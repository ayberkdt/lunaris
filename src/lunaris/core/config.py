# config.py
"""
LUNARIS CONFIGURATION FACTORY
=============================

This module acts as the "Builder" and "Single Source of Truth" (SSOT) manager
for the Lunaris propagation environment.

While `common.type_defs` defines the *atomic building blocks* (Bricks),
this module defines the *blueprints* and the *construction logic* (The Building).

Core Responsibilities
---------------------
1) Composition: assemble low-level SSOT dataclasses (GravityConfig, TimeConfig, ...)
   into a single, validated `SimConfig`.
2) Asset management: locate, validate, and resolve paths for external assets
   (SPICE kernels, gravity model files) with a fail-fast strategy.
3) Safety & validation: enforce cross-module consistency checks that cannot be
   caught by type checking alone (e.g., "SRP requires Sun ephemeris vectors").

Design Philosophy
-----------------
- Import-safe: importing this module must not eagerly import heavy dependencies
  (numba/spiceypy). Heavy model modules are imported only inside factory paths.
- Fail-fast on run: missing assets or missing optional dependencies should fail
  when `load_default_config()` is called, not at import time.
- Strict dataclasses: configs are instantiated directly (TypeError on invalid kwargs).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

# --- Local Imports: Common (dependency-light) ---
from lunaris.common.constants import DAY_S
from lunaris.common.force_requirements import force_requirements_for_config
from lunaris.common.paths import data_dir_from_root, project_root_from_file
from lunaris.common.type_defs import (
    GravityConfig,
    InitialState,
    PerturbationFlags,
    PropagatorConfig,
    SolidTideConfig,
    SpacecraftProps,
    TimeConfig,
)

# --- Type Checking Imports (no runtime cost) ---
if TYPE_CHECKING:
    from lunaris.physics.ephemeris import SpiceBuildConfig
    from lunaris.physics.solar_effects import SRPConfig
    from lunaris.physics.surface_effects import AlbedoConfig, ThermalConfig
    from lunaris.physics.third_body_effects import EarthJ2Params


# =============================================================================
# 1) DEFAULT PATHS & ASSET NAMES
# =============================================================================

# Anchor paths at the project root for editable checkouts. Installed/HPC runs can
# override the external data directory with LUNARIS_DATA_DIR or STLRPS_DATA_DIR.
BASE_DIR = project_root_from_file(__file__)
DATA_DIR = data_dir_from_root(BASE_DIR)
KERNEL_DIR = DATA_DIR / "ephemeris_models"
GRAV_DIR = DATA_DIR / "gravity_models"

# Kernel filename candidates (in priority order).
# We support both raw kernels and "text-wrapped" variants some repos ship (*.tls.txt, *.tpc.txt, *.bpc.txt).
_KERNEL_CANDIDATES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("leapseconds", ("naif0012.tls", "naif0012.tls.txt")),
    ("planetary_constants", ("pck00011.tpc", "pck00011.tpc.txt")),
    ("moon_orientation", ("moon_pa_de440_200625.bpc", "moon_pa_de440_200625.bpc.txt")),
    ("planetary_ephemeris", ("de440.bsp",)),
)

_OPTIONAL_KERNEL_CANDIDATES: tuple[tuple[str, tuple[str, ...]], ...] = (
    # Supplies BODY*_GM values for strict ephemeris validation. Older local data
    # bundles may not have this file yet, so include it opportunistically and
    # let lunaris-data verify --strict report the missing asset.
    ("gravity_constants", ("gm_de440.tpc", "gm_de440.tpc.txt")),
)

# Gravity model filename candidates (in priority order).
_GRAVITY_CANDIDATES: tuple[str, ...] = (
    "jggrx_1800f_sha.tab",
    "jggrx_1800f_sha.tab.txt",
)

FigureSizeName = Literal["landscape", "portrait", "standard"]


def _pick_existing_file(folder: Path, candidates: tuple[str, ...], what: str) -> Path:
    """Return the first existing file inside folder among candidates."""
    for name in candidates:
        p = (folder / name)
        if p.exists():
            return p.resolve()

    raise FileNotFoundError(
        f"CRITICAL: Missing {what} in {folder}.\n"
        f"Tried:\n - " + "\n - ".join(str((folder / n).resolve()) for n in candidates)
    )


def _pick_optional_existing_file(folder: Path, candidates: tuple[str, ...]) -> Path | None:
    """Return the first existing optional file inside folder, or None."""
    for name in candidates:
        p = (folder / name)
        if p.exists():
            return p.resolve()
    return None


def _resolve_default_kernel_paths(kernel_dir: Path | str | None = None) -> tuple[str, ...]:
    """Default local SPICE-kernel resolver for the Lunaris config factory only.

    Dependency-light: resolves paths from local filename candidates without
    importing heavy loaders/model modules. Runtime kernel loading/validation is
    performed elsewhere (lunaris.physics.ephemeris).

    ``kernel_dir`` overrides the default ``KERNEL_DIR`` (from ``LUNARIS_DATA_DIR``
    or the repo ``data/``). CLI callers thread ``--kernel-dir`` here so a custom
    kernel location is honored *before* asset resolution, not after the factory
    has already failed.
    """
    base = Path(kernel_dir).expanduser().resolve() if kernel_dir is not None else KERNEL_DIR
    if not base.exists():
        raise FileNotFoundError(
            f"CRITICAL: Lunaris SPICE kernel directory not found: {base}\n"
            f"Expected folder structure: <data_dir>/ephemeris_models"
        )

    out: list[str] = []
    for purpose, candidates in _KERNEL_CANDIDATES:
        out.append(str(_pick_existing_file(base, candidates, what=f"SPICE kernel ({purpose})")))
    for _purpose, candidates in _OPTIONAL_KERNEL_CANDIDATES:
        optional = _pick_optional_existing_file(base, candidates)
        if optional is not None:
            out.append(str(optional))
    return tuple(out)


def _resolve_default_gravity_path(
    grav_dir: Path | str | None = None,
    grav_file_path: Path | str | None = None,
) -> Path:
    """Default local gravity-model resolver for the Lunaris config factory only.

    Dependency-light: resolves the default gravity file from local filename
    candidates without importing heavy loaders/model modules.

    ``grav_file_path`` names an explicit gravity file directly (CLI
    ``--gravity-file-path``); when given it is verified and returned as-is.
    Otherwise ``grav_dir`` overrides the default ``GRAV_DIR`` and the known
    filename candidates are scanned inside it. Both let a custom asset location be
    honored before the factory would otherwise raise.
    """
    if grav_file_path is not None:
        p = Path(grav_file_path).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(
                f"CRITICAL: Lunaris gravity model file not found: {p}"
            )
        return p
    base = Path(grav_dir).expanduser().resolve() if grav_dir is not None else GRAV_DIR
    if not base.exists():
        raise FileNotFoundError(
            f"CRITICAL: Lunaris gravity model directory not found: {base}\n"
            f"Expected folder structure: <data_dir>/gravity_models"
        )
    return _pick_existing_file(base, _GRAVITY_CANDIDATES, what="gravity model")


# =============================================================================
# 2) TOP-LEVEL SIM CONFIG
# =============================================================================

@dataclass(frozen=True, slots=True)
class VisualConfig:
    """Plotting and reporting configuration."""
    default_dpi: int = 150
    save_pdf: bool = False
    save_pngs: bool = True
    interactive: bool = False

    figure_sizes: Mapping[FigureSizeName, tuple[float, float]] = field(
        default_factory=lambda: {
            "landscape": (12.0, 8.0),
            "portrait": (8.0, 12.0),
            "standard": (10.0, 6.0),
        }
    )
    figure_size_default: FigureSizeName = "landscape"

    def __post_init__(self) -> None:
        if self.default_dpi <= 0:
            raise ValueError(f"VisualConfig.default_dpi must be > 0. Got {self.default_dpi}")
        if self.figure_size_default not in self.figure_sizes:
            raise ValueError(
                f"VisualConfig.figure_size_default='{self.figure_size_default}' "
                f"not found in figure_sizes keys={list(self.figure_sizes.keys())}"
            )
        for name, (w, h) in self.figure_sizes.items():
            if w <= 0.0 or h <= 0.0:
                raise ValueError(f"VisualConfig: figure size '{name}' must be positive. Got {(w, h)}")

    def get_figure_size(self, name: FigureSizeName | None = None) -> tuple[float, float]:
        key = name or self.figure_size_default
        w, h = self.figure_sizes[key]
        return (w, h)


@dataclass(frozen=True, slots=True)
class OutputConfig:
    """File output configuration."""
    out_dir: Path = Path("outputs/simulations")
    create_if_missing: bool = True

    make_3d_plots: bool = True
    downsample_3d: int = 10

    def __post_init__(self) -> None:
        if self.downsample_3d < 1:
            raise ValueError(f"OutputConfig.downsample_3d must be >= 1. Got {self.downsample_3d}")

    def ensure_out_dir(self) -> Path:
        p = self.out_dir.expanduser().resolve()
        if self.create_if_missing:
            p.mkdir(parents=True, exist_ok=True)
        return p


@dataclass(frozen=True, slots=True, kw_only=True)
class SimConfig:
    """
    Central SSOT object for a simulation run.

    Notes
    -----
    - `spice`, `srp`, `albedo`, `thermal`, and `earth_j2` are typed using forward
      refs (TYPE_CHECKING) to keep this module import-safe.
    """
    # Mandatory
    gravity: GravityConfig
    spice: SpiceBuildConfig
    initial_state: InitialState

    # Physics
    flags: PerturbationFlags = field(default_factory=PerturbationFlags)
    spacecraft: SpacecraftProps = field(default_factory=SpacecraftProps)

    # Optional model configs (created only if the corresponding flag is enabled)
    srp: SRPConfig | None = None
    albedo: AlbedoConfig | None = None
    thermal: ThermalConfig | None = None
    solid_tides: SolidTideConfig | None = field(default_factory=SolidTideConfig)

    # Numerics & output
    propagator: PropagatorConfig = field(default_factory=PropagatorConfig)
    time: TimeConfig = field(default_factory=TimeConfig)
    visual: VisualConfig = field(default_factory=VisualConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    # Optional extensions
    earth_j2: EarthJ2Params | None = None

    @property
    def total_seconds(self) -> float:
        return self.time.duration_s

    def validate(self) -> None:
        """Cross-field consistency checks."""
        f = self.flags
        req = force_requirements_for_config(
            self,
            request_external_relativity=False,
        )

        # A) Earth J2
        if f.enable_earth_j2 and self.earth_j2 is None:
            raise ValueError(
                "SimConfig: enable_earth_j2=True but earth_j2 params are None."
            )

        # B) Surface/SRP configs should exist if enabled (lazy-created in factory)
        if f.enable_srp and self.srp is None:
            raise ValueError("SimConfig: enable_srp=True but srp config is None.")
        if f.enable_albedo and self.albedo is None:
            raise ValueError("SimConfig: enable_albedo=True but albedo config is None.")
        if f.enable_thermal and self.thermal is None:
            raise ValueError("SimConfig: enable_thermal=True but thermal config is None.")
        if f.enable_tides and self.solid_tides is None:
            raise ValueError("SimConfig: solid tides enabled but solid_tides config is None.")
        if f.enable_tides_k3 and self.solid_tides is not None and self.solid_tides.k3 is None:
            raise ValueError(
                "SimConfig: enable_tides_k3=True requires solid_tides.k3 to be set explicitly."
            )

        # C) Ephemeris requirements (Sun/Earth vectors)
        if req.need_body_vectors and (not getattr(self.spice, "include_third_body", True)):
            raise ValueError(
                "SimConfig: active physics flags require Sun/Earth ephemeris vectors, "
                "but spice.include_third_body is False."
            )


def replace_sim_config(cfg: SimConfig, **changes: Any) -> SimConfig:
    """Return a validated copy of the simulation SSOT.

    Specialized workflows may need overrides that are not CLI arguments.  They
    must still pass through this helper so a dataclass replacement cannot bypass
    ``SimConfig.validate()``.
    """

    updated = replace(cfg, **changes)
    updated.validate()
    return updated


def ensure_model_configs(cfg: SimConfig) -> SimConfig:
    """Create missing model configs required by enabled perturbation flags."""

    changes: dict[str, Any] = {}

    if cfg.flags.enable_srp and cfg.srp is None:
        try:
            from lunaris.physics.solar_effects import SRPConfig
        except ImportError as e:
            raise ImportError(
                "SRP is enabled but lunaris.physics.solar_effects could not be imported."
            ) from e
        changes["srp"] = SRPConfig()

    if cfg.flags.enable_albedo and cfg.albedo is None:
        try:
            from lunaris.physics.surface_effects import AlbedoConfig
        except ImportError as e:
            raise ImportError(
                "Albedo is enabled but lunaris.physics.surface_effects could not be imported."
            ) from e
        changes["albedo"] = AlbedoConfig()

    if cfg.flags.enable_thermal and cfg.thermal is None:
        try:
            from lunaris.physics.surface_effects import ThermalConfig
        except ImportError as e:
            raise ImportError(
                "Thermal IR is enabled but lunaris.physics.surface_effects could not be imported."
            ) from e
        changes["thermal"] = ThermalConfig()

    if cfg.flags.enable_earth_j2 and cfg.earth_j2 is None:
        try:
            from lunaris.physics.third_body_effects import EarthJ2Params
        except ImportError as e:
            raise ImportError(
                "Earth J2 is enabled but lunaris.physics.third_body_effects could not be imported."
            ) from e
        changes["earth_j2"] = EarthJ2Params(
            j2_coeff=1.08262668e-3,
            r_eq_m=6_378_136.3,
            spin_axis_i=(0.0, 0.0, 1.0),
        )

    if not changes:
        return cfg

    return replace(cfg, **changes)


# =============================================================================
# 3) FACTORY
# =============================================================================

def load_default_config(
    *,
    data_dir: Path | str | None = None,
    kernel_dir: Path | str | None = None,
    gravity_file_path: Path | str | None = None,
) -> SimConfig:
    """
    Create, validate, and return the default simulation configuration.

    This function may import heavy modules (lunaris.physics.ephemeris / numba / spiceypy)
    and will raise ImportError with a helpful message if the environment is incomplete.

    Parameters
    ----------
    data_dir :
        External data root to resolve assets from, overriding the default
        (``LUNARIS_DATA_DIR`` / repo ``data/``). ``ephemeris_models`` and
        ``gravity_models`` subdirectories are resolved beneath it.
    kernel_dir :
        SPICE-kernel directory override (takes precedence over ``data_dir`` for
        kernels). Maps the CLI ``--kernel-dir``.
    gravity_file_path :
        Explicit gravity model file override. Maps the CLI ``--gravity-file-path``.

    These overrides are applied *before* asset resolution so a caller supplying a
    custom asset location on a machine without the default ``data/`` layout still
    builds a config, instead of failing during default resolution. Passing nothing
    preserves the historical default-resolution behavior exactly.
    """

    # -------------------------------------------------------------------------
    # STEP 1: Resolve & validate assets (no heavy imports)
    # -------------------------------------------------------------------------
    resolved_data_dir = Path(data_dir).expanduser().resolve() if data_dir is not None else None
    if kernel_dir is None and resolved_data_dir is not None:
        kernel_dir = resolved_data_dir / "ephemeris_models"
    grav_dir = resolved_data_dir / "gravity_models" if resolved_data_dir is not None else None

    kernel_paths = _resolve_default_kernel_paths(kernel_dir)
    grav_path = _resolve_default_gravity_path(grav_dir=grav_dir, grav_file_path=gravity_file_path)

    # -------------------------------------------------------------------------
    # STEP 2: Build sub-configurations
    # -------------------------------------------------------------------------

    # (1) Ephemeris / SPICE config (heavy dependency: spiceypy + numba)
    try:
        from lunaris.physics.ephemeris import SpiceBuildConfig
        # Defaults are stable strings; keep config import-safe by not importing these at module scope.
        DEFAULT_INERTIAL_FRAME = "J2000"
        DEFAULT_FIXED_FRAME = "MOON_PA"
    except ImportError as e:
        raise ImportError(
            "Failed to import lunaris.physics.ephemeris (requires 'spiceypy' and 'numba' to be installed and compatible)."
        ) from e

    spice_cfg = SpiceBuildConfig(
        kernels=tuple(kernel_paths),
        inertial_frame=DEFAULT_INERTIAL_FRAME,
        fixed_frame=DEFAULT_FIXED_FRAME,
        include_third_body=True,  # needed when SRP/Albedo/Thermal/3rd-body flags are enabled
    )

    # (2) Gravity config (strict, dependency-light)
    gravity_cfg = GravityConfig(
        file_path=str(grav_path),
        degree=100,
    )

    # (3) Physics flags (strict)
    flags = PerturbationFlags(
        enable_sh=True,
        enable_3rd_body_sun=False,
        enable_3rd_body_earth=False,
        enable_earth_j2=False,
        enable_srp=False,
        enable_albedo=False,
        enable_thermal=False,
        enable_tides_k2=False,
        enable_tides_k3=False,
        enable_relativity_1pn=False,
    )

    # (4) Time & initial state (strict)
    time_cfg = TimeConfig(
        duration_s=DAY_S,
        output_dt_s=60.0,
        samples_per_period=360,
    )
    init_state = InitialState(
        x=1_837_400.0, y=0.0, z=0.0,
        vx=0.0, vy=1_633.0, vz=0.0,
    )

    # (5) Numerical propagation
    propagator_cfg = PropagatorConfig(method="DOP853")

    # -------------------------------------------------------------------------
    # STEP 3: Optional model configs (import only if enabled)
    # -------------------------------------------------------------------------
    srp_cfg = None
    albedo_cfg = None
    thermal_cfg = None
    solid_tides_cfg = SolidTideConfig()

    if flags.enable_srp:
        try:
            from lunaris.physics.solar_effects import SRPConfig
        except ImportError as e:
            raise ImportError(
                "SRP is enabled but lunaris.physics.solar_effects could not be imported (numba dependency)."
            ) from e
        srp_cfg = SRPConfig()

    if flags.enable_albedo or flags.enable_thermal:
        try:
            from lunaris.physics.surface_effects import AlbedoConfig, ThermalConfig
        except ImportError as e:
            raise ImportError(
                "Albedo/Thermal is enabled but lunaris.physics.surface_effects could not be imported (numba dependency)."
            ) from e

        if flags.enable_albedo:
            albedo_cfg = AlbedoConfig()
        if flags.enable_thermal:
            thermal_cfg = ThermalConfig()

    # Earth J2 (optional)
    earth_j2_params = None
    if flags.enable_earth_j2:
        try:
            from lunaris.physics.third_body_effects import EarthJ2Params
        except ImportError as e:
            raise ImportError(
                "Earth J2 requested (enable_earth_j2=True) but lunaris.physics.third_body_effects is unavailable."
            ) from e

        # Typical values (WGS-84-like)
        R_EARTH_EQ_M = 6_378_136.3
        J2_EARTH = 1.08262668e-3
        earth_j2_params = EarthJ2Params(
            j2_coeff=J2_EARTH,
            r_eq_m=R_EARTH_EQ_M,
            spin_axis_i=(0.0, 0.0, 1.0),
        )

    # -------------------------------------------------------------------------
    # STEP 4: Assemble & validate
    # -------------------------------------------------------------------------
    cfg = SimConfig(
        gravity=gravity_cfg,
        spice=spice_cfg,
        initial_state=init_state,

        flags=flags,
        spacecraft=SpacecraftProps(),

        srp=srp_cfg,
        albedo=albedo_cfg,
        thermal=thermal_cfg,
        solid_tides=solid_tides_cfg,

        time=time_cfg,
        propagator=propagator_cfg,

        visual=VisualConfig(),
        output=OutputConfig(),

        earth_j2=earth_j2_params,
    )

    cfg.validate()
    return cfg


# =============================================================================
# 4) CONVENIENCE ACCESSOR
# =============================================================================

def get_default_config() -> SimConfig:
    """Explicit accessor for the default configuration.

    This is a thin alias for :func:`load_default_config`. There is intentionally
    no module-level default instance: importing this module must never trigger
    asset discovery, SPICE/gravity loading, or optional-dependency imports.
    Callers obtain a config only by explicitly calling this (or the factory).
    """
    return load_default_config()


# A human-runnable smoke check for the default configuration lives in
# ``tools/check_config.py`` (kept out of this module so importing the core
# library never prints or triggers asset discovery as a side effect).


# =============================================================================
# 5) PUBLIC API
# =============================================================================

__all__ = [
    "DATA_DIR",
    "KERNEL_DIR",
    "GRAV_DIR",
    "SimConfig",
    "replace_sim_config",
    "VisualConfig",
    "OutputConfig",
    "load_default_config",
    "get_default_config",
]
