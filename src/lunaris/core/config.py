# config.py
"""
ST_LRPS CONFIGURATION FACTORY
=============================

This module acts as the "Builder" and "Single Source of Truth" (SSOT) manager
for the ST_LRPS environment.

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

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Mapping, Optional, Tuple, TYPE_CHECKING

# --- Local Imports: Common (dependency-light) ---
from lunaris.common.constants import DAY_S
from lunaris.common.paths import data_dir_from_root, project_root_from_file
from lunaris.common.type_defs import (
    GravityConfig,
    InitialState,
    PerturbationFlags,
    PropagatorConfig,
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
_KERNEL_CANDIDATES: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("leapseconds", ("naif0012.tls", "naif0012.tls.txt")),
    ("planetary_constants", ("pck00011.tpc", "pck00011.tpc.txt")),
    ("moon_orientation", ("moon_pa_de440_200625.bpc", "moon_pa_de440_200625.bpc.txt")),
    ("planetary_ephemeris", ("de440.bsp",)),
)

# Gravity model filename candidates (in priority order).
_GRAVITY_CANDIDATES: Tuple[str, ...] = (
    "jggrx_1800f_sha.tab",
    "jggrx_1800f_sha.tab.txt",
)

FigureSizeName = Literal["landscape", "portrait", "standard"]


def _pick_existing_file(folder: Path, candidates: Tuple[str, ...], what: str) -> Path:
    """Return the first existing file inside folder among candidates."""
    for name in candidates:
        p = (folder / name)
        if p.exists():
            return p.resolve()

    raise FileNotFoundError(
        f"CRITICAL: Missing {what} in {folder}.\n"
        f"Tried:\n - " + "\n - ".join(str((folder / n).resolve()) for n in candidates)
    )


def _resolve_default_kernel_paths() -> Tuple[str, ...]:
    """Default local SPICE-kernel resolver for the ST_LRPS config factory only.

    Dependency-light: resolves paths from local filename candidates without
    importing heavy loaders/model modules. Runtime kernel loading/validation is
    performed elsewhere (lunaris.physics.ephemeris).
    """
    if not KERNEL_DIR.exists():
        raise FileNotFoundError(
            f"CRITICAL: ST_LRPS SPICE kernel directory not found: {KERNEL_DIR}\n"
            f"Expected folder structure: {DATA_DIR}/ephemeris_models"
        )

    out: list[str] = []
    for purpose, candidates in _KERNEL_CANDIDATES:
        out.append(str(_pick_existing_file(KERNEL_DIR, candidates, what=f"SPICE kernel ({purpose})")))
    return tuple(out)


def _resolve_default_gravity_path() -> Path:
    """Default local gravity-model resolver for the ST_LRPS config factory only.

    Dependency-light: resolves the default gravity file from local filename
    candidates without importing heavy loaders/model modules.
    """
    if not GRAV_DIR.exists():
        raise FileNotFoundError(
            f"CRITICAL: ST_LRPS gravity model directory not found: {GRAV_DIR}\n"
            f"Expected folder structure: {DATA_DIR}/gravity_models"
        )
    return _pick_existing_file(GRAV_DIR, _GRAVITY_CANDIDATES, what="gravity model")


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

    figure_sizes: Mapping[FigureSizeName, Tuple[float, float]] = field(
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

    def get_figure_size(self, name: Optional[FigureSizeName] = None) -> Tuple[float, float]:
        key = name or self.figure_size_default
        return tuple(self.figure_sizes[key])


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
    spice: "SpiceBuildConfig"
    initial_state: InitialState

    # Physics
    flags: PerturbationFlags = field(default_factory=PerturbationFlags)
    spacecraft: SpacecraftProps = field(default_factory=SpacecraftProps)

    # Optional model configs (created only if the corresponding flag is enabled)
    srp: Optional["SRPConfig"] = None
    albedo: Optional["AlbedoConfig"] = None
    thermal: Optional["ThermalConfig"] = None

    # Numerics & output
    propagator: PropagatorConfig = field(default_factory=PropagatorConfig)
    time: TimeConfig = field(default_factory=TimeConfig)
    visual: VisualConfig = field(default_factory=VisualConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    # Optional extensions
    earth_j2: Optional["EarthJ2Params"] = None

    @property
    def total_seconds(self) -> float:
        return self.time.duration_s

    def validate(self) -> None:
        """Cross-field consistency checks."""
        f = self.flags

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

        # C) Ephemeris requirements (Sun/Earth vectors)
        need_sun_vec = (
            f.enable_srp
            or f.enable_albedo
            or f.enable_thermal
            or f.enable_3rd_body_sun
            or f.enable_tides_k2
            or f.enable_tides_k3
        )
        need_earth_vec = (
            f.enable_3rd_body_earth
            or f.enable_earth_j2
            or f.enable_tides_k2
            or f.enable_tides_k3
        )
        need_vectors = bool(need_sun_vec or need_earth_vec)
        if need_vectors and (not getattr(self.spice, "include_third_body", True)):
            raise ValueError(
                "SimConfig: active physics flags require Sun/Earth ephemeris vectors, "
                "but spice.include_third_body is False."
            )


# =============================================================================
# 3) FACTORY
# =============================================================================

def load_default_config() -> SimConfig:
    """
    Create, validate, and return the default simulation configuration.

    This function may import heavy modules (lunaris.physics.ephemeris / numba / spiceypy)
    and will raise ImportError with a helpful message if the environment is incomplete.
    """

    # -------------------------------------------------------------------------
    # STEP 1: Resolve & validate assets (no heavy imports)
    # -------------------------------------------------------------------------
    kernel_paths = _resolve_default_kernel_paths()
    grav_path = _resolve_default_gravity_path()

    # -------------------------------------------------------------------------
    # STEP 2: Build sub-configurations
    # -------------------------------------------------------------------------

    # (1) Ephemeris / SPICE config (heavy dependency: spiceypy + numba)
    try:
        from lunaris.physics.ephemeris import SpiceBuildConfig
        # Defaults are stable strings; keep config import-safe by not importing these at module scope.
        DEFAULT_INERTIAL_FRAME = "J2000"
        DEFAULT_FIXED_FRAME = "MOON_PA"
    except Exception as e:
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

    if flags.enable_srp:
        try:
            from lunaris.physics.solar_effects import SRPConfig
        except Exception as e:
            raise ImportError(
                "SRP is enabled but models.solar_effects could not be imported (numba dependency)."
            ) from e
        srp_cfg = SRPConfig()

    if flags.enable_albedo or flags.enable_thermal:
        try:
            from lunaris.physics.surface_effects import AlbedoConfig, ThermalConfig
        except Exception as e:
            raise ImportError(
                "Albedo/Thermal is enabled but models.surface_effects could not be imported (numba dependency)."
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
        except Exception as e:
            raise ImportError(
                "Earth J2 requested (enable_earth_j2=True) but models.third_body_effects is unavailable."
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


# =============================================================================
# 5) SMOKE TEST
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🛠️   ST_LRPS CONFIGURATION CHECK")
    print("=" * 60 + "\n")

    try:
        test_cfg = load_default_config()
        print("✅ [PASS] load_default_config() successful.")
    except Exception as e:
        print(f"❌ [FAIL] Could not load config: {e}")
        raise

    print("\n📂 [PATHS]")
    print(f"   Gravity Model : {test_cfg.gravity.file_path}")
    print(f"   SPICE Kernels : {len(test_cfg.spice.kernels)} selected.")

    print("\n⚙️  [PHYSICS FLAGS]")
    print(f"   Spherical Harmonics : {test_cfg.flags.enable_sh}")
    print(f"   3rd Body (Sun/Earth): {test_cfg.flags.enable_3rd_body_sun} / {test_cfg.flags.enable_3rd_body_earth}")
    print(f"   Earth J2            : {test_cfg.flags.enable_earth_j2}")
    print(f"   Tides (K2/K3)       : {test_cfg.flags.enable_tides_k2} / {test_cfg.flags.enable_tides_k3}")
    print(f"   Relativity (1PN)    : {test_cfg.flags.enable_relativity_1pn}")

    print("\n🚀 [MISSION PARAMETERS]")
    print(f"   Spacecraft Mass : {test_cfg.spacecraft.mass_kg} kg")
    print(f"   Total Duration  : {test_cfg.total_seconds / 86400.0:.2f} days")
    dt_display = test_cfg.time.output_dt_s
    dt_str = f"{dt_display}s" if dt_display is not None else "Auto (Variable)"
    print(f"   Time Step (Out) : {dt_str}")
    print(f"   Propagator      : {test_cfg.propagator.method} (Tol: {test_cfg.propagator.rtol})")

    print("\n" + "=" * 60)
    print("✅ CONFIGURATION INTEGRITY CHECK COMPLETE.")
    print("=" * 60 + "\n")


# =============================================================================
# 6) PUBLIC API
# =============================================================================

__all__ = [
    "DATA_DIR",
    "KERNEL_DIR",
    "GRAV_DIR",
    "SimConfig",
    "VisualConfig",
    "OutputConfig",
    "load_default_config",
    "get_default_config",
]
