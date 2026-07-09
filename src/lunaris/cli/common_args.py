# cli/common_args.py
"""
Lunaris shared CLI helpers.

Pure, import-safe argument helpers shared by the command-line entry points
(``main.py`` and ``batch_runner.py``). This module is intentionally
dependency-light: it imports only from the dependency-light ``common`` layer
at module scope. Heavy modules (loaders / physics / core) are imported lazily
inside the functions that need them, never at import time.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from lunaris.common.constants import DAY_S, R_MOON
from lunaris.common.force_requirements import force_requirements_for_config
from lunaris.common.type_defs import GravityBackend, SolidTideConfig

if TYPE_CHECKING:
    # Typing-only import keeps this module import-safe.
    from lunaris.core.config import SimConfig


_BOOL_TRUE = {"1", "true", "t", "yes", "y", "on"}
_BOOL_FALSE = {"0", "false", "f", "no", "n", "off"}


def str2bool(v: Any) -> bool:
    """argparse-friendly bool parser (strict)."""
    if isinstance(v, bool):
        return v
    if v is None:
        raise argparse.ArgumentTypeError("Boolean value expected, got None.")
    s = str(v).strip().lower()
    if s in _BOOL_TRUE:
        return True
    if s in _BOOL_FALSE:
        return False
    raise argparse.ArgumentTypeError(
        f"Boolean value expected (one of {sorted(_BOOL_TRUE | _BOOL_FALSE)}), got '{v}'."
    )


def parse_tide_bodies(v: Any) -> tuple[str, ...]:
    """Parse a comma-separated tide body list into a validated tuple."""
    parts = tuple(p.strip().lower() for p in str(v).split(",") if p.strip())
    try:
        return SolidTideConfig(tide_bodies=parts).tide_bodies
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def parse_adaptive_table(s: str) -> tuple[tuple[float, int], ...] | None:
    """Parse adaptive-degree table from CLI.

    Expected format:
        "alt_km:deg,alt_km:deg,..."

    Returns:
        tuple of (alt_km, degree) rows (strictly ascending in alt_km).
    """
    if s is None or str(s).strip() == "":
        return None

    pairs: list[tuple[float, int]] = []
    for i, chunk in enumerate(str(s).split(",")):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            alt_str, deg_str = (p.strip() for p in chunk.split(":", 1))
            alt = float(alt_str)
            deg = int(deg_str)
        except Exception as e:
            raise argparse.ArgumentTypeError(
                f"Invalid --adaptive-table at item {i}: '{chunk}'. "
                "Expected 'alt_km:deg' (e.g. '50:60,200:180')."
            ) from e

        if alt < 0:
            raise argparse.ArgumentTypeError(f"adaptive-table item {i}: altitude must be >= 0 km.")
        if deg < 0:
            raise argparse.ArgumentTypeError(f"adaptive-table item {i}: degree must be >= 0.")
        pairs.append((alt, deg))

    # strictly ascending altitude thresholds
    for i in range(1, len(pairs)):
        if pairs[i][0] <= pairs[i - 1][0]:
            raise argparse.ArgumentTypeError("adaptive-table must be strictly ascending in altitude thresholds.")

    return tuple(pairs) if pairs else None


def resolve_orbit_elements(args: argparse.Namespace) -> dict[str, float]:
    """Resolve orbit COEs from CLI args (strict validation).

    Priority:
      1) hp/ha -> compute (a,e)
      2) a/e direct
      3) alt-km circular
    Angles default to 0 if omitted.
    """
    R_km = float(R_MOON) / 1000.0

    inc_deg = float(args.inc_deg) if args.inc_deg is not None else 0.0
    raan_deg = float(args.raan_deg) if args.raan_deg is not None else 0.0
    argp_deg = float(args.argp_deg) if args.argp_deg is not None else 0.0
    ta_deg = float(args.ta_deg) if args.ta_deg is not None else 0.0

    if args.hp_km is not None and args.ha_km is not None:
        hp = float(args.hp_km)
        ha = float(args.ha_km)
        if hp < 0 or ha < 0:
            raise ValueError("hp_km/ha_km must be >= 0.")
        rp_km = R_km + hp
        ra_km = R_km + ha
        a_km = 0.5 * (rp_km + ra_km)
        e = (ra_km - rp_km) / (ra_km + rp_km)

    elif args.a_km is not None and args.e is not None:
        a_km = float(args.a_km)
        e = float(args.e)
        if a_km <= 0:
            raise ValueError("a_km must be > 0.")
        if not (0.0 <= e < 1.0):
            raise ValueError("e must satisfy 0 <= e < 1 for elliptic orbits.")

    elif args.alt_km is not None:
        alt = float(args.alt_km)
        if alt < 0:
            raise ValueError("alt_km must be >= 0.")
        a_km = R_km + alt
        e = 0.0

    else:
        raise ValueError("No orbit init provided. Use --hp-km/--ha-km or --a-km/--e or --alt-km.")

    return {
        "a_km": float(a_km),
        "e": float(e),
        "inc_deg": float(inc_deg),
        "raan_deg": float(raan_deg),
        "argp_deg": float(argp_deg),
        "ta_deg": float(ta_deg),
    }


def init_surface_provider(args: argparse.Namespace) -> Any | None:
    """Load surface provider strictly when CLI roots are provided.

    Contract:
      - Returns an object implementing as_numba_dict()->dict for core.dynamics
      - Also exposes .grids().topo for topo-aware impact events (optional)
    """
    if args.ldem_root is None and args.albedo_root is None:
        return None

    # Local import to avoid heavy imports unless requested
    try:
        from lunaris.loaders.io_surface import FileBackedSurfaceProvider
    except Exception as e:
        raise ImportError(
            "Surface grids requested, but 'loaders.io_surface' is not importable. "
            "Check that the loaders package exists on PYTHONPATH."
        ) from e

    return FileBackedSurfaceProvider(
        ldem_root=str(args.ldem_root) if args.ldem_root is not None else None,
        albedo_root=str(args.albedo_root) if args.albedo_root is not None else None,
        ldem_ppd=int(args.ldem_ppd) if args.ldem_ppd is not None else None,
    )


def need_ephemeris(cfg: SimConfig, topo_requested: bool) -> bool:
    """Return True if any enabled physics (or topo) requires ephemeris tables."""
    req = force_requirements_for_config(
        cfg,
        request_external_relativity=True,
    )
    return bool(req.need_ephem or topo_requested)


_FieldPatch = tuple[str, str, Callable[[Any], Any]]

def patch_dataclass(obj, args: argparse.Namespace, patches: Sequence[_FieldPatch]):
    """Apply a table of (arg_name, field_name, caster) patches to a frozen dataclass."""
    replacements: dict[str, Any] = {}
    for arg_name, field_name, caster in patches:
        val = getattr(args, arg_name, None)
        if val is not None:
            replacements[field_name] = caster(val)
    return replace(obj, **replacements) if replacements else obj

_TIME_PATCHES: tuple[_FieldPatch, ...] = (
    ("output_dt_s", "output_dt_s", float),
    ("samples_per_period", "samples_per_period", int),
)

_SPACECRAFT_PATCHES: tuple[_FieldPatch, ...] = (
    ("mass_kg", "mass_kg", float),
    ("area_m2", "area_m2", float),
    ("cd", "cd", float),
    ("cr", "cr", float),
)

_FLAGS_PATCHES: tuple[_FieldPatch, ...] = (
    ("enable_sh", "enable_sh", bool),
    ("enable_3rd_body_sun", "enable_3rd_body_sun", bool),
    ("enable_3rd_body_earth", "enable_3rd_body_earth", bool),
    ("enable_earth_j2", "enable_earth_j2", bool),
    ("enable_srp", "enable_srp", bool),
    ("enable_albedo", "enable_albedo", bool),
    ("enable_thermal", "enable_thermal", bool),
    ("enable_relativity_1pn", "enable_relativity_1pn", bool),
)

_TIDES_PATCHES: tuple[_FieldPatch, ...] = (
    ("tide_bodies", "tide_bodies", tuple),
    ("tide_k2", "k2", float),
    ("tide_k3", "k3", float),
    ("tide_r_ref_m", "r_ref_m", float),
)

_GRAVITY_PATCHES: tuple[_FieldPatch, ...] = (
    ("gravity_file_path", "file_path", str),
    ("degree", "degree", int),
)

_PROPAGATOR_PATCHES: tuple[_FieldPatch, ...] = (
    ("method", "method", lambda v: str(v).strip()),
    ("user_max_step_s", "user_max_step_s", float),
    ("rtol", "rtol", float),
    ("atol", "atol", float),
    ("compute_2body_baseline", "compute_2body_baseline", bool),
    ("enable_telemetry", "enable_telemetry", bool),
    ("telem_cadence_s", "telem_cadence_s", float),
)

_OUTPUT_PATCHES: tuple[_FieldPatch, ...] = (
    ("out_dir", "out_dir", lambda v: Path(str(v)).expanduser()),
    ("make_3d_plots", "make_3d_plots", bool),
    ("downsample_3d", "downsample_3d", int),
)

_THERMAL_PATCHES: tuple[_FieldPatch, ...] = (
    ("thermal_mode", "thermal_mode", str),
    ("thermal_temperature_k", "temperature_K", float),
    ("thermal_night_temperature_k", "night_temperature_K", float),
    ("thermal_emissivity", "surface_emissivity", float),
    ("thermal_surface_albedo", "surface_albedo", float),
    ("thermal_floor_flux_w_m2", "thermal_floor_flux_W_m2", float),
    ("thermal_facet_lat_count", "facet_lat_count", int),
    ("thermal_facet_lon_count", "facet_lon_count", int),
)

_ALBEDO_PATCHES: tuple[_FieldPatch, ...] = (
    ("albedo_model", "albedo_model", str),
    ("albedo_mode", "albedo_mode", str),
    ("albedo_pressure_coefficient", "albedo_pressure_coefficient", float),
    ("albedo_facet_lat_count", "facet_lat_count", int),
    ("albedo_facet_lon_count", "facet_lon_count", int),
    ("albedo_require_provider", "require_surface_provider", bool),
    ("albedo_enable_eclipse", "enable_eclipse", bool),
)


def apply_args_to_config(cfg: SimConfig, args: argparse.Namespace) -> SimConfig:
    # Lazy import keeps module import light: common.time_utils transitively
    # pulls numba/scipy, which we only need when actually applying overrides.
    from lunaris.common.time_utils import normalize_iso_datetime_to_utc_string

    # --- Time ---
    if args.start_date is not None:
        cfg = replace(
            cfg,
            time=replace(
                cfg.time,
                start_date=normalize_iso_datetime_to_utc_string(
                    str(args.start_date).strip(),
                    precision=0,
                ),
            ),
        )

    if args.days is not None:
        cfg = replace(cfg, time=replace(cfg.time, duration_s=float(args.days) * DAY_S))
    elif args.hours is not None:
        cfg = replace(cfg, time=replace(cfg.time, duration_s=float(args.hours) * 3600.0))

    cfg = replace(cfg, time=patch_dataclass(cfg.time, args, _TIME_PATCHES))

    # --- Spacecraft ---
    cfg = replace(cfg, spacecraft=patch_dataclass(cfg.spacecraft, args, _SPACECRAFT_PATCHES))

    # --- Flags (PerturbationFlags) ---
    flags = patch_dataclass(cfg.flags, args, _FLAGS_PATCHES)

    # Tides mapping: clean CLI -> internal k2/k3 booleans (dataclass constraint: k3 => k2)
    if args.enable_tides is not None or args.tides_kind is not None:
        tides_on = bool(args.enable_tides) if args.enable_tides is not None else True  # kind implies on
        if not tides_on:
            flags = replace(flags, enable_tides_k2=False, enable_tides_k3=False)
        else:
            kind = str(args.tides_kind).strip().lower() if args.tides_kind is not None else "k2"
            if kind == "k3":
                flags = replace(flags, enable_tides_k2=True, enable_tides_k3=True)
            else:
                flags = replace(flags, enable_tides_k2=True, enable_tides_k3=False)

    cfg = replace(cfg, flags=flags)

    # --- Thermal IR configuration ---
    thermal_arg_names = (
        "thermal_mode",
        "thermal_temperature_k",
        "thermal_night_temperature_k",
        "thermal_emissivity",
        "thermal_surface_albedo",
        "thermal_ir_coefficient",
        "thermal_floor_flux_w_m2",
        "thermal_facet_lat_count",
        "thermal_facet_lon_count",
    )
    thermal_requested = bool(flags.enable_thermal) or any(
        getattr(args, name, None) is not None for name in thermal_arg_names
    )
    if thermal_requested:
        from lunaris.physics.surface_effects import ThermalConfig

        th = cfg.thermal if cfg.thermal is not None else ThermalConfig()
        th = patch_dataclass(th, args, _THERMAL_PATCHES)
        if getattr(args, "thermal_ir_coefficient", None) is not None:
            coeff = float(args.thermal_ir_coefficient)
            th = replace(th, ir_pressure_coefficient=coeff, k_thermal=coeff)
        cfg = replace(cfg, thermal=th)

    # --- Lunar albedo configuration ---
    albedo_arg_names = (
        "albedo_root",
        "albedo_model",
        "albedo_mode",
        "albedo_const",
        "albedo_pressure_coefficient",
        "albedo_facet_lat_count",
        "albedo_facet_lon_count",
        "albedo_require_provider",
        "albedo_enable_eclipse",
    )
    albedo_requested = bool(flags.enable_albedo) or any(
        getattr(args, name, None) is not None for name in albedo_arg_names
    )
    if albedo_requested:
        from lunaris.physics.surface_effects import AlbedoConfig

        alb = cfg.albedo if cfg.albedo is not None else AlbedoConfig()
        alb = patch_dataclass(alb, args, _ALBEDO_PATCHES)
        # Convenience: supplying an albedo raster but no explicit mode selects the
        # scaled-DN grid source, preserving the historical "--albedo-root drives
        # albedo" behavior with the facet model.
        if getattr(args, "albedo_mode", None) is None and getattr(args, "albedo_root", None) is not None:
            alb = replace(alb, albedo_mode="scaled_dn_grid")
        if getattr(args, "albedo_const", None) is not None:
            alb = replace(alb, albedo_const=float(args.albedo_const), A_moon=float(args.albedo_const))
        cfg = replace(cfg, albedo=alb)
        
        # User explicitly supplied albedo settings -> automatically enable the force
        # unless they explicitly turned it off via --enable-albedo=False.
        if any(getattr(args, name, None) is not None for name in albedo_arg_names):
            if getattr(args, "enable_albedo", None) is not False:
                flags = replace(flags, enable_albedo=True)
                cfg = replace(cfg, flags=flags)

    # --- Solid tide configuration ---
    tide_cfg = cfg.solid_tides if cfg.solid_tides is not None else SolidTideConfig()
    tide_cfg = patch_dataclass(tide_cfg, args, _TIDES_PATCHES)
    cfg = replace(cfg, solid_tides=tide_cfg)

    # --- Gravity config (GravityConfig) ---
    grav_cfg = cfg.gravity
    new_backend = str(getattr(grav_cfg, "backend", "classic_sh") or "classic_sh")
    new_surrogate_dir = str(getattr(grav_cfg, "st_lrps_model_dir", "") or "")
    if args.gravity_backend is not None:
        new_backend = str(args.gravity_backend)
    if args.surrogate_gravity_model_dir is not None:
        new_surrogate_dir = str(Path(str(args.surrogate_gravity_model_dir)).expanduser().resolve())
    # ``new_backend`` is a raw CLI string; GravityConfig.__post_init__ is the SSOT
    # that validates/normalizes it (no duplicate CLI-side validation). The cast
    # only bridges the static Literal type at this argparse boundary.
    grav_cfg = replace(
        grav_cfg,
        backend=cast(GravityBackend, new_backend),
        st_lrps_model_dir=new_surrogate_dir,
    )
    grav_cfg = patch_dataclass(grav_cfg, args, _GRAVITY_PATCHES)

    if args.adaptive_enabled is not None:
        grav_cfg = replace(grav_cfg, adaptive=replace(grav_cfg.adaptive, enabled=bool(args.adaptive_enabled)))
    if args.adaptive_table is not None:
        grav_cfg = replace(grav_cfg, adaptive=replace(grav_cfg.adaptive, altitude_table=args.adaptive_table))
        if args.adaptive_enabled is None:
            grav_cfg = replace(grav_cfg, adaptive=replace(grav_cfg.adaptive, enabled=True))

    cfg = replace(cfg, gravity=grav_cfg)

    # --- Propagator config (PropagatorConfig) ---
    prop_cfg = patch_dataclass(cfg.propagator, args, _PROPAGATOR_PATCHES)
    cfg = replace(cfg, propagator=prop_cfg)

    # --- Output config (OutputConfig) ---
    out_cfg = patch_dataclass(cfg.output, args, _OUTPUT_PATCHES)
    cfg = replace(cfg, output=out_cfg)

    # --- Kernel dir remap (strict by filename) ---
    if args.kernel_dir is not None:
        kd = Path(str(args.kernel_dir)).expanduser().resolve()
        new_kernels = tuple(str(kd / Path(str(k)).name) for k in cfg.spice.kernels)
        cfg = replace(cfg, spice=replace(cfg.spice, kernels=new_kernels))

    # Final validation (fail-fast)
    from lunaris.core.config import ensure_model_configs

    cfg = ensure_model_configs(cfg)
    cfg.validate()
    return cfg


__all__ = [
    "str2bool",
    "parse_tide_bodies",
    "parse_adaptive_table",
    "resolve_orbit_elements",
    "init_surface_provider",
    "need_ephemeris",
    "apply_args_to_config",
]
