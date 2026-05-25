# cli/common_args.py
"""
ST_LRPS shared CLI helpers.

Pure, import-safe argument helpers shared by the command-line entry points
(``main.py`` and ``mc_runner.py``). This module is intentionally
dependency-light: it imports only from the dependency-light ``common`` layer
at module scope. Heavy modules (loaders / models / core) are imported lazily
inside the functions that need them, never at import time.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING

from common.constants import R_MOON, DAY_S

if TYPE_CHECKING:
    # Typing-only import keeps this module import-safe.
    from config import SimConfig


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


def parse_adaptive_table(s: str) -> Optional[Tuple[Tuple[float, int], ...]]:
    """Parse adaptive-degree table from CLI.

    Expected format:
        "alt_km:deg,alt_km:deg,..."

    Returns:
        tuple of (alt_km, degree) rows (strictly ascending in alt_km).
    """
    if s is None or str(s).strip() == "":
        return None

    pairs: list[Tuple[float, int]] = []
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


def resolve_orbit_elements(args: argparse.Namespace) -> Dict[str, float]:
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


def init_surface_provider(args: argparse.Namespace) -> Optional[Any]:
    """Load surface provider strictly when CLI roots are provided.

    Contract:
      - Returns an object implementing as_numba_dict()->dict for core.dynamics
      - Also exposes .grids().topo for topo-aware impact events (optional)
    """
    if args.ldem_root is None and args.albedo_root is None:
        return None

    # Local import to avoid heavy imports unless requested
    try:
        from loaders.io_surface import FileBackedSurfaceProvider
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


def need_ephemeris(cfg: "SimConfig", topo_requested: bool) -> bool:
    """Return True if any enabled physics (or topo) requires ephemeris tables."""
    f = cfg.flags
    physics_need = (
        f.enable_sh
        or f.enable_3rd_body_sun
        or f.enable_3rd_body_earth
        or f.enable_earth_j2
        or f.enable_srp
        or f.enable_albedo
        or f.enable_thermal
        or f.enable_surface_forces
        or f.enable_tides_k2
        or f.enable_tides_k3
        or f.enable_relativity_1pn
    )
    return bool(physics_need or topo_requested)


def apply_args_to_config(cfg: "SimConfig", args: argparse.Namespace) -> "SimConfig":
    # Lazy import keeps module import light: common.time_utils transitively
    # pulls numba/scipy, which we only need when actually applying overrides.
    from common.time_utils import normalize_iso_datetime_to_utc_string

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

    if args.output_dt_s is not None:
        cfg = replace(cfg, time=replace(cfg.time, output_dt_s=float(args.output_dt_s)))
    if args.samples_per_period is not None:
        cfg = replace(cfg, time=replace(cfg.time, samples_per_period=int(args.samples_per_period)))

    # --- Spacecraft ---
    sc = cfg.spacecraft
    if args.mass_kg is not None:
        sc = replace(sc, mass_kg=float(args.mass_kg))
    if args.area_m2 is not None:
        sc = replace(sc, area_m2=float(args.area_m2))
    if args.cd is not None:
        sc = replace(sc, cd=float(args.cd))
    if args.cr is not None:
        sc = replace(sc, cr=float(args.cr))
    cfg = replace(cfg, spacecraft=sc)

    # --- Flags (PerturbationFlags) ---
    flags = cfg.flags
    if args.enable_sh is not None:
        flags = replace(flags, enable_sh=bool(args.enable_sh))
    if args.enable_3rd_body_sun is not None:
        flags = replace(flags, enable_3rd_body_sun=bool(args.enable_3rd_body_sun))
    if args.enable_3rd_body_earth is not None:
        flags = replace(flags, enable_3rd_body_earth=bool(args.enable_3rd_body_earth))
    if args.enable_earth_j2 is not None:
        flags = replace(flags, enable_earth_j2=bool(args.enable_earth_j2))
    if args.enable_srp is not None:
        flags = replace(flags, enable_srp=bool(args.enable_srp))
    if args.enable_albedo is not None:
        flags = replace(flags, enable_albedo=bool(args.enable_albedo))
    if args.enable_thermal is not None:
        flags = replace(flags, enable_thermal=bool(args.enable_thermal))

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

    if args.enable_relativity_1pn is not None:
        flags = replace(flags, enable_relativity_1pn=bool(args.enable_relativity_1pn))

    cfg = replace(cfg, flags=flags)

    # --- Gravity config (GravityConfig) ---
    grav_cfg = cfg.gravity
    new_backend = str(getattr(grav_cfg, "backend", "classic_sh") or "classic_sh")
    new_surrogate_dir = str(getattr(grav_cfg, "st_lrps_model_dir", "") or "")
    if args.gravity_backend is not None:
        new_backend = str(args.gravity_backend)
    if args.surrogate_gravity_model_dir is not None:
        new_surrogate_dir = str(Path(str(args.surrogate_gravity_model_dir)).expanduser().resolve())
    grav_cfg = replace(grav_cfg, backend=new_backend, st_lrps_model_dir=new_surrogate_dir)
    if args.gravity_file_path is not None:
        grav_cfg = replace(grav_cfg, file_path=str(args.gravity_file_path))
    if args.degree is not None:
        grav_cfg = replace(grav_cfg, degree=int(args.degree))

    if args.adaptive_enabled is not None:
        grav_cfg = replace(grav_cfg, adaptive=replace(grav_cfg.adaptive, enabled=bool(args.adaptive_enabled)))
    if args.adaptive_table is not None:
        grav_cfg = replace(grav_cfg, adaptive=replace(grav_cfg.adaptive, altitude_table=args.adaptive_table))
        if args.adaptive_enabled is None:
            grav_cfg = replace(grav_cfg, adaptive=replace(grav_cfg.adaptive, enabled=True))

    cfg = replace(cfg, gravity=grav_cfg)

    # --- Propagator config (PropagatorConfig) ---
    prop_cfg = cfg.propagator
    if args.method is not None:
        prop_cfg = replace(prop_cfg, method=str(args.method).strip())
    if args.user_max_step_s is not None:
        prop_cfg = replace(prop_cfg, user_max_step_s=float(args.user_max_step_s))
    if args.rtol is not None:
        prop_cfg = replace(prop_cfg, rtol=float(args.rtol))
    if args.atol is not None:
        prop_cfg = replace(prop_cfg, atol=float(args.atol))
    cfg = replace(cfg, propagator=prop_cfg)

    # --- Output config (OutputConfig) ---
    out_cfg = cfg.output
    if args.out_dir is not None:
        out_cfg = replace(out_cfg, out_dir=Path(str(args.out_dir)).expanduser())
    if args.make_3d_plots is not None:
        out_cfg = replace(out_cfg, make_3d_plots=bool(args.make_3d_plots))
    if args.downsample_3d is not None:
        out_cfg = replace(out_cfg, downsample_3d=int(args.downsample_3d))
    cfg = replace(cfg, output=out_cfg)

    # --- Kernel dir remap (strict by filename) ---
    if args.kernel_dir is not None:
        kd = Path(str(args.kernel_dir)).expanduser().resolve()
        new_kernels = tuple(str(kd / Path(str(k)).name) for k in cfg.spice.kernels)
        cfg = replace(cfg, spice=replace(cfg.spice, kernels=new_kernels))

    # Final validation (fail-fast)
    cfg.validate()
    return cfg


__all__ = [
    "str2bool",
    "parse_adaptive_table",
    "resolve_orbit_elements",
    "init_surface_provider",
    "need_ephemeris",
    "apply_args_to_config",
]
