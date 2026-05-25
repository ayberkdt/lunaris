# -*- coding: utf-8 -*-
"""
ST_LRPS - main entry point (STRICT, config.py + common.type_defs aligned)

Contract
--------
- config.py (SimConfig) is the single source of truth (SSOT).
- main.py only applies explicit CLI overrides and wires modules together.
- Shared, pure CLI helpers live in cli.common_args (import-safe).
- NO backward-compat aliases / legacy flags / schema-drift adapters.
"""

# =============================================================================
# 0.                               IMPORTS
# =============================================================================

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Sequence, TYPE_CHECKING

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent

# Determine project root by locating 'config.py'
if (SCRIPT_DIR / "config.py").is_file():
    PROJECT_ROOT = SCRIPT_DIR
elif (SCRIPT_DIR.parent / "config.py").is_file():
    PROJECT_ROOT = SCRIPT_DIR.parent
else:
    PROJECT_ROOT = SCRIPT_DIR

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))



from config import load_default_config, SimConfig  # noqa: E402

# Common layer is dependency-light and safe to import at module import time.
from common.constants import R_MOON, MU_MOON, DEG2RAD, DAY_S  # noqa: E402
from common.time_utils import normalize_iso_datetime_to_utc_string  # noqa: E402
from common.type_defs import InitialState, PropagationResult  # noqa: E402
from models.gravity_adapter import adapt_gravity_model as _shared_adapt_gravity_model  # noqa: E402

# Shared, pure CLI helpers (import-safe; heavy imports are lazy inside them).
from cli.common_args import (  # noqa: E402
    apply_args_to_config,
    init_surface_provider,
    need_ephemeris,
    parse_adaptive_table,
    resolve_orbit_elements,
    str2bool,
)

# Heavy modules (models/core/loaders/analysis) are intentionally NOT imported
# at module import time. They are imported inside runtime functions (main/init_*)
# to keep CLI startup and `import main` resilient even without optional deps.

if TYPE_CHECKING:
    # Typing-only imports (no runtime cost)
    from models.ephemeris import EphemerisManager
    from core.dynamics import DynamicsEngine



# =============================================================================
# 1.                            HELPERS
# =============================================================================
#
# Shared, pure CLI helpers (str2bool, parse_adaptive_table, resolve_orbit_elements,
# init_surface_provider, need_ephemeris, apply_args_to_config) live in
# cli.common_args and are imported above. Only runtime-heavy wiring helpers
# remain in this module.


def init_ephemeris(cfg: SimConfig, tf_s: float) -> "EphemerisManager":
    """Build ephemeris tables using strict EphemerisManager factory.

    Notes:
    - Uses cfg.time.start_date and cfg.time.output_dt_s as the sampling grid.
    - Adds a small duration buffer to avoid interpolation edge issues near tf.
    - Derives whether Sun/Earth vector tables are needed from the active force
      model flags. SH/topography-only runs still get Moon-fixed attitude data,
      but they no longer pay for unnecessary third-body sampling.
    """
    start_utc = str(cfg.time.start_date).strip()
    if not start_utc:
        raise ValueError("cfg.time.start_date is empty.")

    tf_s_buffered = float(tf_s) + 0.1 * DAY_S
    time_cfg = replace(cfg.time, duration_s=tf_s_buffered)
    flags = cfg.flags

    need_body_vectors = bool(
        flags.enable_3rd_body_sun
        or flags.enable_3rd_body_earth
        or flags.enable_earth_j2
        or flags.enable_srp
        or flags.enable_albedo
        or flags.enable_thermal
        or flags.enable_tides_k2
        or flags.enable_tides_k3
    )
    spice_cfg = replace(cfg.spice, include_third_body=need_body_vectors)

    # Local import: models.ephemeris can be heavy (spiceypy/numba)
    from models.ephemeris import EphemerisManager

    return EphemerisManager.from_time_and_spice(
        time_cfg,
        spice_cfg,
        auto_fix_kernel_paths=True,
        need_moon_fixed_rotation=True,
    )


def _extract_rv6(
    y0: Any,
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[float], Optional[float]]:
    """Extract (rx, ry, rz, vx, vy, vz) from strict initial-state container styles.

    Supported:
      - common.type_defs.InitialState: attributes x,y,z,vx,vy,vz or .to_array()
      - core.state.OrbitState: packed vector via .y (len>=6)
      - Generic containers: attributes position/velocity or r_m/v_ms (3,)
      - Array-like (len>=6)

    Returns (None,... ) if extraction fails.
    """
    try:
        if y0 is None:
            return (None, None, None, None, None, None)

        # SSOT initial state dataclass (common.type_defs.InitialState)
        if hasattr(y0, "to_array"):
            arr = getattr(y0, "to_array")()
            return (float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3]), float(arr[4]), float(arr[5]))

        if all(hasattr(y0, k) for k in ("x", "y", "z", "vx", "vy", "vz")):
            return (
                float(getattr(y0, "x")),
                float(getattr(y0, "y")),
                float(getattr(y0, "z")),
                float(getattr(y0, "vx")),
                float(getattr(y0, "vy")),
                float(getattr(y0, "vz")),
            )

        # Other state containers
        if hasattr(y0, "position") and hasattr(y0, "velocity"):
            r = getattr(y0, "position")
            v = getattr(y0, "velocity")
            return (float(r[0]), float(r[1]), float(r[2]), float(v[0]), float(v[1]), float(v[2]))

        if hasattr(y0, "r_m") and hasattr(y0, "v_ms"):
            r = getattr(y0, "r_m")
            v = getattr(y0, "v_ms")
            return (float(r[0]), float(r[1]), float(r[2]), float(v[0]), float(v[1]), float(v[2]))

        if hasattr(y0, "y"):
            y = getattr(y0, "y")
            return (float(y[0]), float(y[1]), float(y[2]), float(y[3]), float(y[4]), float(y[5]))

        y = y0  # assume array-like
        return (float(y[0]), float(y[1]), float(y[2]), float(y[3]), float(y[4]), float(y[5]))
    except Exception:
        return (None, None, None, None, None, None)


def print_summary(cfg: SimConfig, orbit_params: Optional[Dict[str, float]], y0: Any) -> None:
    """Pretty-print a run summary (CLI-oriented)."""
    f = cfg.flags
    sc = cfg.spacecraft

    print("=" * 64)
    print("ST_LRPS RUNNER (STRICT)")
    print("=" * 64)
    print("[Time]")
    print(f"  start_date   : {cfg.time.start_date}")
    print(f"  duration     : {cfg.time.duration_s:.1f} s  ({cfg.time.duration_days:.6f} days)")
    print(f"  output_dt_s  : {cfg.time.output_dt_s}")
    print(f"  samples/period (if output_dt_s is None): {cfg.time.samples_per_period}")
    print("")
    print("[Output]")
    print(f"  out_dir      : {cfg.output.out_dir}")
    print(f"  make_3d_plots : {cfg.output.make_3d_plots}")
    print(f"  downsample_3d : {cfg.output.downsample_3d}")
    print("")
    print("[Spacecraft]")
    print(f"  mass_kg      : {sc.mass_kg}")
    print(f"  area_m2      : {sc.area_m2}")
    print(f"  cd / cr      : {sc.cd} / {sc.cr}")
    print("")
    print("[Gravity]")
    print(f"  backend      : {cfg.gravity.backend}")
    if cfg.gravity.uses_st_lrps:
        print(f"  model_dir    : {cfg.gravity.st_lrps_model_dir}")
    else:
        print(f"  file_path    : {cfg.gravity.file_path}")
        print(f"  degree       : {cfg.gravity.degree}")
        print(f"  adaptive     : enabled={cfg.gravity.adaptive.enabled} table={cfg.gravity.adaptive.altitude_table is not None}")
    print("")
    print("[Forces]")
    print(f"  High-fidelity gravity: {f.enable_sh}")
    print(f"  Third-body Sun       : {f.enable_3rd_body_sun}")
    print(f"  Third-body Earth     : {f.enable_3rd_body_earth}")
    print(f"  Earth J2             : {f.enable_earth_j2}")
    print(f"  SRP                  : {f.enable_srp}")
    print(f"  Albedo               : {f.enable_albedo}")
    print(f"  Thermal              : {f.enable_thermal}")
    tides_on = bool(f.enable_tides_k2 or f.enable_tides_k3)
    tides_kind = "k3" if f.enable_tides_k3 else ("k2" if f.enable_tides_k2 else "off")
    print(f"  Tides                : {tides_on} (kind={tides_kind})")
    print(f"  Relativity (1PN)     : {f.enable_relativity_1pn}")
    print("")
    print("[Initial State]")
    if orbit_params:
        print(f"  COE: a={orbit_params['a_km']:.3f} km e={orbit_params['e']:.6f} i={orbit_params['inc_deg']:.3f} deg")
    else:
        print("  COE: (from cfg.initial_state)")

    rx, ry, rz, vx, vy, vz = _extract_rv6(y0)
    if rx is None:
        print(f"  r0 [m]  : (unavailable; initial_state={type(y0).__name__})")
        print("  v0 [m/s]: (unavailable)")
    else:
        print(f"  r0 [m]  : ({rx:.3f}, {ry:.3f}, {rz:.3f})")
        print(f"  v0 [m/s]: ({vx:.6f}, {vy:.6f}, {vz:.6f})")

    print("=" * 64)


def median_dt(t_arr: Any) -> Optional[float]:
    """Median sampling interval for a time array."""
    try:
        t = np.asarray(t_arr, dtype=float).ravel()
        if t.size < 3:
            return None
        dt = np.diff(t)
        if dt.size == 0:
            return None
        return float(np.median(dt))
    except Exception:
        return None



# =============================================================================
# 2.                                  CLI
# =============================================================================

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ST_LRPS Runner (STRICT; config.py + common.type_defs aligned)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---------------------------
    # Time (TimeConfig)
    # ---------------------------
    g_time = parser.add_argument_group("Time")
    g_time.add_argument(
        "--start-date",
        type=str,
        help=(
            "UTC start date in ISO format. Naive timestamps are interpreted as UTC; "
            "explicit offsets are accepted and normalized to UTC "
            "(e.g. 2026-01-19T12:00:00Z or 2026-01-19T15:00:00+03:00)."
        ),
    )
    dur = g_time.add_mutually_exclusive_group()
    dur.add_argument("--days", type=float, help="Simulation duration [days]")
    dur.add_argument("--hours", type=float, help="Simulation duration [hours]")

    # aligned with TimeConfig.output_dt_s
    g_time.add_argument("--output-dt-s", type=float, help="Fixed output spacing [s] (omit to keep config default)")
    g_time.add_argument("--samples-per-period", type=int, help="Used when output_dt_s is None")

    # ---------------------------
    # Orbit init
    # ---------------------------
    g_orbit = parser.add_argument_group("Orbit Init (choose one)")
    g_orbit.add_argument("--hp-km", type=float, help="Periselene altitude [km]")
    g_orbit.add_argument("--ha-km", type=float, help="Aposelene altitude [km]")
    g_orbit.add_argument("--a-km", type=float, help="Semi-major axis [km]")
    g_orbit.add_argument("--e", type=float, help="Eccentricity (0 <= e < 1)")
    g_orbit.add_argument("--alt-km", type=float, help="Circular orbit altitude [km]")

    g_orbit.add_argument("--inc-deg", type=float, help="Inclination [deg]")
    g_orbit.add_argument("--raan-deg", type=float, help="RAAN [deg]")
    g_orbit.add_argument("--argp-deg", type=float, help="Argument of periapsis [deg]")
    g_orbit.add_argument("--ta-deg", type=float, help="True anomaly [deg]")

    # ---------------------------
    # Physics flags (PerturbationFlags)
    # ---------------------------
    g_phys = parser.add_argument_group("Physics Flags")
    g_phys.add_argument("--enable-sh", type=str2bool, help="Enable spherical harmonics gravity (on/off)")
    g_phys.add_argument("--enable-3rd-body-sun", type=str2bool, help="Enable Sun third-body (on/off)")
    g_phys.add_argument("--enable-3rd-body-earth", type=str2bool, help="Enable Earth third-body (on/off)")
    g_phys.add_argument("--enable-earth-j2", type=str2bool, help="Enable differential Earth J2 (on/off)")
    g_phys.add_argument("--enable-srp", type=str2bool, help="Enable SRP (on/off)")
    g_phys.add_argument("--enable-albedo", type=str2bool, help="Enable lunar albedo pressure (on/off)")
    g_phys.add_argument("--enable-thermal", type=str2bool, help="Enable lunar thermal pressure (on/off)")

    # clean tides contract -> maps to enable_tides_k2/enable_tides_k3
    g_phys.add_argument("--enable-tides", type=str2bool, help="Enable solid tides (on/off)")
    g_phys.add_argument("--tides-kind", choices=("k2", "k3"), help="Tides model kind (k2 or k3)")

    g_phys.add_argument("--enable-relativity-1pn", type=str2bool, help="Enable relativity 1PN (on/off)")

    # ---------------------------
    # Gravity model (GravityConfig)
    # ---------------------------
    g_grav = parser.add_argument_group("Gravity Model")
    g_grav.add_argument(
        "--gravity-backend",
        choices=("classic_sh", "st_lrps"),
        help="Central gravity backend: classical spherical harmonics or ST-LRPS surrogate.",
    )
    g_grav.add_argument("--gravity-file-path", type=str, help="Gravity model file path (.tab/.gfc/.shbdr)")
    g_grav.add_argument(
        "--surrogate-gravity-model-dir",
        type=str,
        help="Trained ST-LRPS gravity run directory (config.json + a checkpoint under checkpoints/)",
    )
    g_grav.add_argument("--degree", type=int, help="Max SH degree (Nmax)")
    g_grav.add_argument("--adaptive-enabled", type=str2bool, help="Enable adaptive SH degree (on/off)")
    g_grav.add_argument("--adaptive-table", type=parse_adaptive_table, help="alt:deg,alt:deg (ascending)")

    # ---------------------------
    # Spacecraft (SpacecraftProps)
    # ---------------------------
    g_sc = parser.add_argument_group("Spacecraft")
    g_sc.add_argument("--mass-kg", type=float, help="Mass [kg]")
    g_sc.add_argument("--area-m2", type=float, help="Area [m^2]")
    g_sc.add_argument("--cd", type=float, help="Cd [-]")
    g_sc.add_argument("--cr", type=float, help="Cr [-]")

    # ---------------------------
    # Numerics (PropagatorConfig)
    # ---------------------------
    g_num = parser.add_argument_group("Numerics")
    g_num.add_argument("--method", type=str, help="Integrator method string (e.g. DOP853, RK45, VV)")
    g_num.add_argument("--user-max-step-s", type=float, help="Max internal solver step [s]")
    g_num.add_argument("--rtol", type=float, help="Relative tolerance")
    g_num.add_argument("--atol", type=float, help="Absolute tolerance")

    # ---------------------------
    # Output & Assets (OutputConfig + assets)
    # ---------------------------
    g_io = parser.add_argument_group("I/O & Assets")
    g_io.add_argument("--out-dir", type=str, help="Output directory")
    g_io.add_argument("--make-3d-plots", type=str2bool, help="Generate 3D plots/animation outputs (on/off)")
    g_io.add_argument("--downsample-3d", type=int, help="3D plot downsample factor")
    g_io.add_argument("--kernel-dir", type=str, help="Directory containing SPICE kernels (renames by filename match)")
    g_io.add_argument("--ldem-root", type=str, help="LOLA LDEM root directory")
    g_io.add_argument("--albedo-root", type=str, help="LOLA Albedo root directory")
    g_io.add_argument("--ldem-ppd", type=int, help="Surface resolution (pixels per degree)")

    args = parser.parse_args(args=argv)
    validate_args(parser, args)
    return args


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    # Orbit init validation
    hp_ha_any = (args.hp_km is not None) or (args.ha_km is not None)
    a_e_any = (args.a_km is not None) or (args.e is not None)
    alt = args.alt_km is not None

    if hp_ha_any and (args.hp_km is None or args.ha_km is None):
        parser.error("Provide BOTH --hp-km and --ha-km together.")
    if a_e_any and (args.a_km is None or args.e is None):
        parser.error("Provide BOTH --a-km and --e together.")
    if args.e is not None and not (0.0 <= args.e < 1.0):
        parser.error("--e must satisfy 0 <= e < 1.")
    if (args.hp_km is not None and args.ha_km is not None) and (args.a_km is not None or args.e is not None):
        parser.error("Choose ONE orbit init mode: (--hp-km,--ha-km) OR (--a-km,--e) OR (--alt-km).")
    if alt and (hp_ha_any or a_e_any):
        parser.error("Choose ONE orbit init mode: (--alt-km) cannot be combined with other orbit init flags.")


    # If user provides orbital angles without an explicit orbit-init mode, fail fast.
    angles_any = any(
        getattr(args, k) is not None
        for k in ("inc_deg", "raan_deg", "argp_deg", "ta_deg")
    )
    base_any = (
        (args.hp_km is not None and args.ha_km is not None)
        or (args.a_km is not None and args.e is not None)
        or alt
    )
    if angles_any and not base_any:
        parser.error(
            "Orbit angle flags (--inc-deg/--raan-deg/--argp-deg/--ta-deg) require an orbit init mode "
            "(--hp-km/--ha-km or --a-km/--e or --alt-km)."
        )

    # start-date format (if provided)
    if args.start_date is not None:
        s = str(args.start_date).strip()
        try:
            normalize_iso_datetime_to_utc_string(s, precision=0)
        except Exception as exc:
            parser.error(
                "--start-date must be an ISO-like timestamp such as "
                "yyyy-MM-ddTHH:mm:ssZ or yyyy-MM-ddTHH:mm:ss+03:00 "
                f"(details: {exc})"
            )

    # numeric sanity
    if args.days is not None and args.days <= 0:
        parser.error("--days must be positive.")
    if args.hours is not None and args.hours <= 0:
        parser.error("--hours must be positive.")
    if args.output_dt_s is not None and args.output_dt_s <= 0:
        parser.error("--output-dt-s must be positive.")
    if args.samples_per_period is not None and args.samples_per_period < 2:
        parser.error("--samples-per-period must be >= 2.")
    if args.degree is not None and args.degree < 0:
        parser.error("--degree must be >= 0.")
    if args.downsample_3d is not None and args.downsample_3d < 1:
        parser.error("--downsample-3d must be >= 1.")
    if args.ldem_ppd is not None and args.ldem_ppd <= 0:
        parser.error("--ldem-ppd must be positive.")
    if args.user_max_step_s is not None and args.user_max_step_s <= 0:
        parser.error("--user-max-step-s must be positive.")

    # adaptive table implies adaptive enabled unless user explicitly disabled
    if args.adaptive_table is not None and args.adaptive_enabled is False:
        parser.error("--adaptive-table requires --adaptive-enabled on (or omit --adaptive-enabled).")

    # tides-kind implies enable-tides unless user explicitly forced off
    if args.tides_kind is not None and args.enable_tides is False:
        parser.error("--tides-kind requires --enable-tides on (or omit --enable-tides).")

    # path sanity
    if args.kernel_dir is not None:
        kd = Path(str(args.kernel_dir)).expanduser()
        if not kd.exists() or not kd.is_dir():
            parser.error(f"--kernel-dir must be an existing directory: {kd}")

    if args.ldem_root is not None:
        p = Path(str(args.ldem_root)).expanduser()
        if not p.exists() or not p.is_dir():
            parser.error(f"--ldem-root must be an existing directory: {p}")

    if args.albedo_root is not None:
        p = Path(str(args.albedo_root)).expanduser()
        if not p.exists() or not p.is_dir():
            parser.error(f"--albedo-root must be an existing directory: {p}")

    if args.surrogate_gravity_model_dir is not None:
        # Artifact validation: directory existence, config.json, and a usable
        # checkpoint (ckpt_best.pt OR ckpt_last.pt) are delegated to the
        # canonical helper. Do NOT reimplement these checks here.
        from common.montecarlo_defs import validate_st_lrps_model_dir

        try:
            model_dir = validate_st_lrps_model_dir(args.surrogate_gravity_model_dir)
        except ValueError as exc:
            parser.error(f"--surrogate-gravity-model-dir: {exc}")

        # Semantic validation (distinct from artifact validation above): confirm
        # the run was trained on a lunar gravity config. Not covered by the
        # artifact helper, so it is kept here and clearly separated.
        try:
            from st_lrps.data.dataset_parameters import looks_like_lunar_run_config
        except ImportError:
            looks_like_lunar_run_config = None
        if looks_like_lunar_run_config is not None:
            run_cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
            if not looks_like_lunar_run_config(run_cfg):
                parser.error(
                    "--surrogate-gravity-model-dir does not look like a lunar-trained ST-LRPS run: "
                    f"{model_dir}"
                )


# =============================================================================
# 3.                                 MAIN
# =============================================================================

def _y0_to_array(y0: Any) -> np.ndarray:
    """Strict: produce a 1D float array (>=6) for propagate()."""
    if y0 is None:
        raise ValueError("Initial state (y0) is None.")

    # common.type_defs.InitialState
    if hasattr(y0, "to_array"):
        arr = np.asarray(getattr(y0, "to_array")(), dtype=float).reshape(-1)
    # core.state.OrbitState (or similar): packed vector via `.y`
    elif hasattr(y0, "y"):
        arr = np.asarray(getattr(y0, "y"), dtype=float).reshape(-1)
    # Plain object with x,y,z,vx,vy,vz
    elif all(hasattr(y0, k) for k in ("x", "y", "z", "vx", "vy", "vz")):
        arr = np.asarray(
            (
                getattr(y0, "x"), getattr(y0, "y"), getattr(y0, "z"),
                getattr(y0, "vx"), getattr(y0, "vy"), getattr(y0, "vz"),
            ),
            dtype=float,
        ).reshape(-1)
    else:
        arr = np.asarray(y0, dtype=float).reshape(-1)

    if arr.size < 6:
        raise ValueError(f"Initial state must have at least 6 elements, got {arr.size}.")
    return arr.astype(float, copy=False)


def _adapt_gravity_model(g: Any) -> Any:
    """Normalize gravity-model attributes through the shared adapter module."""
    return _shared_adapt_gravity_model(g)


def main() -> int:
    args = parse_args()

    # Load & override
    try:
        cfg = load_default_config()
        cfg = apply_args_to_config(cfg, args)
    except Exception as e:
        print(f"[FATAL] Config init failed: {e}")
        return 1

    # Ensure output dir exists
    try:
        out_dir = cfg.output.ensure_out_dir()
    except Exception as e:
        print(f"[FATAL] Output directory failure: {e}")
        return 1

    # Gravity model (STRICT)
    gravity_core = None
    mu = float(MU_MOON)
    try:
        if bool(cfg.flags.enable_sh) and cfg.gravity.uses_st_lrps:
            from models.surrogate_gravity import SurrogateGravityModel

            gravity_core = SurrogateGravityModel.from_model_dir(
                cfg.gravity.st_lrps_model_dir,
                mu_override=float(MU_MOON),
                r_ref_override=float(R_MOON),
                device_preference="cpu",
            )
            mu = float(getattr(gravity_core, "GM_m3s2", MU_MOON))
        else:
            # Local import: spherical harmonics can trigger Numba compilation
            from models.spherical_harmonics import GravityModel

            deg = int(cfg.gravity.degree) if cfg.gravity.degree is not None else None
            gravity = GravityModel.from_file(
                path=str(cfg.gravity.file_path),
                requested_degree=deg,
            )
            gravity_core = _adapt_gravity_model(gravity) if bool(cfg.flags.enable_sh) else None
            # Prefer model's mu (m^3/s^2); fallback to constants
            mu = float(getattr(gravity, "mu", MU_MOON))
    except Exception as e:
        print(f"[FATAL] Gravity model init failed: {e}")
        return 1

    # Surface grids (CLI-requested only)
    topo_requested = bool(args.ldem_root or args.albedo_root)
    surface_provider: Optional[Any] = None
    if topo_requested or cfg.flags.enable_surface_forces:
        try:
            surface_provider = init_surface_provider(args)
        except Exception as e:
            print(f"[FATAL] Surface grids load failed: {e}")
            return 1

        if cfg.flags.enable_surface_forces and surface_provider is None:
            print("[FATAL] Surface forces enabled, but no surface grids loaded. Provide --ldem-root/--albedo-root.")
            return 1

    # Topography grid for topo-aware impact events (optional)
    topo_grid = None
    if surface_provider is not None and hasattr(surface_provider, "grids"):
        try:
            topo_grid = surface_provider.grids().topo  # type: ignore[attr-defined]
        except Exception:
            topo_grid = None

    # Ephemeris if needed
    ephem_mgr: Optional[EphemerisManager] = None
    if need_ephemeris(cfg, topo_requested=topo_requested):
        try:
            ephem_mgr = init_ephemeris(cfg, tf_s=float(cfg.time.duration_s))
        except Exception as e:
            print(f"[FATAL] Ephemeris init failed: {e}")
            return 1

    # Initial state: if orbit init flags provided -> COE -> Cartesian; else cfg.initial_state
    orbit_params: Optional[Dict[str, float]] = None
    y0: InitialState = cfg.initial_state

    orbit_init_requested = any(
        v is not None
        for v in (
            args.hp_km, args.ha_km, args.a_km, args.e, args.alt_km,
            args.inc_deg, args.raan_deg, args.argp_deg, args.ta_deg,
        )
    )
    if orbit_init_requested:
        try:
            orbit_params = resolve_orbit_elements(args)
            a_m = float(orbit_params["a_km"]) * 1000.0
            e = float(orbit_params["e"])
            inc = float(orbit_params["inc_deg"]) * DEG2RAD
            raan = float(orbit_params["raan_deg"]) * DEG2RAD
            argp = float(orbit_params["argp_deg"]) * DEG2RAD
            ta = float(orbit_params["ta_deg"]) * DEG2RAD

            # Canonical SSOT conversion (no silent fallback): a failure here is fatal.
            from core.state import create_state_from_keplerian

            y0 = create_state_from_keplerian(
                semi_major_axis=a_m,
                eccentricity=e,
                inclination=inc,
                raan=raan,
                argp=argp,
                true_anomaly=ta,
                mu=mu,
            )
        except Exception as e:
            print(f"[FATAL] Orbit init failed: {e}")
            return 1

    print_summary(cfg, orbit_params, y0)

    # Build dynamics engine
    try:
        # Local import: avoid importing core at module import time
        from core.dynamics import DynamicsEngine

        engine = DynamicsEngine(
            sc_props=cfg.spacecraft,
            flags=cfg.flags,
            gravity_model=gravity_core,
            gravity_adaptive=(None if cfg.gravity.uses_st_lrps else cfg.gravity.adaptive),
            ephem_manager=ephem_mgr,
            surface_provider=surface_provider,
            earth_j2=cfg.earth_j2,
        )
        _ = engine.build_rhs()  # triggers warmup / JIT (if enabled)
    except Exception as e:
        print(f"[FATAL] Dynamics engine init failed: {e}")
        return 1

    # Propagate
    print(f"[RUN] Propagating for {cfg.time.duration_days:.6f} days ...")
    t0 = time.perf_counter()
    try:
        # Local import: avoid importing core at module import time
        from core.propagator import propagate

        result: PropagationResult = propagate(
            dynamics=engine,
            y0=_y0_to_array(y0),
            cfg=cfg.propagator,
            time_cfg=cfg.time,
            topo_grid=topo_grid,
        )
    except Exception as e:
        print(f"[FATAL] Propagation failed: {e}")
        return 1

    t_prop = time.perf_counter() - t0
    print(f"[DONE] Propagation finished in {t_prop:.3f} s.")

    # Save config snapshot
    try:
        with open(out_dir / "run_config.json", "w", encoding="utf-8") as f:
            json.dump(asdict(cfg), f, indent=2, default=str)
    except Exception:
        print("[WARN] Could not write run_config.json")

    # Metadata (derive dt from result.t if possible)
    dt_used = None
    if getattr(result, "t", None) is not None:
        dt_used = median_dt(result.t)

    meta = {
        "propagator_method": cfg.propagator.method,
        "rtol": cfg.propagator.rtol,
        "atol": cfg.propagator.atol,
        "output_dt_s": cfg.time.output_dt_s,          # strict key
        "output_dt_s_measured": dt_used,              # optional diagnostic
        "degree": cfg.gravity.degree,
        "mu_m3s2": mu,
        "spacecraft": {
            "mass_kg": cfg.spacecraft.mass_kg,
            "area_m2": cfg.spacecraft.area_m2,
            "cd": cfg.spacecraft.cd,
            "cr": cfg.spacecraft.cr,
        },
        "propagation_time_s": t_prop,
        "duration_s": cfg.time.duration_s,
    }

    # Reports / plots
    try:
        from analysis.postprocess import process_simulation_results
        from analysis.reporting.manager import plot_all

        hist = process_simulation_results(result, ctx=engine, cfg=cfg)
        plot_all(
            history=hist,
            out_dir=str(out_dir),
            meta=meta,
            ctx=engine,
            title_prefix="ST_LRPS",
            use_run_subdir=True,
            visual_cfg=cfg.visual,
            save_pdf=True,
        )
    except ImportError:
        print("[WARN] analysis.reporting.manager not found; skipping plots.")
    except Exception as e:
        print(f"[ERROR] Plot/report failed: {e}")

    # 3D visualization (optional)
    if cfg.output.make_3d_plots:
        try:
            from visualization.orbit_animation import render_orbit_animation

            render_orbit_animation(
                result=result,
                config=cfg,
                output_file=str(out_dir / "orbit_3d.mp4"),
            )
        except ImportError:
            print("[WARN] visualization.orbit_animation not found; skipping 3D render.")
        except Exception as e:
            print(f"[ERROR] 3D render failed: {e}")

    print("[OK] Finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
