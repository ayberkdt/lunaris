"""Runtime wiring for the main ``lunaris`` propagation command.

The CLI command stays thin: parse arguments, apply them to ``SimConfig``, build
runtime providers lazily, propagate, and hand results to reporting.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, replace
from typing import TYPE_CHECKING, Any

import numpy as np

from lunaris.cli.common_args import (
    apply_args_to_config,
    init_surface_provider,
    need_ephemeris,
    resolve_orbit_elements,
)
from lunaris.cli.options import parse_args
from lunaris.cli.summary import median_dt, print_summary
from lunaris.common.constants import DAY_S, DEG2RAD, MU_MOON, R_MOON
from lunaris.common.force_requirements import force_requirements_for_config
from lunaris.common.type_defs import InitialState, PropagationResult
from lunaris.core.config import SimConfig, load_default_config

if TYPE_CHECKING:
    from lunaris.physics.ephemeris import EphemerisManager


def build_run_diagnostics_payload(result: Any, method: str) -> dict[str, Any]:
    """Flatten engine-reported run diagnostics into a JSON-safe dict.

    Sources only values the propagator itself computed
    (``PropagationResult.diagnostics`` plus the impact/stop outcome). Non-finite
    numbers are dropped rather than serialized, so consumers never see ``NaN``
    from e.g. an unavailable ``nfev``.
    """
    payload: dict[str, Any] = {}

    diagnostics = getattr(result, "diagnostics", None) or {}
    for key, value in diagnostics.items():
        if isinstance(value, int | float):
            if np.isfinite(value):
                payload[key] = float(value)
        elif isinstance(value, str | bool):
            payload[key] = value
        elif isinstance(value, list | tuple):
            payload[key] = [str(v) for v in value]

    if method:
        payload["method"] = str(method)

    payload["impacted"] = bool(getattr(result, "impacted", False))
    t_impact = getattr(result, "t_impact_s", None)
    if isinstance(t_impact, int | float) and np.isfinite(t_impact):
        payload["t_impact_s"] = float(t_impact)
    stop_reason = getattr(result, "stop_reason", None)
    if stop_reason:
        payload["stop_reason"] = str(stop_reason)

    return payload


def init_ephemeris(cfg: SimConfig, tf_s: float) -> EphemerisManager:
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
    req = force_requirements_for_config(
        cfg,
        request_external_relativity=True,
    )
    spice_cfg = replace(cfg.spice, include_third_body=req.need_body_vectors)

    # Local import: lunaris.physics.ephemeris can be heavy (spiceypy/numba)
    from lunaris.physics.ephemeris import EphemerisManager

    return EphemerisManager.from_time_and_spice(
        time_cfg,
        spice_cfg,
        auto_fix_kernel_paths=True,
        need_moon_fixed_rotation=True,
    )


def _y0_to_array(y0: Any) -> np.ndarray:
    """Strict: produce a 1D float array (>=6) for propagate()."""
    if y0 is None:
        raise ValueError("Initial state (y0) is None.")

    # common.type_defs.InitialState
    if hasattr(y0, "to_array"):
        arr = np.asarray(y0.to_array(), dtype=float).reshape(-1)
    # core.state.OrbitState (or similar): packed vector via `.y`
    elif hasattr(y0, "y"):
        arr = np.asarray(y0.y, dtype=float).reshape(-1)
    # Plain object with x,y,z,vx,vy,vz
    elif all(hasattr(y0, k) for k in ("x", "y", "z", "vx", "vy", "vz")):
        arr = np.asarray(
            (
                y0.x, y0.y, y0.z,
                y0.vx, y0.vy, y0.vz,
            ),
            dtype=float,
        ).reshape(-1)
    else:
        arr = np.asarray(y0, dtype=float).reshape(-1)

    if arr.size < 6:
        raise ValueError(f"Initial state must have at least 6 elements, got {arr.size}.")
    return arr.astype(float, copy=False)


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
            from lunaris.surrogate.runtime import SurrogateGravityModel

            gravity_core = SurrogateGravityModel.from_model_dir(
                cfg.gravity.st_lrps_model_dir,
                mu_override=float(MU_MOON),
                r_ref_override=float(R_MOON),
                device_preference="cpu",
            )
            mu = float(getattr(gravity_core, "GM_m3s2", MU_MOON))
        else:
            # Local import: spherical harmonics can trigger Numba compilation
            from lunaris.physics.spherical_harmonics import GravityModel

            deg = int(cfg.gravity.degree) if cfg.gravity.degree is not None else None
            gravity = GravityModel.from_file(
                path=str(cfg.gravity.file_path),
                requested_degree=deg,
            )
            # GravityModel already satisfies the dynamics gravity contract
            # (degree_max, R_ref_m, GM_m3s2, Cnm ... ws) directly.
            gravity_core = gravity if bool(cfg.flags.enable_sh) else None
            # Prefer model's mu (m^3/s^2); fallback to constants
            mu = float(getattr(gravity, "mu", MU_MOON))
    except Exception as e:
        print(f"[FATAL] Gravity model init failed: {e}")
        return 1

    # Surface grids (CLI-requested only). Whether the active force set needs a
    # surface provider comes from the shared force-requirements SSOT, so this
    # stays in lockstep with SimConfig.validate() and DynamicsEngine.
    topo_requested = bool(args.ldem_root or args.albedo_root)
    surface_provider: Any | None = None
    surface_req = force_requirements_for_config(cfg)
    if topo_requested or surface_req.need_surface_provider:
        try:
            surface_provider = init_surface_provider(args)
        except Exception as e:
            print(f"[FATAL] Surface grids load failed: {e}")
            return 1

        if surface_req.albedo_needs_provider and surface_provider is None:
            print(
                "[FATAL] Albedo grid mode enabled, but no albedo grids loaded. "
                "Provide --albedo-root or use --albedo-mode constant_albedo."
            )
            return 1
        if surface_req.use_thermal_grid and surface_provider is None:
            print("[FATAL] Thermal temperature_grid mode requires surface temperature data. Provide a compatible surface provider.")
            return 1

    # Topography grid for topo-aware impact events (optional)
    topo_grid = None
    if surface_provider is not None and hasattr(surface_provider, "grids"):
        try:
            topo_grid = surface_provider.grids().topo  # type: ignore[attr-defined]
        except Exception:
            topo_grid = None

    # Ephemeris if needed
    ephem_mgr: EphemerisManager | None = None
    if need_ephemeris(cfg, topo_requested=topo_requested):
        try:
            ephem_mgr = init_ephemeris(cfg, tf_s=float(cfg.time.duration_s))
        except Exception as e:
            print(f"[FATAL] Ephemeris init failed: {e}")
            return 1

    # Initial state: if orbit init flags provided -> COE -> Cartesian; else cfg.initial_state
    orbit_params: dict[str, float] | None = None
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
            from lunaris.core.state import create_state_from_keplerian

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
        from lunaris.core.dynamics import DynamicsEngine

        engine = DynamicsEngine(
            sc_props=cfg.spacecraft,
            flags=cfg.flags,
            gravity_model=gravity_core,
            gravity_adaptive=(None if cfg.gravity.uses_st_lrps else cfg.gravity.adaptive),
            ephem_manager=ephem_mgr,
            surface_provider=surface_provider,
            earth_j2=cfg.earth_j2,
            srp=cfg.srp,
            thermal=cfg.thermal,
            albedo=cfg.albedo,
            solid_tides=cfg.solid_tides,
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
        from lunaris.core.propagation.propagator import propagate

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

    # Structured engine diagnostics: one machine-readable line for the desktop
    # UI plus a JSON file next to run_config.json. This only re-emits values
    # the propagator already computed -- nothing is estimated here.
    diag_payload = build_run_diagnostics_payload(result, cfg.propagator.method)
    if diag_payload:
        print("[DIAG] " + json.dumps(diag_payload, sort_keys=True))
        try:
            with open(out_dir / "run_diagnostics.json", "w", encoding="utf-8") as f:
                json.dump(diag_payload, f, indent=2, sort_keys=True)
        except Exception:
            print("[WARN] Could not write run_diagnostics.json")

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
        "output_epoch_count": int(len(result.t)) if getattr(result, "t", None) is not None else None,
        "output_points_cap": int(cfg.time.max_points_cap),
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
        from lunaris.analysis.postprocess import process_simulation_results
        from lunaris.analysis.reporting.manager import plot_all

        hist = process_simulation_results(result, ctx=engine, cfg=cfg)
        plot_all(
            history=hist,
            out_dir=str(out_dir),
            meta=meta,
            ctx=engine,
            title_prefix="Lunaris",
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
            from lunaris.visualization.orbit_animation import render_orbit_animation

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


def main_entry() -> int:
    """Console-script entry point."""
    return main()


__all__ = ["init_ephemeris", "_y0_to_array", "main", "main_entry"]
