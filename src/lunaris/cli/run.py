"""Runtime wiring for the main ``lunaris`` propagation command.

The CLI command stays thin: parse arguments, apply them to ``SimConfig``, build
runtime providers lazily, propagate, and hand results to reporting.
"""

from __future__ import annotations

import json
import time
import traceback
from argparse import Namespace
from dataclasses import asdict, dataclass, replace
from pathlib import Path
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
from lunaris.common.state_vector import normalize_cartesian_state
from lunaris.common.type_defs import PropagationResult
from lunaris.core.config import SimConfig, load_default_config

if TYPE_CHECKING:
    from lunaris.physics.ephemeris import EphemerisManager


_EXPECTED_RUNTIME_EXCEPTIONS: tuple[type[Exception], ...] = (
    FileNotFoundError,
    PermissionError,
    OSError,
    ImportError,
    ValueError,
    RuntimeError,
)


@dataclass(slots=True)
class SurfaceSetup:
    provider: Any | None
    topo_grid: Any | None
    topo_requested: bool


@dataclass(slots=True)
class PropagationRun:
    result: PropagationResult
    elapsed_s: float


def _emit_failure(stage: str, exc: Exception, *, debug_tracebacks: bool) -> None:
    expected = isinstance(exc, _EXPECTED_RUNTIME_EXCEPTIONS)
    prefix = "[FATAL]" if expected else "[FATAL:UNEXPECTED]"
    print(f"{prefix} {stage}: {exc}")
    if not expected and debug_tracebacks:
        traceback.print_exc()


def _emit_optional_failure(stage: str, exc: Exception, *, debug_tracebacks: bool) -> None:
    expected = isinstance(exc, _EXPECTED_RUNTIME_EXCEPTIONS)
    prefix = "[ERROR]" if expected else "[ERROR:UNEXPECTED]"
    print(f"{prefix} {stage}: {exc}")
    if not expected and debug_tracebacks:
        traceback.print_exc()


def load_runtime_config(args: Namespace) -> SimConfig:
    cfg = load_default_config()
    return apply_args_to_config(cfg, args)


def build_gravity_provider(cfg: SimConfig) -> tuple[Any | None, float]:
    gravity_core: Any | None = None
    mu = float(MU_MOON)
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
        # Local import: spherical harmonics can trigger Numba compilation.
        from lunaris.physics.spherical_harmonics import GravityModel

        deg = int(cfg.gravity.degree) if cfg.gravity.degree is not None else None
        gravity = GravityModel.from_file(
            path=str(cfg.gravity.file_path),
            requested_degree=deg,
        )
        # GravityModel already satisfies the dynamics gravity contract directly.
        gravity_core = gravity if bool(cfg.flags.enable_sh) else None
        mu = float(getattr(gravity, "mu", MU_MOON))
    return gravity_core, mu


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
    """Strict: produce the exact 6/7-element float64 vector propagate() supports.

    Accepts ``InitialState`` (``to_array()``), ``OrbitState``-like packed ``.y``
    vectors, plain x/y/z/vx/vy/vz records, or raw array-likes. Oversized states
    are rejected here rather than failing later inside DynamicsEngine.
    """
    return normalize_cartesian_state(y0, allow_mass=True, name="Initial state")


def build_surface_provider_if_needed(cfg: SimConfig, args: Namespace) -> SurfaceSetup:
    # Surface grids (CLI-requested only). Whether the active force set needs a
    # surface provider comes from the shared force-requirements SSOT, so this
    # stays in lockstep with SimConfig.validate() and DynamicsEngine.
    topo_requested = bool(args.ldem_root or args.albedo_root)
    surface_provider: Any | None = None
    surface_req = force_requirements_for_config(cfg)
    if topo_requested or surface_req.need_surface_provider:
        surface_provider = init_surface_provider(args)

        if surface_req.albedo_needs_provider and surface_provider is None:
            raise RuntimeError(
                "Albedo grid mode enabled, but no albedo grids loaded. "
                "Provide --albedo-root or use --albedo-mode constant_albedo."
            )
        if surface_req.use_thermal_grid and surface_provider is None:
            raise RuntimeError(
                "Thermal temperature_grid mode requires surface temperature data. "
                "Provide a compatible surface provider."
            )

    topo_grid = None
    if surface_provider is not None and hasattr(surface_provider, "grids"):
        try:
            topo_grid = surface_provider.grids().topo
        except (AttributeError, TypeError, ValueError):
            topo_grid = None
    return SurfaceSetup(
        provider=surface_provider,
        topo_grid=topo_grid,
        topo_requested=topo_requested,
    )


def build_ephemeris_if_needed(cfg: SimConfig, *, topo_requested: bool) -> EphemerisManager | None:
    if not need_ephemeris(cfg, topo_requested=topo_requested):
        return None
    return init_ephemeris(cfg, tf_s=float(cfg.time.duration_s))


def orbit_init_requested(args: Namespace) -> bool:
    return any(
        v is not None
        for v in (
            args.hp_km,
            args.ha_km,
            args.a_km,
            args.e,
            args.alt_km,
            args.inc_deg,
            args.raan_deg,
            args.argp_deg,
            args.ta_deg,
        )
    )


def resolve_initial_state(
    cfg: SimConfig,
    args: Namespace,
    *,
    mu: float,
) -> tuple[Any, dict[str, float] | None]:
    if not orbit_init_requested(args):
        return cfg.initial_state, None

    orbit_params = resolve_orbit_elements(args)
    a_m = float(orbit_params["a_km"]) * 1000.0
    ecc = float(orbit_params["e"])
    inc = float(orbit_params["inc_deg"]) * DEG2RAD
    raan = float(orbit_params["raan_deg"]) * DEG2RAD
    argp = float(orbit_params["argp_deg"]) * DEG2RAD
    ta = float(orbit_params["ta_deg"]) * DEG2RAD

    # Canonical SSOT conversion (no silent fallback): a failure here is fatal.
    from lunaris.core.state import create_state_from_keplerian

    y0 = create_state_from_keplerian(
        semi_major_axis=a_m,
        eccentricity=ecc,
        inclination=inc,
        raan=raan,
        argp=argp,
        true_anomaly=ta,
        mu=mu,
    )
    return y0, orbit_params


def build_engine(
    cfg: SimConfig,
    *,
    gravity_core: Any | None,
    ephem_mgr: EphemerisManager | None,
    surface_provider: Any | None,
) -> Any:
    # Local import: avoid importing core at module import time.
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
    return engine


def run_propagation(
    engine: Any,
    cfg: SimConfig,
    *,
    y0: Any,
    topo_grid: Any | None,
) -> PropagationRun:
    print(f"[RUN] Propagating for {cfg.time.duration_days:.6f} days ...")
    t0 = time.perf_counter()
    # Local import: avoid importing core at module import time.
    from lunaris.core.propagation.propagator import propagate

    result: PropagationResult = propagate(
        dynamics=engine,
        y0=_y0_to_array(y0),
        cfg=cfg.propagator,
        time_cfg=cfg.time,
        topo_grid=topo_grid,
    )
    elapsed_s = time.perf_counter() - t0
    print(f"[DONE] Propagation finished in {elapsed_s:.3f} s.")
    return PropagationRun(result=result, elapsed_s=elapsed_s)


def write_run_artifacts(
    out_dir: Path,
    cfg: SimConfig,
    diag_payload: dict[str, Any],
) -> None:
    if diag_payload:
        print("[DIAG] " + json.dumps(diag_payload, sort_keys=True))
        try:
            with open(out_dir / "run_diagnostics.json", "w", encoding="utf-8") as f:
                json.dump(diag_payload, f, indent=2, sort_keys=True)
        except OSError:
            print("[WARN] Could not write run_diagnostics.json")

    try:
        with open(out_dir / "run_config.json", "w", encoding="utf-8") as f:
            json.dump(asdict(cfg), f, indent=2, default=str)
    except OSError:
        print("[WARN] Could not write run_config.json")


def build_run_meta(
    cfg: SimConfig,
    result: PropagationResult,
    *,
    mu: float,
    propagation_time_s: float,
) -> dict[str, Any]:
    dt_used = None
    if getattr(result, "t", None) is not None:
        dt_used = median_dt(result.t)

    return {
        "propagator_method": cfg.propagator.method,
        "rtol": cfg.propagator.rtol,
        "atol": cfg.propagator.atol,
        "output_dt_s": cfg.time.output_dt_s,
        "output_dt_s_measured": dt_used,
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
        "propagation_time_s": propagation_time_s,
        "duration_s": cfg.time.duration_s,
    }


def render_reports(
    *,
    result: PropagationResult,
    engine: Any,
    cfg: SimConfig,
    out_dir: Path,
    meta: dict[str, Any],
) -> None:
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


def render_optional_3d(result: PropagationResult, cfg: SimConfig, out_dir: Path) -> None:
    if not cfg.output.make_3d_plots:
        return

    from lunaris.visualization.orbit_animation import render_orbit_animation

    render_orbit_animation(
        result=result,
        config=cfg,
        output_file=str(out_dir / "orbit_3d.mp4"),
    )


def main() -> int:
    args = parse_args()
    debug_tracebacks = bool(getattr(args, "debug_tracebacks", False))

    try:
        cfg = load_runtime_config(args)
    except Exception as e:
        _emit_failure("Config init failed", e, debug_tracebacks=debug_tracebacks)
        return 1

    try:
        out_dir = Path(cfg.output.ensure_out_dir())
    except Exception as e:
        _emit_failure("Output directory failure", e, debug_tracebacks=debug_tracebacks)
        return 1

    try:
        gravity_core, mu = build_gravity_provider(cfg)
    except Exception as e:
        _emit_failure("Gravity model init failed", e, debug_tracebacks=debug_tracebacks)
        return 1

    try:
        surface = build_surface_provider_if_needed(cfg, args)
    except Exception as e:
        _emit_failure("Surface grids load failed", e, debug_tracebacks=debug_tracebacks)
        return 1

    try:
        ephem_mgr = build_ephemeris_if_needed(cfg, topo_requested=surface.topo_requested)
    except Exception as e:
        _emit_failure("Ephemeris init failed", e, debug_tracebacks=debug_tracebacks)
        return 1

    try:
        y0, orbit_params = resolve_initial_state(cfg, args, mu=mu)
    except Exception as e:
        _emit_failure("Orbit init failed", e, debug_tracebacks=debug_tracebacks)
        return 1

    print_summary(cfg, orbit_params, y0)

    try:
        engine = build_engine(
            cfg,
            gravity_core=gravity_core,
            ephem_mgr=ephem_mgr,
            surface_provider=surface.provider,
        )
    except Exception as e:
        _emit_failure("Dynamics engine init failed", e, debug_tracebacks=debug_tracebacks)
        return 1

    try:
        run = run_propagation(
            engine,
            cfg,
            y0=y0,
            topo_grid=surface.topo_grid,
        )
    except Exception as e:
        _emit_failure("Propagation failed", e, debug_tracebacks=debug_tracebacks)
        return 1

    diag_payload = build_run_diagnostics_payload(run.result, cfg.propagator.method)
    try:
        write_run_artifacts(out_dir, cfg, diag_payload)
    except Exception as e:
        print(f"[WARN] Could not write run artifacts: {e}")

    meta = build_run_meta(cfg, run.result, mu=mu, propagation_time_s=run.elapsed_s)

    try:
        render_reports(
            result=run.result,
            engine=engine,
            cfg=cfg,
            out_dir=out_dir,
            meta=meta,
        )
    except ImportError:
        print("[WARN] analysis.reporting.manager not found; skipping plots.")
    except Exception as e:
        _emit_optional_failure("Plot/report failed", e, debug_tracebacks=debug_tracebacks)

    try:
        render_optional_3d(run.result, cfg, out_dir)
    except ImportError:
        print("[WARN] visualization.orbit_animation not found; skipping 3D render.")
    except Exception as e:
        _emit_optional_failure("3D render failed", e, debug_tracebacks=debug_tracebacks)

    print("[OK] Finished.")
    return 0


def main_entry() -> int:
    """Console-script entry point."""
    return main()


__all__ = [
    "build_engine",
    "build_ephemeris_if_needed",
    "build_gravity_provider",
    "build_run_diagnostics_payload",
    "build_run_meta",
    "build_surface_provider_if_needed",
    "init_ephemeris",
    "load_runtime_config",
    "main",
    "main_entry",
    "orbit_init_requested",
    "render_optional_3d",
    "render_reports",
    "resolve_initial_state",
    "run_propagation",
    "write_run_artifacts",
    "_y0_to_array",
]
