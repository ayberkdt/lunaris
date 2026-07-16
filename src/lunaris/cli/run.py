"""Runtime wiring for the main ``lunaris`` propagation command.

The CLI command stays thin: parse arguments, apply them to ``SimConfig``, build
runtime providers lazily, propagate, and hand results to reporting.
"""

from __future__ import annotations

import contextlib
import json
import sys
import time
import traceback
from argparse import Namespace
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

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
from lunaris.common.type_defs import InitialState, PropagationResult
from lunaris.core.config import SimConfig, load_default_config, replace_sim_config

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


_T = TypeVar("_T")


@dataclass(slots=True)
class SurfaceSetup:
    provider: Any | None
    topo_grid: Any | None
    topo_requested: bool


@dataclass(slots=True)
class PropagationRun:
    result: PropagationResult
    elapsed_s: float


class CliStageError(RuntimeError):
    """Attach user-facing stage context while preserving the original exception."""

    def __init__(self, stage: str, cause: Exception) -> None:
        super().__init__(f"{stage}: {cause}")
        self.stage = str(stage)
        self.cause = cause


def _run_stage(stage: str, operation: Callable[[], _T]) -> _T:
    try:
        return operation()
    except Exception as exc:
        raise CliStageError(stage, exc) from exc


def _unwrap_stage_failure(stage: str, exc: Exception) -> tuple[str, Exception]:
    if isinstance(exc, CliStageError):
        return exc.stage, exc.cause
    return stage, exc


def _emit_failure(stage: str, exc: Exception, *, debug_tracebacks: bool) -> None:
    stage, root_exc = _unwrap_stage_failure(stage, exc)
    expected = isinstance(root_exc, _EXPECTED_RUNTIME_EXCEPTIONS)
    prefix = "[FATAL]" if expected else "[FATAL:UNEXPECTED]"
    print(f"{prefix} {stage}: {root_exc}")
    if not expected and debug_tracebacks:
        traceback.print_exception(type(root_exc), root_exc, root_exc.__traceback__)


def _emit_optional_failure(stage: str, exc: Exception, *, debug_tracebacks: bool) -> None:
    stage, root_exc = _unwrap_stage_failure(stage, exc)
    expected = isinstance(root_exc, _EXPECTED_RUNTIME_EXCEPTIONS)
    prefix = "[ERROR]" if expected else "[ERROR:UNEXPECTED]"
    print(f"{prefix} {stage}: {root_exc}")
    if not expected and debug_tracebacks:
        traceback.print_exception(type(root_exc), root_exc, root_exc.__traceback__)


def _debug_tracebacks_requested(argv: Sequence[str] | None = None) -> bool:
    values = sys.argv[1:] if argv is None else argv
    return "--debug-tracebacks" in values


def _warn_optional_failure(stage: str, operation: Callable[[], None]) -> None:
    try:
        operation()
    except Exception as exc:
        print(f"[WARN] {stage}: {exc}")


def _run_optional_output(
    stage: str,
    operation: Callable[[], Any],
    *,
    debug_tracebacks: bool,
    import_warning: str,
) -> None:
    try:
        operation()
    except ImportError:
        print(import_warning)
    except Exception as exc:
        _emit_optional_failure(stage, exc, debug_tracebacks=debug_tracebacks)


def load_runtime_config(args: Namespace) -> SimConfig:
    # Thread asset-path overrides into the factory *before* default asset
    # resolution, so --kernel-dir / --gravity-file-path work on a machine without
    # the repo default data/ layout (otherwise load_default_config would fail
    # while resolving defaults, before apply_args_to_config could override them).
    cfg = load_default_config(
        kernel_dir=getattr(args, "kernel_dir", None),
        gravity_file_path=getattr(args, "gravity_file_path", None),
    )
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


def _telemetry_enabled(cfg: SimConfig) -> bool:
    propagator = cfg.propagator
    cadence = float(getattr(propagator, "telem_cadence_s", 0.0) or 0.0)
    return bool(getattr(propagator, "enable_telemetry", False)) or cadence > 0.0


def _prepare_telemetry_run(
    cfg: SimConfig,
    *,
    out_dir: Path | None = None,
    artifact_requested: bool = False,
) -> tuple[SimConfig, str | None]:
    """Assign the run id that ties [TELEMETRY] samples to the meta line.

    When the replay artifact is requested, the emitter additionally mirrors
    every line into a hidden ``.part`` file under ``out_dir``;
    :func:`_finalize_telemetry_artifact` moves it into the canonical run
    directory once that directory exists (it is only created after the
    propagation finishes).
    """
    if not _telemetry_enabled(cfg):
        return cfg, None
    from lunaris.core.config import replace_sim_config
    from lunaris.core.propagation.telemetry_emitter import generate_run_id

    run_id = cfg.propagator.telemetry_run_id or generate_run_id()
    sink_path = cfg.propagator.telemetry_sink_path
    if artifact_requested and not sink_path and out_dir is not None:
        sink_path = str(out_dir / f".telemetry_{run_id}.ndjson.part")
    if run_id == cfg.propagator.telemetry_run_id and sink_path == cfg.propagator.telemetry_sink_path:
        return cfg, run_id
    cfg = replace_sim_config(
        cfg,
        propagator=replace(
            cfg.propagator,
            telemetry_run_id=run_id,
            telemetry_sink_path=sink_path,
        ),
    )
    return cfg, run_id


def _finalize_telemetry_artifact(cfg: SimConfig, run_dir: Path) -> None:
    """Move the streamed ``.part`` telemetry mirror into the run directory."""
    part = str(cfg.propagator.telemetry_sink_path or "")
    if not part:
        return
    from lunaris.common.telemetry_contract import TELEMETRY_ARTIFACT_NAME

    part_path = Path(part)
    if not part_path.is_file():
        return
    try:
        target = run_dir / TELEMETRY_ARTIFACT_NAME
        part_path.replace(target)
    except OSError:
        print("[WARN] Could not finalize telemetry.ndjson artifact.")


def _git_commit_or_none() -> str | None:
    """Best-effort short commit hash (None for wheel installs / no git)."""
    import subprocess

    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=3.0,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = proc.stdout.strip()
    return commit if proc.returncode == 0 and commit else None


def build_telemetry_provenance(cfg: SimConfig, run_id: str, *, mu: float | None) -> Any:
    """Run-level provenance for the [TELEMETRY_META] line.

    Only facts known *before* propagation are claimed here; effective-runtime
    facts (rhs_path, integration backend, wall time) arrive with the end-of-run
    [DIAG] payload and are merged by the consumer. Absent knowledge stays
    absent — the monitor renders it as "Unavailable".
    """
    from lunaris.common.provenance import sha256_text
    from lunaris.common.telemetry_contract import TelemetryProvenance
    from lunaris.core.propagation.telemetry_emitter import INERTIAL_FRAME_LABEL

    config_sha256: str | None = None
    with contextlib.suppress(TypeError, ValueError):
        config_sha256 = sha256_text(
            json.dumps(asdict(cfg), sort_keys=True, default=str)
        )[:12]

    gravity = cfg.gravity
    gravity_model: str | None = None
    st_lrps_artifact: str | None = None
    if gravity.uses_st_lrps:
        st_lrps_artifact = gravity.st_lrps_model_dir or None
        gravity_model = Path(gravity.st_lrps_model_dir).name if gravity.st_lrps_model_dir else None
    elif gravity.file_path:
        gravity_model = Path(gravity.file_path).name

    cadence = float(cfg.propagator.telem_cadence_s or 0.0)
    return TelemetryProvenance(
        run_id=run_id,
        integrator=str(cfg.propagator.method),
        gravity_backend=str(gravity.backend),
        gravity_model=gravity_model,
        sh_degree=gravity.degree,
        adaptive_degree=bool(getattr(gravity.adaptive, "enabled", False)) or None,
        st_lrps_artifact=st_lrps_artifact,
        config_sha256=config_sha256,
        git_commit=_git_commit_or_none(),
        frame_inertial=INERTIAL_FRAME_LABEL,
        mu_m3s2=mu,
        telemetry_cadence_s=cadence if cadence > 0.0 else None,
    )


def _emit_telemetry_meta(cfg: SimConfig, run_id: str | None, *, mu: float | None) -> None:
    """Print the [TELEMETRY_META] line (best-effort; never fails the run)."""
    if run_id is None:
        return
    try:
        from lunaris.common.telemetry_contract import encode_meta_line

        line = encode_meta_line(build_telemetry_provenance(cfg, run_id, mu=mu))
        print(line, flush=True)
        sink = str(cfg.propagator.telemetry_sink_path or "")
        if sink:
            with contextlib.suppress(OSError):
                with open(sink, "a", encoding="utf-8", newline="\n") as handle:
                    handle.write(line + "\n")
    except Exception:
        print("[WARN] Could not emit telemetry meta line.")


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
        "body_radius_m": float(R_MOON),
        "spacecraft": {
            "mass_kg": cfg.spacecraft.mass_kg,
            "area_m2": cfg.spacecraft.area_m2,
            "cd": cfg.spacecraft.cd,
            "cr": cfg.spacecraft.cr,
        },
        "propagation_time_s": propagation_time_s,
        "duration_s": cfg.time.duration_s,
    }


def create_canonical_run_dir(out_dir: Path) -> Path:
    """Create the single timestamped leaf that holds *all* artifacts of one run.

    Config, diagnostics, PNGs, the PDF report, and any 3D render are written into
    this one directory so the Results indexer sees a self-contained run: a
    ``run_config.json`` sibling to its own figures and reports. Writing config to
    ``out_dir`` while reporting created its own timestamped subdirectory used to
    split every run across two directories, leaving the gallery empty and letting
    re-used output roots overwrite the top-level config.
    """
    from lunaris.analysis.reporting.manager import create_run_directory

    return Path(create_run_directory(str(out_dir), prefix="run"))


def render_reports(
    *,
    result: PropagationResult,
    engine: Any,
    cfg: SimConfig,
    out_dir: Path,
    meta: dict[str, Any],
    preset: str = "standard",
) -> dict[str, Any]:
    from lunaris.analysis.reporting.manager import generate_run_package

    outputs = generate_run_package(
        result=result,
        config=cfg,
        out_dir=out_dir,
        ctx=engine,
        meta=meta,
        preset=preset,
    )
    notification = {
        "status": "success",
        "run_dir": str(out_dir.resolve()),
        "report_pdf": str(Path(outputs["pdf"]).resolve()),
        "report_markdown": str(Path(outputs["report_markdown"]).resolve()),
        "metrics_json": str(Path(outputs["metrics"]).resolve()),
        "preset": preset,
    }
    print("[REPORT] " + json.dumps(notification, sort_keys=True))
    return outputs


def render_optional_3d(result: PropagationResult, cfg: SimConfig, out_dir: Path) -> None:
    if not cfg.output.make_3d_plots:
        return

    from lunaris.visualization.orbit_animation import render_orbit_animation

    render_orbit_animation(
        result=result,
        config=cfg,
        output_file=str(out_dir / "orbit_3d.mp4"),
    )


def run_pipeline(args: Namespace) -> int:
    """Execute the propagation command after argument parsing."""
    debug_tracebacks = bool(getattr(args, "debug_tracebacks", False))

    cfg = _run_stage("Config init failed", lambda: load_runtime_config(args))
    out_dir = _run_stage("Output directory failure", lambda: Path(cfg.output.ensure_out_dir()))
    cfg, telemetry_run_id = _prepare_telemetry_run(
        cfg,
        out_dir=out_dir,
        artifact_requested=bool(getattr(args, "telemetry_artifact", None)),
    )
    gravity_core, mu = _run_stage("Gravity model init failed", lambda: build_gravity_provider(cfg))
    surface = _run_stage("Surface grids load failed", lambda: build_surface_provider_if_needed(cfg, args))
    ephem_mgr = _run_stage(
        "Ephemeris init failed",
        lambda: build_ephemeris_if_needed(cfg, topo_requested=surface.topo_requested),
    )
    y0, orbit_params = _run_stage("Orbit init failed", lambda: resolve_initial_state(cfg, args, mu=mu))
    effective_y0 = _y0_to_array(y0)[:6]
    cfg = replace_sim_config(
        cfg,
        initial_state=InitialState(
            x=float(effective_y0[0]),
            y=float(effective_y0[1]),
            z=float(effective_y0[2]),
            vx=float(effective_y0[3]),
            vy=float(effective_y0[4]),
            vz=float(effective_y0[5]),
        ),
    )
    _run_stage("Run summary failed", lambda: print_summary(cfg, orbit_params, y0))

    engine = _run_stage(
        "Dynamics engine init failed",
        lambda: build_engine(
            cfg,
            gravity_core=gravity_core,
            ephem_mgr=ephem_mgr,
            surface_provider=surface.provider,
        ),
    )

    _emit_telemetry_meta(cfg, telemetry_run_id, mu=float(mu))

    run = _run_stage(
        "Propagation failed",
        lambda: run_propagation(
            engine,
            cfg,
            y0=y0,
            topo_grid=surface.topo_grid,
        ),
    )

    # One canonical run directory holds every artifact of this run (config,
    # diagnostics, PNGs, PDF, 3D). Falls back to out_dir if creation fails so a
    # completed propagation still writes something.
    run_dir = out_dir
    try:
        run_dir = create_canonical_run_dir(out_dir)
    except OSError:
        print("[WARN] Could not create canonical run directory; using output root.")

    diag_payload = build_run_diagnostics_payload(run.result, cfg.propagator.method)
    # SPICE provenance chain (kernels/hashes/ET window) into the run diagnostics,
    # so a single run's ephemeris evidence sits alongside its config and figures.
    if ephem_mgr is not None and hasattr(ephem_mgr, "kernel_provenance"):
        with contextlib.suppress(Exception):
            diag_payload["spice_kernels"] = ephem_mgr.kernel_provenance()
    run.result.diagnostics = dict(diag_payload)
    _warn_optional_failure(
        "Could not write run artifacts",
        lambda: write_run_artifacts(run_dir, cfg, diag_payload),
    )
    _finalize_telemetry_artifact(cfg, run_dir)

    meta = build_run_meta(cfg, run.result, mu=mu, propagation_time_s=run.elapsed_s)
    meta.update(
        {
            "ldem_root": getattr(args, "ldem_root", None),
            "albedo_root": getattr(args, "albedo_root", None),
        }
    )

    _run_optional_output(
        "Plot/report failed",
        lambda: render_reports(
            result=run.result,
            engine=engine,
            cfg=cfg,
            out_dir=run_dir,
            meta=meta,
            preset=str(getattr(args, "report_preset", "standard")),
        ),
        debug_tracebacks=debug_tracebacks,
        import_warning="[WARN] analysis.reporting.manager not found; skipping plots.",
    )

    _run_optional_output(
        "3D render failed",
        lambda: render_optional_3d(run.result, cfg, run_dir),
        debug_tracebacks=debug_tracebacks,
        import_warning="[WARN] visualization.orbit_animation not found; skipping 3D render.",
    )

    print("[OK] Finished.")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    return run_pipeline(args)


def main_entry() -> int:
    """Console-script entry point."""
    try:
        return main()
    except Exception as exc:
        _emit_failure(
            "Runtime failed",
            exc,
            debug_tracebacks=_debug_tracebacks_requested(),
        )
        return 1


__all__ = [
    "build_engine",
    "build_ephemeris_if_needed",
    "build_gravity_provider",
    "build_run_diagnostics_payload",
    "build_run_meta",
    "build_surface_provider_if_needed",
    "create_canonical_run_dir",
    "init_ephemeris",
    "load_runtime_config",
    "main",
    "main_entry",
    "orbit_init_requested",
    "render_optional_3d",
    "render_reports",
    "resolve_initial_state",
    "run_pipeline",
    "run_propagation",
    "write_run_artifacts",
    "_y0_to_array",
]
