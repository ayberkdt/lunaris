"""High-level orbit propagation orchestration.

The heavy helper families live in sibling modules under
``lunaris.core.propagation``. This module keeps the public ``propagate`` entry
point and owns the propagation orchestration surface.
"""

from __future__ import annotations

import json
import logging
import math
import time
import warnings
from collections.abc import Callable, Sequence
from types import SimpleNamespace
from typing import Any

import numpy as np
from scipy.integrate import solve_ivp

from lunaris.common.constants import R_MOON
from lunaris.common.math_utils import nyquist_max_step_s, specific_energy_drift_stats
from lunaris.common.type_defs import PropagationResult, PropagatorConfig, TimeConfig
from lunaris.core.dynamics import DynamicsEngine
from lunaris.core.propagation.checkpoint import (
    _atomic_save_npz,
    _checkpoint_metadata,
    _stop_requested,
)
from lunaris.core.propagation.diagnostics import build_propagation_diagnostics
from lunaris.core.propagation.events import (
    EventOutcome,
    _build_r_i_to_bf_from_rot_table,
    _terminal_event_endpoint,
    _wrap_event_first6,
    build_events,
    event_outcome_from_solver_events,
)
from lunaris.core.propagation.fixed_step_runner import run_fixed_step_propagation
from lunaris.core.propagation.integrators.fixed_step import (
    _integrate_fixed_step,
    _is_fixed_step_method,
    symplectic_breaks_separability,
    symplectic_nonconservative_gravity,
    symplectic_nonconservative_violations,
)
from lunaris.core.propagation.integrators.scipy import _resolve_scipy_method
from lunaris.core.propagation.plans import (
    IntegrationPlan,
    StepSizePlan,
    TimeGridPlan,
    _osculating_periapsis_alt_km,
    resolve_integration_plan,
    resolve_step_size_policy,
    resolve_time_grid_plan,
)
from lunaris.core.propagation.result import _as_state_array
from lunaris.core.propagation.telemetry import (
    _build_surface_radius_sampler,
    _make_telem_dict,
)
from lunaris.core.propagation.time_grid import (
    _get_ref_radius_and_mu,
    _get_sh_degree,
    make_time_grid,
)

logger = logging.getLogger(__name__)


def _resolve_atol(cfg: PropagatorConfig, n_state: int) -> float | np.ndarray:
    """Resolve the absolute-tolerance argument for solve_ivp.

    Returns the scalar ``cfg.atol`` unless ``atol_pos``/``atol_vel`` are set, in
    which case it returns a length-``n_state`` vector that applies the
    position/velocity bounds to the first six components and keeps the scalar
    ``atol`` for any extra (augmented) components. Keeping the scalar path
    byte-identical preserves the existing default behavior.
    """
    atol_scalar = float(getattr(cfg, "atol", 1e-12))
    atol_pos = getattr(cfg, "atol_pos", None)
    atol_vel = getattr(cfg, "atol_vel", None)
    if atol_pos is None and atol_vel is None:
        return atol_scalar

    n = int(n_state)
    vec: np.ndarray = np.full(n, atol_scalar, dtype=np.float64)
    if n >= 6:
        if atol_pos is not None:
            vec[0:3] = float(atol_pos)
        if atol_vel is not None:
            vec[3:6] = float(atol_vel)
    return vec


def _rhs_path_for_diagnostics(dynamics: Any) -> str:
    """Return the RHS implementation path stamped into run diagnostics."""

    raw: Any = None
    prep = getattr(dynamics, "_prep", None)
    if isinstance(prep, dict):
        raw = prep.get("rhs_path")
    elif prep is not None:
        try:
            raw = prep["rhs_path"]
        except (KeyError, TypeError):
            raw = None

    if raw is not None:
        text = str(raw).strip()
        if text == "surrogate_python":
            return "surrogate_python_autograd"
        if text:
            return text

    grav = getattr(dynamics, "grav", None)
    if getattr(grav, "model_kind", None) == "st_lrps":
        return "surrogate_python_autograd"
    if grav is None:
        return "point_mass_python"
    return "unknown"


def propagate(
    dynamics: DynamicsEngine,
    y0: Any,
    cfg: PropagatorConfig,
    *,
    time_cfg: TimeConfig | None = None,
    topo_grid: Any = None,
    extra_events: Sequence[Callable[[float, np.ndarray], float]] | None = None,
) -> PropagationResult:
    """
    Propagate the trajectory for a configured duration and output sampling grid.

    SSOT
    ----
    Duration and output sampling are owned by TimeConfig:
        time_cfg.duration_s
        time_cfg.output_dt_s

    Notes
    -----
    - t0 (start epoch in seconds) is owned by TimeConfig when provided (time_cfg.t0_s).
    - max_points_cap / verbosity / integration tolerances remain in PropagatorConfig unless also provided in TimeConfig.
    """
    y0_arr = _as_state_array(y0)

    t_wall0 = time.perf_counter()

    verbose = bool(getattr(cfg, "verbose", False))

    # Normalize optional filesystem paths to str (or None) for string ops in helpers.
    stop_file: str | None = None
    try:
        sf = getattr(cfg, "stop_file", None)
        stop_file = (str(sf) if sf else None)
    except (AttributeError, TypeError, ValueError):
        stop_file = None

    checkpoint_path: str | None = None
    try:
        cp = getattr(cfg, "checkpoint_path", None)
        checkpoint_path = (str(cp) if cp else None)
    except (AttributeError, TypeError, ValueError):
        checkpoint_path = None

    # -------------------------------------------------------------------------
    # 1) Resolve time grid (STRICT: TimeConfig required)
    # -------------------------------------------------------------------------
    time_plan = resolve_time_grid_plan(
        dynamics=dynamics,
        y0=y0_arr,
        cfg=cfg,
        time_cfg=time_cfg,
        verbose=verbose,
    )
    t0 = time_plan.t0
    tf = time_plan.tf
    dur_s = time_plan.duration_s
    dt_out = time_plan.realized_output_dt_s
    t_eval = time_plan.t_eval

    rhs = dynamics.build_rhs()
    rhs_path = _rhs_path_for_diagnostics(dynamics)
    R_ref_m, mu_m3s2 = _get_ref_radius_and_mu(dynamics)

    # Terrain-aware telemetry is optional. The actual hybrid impact event uses a
    # similar capability deeper in the propagator, but surfacing the sampled
    # local radius here lets the desktop UI explain *why* a run stopped near the
    # surface instead of only showing mean-radius altitude.
    telem_r_i_to_bf: Callable[[float, np.ndarray], np.ndarray] | None = None
    telem_surface_radius_m: Callable[[float, float], float] | None = None
    if topo_grid is not None:
        try:
            telem_r_i_to_bf = _build_r_i_to_bf_from_rot_table(dynamics)
            if telem_r_i_to_bf is not None:
                telem_surface_radius_m = _build_surface_radius_sampler(topo_grid)
        except (AttributeError, KeyError, IndexError, TypeError, ValueError):
            telem_r_i_to_bf = None
            telem_surface_radius_m = None

    # Optional: stream compact JSON telemetry for UI live plots/progress.
    # Controlled explicitly via config (no env-var "magic"). Opt-in: a library
    # caller (validation harness, batch job, test) must not have stdout polluted
    # with JSON lines unless it asks for telemetry. The desktop UI sets
    # ``cfg.enable_telemetry = True``. Back-compat: a positive ``telem_cadence_s``
    # also enables it.
    _telem_cadence_cfg = float(getattr(cfg, "telem_cadence_s", getattr(cfg, "telemetry_cadence_s", 0.0)) or 0.0)
    enable_telem_json = bool(getattr(cfg, "enable_telemetry", False)) or _telem_cadence_cfg > 0.0
    telem_cadence_s: float = _telem_cadence_cfg
    if enable_telem_json and telem_cadence_s <= 0.0:
        hb_h = float(getattr(cfg, "heartbeat_hours", 0.0) or 0.0)
        if hb_h > 0.0:
            telem_cadence_s = max(5.0, hb_h * 3600.0)
        else:
            # Fallback: ~60 output samples, but at least 60s
            telem_cadence_s = max(60.0, float(dt_out) * 60.0)

    if enable_telem_json and telem_cadence_s > 0.0:
        last_telem_t = float(t0) - float(telem_cadence_s)
        rhs_base = rhs

        def rhs(t: float, y: np.ndarray) -> np.ndarray:
            nonlocal last_telem_t
            dy = rhs_base(t, y)
            if (float(t) - float(last_telem_t)) >= float(telem_cadence_s):
                telem = _make_telem_dict(
                    t_s=float(t - t0),
                    y=y,
                    R_ref_m=float(R_ref_m),
                    mu_m3s2=float(mu_m3s2),
                    t_frame_s=float(t),
                    r_i_to_bf=telem_r_i_to_bf,
                    surface_radius_m=telem_surface_radius_m,
                )
                if telem is not None:
                    print(json.dumps(telem, separators=(",", ":")), flush=True)
                last_telem_t = float(t)
            return dy

    # -------------------------------------------------------------------------
    # 3) Max-step logic (Nyquist cap vs user cap)
    # -------------------------------------------------------------------------
    degree = _get_sh_degree(dynamics)
    topo_present = topo_grid is not None
    step_plan = resolve_step_size_policy(
        cfg=cfg,
        y0=y0_arr,
        R_ref_m=float(R_ref_m),
        mu_m3s2=float(mu_m3s2),
        sh_degree=int(degree),
        output_dt_s=float(dt_out),
        topo_present=topo_present,
        nyquist_func=nyquist_max_step_s,
    )
    max_step = step_plan.actual_max_step_s
    if step_plan.user_max_step_s is None:
        if verbose:
            logger.info(
                "[STEP] max_step_s=%0.6f (reason=%s, deg=%d)",
                max_step,
                step_plan.limiting_reason,
                step_plan.sh_degree,
            )
    else:
        if verbose:
            logger.info(
                "[STEP] user_max_step=%gs, nyquist=%0.6fs -> using %0.6fs (reason=%s)",
                step_plan.user_max_step_s,
                step_plan.nyquist_max_step_s,
                max_step,
                step_plan.limiting_reason,
            )

    integration_plan = resolve_integration_plan(cfg, duration_s=dur_s)
    checkpoint_meta = _checkpoint_metadata(method=integration_plan.method, config=cfg)

    # -------------------------------------------------------------------------
    # 4) Events
    # -------------------------------------------------------------------------
    events = build_events(dynamics, cfg, topo_grid=topo_grid, add_stop_event=bool(stop_file))
    if extra_events:
        for ev in list(extra_events):
            events.append(_wrap_event_first6(ev))

    # -------------------------------------------------------------------------
    # 4b) Symplectic guard
    # -------------------------------------------------------------------------
    # Symplectic / structure-preserving methods (VV, PEFRL, Yoshida) only keep
    # their bounded-energy-drift guarantee for conservative, position-only forces
    # (SH gravity, third-body, Earth J2). When a non-conservative perturbation is
    # active (SRP / albedo / thermal IR) that guarantee is void; when a
    # velocity-dependent force is active (1PN relativity) the acceleration-based
    # steppers additionally sample the force inconsistently. We don't silently
    # change the method -- the choice belongs to the caller -- but we must not let
    # the inconsistency pass unflagged. See ``_SYMPLECTIC_VOIDING_FLAGS``.
    _method = getattr(cfg, "method", "DOP853")
    _flags = getattr(dynamics, "flags", None)
    _violations = symplectic_nonconservative_violations(_method, _flags)
    # The gravity model itself can void the guarantee: a non-conservative
    # surrogate (one whose acceleration is not the gradient of a scalar
    # potential) breaks bounded energy drift even with every perturbation flag
    # off. potential_autograd surrogates and classical SH stay exempt.
    _violations += symplectic_nonconservative_gravity(_method, getattr(dynamics, "grav", None))
    if _violations:
        _msg = (
            f"Symplectic method {str(_method)!r} is active together with "
            f"non-conservative perturbation(s): {', '.join(_violations)}. "
            "The bounded-energy-drift guarantee of symplectic integrators only "
            "holds for conservative, position-only forces (gravity, third-body, "
            "Earth J2). Energy drift may now be unbounded; prefer RK4 or an "
            "adaptive method (DOP853/RK45) for these dynamics."
        )
        if symplectic_breaks_separability(_method, _flags):
            _msg += (
                " Note: 1PN relativity is velocity-dependent, so the "
                "acceleration-form stepper also evaluates it at an inconsistent "
                "intermediate velocity (not just non-symplectically)."
            )
        # In normal use this is a warning (the method choice belongs to the
        # caller). For a paper / validation run, an invalid energy-behavior claim
        # must not pass silently: ``strict_symplectic`` escalates it to a hard
        # failure so a benchmark cannot quietly ship a symplectic result whose
        # bounded-drift guarantee is void.
        if bool(getattr(cfg, "strict_symplectic", False)):
            raise ValueError(
                _msg + " strict_symplectic=True: refusing to run a symplectic method with "
                "non-conservative forces for a paper/validation run."
            )
        warnings.warn(_msg, RuntimeWarning, stacklevel=2)

    # -------------------------------------------------------------------------
    # 5) Integrate
    # -------------------------------------------------------------------------
    if integration_plan.backend == "fixed_step":
        res = run_fixed_step_propagation(
            rhs=rhs,
            t_eval=t_eval,
            y0=y0_arr,
            max_step_s=max_step,
            method=integration_plan.method,
            events=events,
            R_ref_m=float(R_ref_m),
            mu_m3s2=float(mu_m3s2),
            verbose=verbose,
            heartbeat_hours=float(getattr(cfg, "heartbeat_hours", 0.0)),
            stop_file=stop_file,
            checkpoint_path=checkpoint_path,
            checkpoint_metadata=checkpoint_meta,
            logger=logger,
        )

    else:
        # TODO(P3): extract the SciPy/chunked branch into ``scipy_runner.py` once
        # the diagnostics and fixed-step seams have stayed stable.
        if solve_ivp is None:
            raise ImportError("SciPy is required for adaptive integration (solve_ivp not available).")

        method = integration_plan.method

        # Per-component (vector) atol when configured: position and velocity differ
        # by ~3 orders of magnitude, so a single scalar over-tightens one of them.
        atol_arg = _resolve_atol(cfg, int(y0_arr.size))

        if verbose:
            if isinstance(atol_arg, np.ndarray):
                logger.info(
                    f"[PROP] solve_ivp method={method} | dt_out={dt_out:g}s | max_step={max_step:.6f}s "
                    f"| atol=vector(pos={getattr(cfg, 'atol_pos', None)}, vel={getattr(cfg, 'atol_vel', None)})"
                )
            else:
                logger.info(
                    f"[PROP] solve_ivp method={method} | dt_out={dt_out:g}s | max_step={max_step:.6f}s"
                )

        def _solve_span(t_start: float, t_end: float, y_start: np.ndarray, t_eval_span: np.ndarray):
            return solve_ivp(
                fun=rhs,
                t_span=(float(t_start), float(t_end)),
                y0=np.asarray(y_start, dtype=np.float64),
                method=method,
                t_eval=np.asarray(t_eval_span, dtype=np.float64),
                rtol=float(getattr(cfg, "rtol", 1e-9)),
                atol=atol_arg,
                max_step=float(max_step),
                events=(events if events else None),
                dense_output=False,
                vectorized=False,
            )

        chunk_s = integration_plan.chunk_s

        checkpoint_every_chunk = bool(getattr(cfg, "checkpoint_every_chunk", False))
        integration_failed = False
        integration_failure_message: str | None = None
        stopped_early = False
        stop_reason = None
        chunk_idx = 0

        if chunk_s is None:
            sol = _solve_span(t0, tf, y0_arr, t_eval)
            t_cat = np.asarray(sol.t, dtype=np.float64)
            y_cat = np.asarray(sol.y, dtype=np.float64)
            t_events = [np.asarray(te, dtype=np.float64) for te in (sol.t_events or [])]
            y_events = [np.asarray(ye, dtype=np.float64) for ye in (sol.y_events or [])]
            # Keep stopped_early consistent with a terminal-event stop (status==1)
            # so callers never see stop_reason set while stopped_early is False.
            if int(getattr(sol, "status", 0)) == 1:
                stopped_early = True
            elif not bool(getattr(sol, "success", True)):
                stopped_early = True
                stop_reason = "integration failed"
                integration_failed = True
                integration_failure_message = str(getattr(sol, "message", "integration failed"))
        else:
            t_parts: list[np.ndarray] = []
            y_parts: list[np.ndarray] = []

            n_ev = len(events) if events else 0
            t_events_acc: list[list[np.ndarray]] = [[] for _ in range(n_ev)]
            y_events_acc: list[list[np.ndarray]] = [[] for _ in range(n_ev)]

            y_curr = y0_arr.copy()
            t_curr = float(t0)

            while t_curr < tf - 1e-12:
                if _stop_requested(stop_file) and (not bool(getattr(cfg, "stop_event_in_scipy", False))):
                    stopped_early = True
                    stop_reason = "stop file"
                    break

                t_next = min(tf, t_curr + float(chunk_s))
                mask = (t_eval >= t_curr - 1e-12) & (t_eval <= t_next + 1e-12)
                t_eval_span = t_eval[mask]
                if t_eval_span.size < 2:
                    t_eval_span = np.array([t_curr, t_next], dtype=np.float64)

                sol_k = _solve_span(t_curr, t_next, y_curr, t_eval_span)

                sol_k_status = int(getattr(sol_k, "status", 0))
                if sol_k_status != 1 and not bool(getattr(sol_k, "success", True)):
                    stopped_early = True
                    stop_reason = "integration failed"
                    integration_failed = True
                    integration_failure_message = str(getattr(sol_k, "message", "integration failed"))
                    break

                sol_t = np.asarray(sol_k.t, dtype=np.float64)
                sol_y = np.asarray(sol_k.y, dtype=np.float64)
                if sol_t.size == 0 or sol_y.ndim != 2 or sol_y.shape[1] != sol_t.size:
                    stopped_early = True
                    stop_reason = "integration failed"
                    integration_failed = True
                    integration_failure_message = "solve_ivp returned an invalid chunk shape"
                    break

                terminal_endpoint = (
                    _terminal_event_endpoint(sol_k, events, state_size=y0_arr.size)
                    if sol_k_status == 1
                    else None
                )
                if terminal_endpoint is not None:
                    t_terminal, y_terminal = terminal_endpoint
                    keep = sol_t <= (t_terminal + 1e-9)
                    sol_t = sol_t[keep]
                    sol_y = sol_y[:, keep]
                    if sol_t.size == 0 or abs(float(sol_t[-1]) - t_terminal) > 1e-9:
                        sol_t = np.concatenate([sol_t, np.asarray([t_terminal], dtype=np.float64)])
                        sol_y = np.concatenate([sol_y, y_terminal.reshape(-1, 1)], axis=1)
                    else:
                        sol_y[:, -1] = y_terminal

                if not t_parts:
                    t_parts.append(sol_t)
                    y_parts.append(sol_y)
                else:
                    t_parts.append(sol_t[1:])
                    y_parts.append(sol_y[:, 1:])

                if getattr(sol_k, "t_events", None) is not None:
                    for i in range(n_ev):
                        te = sol_k.t_events[i] if i < len(sol_k.t_events) else np.array([], dtype=np.float64)
                        ye = sol_k.y_events[i] if i < len(sol_k.y_events) else np.zeros((0, y0_arr.size), dtype=np.float64)
                        t_events_acc[i].append(np.asarray(te, dtype=np.float64))
                        y_events_acc[i].append(np.asarray(ye, dtype=np.float64))

                # Advance the running state to the END of this completed chunk
                # BEFORE checkpointing, so a "latest/state/last" checkpoint records
                # the chunk end (a valid resume point) rather than its start.
                y_curr = np.asarray(sol_y[:, -1], dtype=np.float64).copy()
                t_curr = float(sol_t[-1])

                if checkpoint_path and checkpoint_every_chunk:
                    try:
                        ck_mode = str(getattr(cfg, "checkpoint_mode", "full")).strip().lower()
                        if ck_mode in ("latest", "state", "last"):
                            _atomic_save_npz(
                                checkpoint_path,
                                t=np.asarray([t_curr], dtype=np.float64),
                                y_row=y_curr.reshape(1, -1),
                                **checkpoint_meta,
                            )
                        elif ck_mode in ("chunks", "chunk"):
                            base = str(checkpoint_path)
                            chunk_path = f"{base}.chunk{chunk_idx:06d}.npz"
                            _atomic_save_npz(
                                chunk_path,
                                t=sol_t,
                                y_row=sol_y.T,
                                **checkpoint_meta,
                            )
                            _atomic_save_npz(
                                checkpoint_path,
                                t=np.asarray([t_curr], dtype=np.float64),
                                y_row=y_curr.reshape(1, -1),
                                **checkpoint_meta,
                            )
                        else:
                            t_tmp = np.concatenate(t_parts) if t_parts else np.array([], dtype=np.float64)
                            y_tmp = np.concatenate(y_parts, axis=1) if y_parts else np.zeros((y0_arr.size, 0), dtype=np.float64)
                            _atomic_save_npz(
                                checkpoint_path,
                                t=t_tmp,
                                y_row=y_tmp.T,
                                **checkpoint_meta,
                            )
                    except (OSError, ValueError) as exc:
                        warnings.warn(f"Checkpoint write failed: {exc}", RuntimeWarning, stacklevel=2)

                if sol_k_status == 1:
                    stopped_early = True
                    stop_reason = "event"
                    break

                # y_curr/t_curr were already advanced to the chunk end above.
                chunk_idx += 1

            t_cat = np.concatenate(t_parts) if t_parts else np.array([t0], dtype=np.float64)
            y_cat = np.concatenate(y_parts, axis=1) if y_parts else y0_arr.reshape(-1, 1)

            t_events = [np.concatenate(ch) if ch else np.array([], dtype=np.float64) for ch in t_events_acc]
            y_events = [np.concatenate(ch, axis=0) if ch else np.zeros((0, y0_arr.size), dtype=np.float64) for ch in y_events_acc]

            sol = SimpleNamespace(
                t=t_cat,
                y=y_cat,
                t_events=t_events,
                y_events=y_events,
                success=not integration_failed,
                status=(-1 if integration_failed else (1 if stopped_early else 0)),
                message=(
                    integration_failure_message
                    if integration_failed
                    else ("chunked ok" if not stopped_early else "stopped early")
                ),
                nfev=np.nan,
            )

        y_row = np.asarray(y_cat, dtype=np.float64).T

        outcome = event_outcome_from_solver_events(
            events=events,
            t_events=list(t_events),
            y_events=list(y_events),
            stopped_early=bool(stopped_early),
            stop_reason=stop_reason,
            stop_file=stop_file,
            stop_event_in_scipy=bool(getattr(cfg, "stop_event_in_scipy", False)),
        )

        if checkpoint_path and (not integration_failed) and not (chunk_s is not None and checkpoint_every_chunk):
            try:
                _atomic_save_npz(
                    checkpoint_path,
                    t=np.asarray(t_cat, dtype=np.float64),
                    y_row=y_row,
                    **checkpoint_meta,
                )
            except (OSError, ValueError) as exc:
                warnings.warn(f"Checkpoint write failed: {exc}", RuntimeWarning, stacklevel=2)

        res = PropagationResult(
            t=np.asarray(t_cat, dtype=np.float64),
            y=y_row,
            ode=sol,
            t_events=list(t_events),
            y_events=list(y_events),
            impacted=outcome.impacted,
            t_impact_s=outcome.t_impact_s,
            y_impact=outcome.y_impact,
            stopped_early=outcome.stopped_early,
            stop_reason=outcome.stop_reason,
            t_stop_s=outcome.t_stop_s,
            diagnostics={},
        )

    # -------------------------------------------------------------------------
    # 6) Diagnostics + Optional 2-body baseline
    # -------------------------------------------------------------------------
    wall = time.perf_counter() - t_wall0
    res.diagnostics = build_propagation_diagnostics(
        dynamics=dynamics,
        cfg=cfg,
        result=res,
        time_plan=time_plan,
        step_plan=step_plan,
        integration_plan=integration_plan,
        degree=int(degree),
        output_dt_s=float(dt_out),
        max_step_s=float(max_step),
        wall_time_s=float(wall),
        rhs_path=rhs_path,
        symplectic_violations=_violations,
        y0=y0_arr,
        R_ref_m=float(R_ref_m),
        mu_m3s2=float(mu_m3s2),
        verbose=verbose,
        logger=logger,
    )

    if bool(getattr(cfg, "compute_2body_baseline", False)):
        res.baseline = _compute_2body_baseline(
            t_eval=res.t,
            y0=y0_arr[:6],
            mu_m3s2=float(mu_m3s2),
            cfg=cfg,
            max_step=float(max_step),
        )

    return res

def _compute_2body_baseline(
    *,
    t_eval: np.ndarray,
    y0: np.ndarray,
    mu_m3s2: float,
    cfg: PropagatorConfig,
    max_step: float,
) -> PropagationResult | None:
    """Compute a simple 2-body (central-gravity) reference trajectory.

    This is a diagnostic baseline to compare against the full dynamics model.
    Returns None if the time grid is invalid or SciPy is unavailable (adaptive path).
    """
    t_eval = np.asarray(t_eval, dtype=np.float64).reshape(-1)
    if t_eval.size < 2 or np.any(np.diff(t_eval) <= 0.0):
        return None

    y0 = np.asarray(y0, dtype=np.float64).reshape(-1)
    if y0.size < 6:
        return None
    y0 = y0[:6].copy()

    mu = float(mu_m3s2)
    if (not np.isfinite(mu)) or mu <= 0.0:
        return None

    def rhs2(t: float, y: np.ndarray) -> np.ndarray:
        rx, ry, rz = float(y[0]), float(y[1]), float(y[2])
        vx, vy, vz = float(y[3]), float(y[4]), float(y[5])

        r2 = rx * rx + ry * ry + rz * rz
        r2 = max(r2, 1e-30)
        inv_r = 1.0 / math.sqrt(r2)
        inv_r3 = inv_r * inv_r * inv_r

        ax = -mu * rx * inv_r3
        ay = -mu * ry * inv_r3
        az = -mu * rz * inv_r3

        dy: np.ndarray = np.empty(6, dtype=np.float64)
        dy[0] = vx
        dy[1] = vy
        dy[2] = vz
        dy[3] = ax
        dy[4] = ay
        dy[5] = az
        return dy

    # Fixed-step baseline (symplectic / RKN / RK4)
    if _is_fixed_step_method(cfg.method):
        ode_like, _, _, _, _, _, _ = _integrate_fixed_step(
            rhs=rhs2,
            t_eval=t_eval,
            y0=y0,
            max_step=float(max_step),
            method=str(cfg.method),
            events=None,
            R_ref_m=float(R_MOON),
            mu_m3s2=float(mu),
            verbose=False,
            heartbeat_hours=0.0,
            stop_file=None,
            checkpoint_path=None,
        )
        t_out = np.asarray(ode_like.t, dtype=np.float64)
        y_row = np.asarray(ode_like.y, dtype=np.float64).T
        # 2-body is conservative + autonomous, so this drift is purely numerical.
        diag = {"baseline": 1.0, "solver": "fixed-step", "success": 1.0}
        diag.update(specific_energy_drift_stats(t_out, y_row, float(mu)))
        return PropagationResult(
            t=t_out,
            y=y_row,
            ode=ode_like,
            diagnostics=diag,
        )

    # Adaptive baseline (solve_ivp)
    if solve_ivp is None:
        return None

    method = _resolve_scipy_method(cfg.method)
    rtol = float(getattr(cfg, "baseline_rtol", getattr(cfg, "rtol", 1e-9)))
    atol = float(getattr(cfg, "baseline_atol", getattr(cfg, "atol", 1e-12)))

    try:
        sol = solve_ivp(
            fun=rhs2,
            t_span=(float(t_eval[0]), float(t_eval[-1])),
            y0=y0,
            method=method,
            t_eval=t_eval,
            rtol=rtol,
            atol=atol,
            max_step=float(max_step),
            events=None,
            dense_output=False,
            vectorized=False,
        )
    except (ValueError, TypeError):
        # Only an unsupported/misspecified integrator method falls back to the
        # DOP853 baseline; a genuine integration failure must surface, not be
        # silently retried with a different method.
        sol = solve_ivp(
            fun=rhs2,
            t_span=(float(t_eval[0]), float(t_eval[-1])),
            y0=y0,
            method="DOP853",
            t_eval=t_eval,
            rtol=rtol,
            atol=atol,
            max_step=float(max_step),
            events=None,
            dense_output=False,
            vectorized=False,
        )

    t_b = np.asarray(sol.t, dtype=np.float64)
    y_b = np.asarray(sol.y, dtype=np.float64).T
    # 2-body is conservative + autonomous, so this drift is purely numerical.
    diag = {"baseline": 1.0, "solver": method, "success": float(bool(getattr(sol, "success", True)))}
    diag.update(specific_energy_drift_stats(t_b, y_b, float(mu)))
    return PropagationResult(
        t=t_b,
        y=y_b,
        ode=sol,
        diagnostics=diag,
    )


__all__ = [
    "propagate",
    "PropagationResult",
    "EventOutcome",
    "TimeGridPlan",
    "StepSizePlan",
    "IntegrationPlan",
    "build_events",
    "make_time_grid",
    "resolve_time_grid_plan",
    "resolve_step_size_policy",
    "resolve_integration_plan",
    "event_outcome_from_solver_events",
    "_osculating_periapsis_alt_km",
]
