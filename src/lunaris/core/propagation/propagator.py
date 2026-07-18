"""High-level orbit propagation orchestration.

The heavy helper families live in sibling modules under
``lunaris.core.propagation``. This module keeps the public ``propagate`` entry
point and owns the propagation orchestration surface.
"""

from __future__ import annotations

import logging
import math
import time
import warnings
from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import Any

import numpy as np
from scipy.integrate import solve_ivp

from lunaris.common.constants import R_MOON
from lunaris.common.math_utils import nyquist_max_step_s, specific_energy_drift_stats
from lunaris.common.type_defs import PropagationResult, PropagatorConfig, TimeConfig
from lunaris.core.dynamics import DynamicsEngine
from lunaris.core.propagation.checkpoint import _checkpoint_metadata
from lunaris.core.propagation.diagnostics import build_propagation_diagnostics
from lunaris.core.propagation.events import (
    EventOutcome,
    _build_r_i_to_bf_from_rot_table,
    _wrap_event_first6,
    build_events,
    event_outcome_from_solver_events,
)
from lunaris.core.propagation.fixed_step_runner import run_fixed_step_propagation
from lunaris.core.propagation.integrators.fixed_step import (
    _integrate_fixed_step,
    _is_fixed_step_method,
    accel_form_velocity_dependence_violations,
    symplectic_breaks_separability,
    symplectic_discontinuous_gravity,
    symplectic_impulsive_maneuver_violations,
    symplectic_nonconservative_gravity,
    symplectic_nonconservative_violations,
)
from lunaris.core.propagation.integrators.scipy import _resolve_scipy_method
from lunaris.core.propagation.plans import (
    IntegrationPlan,
    ManeuverPlan,
    StepSizePlan,
    TimeGridPlan,
    _osculating_periapsis_alt_km,
    apply_impulsive_maneuver,
    resolve_integration_plan,
    resolve_step_size_policy,
    resolve_time_grid_plan,
)
from lunaris.core.propagation.result import _as_state_array
from lunaris.core.propagation.scipy_runner import run_scipy_propagation
from lunaris.core.propagation.telemetry import (
    _build_surface_radius_sampler,
    _make_telem_dict,  # noqa: F401  # canonical import path for tests/back-compat
)
from lunaris.core.propagation.time_grid import (
    get_ref_radius_and_mu,
    get_sh_degree,
    make_time_grid,
)

logger = logging.getLogger(__name__)


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
    maneuver_plan: ManeuverPlan | None = None,
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

    if maneuver_plan is not None and maneuver_plan.maneuvers:
        return _propagate_with_maneuvers(
            dynamics=dynamics,
            y0=y0_arr,
            cfg=cfg,
            time_cfg=time_cfg,
            topo_grid=topo_grid,
            extra_events=extra_events,
            maneuver_plan=maneuver_plan,
        )

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

    # The propagator owns the invariant that the solver state dimension is
    # fixed for an integration. Use the trusted hot-path callable; external
    # callers of DynamicsEngine.build_rhs retain validation on every call.
    trusted_builder = getattr(dynamics, "_build_solver_rhs", None)
    rhs = trusted_builder() if callable(trusted_builder) else dynamics.build_rhs()
    rhs_path = _rhs_path_for_diagnostics(dynamics)
    R_ref_m, mu_m3s2 = get_ref_radius_and_mu(dynamics)

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
    telem_emitter: Any | None = None
    telem_cadence_s: float = _telem_cadence_cfg
    if enable_telem_json and telem_cadence_s <= 0.0:
        hb_h = float(getattr(cfg, "heartbeat_hours", 0.0) or 0.0)
        if hb_h > 0.0:
            telem_cadence_s = max(5.0, hb_h * 3600.0)
        else:
            # Fallback: ~60 output samples, but at least 60s
            telem_cadence_s = max(60.0, float(dt_out) * 60.0)

    if enable_telem_json and telem_cadence_s > 0.0:
        # Structured lunaris_telemetry_v1 emission ([TELEMETRY] {json} lines).
        # The cadence gate stays here — one float comparison per RHS call —
        # These gated samples are transient ``rhs_probe`` observations;
        # scientific output-state samples are emitted from PropagationResult
        # after integration.
        from lunaris.core.propagation.telemetry_emitter import build_emitter_from_config

        telem_emitter = build_emitter_from_config(
            cfg,
            t0_s=float(t0),
            reference_radius_m=float(R_ref_m),
            mu_m3s2=float(mu_m3s2),
            r_i_to_bf=telem_r_i_to_bf,
            surface_radius_m=telem_surface_radius_m,
        )
        last_telem_t = float(t0) - float(telem_cadence_s)
        rhs_base = rhs

        def rhs(t: float, y: np.ndarray) -> np.ndarray:
            nonlocal last_telem_t
            dy = rhs_base(t, y)
            if (float(t) - float(last_telem_t)) >= float(telem_cadence_s):
                telem_emitter.emit_rhs_probe(float(t), y)
                last_telem_t = float(t)
            return dy

    # -------------------------------------------------------------------------
    # 3) Max-step logic (Nyquist cap vs user cap)
    # -------------------------------------------------------------------------
    degree = get_sh_degree(dynamics)
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
    elif verbose:
        logger.info(
            "[STEP] user_max_step=%gs, nyquist=%0.6fs -> using %0.6fs (reason=%s)",
            step_plan.user_max_step_s,
            step_plan.nyquist_max_step_s,
            max_step,
            step_plan.limiting_reason,
        )

    integration_plan = resolve_integration_plan(cfg, duration_s=dur_s)
    max_internal_steps = int(getattr(cfg, "max_internal_steps", 1_000_000))
    if integration_plan.backend == "scipy":
        # ``solve_ivp`` does not expose an accepted-step hard limit. Its
        # max_step still establishes a strict lower bound, which lets us reject
        # configurations guaranteed to exceed the requested safety budget before
        # expensive integration begins. Fixed-step receives an exact count in
        # ``_integrate_fixed_step`` below.
        minimum_internal_steps = int(math.ceil((tf - t0) / float(max_step)))
        if minimum_internal_steps > max_internal_steps:
            raise ValueError(
                "Adaptive propagation requires at least "
                f"{minimum_internal_steps} internal steps from duration/max_step, "
                f"exceeding max_internal_steps={max_internal_steps}. "
                "Note: for the adaptive (SciPy) backend this is a preflight "
                "feasibility guard on the minimum implied step count only — "
                "solve_ivp itself has no hard cap on accepted steps, so the "
                "actual step count during integration is not limited by "
                "max_internal_steps. Increase the limit or use a larger max_step."
            )
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

    # Every acceleration-form stepper assumes ``a = f(t, r)``; a velocity-
    # dependent force (1PN relativity) is sampled at a stale velocity. The
    # symplectic branch above already reports this for VV/Yoshida/PEFRL, but
    # RKN4 is acceleration-form without being symplectic, so it needs its own
    # guard or the inconsistency passes silently.
    _accel_violations = accel_form_velocity_dependence_violations(_method, _flags)
    if _accel_violations and not symplectic_breaks_separability(_method, _flags):
        _amsg = (
            f"Acceleration-form method {str(_method)!r} assumes a = f(t, r), but "
            f"velocity-dependent force(s) are active: {', '.join(_accel_violations)}. "
            "Stage accelerations are evaluated with a stale/inconsistent "
            "intermediate velocity, causing a silent accuracy loss. Prefer RK4 "
            "or an adaptive method (DOP853/RK45) for velocity-dependent dynamics."
        )
        if bool(getattr(cfg, "strict_symplectic", False)):
            raise ValueError(
                _amsg + " strict_symplectic=True: refusing to run an acceleration-form "
                "method with velocity-dependent forces for a paper/validation run."
            )
        warnings.warn(_amsg, RuntimeWarning, stacklevel=2)

    # Adaptive-degree SH gravity switches degree at discrete altitude
    # thresholds, so the field is a discontinuous function of position — the
    # smooth-Hamiltonian assumption behind the symplectic bounded-drift
    # argument. Read from the prepared gravity pack (build_rhs above stamped
    # ``dynamics._prep``); absent/foreign dynamics objects skip the guard.
    _prep = getattr(dynamics, "_prep", None)
    _gpack = _prep.get("grav") if isinstance(_prep, dict) else None
    _adaptive_violations = symplectic_discontinuous_gravity(_method, _gpack)
    if _adaptive_violations:
        _dmsg = (
            f"Symplectic method {str(_method)!r} is active together with "
            f"{', '.join(_adaptive_violations)}. The bounded-energy-drift "
            "guarantee of symplectic integrators assumes a smooth Hamiltonian; "
            "each degree-threshold crossing injects an energy kick, so drift "
            "may accumulate on orbits that cross thresholds. Use a fixed SH "
            "degree with symplectic methods, or prefer RK4 or an adaptive "
            "method (DOP853/RK45) with adaptive degree."
        )
        if bool(getattr(cfg, "strict_symplectic", False)):
            raise ValueError(
                _dmsg + " strict_symplectic=True: refusing to run a symplectic method "
                "with a discontinuous (adaptive-degree) gravity field for a "
                "paper/validation run."
            )
        warnings.warn(_dmsg, RuntimeWarning, stacklevel=2)

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
            max_internal_steps=max_internal_steps,
            logger=logger,
        )

    else:
        res = run_scipy_propagation(
            rhs=rhs,
            t_eval=t_eval,
            y0=y0_arr,
            t0=float(t0),
            tf=float(tf),
            method=integration_plan.method,
            max_step_s=float(max_step),
            chunk_s=integration_plan.chunk_s,
            cfg=cfg,
            events=events,
            stop_file=stop_file,
            checkpoint_path=checkpoint_path,
            checkpoint_metadata=checkpoint_meta,
            output_dt_s=float(dt_out),
            verbose=verbose,
            logger=logger,
        )

    # -------------------------------------------------------------------------
    # 6) Diagnostics + Optional 2-body baseline
    # -------------------------------------------------------------------------
    wall = time.perf_counter() - t_wall0
    if telem_emitter is not None:
        # Emit the exact solver-returned rows.  This is the shared scientific
        # trajectory boundary for adaptive and fixed-step propagation.
        telem_emitter.emit_trajectory(res.t, res.y)
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
    if telem_emitter is not None:
        for key, value in telem_emitter.diagnostics.as_dict().items():
            res.diagnostics[f"telemetry_{key}"] = value

    if bool(getattr(cfg, "compute_2body_baseline", False)):
        res.baseline = _compute_2body_baseline(
            t_eval=res.t,
            y0=y0_arr[:6],
            mu_m3s2=float(mu_m3s2),
            cfg=cfg,
            max_step=float(max_step),
        )

    return res


def _propagate_with_maneuvers(
    *,
    dynamics: DynamicsEngine,
    y0: np.ndarray,
    cfg: PropagatorConfig,
    time_cfg: TimeConfig | None,
    topo_grid: Any,
    extra_events: Sequence[Callable[[float, np.ndarray], float]] | None,
    maneuver_plan: ManeuverPlan,
) -> PropagationResult:
    """Segment a propagation at burns and publish one post-burn row per boundary.

    Terminal events have precedence over a burn at the same epoch. Checkpoint
    artifacts are rejected because the current checkpoint contract is
    diagnostic/write-only and cannot prove whether a boundary impulse was
    already applied.
    """
    if time_cfg is None:
        raise ValueError("time_cfg is required for maneuver propagation")
    t0 = float(time_cfg.t0_s)
    tf = t0 + float(time_cfg.duration_s)
    maneuver_plan.validate_window(t0, tf)
    if getattr(cfg, "checkpoint_path", None):
        raise ValueError(
            "checkpoint_path is not supported with maneuver_plan: the current "
            "checkpoint schema has no burn-application cursor and resume could "
            "double-apply an impulse"
        )

    maneuver_violations = symplectic_impulsive_maneuver_violations(cfg.method, maneuver_plan)
    if maneuver_violations:
        message = (
            f"Symplectic method {cfg.method!r} cannot carry its smooth-flow "
            "bounded-energy-drift guarantee across an impulsive maneuver."
        )
        if cfg.strict_symplectic:
            raise ValueError(message + " strict_symplectic=True: refusing the run.")
        warnings.warn(message, RuntimeWarning, stacklevel=2)

    telemetry_disabled = bool(cfg.enable_telemetry or cfg.telem_cadence_s > 0.0)
    segment_cfg = replace(
        cfg,
        compute_2body_baseline=False,
        enable_telemetry=False,
        telem_cadence_s=0.0,
        telemetry_sink_path="",
    )

    t_rows: list[float] = []
    y_rows: list[np.ndarray] = []
    event_t_parts: list[list[np.ndarray]] = []
    event_y_parts: list[list[np.ndarray]] = []
    applied: list[dict[str, Any]] = []
    segments: list[PropagationResult] = []
    current_t = t0
    current_y = np.asarray(y0, dtype=np.float64).copy()
    terminal_result: PropagationResult | None = None
    wall_start = time.perf_counter()

    def append_segment(segment: PropagationResult) -> None:
        nonlocal terminal_result
        start = 1 if t_rows and np.isclose(segment.t[0], t_rows[-1], rtol=0.0, atol=1e-12) else 0
        t_rows.extend(float(value) for value in segment.t[start:])
        y_rows.extend(np.asarray(row, dtype=np.float64).copy() for row in segment.y[start:])
        while len(event_t_parts) < len(segment.t_events):
            event_t_parts.append([])
            event_y_parts.append([])
        for index, values in enumerate(segment.t_events):
            event_t_parts[index].append(np.asarray(values, dtype=np.float64))
        for index, values in enumerate(segment.y_events):
            event_y_parts[index].append(np.asarray(values, dtype=np.float64))
        segments.append(segment)
        if segment.impacted or segment.stopped_early:
            terminal_result = segment

    def apply_boundary_burn(maneuver: Any) -> None:
        nonlocal current_y
        current_y, record = apply_impulsive_maneuver(current_y, maneuver)
        applied.append(record)
        if t_rows and np.isclose(t_rows[-1], maneuver.t_burn_s, rtol=0.0, atol=1e-12):
            y_rows[-1] = current_y.copy()
        else:
            t_rows.append(float(maneuver.t_burn_s))
            y_rows.append(current_y.copy())

    for maneuver in maneuver_plan.maneuvers:
        burn_t = float(maneuver.t_burn_s)
        if burn_t > current_t:
            segment_time = replace(time_cfg, t0_s=current_t, duration_s=burn_t - current_t)
            segment = propagate(
                dynamics,
                current_y,
                segment_cfg,
                time_cfg=segment_time,
                topo_grid=topo_grid,
                extra_events=extra_events,
            )
            append_segment(segment)
            current_t = float(segment.t[-1])
            current_y = np.asarray(segment.y[-1], dtype=np.float64).copy()
            if terminal_result is not None:
                break
            if not np.isclose(current_t, burn_t, rtol=0.0, atol=1e-9):
                raise RuntimeError("maneuver segment did not reach its burn boundary")
        apply_boundary_burn(maneuver)
        current_t = burn_t

    if terminal_result is None and current_t < tf:
        segment_time = replace(time_cfg, t0_s=current_t, duration_s=tf - current_t)
        segment = propagate(
            dynamics,
            current_y,
            segment_cfg,
            time_cfg=segment_time,
            topo_grid=topo_grid,
            extra_events=extra_events,
        )
        append_segment(segment)

    if not t_rows:
        t_rows.append(t0)
        y_rows.append(current_y.copy())

    last = terminal_result or (segments[-1] if segments else None)
    diagnostics = dict(last.diagnostics) if last is not None else {}
    diagnostics.update(
        {
            "maneuver_plan_sha256": maneuver_plan.plan_sha256,
            "maneuver_output_contract": "single_timestamp_post_burn_v1",
            "maneuver_event_precedence": "terminal_event_before_burn",
            "maneuvers_requested": len(maneuver_plan.maneuvers),
            "maneuvers_applied": applied,
            "maneuver_segments_completed": len(segments),
            "maneuver_checkpoint_policy": "unsupported_fail_closed",
            "wall_time_s": float(time.perf_counter() - wall_start),
        }
    )
    if cfg.compute_2body_baseline:
        diagnostics["baseline_disabled_reason"] = (
            "maneuver-aware two-body baseline is not implemented; no silent no-burn baseline emitted"
        )
    if telemetry_disabled:
        diagnostics["telemetry_disabled_reason"] = (
            "segmented maneuver telemetry requires a discontinuity-aware replay schema"
        )

    def combine(parts: list[np.ndarray], *, state: bool) -> np.ndarray:
        nonempty = [part for part in parts if part.size]
        if nonempty:
            return np.concatenate(nonempty, axis=0)
        n_state = int(y0.size)
        return np.empty((0, n_state), dtype=np.float64) if state else np.empty(0, dtype=np.float64)

    impacted = bool(last.impacted) if last is not None else False
    stopped_early = bool(last.stopped_early) if last is not None else False
    return PropagationResult(
        t=np.asarray(t_rows, dtype=np.float64),
        y=np.vstack(y_rows),
        ode=None,
        t_events=[combine(parts, state=False) for parts in event_t_parts],
        y_events=[combine(parts, state=True) for parts in event_y_parts],
        impacted=impacted,
        t_impact_s=(last.t_impact_s if last is not None else None),
        y_impact=(last.y_impact if last is not None else None),
        stopped_early=stopped_early,
        stop_reason=(last.stop_reason if last is not None else None),
        t_stop_s=(last.t_stop_s if last is not None else None),
        diagnostics=diagnostics,
        baseline=None,
    )


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
