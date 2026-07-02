"""High-level orbit propagation orchestration.

The heavy helper families live in sibling modules under
``lunaris.core.propagation``. This module keeps the public ``propagate`` entry
point and owns the propagation orchestration surface.
"""

from __future__ import annotations

import json
import math
import time
import warnings
from collections.abc import Callable, Sequence
from types import SimpleNamespace
from typing import Any

import numpy as np
from scipy.integrate import solve_ivp

from lunaris.common.constants import R_MOON
from lunaris.common.math_utils import (
    nyquist_max_step_s,
    recommended_sh_degree,
    specific_energy_drift_stats,
)
from lunaris.common.type_defs import PropagationResult, PropagatorConfig, TimeConfig
from lunaris.core.dynamics import DynamicsEngine
from lunaris.core.propagation.checkpoint import _atomic_save_npz, _stop_requested
from lunaris.core.propagation.events import (
    _build_r_i_to_bf_from_rot_table,
    _find_event_index,
    _get_detect_impact,
    _get_impact_alt_km,
    _terminal_event_endpoint,
    _wrap_event_first6,
    build_events,
)
from lunaris.core.propagation.integrators.fixed_step import (
    _ACCEL_METHODS,
    _RHS_METHODS,
    _accel_stepper,
    _fixed_step_requires_6d,
    _integrate_fixed_step,
    _is_fixed_step_method,
    _is_symplectic_method,
    symplectic_breaks_separability,
    symplectic_nonconservative_gravity,
    symplectic_nonconservative_violations,
)
from lunaris.core.propagation.integrators.rk import _rk4_step_full, _rk8_step_full
from lunaris.core.propagation.integrators.scipy import _resolve_scipy_method
from lunaris.core.propagation.integrators.symplectic import (
    _Y4_WEIGHTS,
    _Y6_WEIGHTS,
    _Y8_WEIGHTS,
    _composition_weights,
)
from lunaris.core.propagation.result import _as_state_array
from lunaris.core.propagation.telemetry import (
    _build_surface_radius_sampler,
    _make_telem_dict,
)
from lunaris.core.propagation.time_grid import (
    _clamp_output_dt,
    _get_ref_radius_and_mu,
    _get_sh_degree,
    _norm_method,
    make_time_grid,
)


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
    vec = np.full(n, atol_scalar, dtype=np.float64)
    if n >= 6:
        if atol_pos is not None:
            vec[0:3] = float(atol_pos)
        if atol_vel is not None:
            vec[3:6] = float(atol_vel)
    return vec


def _osculating_periapsis_alt_km(y: Any, mu_m3s2: float, R_ref_m: float) -> float | None:
    """Return osculating conic periapsis altitude from a Cartesian state.

    The p/(1+e) form works for elliptic, parabolic, and hyperbolic conics when
    angular momentum is non-zero. ``None`` means the state is too degenerate to
    infer a useful periapsis for step-size policy.
    """
    arr = np.asarray(y, dtype=np.float64).reshape(-1)
    if arr.size < 6:
        return None

    mu = float(mu_m3s2)
    R_ref = float(R_ref_m)
    if (not np.isfinite(mu)) or mu <= 0.0 or (not np.isfinite(R_ref)) or R_ref <= 0.0:
        return None

    r = arr[0:3]
    v = arr[3:6]
    if not (np.all(np.isfinite(r)) and np.all(np.isfinite(v))):
        return None

    rn = float(np.linalg.norm(r))
    h_vec = np.cross(r, v)
    h2 = float(np.dot(h_vec, h_vec))
    if rn <= 0.0 or h2 <= 0.0:
        return None

    ecc_vec = np.cross(v, h_vec) / mu - r / rn
    ecc = float(np.linalg.norm(ecc_vec))
    if (not np.isfinite(ecc)) or ecc < 0.0:
        return None

    rp = (h2 / mu) / (1.0 + ecc)
    if (not np.isfinite(rp)) or rp <= 0.0:
        return None
    return float((rp - R_ref) / 1000.0)


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
    except Exception:
        stop_file = None

    checkpoint_path: str | None = None
    try:
        cp = getattr(cfg, "checkpoint_path", None)
        checkpoint_path = (str(cp) if cp else None)
    except Exception:
        checkpoint_path = None

    # -------------------------------------------------------------------------
    # 1) Resolve time grid (STRICT: TimeConfig required)
    # -------------------------------------------------------------------------
    if time_cfg is None:
        raise ValueError("time_cfg is required (STRICT). Provide TimeConfig(duration_s=..., output_dt_s=...).")

    if getattr(time_cfg, "duration_s", None) is None:
        raise ValueError("time_cfg.duration_s is required and must be finite/positive.")
    dt_out_raw = getattr(time_cfg, "output_dt_s", None)
    dur_s = float(time_cfg.duration_s)
    if dur_s <= 0.0 or (not np.isfinite(dur_s)):
        raise ValueError("Duration must be positive and finite.")

    # Start/end times (t0 belongs to TimeConfig; default 0 if omitted)
    t0 = float(getattr(time_cfg, "t0_s", 0.0) or 0.0)
    if not np.isfinite(t0):
        raise ValueError("time_cfg.t0_s must be finite.")
    tf = t0 + dur_s

    # Resolve output sampling step
    if dt_out_raw is None:
        # Allow "output_dt_s=None" by deriving a reasonable sampling step from the
        # osculating Keplerian period estimated from the initial state (Kepler two-body).
        # (This matches the intent of TimeConfig.samples_per_period.)
        _, mu = _get_ref_radius_and_mu(dynamics)
        mu = float(mu)

        r0 = float(np.linalg.norm(y0_arr[:3]))
        v0 = float(np.linalg.norm(y0_arr[3:6]))
        if not (math.isfinite(r0) and math.isfinite(v0) and r0 > 0.0 and mu > 0.0):
            raise ValueError("Cannot derive output_dt_s: invalid initial state or mu.")

        denom = (2.0 / r0) - (v0 * v0 / mu)
        if denom <= 0.0 or (not math.isfinite(denom)):
            raise ValueError(
                "time_cfg.output_dt_s is None, but the orbit appears unbound/degenerate. "
                "Set output_dt_s explicitly."
            )

        a = 1.0 / denom
        T = 2.0 * math.pi * math.sqrt((a * a * a) / mu)

        spp = int(getattr(time_cfg, "samples_per_period", 360) or 360)
        spp = max(1, spp)
        dt_out_user = float(T) / float(spp)
    else:
        dt_out_user = float(dt_out_raw)

    if dt_out_user <= 0.0 or (not np.isfinite(dt_out_user)):
        raise ValueError("time_cfg.output_dt_s must be positive and finite.")
    # Cap output points (owned by PropagatorConfig; TimeConfig may optionally override)
    max_points_cap = int(getattr(time_cfg, "max_points_cap", getattr(cfg, "max_points_cap", 200_000)))

    dt_out = _clamp_output_dt(t0, tf, float(dt_out_user), max_points_cap, verbose)
    t_eval = make_time_grid(t0, tf, dt_out)

    rhs = dynamics.build_rhs()
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
        except Exception:
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

    nyq_max: float | None = None
    nyq_r_min_alt_km: float | None = None
    if bool(getattr(cfg, "use_nyquist_max_step", False)):
        try:
            guard_alt_km = float(_get_impact_alt_km(cfg) if _get_detect_impact(cfg) else 0.0)
            if topo_present:
                guard_alt_km = 0.0
            peri_alt_km = _osculating_periapsis_alt_km(y0_arr, float(mu_m3s2), float(R_ref_m))
            nyq_r_min_alt_km = (
                max(float(guard_alt_km), float(peri_alt_km))
                if peri_alt_km is not None
                else float(guard_alt_km)
            )
            nyq_max = float(nyquist_max_step_s(
                R_ref_m=float(R_ref_m),
                mu_m3s2=float(mu_m3s2),
                degree=int(max(1, degree)),
                r_min_alt_km=float(nyq_r_min_alt_km),
                safety_div=float(getattr(cfg, "nyquist_safety_div", 8.0)),
                v_margin=float(getattr(cfg, "nyquist_v_margin", 1.2)),
            ))
        except Exception:
            nyq_max = None

    if nyq_max is None or (not np.isfinite(nyq_max)) or nyq_max <= 0.0:
        nyq_max = float(dt_out)

    user_max_step_s = getattr(cfg, "user_max_step_s", None)
    if user_max_step_s is None:
        max_step = float(nyq_max)
        if verbose:
            print(f"[STEP] Nyquist max_step_s={max_step:.6f} (deg={degree})", flush=True)
    else:
        max_step = min(float(user_max_step_s), float(nyq_max))
        if verbose:
            print(f"[STEP] user_max_step={float(user_max_step_s):g}s, nyquist={nyq_max:.6f}s -> using {max_step:.6f}s", flush=True)

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
    # The gravity model itself can void the guarantee: a force_direct ST-LRPS
    # surrogate predicts acceleration directly (no underlying scalar potential),
    # so it is non-conservative by construction even with every perturbation
    # flag off. potential_autograd surrogates and classical SH stay exempt.
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
    if _is_fixed_step_method(getattr(cfg, "method", "DOP853")):
        meth_name = str(getattr(cfg, "method", "VV"))
        if verbose:
            print(f"[PROP] Fixed-step {meth_name}: dt_out={dt_out:g}s, max_step={max_step:.6f}s", flush=True)

        if _fixed_step_requires_6d(meth_name) and y0_arr.size != 6:
            raise ValueError(
                f"Fixed-step method {meth_name!r} (symplectic/Nystrom) supports only the 6D state "
                "[x,y,z,vx,vy,vz]. "
                f"Got initial state size={int(y0_arr.size)}. Use RK4 or a SciPy integrator (e.g., "
                "DOP853/RK45) for augmented states."
            )

        y0_fixed = y0_arr[:6] if _fixed_step_requires_6d(meth_name) else y0_arr

        ode_like, impacted, t_imp, y_imp, stopped_early, stop_reason, t_stop = _integrate_fixed_step(
            rhs=rhs,
            t_eval=t_eval,
            y0=y0_fixed,
            max_step=max_step,
            method=meth_name,
            events=events,
            R_ref_m=float(R_ref_m),
            mu_m3s2=float(mu_m3s2),
            verbose=verbose,
            heartbeat_hours=float(getattr(cfg, "heartbeat_hours", 0.0)),
            stop_file=stop_file,
            checkpoint_path=checkpoint_path,
        )

        t_out = np.asarray(ode_like.t, dtype=np.float64)
        y_row = np.asarray(ode_like.y, dtype=np.float64).T  # (N,6)

        res = PropagationResult(
            t=t_out,
            y=y_row,
            ode=ode_like,
            impacted=bool(impacted),
            t_impact_s=(float(t_imp) if t_imp is not None else None),
            y_impact=(np.asarray(y_imp, dtype=np.float64) if y_imp is not None else None),
            stopped_early=bool(stopped_early),
            stop_reason=stop_reason if stop_reason else ("impact" if impacted else None),
            t_stop_s=t_stop,
            diagnostics={},
        )

    else:
        if solve_ivp is None:
            raise ImportError("SciPy is required for adaptive integration (solve_ivp not available).")

        method = _resolve_scipy_method(getattr(cfg, "method", "DOP853"))

        # Per-component (vector) atol when configured: position and velocity differ
        # by ~3 orders of magnitude, so a single scalar over-tightens one of them.
        atol_arg = _resolve_atol(cfg, int(y0_arr.size))

        if verbose:
            if isinstance(atol_arg, np.ndarray):
                print(
                    f"[PROP] solve_ivp method={method} | dt_out={dt_out:g}s | max_step={max_step:.6f}s "
                    f"| atol=vector(pos={getattr(cfg, 'atol_pos', None)}, vel={getattr(cfg, 'atol_vel', None)})",
                    flush=True,
                )
            else:
                print(f"[PROP] solve_ivp method={method} | dt_out={dt_out:g}s | max_step={max_step:.6f}s", flush=True)

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

        total_span = tf - t0
        chunk_s = getattr(cfg, "chunk_s", None)
        if chunk_s is not None:
            chunk_s = float(chunk_s)
            if chunk_s <= 0.0 or chunk_s >= total_span:
                chunk_s = None

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
                            )
                        elif ck_mode in ("chunks", "chunk"):
                            base = str(checkpoint_path)
                            chunk_path = f"{base}.chunk{chunk_idx:06d}.npz"
                            _atomic_save_npz(chunk_path, t=sol_t, y_row=sol_y.T)
                            _atomic_save_npz(
                                checkpoint_path,
                                t=np.asarray([t_curr], dtype=np.float64),
                                y_row=y_curr.reshape(1, -1),
                            )
                        else:
                            t_tmp = np.concatenate(t_parts) if t_parts else np.array([], dtype=np.float64)
                            y_tmp = np.concatenate(y_parts, axis=1) if y_parts else np.zeros((y0_arr.size, 0), dtype=np.float64)
                            _atomic_save_npz(checkpoint_path, t=t_tmp, y_row=y_tmp.T)
                    except Exception as exc:
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

        impacted = False
        t_imp = None
        y_imp = None
        idx_impact = _find_event_index(events, "impact")
        if idx_impact is not None:
            try:
                if idx_impact < len(t_events) and np.asarray(t_events[idx_impact]).size > 0:
                    impacted = True
                    t_imp = float(np.asarray(t_events[idx_impact])[0])
                    y_imp = np.asarray(np.asarray(y_events[idx_impact])[0], dtype=np.float64)
            except Exception:
                pass

        try:
            if impacted:
                stop_reason = "impact"
            elif stop_reason is None:
                idx_stop = _find_event_index(events, "stop")
                if (
                    stop_file and bool(getattr(cfg, "stop_event_in_scipy", False))
                    and idx_stop is not None and idx_stop < len(t_events)
                    and np.asarray(t_events[idx_stop]).size > 0
                ):
                    stop_reason = "stop file"
                if stop_reason is None and any((te is not None and np.asarray(te).size > 0) for te in t_events):
                    stop_reason = "event"
        except Exception:
            pass

        t_stop = None
        if stop_file and bool(getattr(cfg, "stop_event_in_scipy", False)) and (not impacted):
            idx_stop = _find_event_index(events, "stop")
            if idx_stop is not None:
                try:
                    if idx_stop < len(t_events) and np.asarray(t_events[idx_stop]).size > 0:
                        t_stop = float(np.asarray(t_events[idx_stop])[0])
                except Exception:
                    pass

        if checkpoint_path and (not integration_failed) and not (chunk_s is not None and checkpoint_every_chunk):
            try:
                _atomic_save_npz(checkpoint_path, t=np.asarray(t_cat, dtype=np.float64), y_row=y_row)
            except Exception as exc:
                warnings.warn(f"Checkpoint write failed: {exc}", RuntimeWarning, stacklevel=2)

        res = PropagationResult(
            t=np.asarray(t_cat, dtype=np.float64),
            y=y_row,
            ode=sol,
            t_events=list(t_events),
            y_events=list(y_events),
            impacted=bool(impacted),
            t_impact_s=(float(t_imp) if t_imp is not None else None),
            y_impact=(np.asarray(y_imp, dtype=np.float64) if y_imp is not None else None),
            stopped_early=bool(stopped_early) or bool(impacted),
            stop_reason=stop_reason,
            t_stop_s=t_stop,
            diagnostics={},
        )

    # -------------------------------------------------------------------------
    # 6) Diagnostics + Optional 2-body baseline
    # -------------------------------------------------------------------------
    wall = time.perf_counter() - t_wall0
    nfev = float(getattr(res.ode, "nfev", np.nan)) if res.ode is not None else np.nan
    res.diagnostics = {
        "wall_time_s": float(wall),
        "output_dt_s": float(dt_out),
        "max_step_s": float(max_step),
        "degree": float(degree),
        "n_points": float(res.t.size),
        "nfev": float(nfev) if np.isfinite(nfev) else float("nan"),
        "method_symplectic": float(1.0 if _is_symplectic_method(getattr(cfg, "method", "DOP853")) else 0.0),
        "symplectic_violation": float(1.0 if _violations else 0.0),
    }
    if nyq_r_min_alt_km is not None:
        res.diagnostics["nyquist_r_min_alt_km"] = float(nyq_r_min_alt_km)
    if _violations:
        res.diagnostics["symplectic_violation_forces"] = list(_violations)

    # Energy / angular-momentum drift over the trajectory. This is a combined
    # physical+numerical drift on the full run (a bounded oscillation is physical;
    # a monotone secular trend signals numerical error). It is a *pure* numerical
    # accuracy proxy on the conservative/autonomous 2-body baseline below.
    try:
        for _k, _v in specific_energy_drift_stats(res.t, res.y, float(mu_m3s2)).items():
            res.diagnostics[_k] = float(_v)
    except Exception:
        pass

    # SH truncation-degree adequacy for the orbit's periapsis altitude. Below the
    # recommended degree, low-degree truncation (not the integrator) is the
    # dominant position-error term, so we surface it rather than let it pass.
    try:
        if int(degree) >= 2 and y0_arr.size >= 6 and float(mu_m3s2) > 0.0:
            alt_peri_km = _osculating_periapsis_alt_km(y0_arr, float(mu_m3s2), float(R_ref_m))
            if alt_peri_km is not None:
                rec_deg = recommended_sh_degree(alt_peri_km, float(R_ref_m))
                res.diagnostics["periapsis_alt_km"] = float(alt_peri_km)
                res.diagnostics["recommended_degree"] = float(rec_deg)
                if rec_deg > int(degree):
                    _deg_msg = (
                        f"SH truncation degree={int(degree)} may be too low for periapsis "
                        f"altitude {alt_peri_km:.1f} km: upward continuation suggests degree "
                        f">= {rec_deg} to retain gravity signal above the 1e-3 floor. "
                        "Low-degree truncation, not the integrator, is then the dominant "
                        "position-error term."
                    )
                    warnings.warn(_deg_msg, RuntimeWarning, stacklevel=2)
                    if verbose:
                        print(f"[GRAV] {_deg_msg}", flush=True)
    except Exception:
        pass

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

        dy = np.empty(6, dtype=np.float64)
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
    except Exception:
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
    "make_time_grid",
    "build_events",
    "_ACCEL_METHODS",
    "_RHS_METHODS",
    "_Y4_WEIGHTS",
    "_Y6_WEIGHTS",
    "_Y8_WEIGHTS",
    "_accel_stepper",
    "_composition_weights",
    "_is_fixed_step_method",
    "_is_symplectic_method",
    "symplectic_nonconservative_violations",
    "symplectic_nonconservative_gravity",
    "symplectic_breaks_separability",
    "_norm_method",
    "_rk4_step_full",
    "_rk8_step_full",
]
