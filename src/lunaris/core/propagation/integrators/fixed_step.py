"""Fixed-step propagation driver and method dispatch."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import numpy as np

from lunaris.core.propagation.checkpoint import _atomic_save_npz, _stop_requested
from lunaris.core.propagation.events import _event_crossed, _refine_event_time_bisect
from lunaris.core.propagation.integrators.rk import _rk4_step_full, _rk8_step_full, _rkn4_step
from lunaris.core.propagation.integrators.symplectic import (
    _pefrl_step,
    _vv_step,
    _y4_step,
    _y6_step,
    _y8_step,
)
from lunaris.core.propagation.telemetry import _make_telem_dict
from lunaris.core.propagation.time_grid import _norm_method


# Acceleration-based symplectic + Nystrom methods operate on the 6-D [r, v]
# state only. RK4 operates on the full state (augmented states allowed).
_ACCEL_METHODS: dict[str, str] = {
    "VV": "VV", "VERLET": "VV", "STORMER_VERLET": "VV", "STÖRMER_VERLET": "VV",
    "LEAPFROG": "VV",
    "Y4": "Y4", "YOSHIDA4": "Y4",
    "Y6": "Y6", "YOSHIDA6": "Y6",
    "Y8": "Y8", "YOSHIDA8": "Y8",
    "PEFRL": "PEFRL",
    "RKN4": "RKN4", "RKN": "RKN4",
}
_SYMPLECTIC_CANONICAL = frozenset({"VV", "Y4", "Y6", "Y8", "PEFRL"})
_RHS_METHODS: dict[str, str] = {"RK4": "RK4", "RK8": "RK8"}

def _is_symplectic_method(method: str) -> bool:
    """True for the structure-preserving fixed-step methods (Verlet/PEFRL/Yoshida)."""
    canonical = _ACCEL_METHODS.get(_norm_method(method))
    return canonical in _SYMPLECTIC_CANONICAL

def _is_fixed_step_method(method: str) -> bool:
    """True for every in-house fixed-step method (symplectic, Nystrom, or RK)."""
    m = _norm_method(method)
    return (m in _ACCEL_METHODS) or (m in _RHS_METHODS)

def _fixed_step_requires_6d(method: str) -> bool:
    """Acceleration-based methods (symplectic + RKN) support only the 6-D state."""
    return _norm_method(method) in _ACCEL_METHODS

def _accel_stepper(canonical: str) -> Callable[[Callable[[float, np.ndarray], np.ndarray], float, np.ndarray, float], np.ndarray]:
    return {
        "VV": _vv_step,
        "Y4": _y4_step,
        "Y6": _y6_step,
        "Y8": _y8_step,
        "PEFRL": _pefrl_step,
        "RKN4": _rkn4_step,
    }[canonical]

def _build_fixed_stepper(
    method: str,
    rhs: Callable[[float, np.ndarray], np.ndarray],
    accel: Callable[[float, np.ndarray], np.ndarray],
) -> tuple[Callable[[float, np.ndarray, float], np.ndarray], bool]:
    """Return ``(step(t, y, h) -> y_next, requires_6d)`` for a fixed-step method."""
    m = _norm_method(method)
    if m in _RHS_METHODS:
        rhs_stepper = _rk8_step_full if _RHS_METHODS[m] == "RK8" else _rk4_step_full

        def step_rhs(t: float, y: np.ndarray, h: float) -> np.ndarray:
            return rhs_stepper(rhs, t, y, h)

        return step_rhs, False

    canonical = _ACCEL_METHODS.get(m)
    if canonical is None:
        raise ValueError(f"Unknown fixed-step method: {method!r}")

    base = _accel_stepper(canonical)

    def step_accel(t: float, y: np.ndarray, h: float) -> np.ndarray:
        return base(accel, t, y, h)

    return step_accel, True

def _integrate_fixed_step(
    rhs: Callable[[float, np.ndarray], np.ndarray],
    t_eval: np.ndarray,
    y0: np.ndarray,
    *,
    max_step: float,
    method: str,
    events: list[Callable[[float, np.ndarray], float]] | None,
    R_ref_m: float,
    mu_m3s2: float,
    verbose: bool,
    heartbeat_hours: float,
    stop_file: str | None,
    checkpoint_path: str | None,
) -> tuple[Any, bool, float | None, np.ndarray | None, bool, str | None, float | None]:
    """Integrate with an in-house fixed-step method (symplectic, RKN, or RK4).

    Notes
    -----
    - Acceleration-based methods (Verlet/PEFRL/Yoshida/RKN4) support ONLY the 6D
      state [r,v]. The full-state RK4 also accepts augmented states (e.g. mass).
    - Events are supported (including non-impact events) and can be refined inside
      each step using a bisection scheme.
    """
    t_eval = np.asarray(t_eval, dtype=np.float64)
    if t_eval.size < 2 or np.any(np.diff(t_eval) <= 0.0):
        raise ValueError("t_eval must be strictly increasing and contain at least 2 points.")

    y0 = np.asarray(y0, dtype=np.float64).reshape(-1)
    if y0.size < 6:
        raise ValueError("Fixed-step integration requires a state of at least 6 elements [x,y,z,vx,vy,vz].")
    if _fixed_step_requires_6d(method) and y0.size != 6:
        raise ValueError(
            f"Fixed-step method {method!r} (symplectic/Nystrom) supports only the 6D state [r,v]; "
            f"got size={int(y0.size)}. Use RK4 or a SciPy integrator for augmented states."
        )

    max_step = float(max_step)
    if (not np.isfinite(max_step)) or max_step <= 0.0:
        raise ValueError("max_step must be positive and finite for fixed-step integration.")

    # Acceleration adapter: avoid extra allocations when rhs already returns ndarray
    def accel(t: float, y6: np.ndarray) -> np.ndarray:
        dy = rhs(t, y6)
        if isinstance(dy, np.ndarray):
            return dy[3:6]
        # Fallback (should be rare)
        a = np.asarray(dy, dtype=np.float64).reshape(-1)
        return a[3:6]

    step, _requires_6d = _build_fixed_stepper(method, rhs, accel)

    # ------------------------------------------------------------------
    # Events: support all events, with optional refinement
    # ------------------------------------------------------------------
    ev_list: list[Callable[[float, np.ndarray], float]] = list(events) if events else []

    n_ev = len(ev_list)
    t_events_acc: list[list[float]] = [[] for _ in range(n_ev)]
    y_events_acc: list[list[np.ndarray]] = [[] for _ in range(n_ev)]

    # Initialize previous event values at the start time
    t_start = float(t_eval[0])
    g_prev: list[float] = []
    for ev in ev_list:
        try:
            g_prev.append(float(ev(t_start, y0)))
        except Exception:
            g_prev.append(float("nan"))

    t_list: list[float] = [t_start]
    y_list: list[np.ndarray] = [y0.copy()]

    impacted = False
    t_imp: float | None = None
    y_imp: np.ndarray | None = None

    stopped_early = False
    stop_reason: str | None = None
    t_stop: float | None = None

    last_hb_hr = 0.0
    alt_min_km = float("inf")
    alt_max_km = float("-inf")




    # Refinement controls (can be overridden by attaching attrs to rhs)
    refine_tol_s = float(getattr(rhs, "_fixed_step_event_tol_s", 1e-6))
    refine_max_iter = int(getattr(rhs, "_fixed_step_event_max_iter", 30))

    for k in range(t_eval.size - 1):
        if _stop_requested(stop_file):
            stopped_early = True
            stop_reason = "stop file"
            break

        t_seg0 = float(t_list[-1])
        t_target = float(t_eval[k + 1])
        dt_seg = t_target - t_seg0
        if dt_seg <= 0.0:
            continue

        n_sub = int(math.ceil(dt_seg / max_step)) if dt_seg > max_step else 1
        n_sub = max(1, n_sub)
        h = dt_seg / float(n_sub)

        y_curr = y_list[-1].copy()

        for j in range(n_sub):
            tj = t_seg0 + j * h
            y_next = step(tj, y_curr, h)
            t_next = tj + h

            earliest_terminal: tuple[float, int, np.ndarray] | None = None  # (t_event, idx, y_event)

            for i, ev in enumerate(ev_list):
                try:
                    g0 = float(g_prev[i])
                    g1 = float(ev(t_next, y_next))
                except Exception:
                    g_prev[i] = float("nan")
                    continue

                direction = float(getattr(ev, "direction", 0.0))
                terminal = bool(getattr(ev, "terminal", False))

                if _event_crossed(g0, g1, direction):
                    # Refine root within this substep
                    try:
                        t_ev, y_ev = _refine_event_time_bisect(
                            step=step,
                            ev=ev,
                            t0=tj,
                            y0=y_curr,
                            h=h,
                            g0=g0,
                            g1=g1,
                            max_iter=refine_max_iter,
                            tol_s=refine_tol_s,
                        )
                    except Exception:
                        # Fallback: linear interpolation in event function value
                        denom = (g0 - g1)
                        if denom != 0.0 and np.isfinite(denom):
                            tau = float(min(1.0, max(0.0, g0 / denom)))
                        else:
                            tau = 0.5
                        t_ev = tj + tau * h
                        y_ev = np.asarray(y_curr + tau * (y_next - y_curr), dtype=np.float64)

                    t_events_acc[i].append(float(t_ev))
                    y_events_acc[i].append(np.asarray(y_ev, dtype=np.float64))

                    if terminal:
                        if (earliest_terminal is None) or (t_ev < earliest_terminal[0]):
                            earliest_terminal = (float(t_ev), i, np.asarray(y_ev, dtype=np.float64))

                # Update previous value for next substep
                g_prev[i] = g1

            # Terminal event: stop at earliest terminal root in this substep
            if earliest_terminal is not None:
                t_ev, i_ev, y_ev = earliest_terminal
                ev_role = getattr(ev_list[i_ev], "_event_role", None)

                t_list.append(float(t_ev))
                y_list.append(np.asarray(y_ev, dtype=np.float64))

                stopped_early = True
                if ev_role == "impact":
                    impacted = True
                    t_imp = float(t_ev)
                    y_imp = np.asarray(y_ev, dtype=np.float64)
                    stop_reason = "impact"
                elif ev_role in ("stop", "stop_file", "stopfile"):
                    stop_reason = "stop file"
                else:
                    stop_reason = "event"
                break

            # No terminal event: accept full step
            y_curr = np.asarray(y_next, dtype=np.float64)

            # Periodic stop-file polling
            if (j % 50) == 0 and _stop_requested(stop_file):
                stopped_early = True
                stop_reason = "stop file"
                break

        if stopped_early:
            break

        t_list.append(float(t_target))
        y_list.append(np.asarray(y_curr, dtype=np.float64))

        # Heartbeat
        if heartbeat_hours and heartbeat_hours > 0.0:
            t_hr = (t_target - float(t_eval[0])) / 3600.0
            alt_now_km = (float(np.linalg.norm(y_curr[0:3])) - float(R_ref_m)) / 1000.0
            alt_min_km = min(alt_min_km, alt_now_km)
            alt_max_km = max(alt_max_km, alt_now_km)
            if (t_hr - last_hb_hr) >= float(heartbeat_hours):
                if verbose:
                    print(
                        f"[HB] t={t_hr:7.2f} h | alt={alt_now_km:9.3f} km | min={alt_min_km:9.3f} | max={alt_max_km:9.3f}",
                        flush=True,
                    )
                last_hb_hr = t_hr

    if stopped_early and t_list:
        t_stop = float(t_list[-1])

    t_arr = np.asarray(t_list, dtype=np.float64)
    y_arr = np.asarray(y_list, dtype=np.float64)  # (N,6)

    # Convert accumulated event hits
    t_events = [np.asarray(te, dtype=np.float64) for te in t_events_acc]
    y_events = [
        (np.vstack(ye).astype(np.float64, copy=False) if len(ye) else np.zeros((0, y0.size), dtype=np.float64))
        for ye in y_events_acc
    ]

    ode_like = SimpleNamespace(
        t=t_arr,
        y=y_arr.T,  # mimic SciPy (6,N)
        success=True,
        status=(1 if (impacted or stopped_early) else 0),
        message=("fixed-step ok" if not stopped_early else "stopped early"),
        nfev=np.nan,
        t_events=t_events,
        y_events=y_events,
    )

    if checkpoint_path:
        try:
            _atomic_save_npz(checkpoint_path, t=t_arr, y_row=y_arr)
        except Exception as exc:
            import warnings
            warnings.warn(f"Checkpoint write failed: {exc}", RuntimeWarning, stacklevel=2)

    return (
        ode_like,
        impacted,
        (float(t_imp) if t_imp is not None else None),
        (np.asarray(y_imp, dtype=np.float64) if y_imp is not None else None),
        stopped_early,
        stop_reason,
        t_stop,
    )


__all__ = [
    "_ACCEL_METHODS",
    "_SYMPLECTIC_CANONICAL",
    "_RHS_METHODS",
    "_is_symplectic_method",
    "_is_fixed_step_method",
    "_fixed_step_requires_6d",
    "_accel_stepper",
    "_build_fixed_stepper",
    "_integrate_fixed_step",
]
