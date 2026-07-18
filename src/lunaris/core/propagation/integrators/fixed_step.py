"""Fixed-step propagation driver and method dispatch."""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import numpy as np

from lunaris.common.integrator_methods import FIXED_STEP_METHOD_ALIASES
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
from lunaris.core.propagation.time_grid import _norm_method

logger = logging.getLogger(__name__)

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

if set(_ACCEL_METHODS) | set(_RHS_METHODS) != set(FIXED_STEP_METHOD_ALIASES):
    raise RuntimeError("Fixed-step method registry drifted from lunaris.common.integrator_methods.")

def _is_symplectic_method(method: str) -> bool:
    """True for the structure-preserving fixed-step methods (Verlet/PEFRL/Yoshida)."""
    canonical = _ACCEL_METHODS.get(_norm_method(method))
    return canonical in _SYMPLECTIC_CANONICAL

# Perturbations that void the symplectic guarantee. SH gravity, third-body, and
# Earth J2 are position-only (conservative): a symplectic integrator still
# preserves a modified Hamiltonian, so it remains the correct choice for them.
# The flags below are different:
#   - SRP / albedo / thermal IR are non-conservative external forces (not
#     derivable from a Moon-centered potential; discontinuous across eclipse),
#     so the bounded-energy-drift guarantee no longer holds.
#   - 1PN relativity is *velocity-dependent*, which additionally breaks the
#     separable ``H = T(p) + V(q)`` form the acceleration-based steppers assume
#     (they sample the force at a partially updated velocity). This is worse than
#     "merely non-conservative" and is flagged separately.
# Each entry is ``(flag_attr, human_label, breaks_separability)``.
_SYMPLECTIC_VOIDING_FLAGS: tuple[tuple[str, str, bool], ...] = (
    ("enable_srp", "SRP", False),
    ("enable_albedo", "albedo", False),
    ("enable_thermal", "thermal IR", False),
    ("enable_relativity_1pn", "1PN relativity", True),
)

def symplectic_nonconservative_violations(method: str, flags: Any) -> list[str]:
    """Return human labels of active perturbations that void symplecticity.

    Empty when ``method`` is not symplectic or when only conservative
    (position-only) perturbations are active. ``flags`` is any object exposing
    the :class:`PerturbationFlags` boolean attributes; missing attributes are
    treated as ``False`` so this is safe on partial/legacy flag objects.
    """
    if flags is None or not _is_symplectic_method(method):
        return []
    out: list[str] = []
    for attr, label, _breaks_sep in _SYMPLECTIC_VOIDING_FLAGS:
        if bool(getattr(flags, attr, False)):
            out.append(label)
    # Legacy thermal flag alias (some callers set enable_thermal_ir directly).
    if "thermal IR" not in out and bool(getattr(flags, "enable_thermal_ir", False)):
        out.append("thermal IR")
    return out

def symplectic_nonconservative_gravity(method: str, gravity_model: Any) -> list[str]:
    """Return labels when the gravity provider itself voids symplecticity.

    Classical SH gravity is the gradient of a scalar potential (conservative),
    and so is the supported ST-LRPS ``potential_autograd`` surrogate (its
    acceleration is the autograd gradient of a learned potential). A gravity
    provider that is *not* conservative by construction voids the
    bounded-energy-drift guarantee of a symplectic method. This is decided from
    the provider's ``is_conservative`` taxonomy flag (never an ``isinstance``
    check), so it stays correct if a non-conservative kind is ever reintroduced.
    Empty when ``method`` is not symplectic or no surrogate provider is attached.
    """
    if gravity_model is None or not _is_symplectic_method(method):
        return []
    if getattr(gravity_model, "model_kind", None) != "st_lrps":
        return []
    is_conservative = getattr(gravity_model, "is_conservative", True)
    if not bool(is_conservative):
        return ["non-conservative surrogate gravity (bounded energy drift not guaranteed)"]
    return []

def accel_form_velocity_dependence_violations(method: str, flags: Any) -> list[str]:
    """Return labels of active velocity-dependent forces under an acceleration-form method.

    Every acceleration-based stepper (symplectic *and* RKN4) assumes
    ``a = f(t, r)``: stage accelerations are evaluated with the step-start (or a
    partially updated) velocity. A velocity-dependent force such as 1PN
    relativity is therefore sampled at a stale/inconsistent velocity — a silent
    accuracy loss that applies to RKN4 just as much as to the symplectic
    methods, even though RKN4 carries no symplectic guarantee to void. Empty
    when ``method`` is not acceleration-form or no velocity-dependent force is
    active. ``flags`` follows the same duck-typed contract as
    :func:`symplectic_nonconservative_violations`.
    """
    if flags is None or _norm_method(method) not in _ACCEL_METHODS:
        return []
    return [
        label
        for attr, label, breaks_sep in _SYMPLECTIC_VOIDING_FLAGS
        if breaks_sep and bool(getattr(flags, attr, False))
    ]

def symplectic_discontinuous_gravity(method: str, gravity_pack: Any) -> list[str]:
    """Return labels when adaptive-degree SH gravity voids the symplectic argument.

    Adaptive-degree gravity switches the evaluated SH degree at discrete
    altitude thresholds, so the acceleration is a *discontinuous* function of
    position. The bounded-energy-drift argument of a symplectic integrator
    assumes a smooth Hamiltonian; an orbit crossing a threshold each revolution
    accumulates energy kicks. Empty when ``method`` is not symplectic or the
    pack does not enable adaptive degree. ``gravity_pack`` is any object
    exposing the ``_GravPack.adaptive_enabled`` boolean; ``None`` is safe.
    """
    if gravity_pack is None or not _is_symplectic_method(method):
        return []
    if bool(getattr(gravity_pack, "adaptive_enabled", False)):
        return ["adaptive-degree SH gravity (field discontinuous at altitude thresholds)"]
    return []


def symplectic_impulsive_maneuver_violations(method: str, maneuver_plan: Any) -> list[str]:
    """Return a violation for a discontinuous velocity jump under a symplectic method."""
    if not _is_symplectic_method(method):
        return []
    maneuvers = getattr(maneuver_plan, "maneuvers", ())
    return ["impulsive maneuver velocity discontinuity"] if maneuvers else []

def symplectic_breaks_separability(method: str, flags: Any) -> bool:
    """True when a *velocity-dependent* force is active under a symplectic method.

    This is the stronger failure mode: the acceleration-based steppers assume
    ``a = f(t, r)`` and sample velocity-dependent forces (1PN relativity)
    inconsistently, not just non-symplectically.
    """
    if flags is None or not _is_symplectic_method(method):
        return False
    return any(
        breaks_sep and bool(getattr(flags, attr, False))
        for attr, _label, breaks_sep in _SYMPLECTIC_VOIDING_FLAGS
    )

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


def _event_label(index: int, event: Callable[[float, np.ndarray], float]) -> str:
    """Return stable context for failures in an event callback."""
    role = str(getattr(event, "_event_role", "") or "").strip()
    return f"event index {index}" + (f" ({role})" if role else "")


def _evaluate_event(
    event: Callable[[float, np.ndarray], float],
    *,
    index: int,
    t_s: float,
    y: np.ndarray,
) -> float:
    """Evaluate one fixed-step event without hiding callback failures."""
    label = _event_label(index, event)
    try:
        value = float(event(float(t_s), y))
    except Exception as exc:
        raise RuntimeError(
            f"Fixed-step {label} evaluation failed at t={float(t_s):.12g} s."
        ) from exc
    if not np.isfinite(value):
        raise ValueError(
            f"Fixed-step {label} returned non-finite value {value!r} "
            f"at t={float(t_s):.12g} s."
        )
    return value


def _planned_fixed_step_count(t_eval: np.ndarray, max_step: float) -> int:
    """Return the exact number of fixed substeps implied by the output grid."""
    return sum(
        max(1, int(math.ceil(float(t_eval[i + 1] - t_eval[i]) / max_step)))
        for i in range(t_eval.size - 1)
    )

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
    checkpoint_metadata: dict[str, Any] | None = None,
    max_internal_steps: int | None = None,
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

    planned_internal_steps = _planned_fixed_step_count(t_eval, max_step)
    if max_internal_steps is not None:
        limit = int(max_internal_steps)
        if limit < 1:
            raise ValueError("max_internal_steps must be >= 1 for fixed-step integration.")
        if planned_internal_steps > limit:
            raise ValueError(
                "Fixed-step propagation would require "
                f"{planned_internal_steps} internal steps, exceeding "
                f"max_internal_steps={limit}. Increase the limit or use a larger max_step."
            )

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
    g_prev = [
        _evaluate_event(ev, index=i, t_s=t_start, y=y0)
        for i, ev in enumerate(ev_list)
    ]

    t_list: list[float] = [t_start]
    y_list: list[np.ndarray] = [y0.copy()]

    impacted = False
    t_imp: float | None = None
    y_imp: np.ndarray | None = None

    stopped_early = False
    stop_reason: str | None = None
    t_stop: float | None = None

    executed_internal_steps = 0

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
            executed_internal_steps += 1

            # Phase 1: collect every refined crossing in this substep as a
            # candidate. Nothing is committed yet — a terminal root may
            # invalidate candidates that land after it.
            crossings: list[tuple[int, float, np.ndarray, bool]] = []  # (idx, t_event, y_event, terminal)

            for i, ev in enumerate(ev_list):
                g0 = float(g_prev[i])
                if not np.isfinite(g0):
                    raise ValueError(
                        f"Fixed-step {_event_label(i, ev)} has a non-finite prior value "
                        f"at t={tj:.12g} s."
                    )
                g1 = _evaluate_event(ev, index=i, t_s=t_next, y=y_next)

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
                    except Exception as exc:
                        raise RuntimeError(
                            f"Fixed-step {_event_label(i, ev)} refinement failed "
                            f"inside [{tj:.12g}, {t_next:.12g}] s."
                        ) from exc

                    if not np.isfinite(t_ev) or not np.all(np.isfinite(y_ev)):
                        raise ValueError(
                            f"Fixed-step {_event_label(i, ev)} refinement returned "
                            "a non-finite time or state."
                        )

                    crossings.append((i, float(t_ev), np.asarray(y_ev, dtype=np.float64), terminal))
                elif terminal and g0 == 0.0 and (
                    direction == 0.0
                    or (direction < 0.0 and g1 <= 0.0)
                    or (direction > 0.0 and g1 >= 0.0)
                ):
                    # SciPy parity: ``solve_ivp`` treats g(t_prev)=0 followed by a
                    # value on (or at) the triggering side as an event at t_prev
                    # itself. The exact-zero suppression in ``_event_crossed`` only
                    # protects *non-terminal* events (peri/apo) from re-detecting
                    # the same root; a terminal event cannot recur because
                    # integration stops at its first root, so a zero prior value
                    # here means the trajectory starts on the terminal boundary
                    # (surface starts, checkpoint restarts, exact rounding) and
                    # must be reported instead of silently skipped.
                    crossings.append((i, float(tj), y_curr.copy(), True))

                # Update previous value for next substep
                g_prev[i] = g1

            earliest_terminal: tuple[float, int, np.ndarray] | None = None  # (t_event, idx, y_event)
            for i, t_ev, y_ev, terminal in crossings:
                if terminal and ((earliest_terminal is None) or (t_ev < earliest_terminal[0])):
                    earliest_terminal = (t_ev, i, y_ev)

            # Phase 2: commit only roots at or before the earliest terminal
            # root (SciPy parity: ``solve_ivp`` discards roots later than the
            # terminal root, e.g. a periselene hit "after" the impact).
            for i, t_ev, y_ev, _terminal in crossings:
                if (earliest_terminal is None) or (t_ev <= earliest_terminal[0]):
                    t_events_acc[i].append(t_ev)
                    y_events_acc[i].append(y_ev)

            # Terminal event: stop at earliest terminal root in this substep
            if earliest_terminal is not None:
                t_ev, i_ev, y_ev = earliest_terminal
                ev_role = getattr(ev_list[i_ev], "_event_role", None)

                # A root exactly at the last committed output time (t0-boundary
                # terminal events) must not duplicate that grid point.
                if float(t_ev) > float(t_list[-1]):
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

            # Periodic stop-file polling. The substep just accepted must be
            # committed before breaking: otherwise the final physical state is
            # lost, t_stop falls back to the previous output-grid time, and the
            # checkpoint resumes from stale arrays.
            if (j % 50) == 0 and _stop_requested(stop_file):
                t_list.append(float(t_next))
                y_list.append(y_curr.copy())
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
                    logger.info(
                        "[HB] t=%7.2f h | alt=%9.3f km | min=%9.3f | max=%9.3f",
                        t_hr,
                        alt_now_km,
                        alt_min_km,
                        alt_max_km,
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
        # ``internal_step_count`` reports the steps actually executed so an
        # early-terminated run (impact / terminal event / stop file) is never
        # accounted as if the whole mission had been integrated. The planned
        # total is kept alongside for capacity/provenance comparisons.
        internal_step_count=int(executed_internal_steps),
        executed_internal_step_count=int(executed_internal_steps),
        planned_internal_step_count=int(planned_internal_steps),
        max_internal_steps=(None if max_internal_steps is None else int(max_internal_steps)),
        t_events=t_events,
        y_events=y_events,
    )

    if checkpoint_path:
        try:
            _atomic_save_npz(
                checkpoint_path,
                t=t_arr,
                y_row=y_arr,
                **(checkpoint_metadata or {}),
            )
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
    "_SYMPLECTIC_VOIDING_FLAGS",
    "symplectic_nonconservative_violations",
    "symplectic_nonconservative_gravity",
    "symplectic_discontinuous_gravity",
    "symplectic_impulsive_maneuver_violations",
    "symplectic_breaks_separability",
    "accel_form_velocity_dependence_violations",
    "_is_fixed_step_method",
    "_fixed_step_requires_6d",
    "_accel_stepper",
    "_build_fixed_stepper",
    "_evaluate_event",
    "_integrate_fixed_step",
    "_planned_fixed_step_count",
]
