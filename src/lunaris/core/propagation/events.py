"""Event construction, terrain-impact, and event-refinement helpers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from lunaris.common.math_utils import quat_rotate_np, quat_slerp_np
from lunaris.common.type_defs import PropagatorConfig
from lunaris.core.dynamics import DynamicsEngine
from lunaris.core.events import (
    make_aposelene_event,
    make_hybrid_impact_event,
    make_impact_event,
    make_periselene_event,
)
from lunaris.core.propagation.checkpoint import _stop_requested
from lunaris.core.propagation.time_grid import _get_ref_radius_and_mu


@dataclass(frozen=True, slots=True)
class EventOutcome:
    """Normalized terminal-event/impact/stop-file outcome."""

    impacted: bool
    t_impact_s: float | None
    y_impact: np.ndarray | None
    stopped_early: bool
    stop_reason: str | None
    t_stop_s: float | None

    def __post_init__(self) -> None:
        impacted = bool(self.impacted)
        stopped_early = bool(self.stopped_early)
        stop_reason = self.stop_reason
        t_stop_s = self.t_stop_s

        y_impact = None
        if self.y_impact is not None:
            y_impact = np.ascontiguousarray(self.y_impact, dtype=np.float64)

        if impacted:
            stopped_early = True
            stop_reason = "impact"
            if self.t_impact_s is not None and t_stop_s is None:
                t_stop_s = float(self.t_impact_s)

        if stop_reason:
            stopped_early = True

        object.__setattr__(self, "impacted", impacted)
        object.__setattr__(self, "t_impact_s", None if self.t_impact_s is None else float(self.t_impact_s))
        object.__setattr__(self, "y_impact", y_impact)
        object.__setattr__(self, "stopped_early", stopped_early)
        object.__setattr__(self, "stop_reason", stop_reason)
        object.__setattr__(self, "t_stop_s", None if t_stop_s is None else float(t_stop_s))


def _first_event_time(
    t_events: list[np.ndarray],
    index: int | None,
) -> float | None:
    if index is None or index >= len(t_events):
        return None
    try:
        t_ev = np.asarray(t_events[index], dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if t_ev.size == 0:
        return None
    t0 = float(t_ev[0])
    return t0 if np.isfinite(t0) else None


def _first_event_state(
    y_events: list[np.ndarray],
    index: int | None,
) -> np.ndarray | None:
    if index is None or index >= len(y_events):
        return None
    try:
        y_ev = np.asarray(y_events[index], dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if y_ev.size == 0:
        return None
    if y_ev.ndim == 1:
        y_ev = y_ev.reshape(1, -1)
    return np.asarray(y_ev[0], dtype=np.float64).reshape(-1)


def _earliest_terminal_event_time(
    events: list[Callable[[float, np.ndarray], float]] | None,
    t_events: list[np.ndarray],
) -> float | None:
    if not events:
        return None

    best: float | None = None
    for i, ev in enumerate(events):
        if not bool(getattr(ev, "terminal", False)):
            continue
        t0 = _first_event_time(t_events, i)
        if t0 is None:
            continue
        if best is None or t0 < best:
            best = t0
    return best


def event_outcome_from_solver_events(
    *,
    events: list[Callable[[float, np.ndarray], float]] | None,
    t_events: list[np.ndarray],
    y_events: list[np.ndarray],
    stopped_early: bool,
    stop_reason: str | None,
    stop_file: str | None,
    stop_event_in_scipy: bool,
) -> EventOutcome:
    """Normalize SciPy/fake-SciPy event payloads into a stable outcome object."""
    impacted = False
    t_imp: float | None = None
    y_imp: np.ndarray | None = None

    idx_impact = _find_event_index(events, "impact")
    if idx_impact is not None and _first_event_time(t_events, idx_impact) is not None:
        try:
            t_imp = _first_event_time(t_events, idx_impact)
            y_imp = _first_event_state(y_events, idx_impact)
            if t_imp is None or y_imp is None:
                raise ValueError("missing impact event time or state")
            impacted = True
        except (TypeError, ValueError) as exc:
            # Losing the impact flag silently would flip a result-affecting
            # boolean (R29b): a malformed solver event payload must be visible.
            import warnings
            warnings.warn(
                f"Impact-event extraction failed on malformed solver output: {exc}. "
                "The trajectory is reported as non-impacting.",
                RuntimeWarning,
                stacklevel=2,
            )
            t_imp = None
            y_imp = None

    if impacted:
        return EventOutcome(
            impacted=True,
            t_impact_s=t_imp,
            y_impact=y_imp,
            stopped_early=True,
            stop_reason="impact",
            t_stop_s=t_imp,
        )

    reason = stop_reason
    t_stop: float | None = None

    idx_stop = _find_event_index(events, "stop")
    if stop_file and bool(stop_event_in_scipy) and idx_stop is not None:
        t_stop_candidate = _first_event_time(t_events, idx_stop)
        if t_stop_candidate is not None:
            reason = "stop file"
            t_stop = t_stop_candidate

    if reason is None and bool(stopped_early):
        t_terminal = _earliest_terminal_event_time(events, t_events)
        if t_terminal is not None:
            reason = "event"
            t_stop = t_terminal

    if t_stop is None and reason == "event":
        t_stop = _earliest_terminal_event_time(events, t_events)

    return EventOutcome(
        impacted=False,
        t_impact_s=None,
        y_impact=None,
        stopped_early=bool(stopped_early),
        stop_reason=reason,
        t_stop_s=t_stop,
    )


def _build_r_i_to_bf_from_rot_table(
    dynamics: DynamicsEngine,
) -> Callable[[float, np.ndarray], np.ndarray] | None:
    """Build an inertial→body-fixed position mapper from an ephemeris quaternion table.

    Expected rotation table format:
      - q_i2f(t): quaternion (w, x, y, z) mapping inertial → fixed/body frame

    Supported provider:
      - eph.tables -> dt_s, q_i2f_tab

    Returns None if no valid rotation table is available.

    Why single-row tables are supported
    -----------------------------------
    Short runs can legitimately produce an ephemeris attitude table with only
    one quaternion sample. In that case the rotation is still perfectly usable:
    it simply behaves as a constant inertial→fixed transform over the requested
    interval. Rejecting `N=1` tables would unnecessarily disable:

    - terrain-aware telemetry, and
    - hybrid impact events

    for otherwise valid short-duration propagations.
    """
    eph = getattr(dynamics, "ephem", None)
    if eph is None:
        return None

    def _try_get_dt_and_qtab() -> tuple[float, np.ndarray] | None:
        # R29b-justified: capability probe over heterogeneous ephemeris shapes;
        # returning None only disables the *optional* terrain/telemetry rotation
        # helper, never the propagation itself.
        # Tables-style (strict)
        if hasattr(eph, "tables"):

            try:
                tab = eph.tables
                dt = float(getattr(tab, "dt_s", 0.0))
                q = getattr(tab, "q_i2f_tab", None)
            except (AttributeError, TypeError, ValueError):
                dt = 0.0
                q = None
            if dt > 0.0 and q is not None:
                try:
                    qtab = np.asarray(q, dtype=np.float64)
                    return dt, qtab
                except (TypeError, ValueError):
                    pass

        return None

    got = _try_get_dt_and_qtab()
    if got is None:
        return None

    dt, q_tab = got
    if dt <= 0.0 or q_tab.ndim != 2 or q_tab.shape[1] != 4 or q_tab.shape[0] < 1:
        return None

    q_tab = np.ascontiguousarray(q_tab, dtype=np.float64)
    n = int(q_tab.shape[0])
    dt_f = float(dt)

    if n == 1:
        q_const = np.asarray(q_tab[0], dtype=np.float64).reshape(4)

        def r_i_to_bf_constant(_t: float, r_i: np.ndarray) -> np.ndarray:
            """
            Apply a constant inertial→fixed quaternion.

            The timestamp is intentionally ignored because a single-table sample
            means the caller asked for a degenerate/constant frame history.
            """

            r = np.asarray(r_i, dtype=np.float64).reshape(3)
            return quat_rotate_np(q_const, r)

        return r_i_to_bf_constant

    def r_i_to_bf(t: float, r_i: np.ndarray) -> np.ndarray:
        # t assumed seconds aligned with ephemeris start
        u = float(t) / dt_f
        if u <= 0.0:
            q = q_tab[0]
        elif u >= float(n - 1):
            q = q_tab[n - 1]
        else:
            i = int(u)  # floor for u>=0
            a = float(u - i)
            q = quat_slerp_np(q_tab[i], q_tab[i + 1], a)

        r = np.asarray(r_i, dtype=np.float64).reshape(3)
        return quat_rotate_np(q, r)

    return r_i_to_bf

def _wrap_event_first6(ev: Callable[[float, np.ndarray], float]) -> Callable[[float, np.ndarray], float]:
    """Wrap an event function so it ignores any augmented state beyond the first 6 elements."""
    def g(t: float, y: np.ndarray) -> float:
        return float(ev(t, y[:6]))

    # SciPy event attributes
    g.terminal = bool(getattr(ev, "terminal", False))     # type: ignore[attr-defined]
    g.direction = float(getattr(ev, "direction", 0.0))    # type: ignore[attr-defined]

    # Help debugging / introspection
    g.__name__ = getattr(ev, "__name__", "event_first6")
    g.__doc__ = getattr(ev, "__doc__", None)
    return g

def _get_event_cfg(cfg: Any) -> Any:
    return getattr(cfg, "events", None)

# R29b-justified (getter family below): config objects are polymorphic
# (dataclass / SimpleNamespace / UI adapters); a non-coercible attribute means
# "not configured" and falls back to the documented default.
def _get_cfg_bool(cfg: Any, name: str, default: bool) -> bool:
    evc = _get_event_cfg(cfg)
    if evc is not None and hasattr(evc, name):
        try:
            return bool(getattr(evc, name))
        except (TypeError, ValueError):
            pass
    if hasattr(cfg, name):
        try:
            return bool(getattr(cfg, name))
        except (TypeError, ValueError):
            pass
    return bool(default)

def _get_cfg_float(cfg: Any, name: str, default: float) -> float:
    evc = _get_event_cfg(cfg)
    if evc is not None and hasattr(evc, name):
        try:
            return float(getattr(evc, name))
        except (TypeError, ValueError):
            pass
    if hasattr(cfg, name):
        try:
            return float(getattr(cfg, name))
        except (TypeError, ValueError):
            pass
    return float(default)

def _get_detect_impact(cfg: Any) -> bool:
    return _get_cfg_bool(cfg, "detect_impact", True)

def _get_impact_alt_km(cfg: Any) -> float:
    return _get_cfg_float(cfg, "impact_alt_km", 0.0)

def _get_enable_peri_apo_events(cfg: Any) -> bool:
    return _get_cfg_bool(cfg, "enable_peri_apo_events", True)

def _find_event_index(events: list[Callable[[float, np.ndarray], float]] | None, role: str) -> int | None:
    if not events:
        return None
    for i, ev in enumerate(events):
        if getattr(ev, "_event_role", None) == role:
            return i
    return None

def _terminal_event_endpoint(
    sol: Any,
    events: list[Callable[[float, np.ndarray], float]] | None,
    *,
    state_size: int,
) -> tuple[float, np.ndarray] | None:
    """Return the earliest terminal event endpoint recorded by a SciPy solution."""
    if not events or getattr(sol, "t_events", None) is None:
        return None

    y_events_raw = getattr(sol, "y_events", None)
    if y_events_raw is None:
        return None

    best: tuple[float, np.ndarray] | None = None
    for i, ev in enumerate(events):
        if not bool(getattr(ev, "terminal", False)):
            continue
        try:
            t_ev = np.asarray(sol.t_events[i], dtype=np.float64).reshape(-1)
        except (TypeError, ValueError):
            # R29b-justified: a malformed per-event array from the solver means
            # "this event did not fire usably"; other terminal events are still
            # considered, and no-event runs keep their full trajectory.
            continue
        if t_ev.size == 0 or i >= len(y_events_raw):
            continue

        try:
            y_ev = np.asarray(y_events_raw[i], dtype=np.float64)
        except (TypeError, ValueError):
            # R29b-justified: same rationale as t_events above.
            continue
        if y_ev.size == 0:
            continue
        if y_ev.ndim == 1:
            y_ev = y_ev.reshape(1, -1)

        t0 = float(t_ev[0])
        y0 = np.asarray(y_ev[0], dtype=np.float64).reshape(-1)
        if y0.size != state_size or not np.isfinite(t0) or not np.all(np.isfinite(y0)):
            continue
        if best is None or t0 < best[0]:
            best = (t0, y0)

    return best

def build_events(
    dynamics: DynamicsEngine,
    cfg: PropagatorConfig,
    *,
    topo_grid: Any = None,
    add_stop_event: bool = True,
) -> list[Callable[[float, np.ndarray], float]]:
    """Build SciPy-compatible event callables based on PropagatorConfig (+ optional topo grid)."""
    R_ref, _mu = _get_ref_radius_and_mu(dynamics)

    detect_impact = _get_detect_impact(cfg)
    impact_alt_km = _get_impact_alt_km(cfg)
    impact_alt_m = float(impact_alt_km) * 1000.0

    events: list[Callable[[float, np.ndarray], float]] = []

    # Impact event (terminal)
    if detect_impact:
        if topo_grid is not None:
            r_i_to_bf = _build_r_i_to_bf_from_rot_table(dynamics)
            if r_i_to_bf is not None:
                ev_imp = make_hybrid_impact_event(
                    R_ref_m=float(R_ref),
                    impact_alt_m=float(impact_alt_m),
                    topo=topo_grid,
                    r_i_to_bf=r_i_to_bf,
                    switch_alt_m=float(getattr(cfg, "hybrid_switch_alt_m", 11_000.0)),
                    kind=str(getattr(cfg, "hybrid_kind", "bilinear")),
                    terminal=True,
                )
            else:
                ev_imp = make_impact_event(R_ref_m=float(R_ref), impact_alt_m=float(impact_alt_m), terminal=True)
        else:
            ev_imp = make_impact_event(R_ref_m=float(R_ref), impact_alt_m=float(impact_alt_m), terminal=True)

        ev_imp6 = _wrap_event_first6(ev_imp)
        ev_imp6._event_role = "impact"  # type: ignore[attr-defined]
        events.append(ev_imp6)

    # Peri/Apo events (non-terminal)
    if _get_enable_peri_apo_events(cfg):
        ev_peri = _wrap_event_first6(make_periselene_event(terminal=False))
        ev_apo = _wrap_event_first6(make_aposelene_event(terminal=False))
        ev_peri._event_role = "peri"  # type: ignore[attr-defined]
        ev_apo._event_role = "apo"  # type: ignore[attr-defined]
        events.append(ev_peri)
        events.append(ev_apo)

    # Stop file event (optional) – use +/-1 for sign change
    stop_file = getattr(cfg, "stop_file", None)
    stop_in_scipy = bool(getattr(cfg, "stop_event_in_scipy", False))
    if add_stop_event and stop_file and stop_in_scipy:
        def _stop_ev(t: float, y: np.ndarray) -> float:
            # Use +/-1 so a change in stop-file state produces a sign change
            return -1.0 if _stop_requested(str(stop_file)) else 1.0
        _stop_ev.terminal = True           # type: ignore[attr-defined]
        _stop_ev.direction = 0.0           # type: ignore[attr-defined]
        _stop_ev._event_role = "stop"  # type: ignore[attr-defined]
        events.append(_stop_ev)

    return events

def _event_crossed(g0: float, g1: float, direction: float = 0.0) -> bool:
    """Return True if an event root is bracketed in [g0,g1] given direction."""
    if not (np.isfinite(g0) and np.isfinite(g1)):
        return False
    # Treat exact zeros robustly
    if g0 == 0.0:
        # If we start exactly at root, do not trigger unless moving away then back.
        # Here we simply ignore the start-point root to avoid duplicate detections.
        return False
    if direction > 0.0:
        return (g0 < 0.0) and (g1 >= 0.0)
    if direction < 0.0:
        return (g0 > 0.0) and (g1 <= 0.0)
    return (g0 > 0.0) != (g1 > 0.0) or (g1 == 0.0)

def _refine_event_time_bisect(
    *,
    step: Callable[[float, np.ndarray, float], np.ndarray],
    ev: Callable[[float, np.ndarray], float],
    t0: float,
    y0: np.ndarray,
    h: float,
    g0: float,
    g1: float,
    max_iter: int = 30,
    tol_s: float = 1e-6,
) -> tuple[float, np.ndarray]:
    """Refine a single event root inside a step using bisection + final linear-in-g correction.

    We re-integrate short substeps from (t0,y0) to candidate times. This does NOT modify the
    main integrator's state advancement (y_next is still computed once in the main loop), so
    symplectic stepping remains unchanged; we only improve the reported event time/state.
    """
    # Ensure a valid bracket
    if not (np.isfinite(g0) and np.isfinite(g1)):
        t_lin = t0
        return t_lin, y0

    a = 0.0
    b = 1.0
    ga = float(g0)
    gb = float(g1)

    # Cache endpoints' states when needed
    yb = None

    # Early exit if already extremely close
    if abs(h) <= tol_s:
        yb = step(t0, y0, h)
        return t0 + h, np.asarray(yb, dtype=np.float64)

    # Bisection iterations
    for _ in range(max_iter):
        if (b - a) * abs(h) <= tol_s:
            break
        m = 0.5 * (a + b)
        hm = m * h
        ym = step(t0, y0, hm)
        gm = float(ev(t0 + hm, ym))

        # Narrow the bracket by sign
        if (ga > 0.0) == (gm > 0.0):
            a = m
            ga = gm
        else:
            b = m
            gb = gm
            yb = ym

    # If yb not computed (possible if we always moved left), compute it
    if yb is None:
        yb = step(t0, y0, b * h)
        gb = float(ev(t0 + b * h, yb))

    # Final linear-in-g correction inside last bracket (a,b)
    denom = (gb - ga)
    if denom != 0.0 and np.isfinite(denom):
        tau = a + (-ga) * (b - a) / denom
        tau = float(min(1.0, max(0.0, tau)))
    else:
        tau = b

    ht = tau * h
    yt = step(t0, y0, ht)
    return float(t0 + ht), np.asarray(yt, dtype=np.float64)


__all__ = [
    "EventOutcome",
    "_build_r_i_to_bf_from_rot_table",
    "_wrap_event_first6",
    "_get_event_cfg",
    "_get_cfg_bool",
    "_get_cfg_float",
    "_get_detect_impact",
    "_get_impact_alt_km",
    "_get_enable_peri_apo_events",
    "_find_event_index",
    "_terminal_event_endpoint",
    "event_outcome_from_solver_events",
    "build_events",
    "_event_crossed",
    "_refine_event_time_bisect",
]
