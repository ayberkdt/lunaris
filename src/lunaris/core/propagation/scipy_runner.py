"""SciPy (``solve_ivp``) propagation orchestration.

Owns the adaptive-integration branch of :func:`lunaris.core.propagation.propagate`:
the normal single-span solve and the chunked solve with per-chunk checkpointing,
terminal-event endpoint preservation, event accumulation and stop-file handling.
Extracted from ``propagator.py`` (seam-cleanup item 5, P3 slice); behavior is
preserved exactly — no numerical change.
"""

from __future__ import annotations

import logging
import math
import warnings
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import numpy as np
from scipy.integrate import solve_ivp

from lunaris.common.type_defs import PropagationResult, PropagatorConfig
from lunaris.core.dynamics.contracts import RestartPlan
from lunaris.core.propagation.checkpoint import _atomic_save_npz, _stop_requested
from lunaris.core.propagation.events import (
    _terminal_event_endpoint,
    event_outcome_from_solver_events,
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
    vec: np.ndarray = np.full(n, atol_scalar, dtype=np.float64)
    if n >= 6:
        if atol_pos is not None:
            vec[0:3] = float(atol_pos)
        if atol_vel is not None:
            vec[3:6] = float(atol_vel)
    return vec


# A non-terminal event whose start value is within this many times its measured
# rounding sensitivity of zero starts on its surface (see `_rounding_band`); the
# factor covers the few float64 operations between a state component and the
# event value with a wide margin.
_SURFACE_BAND_FACTOR = 64.0
# Size of the probe perturbation of each state component, in units of
# eps * |y_i| (a few ulps of the component).
_SURFACE_PROBE_ULPS = 4.0


def _rounding_band(event: Callable[[float, np.ndarray], float], t: float, y: np.ndarray, g0: float) -> float:
    """Rounding level of ``event`` at ``(t, y)``: how far a few-ulp state change moves it.

    Each component of ``y`` is moved by ``_SURFACE_PROBE_ULPS`` ulps in turn and
    the largest change of the event value, times ``_SURFACE_BAND_FACTOR``, is
    returned; NaN when a probe is not finite.
    """
    state = np.asarray(y, dtype=np.float64)
    eps = float(np.finfo(np.float64).eps)
    largest = 0.0
    for index in range(state.size):
        delta = _SURFACE_PROBE_ULPS * eps * abs(float(state[index]))
        if delta == 0.0:
            continue
        probe = state.copy()
        probe[index] += delta
        change = abs(float(event(t, probe)) - g0)
        if not math.isfinite(change):
            return math.nan
        largest = max(largest, change)
    return _SURFACE_BAND_FACTOR * largest


# A chunk starts from the previous chunk's end state, read off its interpolant,
# and a segment after a restart root starts from the exact solve's state while
# the root came from the first solve's interpolant (2.4e-6 m apart on Sidera's
# 400 km Venus orbit at 60 s steps). An event whose root lies on the boundary is
# then a rounding-level fraction of its value over the last step away from
# zero, on either side. Within this fraction of that value the side is taken
# from the step end before the boundary. A distinct crossing that close to the
# boundary (a millionth of a step) is not resolved.
_ROOT_REACH = 1.0e-6


def _step_consistent_events(
    events: list[Callable[[float, np.ndarray], float]],
    seeds: list[float | None] | None = None,
) -> list[Callable[[float, np.ndarray], float]]:
    """Wrap events so a root search sees the signs that SciPy detected.

    ``solve_ivp`` decides that an event crossed from its values at the step
    ends, computed on the solver states, and then calls ``brentq`` on the event
    of the step interpolant between the same two times. The interpolant's end
    value differs from the solver state by rounding, so an event whose value is
    at rounding level there (``r . v`` on a circular orbit, ~1e-16 of
    ``|r| |v|``) can change sign between the two, and ``brentq`` raises "f(a)
    and f(b) must have different signs" from inside the solver (reproduced for
    RK45, Radau and LSODA on a circular lunar orbit). Each wrapper remembers the
    values at the two latest step ends (evaluations at a time past every
    earlier one) and returns that value again at the same time when a fresh
    evaluation has the other sign. A fresh value of the same sign is returned
    unchanged, so a run without the inconsistency is bit-for-bit what it was.

    A non-terminal event that starts on its surface (first value within
    `_rounding_band` of zero) returns NaN, which SciPy never reads as a
    crossing, until the first step end at which it leaves that band; its
    rounding-level sign flips over SciPy's tiny first steps would otherwise
    give repeated roots at the start. Terminal events are not touched there: a
    start on the impact sphere moving inward stays an impact at the start.
    Ported from Sidera (commit 7e3deea4).

    A step end where the event is exactly zero (a root that falls on it) is
    reported as the smallest value past zero, on the side away from the
    previous step end: the step that reaches the root reports it, and the next
    step, which starts there, does not report it again.

    ``seeds`` (one signed value or ``None`` per event) is for a span that
    continues another (a chunk boundary, or a restart root of
    :func:`_solve_with_restarts`, from :func:`_side_after_root`) rather than
    the start of the run: an event, terminal or not, whose first value is
    within the band, or within `_ROOT_REACH` of the seed's magnitude, reports
    that value on the seed's side, the side the previous span ended on. A
    root on the boundary is then reported once: by the previous span if it
    reached it, else in the first step of this one. Each wrapper exposes the
    step ends it remembers as ``step_ends``.
    """
    tiny = float(np.finfo(np.float64).tiny)

    def consistent(
        event: Callable[[float, np.ndarray], float], seed: float | None,
    ) -> Callable[[float, np.ndarray], float]:
        step_ends: list[tuple[float, float]] = []
        terminal = bool(getattr(event, "terminal", False))
        started = False
        surface_band: float | None = None

        def g(t: float, y: np.ndarray) -> float:
            nonlocal started, surface_band
            t_key = float(t)
            value = event(t, y)
            if not started and (seed is not None or not terminal):
                started = True
                g0 = float(value)
                if math.isfinite(g0):
                    band = _rounding_band(event, t, y, g0)
                    if seed is not None:
                        reach = max(band if math.isfinite(band) else 0.0, _ROOT_REACH * abs(seed))
                        if abs(g0) <= reach:
                            value = math.copysign(max(band if math.isfinite(band) else 0.0, tiny), seed)
                    elif math.isfinite(band) and abs(g0) <= band:
                        surface_band = band
            frontier = not step_ends or t_key > step_ends[-1][0]
            if surface_band is not None:
                if not frontier:
                    return math.nan
                if abs(float(value)) <= surface_band:
                    step_ends.append((t_key, math.nan))
                    del step_ends[:-2]
                    return math.nan
                surface_band = None
            for t_end, end_value in step_ends:
                if t_end == t_key:
                    fresh = float(value)
                    same_side = fresh == end_value or (fresh > 0.0 and end_value > 0.0) or (
                        fresh < 0.0 and end_value < 0.0
                    )
                    return value if same_side else end_value
            if frontier:
                previous = step_ends[-1][1] if step_ends else math.nan
                if float(value) == 0.0 and math.isfinite(previous) and previous != 0.0:
                    # A root exactly on a step end belongs to the step that
                    # reaches it: SciPy reads 0 as a crossing at the end of
                    # that step and again at the start of the next.
                    value = -math.copysign(tiny, previous)
                step_ends.append((t_key, float(value)))
                del step_ends[:-2]
            return value

        g.terminal = getattr(event, "terminal", False)  # type: ignore[attr-defined]
        g.direction = getattr(event, "direction", 0.0)  # type: ignore[attr-defined]
        g.step_ends = step_ends  # type: ignore[attr-defined]
        return g

    signs = list(seeds) if seeds is not None else [None] * len(events)
    return [consistent(event, seed) for event, seed in zip(events, signs, strict=True)]


def _side_after_root(
    wrapper: Callable[[float, np.ndarray], float], times: np.ndarray, t_root: float, forward: bool,
) -> float | None:
    """The side an event is on just past a restart root, or ``None`` when unknown.

    The sign is the side (+/-); the magnitude is the event's value at the
    step end before the root, the scale `_ROOT_REACH` is taken of.

    The value at the root is at rounding level for an event whose surface the
    root lies on, and the step end after it lies past roots SciPy did not
    report (it reports none after a terminal root), so the side comes from the
    step end before the root, flipped once per root reported after it.
    Ported from Sidera (fc6bae19).
    """
    before = [
        (t, v) for t, v in getattr(wrapper, "step_ends", [])
        if (t < t_root if forward else t > t_root) and math.isfinite(v) and v != 0.0
    ]
    if not before:
        return None
    t_before, value = before[-1]
    crossings = int(np.count_nonzero((times > t_before) if forward else (times < t_before)))
    side = abs(value) if value > 0.0 else -abs(value)
    return side if crossings % 2 == 0 else -side


def _armed_restart_event(
    event: Callable[[float, np.ndarray], float], t_start: float, *, on_root: bool = True,
) -> Callable[[float, np.ndarray], float]:
    """A terminal copy of ``event`` that cannot fire on the root it starts from.

    A segment restarted at a root begins with the event at rounding level, and
    its sign there is noise. The copy returns NaN at the segment's first
    evaluation (SciPy then sees no crossing over the first step, which is
    the one leaving that root) and the event's own value everywhere else.
    For the first segment of a span (``on_root=False``: the run's or a
    chunk's start, not a root) the start value is suppressed only when it is
    within `_rounding_band` of zero, so a contact inside the first step is
    still found. Ported from Sidera (fc6bae19).
    """
    first = True

    def g(t: float, y: np.ndarray) -> float:
        nonlocal first
        if first:
            first = False
            if float(t) == float(t_start):
                if on_root:
                    return math.nan
                value = float(event(t, y))
                band = _rounding_band(event, t, y, value)
                return math.nan if (value == 0.0 or (math.isfinite(band) and abs(value) <= band)) else value
        return float(event(t, y))

    g.terminal = True  # type: ignore[attr-defined]
    g.direction = 0.0  # type: ignore[attr-defined]
    return g


def _solve_with_restarts(
    solve_once: Callable[..., object],
    t_start: float,
    t_end: float,
    y_start: np.ndarray,
    t_eval_span: np.ndarray,
    user_events: list[Callable[[float, np.ndarray], float]],
    restart_events: list[Callable[[float, np.ndarray], float]],
    segment_max_step: Callable[[float, np.ndarray], float | None] | None = None,
    seeds: list[float | None] | None = None,
) -> SimpleNamespace:
    """Integrate ``[t_start, t_end]`` in segments that end at restart-event roots.

    An embedded Runge-Kutta pair estimates its error from stages it shares
    with the solution, so a kink of the right-hand side inside a step (a
    shadow boundary, where the radiation force stops being smooth) corrupts
    both and the estimate stays small: the step is accepted with an error the
    tolerance never saw. Stopping at each root and restarting there puts every
    kink on a step boundary.

    Each segment is solved once, with the user events and the restart events,
    up to the next root; SciPy cuts the step that holds the root there. The
    state it reports at the root comes from that step's interpolant, whose
    error the step control never sees, so the cut step alone is solved again,
    from its start state (the solver's own, kept by an event that never
    fires) to the root, with the same first step; the next segment starts
    from that step end. The segment's samples and user events up to the cut
    step come from the first solve, the samples inside it from the second.
    The user events therefore see every time once and in order, as in a
    single solve: an event that keeps state of its own
    (`make_max_eclipse_duration_event`) or reads state outside the trajectory
    (the stop file) behaves as without restarts. ``user_events`` are wrapped
    afresh for each segment (their wrappers keep step-end state). A user
    event whose root is also a restart root sits at rounding level at the end
    of one segment and at the start of the next; each segment after a root
    therefore starts its user events on the side they were on past that root
    (:func:`_side_after_root`), so the crossing is reported once.

    ``solve_once(t0, t1, y0, t_eval, events, step_cap, *, first_step=None)``
    is one ``solve_ivp`` call. Returns the ``solve_ivp`` result fields the
    callers read (``t``, ``y``, ``t_events``/``y_events`` of the user events,
    ``status``, ``success``, ``message``, ``nfev``) plus ``restarts``. A user
    terminal event ends the span as in a single solve. ``seeds`` are the user
    events' sides at ``t_start`` when the span continues another (a chunk
    boundary), as for a restart root; the result's ``user_wrappers`` are the
    last segment's, for the next span's seeds.

    Ported from Sidera (2ddf461e, with the review fixes of fc6bae19), without
    the dense-output stitching, which the Lunaris SciPy path never asks for.
    """
    n_user = len(user_events)
    grid = np.asarray(t_eval_span, dtype=np.float64)
    forward = float(t_end) >= float(t_start)

    def ahead(a: float, b: float) -> bool:
        """``a`` lies past ``b`` in the integration direction."""
        return a > b if forward else a < b

    def between(lo: float, hi: float) -> np.ndarray:
        """Grid times from ``lo`` to ``hi`` (both included), ``lo`` first in the integration direction."""
        return grid[(grid >= min(lo, hi)) & (grid <= max(lo, hi))]

    t_cur = float(t_start)
    y_cur = np.asarray(y_start, dtype=np.float64)
    t_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    te_parts: list[list[np.ndarray]] = [[] for _ in range(n_user)]
    ye_parts: list[list[np.ndarray]] = [[] for _ in range(n_user)]
    nfev = 0
    status = 0
    message = "The solver successfully reached the end of the integration interval."
    restarts = 0

    def keep(sol: Any, *, until: float | None = None, after: float | None = None, events: bool = True) -> None:
        """Append ``sol``'s samples (past ``after``, not past ``until``) and user events."""
        seg_t = np.asarray(sol.t, dtype=np.float64)
        seg_y: np.ndarray = np.asarray(sol.y, dtype=np.float64).reshape(y_cur.size, -1)
        mask = np.ones(seg_t.size, dtype=bool)
        if until is not None:
            mask &= np.array([not ahead(float(value), until) for value in seg_t], dtype=bool)
        if after is not None:
            mask &= np.array([ahead(float(value), after) for value in seg_t], dtype=bool)
        seg_t, seg_y = seg_t[mask], seg_y[:, mask]
        if t_parts and seg_t.size:
            fresh = np.array([ahead(float(value), float(t_parts[-1][-1])) for value in seg_t], dtype=bool)
            seg_t, seg_y = seg_t[fresh], seg_y[:, fresh]
        if seg_t.size:
            t_parts.append(seg_t)
            y_parts.append(seg_y)
        if events:
            t_events = list(getattr(sol, "t_events", None) or [])
            y_events = list(getattr(sol, "y_events", None) or [])
            for i in range(n_user):
                te_parts[i].append(np.asarray(t_events[i], dtype=np.float64).reshape(-1))
                ye_parts[i].append(np.asarray(y_events[i], dtype=np.float64).reshape(-1, y_cur.size))

    def step_end_recorder() -> Callable[[float, np.ndarray], float]:
        """An event that never fires and keeps the last two step ends SciPy evaluated (exact solver states)."""
        ends: list[tuple[float, np.ndarray]] = []

        def g(t: float, y: np.ndarray) -> float:
            if not ends or ahead(float(t), ends[-1][0]):
                ends.append((float(t), np.array(y, dtype=np.float64, copy=True)))
                del ends[:-2]
            return 1.0

        g.terminal = False  # type: ignore[attr-defined]
        g.direction = 0.0  # type: ignore[attr-defined]
        g.ends = ends  # type: ignore[attr-defined]
        return g

    def user_terminal(sol: Any) -> bool:
        t_events = list(getattr(sol, "t_events", None) or [])
        return any(
            bool(getattr(user_events[i], "terminal", False)) and np.asarray(t_events[i]).size > 0
            for i in range(n_user)
        )

    user_wrappers: list[Callable[[float, np.ndarray], float]] = []
    while True:
        cap = segment_max_step(t_cur, y_cur) if segment_max_step is not None else None
        segment_eval = between(t_cur, t_end)
        restart_copies = [_armed_restart_event(event, t_cur, on_root=restarts > 0) for event in restart_events]
        user_wrappers = _step_consistent_events(user_events, seeds)
        recorder = step_end_recorder()
        locate: Any = solve_once(t_cur, t_end, y_cur, segment_eval, user_wrappers + restart_copies + [recorder], cap)
        nfev += int(getattr(locate, "nfev", 0))
        t_events = list(getattr(locate, "t_events", None) or [])
        roots = [
            float(np.asarray(t_events[n_user + k])[-1])
            for k in range(len(restart_events))
            if np.asarray(t_events[n_user + k]).size
        ]
        # The root met first in the integration direction.
        first_root = (min(roots) if forward else max(roots)) if roots else None
        located = (
            bool(getattr(locate, "success", True))
            and int(getattr(locate, "status", 0)) == 1
            and not user_terminal(locate)
            and bool(roots)
        )
        if not located or first_root is None or not ahead(float(t_end), first_root):
            keep(locate)
            if not bool(getattr(locate, "success", True)):
                status = int(getattr(locate, "status", -1))
                message = str(getattr(locate, "message", "integration failed"))
            elif int(getattr(locate, "status", 0)) == 1 and not located:
                status = 1
                message = str(getattr(locate, "message", "A termination event occurred."))
            break
        t_root = float(first_root)
        # SciPy cut the step holding the root at the root: its start is the
        # last step end at or before the root (brentq can return the step's own
        # start), whose solver state the recorder kept, and the cut step alone
        # is solved again from there.
        t_last, y_last = [end for end in recorder.ends if not ahead(end[0], t_root)][-1]  # type: ignore[attr-defined]
        exact_eval = np.unique(np.concatenate((between(t_last, t_root), [t_root])))
        if not forward:
            exact_eval = exact_eval[::-1]
        if abs(t_root - t_last) <= 64.0 * float(np.spacing(max(abs(t_root), 1.0))):
            # No step to solve: the step end's own state, or within a few ulp
            # of it the first solve's state at the root, stands.
            if t_root == t_last:
                y_root = np.asarray(y_last, dtype=np.float64)
            else:
                k_root = next(k for k in range(len(restart_events)) if np.asarray(t_events[n_user + k]).size
                              and float(np.asarray(t_events[n_user + k])[-1]) == t_root)
                y_root = np.asarray(locate.y_events[n_user + k_root], dtype=np.float64)[-1]
            exact: Any = SimpleNamespace(t=np.array([t_root]), y=y_root.reshape(-1, 1), success=True, nfev=0)
        else:
            exact = solve_once(t_last, t_root, y_last, exact_eval, [], cap, first_step=abs(t_root - t_last))
        nfev += int(getattr(exact, "nfev", 0))
        keep(locate, until=t_last)
        if not bool(getattr(exact, "success", True)):
            keep(exact, after=t_last, events=False)
            status = int(getattr(exact, "status", -1))
            message = str(getattr(exact, "message", "integration failed"))
            break
        # The root sample is a restart point, not output, unless the grid asked for it.
        root_on_grid = bool(np.isin([t_root], grid)[0])
        keep(exact, after=t_last, until=t_root if root_on_grid else float(np.nextafter(t_root, t_last)), events=False)
        seeds = [
            _side_after_root(user_wrappers[i], np.asarray(t_events[i], dtype=np.float64).reshape(-1), t_root, forward)
            for i in range(n_user)
        ]
        t_cur = t_root
        y_cur = np.asarray(exact.y, dtype=np.float64).reshape(y_cur.size, -1)[:, -1].copy()
        restarts += 1

    t_out = np.concatenate(t_parts) if t_parts else np.zeros(0, dtype=np.float64)
    y_out = np.concatenate(y_parts, axis=1) if y_parts else np.zeros((y_cur.size, 0), dtype=np.float64)
    return SimpleNamespace(
        t=t_out,
        y=y_out,
        t_events=[np.concatenate(parts) if parts else np.zeros(0) for parts in te_parts],
        y_events=[np.concatenate(parts, axis=0) if parts else np.zeros((0, y_cur.size)) for parts in ye_parts],
        status=status,
        success=status >= 0,
        message=message,
        nfev=nfev,
        restarts=restarts,
        user_wrappers=user_wrappers,
    )


def _grid_tolerance_s(grid: np.ndarray, chunk_s: float) -> float:
    """Absolute time tolerance for 'on the output grid' [s].

    A small fraction of the finer of the grid spacing and the chunk length:
    far above the rounding a chain of ``t += chunk_s`` accumulates, far below
    any real sample spacing, and independent of the absolute epoch.
    """
    spacing = float(np.min(np.diff(grid))) if grid.size > 1 else float(chunk_s)
    return 1.0e-7 * max(min(spacing, float(chunk_s)), 1.0e-9)


def _nearest_grid_distance(times: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Distance of each time to the nearest point of the sorted output grid [s]."""
    if grid.size == 0:
        return np.full(np.shape(times), np.inf)
    right = np.clip(np.searchsorted(grid, times), 0, grid.size - 1)
    left = np.clip(right - 1, 0, grid.size - 1)
    return np.minimum(np.abs(times - grid[left]), np.abs(times - grid[right]))


def run_scipy_propagation(
    *,
    rhs: Callable[[float, np.ndarray], np.ndarray],
    t_eval: np.ndarray,
    y0: np.ndarray,
    t0: float,
    tf: float,
    method: str,
    max_step_s: float,
    chunk_s: float | None,
    cfg: PropagatorConfig,
    events: list[Callable[[float, np.ndarray], float]],
    stop_file: str | None,
    checkpoint_path: str | None,
    checkpoint_metadata: dict[str, object],
    output_dt_s: float,
    verbose: bool,
    logger: logging.Logger,
    restart_plan: RestartPlan | None = None,
) -> PropagationResult:
    """Run the SciPy adaptive branch and assemble a propagation result.

    The returned result carries an empty ``diagnostics`` dict; the caller
    (:func:`propagate`) attaches the full diagnostics afterwards.

    ``restart_plan`` holds the dynamics' non-smooth points (functions whose
    zero is a kink of the right-hand side, such as a shadow boundary). Each
    span is integrated up to the next such root and restarted there, so no
    step straddles a kink; see :func:`_solve_with_restarts`. Without events
    the solve is exactly the single ``solve_ivp`` call it always was.
    """
    if solve_ivp is None:
        raise ImportError("SciPy is required for adaptive integration (solve_ivp not available).")

    y0_arr = np.asarray(y0, dtype=np.float64)
    checkpoint_meta = checkpoint_metadata
    max_step = float(max_step_s)
    dt_out = float(output_dt_s)

    # Per-component (vector) atol when configured: position and velocity differ
    # by ~3 orders of magnitude, so a single scalar over-tightens one of them.
    atol_arg = _resolve_atol(cfg, int(y0_arr.size))

    if verbose:
        if isinstance(atol_arg, np.ndarray):
            logger.info(
                "[PROP] solve_ivp method=%s | dt_out=%gs | max_step=%.6fs | atol=vector(pos=%s, "
                "vel=%s)",
                method,
                dt_out,
                max_step,
                getattr(cfg, 'atol_pos', None),
                getattr(cfg, 'atol_vel', None),
            )
        else:
            logger.info(
                "[PROP] solve_ivp method=%s | dt_out=%gs | max_step=%.6fs",
                method, dt_out, max_step,
            )

    span_wrappers: list[Callable[[float, np.ndarray], float]] = []

    def _solve_once(
        t_start: float,
        t_end: float,
        y_start: np.ndarray,
        t_eval_span: np.ndarray,
        span_events: list[Callable[[float, np.ndarray], float]] | None,
        step_cap: float | None = None,
        *,
        first_step: float | None = None,
    ):
        # ``first_step`` is passed only when set, so a run without restarts is
        # the very ``solve_ivp`` call it was before them.
        extra: dict[str, float] = {} if first_step is None else {"first_step": float(first_step)}
        return solve_ivp(
            fun=rhs,
            t_span=(float(t_start), float(t_end)),
            y0=np.asarray(y_start, dtype=np.float64),
            method=method,
            t_eval=np.asarray(t_eval_span, dtype=np.float64),
            rtol=float(cfg.rtol),
            atol=atol_arg,
            max_step=float(max_step) if step_cap is None else min(float(max_step), float(step_cap)),
            events=span_events or None,
            dense_output=False,
            vectorized=False,
            **extra,
        )

    def _solve_span(
        t_start: float, t_end: float, y_start: np.ndarray, t_eval_span: np.ndarray,
        seeds: list[float | None] | None = None,
    ):
        if restart_plan is None or not restart_plan.events:
            wrappers = _step_consistent_events(events, seeds) if events else []
            span_wrappers[:] = wrappers
            return _solve_once(t_start, t_end, y_start, t_eval_span, wrappers)
        result = _solve_with_restarts(
            _solve_once, t_start, t_end, y_start, t_eval_span, list(events or []),
            list(restart_plan.events), restart_plan.max_step_s, seeds,
        )
        span_wrappers[:] = list(getattr(result, "user_wrappers", []))
        return result

    def _boundary_seeds(t_boundary: float) -> list[float | None] | None:
        """The events' sides at a chunk boundary: their values at the last step end there.

        The next chunk starts from the same state, so an event whose root lies on or just
        past the boundary is at rounding level at its start; without the side it is read
        as starting on its surface and the crossing is dropped (chunk_s = T/2 from a
        periapsis lost every apsis). A root just before the boundary was reported by the
        previous chunk, which leaves the post-crossing side.
        """
        if not events or len(span_wrappers) != len(events):
            return None
        sides: list[float | None] = []
        for wrapper in span_wrappers:
            ends = getattr(wrapper, "step_ends", None) or []
            t_end, value = ends[-1] if ends else (math.nan, math.nan)
            usable = t_end == t_boundary and math.isfinite(value) and value != 0.0
            sides.append(float(value) if usable else None)
        return sides

    checkpoint_every_chunk = bool(getattr(cfg, "checkpoint_every_chunk", False))
    integration_failed = False
    integration_failure_message: str | None = None
    stopped_early = False
    stop_reason = None
    chunk_idx = 0

    if _stop_requested(stop_file):
        # A stop file present before the first step stops the run at t0, as on
        # the fixed-step path. The in-solver stop event cannot see it: its sign
        # never changes while the file stays in place.
        logger.warning("Stop file %s exists before the run started; stopping at t0 = %.12g s.", stop_file, float(t0))
        stopped_early = True
        stop_reason = "stop file"
        t_cat: np.ndarray = np.asarray([t0], dtype=np.float64)
        y_cat: np.ndarray = y0_arr.reshape(-1, 1)
        t_events: list[np.ndarray] = [np.array([], dtype=np.float64) for _ in (events or [])]
        y_events: list[np.ndarray] = [np.zeros((0, y0_arr.size), dtype=np.float64) for _ in (events or [])]
        sol = SimpleNamespace(
            t=t_cat, y=y_cat, t_events=t_events, y_events=y_events,
            success=True, status=1, message="stop file present before start", nfev=0,
        )
    elif chunk_s is None:
        sol = _solve_span(t0, tf, y0_arr, t_eval)
        t_cat = np.asarray(sol.t, dtype=np.float64)
        y_cat = np.asarray(sol.y, dtype=np.float64)
        t_events = [np.asarray(te, dtype=np.float64) for te in (sol.t_events or [])]
        y_events = [np.asarray(ye, dtype=np.float64) for ye in (sol.y_events or [])]
        # Keep stopped_early consistent with a terminal-event stop (status==1)
        # so callers never see stop_reason set while stopped_early is False.
        if int(getattr(sol, "status", 0)) == 1:
            stopped_early = True
            terminal_endpoint = _terminal_event_endpoint(sol, events, state_size=y0_arr.size)
            if terminal_endpoint is not None:
                t_terminal, y_terminal = terminal_endpoint
                keep = t_cat <= (t_terminal + 1e-9)
                t_cat = t_cat[keep]
                y_cat = y_cat[:, keep]
                if t_cat.size == 0 or abs(float(t_cat[-1]) - t_terminal) > 1e-9:
                    t_cat = np.concatenate(
                        [t_cat, np.asarray([t_terminal], dtype=np.float64)]
                    )
                    y_cat = np.concatenate(
                        [y_cat, y_terminal.reshape(-1, 1)], axis=1
                    )
                else:
                    y_cat[:, -1] = y_terminal
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
        last_event_t = [-np.inf] * n_ev

        y_curr = y0_arr.copy()
        t_curr = float(t0)
        grid = np.asarray(t_eval, dtype=np.float64)
        grid_tol = _grid_tolerance_s(grid, float(chunk_s))
        last_kept_t = -np.inf

        while t_curr < tf - 1e-12:
            # Polled regardless of ``stop_event_in_scipy``: the in-solver stop
            # event cannot fire for a file created between two chunks, whose
            # sign is already "stop" at the next chunk's first evaluation.
            if _stop_requested(stop_file):
                stopped_early = True
                stop_reason = "stop file"
                break

            t_next = min(tf, t_curr + float(chunk_s))
            # Repeated ``t += chunk_s`` drifts by rounding; a chunk end within
            # the grid tolerance of an output time is that output time.
            nearest_index = int(np.argmin(np.abs(grid - t_next))) if grid.size else -1
            if nearest_index >= 0 and abs(float(grid[nearest_index]) - t_next) <= grid_tol:
                t_next = min(tf, float(grid[nearest_index]))
            mask = (t_eval >= t_curr - grid_tol) & (t_eval <= t_next + grid_tol)
            # Both chunk ends are always sampled, on the grid or not: the next
            # chunk restarts from the state at t_next, not from the last output
            # sample, so no interval is integrated twice and no event is
            # recorded twice. Only output-grid samples are kept below.
            t_eval_span = np.clip(t_eval[mask], t_curr, t_next)
            t_eval_span = np.unique(np.concatenate(([t_curr], t_eval_span, [t_next])).astype(np.float64))

            sol_k = _solve_span(t_curr, t_next, y_curr, t_eval_span, _boundary_seeds(t_curr) if chunk_idx else None)

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

            # Keep output-grid samples (and a terminal event endpoint) that are
            # new: a chunk's first sample repeats the previous chunk's last one,
            # and off-grid chunk ends are restart points, not output.
            on_grid = _nearest_grid_distance(sol_t, grid) <= grid_tol
            if terminal_endpoint is not None:
                on_grid[-1] = True
            keep_out = on_grid & (sol_t > last_kept_t + 1e-12)
            if np.any(keep_out):
                t_parts.append(sol_t[keep_out])
                y_parts.append(sol_y[:, keep_out])
                last_kept_t = float(sol_t[keep_out][-1])

            if getattr(sol_k, "t_events", None) is not None:
                for i in range(n_ev):
                    te = sol_k.t_events[i] if i < len(sol_k.t_events) else np.array([], dtype=np.float64)
                    ye = sol_k.y_events[i] if i < len(sol_k.y_events) else np.zeros((0, y0_arr.size), dtype=np.float64)
                    te_arr: np.ndarray = np.asarray(te, dtype=np.float64).reshape(-1)
                    # solve_ivp returns an empty 1-D array for a chunk without
                    # events; give it the (0, n) shape the concatenation needs.
                    ye_arr: np.ndarray = np.asarray(ye, dtype=np.float64).reshape(-1, y0_arr.size)
                    # A root exactly on the chunk boundary is reported by both
                    # chunks (g = 0 at the end of one and the start of the
                    # next); keep the first report only.
                    if te_arr.size and last_event_t[i] >= t_curr - grid_tol:
                        fresh = te_arr > t_curr + grid_tol
                        te_arr = te_arr[fresh]
                        ye_arr = ye_arr[fresh]
                    if te_arr.size:
                        last_event_t[i] = float(te_arr[-1])
                    t_events_acc[i].append(te_arr)
                    y_events_acc[i].append(ye_arr)

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
        t_last_s=(float(t_cat[-1]) if np.asarray(t_cat).size else None),
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

    return PropagationResult(
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


__all__ = ["run_scipy_propagation"]
