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

import numpy as np
from scipy.integrate import solve_ivp

from lunaris.common.type_defs import PropagationResult, PropagatorConfig
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


def _step_consistent_events(
    events: list[Callable[[float, np.ndarray], float]],
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
    """

    def consistent(event: Callable[[float, np.ndarray], float]) -> Callable[[float, np.ndarray], float]:
        step_ends: list[tuple[float, float]] = []
        terminal = bool(getattr(event, "terminal", False))
        started = False
        surface_band: float | None = None

        def g(t: float, y: np.ndarray) -> float:
            nonlocal started, surface_band
            t_key = float(t)
            value = event(t, y)
            if not terminal and not started:
                started = True
                g0 = float(value)
                if math.isfinite(g0):
                    band = _rounding_band(event, t, y, g0)
                    if math.isfinite(band) and abs(g0) <= band:
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
                step_ends.append((t_key, float(value)))
                del step_ends[:-2]
            return value

        g.terminal = getattr(event, "terminal", False)  # type: ignore[attr-defined]
        g.direction = getattr(event, "direction", 0.0)  # type: ignore[attr-defined]
        return g

    return [consistent(event) for event in events]


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
) -> PropagationResult:
    """Run the SciPy adaptive branch and assemble a propagation result.

    The returned result carries an empty ``diagnostics`` dict; the caller
    (:func:`propagate`) attaches the full diagnostics afterwards.
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
            events=(_step_consistent_events(events) if events else None),
            dense_output=False,
            vectorized=False,
        )

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
