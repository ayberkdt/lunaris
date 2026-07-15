"""SciPy (``solve_ivp``) propagation orchestration.

Owns the adaptive-integration branch of :func:`lunaris.core.propagation.propagate`:
the normal single-span solve and the chunked solve with per-chunk checkpointing,
terminal-event endpoint preservation, event accumulation and stop-file handling.
Extracted from ``propagator.py`` (seam-cleanup item 5, P3 slice); behavior is
preserved exactly — no numerical change.
"""

from __future__ import annotations

import logging
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
            events=(events or None),
            dense_output=False,
            vectorized=False,
        )

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
