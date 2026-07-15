"""Fixed-step propagation orchestration wrapper."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import numpy as np

from lunaris.common.type_defs import PropagationResult
from lunaris.core.propagation.events import EventOutcome
from lunaris.core.propagation.integrators.fixed_step import (
    _fixed_step_requires_6d,
    _integrate_fixed_step,
)


def run_fixed_step_propagation(
    *,
    rhs: Callable[[float, np.ndarray], np.ndarray],
    t_eval: np.ndarray,
    y0: np.ndarray,
    max_step_s: float,
    method: str,
    events: list[Callable[[float, np.ndarray], float]] | None,
    R_ref_m: float,
    mu_m3s2: float,
    verbose: bool,
    heartbeat_hours: float,
    stop_file: str | None,
    checkpoint_path: str | None,
    checkpoint_metadata: dict[str, Any] | None,
    max_internal_steps: int | None,
    logger: logging.Logger,
) -> PropagationResult:
    """Run the fixed-step branch and assemble a propagation result."""

    meth_name = str(method)
    if verbose:
        t_eval_arr = np.asarray(t_eval, dtype=np.float64)
        dt_out = (
            float(t_eval_arr[1] - t_eval_arr[0])
            if t_eval_arr.size >= 2
            else float("nan")
        )
        logger.info(
            "[PROP] Fixed-step %s: dt_out=%gs, max_step=%.6fs",
            meth_name,
            dt_out,
            float(max_step_s),
        )

    y0_arr = np.asarray(y0, dtype=np.float64).reshape(-1)
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
        max_step=float(max_step_s),
        method=meth_name,
        events=events,
        R_ref_m=float(R_ref_m),
        mu_m3s2=float(mu_m3s2),
        verbose=verbose,
        heartbeat_hours=float(heartbeat_hours),
        stop_file=stop_file,
        checkpoint_path=checkpoint_path,
        checkpoint_metadata=checkpoint_metadata,
        max_internal_steps=max_internal_steps,
    )

    t_out = np.asarray(ode_like.t, dtype=np.float64)
    y_row = np.asarray(ode_like.y, dtype=np.float64).T
    outcome = EventOutcome(
        impacted=bool(impacted),
        t_impact_s=(float(t_imp) if t_imp is not None else None),
        y_impact=(np.asarray(y_imp, dtype=np.float64) if y_imp is not None else None),
        stopped_early=bool(stopped_early),
        stop_reason=stop_reason or ("impact" if impacted else None),
        t_stop_s=t_stop,
    )

    return PropagationResult(
        t=t_out,
        y=y_row,
        ode=ode_like,
        t_events=list(getattr(ode_like, "t_events", [])),
        y_events=list(getattr(ode_like, "y_events", [])),
        impacted=outcome.impacted,
        t_impact_s=outcome.t_impact_s,
        y_impact=outcome.y_impact,
        stopped_early=outcome.stopped_early,
        stop_reason=outcome.stop_reason,
        t_stop_s=outcome.t_stop_s,
        diagnostics={},
    )


__all__ = ["run_fixed_step_propagation"]
