"""Step-size convergence gate for fixed-step integrators.

An external NASA-grade review noted that Lunaris ships several fixed-step
integrators with per-step error tests, but does not enforce a *convergence gate*
inside its scientific-benchmark flow: a run that claims a given integrator order
should demonstrate, on a refined step sequence, that the observed order matches
the nominal one before its numbers back a claim.

This module is the reusable primitive for that gate. It is deliberately
propagation-agnostic: callers provide the three refined *solutions* (or their
errors against a reference), and this module computes the observed order and a
Richardson error estimate, then judges them against a tolerance band.

Two estimators, matching the two situations that arise in practice:

* **Reference-free** (:func:`observed_order_from_solutions`) — no exact solution
  is known, so the order is read from successive solution differences on a
  ``dt, dt/2, dt/4`` sequence: ``p = log_r(|u_c - u_m| / |u_m - u_f|)``.
* **Reference-based** (:func:`observed_order_from_errors`) — an exact/reference
  solution is available (e.g. an analytic Kepler orbit), so the order is read
  directly from the error ratio: ``p = log_r(e_c / e_m)``.

The gate result serialises to a manifest dict via
:meth:`ConvergenceGateResult.to_manifest`, so a benchmark/paper-evidence run can
stamp ``observed_order`` and ``richardson_error_estimate`` alongside its other
provenance. In ``paper_safe`` runs that intentionally skip the gate, record the
explicit :func:`skipped_gate` marker rather than omitting the field, so a reader
can tell "gate passed" from "gate not run".
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = [
    "ConvergenceGateResult",
    "observed_order_from_solutions",
    "observed_order_from_errors",
    "richardson_error_estimate",
    "evaluate_convergence",
    "run_convergence_gate",
    "skipped_gate",
]


def _norm(value: Any) -> float:
    """L2 norm of a scalar or vector solution/difference."""
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    return float(np.linalg.norm(arr))


def observed_order_from_solutions(
    u_coarse: Any,
    u_medium: Any,
    u_fine: Any,
    *,
    refinement: float = 2.0,
) -> float:
    """Observed order from three solutions on a ``dt, dt/2, dt/4`` sequence.

    ``p = log_r(|u_c - u_m| / |u_m - u_f|)`` with refinement ratio ``r``. Used
    when no exact solution is available. Returns ``inf`` when the finer
    difference underflows to zero (already at round-off — treat as converged).
    """
    if refinement <= 1.0:
        raise ValueError(f"refinement must be > 1, got {refinement!r}")
    d_coarse = _norm(np.asarray(u_coarse, np.float64) - np.asarray(u_medium, np.float64))
    d_fine = _norm(np.asarray(u_medium, np.float64) - np.asarray(u_fine, np.float64))
    if d_fine == 0.0:
        return math.inf
    if d_coarse == 0.0:
        return 0.0
    return math.log(d_coarse / d_fine) / math.log(refinement)


def observed_order_from_errors(
    error_coarse: float,
    error_medium: float,
    *,
    refinement: float = 2.0,
) -> float:
    """Observed order from two errors against a known reference.

    ``p = log_r(e_c / e_m)``. Returns ``inf`` when the finer error underflows.
    """
    if refinement <= 1.0:
        raise ValueError(f"refinement must be > 1, got {refinement!r}")
    e_c = abs(float(error_coarse))
    e_m = abs(float(error_medium))
    if e_m == 0.0:
        return math.inf
    if e_c == 0.0:
        return 0.0
    return math.log(e_c / e_m) / math.log(refinement)


def richardson_error_estimate(
    u_medium: Any,
    u_fine: Any,
    order: float,
    *,
    refinement: float = 2.0,
) -> float:
    """Richardson estimate of the finest solution's error.

    ``err_fine ~= |u_f - u_m| / (r^p - 1)`` — the standard extrapolation bound
    on the remaining discretisation error at the finest step.
    """
    denom = refinement**order - 1.0
    if denom <= 0.0 or not math.isfinite(denom):
        return math.inf
    return _norm(np.asarray(u_fine, np.float64) - np.asarray(u_medium, np.float64)) / denom


@dataclass(frozen=True)
class ConvergenceGateResult:
    """Verdict of a step-size convergence gate.

    ``passed`` is ``None`` when the gate was skipped (see :func:`skipped_gate`),
    ``True``/``False`` when it ran.
    """

    expected_order: float
    observed_order: float
    richardson_error_estimate: float
    order_tol: float
    passed: bool | None
    dts: tuple[float, ...] = ()
    status: str = "evaluated"
    note: str = ""
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_manifest(self) -> dict[str, Any]:
        """JSON-safe dict for stamping into a run/benchmark manifest."""
        return {
            "status": self.status,
            "expected_order": _finite_or_none(self.expected_order),
            "observed_order": _finite_or_none(self.observed_order),
            "richardson_error_estimate": _finite_or_none(self.richardson_error_estimate),
            "order_tol": _finite_or_none(self.order_tol),
            "passed": self.passed,
            "dts": list(self.dts),
            "note": self.note,
        }


def _finite_or_none(value: float) -> float | None:
    v = float(value)
    return v if math.isfinite(v) else None


def evaluate_convergence(
    *,
    expected_order: float,
    observed_order: float,
    richardson_error_estimate: float,
    order_tol: float = 0.5,
    dts: Sequence[float] = (),
    note: str = "",
    diagnostics: dict[str, Any] | None = None,
) -> ConvergenceGateResult:
    """Judge an observed order against an expected-order tolerance band.

    The gate passes when ``|observed - expected| <= order_tol`` (an observed
    order *above* expected is fine — superconvergence — but a large positive
    deviation usually means the error is at the round-off floor, which the caller
    can inspect via the Richardson estimate). ``inf`` observed order (finest
    difference at round-off) passes: the scheme has converged to machine
    precision.
    """
    if math.isinf(observed_order):
        passed = True
    else:
        passed = abs(observed_order - expected_order) <= order_tol
    return ConvergenceGateResult(
        expected_order=float(expected_order),
        observed_order=float(observed_order),
        richardson_error_estimate=float(richardson_error_estimate),
        order_tol=float(order_tol),
        passed=bool(passed),
        dts=tuple(float(d) for d in dts),
        status="evaluated",
        note=note,
        diagnostics=dict(diagnostics or {}),
    )


def skipped_gate(reason: str, *, expected_order: float = float("nan")) -> ConvergenceGateResult:
    """A gate marker for runs that intentionally do not run the gate.

    ``paper_safe`` reference runs may fix the step deliberately and skip the
    refinement sweep; recording this keeps "gate passed" distinct from "gate not
    run" in the manifest.
    """
    return ConvergenceGateResult(
        expected_order=float(expected_order),
        observed_order=float("nan"),
        richardson_error_estimate=float("nan"),
        order_tol=float("nan"),
        passed=None,
        status="skipped",
        note=reason,
    )


def run_convergence_gate(
    solve: Any,
    *,
    base_dt: float,
    expected_order: float,
    reference: Any = None,
    refinement: float = 2.0,
    order_tol: float = 0.5,
) -> ConvergenceGateResult:
    """Run a ``dt, dt/refinement, dt/refinement^2`` sweep and gate the result.

    Parameters
    ----------
    solve:
        Callable ``solve(dt) -> solution`` returning a comparable solution
        (scalar, vector, or array; e.g. a final state or an error metric). Called
        three times, at ``base_dt``, ``base_dt/refinement``, ``base_dt/refinement^2``.
    base_dt:
        Coarsest step of the sweep.
    expected_order:
        Nominal integrator order (e.g. 4 for classical RK4).
    reference:
        Optional exact/reference solution. When given, the order is read from the
        error ratio against it; otherwise from successive solution differences.
    """
    if base_dt <= 0.0 or not math.isfinite(base_dt):
        raise ValueError(f"base_dt must be finite and > 0, got {base_dt!r}")
    dts = (base_dt, base_dt / refinement, base_dt / (refinement * refinement))
    u_c, u_m, u_f = (solve(dt) for dt in dts)

    if reference is not None:
        ref = np.asarray(reference, dtype=np.float64)
        e_c = _norm(np.asarray(u_c, np.float64) - ref)
        e_m = _norm(np.asarray(u_m, np.float64) - ref)
        e_f = _norm(np.asarray(u_f, np.float64) - ref)
        order = observed_order_from_errors(e_c, e_m, refinement=refinement)
        rich = richardson_error_estimate(u_m, u_f, order, refinement=refinement)
        diagnostics = {"error_coarse": e_c, "error_medium": e_m, "error_fine": e_f}
    else:
        order = observed_order_from_solutions(u_c, u_m, u_f, refinement=refinement)
        rich = richardson_error_estimate(u_m, u_f, order, refinement=refinement)
        diagnostics = {}

    return evaluate_convergence(
        expected_order=expected_order,
        observed_order=order,
        richardson_error_estimate=rich,
        order_tol=order_tol,
        dts=dts,
        note=(
            "reference-based" if reference is not None else "reference-free"
        )
        + f" convergence over dt={dts[0]:.6g}/{dts[1]:.6g}/{dts[2]:.6g} s",
        diagnostics=diagnostics,
    )
