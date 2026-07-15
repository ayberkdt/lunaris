"""Tests for the step-size convergence gate (validation.convergence).

The gate is a propagation-agnostic primitive: given three refined solutions (or
errors against a reference), it must recover the integrator's observed order,
produce a Richardson error estimate, and pass/fail against a tolerance band.
These tests drive it with analytic error models of known order so the recovered
order is exactly predictable.
"""

from __future__ import annotations

import math

from lunaris.validation.convergence import (
    ConvergenceGateResult,
    evaluate_convergence,
    observed_order_from_errors,
    observed_order_from_solutions,
    richardson_error_estimate,
    run_convergence_gate,
    skipped_gate,
)


def test_reference_based_order_recovers_p_for_power_law_error():
    # error(dt) = C * dt^p ; ratio over a factor-2 refinement recovers p exactly.
    for p in (2.0, 4.0, 8.0):
        c, dt = 3.0, 0.1
        e_coarse = c * dt**p
        e_medium = c * (dt / 2) ** p
        assert math.isclose(observed_order_from_errors(e_coarse, e_medium), p, rel_tol=1e-12)


def test_reference_free_order_recovers_p_from_solution_differences():
    # u(dt) = u_exact + C*dt^p ; successive differences give order p.
    for p in (2.0, 4.0):
        u_exact, c, dt = 100.0, 5.0, 0.2
        u_c = u_exact + c * dt**p
        u_m = u_exact + c * (dt / 2) ** p
        u_f = u_exact + c * (dt / 4) ** p
        order = observed_order_from_solutions(u_c, u_m, u_f)
        assert math.isclose(order, p, rel_tol=1e-9)


def test_observed_order_handles_roundoff_floor():
    # Identical finer solutions -> finest difference is zero -> treat as converged.
    assert observed_order_from_solutions(1.0, 1.0 + 1e-9, 1.0 + 1e-9) == math.inf
    assert observed_order_from_errors(1e-3, 0.0) == math.inf


def test_richardson_error_estimate_matches_leading_term():
    # For u = u_exact + C*dt^p, |u_f - u_m| / (2^p - 1) ~= C*(dt/4)^p (the fine error).
    u_exact, c, dt, p = 0.0, 2.0, 0.4, 4.0
    u_m = u_exact + c * (dt / 2) ** p
    u_f = u_exact + c * (dt / 4) ** p
    est = richardson_error_estimate(u_m, u_f, p)
    true_fine_error = abs(u_f - u_exact)
    assert math.isclose(est, true_fine_error, rel_tol=1e-9)


def test_gate_passes_within_band_and_fails_outside():
    good = evaluate_convergence(
        expected_order=4.0, observed_order=4.1, richardson_error_estimate=1e-9
    )
    assert good.passed is True
    bad = evaluate_convergence(
        expected_order=4.0, observed_order=2.7, richardson_error_estimate=1e-3
    )
    assert bad.passed is False
    # Superconvergence to the round-off floor (inf) passes.
    conv = evaluate_convergence(
        expected_order=4.0, observed_order=math.inf, richardson_error_estimate=0.0
    )
    assert conv.passed is True


def test_run_gate_end_to_end_with_reference():
    # A synthetic 4th-order solver: solution = reference + C*dt^4.
    reference = 42.0

    def solve(dt: float) -> float:
        return reference + 1.5 * dt**4

    result = run_convergence_gate(
        solve, base_dt=0.1, expected_order=4.0, reference=reference
    )
    assert isinstance(result, ConvergenceGateResult)
    assert math.isclose(result.observed_order, 4.0, rel_tol=1e-9)
    assert result.passed is True
    manifest = result.to_manifest()
    assert manifest["status"] == "evaluated"
    assert manifest["passed"] is True
    assert math.isclose(manifest["observed_order"], 4.0, rel_tol=1e-9)


def test_skipped_gate_is_distinct_from_pass():
    result = skipped_gate("paper_safe: fixed reference step, no refinement sweep")
    assert result.passed is None
    manifest = result.to_manifest()
    assert manifest["status"] == "skipped"
    assert manifest["passed"] is None
    # Non-finite fields serialise to None, never NaN, in the manifest.
    assert manifest["observed_order"] is None
