# tests/test_integrator_estimates.py
"""Unit tests for the pure solver estimate / validation helpers (no Qt)."""

from __future__ import annotations

import pytest

from lunaris.ui.core.integrator_catalog import spec_for_label
from lunaris.ui.core.integrator_estimates import (
    accuracy_label,
    estimate_fixed_step_cost,
    evals_per_step,
    validate_solver_inputs,
)


def _spec(label):
    s = spec_for_label(label)
    assert s is not None
    return s


@pytest.mark.parametrize(
    "key,expected",
    [("VV", 2), ("PEFRL", 4), ("YOSHIDA4", 6), ("YOSHIDA6", 18),
     ("YOSHIDA8", 54), ("RKN4", 3), ("RK4", 4), ("RK8", 24)],
)
def test_evals_per_step_matches_implementation(key, expected):
    assert evals_per_step(key) == expected


def test_evals_per_step_none_for_adaptive():
    assert evals_per_step("DOP853") is None


@pytest.mark.parametrize(
    "rtol,label",
    [
        (1e-12, "Very high accuracy"),
        (1e-10, "High accuracy"),
        (1e-8, "Balanced"),
        (1e-5, "Coarse"),
        (1e-2, "Very coarse"),
        (0.0, "—"),
        ("", "—"),
        (None, "—"),
    ],
)
def test_accuracy_label(rtol, label):
    assert accuracy_label(rtol) == label


def test_cost_adaptive_has_no_fixed_estimate():
    est = estimate_fixed_step_cost(_spec("DOP853 (Adaptive)"), 86400.0, None)
    assert est.mode == "adaptive"
    assert est.n_steps is None and est.total_evals is None


def test_cost_auto_when_step_blank():
    est = estimate_fixed_step_cost(_spec("RK4 (Fixed-step)"), 86400.0, None)
    assert est.mode == "auto"


def test_cost_fixed_counts_steps_and_evals():
    # 1 day at 60 s steps -> 1440 steps; RK4 = 4 evals/step.
    est = estimate_fixed_step_cost(_spec("RK4 (Fixed-step)"), 86400.0, 60.0)
    assert est.mode == "fixed"
    assert est.n_steps == 1440
    assert est.total_evals == 1440 * 4


def test_cost_fixed_uses_method_specific_evals():
    est = estimate_fixed_step_cost(_spec("YOSHIDA8 (Symplectic)"), 86400.0, 60.0)
    assert est.total_evals == 1440 * 54


def test_cost_unknown_without_duration():
    est = estimate_fixed_step_cost(_spec("RK4 (Fixed-step)"), 0.0, 60.0)
    assert est.mode == "unknown"


def test_validate_adaptive_rtol_band():
    spec = _spec("DOP853 (Adaptive)")
    assert validate_solver_inputs(spec, duration_s=86400.0, step_s=None, rtol="1e-10") == []
    # too loose -> warning
    issues = validate_solver_inputs(spec, duration_s=86400.0, step_s=None, rtol="1e-1")
    assert issues and issues[0][0] == "warning"
    # non-positive -> error
    issues = validate_solver_inputs(spec, duration_s=86400.0, step_s=None, rtol="-1")
    assert issues and issues[0][0] == "error"


def test_validate_fixed_step_resolution():
    spec = _spec("RK4 (Fixed-step)")
    # Healthy resolution -> no issues.
    assert validate_solver_inputs(spec, duration_s=86400.0, step_s=60.0) == []
    # Step >= duration -> warning.
    issues = validate_solver_inputs(spec, duration_s=3600.0, step_s=7200.0)
    assert issues and issues[0][0] == "warning"
    # Too few steps -> warning.
    issues = validate_solver_inputs(spec, duration_s=1000.0, step_s=100.0)
    assert issues and issues[0][0] == "warning"
    # Non-positive step -> error.
    issues = validate_solver_inputs(spec, duration_s=86400.0, step_s=-1.0)
    assert issues and issues[0][0] == "error"


def test_validate_none_spec_is_empty():
    assert validate_solver_inputs(None, duration_s=1.0, step_s=1.0) == []
