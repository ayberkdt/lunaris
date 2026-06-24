# tests/test_solver_tolerance_defaults.py
"""Method-aware tolerance defaults (pure-Python, no Qt)."""

from __future__ import annotations

import pytest

from lunaris.ui.core.solver_policy import (
    DEFAULT_ADAPTIVE_ATOL,
    DEFAULT_ADAPTIVE_RTOL,
    choose_solver_tolerances,
    method_default_tolerances,
)


def test_dop853_default_matches_backend_ssot():
    assert method_default_tolerances("DOP853 (Adaptive)") == (DEFAULT_ADAPTIVE_RTOL, DEFAULT_ADAPTIVE_ATOL)


def test_low_order_method_gets_looser_default():
    rtol_dop, _ = method_default_tolerances("DOP853")
    rtol_rk23, _ = method_default_tolerances("RK23 (Adaptive)")
    assert rtol_rk23 > rtol_dop  # RK23 (3rd order) should not chase a tight default


def test_stiff_methods_have_their_own_defaults():
    for label in ("RADAU (Adaptive · stiff)", "BDF (Adaptive · stiff)", "LSODA (Adaptive · auto)"):
        rtol, atol = method_default_tolerances(label)
        assert 0.0 < atol < rtol <= 1e-3


def test_fixed_step_and_unknown_fall_back_to_global_pair():
    assert method_default_tolerances("RK4 (Fixed-step)") == (DEFAULT_ADAPTIVE_RTOL, DEFAULT_ADAPTIVE_ATOL)
    assert method_default_tolerances("NOT_A_METHOD") == (DEFAULT_ADAPTIVE_RTOL, DEFAULT_ADAPTIVE_ATOL)
    assert method_default_tolerances("") == (DEFAULT_ADAPTIVE_RTOL, DEFAULT_ADAPTIVE_ATOL)


def test_bare_key_resolves():
    assert method_default_tolerances("RK45") == method_default_tolerances("RK45 (Adaptive)")


def test_blank_rtol_uses_method_default_in_choose():
    # RK23 blank rtol -> its looser default, not the global 1e-10.
    rtol, _ = choose_solver_tolerances("RK23 (Adaptive)", rtol="", atol=None)
    assert rtol == method_default_tolerances("RK23")[0]
    assert rtol > DEFAULT_ADAPTIVE_RTOL


def test_explicit_rtol_is_never_overridden_by_method_default():
    # An explicit value wins regardless of method.
    rtol, _ = choose_solver_tolerances("RK23 (Adaptive)", rtol="1e-9", atol=None)
    assert rtol == 1e-9


def test_explicit_rtol_atol_derivation_is_method_independent():
    # The derive floor stays global so existing behaviour is preserved.
    for label in ("DOP853 (Adaptive)", "RK45", "RK23 (Adaptive)"):
        rtol, atol = choose_solver_tolerances(label, rtol="1e-8", atol="")
        assert rtol == 1e-8
        assert atol == 1e-10
