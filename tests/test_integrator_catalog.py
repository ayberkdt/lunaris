# tests/test_integrator_catalog.py
"""Unit tests for the UI integrator catalog (pure-Python, no Qt required)."""

from __future__ import annotations

import pytest

from lunaris.core import propagator as P
from lunaris.ui.core.integrator_catalog import (
    INTEGRATOR_CATALOG,
    catalog_labels,
    grouped_labels,
    spec_for_label,
)
from lunaris.ui.core.solver_policy import solver_method_is_adaptive


def test_label_starts_with_backend_key():
    # Command building / session restore take the first whitespace token as the
    # backend method, so every label must start with its key.
    for spec in INTEGRATOR_CATALOG:
        assert spec.label.split()[0] == spec.key, spec.label


def test_keys_are_recognized_by_the_propagator():
    for spec in INTEGRATOR_CATALOG:
        if spec.family == "adaptive":
            # SciPy adaptive methods are dispatched by name, not in-house.
            assert not P._is_fixed_step_method(spec.key), spec.key
        else:
            assert P._is_fixed_step_method(spec.key), spec.key


def test_family_matches_solver_policy_adaptive_flag():
    # The card's family classification must agree with the tolerance policy.
    for spec in INTEGRATOR_CATALOG:
        assert spec.is_adaptive == solver_method_is_adaptive(spec.label), spec.label


def test_symplectic_specs_align_with_propagator():
    for spec in INTEGRATOR_CATALOG:
        if spec.family == "symplectic":
            assert P._is_symplectic_method(spec.key), spec.key
        else:
            assert not P._is_symplectic_method(spec.key), spec.key


def test_fixed_step_methods_use_fixed_step_mode():
    for spec in INTEGRATOR_CATALOG:
        assert spec.uses_fixed_step == (not spec.is_adaptive), spec.label


@pytest.mark.parametrize("label", catalog_labels())
def test_spec_for_label_roundtrip(label):
    spec = spec_for_label(label)
    assert spec is not None and spec.label == label


@pytest.mark.parametrize("bare", ["DOP853", "rk4", "YOSHIDA8", "RKN4"])
def test_spec_for_label_accepts_bare_keys(bare):
    spec = spec_for_label(bare)
    assert spec is not None and spec.key.upper() == bare.upper()


def test_spec_for_label_unknown_returns_none():
    assert spec_for_label("NOT_A_METHOD") is None
    assert spec_for_label("") is None
    assert spec_for_label(None) is None


def test_grouped_labels_cover_catalog_in_order():
    flat = [label for _, labels in grouped_labels() for label in labels]
    assert flat == catalog_labels()
    families = [fam for fam, _ in grouped_labels()]
    # Families appear as contiguous, non-repeating blocks.
    assert len(families) == len(set(families))
