# tests/test_integrator_catalog.py
"""Unit tests for the UI integrator catalog (pure-Python, no Qt required)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from lunaris.common.type_defs import PerturbationFlags
from lunaris.core.propagation.integrators.fixed_step import (
    _is_fixed_step_method,
    _is_symplectic_method,
    symplectic_breaks_separability,
    symplectic_nonconservative_violations,
)
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
            assert not _is_fixed_step_method(spec.key), spec.key
        else:
            assert _is_fixed_step_method(spec.key), spec.key


def test_family_matches_solver_policy_adaptive_flag():
    # The card's family classification must agree with the tolerance policy.
    for spec in INTEGRATOR_CATALOG:
        assert spec.is_adaptive == solver_method_is_adaptive(spec.label), spec.label


def test_symplectic_specs_align_with_propagator():
    for spec in INTEGRATOR_CATALOG:
        if spec.family == "symplectic":
            assert _is_symplectic_method(spec.key), spec.key
        else:
            assert not _is_symplectic_method(spec.key), spec.key


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


def test_symplectic_guard_flags_nonconservative_forces():
    # SRP under a symplectic method voids the bounded-drift guarantee -> flagged.
    flags = PerturbationFlags(enable_sh=True, enable_srp=True)
    assert symplectic_nonconservative_violations("PEFRL", flags) == ["SRP"]
    # ... but SRP under a non-symplectic method is fine (no guarantee to void).
    assert symplectic_nonconservative_violations("DOP853", flags) == []
    assert symplectic_nonconservative_violations("RK4", flags) == []


def test_symplectic_guard_silent_for_conservative_only():
    # Gravity + third-body + Earth J2 are position-only (conservative); a
    # symplectic integrator is the correct choice and must NOT be flagged.
    flags = PerturbationFlags(
        enable_sh=True, enable_3rd_body_sun=True, enable_3rd_body_earth=True
    )
    for method in ("VV", "PEFRL", "YOSHIDA4", "YOSHIDA6", "YOSHIDA8"):
        assert symplectic_nonconservative_violations(method, flags) == []
        assert not symplectic_breaks_separability(method, flags)


def test_symplectic_guard_collects_all_active_nonconservative_forces():
    flags = PerturbationFlags(
        enable_sh=True, enable_srp=True, enable_albedo=True,
        enable_thermal=True, enable_relativity_1pn=True,
    )
    violations = symplectic_nonconservative_violations("YOSHIDA6", flags)
    assert set(violations) == {"SRP", "albedo", "thermal IR", "1PN relativity"}


def test_symplectic_guard_separability_only_for_velocity_dependent():
    # 1PN relativity is velocity-dependent -> breaks separability (worse mode).
    rel = PerturbationFlags(enable_sh=True, enable_relativity_1pn=True)
    assert symplectic_breaks_separability("PEFRL", rel)
    # SRP is non-conservative but position/time-only -> not a separability break.
    srp = PerturbationFlags(enable_sh=True, enable_srp=True)
    assert not symplectic_breaks_separability("PEFRL", srp)


def test_symplectic_guard_safe_on_partial_flag_objects_and_none():
    # Helper must tolerate arbitrary flag-like objects and None.
    assert symplectic_nonconservative_violations("PEFRL", None) == []
    partial = SimpleNamespace(enable_srp=True)  # missing other attrs
    assert symplectic_nonconservative_violations("PEFRL", partial) == ["SRP"]


def test_grouped_labels_cover_catalog_in_order():
    flat = [label for _, labels in grouped_labels() for label in labels]
    assert flat == catalog_labels()
    families = [fam for fam, _ in grouped_labels()]
    # Families appear as contiguous, non-repeating blocks.
    assert len(families) == len(set(families))
