# tests/test_type_defs_contracts.py
# -*- coding: utf-8 -*-
"""
Configuration / dataclass *contract* tests for ``lunaris.common.type_defs``.

This module deliberately complements (rather than duplicates) ``test_type_defs.py``.
It pins down the contracts most likely to hide a *silent* scientific or
configuration mistake:

- ``SpacecraftProps.ballistic_coefficient`` edge cases (zero area / zero Cd → inf),
- ``GravityConfig`` backend validation (rejecting typos, case/whitespace handling,
  the ``uses_st_lrps`` switch, and the per-backend required-field rules),
- ``InitialState`` returning *fresh, independent* float64 arrays (no shared
  references that could be mutated under a caller's feet),
- a few invariant checks (frozen dataclasses, tides degree/kind matrix).

All tests are pure-Python, deterministic, and require no external data.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from lunaris.common.type_defs import (
    AdaptiveDegreeConfig,
    GravityConfig,
    InitialState,
    PerturbationFlags,
    SpacecraftProps,
)


# =============================================================================
# SpacecraftProps — ballistic coefficient & validation edges
# =============================================================================

def test_ballistic_coefficient_normal_case():
    sp = SpacecraftProps(mass_kg=500.0, area_m2=2.0, cd=2.5, cr=1.4)
    # BC = m / (Cd * A) = 500 / (2.5 * 2.0) = 100
    assert sp.ballistic_coefficient == pytest.approx(100.0)


@pytest.mark.parametrize("kwargs", [
    dict(mass_kg=100.0, area_m2=0.0, cd=2.2),   # zero area  -> no drag coupling
    dict(mass_kg=100.0, area_m2=2.0, cd=0.0),   # zero Cd     -> no drag coupling
])
def test_ballistic_coefficient_is_infinite_without_drag_coupling(kwargs):
    sp = SpacecraftProps(**kwargs)
    assert math.isinf(sp.ballistic_coefficient)


@pytest.mark.parametrize("kwargs", [
    dict(mass_kg=0.0),                 # mass must be strictly positive
    dict(mass_kg=-1.0),
    dict(area_m2=-0.1),                # area must be >= 0
    dict(cd=-0.1),                     # cd must be >= 0
    dict(cr=-0.1),                     # cr must be >= 0
])
def test_spacecraft_props_rejects_invalid(kwargs):
    with pytest.raises(ValueError):
        SpacecraftProps(**kwargs)


def test_spacecraft_props_is_frozen():
    sp = SpacecraftProps()
    with pytest.raises(dataclasses.FrozenInstanceError):
        sp.mass_kg = 10.0  # type: ignore[misc]


# =============================================================================
# GravityConfig — backend validation & required-field rules
# =============================================================================

def test_gravity_config_rejects_unknown_backend():
    with pytest.raises(ValueError, match="classic_sh.*st_lrps|backend"):
        GravityConfig(file_path="grav.tab", backend="nope")


def test_gravity_config_backend_is_case_and_whitespace_insensitive():
    # Mixed case / padded backend names normalize for validation.
    cfg = GravityConfig(file_path="", backend="  ST_LRPS ", st_lrps_model_dir="run")
    assert cfg.uses_st_lrps is True

    cfg2 = GravityConfig(file_path="grav.tab", backend="Classic_SH")
    assert cfg2.uses_st_lrps is False


def test_gravity_config_classic_sh_requires_file_path():
    with pytest.raises(ValueError, match="file_path"):
        GravityConfig(file_path="", backend="classic_sh")


def test_gravity_config_st_lrps_requires_model_dir():
    with pytest.raises(ValueError, match="st_lrps_model_dir"):
        GravityConfig(file_path="grav.tab", backend="st_lrps", st_lrps_model_dir="")


def test_gravity_config_rejects_negative_degree():
    with pytest.raises(ValueError, match="degree"):
        GravityConfig(file_path="grav.tab", degree=-3)


def test_gravity_config_adaptive_min_degree_cannot_exceed_max_degree():
    with pytest.raises(ValueError, match="min_degree"):
        GravityConfig(
            file_path="grav.tab",
            degree=10,
            adaptive=AdaptiveDegreeConfig(min_degree=20),
        )


def test_gravity_config_allows_adaptive_min_degree_equal_to_degree():
    # Boundary: equal is allowed (only strictly greater is rejected).
    cfg = GravityConfig(
        file_path="grav.tab",
        degree=10,
        adaptive=AdaptiveDegreeConfig(min_degree=10),
    )
    assert cfg.adaptive.min_degree == 10


# =============================================================================
# AdaptiveDegreeConfig — power / quantization / table monotonicity
# =============================================================================

@pytest.mark.parametrize("kwargs", [
    dict(power=0.0),                # power must be > 0
    dict(power=-1.0),
    dict(min_degree=-1),            # min_degree must be >= 0
    dict(quantization_step=0),      # step must be >= 1
])
def test_adaptive_degree_config_rejects_invalid_scalars(kwargs):
    with pytest.raises(ValueError):
        AdaptiveDegreeConfig(**kwargs)


def test_adaptive_degree_config_rejects_non_increasing_table():
    with pytest.raises(ValueError, match="increasing"):
        AdaptiveDegreeConfig(altitude_table=((200.0, 40), (100.0, 60)))


# =============================================================================
# PerturbationFlags — tides coupling + derived properties
# =============================================================================

def test_tides_k3_requires_k2():
    with pytest.raises(ValueError, match="enable_tides_k3"):
        PerturbationFlags(enable_tides_k3=True, enable_tides_k2=False)


@pytest.mark.parametrize(
    "flags,expect_degree,expect_kind",
    [
        (dict(), 0, "none"),
        (dict(enable_tides_k2=True), 2, "k2"),
        (dict(enable_tides_k2=True, enable_tides_k3=True), 3, "k3"),
    ],
)
def test_tides_degree_and_kind_matrix(flags, expect_degree, expect_kind):
    pf = PerturbationFlags(**flags)
    assert pf.tides_degree == expect_degree
    assert pf.tides_kind == expect_kind
    assert pf.enable_tides == (expect_degree > 0)


def test_third_body_and_surface_force_properties():
    assert PerturbationFlags(enable_3rd_body_earth=True).enable_third_body is True
    assert PerturbationFlags(enable_earth_j2=True).enable_third_body is True
    assert PerturbationFlags(enable_albedo=True).enable_surface_forces is True
    assert PerturbationFlags(enable_thermal=True).enable_surface_forces is True
    # Nothing enabled -> both derived switches are False.
    base = PerturbationFlags()
    assert base.enable_third_body is False
    assert base.enable_surface_forces is False


# =============================================================================
# InitialState — float64 contract + independence (no shared references)
# =============================================================================

def test_initial_state_to_array_is_float64_and_correct():
    st = InitialState(x=1.0, y=2.0, z=3.0, vx=-4.0, vy=-5.0, vz=-6.0)
    arr = st.to_array()
    assert arr.dtype == np.float64
    assert arr.shape == (6,)
    np.testing.assert_array_equal(arr, [1.0, 2.0, 3.0, -4.0, -5.0, -6.0])


def test_initial_state_returns_independent_arrays_each_call():
    """Each accessor must return a fresh array; mutating one must not affect others
    or subsequent calls (guards against a shared-buffer aliasing bug)."""
    st = InitialState(x=1.0, y=2.0, z=3.0, vx=4.0, vy=5.0, vz=6.0)

    a1 = st.to_array()
    a2 = st.to_array()
    assert a1 is not a2
    a1[0] = 999.0
    assert a2[0] == 1.0            # mutation did not leak
    assert st.to_array()[0] == 1.0  # source dataclass is unaffected (frozen)

    r = st.r_vec()
    v = st.v_vec()
    assert r.dtype == np.float64 and v.dtype == np.float64
    np.testing.assert_array_equal(r, [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(v, [4.0, 5.0, 6.0])
    r[0] = -1.0
    assert st.r_vec()[0] == 1.0    # r_vec returns a fresh array too
