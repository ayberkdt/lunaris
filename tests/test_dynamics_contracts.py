# tests/test_dynamics_contracts.py
# -*- coding: utf-8 -*-
"""
Dependency / provider *contract* tests for ``lunaris.core.dynamics``.

These tests pin down ``DynamicsEngine._validate_dependencies()`` and the strict
provider-extraction helpers WITHOUT requiring real SPICE kernels, gravity files,
or Numba RHS compilation. ``DynamicsEngine.__init__`` runs validation eagerly, so
constructing an engine is enough to exercise the contract; we never call
``build_rhs()`` for the heavy classical path here.

Why this matters: a misconfigured force model that *silently* runs with the wrong
provider (or no provider) is exactly the class of bug that produces plausible-but-
wrong physics. Each test asserts a clear, early failure.
"""

from __future__ import annotations

import numpy as np
import pytest

from lunaris.common.type_defs import PerturbationFlags, SpacecraftProps
from lunaris.core.dynamics import (
    DynamicsEngine,
    _is_surrogate_gravity_provider,
    extract_gravity_strict,
)


# -----------------------------------------------------------------------------
# Helpers / fixtures
# -----------------------------------------------------------------------------
def _sc(**kw) -> SpacecraftProps:
    base = dict(mass_kg=10.0, area_m2=1.0, cd=2.2, cr=1.5)
    base.update(kw)
    return SpacecraftProps(**base)


class _Workspace:
    """Minimal Numba-style scratch workspace expected by the strict gravity contract."""

    def __init__(self, nmax: int):
        n = nmax + 1
        self.P = np.zeros((n, n), dtype=np.float64)
        self.dP = np.zeros((n, n), dtype=np.float64)
        self.cos_m = np.zeros(n, dtype=np.float64)
        self.sin_m = np.zeros(n, dtype=np.float64)


def _good_gravity_model(nmax: int = 2):
    """A minimal object satisfying the strict classical-SH gravity contract."""
    n = nmax + 1

    class _G:
        degree_max = nmax
        R_ref_m = 1_737_400.0
        GM_m3s2 = 4.9048695e12
        Cnm = np.zeros((n, n), dtype=np.float64)
        Snm = np.zeros((n, n), dtype=np.float64)
        diag = np.zeros(n, dtype=np.float64)
        subdiag = np.zeros(n, dtype=np.float64)
        A = np.zeros(n, dtype=np.float64)
        B = np.zeros(n, dtype=np.float64)
        scale_m = np.ones(n, dtype=np.float64)
        ws = _Workspace(nmax)

    return _G()


class _StubSurrogate:
    model_kind = "st_lrps"
    R_ref_m = 1_737_400.0
    GM_m3s2 = 4.9048695e12

    def acceleration_fixed(self, r_fixed):
        return np.array([-1.0, 0.0, 0.0], dtype=np.float64)


# =============================================================================
# _validate_dependencies — required providers
# =============================================================================

def test_enable_sh_without_gravity_model_raises():
    with pytest.raises(ValueError, match="enable_sh=True but gravity_model is None"):
        DynamicsEngine(_sc(), PerturbationFlags(enable_sh=True),
                       gravity_model=None, allow_identity_rotation=True)


def test_enable_albedo_without_surface_provider_raises():
    with pytest.raises(ValueError, match="surface_provider is None"):
        DynamicsEngine(
            _sc(),
            PerturbationFlags(enable_sh=False, enable_albedo=True),
            surface_provider=None,
            allow_identity_rotation=True,
        )


@pytest.mark.parametrize("bad_sc", [
    _sc(area_m2=0.0),   # SRP/albedo need a positive optical area
    _sc(cr=0.0),        # cr must be in (0, 2.5]
    _sc(cr=3.0),
])
def test_srp_rejects_invalid_spacecraft_properties(bad_sc):
    # The spacecraft-property check fires before the ephemeris check, so no
    # ephemeris stub is required to reach it.
    with pytest.raises(ValueError):
        DynamicsEngine(bad_sc, PerturbationFlags(enable_sh=False, enable_srp=True),
                       allow_identity_rotation=True)


def test_third_body_sun_requires_ephemeris():
    with pytest.raises(ValueError, match="Ephemeris is required"):
        DynamicsEngine(
            _sc(),
            PerturbationFlags(enable_sh=False, enable_3rd_body_sun=True),
            ephem_manager=None,
            allow_identity_rotation=True,
        )


def test_earth_j2_without_params_raises():
    with pytest.raises(ValueError, match="earth_j2 params are None"):
        DynamicsEngine(
            _sc(),
            PerturbationFlags(enable_sh=False, enable_earth_j2=True),
            earth_j2=None,
            allow_identity_rotation=True,
        )


# =============================================================================
# allow_identity_rotation only substitutes the frame quaternion
# =============================================================================

def test_allow_identity_rotation_substitutes_quaternion_only():
    """With SH enabled and no ephemeris, the engine needs q_i2f. That single
    dependency may be replaced by identity, but only when explicitly allowed."""
    grav = object()  # non-None, non-surrogate -> use_sh path needs q

    # allow_identity_rotation=False -> q must come from ephemeris -> error.
    with pytest.raises(ValueError, match="q_i2f"):
        DynamicsEngine(_sc(), PerturbationFlags(enable_sh=True),
                       gravity_model=grav, ephem_manager=None,
                       allow_identity_rotation=False)

    # allow_identity_rotation=True -> identity q is accepted; construction succeeds.
    eng = DynamicsEngine(_sc(), PerturbationFlags(enable_sh=True),
                         gravity_model=grav, ephem_manager=None,
                         allow_identity_rotation=True)
    assert eng is not None


def test_allow_identity_rotation_does_not_substitute_sun_or_earth_vectors():
    """Sun/Earth *vectors* are physical inputs, not a frame convention: identity
    rotation must NOT paper over a missing ephemeris when a vector is required."""
    with pytest.raises(ValueError, match="Ephemeris is required"):
        DynamicsEngine(
            _sc(),
            PerturbationFlags(enable_sh=False, enable_3rd_body_sun=True),
            ephem_manager=None,
            allow_identity_rotation=True,  # cannot rescue a missing Sun vector
        )


# =============================================================================
# Unsupported or missing dependencies must fail loudly (not silently no-op)
# =============================================================================

def test_thermal_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="Thermal"):
        DynamicsEngine(_sc(), PerturbationFlags(enable_sh=False, enable_thermal=True),
                       allow_identity_rotation=True)


def test_solid_tides_require_ephemeris_vectors():
    # enable_sh disabled so the gravity-model check does not fire first.
    with pytest.raises(ValueError, match="Ephemeris is required"):
        DynamicsEngine(_sc(), PerturbationFlags(enable_sh=False, enable_tides_k2=True),
                       allow_identity_rotation=True)


# =============================================================================
# Surrogate-gravity provider detection
# =============================================================================

def test_surrogate_provider_detection():
    assert _is_surrogate_gravity_provider(_StubSurrogate()) is True
    assert _is_surrogate_gravity_provider(object()) is False
    assert _is_surrogate_gravity_provider(None) is False

    # An object claiming the kind but lacking acceleration_fixed is NOT a surrogate.
    class _Partial:
        model_kind = "st_lrps"
    assert _is_surrogate_gravity_provider(_Partial()) is False


def test_engine_routes_surrogate_provider_through_python_path():
    eng = DynamicsEngine(_sc(), PerturbationFlags(enable_sh=True),
                         gravity_model=_StubSurrogate(), ephem_manager=None,
                         allow_identity_rotation=True)
    req = eng._requirements()
    assert req["use_sh"] is True
    assert req["use_surrogate_gravity"] is True


# =============================================================================
# Strict classical-SH gravity contract (extract_gravity_strict)
# =============================================================================

def test_extract_gravity_strict_accepts_valid_model():
    nmax, r_ref, gm, *_ = extract_gravity_strict(_good_gravity_model(nmax=3))
    assert nmax == 3
    assert r_ref > 0.0 and gm > 0.0


def test_extract_gravity_strict_rejects_none():
    with pytest.raises(ValueError, match="gravity_model is None"):
        extract_gravity_strict(None)


def test_extract_gravity_strict_requires_degree_max():
    g = _good_gravity_model()
    delattr(type(g), "degree_max")
    with pytest.raises(AttributeError, match="degree_max"):
        extract_gravity_strict(g)


def test_extract_gravity_strict_requires_workspace():
    class _NoWs:
        degree_max = 2
        R_ref_m = 1_737_400.0
        GM_m3s2 = 4.9048695e12
        Cnm = np.zeros((3, 3))
        Snm = np.zeros((3, 3))
        diag = np.zeros(3)
        subdiag = np.zeros(3)
        A = np.zeros(3)
        B = np.zeros(3)
        scale_m = np.ones(3)

    with pytest.raises(AttributeError, match="ws.*make_workspace|workspace"):
        extract_gravity_strict(_NoWs())


def test_extract_gravity_strict_rejects_too_small_coeff_arrays():
    class _Small:
        degree_max = 5            # claims degree 5 but coeff arrays are 1x1
        R_ref_m = 1_737_400.0
        GM_m3s2 = 4.9048695e12
        Cnm = np.zeros((1, 1))
        Snm = np.zeros((1, 1))
        diag = np.zeros(1)
        subdiag = np.zeros(1)
        A = np.zeros(1)
        B = np.zeros(1)
        scale_m = np.ones(6)
        ws = _Workspace(5)

    with pytest.raises(ValueError, match="Cnm shape too small"):
        extract_gravity_strict(_Small())


@pytest.mark.parametrize("attr,value", [("GM_m3s2", -1.0), ("R_ref_m", 0.0)])
def test_extract_gravity_strict_rejects_nonpositive_scalars(attr, value):
    g = _good_gravity_model()
    setattr(type(g), attr, value)
    with pytest.raises(ValueError, match="must be positive"):
        extract_gravity_strict(g)
