# LUNAR_SIMULATION/tests/test_state.py
# -*- coding: utf-8 -*-
"""
Test Suite for Core State & Geometry Engine
===========================================

Validates the integrity of the 6-DOF Cartesian state vector, orbital element conversions,
and geometric utility helpers defined in `core.state`.

Run with:
    pytest tests/test_state.py -vv
"""

import math
import numpy as np
import sys
from pathlib import Path
import pytest
from numpy.testing import assert_allclose, assert_array_equal


# -----------------------------------------------------------------------------
# Import helper (run from repo root without installing)
# -----------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# Import the module under test
from core.state import (
    # Constants
    STATE_SIZE,
    
    # Containers
    OrbitState,
    ClassicalElements,
    
    # Packing
    pack_orbital_state,
    unpack_orbital_state,
    
    # Coordinate Transformations
    keplerian_to_cartesian,
    cartesian_to_keplerian,
    keplerian_to_state_vector,
    create_state_from_keplerian,
    
    # Geometry Helpers
    calculate_periapsis_apoapsis_radii,
    calculate_ae_from_radii,
    calculate_ae_from_altitudes,
    calculate_altitudes_from_ae
)

# --- Test Constants (Lunar Context) ---
MU_MOON = 4.90486959e12  # m^3/s^2
R_MOON  = 1.7374e6       # m


# =============================================================================
# 1. Packing & Unpacking Tests
# =============================================================================

def test_pack_unpack_integrity():
    """Verifies that packing and unpacking are lossless and shape-correct."""
    r_in = np.array([1000.0, 2000.0, 3000.0])
    v_in = np.array([-10.0, -20.0, -30.0])

    # Pack
    state_vec = pack_orbital_state(r_in, v_in)
    
    assert state_vec.shape == (STATE_SIZE,)
    assert state_vec.dtype == np.float64
    assert_array_equal(state_vec[:3], r_in)
    assert_array_equal(state_vec[3:], v_in)

    # Unpack (Copy)
    r_out, v_out = unpack_orbital_state(state_vec, copy=True)
    assert_array_equal(r_out, r_in)
    assert_array_equal(v_out, v_in)
    assert r_out is not state_vec[:3]  # Ensure it's a copy

    # Unpack (View)
    r_view, v_view = unpack_orbital_state(state_vec, copy=False)
    assert_array_equal(r_view, r_in)
    assert r_view.base is state_vec  # Ensure it's a view


def test_validation_helpers_reject_bad_inputs():
    """Ensures input validators catch NaNs and wrong shapes."""
    bad_vec = np.array([1.0, np.nan, 3.0])
    short_vec = np.array([1.0, 2.0])
    
    with pytest.raises(ValueError, match="non-finite"):
        pack_orbital_state(bad_vec, [0,0,0])
        
    with pytest.raises(ValueError, match="elements"):
        pack_orbital_state(short_vec, [0,0,0])


# =============================================================================
# 2. Geometric & Altitude Helper Tests
# =============================================================================

def test_elliptic_geometry_conversions():
    """Test round-trip conversion between (hp, ha) and (a, e)."""
    # Case: 100 km x 100 km circular orbit
    hp_in = 100.0
    ha_in = 100.0
    
    a, e = calculate_ae_from_altitudes(R_MOON, hp_in, ha_in)
    assert np.isclose(e, 0.0)
    assert np.isclose(a, R_MOON + 100_000.0)
    
    hp_out, ha_out = calculate_altitudes_from_ae(R_MOON, a, e)
    assert np.isclose(hp_out, hp_in)
    assert np.isclose(ha_out, ha_in)

    # Case: 100 km x 5000 km elliptic orbit
    hp_in = 100.0
    ha_in = 5000.0
    
    a, e = calculate_ae_from_altitudes(R_MOON, hp_in, ha_in)
    assert 0.0 < e < 1.0
    
    hp_out, ha_out = calculate_altitudes_from_ae(R_MOON, a, e)
    assert_allclose([hp_out, ha_out], [hp_in, ha_in], rtol=1e-12)


# =============================================================================
# 3. Coordinate Transformation Tests (COE <-> Cartesian)
# =============================================================================

def test_keplerian_cartesian_roundtrip():
    """
    Full round-trip test:
    COE -> Cartesian -> COE -> Cartesian
    Verifies that physics is preserved across coordinate frames.
    """
    # Define a standard elliptic orbit
    a_in    = R_MOON + 500e3  # 500 km altitude
    e_in    = 0.01
    inc_in  = math.radians(45.0)
    raan_in = math.radians(10.0)
    argp_in = math.radians(90.0)
    ta_in   = math.radians(180.0)

    # 1. COE -> Cartesian
    r, v = keplerian_to_cartesian(a_in, e_in, inc_in, raan_in, argp_in, ta_in, mu=MU_MOON)

    # 2. Cartesian -> COE
    a_out, e_out, inc_out, raan_out, argp_out, ta_out = cartesian_to_keplerian(
        r, v, mu=MU_MOON, wrap_angles=True
    )

    # 3. Compare Elements (Note: Angles can be tricky due to wrapping, but specific case is safe)
    assert_allclose(a_out, a_in, rtol=1e-12)
    assert_allclose(e_out, e_in, atol=1e-12)
    assert_allclose(inc_out, inc_in, atol=1e-12)
    
    # 4. Final verification: Convert back to Cartesian and compare vectors (Physical Truth)
    r2, v2 = keplerian_to_cartesian(a_out, e_out, inc_out, raan_out, argp_out, ta_out, mu=MU_MOON)
    
    assert_allclose(r2, r, rtol=1e-12, atol=1e-6)
    assert_allclose(v2, v, rtol=1e-12, atol=1e-9)


def test_singularity_guards():
    """Test that invalid orbits raise appropriate errors."""
    
    # Parabolic singularity (e=1.0)
    with pytest.raises(ValueError, match="Parabolic"):
        keplerian_to_cartesian(10000e3, 1.0, 0, 0, 0, 0)

    # Hyperbolic impact (e > 1 but a > 0 is invalid convention here)
    with pytest.raises(ValueError, match="Hyperbolic"):
        keplerian_to_cartesian(10000e3, 1.5, 0, 0, 0, 0) # Should utilize a < 0 for hyperbolic

    # Non-finite inputs
    with pytest.raises(ValueError, match="finite"):
        keplerian_to_cartesian(np.nan, 0, 0, 0, 0, 0)


# =============================================================================
# 4. High-Level Container Tests (OrbitState & ClassicalElements)
# =============================================================================

def test_orbit_state_behavior():
    """Test the OOP wrapper 'OrbitState'."""
    r = np.array([R_MOON + 1000.0, 0.0, 0.0])
    v = np.array([0.0, 1600.0, 0.0])
    
    state = OrbitState(r, v)
    
    # Check properties
    assert np.isclose(state.r_mag, R_MOON + 1000.0)
    assert np.isclose(state.v_mag, 1600.0)
    
    # Check energy
    expected_energy = 0.5 * 1600**2 - MU_MOON / (R_MOON + 1000.0)
    assert np.isclose(state.compute_specific_energy(MU_MOON), expected_energy)
    
    # Check y-property
    y = state.y
    assert y.shape == (6,)
    assert_array_equal(y[:3], r)


def test_classical_elements_behavior():
    """Test the OOP wrapper 'ClassicalElements'."""
    ce = ClassicalElements(
        a=8000e3, e=0.1, inc=0.5, raan=0.1, argp=0.2, ta=0.0
    )
    
    # Check normalization returns a new instance
    ce_norm = ce.normalized()
    assert isinstance(ce_norm, ClassicalElements)
    assert ce_norm is not ce
    
    # Check conversion to OrbitState
    os = ce.to_orbit_state(mu=MU_MOON)
    assert isinstance(os, OrbitState)
    assert os.r_mag > 0

# =============================================================================
# 5. Factory Tests
# =============================================================================

def test_create_state_factory():
    """Test the 'overloaded' factory function."""
    args = (8000e3, 0.0, 0.0, 0.0, 0.0, 0.0)
    
    # Return Object
    obj = create_state_from_keplerian(*args, mu=MU_MOON, return_array=False)
    assert isinstance(obj, OrbitState)
    
    # Return Array
    arr = create_state_from_keplerian(*args, mu=MU_MOON, return_array=True)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (6,)


if __name__ == "__main__":
    # Allow running this script directly for quick debugging
    import sys
    sys.exit(pytest.main(["-vv", __file__]))