# tests/test_state_geometry.py
# -*- coding: utf-8 -*-
"""
Orbital-geometry invariants for ``lunaris.core.state``.

Complements ``test_state.py`` by focusing on the cases most prone to *silent*
geometric errors:

- circular / equatorial *singular* orbits, where RAAN and argument-of-periapsis
  are undefined — here we compare reconstructed Cartesian state and physical
  invariants (h, energy) rather than forcing arbitrary angle conventions,
- near-circular round-trip stability,
- inclination / RAAN / argument wrapping into ``[0, 2*pi)``,
- rejection of parabolic / hyperbolic / negative-eccentricity / non-finite inputs,
- vis-viva energy consistency (eps == -mu / (2a)) for bound two-body orbits,
- altitude <-> (a, e) round-trips.

Lunar constants are used locally so the tests stay independent of the production
constant module's exact values.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from lunaris.core.state import (
    OrbitState,
    calculate_ae_from_altitudes,
    calculate_altitudes_from_ae,
    cartesian_to_keplerian,
    keplerian_to_cartesian,
)

MU_MOON = 4.90486959e12  # m^3/s^2
R_MOON = 1.7374e6        # m


def _specific_energy(r, v, mu):
    r = np.asarray(r, float)
    v = np.asarray(v, float)
    return 0.5 * float(v @ v) - mu / float(np.linalg.norm(r))


# =============================================================================
# Singular orbits: compare invariants, not undefined angles
# =============================================================================

def test_circular_equatorial_singularity_reconstructs_state_and_invariants():
    """e=0, i=0 leaves RAAN and argp undefined. The conversion must not blow up,
    and a COE -> Cartesian -> COE -> Cartesian loop must preserve the physical
    state and invariants regardless of how the singular angles are reported."""
    a = R_MOON + 200e3
    ta = math.radians(30.0)
    r, v = keplerian_to_cartesian(a, 0.0, 0.0, 0.0, 0.0, ta, mu=MU_MOON)

    els = cartesian_to_keplerian(r, v, mu=MU_MOON)
    a_out, e_out = els[0], els[1]
    assert a_out == pytest.approx(a, rel=1e-9)
    assert e_out == pytest.approx(0.0, abs=1e-9)

    # Reconstructed Cartesian state must match the original (angles may be reported
    # arbitrarily for the singular case, but the physical state cannot change).
    r2, v2 = keplerian_to_cartesian(*els, mu=MU_MOON)
    np.testing.assert_allclose(r2, r, rtol=1e-9, atol=1e-3)
    np.testing.assert_allclose(v2, v, rtol=1e-9, atol=1e-6)

    # Specific angular momentum and energy are frame-independent invariants.
    np.testing.assert_allclose(np.cross(r2, v2), np.cross(r, v), rtol=1e-9, atol=1e-3)
    assert _specific_energy(r2, v2, MU_MOON) == pytest.approx(
        _specific_energy(r, v, MU_MOON), rel=1e-9
    )


def test_polar_circular_orbit_has_zero_z_angular_momentum():
    """A circular polar orbit (i=90deg) in the x-z plane has h purely in-plane,
    so its z-component (which sets the cos(i) used for inclination) is ~0."""
    a = R_MOON + 100e3
    r, v = keplerian_to_cartesian(a, 0.0, math.radians(90.0), 0.0, 0.0, 0.0, mu=MU_MOON)
    h = np.cross(r, v)
    assert abs(h[2]) < 1e-3 * float(np.linalg.norm(h))
    _, _, inc, *_ = cartesian_to_keplerian(r, v, mu=MU_MOON)
    assert inc == pytest.approx(math.radians(90.0), abs=1e-9)


# =============================================================================
# Near-circular round-trip stability + angle wrapping
# =============================================================================

def test_near_circular_inclined_roundtrip_is_stable():
    a = R_MOON + 500e3
    e = 1e-6  # near-circular but not exactly singular
    inc = math.radians(63.4)
    raan = math.radians(135.0)
    argp = math.radians(200.0)
    ta = math.radians(47.0)

    r, v = keplerian_to_cartesian(a, e, inc, raan, argp, ta, mu=MU_MOON)
    a_o, e_o, inc_o, raan_o, argp_o, ta_o = cartesian_to_keplerian(r, v, mu=MU_MOON)

    assert a_o == pytest.approx(a, rel=1e-9)
    assert e_o == pytest.approx(e, abs=1e-7)
    assert inc_o == pytest.approx(inc, abs=1e-9)

    # Round-trip back to Cartesian must be physically identical.
    r2, v2 = keplerian_to_cartesian(a_o, e_o, inc_o, raan_o, argp_o, ta_o, mu=MU_MOON)
    np.testing.assert_allclose(r2, r, rtol=1e-9, atol=1e-3)
    np.testing.assert_allclose(v2, v, rtol=1e-9, atol=1e-6)


def test_returned_angles_are_wrapped_into_unit_circle():
    """All angular elements returned by cartesian_to_keplerian must live in
    [0, 2*pi) so downstream code never sees negative or >2*pi angles."""
    a = R_MOON + 800e3
    r, v = keplerian_to_cartesian(
        a, 0.2, math.radians(28.5), math.radians(300.0), math.radians(250.0),
        math.radians(310.0), mu=MU_MOON,
    )
    _, _, inc, raan, argp, ta = cartesian_to_keplerian(r, v, mu=MU_MOON)
    for ang in (inc, raan, argp, ta):
        assert 0.0 <= ang < 2.0 * math.pi


# =============================================================================
# Invalid conic sections / inputs
# =============================================================================

@pytest.mark.parametrize("a,e,match", [
    (1.0e7, 1.0, "Parabolic"),                 # parabolic singular in (a,e)
    (1.0e7, 1.5, "Hyperbolic"),                # e>1 with a>0 is inconsistent
    (1.0e7, -0.1, "Eccentricity"),             # negative eccentricity
    (-1.0e7, 0.5, "Elliptic"),                 # e<1 needs a>0
])
def test_invalid_conic_sections_are_rejected(a, e, match):
    with pytest.raises(ValueError, match=match):
        keplerian_to_cartesian(a, e, 0.0, 0.0, 0.0, 0.0, mu=MU_MOON)


def test_non_finite_and_nonpositive_mu_rejected():
    with pytest.raises(ValueError, match="finite"):
        keplerian_to_cartesian(math.nan, 0.1, 0.0, 0.0, 0.0, 0.0, mu=MU_MOON)
    with pytest.raises(ValueError):
        keplerian_to_cartesian(1.0e7, 0.1, 0.0, 0.0, 0.0, 0.0, mu=-1.0)


# =============================================================================
# Two-body energy consistency (vis-viva)
# =============================================================================

@pytest.mark.parametrize("e", [0.0, 0.1, 0.5, 0.85])
def test_specific_energy_matches_minus_mu_over_2a(e):
    a = R_MOON + 1_000e3
    state = OrbitState(*keplerian_to_cartesian(a, e, math.radians(30.0), 0.3, 0.4, 1.0, mu=MU_MOON))
    eps = state.compute_specific_energy(MU_MOON)
    assert eps == pytest.approx(-MU_MOON / (2.0 * a), rel=1e-9)
    assert eps < 0.0  # bound orbit


# =============================================================================
# Altitude <-> (a, e) round-trips
# =============================================================================

def test_altitude_to_ae_roundtrip_circular_and_elliptic():
    # Circular: hp == ha => e == 0, a == R_ref + h.
    a, e = calculate_ae_from_altitudes(R_MOON, 100.0, 100.0)
    assert e == pytest.approx(0.0, abs=1e-12)
    assert a == pytest.approx(R_MOON + 100e3, rel=1e-12)

    # Elliptic: round-trip preserves both altitudes.
    a, e = calculate_ae_from_altitudes(R_MOON, 80.0, 1500.0)
    assert 0.0 < e < 1.0
    hp, ha = calculate_altitudes_from_ae(R_MOON, a, e)
    assert hp == pytest.approx(80.0, rel=1e-9)
    assert ha == pytest.approx(1500.0, rel=1e-9)


def test_altitude_conversion_orders_periapsis_below_apoapsis():
    # Even if altitudes are passed swapped, (a, e) must describe a valid orbit
    # with periapsis <= apoapsis.
    a, e = calculate_ae_from_altitudes(R_MOON, 1500.0, 80.0)
    hp, ha = calculate_altitudes_from_ae(R_MOON, a, e)
    assert hp <= ha
