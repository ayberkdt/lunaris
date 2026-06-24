"""Solar radiation pressure and conical-eclipse geometry.

Pins the SRP cannonball model and the Moon/Earth conical shadow factor: magnitude
against the closed form, direction away from the Sun, inverse-square distance
scaling, shadow factor bounded in [0, 1], full sunlight on the day side, exact
zero deep in the umbra, and a continuous monotone penumbra transition. These are
the behaviours the Phase-1 review verified empirically; here they become
regression locks.
"""

from __future__ import annotations

import numpy as np
import pytest

from lunaris.common.constants import AU, P_SUN_1AU, R_EARTH_MEAN, R_MOON_MEAN
from lunaris.common.type_defs import SpacecraftProps
from lunaris.physics.solar_effects import (
    SRPConfig,
    accel_srp,
    compute_srp_accel,
    moon_shadow_factor_conical,
)

_CR, _AREA, _MASS = 1.8, 2.0, 12.0
_SUN = np.array([1.496e11, 0.0, 0.0])  # 1 AU on +X
_NO_EARTH = np.array([0.0, 0.0, 0.0])


def _srp(r, sun=_SUN, earth=_NO_EARTH, moon_ecl=True, earth_ecl=False):
    return np.array(
        accel_srp(
            r[0], r[1], r[2], sun[0], sun[1], sun[2], earth[0], earth[1], earth[2],
            R_MOON_MEAN, R_EARTH_MEAN, AU, P_SUN_1AU, _CR, _AREA, _MASS, moon_ecl, earth_ecl,
        )
    )


def test_full_sun_magnitude_matches_closed_form() -> None:
    r = np.array([R_MOON_MEAN + 100e3, 0.0, 0.0])  # day side, no eclipse
    a = _srp(r)
    d = np.linalg.norm(r - _SUN)
    expected = P_SUN_1AU * _CR * _AREA / _MASS * (AU / d) ** 2
    assert abs(np.linalg.norm(a) - expected) <= 1e-12 * expected


def test_acceleration_points_away_from_sun() -> None:
    r = np.array([R_MOON_MEAN + 300e3, 5.0e5, -2.0e5])
    a = _srp(r)
    sun_to_sc = r - _SUN
    u = sun_to_sc / np.linalg.norm(sun_to_sc)
    # a is parallel to the Sun->spacecraft direction (away from the Sun).
    assert np.dot(a / np.linalg.norm(a), u) > 1.0 - 1e-12


def test_inverse_square_distance_scaling() -> None:
    """|a| * d^2 is constant (d = true Sun->spacecraft distance), so the ratio of
    magnitudes equals (d1/d2)^2 exactly."""
    r = np.array([R_MOON_MEAN + 100e3, 0.0, 0.0])
    sun_far = 2.0 * _SUN
    a1 = _srp(r, sun=_SUN)
    a2 = _srp(r, sun=sun_far)
    d1 = np.linalg.norm(r - _SUN)
    d2 = np.linalg.norm(r - sun_far)
    assert np.linalg.norm(a2) / np.linalg.norm(a1) == pytest.approx((d1 / d2) ** 2, rel=1e-12)
    # |a| d^2 invariant.
    assert np.linalg.norm(a1) * d1**2 == pytest.approx(np.linalg.norm(a2) * d2**2, rel=1e-12)


def test_shadow_factor_is_bounded() -> None:
    rng = np.random.default_rng(3)
    for _ in range(500):
        r = rng.normal(scale=3.0e6, size=3)
        nu = moon_shadow_factor_conical(r[0], r[1], r[2], _SUN[0], _SUN[1], _SUN[2], R_MOON_MEAN)
        assert 0.0 <= nu <= 1.0


def test_day_side_is_full_sunlight() -> None:
    r = np.array([R_MOON_MEAN + 50e3, 0.0, 0.0])  # between Moon and Sun
    assert moon_shadow_factor_conical(r[0], r[1], r[2], _SUN[0], _SUN[1], _SUN[2], R_MOON_MEAN) == 1.0


def test_deep_umbra_zeros_the_force() -> None:
    r = np.array([-(R_MOON_MEAN + 50e3), 0.0, 0.0])  # directly behind the Moon
    nu = moon_shadow_factor_conical(r[0], r[1], r[2], _SUN[0], _SUN[1], _SUN[2], R_MOON_MEAN)
    assert nu == 0.0
    assert np.linalg.norm(_srp(r)) == 0.0


def test_penumbra_is_continuous_and_monotone() -> None:
    """Sweeping the perpendicular offset across the terminator behind the Moon,
    the shadow factor rises monotonically from 0 (umbra) to 1 (sunlight). Near the
    lunar surface the umbra and penumbra cones nearly coincide, so the penumbra is
    only ~tens of km wide; we sample it finely (~10 m steps) to confirm the
    smoothstep is genuinely continuous, not a hard step."""
    x_behind = -(R_MOON_MEAN + 200e3)
    rhos = np.linspace(1.70e6, 1.78e6, 8001)  # straddles the narrow penumbra band
    nus = np.array(
        [moon_shadow_factor_conical(x_behind, rho, 0.0, _SUN[0], _SUN[1], _SUN[2], R_MOON_MEAN) for rho in rhos]
    )
    assert np.all((nus >= 0.0) & (nus <= 1.0))
    assert np.all(np.diff(nus) >= -1e-12)  # monotone non-decreasing
    assert np.max(np.abs(np.diff(nus))) < 0.05  # continuous: no jumps at 10 m resolution
    assert nus[0] == 0.0 and nus[-1] == pytest.approx(1.0)  # umbra -> full sun
    assert np.any((nus > 0.05) & (nus < 0.95))  # genuinely passes through the penumbra


def test_zero_mass_or_area_returns_zero() -> None:
    r = np.array([R_MOON_MEAN + 100e3, 0.0, 0.0])
    assert accel_srp(r[0], r[1], r[2], *_SUN, *_NO_EARTH, R_MOON_MEAN, R_EARTH_MEAN, AU, P_SUN_1AU, _CR, _AREA, 0.0, True, False) == (0.0, 0.0, 0.0)
    assert accel_srp(r[0], r[1], r[2], *_SUN, *_NO_EARTH, R_MOON_MEAN, R_EARTH_MEAN, AU, P_SUN_1AU, _CR, 0.0, _MASS, True, False) == (0.0, 0.0, 0.0)


def test_disabling_moon_eclipse_removes_shadow() -> None:
    """With Moon eclipse off, a spacecraft in shadow still feels SRP (idealised)."""
    r = np.array([-(R_MOON_MEAN + 50e3), 0.0, 0.0])
    assert np.linalg.norm(_srp(r, moon_ecl=True)) == 0.0
    assert np.linalg.norm(_srp(r, moon_ecl=False)) > 0.0


def test_python_wrapper_matches_kernel() -> None:
    r = np.array([R_MOON_MEAN + 250e3, 1.0e5, -3.0e5])
    cfg = SRPConfig(enable_moon_eclipse=True, enable_earth_eclipse=False)
    sc = SpacecraftProps(mass_kg=_MASS, area_m2=_AREA, cr=_CR)
    a_wrap = compute_srp_accel(r, _SUN, _NO_EARTH, sc, cfg)
    np.testing.assert_allclose(a_wrap, _srp(r), rtol=1e-12, atol=0.0)
