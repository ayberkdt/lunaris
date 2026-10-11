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


# ---------------------------------------------------------------------------
# Penumbra = visible solar-disk fraction; Earth eclipse on by default
# ---------------------------------------------------------------------------


def _exact_visible_fraction(r: np.ndarray, sun: np.ndarray, radius: float, n: int = 20000) -> float:
    """Independent reference: solid-angle fraction of the solar cap (radius a
    about the Sun direction) outside the occulter cap (radius b about the body
    centre), integrated ring by ring with the exact azimuth fraction. Uniform
    solar radiance, spherical geometry - no planar approximation."""
    from lunaris.common.constants import R_SUN_MEAN

    d = sun - r
    a = np.arcsin(R_SUN_MEAN / np.linalg.norm(d))
    b = np.arcsin(min(1.0, radius / np.linalg.norm(r)))
    u_sun = d / np.linalg.norm(d)
    u_occ = -r / np.linalg.norm(r)
    c = np.arctan2(np.linalg.norm(np.cross(u_sun, u_occ)), float(np.dot(u_sun, u_occ)))
    th = (np.arange(n) + 0.5) * (a / n)
    w = np.sin(th)
    val = (np.cos(b) - np.cos(th) * np.cos(c)) / (np.sin(th) * np.sin(c))
    frac = np.arccos(np.clip(val, -1.0, 1.0)) / np.pi
    return float(1.0 - np.sum(w * frac) / np.sum(w))


@pytest.mark.parametrize(
    ("radius", "x_behind"),
    [
        (R_MOON_MEAN, 100e3),
        (R_MOON_MEAN, 1.0e7),
        (R_MOON_MEAN, 3.0e8),
        (R_EARTH_MEAN, 1.0e6),
        (R_EARTH_MEAN, 3.84e8),  # Earth's shadow at the Moon's distance
    ],
)
def test_penumbra_equals_exact_disk_overlap(radius: float, x_behind: float) -> None:
    """Across the whole penumbra band the shadow factor is the visible solar-disk
    fraction. The planar lens area (Montenbruck & Gill 2000, sec. 3.4) agrees with the
    exact spherical-cap integral to 3e-4 (measured; the planar approximation of
    the occulter limb dominates); 5e-4 is the budget. The former smoothstep was
    up to 0.046 off here."""
    from lunaris.common.constants import R_SUN_MEAN

    s = float(np.linalg.norm(_SUN))
    r_u = radius * (1.0 - x_behind / (radius * s / (R_SUN_MEAN - radius)))
    r_p = radius * (1.0 + x_behind / (radius * s / (R_SUN_MEAN + radius)))
    for rho in np.linspace(max(r_u, 0.0) * 1.0000001, r_p * 0.9999999, 41):
        r = np.array([-x_behind, rho, 0.0])
        nu = moon_shadow_factor_conical(r[0], r[1], r[2], _SUN[0], _SUN[1], _SUN[2], radius)
        assert abs(nu - _exact_visible_fraction(r, _SUN, radius)) < 5e-4


def test_annular_beyond_umbra_apex_is_closed_form() -> None:
    """Behind the lunar umbra apex (~3.7e8 m) on the shadow axis the Moon is a
    disk inside the solar disk: nu = 1 - b^2/a^2 with apparent radii a, b. The
    smoothstep returned ~0 (total eclipse) here."""
    from lunaris.common.constants import R_SUN_MEAN

    r = np.array([-1.0e9, 0.0, 0.0])
    a = np.arcsin(R_SUN_MEAN / np.linalg.norm(_SUN - r))
    b = np.arcsin(R_MOON_MEAN / 1.0e9)
    nu = moon_shadow_factor_conical(r[0], r[1], r[2], _SUN[0], _SUN[1], _SUN[2], R_MOON_MEAN)
    assert nu == pytest.approx(1.0 - (b / a) ** 2, rel=0.0, abs=1e-12)
    assert abs(nu - _exact_visible_fraction(r + np.array([0.0, 1.0, 0.0]), _SUN, R_MOON_MEAN)) < 5e-4


def test_srp_config_enables_earth_eclipse_by_default() -> None:
    assert SRPConfig().enable_earth_eclipse is True


def test_default_srp_is_zero_in_lunar_eclipse() -> None:
    """Earth between Sun and Moon (total lunar eclipse): a spacecraft on the
    Moon's sunward side is lit as far as the Moon is concerned but sits in the
    Earth's umbra, so the default SRP must vanish - as albedo/thermal IR do."""
    earth = np.array([3.844e8, 0.0, 0.0])
    sun = earth + np.array([1.496e11, 0.0, 0.0])
    r = np.array([R_MOON_MEAN + 100e3, 0.0, 0.0])  # Moon day side
    sc = SpacecraftProps(mass_kg=_MASS, area_m2=_AREA, cr=_CR)
    assert np.linalg.norm(compute_srp_accel(r, sun, earth, sc, SRPConfig())) == 0.0
    off = SRPConfig(enable_earth_eclipse=False)
    assert np.linalg.norm(compute_srp_accel(r, sun, earth, sc, off)) > 0.0


def test_zero_earth_vector_is_ignored_by_earth_eclipse() -> None:
    """Without an Earth ephemeris the Earth vector is zero; the Earth eclipse
    must then be skipped instead of putting the Earth at the Moon's centre
    (which would half-shadow a spacecraft on the terminator)."""
    r = np.array([0.0, R_MOON_MEAN + 100e3, 0.0])  # terminator, Moon eclipse off
    lit = _srp(r, moon_ecl=False, earth_ecl=False)
    gated = _srp(r, earth=_NO_EARTH, moon_ecl=False, earth_ecl=True)
    np.testing.assert_array_equal(gated, lit)
