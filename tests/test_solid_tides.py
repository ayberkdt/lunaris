from __future__ import annotations

import math

import numpy as np
import pytest

from lunaris.common.constants import MU_EARTH, MU_MOON, R_MOON
from lunaris.common.math_utils import quat_rotate_vec
from lunaris.common.type_defs import PerturbationFlags, SolidTideConfig, SpacecraftProps
from lunaris.core.dynamics import DynamicsEngine
from lunaris.physics.solid_tides import (
    accel_solid_tides_numba,
    calc_solid_tide_accel,
    legendre_p2,
    legendre_p2_derivative,
    legendre_p3,
    legendre_p3_derivative,
    solid_tide_accel_degree_numba,
    solid_tide_potential_degree,
)


def _norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))


def _rot(q: tuple[float, float, float, float], v: np.ndarray) -> np.ndarray:
    x, y, z = quat_rotate_vec(q[0], q[1], q[2], q[3], float(v[0]), float(v[1]), float(v[2]))
    return np.asarray((x, y, z), dtype=np.float64)


def _mock_ephem(*, q: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)):
    class _MockEphem:
        def get_data_provider(self):
            return {
                "dt_s": 60.0,
                "r_sun_tab_m": np.asarray([[149_600_000_000.0, 2.0e8, -1.0e8]], dtype=np.float64),
                "r_earth_tab_m": np.asarray([[384_400_000.0, 1.0e6, -2.0e6]], dtype=np.float64),
                "q_i2f_tab": np.asarray([q, q], dtype=np.float64),
            }

    return _MockEphem()


def test_legendre_p2_p3_and_derivatives() -> None:
    c = 0.37
    assert float(legendre_p2(c)) == pytest.approx(0.5 * (3.0 * c * c - 1.0))
    assert float(legendre_p2_derivative(c)) == pytest.approx(3.0 * c)
    assert float(legendre_p3(c)) == pytest.approx(0.5 * (5.0 * c**3 - 3.0 * c))
    assert float(legendre_p3_derivative(c)) == pytest.approx(0.5 * (15.0 * c * c - 3.0))


def test_zero_love_numbers_give_exact_zero_acceleration() -> None:
    r = np.asarray([R_MOON + 120_000.0, 10_000.0, -5_000.0], dtype=np.float64)
    earth = np.asarray([384_400_000.0, 1.0e6, 0.0], dtype=np.float64)
    a = calc_solid_tide_accel(
        r,
        earth,
        mu_body=MU_EARTH,
        r_ref_m=R_MOON,
        k2=0.0,
        k3=0.0,
        use_k2=True,
        use_k3=True,
    )
    np.testing.assert_array_equal(a, np.zeros(3, dtype=np.float64))


def test_acceleration_is_finite_for_realistic_low_lunar_orbit() -> None:
    r = np.asarray([R_MOON + 100_000.0, 25_000.0, -8_000.0], dtype=np.float64)
    earth = np.asarray([384_400_000.0, 0.0, 0.0], dtype=np.float64)
    a = calc_solid_tide_accel(r, earth, mu_body=MU_EARTH, r_ref_m=R_MOON, k2=0.02416)

    assert np.all(np.isfinite(a))
    assert _norm(a) > 0.0

    central = MU_MOON / float(np.dot(r, r))
    ratio = _norm(a) / central
    assert 1.0e-10 < ratio < 1.0e-4


@pytest.mark.parametrize("degree,k_l", [(2, 0.02416), (3, 0.01)])
def test_analytical_gradient_matches_finite_difference(degree: int, k_l: float) -> None:
    r = np.asarray([R_MOON + 160_000.0, 83_000.0, -41_000.0], dtype=np.float64)
    earth = np.asarray([384_400_000.0, 1.7e6, -2.1e6], dtype=np.float64)

    ax, ay, az = solid_tide_accel_degree_numba(
        float(r[0]),
        float(r[1]),
        float(r[2]),
        float(earth[0]),
        float(earth[1]),
        float(earth[2]),
        float(MU_EARTH),
        float(R_MOON),
        float(k_l),
        int(degree),
    )
    analytic = np.asarray((ax, ay, az), dtype=np.float64)

    h = 5.0
    fd = np.zeros(3, dtype=np.float64)
    for i in range(3):
        step = np.zeros(3, dtype=np.float64)
        step[i] = h
        up = solid_tide_potential_degree(
            r + step,
            earth,
            mu_body=MU_EARTH,
            r_ref_m=R_MOON,
            k_l=k_l,
            degree=degree,
        )
        um = solid_tide_potential_degree(
            r - step,
            earth,
            mu_body=MU_EARTH,
            r_ref_m=R_MOON,
            k_l=k_l,
            degree=degree,
        )
        fd[i] = (up - um) / (2.0 * h)

    np.testing.assert_allclose(analytic, fd, rtol=2.0e-5, atol=1.0e-12)


def test_frame_consistency_under_common_rotation() -> None:
    q = (math.cos(0.31), 0.0, 0.0, math.sin(0.31))
    r = np.asarray([R_MOON + 180_000.0, 70_000.0, 45_000.0], dtype=np.float64)
    earth = np.asarray([384_400_000.0, -1.0e6, 3.0e6], dtype=np.float64)

    a = calc_solid_tide_accel(r, earth, mu_body=MU_EARTH, r_ref_m=R_MOON, k2=0.02416)
    a_expected = _rot(q, a)
    a_rotated_inputs = calc_solid_tide_accel(
        _rot(q, r),
        _rot(q, earth),
        mu_body=MU_EARTH,
        r_ref_m=R_MOON,
        k2=0.02416,
    )

    np.testing.assert_allclose(a_rotated_inputs, a_expected, rtol=2.0e-14, atol=1.0e-18)


def test_dynamics_tides_k2_no_longer_raise_not_implemented() -> None:
    sc = SpacecraftProps(mass_kg=12.0, area_m2=0.08, cr=1.3)
    flags = PerturbationFlags(enable_sh=False, enable_tides_k2=True)
    engine = DynamicsEngine(
        sc_props=sc,
        flags=flags,
        ephem_manager=_mock_ephem(),
        solid_tides=SolidTideConfig(tide_bodies=("earth",), k2=0.02416),
    )

    rhs = engine.build_rhs(force_rebuild=True)
    y = np.asarray([R_MOON + 100_000.0, 0.0, 0.0, 0.0, 1_600.0, 0.0], dtype=np.float64)
    dy = rhs(0.0, y)

    assert dy.shape == y.shape
    assert np.all(np.isfinite(dy))
    breakdown = engine.get_acceleration_breakdown(0.0, y)
    assert breakdown["Solid Tides (Earth)"] > 0.0


def test_dynamics_k3_requires_explicit_love_number() -> None:
    sc = SpacecraftProps(mass_kg=12.0, area_m2=0.08, cr=1.3)
    flags = PerturbationFlags(enable_sh=False, enable_tides_k2=True, enable_tides_k3=True)

    with pytest.raises(ValueError, match="k3"):
        DynamicsEngine(
            sc_props=sc,
            flags=flags,
            ephem_manager=_mock_ephem(),
            solid_tides=SolidTideConfig(tide_bodies=("earth",), k2=0.02416),
        )


def test_dynamics_k3_runs_when_love_number_is_configured() -> None:
    sc = SpacecraftProps(mass_kg=12.0, area_m2=0.08, cr=1.3)
    flags = PerturbationFlags(enable_sh=False, enable_tides_k2=True, enable_tides_k3=True)
    engine = DynamicsEngine(
        sc_props=sc,
        flags=flags,
        ephem_manager=_mock_ephem(),
        solid_tides=SolidTideConfig(tide_bodies=("earth",), k2=0.02416, k3=0.01),
    )
    rhs = engine.build_rhs(force_rebuild=True)
    y = np.asarray([R_MOON + 150_000.0, 20_000.0, 0.0, 0.0, 1_590.0, 0.0], dtype=np.float64)
    assert np.all(np.isfinite(rhs(0.0, y)))


def test_disabled_tide_flags_preserve_point_mass_rhs() -> None:
    sc = SpacecraftProps(mass_kg=12.0, area_m2=0.08, cr=1.3)
    flags = PerturbationFlags(enable_sh=False)
    y = np.asarray([R_MOON + 120_000.0, 0.0, 0.0, 0.0, 1_600.0, 0.0], dtype=np.float64)

    rhs_a = DynamicsEngine(sc_props=sc, flags=flags, allow_identity_rotation=True).build_rhs(force_rebuild=True)
    rhs_b = DynamicsEngine(
        sc_props=sc,
        flags=flags,
        solid_tides=SolidTideConfig(tide_bodies=("earth",), k2=0.0),
        allow_identity_rotation=True,
    ).build_rhs(force_rebuild=True)

    np.testing.assert_array_equal(rhs_a(0.0, y), rhs_b(0.0, y))


def test_accel_kernel_can_sum_k2_and_k3() -> None:
    r = np.asarray([R_MOON + 140_000.0, 1_000.0, 2_000.0], dtype=np.float64)
    earth = np.asarray([384_400_000.0, 0.0, 0.0], dtype=np.float64)
    a2 = calc_solid_tide_accel(r, earth, mu_body=MU_EARTH, r_ref_m=R_MOON, k2=0.02416)
    ax, ay, az = accel_solid_tides_numba(
        float(r[0]),
        float(r[1]),
        float(r[2]),
        float(earth[0]),
        float(earth[1]),
        float(earth[2]),
        float(MU_EARTH),
        float(R_MOON),
        0.02416,
        0.0,
        True,
        True,
    )
    np.testing.assert_allclose(np.asarray((ax, ay, az)), a2, rtol=0.0, atol=0.0)
