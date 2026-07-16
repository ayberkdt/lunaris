# tests/test_relativity_effects.py
"""
Pytest suite for the 1PN Schwarzschild relativistic acceleration correction.

This file is migrated from the module-level self-test so it can run in CI.
"""

from __future__ import annotations

import math
import sys

import numpy as np
import pytest

from lunaris.common.constants import MU_SUN

# ---------------------------------------------------------------------------
# Import the module under test.
# ---------------------------------------------------------------------------
from lunaris.physics.relativity_effects import (
    C_SQ,
    EPS_1E12,
    MU_MOON,
    calc_external_1pn_accel,
    calc_schwarzschild_accel,
    de_sitter_components,
    external_1pn_components,
    external_schwarzschild_diff_components,
)

# ---------------------------------------------------------------------------
# Pure-Python reference implementation (no numba), same formula as kernel
# ---------------------------------------------------------------------------

def _schwarzschild_components_ref(
    rx: float, ry: float, rz: float,
    vx: float, vy: float, vz: float,
    mu: float,
) -> tuple[float, float, float]:
    """
    Pure-Python reference (no numba), same formula as the kernel.
    Used only for test comparisons.
    """
    r2 = rx * rx + ry * ry + rz * rz
    if r2 <= EPS_1E12:
        return 0.0, 0.0, 0.0

    r = math.sqrt(r2)
    v2 = vx * vx + vy * vy + vz * vz
    rv = rx * vx + ry * vy + rz * vz

    term_common = mu / (C_SQ * r2 * r)
    alpha = (4.0 * mu / r) - v2
    beta = 4.0 * rv

    ax = term_common * (alpha * rx + beta * vx)
    ay = term_common * (alpha * ry + beta * vy)
    az = term_common * (alpha * rz + beta * vz)
    return ax, ay, az


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session", autouse=True)
def _warmup_numba() -> None:
    """
    Trigger JIT compilation once per test session so timing/noise isn't repeated.
    """
    r_warm = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    v_warm = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    _ = calc_schwarzschild_accel(r_warm, v_warm, float(MU_MOON))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_calc_returns_float64_vec3() -> None:
    r = np.array([2.0e6, -1.0e6, 3.0e6], dtype=np.float64)
    v = np.array([1200.0, -800.0, 50.0], dtype=np.float64)

    a = calc_schwarzschild_accel(r, v, float(MU_MOON))

    assert isinstance(a, np.ndarray)
    assert a.shape == (3,)
    assert a.dtype == np.float64


def test_singularity_protection_near_zero_r() -> None:
    r0 = np.zeros(3, dtype=np.float64)
    v0 = np.array([1.0, 2.0, 3.0], dtype=np.float64)

    a0 = calc_schwarzschild_accel(r0, v0, float(MU_MOON))

    assert np.allclose(a0, 0.0), "Expected zero accel for near-zero position vector"


def test_v_zero_parallel_and_magnitude() -> None:
    # v = 0 case: accel must be parallel to r_vec and match closed-form magnitude
    r1 = np.array([1.2e6, -2.3e6, 0.7e6], dtype=np.float64)
    v1 = np.zeros(3, dtype=np.float64)

    a1 = calc_schwarzschild_accel(r1, v1, float(MU_MOON))

    # Parallel check: r x a ≈ 0
    cross = np.cross(r1, a1)
    # Scaled tolerance: avoid false fails when magnitudes are tiny
    assert np.linalg.norm(cross) <= 1e-12 * (np.linalg.norm(r1) * np.linalg.norm(a1) + 1.0), (
        "For v=0, accel should be parallel to r_vec"
    )

    # Magnitude check:
    # |a| = 4*mu^2 / (c^2 * r^3)  since a_vec = 4*mu^2/(c^2*r^4) * r_vec
    rmag = float(np.linalg.norm(r1))
    expected_mag = 4.0 * float(MU_MOON) ** 2 / (C_SQ * (rmag ** 3))
    got_mag = float(np.linalg.norm(a1))
    assert abs(got_mag - expected_mag) <= 1e-12 * max(expected_mag, 1.0), (
        f"v=0 magnitude mismatch: got {got_mag}, expected {expected_mag}"
    )


def test_reference_agreement() -> None:
    # Deterministic vectors (no randomness) to keep CI stable
    r2 = np.array([2.1e6, 1.7e6, -0.9e6], dtype=np.float64)
    v2 = np.array([-500.0, 1250.0, 200.0], dtype=np.float64)
    mu = float(MU_MOON)

    a_numba = calc_schwarzschild_accel(r2, v2, mu)

    axr, ayr, azr = _schwarzschild_components_ref(
        float(r2[0]), float(r2[1]), float(r2[2]),
        float(v2[0]), float(v2[1]), float(v2[2]),
        mu,
    )
    a_ref = np.array([axr, ayr, azr], dtype=np.float64)

    np.testing.assert_allclose(a_numba, a_ref, rtol=1e-13, atol=0.0)


def test_external_schwarzschild_differential_matches_subtraction() -> None:
    r = np.array([2.0e6, 1.0e5, -5.0e4], dtype=np.float64)
    v = np.array([25.0, 1580.0, -12.0], dtype=np.float64)
    body = np.array([1.495978707e11, 2.0e8, -1.0e8], dtype=np.float64)
    body_v = np.array([0.0, -29_780.0, 15.0], dtype=np.float64)
    mu = float(MU_SUN)

    got = np.array(
        external_schwarzschild_diff_components(
            float(r[0]), float(r[1]), float(r[2]),
            float(v[0]), float(v[1]), float(v[2]),
            float(body[0]), float(body[1]), float(body[2]),
            float(body_v[0]), float(body_v[1]), float(body_v[2]),
            mu,
        ),
        dtype=np.float64,
    )

    sc = np.array(_schwarzschild_components_ref(*(r - body), *(v - body_v), mu), dtype=np.float64)
    moon = np.array(_schwarzschild_components_ref(*(-body), *(-body_v), mu), dtype=np.float64)
    np.testing.assert_allclose(got, sc - moon, rtol=1e-13, atol=0.0)


def test_de_sitter_components_match_precession_formula() -> None:
    v = np.array([0.0, 1600.0, 20.0], dtype=np.float64)
    body = np.array([1.495978707e11, 0.0, 0.0], dtype=np.float64)
    body_v = np.array([0.0, -29_780.0, 0.0], dtype=np.float64)
    mu = float(MU_SUN)

    got = np.array(
        de_sitter_components(
            float(v[0]), float(v[1]), float(v[2]),
            float(body[0]), float(body[1]), float(body[2]),
            float(body_v[0]), float(body_v[1]), float(body_v[2]),
            mu,
        ),
        dtype=np.float64,
    )

    r_body_to_moon = -body
    v_body_to_moon = -body_v
    omega = 1.5 * mu * np.cross(r_body_to_moon, v_body_to_moon) / (
        C_SQ * np.linalg.norm(r_body_to_moon) ** 3
    )
    # Prograde geodetic precession: a = +2 (Omega x v).
    expected = 2.0 * np.cross(omega, v)
    np.testing.assert_allclose(got, expected, rtol=1e-13, atol=0.0)


def test_external_1pn_wrapper_matches_kernel_sum() -> None:
    r = np.array([2.1e6, -3.0e5, 4.0e4], dtype=np.float64)
    v = np.array([120.0, 1500.0, 45.0], dtype=np.float64)
    body = np.array([1.495978707e11, 3.0e8, 2.0e8], dtype=np.float64)
    body_v = np.array([-50.0, -29_700.0, 30.0], dtype=np.float64)
    mu = float(MU_SUN)

    wrapper = calc_external_1pn_accel(r, v, body, body_v, mu)
    kernel = np.array(
        external_1pn_components(
            float(r[0]), float(r[1]), float(r[2]),
            float(v[0]), float(v[1]), float(v[2]),
            float(body[0]), float(body[1]), float(body[2]),
            float(body_v[0]), float(body_v[1]), float(body_v[2]),
            mu,
        ),
        dtype=np.float64,
    )

    np.testing.assert_allclose(wrapper, kernel, rtol=0.0, atol=0.0)


if __name__ == "__main__":
    import sys

    print("This is a pytest test module. Run it with:")
    print("  python -m pytest -vv -rA --durations=10 tests/test_relativity_effects.py")
    sys.exit(0)


