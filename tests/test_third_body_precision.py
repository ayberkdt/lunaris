"""Precision / conditioning regression for the third-body differential gravity kernel.

This suite locks in the cancellation-free (Battin ``F(q)``) formulation adopted by
:func:`lunaris.physics.third_body_effects.accel_third_body_numba`. The physical
quantity is the differential (tidal) acceleration in a Moon-centred frame,

    a = mu * [ (r_tb - r_sc) / |r_tb - r_sc|^3  -  r_tb / |r_tb|^3 ],

which, for a lunar orbiter, is the small difference of two large, nearly-equal
vectors (``|r_sc| / |r_tb|`` ~ 5e-3 for Earth, ~1e-5 for Sun). Evaluating it as a
literal subtraction loses several significant digits; the kernel must instead
stay at machine precision.

Reference strategy
------------------
The high-precision reference is built with the standard-library :mod:`decimal`
module at 50 significant digits. ``Decimal(float(x))`` captures the *exact* binary
value the float64 kernel actually receives, so the comparison isolates the
kernel's numerical conditioning rather than input rounding. No third-party
high-precision dependency is required.
"""

from __future__ import annotations

import math
from decimal import Decimal, getcontext

import numpy as np
import pytest

from lunaris.common.constants import MU_EARTH, MU_SUN
from lunaris.physics.third_body_effects import accel_third_body_numba

getcontext().prec = 50

# Representative Moon-centred geometries [m]. Earth ~3.84e8 m, Sun ~1.5e11 m away;
# the Sun case is the worst-conditioned (smallest |r_sc|/|r_tb|).
_GEOMETRIES = {
    "earth_axial": (
        np.array([1_737_400.0 + 100e3, 0.0, 0.0]),
        np.array([384_400e3, 0.0, 0.0]),
        MU_EARTH,
    ),
    "earth_oblique": (
        np.array([1.2e6, 1.0e6, 0.8e6]),
        np.array([2.7e8, 2.5e8, 0.5e8]),
        MU_EARTH,
    ),
    "sun_oblique": (
        np.array([1_737_400.0 + 50e3, 3.0e5, -2.0e5]),
        np.array([0.5 * 1.496e11, 1.3e11, 0.4e11]),
        MU_SUN,
    ),
    "high_orbit_earth": (
        np.array([1.0e7, 2.0e6, -3.0e6]),
        np.array([3.0e8, 2.4e8, 0.4e8]),
        MU_EARTH,
    ),
}


def _reference_diff_accel(r: np.ndarray, s: np.ndarray, mu: float) -> list[Decimal]:
    """50-digit Decimal reference using the literal differential definition on the
    *exact* float64 inputs."""
    R = [Decimal(float(v)) for v in r]
    S = [Decimal(float(v)) for v in s]
    MU = Decimal(float(mu))
    D = [S[i] - R[i] for i in range(3)]
    d_norm = (D[0] * D[0] + D[1] * D[1] + D[2] * D[2]).sqrt()
    s_norm = (S[0] * S[0] + S[1] * S[1] + S[2] * S[2]).sqrt()
    d3 = d_norm * d_norm * d_norm
    s3 = s_norm * s_norm * s_norm
    return [MU * (D[i] / d3 - S[i] / s3) for i in range(3)]


def _relerr(a, ref: list[Decimal]) -> float:
    num = (
        (Decimal(float(a[0])) - ref[0]) ** 2
        + (Decimal(float(a[1])) - ref[1]) ** 2
        + (Decimal(float(a[2])) - ref[2]) ** 2
    ).sqrt()
    den = (ref[0] * ref[0] + ref[1] * ref[1] + ref[2] * ref[2]).sqrt()
    return float(num / den)


def _direct_difference_f64(r: np.ndarray, s: np.ndarray, mu: float) -> np.ndarray:
    """The pre-change naive form, kept here only as a conditioning baseline."""
    dx, dy, dz = s[0] - r[0], s[1] - r[1], s[2] - r[2]
    d2 = dx * dx + dy * dy + dz * dz
    b2 = s[0] * s[0] + s[1] * s[1] + s[2] * s[2]
    inv_d3 = 1.0 / (d2 * math.sqrt(d2))
    inv_b3 = 1.0 / (b2 * math.sqrt(b2))
    return np.array(
        [
            mu * (dx * inv_d3 - s[0] * inv_b3),
            mu * (dy * inv_d3 - s[1] * inv_b3),
            mu * (dz * inv_d3 - s[2] * inv_b3),
        ]
    )


@pytest.mark.parametrize("name", list(_GEOMETRIES))
def test_matches_high_precision_reference(name: str) -> None:
    """Deployed kernel must agree with the 50-digit reference to ~1e-13 relative,
    i.e. it must NOT lose the leading digits to cancellation."""
    r, s, mu = _GEOMETRIES[name]
    a = accel_third_body_numba(
        float(r[0]), float(r[1]), float(r[2]),
        float(s[0]), float(s[1]), float(s[2]),
        float(mu),
    )
    ref = _reference_diff_accel(r, s, mu)
    assert _relerr(a, ref) < 1e-13


def test_better_conditioned_than_direct_difference_on_sun_term() -> None:
    """Regression guard: anyone reverting to the literal subtraction would
    reintroduce a ~1e-11 error on the Sun term. The deployed kernel must be at
    least 100x closer to the reference than the naive difference."""
    r, s, mu = _GEOMETRIES["sun_oblique"]
    ref = _reference_diff_accel(r, s, mu)
    err_deployed = _relerr(
        accel_third_body_numba(
            float(r[0]), float(r[1]), float(r[2]),
            float(s[0]), float(s[1]), float(s[2]),
            float(mu),
        ),
        ref,
    )
    err_direct = _relerr(_direct_difference_f64(r, s, mu), ref)
    assert err_deployed < err_direct / 100.0
    # And the deployed form is essentially at machine precision.
    assert err_deployed < 1e-13


def test_reduces_to_one_dimensional_closed_form() -> None:
    """Collinear geometry on +X (0 < r < b): a_x = mu (1/(b-r)^2 - 1/b^2)."""
    mu = float(MU_EARTH)
    b = 384_400e3
    for r_x in (100e3, 300e3, 1.0e6, 5.0e6):
        a = accel_third_body_numba(r_x, 0.0, 0.0, b, 0.0, 0.0, mu)
        expected = mu * (1.0 / (b - r_x) ** 2 - 1.0 / b**2)
        assert abs(a[0] - expected) <= 1e-13 * abs(expected)
        assert abs(a[1]) < 1e-18
        assert abs(a[2]) < 1e-18
        # Differential pull is toward the body for a spacecraft inside the orbit.
        assert a[0] > 0.0


def test_degenerate_geometry_returns_exact_zero() -> None:
    """Sub-metre separations hit the singularity guard and must return (0,0,0)."""
    assert accel_third_body_numba(0.1, 0.0, 0.0, 0.2, 0.0, 0.0, float(MU_EARTH)) == (0.0, 0.0, 0.0)
    # Degenerate third-body distance.
    assert accel_third_body_numba(1.0e6, 0.0, 0.0, 1.0e6, 0.0, 0.0, float(MU_SUN)) == (0.0, 0.0, 0.0)


def test_zero_mu_returns_zero() -> None:
    r, s, _ = _GEOMETRIES["earth_axial"]
    a = accel_third_body_numba(float(r[0]), float(r[1]), float(r[2]), float(s[0]), float(s[1]), float(s[2]), 0.0)
    assert a == (0.0, 0.0, 0.0)
