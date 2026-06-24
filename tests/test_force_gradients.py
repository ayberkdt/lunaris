"""Finite-difference gradient validation for the conservative force kernels.

A passing unit test proves the code runs; these tests prove the *physics*: every
conservative perturbation must satisfy ``a = +grad(U)`` (the geodesy sign
convention Lunaris uses, confirmed against the spherical-harmonics kernel) and,
being a gradient field, must be curl-free. We compare each analytic acceleration
kernel against a central finite difference of its own potential and against the
symmetry of its Jacobian.

This is the gate the ``astrodynamics-validation`` skill requires before trusting
a potential→acceleration derivation: analytic vs numerical gradient must agree to
~1e-6 relative in float64.
"""

from __future__ import annotations

import numpy as np
import pytest

from lunaris.common.constants import MU_EARTH, MU_MOON, MU_SUN, R_EARTH_EQUATORIAL, R_MOON
from lunaris.physics.solid_tides import (
    solid_tide_accel_degree_numba,
    solid_tide_potential_degree_numba,
)
from lunaris.physics.spherical_harmonics import GravityModel
from lunaris.physics.third_body_effects import _accel_j2_oblate_unit_k

# Earth J2 nominal (WGS-84-ish), reused across tests.
_J2_EARTH = 1.082_626_68e-3


def _central_grad(potential, r: np.ndarray, h: float) -> np.ndarray:
    """Central finite difference of a scalar potential at r [m]."""
    g = np.empty(3, dtype=np.float64)
    for i in range(3):
        rp = r.copy()
        rm = r.copy()
        rp[i] += h
        rm[i] -= h
        g[i] = (potential(rp) - potential(rm)) / (2.0 * h)
    return g


def _accel_jacobian(accel, r: np.ndarray, h: float) -> np.ndarray:
    """Central-difference Jacobian J[i,j] = d a_i / d x_j of an acceleration field."""
    J = np.empty((3, 3), dtype=np.float64)
    for j in range(3):
        rp = r.copy()
        rm = r.copy()
        rp[j] += h
        rm[j] -= h
        ap = np.asarray(accel(rp), dtype=np.float64)
        am = np.asarray(accel(rm), dtype=np.float64)
        J[:, j] = (ap - am) / (2.0 * h)
    return J


# -----------------------------------------------------------------------------
# Solid-body tides (explicit potential + analytic gradient in the same module)
# -----------------------------------------------------------------------------
_TIDE_BODIES = {
    "earth": (np.array([384_400e3, 5.0e7, -2.0e7]), MU_EARTH),
    "sun": (np.array([0.4e11, 1.3e11, 0.5e11]), MU_SUN),
}


@pytest.mark.parametrize("body", list(_TIDE_BODIES))
@pytest.mark.parametrize("degree,k_l", [(2, 0.02416), (3, 0.0089)])
def test_solid_tide_accel_is_gradient_of_potential(body: str, degree: int, k_l: float) -> None:
    b, mu = _TIDE_BODIES[body]
    r = np.array([R_MOON + 120e3, 3.0e5, -1.5e5], dtype=np.float64)

    def U(rr: np.ndarray) -> float:
        return solid_tide_potential_degree_numba(
            rr[0], rr[1], rr[2], b[0], b[1], b[2], mu, R_MOON, k_l, degree
        )

    a = np.array(
        solid_tide_accel_degree_numba(
            r[0], r[1], r[2], b[0], b[1], b[2], mu, R_MOON, k_l, degree
        )
    )
    g = _central_grad(U, r, h=2.0)
    assert np.linalg.norm(a - g) <= 1e-6 * np.linalg.norm(g)


def test_solid_tide_field_is_curl_free() -> None:
    """A gradient field has a symmetric Jacobian (d a_i/d x_j == d a_j/d x_i)."""
    b, mu = _TIDE_BODIES["earth"]
    r = np.array([R_MOON + 200e3, -4.0e5, 2.2e5], dtype=np.float64)

    def accel(rr: np.ndarray):
        return solid_tide_accel_degree_numba(
            rr[0], rr[1], rr[2], b[0], b[1], b[2], mu, R_MOON, 0.02416, 2
        )

    J = _accel_jacobian(accel, r, h=5.0)
    asym = np.max(np.abs(J - J.T))
    assert asym <= 1e-6 * np.max(np.abs(J))


# -----------------------------------------------------------------------------
# Earth J2 (single-body acceleration vs its zonal potential)
# -----------------------------------------------------------------------------
def test_earth_j2_accel_is_gradient_of_zonal_potential() -> None:
    k = np.array([0.0, 0.0, 1.0])
    r = np.array([7.0e6, 2.0e6, 3.0e6], dtype=np.float64)

    def U(rr: np.ndarray) -> float:
        rn = float(np.linalg.norm(rr))
        rk = float(rr @ k)
        # a = +grad(U) convention: U_J2 = -(mu J2 R^2 / 2) (3 (r.k)^2 - r^2)/r^5
        return -(MU_EARTH * _J2_EARTH * R_EARTH_EQUATORIAL**2 / 2.0) * (3.0 * rk * rk - rn * rn) / rn**5

    a = np.array(
        _accel_j2_oblate_unit_k(
            r[0], r[1], r[2], MU_EARTH, R_EARTH_EQUATORIAL, _J2_EARTH, k[0], k[1], k[2]
        )
    )
    g = _central_grad(U, r, h=50.0)
    assert np.linalg.norm(a - g) <= 1e-6 * np.linalg.norm(g)


def test_earth_j2_field_is_curl_free() -> None:
    k = np.array([0.0, 0.0, 1.0])
    r = np.array([6.9e6, -1.5e6, 2.4e6], dtype=np.float64)

    def accel(rr: np.ndarray):
        return _accel_j2_oblate_unit_k(
            rr[0], rr[1], rr[2], MU_EARTH, R_EARTH_EQUATORIAL, _J2_EARTH, k[0], k[1], k[2]
        )

    J = _accel_jacobian(accel, r, h=100.0)
    assert np.max(np.abs(J - J.T)) <= 1e-6 * np.max(np.abs(J))


# -----------------------------------------------------------------------------
# Spherical-harmonic gravity limiting case
# -----------------------------------------------------------------------------
def test_sh_reduces_to_point_mass_with_zero_coefficients() -> None:
    """With every non-central coefficient zero, SH gravity must equal -mu r/|r|^3."""
    degree = 6
    c = np.zeros((degree + 1, degree + 1), dtype=np.float64)
    s = np.zeros_like(c)
    model = GravityModel.from_arrays(degree, R_MOON, MU_MOON, c, s)

    for r in (
        np.array([R_MOON + 150e3, 4.0e5, -2.0e5]),
        np.array([R_MOON + 1_000e3, 0.0, 0.0]),
        np.array([0.0, R_MOON + 80e3, 0.0]),
    ):
        a = np.array(model.accel_fixed(r, degree=degree))
        a_pm = -MU_MOON * r / np.linalg.norm(r) ** 3
        assert np.linalg.norm(a - a_pm) <= 1e-12 * np.linalg.norm(a_pm)


def test_sh_zonal_field_is_axially_symmetric() -> None:
    """A zonal-only (m=0) field must be invariant under rotation about the polar
    axis: rotating the evaluation point in longitude rotates the acceleration by
    the same angle and leaves its components in the rotating frame unchanged."""
    degree = 4
    c = np.zeros((degree + 1, degree + 1), dtype=np.float64)
    s = np.zeros_like(c)
    c[2, 0] = -9.0e-5  # J2-like zonal term only
    c[4, 0] = 3.0e-6
    model = GravityModel.from_arrays(degree, R_MOON, MU_MOON, c, s)

    r0 = np.array([R_MOON + 200e3, 0.0, 120e3], dtype=np.float64)
    a0 = np.array(model.accel_fixed(r0, degree=degree))

    theta = 0.7
    Rz = np.array(
        [[np.cos(theta), -np.sin(theta), 0.0],
         [np.sin(theta), np.cos(theta), 0.0],
         [0.0, 0.0, 1.0]]
    )
    a_rot = np.array(model.accel_fixed(Rz @ r0, degree=degree))
    # Acceleration at the rotated point must be the rotated acceleration.
    assert np.linalg.norm(a_rot - Rz @ a0) <= 1e-10 * np.linalg.norm(a0)
