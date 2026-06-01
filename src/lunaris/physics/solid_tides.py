# lunaris/physics/solid_tides.py
"""
Elastic lunar solid-body tides
==============================

This module implements the lunar solid-body tide acceleration raised by an
external body (Earth or Sun) from the standard Love-number disturbing
potential. It intentionally models only the instantaneous elastic response:
no time lag, dissipation, ocean tide, or thermal tide terms are included.

For tide-raising body j, spacecraft Moon-fixed vector r, body Moon-fixed vector
R_j, lunar reference radius R, and Love number k_l:

    dU_l = k_l * mu_j / |R_j| * (R / |r|)^(l+1) * (R / |R_j|)^l * P_l(c)

where c = r_hat dot e_j and e_j = R_j / |R_j|. The acceleration convention used
by Lunaris spherical-harmonic gravity is the gradient of this positive
potential, so the returned tide acceleration is grad_r(dU_l):

    grad dU_l =
        A_l * rho^(-(l+2)) *
        [ P_l'(c) * (e_j - c r_hat) - (l + 1) * P_l(c) * r_hat ]

with A_l = k_l * mu_j * R^(2l+1) / |R_j|^(l+1). The implementation evaluates the
same expression using dimensionless radius ratios to avoid large intermediate
powers.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import numpy.typing as npt
from numba import njit

from lunaris.common.constants import R_MOON
from lunaris.common.type_defs import Vec3


_MIN_NORM2 = 1.0


@njit(cache=True, nogil=True, inline="always")
def legendre_p2(c: float) -> float:
    """Degree-2 Legendre polynomial P2(c)."""
    return 0.5 * (3.0 * c * c - 1.0)


@njit(cache=True, nogil=True, inline="always")
def legendre_p2_derivative(c: float) -> float:
    """Derivative dP2/dc."""
    return 3.0 * c


@njit(cache=True, nogil=True, inline="always")
def legendre_p3(c: float) -> float:
    """Degree-3 Legendre polynomial P3(c)."""
    return 0.5 * (5.0 * c * c * c - 3.0 * c)


@njit(cache=True, nogil=True, inline="always")
def legendre_p3_derivative(c: float) -> float:
    """Derivative dP3/dc."""
    return 0.5 * (15.0 * c * c - 3.0)


@njit(cache=True, nogil=True, inline="always")
def _clamp_unit(c: float) -> float:
    if c > 1.0:
        return 1.0
    if c < -1.0:
        return -1.0
    return c


@njit(cache=True, nogil=True)
def solid_tide_potential_degree_numba(
    rx: float,
    ry: float,
    rz: float,
    bx: float,
    by: float,
    bz: float,
    mu_body: float,
    r_ref_m: float,
    k_l: float,
    degree: int,
) -> float:
    """
    Disturbing potential dU_l for l=2 or l=3.

    Inputs are Moon-centered vectors in the same frame, normally Moon-fixed.
    Returns zero for disabled/degenerate cases and for unsupported degrees.
    """
    if mu_body == 0.0 or k_l == 0.0 or r_ref_m <= 0.0:
        return 0.0
    if degree != 2 and degree != 3:
        return 0.0

    rho2 = rx * rx + ry * ry + rz * rz
    d2 = bx * bx + by * by + bz * bz
    if rho2 <= _MIN_NORM2 or d2 <= _MIN_NORM2:
        return 0.0

    rho = math.sqrt(rho2)
    d = math.sqrt(d2)

    inv_rho = 1.0 / rho
    inv_d = 1.0 / d

    c = (rx * bx + ry * by + rz * bz) * inv_rho * inv_d
    c = _clamp_unit(c)

    if degree == 2:
        p_l = legendre_p2(c)
        r_over_rho = r_ref_m * inv_rho
        r_over_d = r_ref_m * inv_d
        return k_l * mu_body * inv_d * (r_over_rho ** 3) * (r_over_d ** 2) * p_l

    p_l = legendre_p3(c)
    r_over_rho = r_ref_m * inv_rho
    r_over_d = r_ref_m * inv_d
    return k_l * mu_body * inv_d * (r_over_rho ** 4) * (r_over_d ** 3) * p_l


@njit(cache=True, nogil=True)
def solid_tide_accel_degree_numba(
    rx: float,
    ry: float,
    rz: float,
    bx: float,
    by: float,
    bz: float,
    mu_body: float,
    r_ref_m: float,
    k_l: float,
    degree: int,
) -> Tuple[float, float, float]:
    """
    Analytical gradient of the degree-l tide potential for l=2 or l=3.

    Returned acceleration is in the same frame as the inputs.
    """
    if mu_body == 0.0 or k_l == 0.0 or r_ref_m <= 0.0:
        return 0.0, 0.0, 0.0
    if degree != 2 and degree != 3:
        return 0.0, 0.0, 0.0

    rho2 = rx * rx + ry * ry + rz * rz
    d2 = bx * bx + by * by + bz * bz
    if rho2 <= _MIN_NORM2 or d2 <= _MIN_NORM2:
        return 0.0, 0.0, 0.0

    rho = math.sqrt(rho2)
    d = math.sqrt(d2)
    inv_rho = 1.0 / rho
    inv_d = 1.0 / d

    rhx = rx * inv_rho
    rhy = ry * inv_rho
    rhz = rz * inv_rho

    ex = bx * inv_d
    ey = by * inv_d
    ez = bz * inv_d

    c = rhx * ex + rhy * ey + rhz * ez
    c = _clamp_unit(c)

    r_over_d = r_ref_m * inv_d
    r_over_rho = r_ref_m * inv_rho

    if degree == 2:
        p_l = legendre_p2(c)
        dp_l = legendre_p2_derivative(c)
        scale = k_l * mu_body / rho2 * (r_over_d ** 3) * (r_over_rho ** 2)
        degree_plus_one = 3.0
    else:
        p_l = legendre_p3(c)
        dp_l = legendre_p3_derivative(c)
        scale = k_l * mu_body / rho2 * (r_over_d ** 4) * (r_over_rho ** 3)
        degree_plus_one = 4.0

    tx = dp_l * (ex - c * rhx) - degree_plus_one * p_l * rhx
    ty = dp_l * (ey - c * rhy) - degree_plus_one * p_l * rhy
    tz = dp_l * (ez - c * rhz) - degree_plus_one * p_l * rhz

    return scale * tx, scale * ty, scale * tz


@njit(cache=True, nogil=True)
def accel_solid_tides_numba(
    rx: float,
    ry: float,
    rz: float,
    bx: float,
    by: float,
    bz: float,
    mu_body: float,
    r_ref_m: float,
    k2: float,
    k3: float,
    use_k2: bool,
    use_k3: bool,
) -> Tuple[float, float, float]:
    """
    Sum enabled degree-2 and degree-3 elastic solid-tide accelerations.

    ``k2 == 0`` and ``k3 == 0`` produce exactly zero contributions because the
    corresponding degree is skipped before any floating-point algebra is done.
    """
    ax = 0.0
    ay = 0.0
    az = 0.0

    if use_k2 and k2 != 0.0:
        a2x, a2y, a2z = solid_tide_accel_degree_numba(
            rx, ry, rz, bx, by, bz, mu_body, r_ref_m, k2, 2
        )
        ax += a2x
        ay += a2y
        az += a2z

    if use_k3 and k3 != 0.0:
        a3x, a3y, a3z = solid_tide_accel_degree_numba(
            rx, ry, rz, bx, by, bz, mu_body, r_ref_m, k3, 3
        )
        ax += a3x
        ay += a3y
        az += a3z

    return ax, ay, az


def _as_vec3(x: npt.ArrayLike, name: str) -> Vec3:
    arr = np.asarray(x, dtype=np.float64)
    if arr.shape != (3,):
        if arr.size == 3:
            arr = arr.reshape(3,)
        else:
            raise ValueError(f"{name} must have shape (3,), got {arr.shape}.")
    return arr


def solid_tide_potential_degree(
    r_sc: npt.ArrayLike,
    r_body: npt.ArrayLike,
    *,
    mu_body: float,
    r_ref_m: float = R_MOON,
    k_l: float,
    degree: int,
) -> float:
    """Python wrapper for ``solid_tide_potential_degree_numba``."""
    r = _as_vec3(r_sc, "r_sc")
    b = _as_vec3(r_body, "r_body")
    return float(
        solid_tide_potential_degree_numba(
            float(r[0]),
            float(r[1]),
            float(r[2]),
            float(b[0]),
            float(b[1]),
            float(b[2]),
            float(mu_body),
            float(r_ref_m),
            float(k_l),
            int(degree),
        )
    )


def calc_solid_tide_accel(
    r_sc: npt.ArrayLike,
    r_body: npt.ArrayLike,
    *,
    mu_body: float,
    r_ref_m: float = R_MOON,
    k2: float = 0.02416,
    k3: float = 0.0,
    use_k2: bool = True,
    use_k3: bool = False,
) -> Vec3:
    """Convenience wrapper returning a newly allocated (3,) acceleration array."""
    r = _as_vec3(r_sc, "r_sc")
    b = _as_vec3(r_body, "r_body")
    ax, ay, az = accel_solid_tides_numba(
        float(r[0]),
        float(r[1]),
        float(r[2]),
        float(b[0]),
        float(b[1]),
        float(b[2]),
        float(mu_body),
        float(r_ref_m),
        float(k2),
        float(k3),
        bool(use_k2),
        bool(use_k3),
    )
    return np.array((ax, ay, az), dtype=np.float64)


__all__ = (
    "legendre_p2",
    "legendre_p2_derivative",
    "legendre_p3",
    "legendre_p3_derivative",
    "solid_tide_potential_degree_numba",
    "solid_tide_accel_degree_numba",
    "accel_solid_tides_numba",
    "solid_tide_potential_degree",
    "calc_solid_tide_accel",
)
