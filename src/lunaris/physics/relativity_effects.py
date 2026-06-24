# ST_LRPS/models/relativity_effects.py
"""
Relativistic Effects: 1PN Corrections
=====================================

This module implements first post-Newtonian (1PN) acceleration corrections used
by the lunar propagation engine:

- the Schwarzschild term for motion about a single central body,
- differential Schwarzschild terms from external bodies, and
- the de Sitter/geodetic precession term from the Moon's motion in an external
  gravity field.

The implementation is split into:
- a small Numba-compiled scalar kernel (allocation-free), and
- thin wrappers for convenience and interoperability with the Python layer.

Design goals
------------
- Small, self-contained implementation with a clear, stable public API.
- High performance in tight loops:
  * Provide an allocation-free API (`calc_schwarzschild_accel_out`) suitable for
    integrator kernels.
  * Keep an ergonomic convenience wrapper (`calc_schwarzschild_accel`) that
    returns a new (3,) float64 array for non-hot paths.
- Consistent units and behavior:
  * Inputs: position [m], velocity [m/s], gravitational parameter mu [m^3/s^2]
  * Output: acceleration correction [m/s^2]
  * Near-zero radius is guarded (returns zero correction) to avoid singularities.
- Minimal runtime dependencies:
  * No SPICE calls; this is purely local physics given (r, v, mu).
  * Numba is used for speed; wrappers remain usable from pure Python.

Runtime vs. testing
-------------------
- This module intentionally avoids module-level self-tests. Verification is
  performed via pytest in:
    `tests/test_relativity_effects.py`

Scope / limitations
-------------------
- This model captures Schwarzschild and de Sitter/geodetic 1PN terms.
  It does not include:
  * frame-dragging (Lense–Thirring),
  * J2/oblateness relativistic couplings,
  * the full Einstein-Infeld-Hoffmann N-body equations,
  * time dilation / clock models.
"""



# =============================================================================
# 0.                               IMPORTS
# =============================================================================

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numba import njit

from lunaris.common.constants import C_LIGHT, EPS_1E12, MU_MOON
from lunaris.common.type_defs import Vec3

# 3) Pre-calculation (Module Level Constant)
C_SQ: float = C_LIGHT * C_LIGHT


# =============================================================================
# 1.                       COMPUTATIONAL KERNELS
# =============================================================================

@njit(cache=True, nogil=True, inline="always")
def _schwarzschild_components(
    rx: float, ry: float, rz: float,
    vx: float, vy: float, vz: float,
    mu: float,
) -> tuple[float, float, float]:
    """
    1PN Schwarzschild acceleration correction (alloc-free scalar kernel).

    a_rel = (mu / (c^2 * r^3)) * [ (4*mu/r - v^2)*r_vec + 4*(r_vec · v_vec)*v_vec ]
    """
    r2 = rx * rx + ry * ry + rz * rz
    if r2 <= EPS_1E12:
        return 0.0, 0.0, 0.0

    inv_r = 1.0 / math.sqrt(r2)          # 1/r
    v2 = vx * vx + vy * vy + vz * vz
    rv = rx * vx + ry * vy + rz * vz     # r · v

    # mu / (c^2 * r^3) = mu * inv_r / (c^2 * r^2)
    term_common = (mu * inv_r) / (C_SQ * r2)

    alpha = 4.0 * mu * inv_r - v2        # (4*mu/r - v^2)
    beta = 4.0 * rv                      # 4*(r·v)

    ax = term_common * (alpha * rx + beta * vx)
    ay = term_common * (alpha * ry + beta * vy)
    az = term_common * (alpha * rz + beta * vz)
    return ax, ay, az


@njit(cache=True, nogil=True, inline="always")
def _external_schwarzschild_diff_components(
    rx: float, ry: float, rz: float,
    vx: float, vy: float, vz: float,
    body_x: float, body_y: float, body_z: float,
    body_vx: float, body_vy: float, body_vz: float,
    mu_body: float,
) -> tuple[float, float, float]:
    """
    Differential external-body Schwarzschild correction in a Moon-centered frame.

    ``body_*`` is the external body's Moon-centered inertial position/velocity.
    The returned term is a_1PN(sc wrt body) - a_1PN(Moon wrt body), so it can
    be added directly to the relative spacecraft equations of motion.
    """
    if mu_body <= 0.0:
        return 0.0, 0.0, 0.0

    b2 = body_x * body_x + body_y * body_y + body_z * body_z
    if b2 <= EPS_1E12:
        return 0.0, 0.0, 0.0

    sc_ax, sc_ay, sc_az = _schwarzschild_components(
        rx - body_x,
        ry - body_y,
        rz - body_z,
        vx - body_vx,
        vy - body_vy,
        vz - body_vz,
        mu_body,
    )
    moon_ax, moon_ay, moon_az = _schwarzschild_components(
        -body_x,
        -body_y,
        -body_z,
        -body_vx,
        -body_vy,
        -body_vz,
        mu_body,
    )
    return sc_ax - moon_ax, sc_ay - moon_ay, sc_az - moon_az


@njit(cache=True, nogil=True, inline="always")
def _de_sitter_components(
    vx: float, vy: float, vz: float,
    body_x: float, body_y: float, body_z: float,
    body_vx: float, body_vy: float, body_vz: float,
    mu_body: float,
) -> tuple[float, float, float]:
    """
    de Sitter/geodetic precession acceleration for Moon-centered relative motion.

    The external body vector is Moon->body. Internally, R and V are the Moon
    state relative to that body. In GR,

        Omega_dS = 3/2 * mu_body / (c^2 |R|^3) * (R x V)

    where R, V are the Moon's position/velocity relative to the external body, so
    Omega_dS points along the Moon's orbital angular momentum about that body. The
    geodetic precession is PROGRADE: the contribution to the satellite equation of
    motion is the Coriolis-like term

        a = +2 * Omega_dS x v

    which makes the orbit plane precess at +Omega_dS (the canonical ~19.2 mas/yr
    for the Sun term). Note the sign: ``+2 Omega x v`` (prograde), not
    ``-2 Omega x v`` (which would precess the orbit retrograde).
    """
    if mu_body <= 0.0:
        return 0.0, 0.0, 0.0

    # R and V: external body -> Moon.
    rx_b = -body_x
    ry_b = -body_y
    rz_b = -body_z
    vx_b = -body_vx
    vy_b = -body_vy
    vz_b = -body_vz

    r2 = rx_b * rx_b + ry_b * ry_b + rz_b * rz_b
    if r2 <= EPS_1E12:
        return 0.0, 0.0, 0.0

    inv_r3 = 1.0 / (r2 * math.sqrt(r2))
    scale = 1.5 * mu_body * inv_r3 / C_SQ

    ox = scale * (ry_b * vz_b - rz_b * vy_b)
    oy = scale * (rz_b * vx_b - rx_b * vz_b)
    oz = scale * (rx_b * vy_b - ry_b * vx_b)

    # a = +2 * (Omega x v)  -> prograde geodetic precession at rate |Omega_dS|.
    ax = 2.0 * (oy * vz - oz * vy)
    ay = 2.0 * (oz * vx - ox * vz)
    az = 2.0 * (ox * vy - oy * vx)
    return ax, ay, az


@njit(cache=True, nogil=True, inline="always")
def _external_1pn_components(
    rx: float, ry: float, rz: float,
    vx: float, vy: float, vz: float,
    body_x: float, body_y: float, body_z: float,
    body_vx: float, body_vy: float, body_vz: float,
    mu_body: float,
) -> tuple[float, float, float]:
    """Combined external-body Schwarzschild differential + de Sitter terms."""
    sx, sy, sz = _external_schwarzschild_diff_components(
        rx, ry, rz,
        vx, vy, vz,
        body_x, body_y, body_z,
        body_vx, body_vy, body_vz,
        mu_body,
    )
    dx, dy, dz = _de_sitter_components(
        vx, vy, vz,
        body_x, body_y, body_z,
        body_vx, body_vy, body_vz,
        mu_body,
    )
    return sx + dx, sy + dy, sz + dz


@njit(cache=True, nogil=True, inline="always")
def calc_schwarzschild_accel_out(r_vec: np.ndarray, v_vec: np.ndarray, mu: float, out: np.ndarray) -> None:
    """
    Allocation-free API for tight loops: writes result into `out` (shape (3,)).
    """
    ax, ay, az = _schwarzschild_components(
        r_vec[0], r_vec[1], r_vec[2],
        v_vec[0], v_vec[1], v_vec[2],
        mu,
    )
    out[0] = ax
    out[1] = ay
    out[2] = az


@njit(cache=True, nogil=True)
def calc_schwarzschild_accel(r_vec: np.ndarray, v_vec: np.ndarray, mu: float) -> np.ndarray:
    """
    Convenience wrapper (allocates a (3,) array). Prefer *_out in integrator loops.
    """
    out = np.empty(3, dtype=np.float64)
    calc_schwarzschild_accel_out(r_vec, v_vec, mu, out)
    return out


def calc_external_1pn_accel(
    r_vec: Vec3,
    v_vec: Vec3,
    body_pos_m: Vec3,
    body_vel_m_s: Vec3,
    mu_body: float,
) -> Vec3:
    """Convenience wrapper for external-body Schwarzschild + de Sitter terms."""
    r = np.asarray(r_vec, dtype=np.float64)
    v = np.asarray(v_vec, dtype=np.float64)
    b = np.asarray(body_pos_m, dtype=np.float64)
    bv = np.asarray(body_vel_m_s, dtype=np.float64)
    ax, ay, az = _external_1pn_components(
        float(r[0]), float(r[1]), float(r[2]),
        float(v[0]), float(v[1]), float(v[2]),
        float(b[0]), float(b[1]), float(b[2]),
        float(bv[0]), float(bv[1]), float(bv[2]),
        float(mu_body),
    )
    return np.asarray((ax, ay, az), dtype=np.float64)



# =============================================================================
# 2.                        MODEL INTERFACE
# =============================================================================

@dataclass(slots=True, frozen=True)
class RelativityModel:
    """
    Relativistic correction model (1PN Schwarzschild convenience wrapper).

    Note: This is a Python-side convenience wrapper; the Numba loop should call
    the njit kernels directly for best performance. External-body 1PN terms are
    provided as standalone kernels because they require ephemeris body states.
    """
    mu: float = MU_MOON  # [m^3/s^2]

    def __post_init__(self) -> None:
        object.__setattr__(self, "mu", float(self.mu))

    def compute_accel(self, r_vec: Vec3, v_vec: Vec3) -> Vec3:
        out = np.empty(3, dtype=np.float64)
        calc_schwarzschild_accel_out(np.asarray(r_vec, dtype=np.float64),
                                     np.asarray(v_vec, dtype=np.float64),
                                     self.mu, out)
        return out



# =============================================================================
# 3.                            PUBLIC API
# =============================================================================

__all__ = (
    # --- Core kernels ---
    "calc_schwarzschild_accel",       # Convenience wrapper (allocates (3,) ndarray)

    "calc_schwarzschild_accel_out",   # Allocation-free: writes into `out` (shape (3,))

    "calc_external_1pn_accel",        # External-body Schwarzschild + de Sitter wrapper

    "_external_schwarzschild_diff_components",

    "_de_sitter_components",

    "_external_1pn_components",

    # --- Model interface ---
    "RelativityModel",                # Convenience wrapper class (holds mu)

)
