# ST_LRPS/models/relativity_effects.py
"""
Relativistic Effects: Schwarzschild 1PN Correction
==================================================

This module implements the Schwarzschild (spherically symmetric) first
post-Newtonian (1PN) acceleration correction for motion about a single central
body.

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
- This model captures only the Schwarzschild 1PN term for a single central body.
  It does not include:
  * frame-dragging (Lense–Thirring),
  * J2/oblateness relativistic couplings,
  * multi-body post-Newtonian effects,
  * time dilation / clock models.
"""



# =============================================================================
# 0.                               IMPORTS
# =============================================================================

from __future__ import annotations

import math
import numpy as np

from typing import Tuple
from dataclasses import dataclass



from lunaris.common.constants import C_LIGHT, MU_MOON, EPS_1E12
from lunaris.common.type_defs import Vec3


from numba import njit

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
) -> Tuple[float, float, float]:
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



# =============================================================================
# 2.                        MODEL INTERFACE
# =============================================================================

@dataclass(slots=True, frozen=True)
class RelativityModel:
    """
    Relativistic correction model (1PN Schwarzschild).

    Note: This is a Python-side convenience wrapper; the Numba loop should call
    the njit kernels directly for best performance.
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

    # --- Model interface ---
    "RelativityModel",                # Convenience wrapper class (holds mu)

)