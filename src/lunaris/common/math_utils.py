# LUNAR_SIMULATION/common/math_utils.py
# -*- coding: utf-8 -*-
"""
Numba-Accelerated Math Utilities
================================

Numba is a required dependency for this module. The functions are written around
JIT-compiled kernels used by the propagation and post-processing pipeline.

Low-level mathematical primitives used across the project. This module is intentionally
independent of higher-level layers (SPICE, file I/O, UI).

Contents
--------
1) Core Tiny Math
   - Small, JIT-friendly helpers such as dot/norm and scalar utilities.

2) Quaternion Arithmetic (scalar-first)
   - Convention: q = [w, x, y, z] where w is the scalar part.
   - Normalization, conjugation, Hamilton product, vector rotation, and SLERP.
   - Implemented as Numba-friendly scalar kernels plus NumPy convenience wrappers.

3) Table-Based Interpolation Kernels
   - SLERP interpolation for quaternion tables.
   - Catmull–Rom spline interpolation for 3D vector tables.
   - For ephemeris/attitude smoothing, resampling, and visualization.

4) Orbital Mechanics (RV -> COE)
   - Conversion from Cartesian state (r, v) to Classical Orbital Elements (COE).
   - Robust handling of singularities (circular and/or equatorial cases).
   - Includes batch conversion with a Numba-parallel kernel for post-processing.

5) Physics Helpers
   - Nyquist-based step-size recommendation for high-degree spherical harmonics
     to reduce aliasing of short-wavelength gravity features.

6) Scalar & Geometry Utilities
   - Domain-protected acos, scalar clamp, angle/longitude wrapping,
     and Cartesian -> (lat, lon, r) conversion for body-fixed coordinates.

7) Grid Samplers
   - Nearest/bilinear samplers for 2D grids (clamp latitude, wrap longitude).
   - Lat/Lon -> grid sampling helper for regularly gridded planetary products.
   - Optional scale/offset + missing-value aware variants for planetary rasters.

Design Goals
------------
- Performance: numerically hot paths are JIT-compiled via @njit(cache=True) when available.
- Robustness: explicit guards for floating-point edge cases (acos domain, near-zero norms).
- Stable surface: a small public API (see __all__) and pure, deterministic behavior.

Constants & Tolerances
----------------------
- Mathematical constants and numeric epsilons are centralized in `common.constants`.
- Numeric epsilons follow scale-explicit naming:
    EPS_1E12, EPS_1E15, EPS2_1E18, ...
  Use EPS2_* for squared-scale guards (e.g., r^2 near-origin).

Conventions
-----------
- Quaternions: scalar-first [w, x, y, z].
- Units: SI (meters, seconds, radians) unless explicitly stated (e.g., *_deg, *_km).
- Arrays:
    * Vectors are float64 arrays of shape (3,).
    * State stacks are shape (6, N) with rows [rx, ry, rz, vx, vy, vz].

Usage
-----
    >>> import numpy as np
    >>> from lunaris.common.math_utils import quat_rotate_np, rv_to_coe_select
    >>> q = np.array([1.0, 0.0, 0.0, 0.0])  # identity quaternion (w,x,y,z)
    >>> v = np.array([1.0, 0.0, 0.0])
    >>> v_rot = quat_rotate_np(q, v)

    >>> r_vec = np.array([1737.4e3 + 100e3, 0.0, 0.0])
    >>> v_vec = np.array([0.0, 1600.0, 0.0])
    >>> mu = 4.9048695e12
    >>> a, e, inc, raan, argp, nu = rv_to_coe_select(r_vec, v_vec, mu, mode="coe6")
"""



# =============================================================================
# 0.                                 IMPORTS
# =============================================================================
from __future__ import annotations

import math
import numpy as np
from typing import Literal, Tuple

from numba import njit, prange  

from .constants import (RAD2DEG, TWO_PI,
                        EPS_1E12, EPS_1E15, EPS_1E18, EPS_1E30)


# =============================================================================
# 1.                              SMALL HELPERS
# =============================================================================

# Used by: state
@njit(cache=True)
def norm3(ax: float, ay: float, az: float) -> float:
    """Return Euclidean norm of a 3D vector: sqrt(ax^2 + ay^2 + az^2)."""
    return math.sqrt(ax*ax + ay*ay + az*az)


# Used by: math_utils, surface_effects, dynamics
@njit(cache=True)
def clamp(x: float, lo: float, hi: float) -> float:
    """
    Clamp a scalar to a closed interval.

    Returns:
        lo if x < lo, hi if x > hi, otherwise x.
    """
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


# Used by: math_utils, state
@njit(cache=True)
def wrap_angle_2pi(angle_rad: float) -> float:
    """
    Wrap an angle [rad] into [0, 2π).

    Rationale
    ---------
    Orbital-element angles are periodic. Using a canonical interval:
    - prevents discontinuities in plots / statistics
    - simplifies downstream comparisons (e.g., RAAN drift)

    Examples
    --------
    -0.1  -> 2π - 0.1
     6.30 -> ~0.0168
    """
    # With a positive modulus, Python/Numba's % returns a value in [0, 2π).
    return angle_rad % TWO_PI



# =============================================================================
# 2.                    QUATERNION KERNELS (Scalar First)
# =============================================================================
# Convention: q = [q0, q1, q2, q3] where q0 is scalar (real) part.

# Used by: ephemeris
@njit(cache=True)
def quat_conj(q0: float, q1: float, q2: float, q3: float) -> Tuple[float, float, float, float]:
    """
    Returns the conjugate of a quaternion.

    For unit quaternions, conjugate equals inverse.
    """
    return q0, -q1, -q2, -q3


# Used by: math_utils, ephemeris, dynamics, postprocess
@njit(cache=True)
def quat_rotate_vec(q0: float, q1: float, q2: float, q3: float,
                    vx: float, vy: float, vz: float) -> Tuple[float, float, float]:
    """
    Rotates a 3D vector v by a unit quaternion q (scalar-first).

    Uses an optimized Rodrigues-like formula:

        t  = 2 * (q_vec × v)
        v' = v + q0 * t + (q_vec × t)

    This avoids explicitly forming q * [0,v] * q_conj.
    """
    # t = 2 * (q_vec x v)
    tx = 2.0 * (q2*vz - q3*vy)
    ty = 2.0 * (q3*vx - q1*vz)
    tz = 2.0 * (q1*vy - q2*vx)

    # (q_vec x t)
    cx = q2*tz - q3*ty
    cy = q3*tx - q1*tz
    cz = q1*ty - q2*tx

    vpx = vx + q0*tx + cx
    vpy = vy + q0*ty + cy
    vpz = vz + q0*tz + cz
    return vpx, vpy, vpz


# Used by: propagator
def quat_rotate_np(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Convenience wrapper for rotating a NumPy 3-vector by a NumPy quaternion.

    This is intentionally *not* JIT-compiled:
    - It performs dtype casting / shape normalization (NumPy operations).
    - Heavy math is already in the JIT kernel `quat_rotate_vec`.
    """
    q = np.asarray(q, dtype=np.float64).ravel()
    v = np.asarray(v, dtype=np.float64).ravel()

    if q.size != 4:
        raise ValueError(f"quat_rotate_np: expected q with 4 elements, got {q.size}")
    if v.size != 3:
        raise ValueError(f"quat_rotate_np: expected v with 3 elements, got {v.size}")

    x, y, z = quat_rotate_vec(
        float(q[0]), float(q[1]), float(q[2]), float(q[3]),
        float(v[0]), float(v[1]), float(v[2])
    )
    out = np.empty(3, dtype=np.float64)
    out[0] = x
    out[1] = y
    out[2] = z
    return out


# Used by: math_utils
@njit(cache=True)
def _quat_normalize_kernel(q0: float, q1: float, q2: float, q3: float) -> Tuple[float, float, float, float]:
    """
    Normalizes a quaternion to unit length.

    Notes
    -----
    - Returns identity quaternion (1,0,0,0) if the norm is too small, preventing NaN/Inf.
    - Includes a fast path if the quaternion is already unit length within tolerance,
      avoiding an expensive sqrt() in stable cases.
    """
    n2 = q0*q0 + q1*q1 + q2*q2 + q3*q3

    # Fast check: avoid sqrt if already nearly unit length
    if abs(n2 - 1.0) < EPS_1E12:
        return q0, q1, q2, q3
    
    # Safety against division by ~0
    if n2 < EPS_1E30:
        return 1.0, 0.0, 0.0, 0.0
    
    n = math.sqrt(n2)

    inv = 1.0 / n
    return q0*inv, q1*inv, q2*inv, q3*inv


# Used by: math_utils
@njit(cache=True)
def _quat_slerp(qA0: float, qA1: float, qA2: float, qA3: float,
               qB0: float, qB1: float, qB2: float, qB3: float,
               t: float) -> Tuple[float, float, float, float]:
    """
    Spherical Linear Interpolation (SLERP) between two unit quaternions.

    Features
    --------
    - Shortest path: if dot(A,B) < 0, flips B to avoid long rotation.
    - Robust acos: clamps dot into [-1,1] to avoid domain errors.
    - Small-angle optimization: uses LERP + normalize when quaternions are nearly identical.

    Parameters
    ----------
    qA : Start quaternion (t=0)
    qB : End quaternion   (t=1)
    t  : Interpolation factor in [0,1]
    """
    # Fast-path for endpoints (avoids trig and guarantees exact endpoint output)
    if t <= 0.0:
        return qA0, qA1, qA2, qA3
    if t >= 1.0:
        return qB0, qB1, qB2, qB3

    dot = qA0*qB0 + qA1*qB1 + qA2*qB2 + qA3*qB3

    # Ensure shortest arc
    if dot < 0.0:
        qB0 = -qB0
        qB1 = -qB1
        qB2 = -qB2
        qB3 = -qB3
        dot = -dot

    # Keep pre-clamp dot for threshold decision (avoids masking tiny numeric issues)
    dot_original = dot

    # Clamp to valid acos domain
    if dot > 1.0:
        dot = 1.0
    elif dot < -1.0:
        dot = -1.0

    # If extremely close: LERP + normalize
    SLERP_THRESHOLD = 0.9995
    if dot_original > SLERP_THRESHOLD:
        q0 = qA0 + t*(qB0 - qA0)
        q1 = qA1 + t*(qB1 - qA1)
        q2 = qA2 + t*(qB2 - qA2)
        q3 = qA3 + t*(qB3 - qA3)
        return _quat_normalize_kernel(q0, q1, q2, q3)

    # Standard SLERP
    theta0 = math.acos(dot)
    sin0 = math.sin(theta0)

    # Safety against division by ~0 (should be rare due to threshold, but protects edge cases)
    if sin0 < EPS_1E12:
        q0 = qA0 + t*(qB0 - qA0)
        q1 = qA1 + t*(qB1 - qA1)
        q2 = qA2 + t*(qB2 - qA2)
        q3 = qA3 + t*(qB3 - qA3)
        return _quat_normalize_kernel(q0, q1, q2, q3)

    # s0 and s1 are the interpolation weights
    # (classic form; numerically clean and easy to audit)
    s0 = math.sin((1.0 - t) * theta0) / sin0
    s1 = math.sin(t * theta0) / sin0

    q0 = s0*qA0 + s1*qB0
    q1 = s0*qA1 + s1*qB1
    q2 = s0*qA2 + s1*qB2
    q3 = s0*qA3 + s1*qB3

    return _quat_normalize_kernel(q0, q1, q2, q3)


# Used by: propagator
def quat_slerp_np(qA: np.ndarray, qB: np.ndarray, t: float) -> np.ndarray:
    """
    Convenience wrapper for SLERP between two NumPy quaternion arrays.
    """
    qA = np.asarray(qA, dtype=np.float64).ravel()
    qB = np.asarray(qB, dtype=np.float64).ravel()

    if qA.size != 4:
        raise ValueError(f"quat_slerp_np: expected qA with 4 elements, got {qA.size}")
    if qB.size != 4:
        raise ValueError(f"quat_slerp_np: expected qB with 4 elements, got {qB.size}")

    q0, q1, q2, q3 = _quat_slerp(
        float(qA[0]), float(qA[1]), float(qA[2]), float(qA[3]),
        float(qB[0]), float(qB[1]), float(qB[2]), float(qB[3]),
        float(t)
    )
    out = np.empty(4, dtype=np.float64)
    out[0] = q0
    out[1] = q1
    out[2] = q2
    out[3] = q3
    return out



# =============================================================================
# 3.                         TABLE INTERPOLATIONS 
# =============================================================================

# Used by: math_utils
@njit(cache=True)
def _table_index_frac(t: float, dt: float, n: int) -> Tuple[int, float]:
    """
    Converts continuous time `t` to a discrete table segment index and a fractional coordinate.

    Intended use
    ------------
    Shared by constant-step table interpolators that assume:
    - constant time step `dt`
    - samples stored in rows 0..n-1
    - interpolation uses rows i and i+1 (so i must be in [0, n-2])

    Notes
    -----
    - This helper clamps i into [0, n-2] and f into [0,1].
    - If dt <= 0 or n < 2, returns (0, 0.0) as a safe fallback.

    Returns
    -------
    i : int
        Base row index for interpolation, clamped to [0, n-2]
    f : float
        Fraction within [0,1] such that:
          t ≈ (i + f) * dt
    """
    if dt <= 0.0 or n < 2:
        return 0, 0.0

    u = t / dt

    # Using floor-like behavior is safer for negative u (even if callers clamp t).
    i = int(math.floor(u))

    # Clamp i so that i+1 is valid
    if i < 0:
        i = 0
    elif i > n - 2:
        i = n - 2

    # Fractional part
    f = u - i

    # Numeric safety: keep in [0,1] against floating noise
    if f < 0.0:
        f = 0.0
    elif f > 1.0:
        f = 1.0

    return i, f


@njit(cache=True)
def _table_endpoint_index(t: float, dt: float, n: int) -> int:
    """Returns endpoint index for constant-step tables, or -1 if interpolation is needed.

    Rules (shared across table interpolators)
    ----------------------------------------
    - n == 1, dt <= 0, t <= 0  -> 0
    - t >= dt*(n-1)            -> n-1
    - otherwise                -> -1
    """
    if n <= 0:
        return -1  # defensive; callers normally handle n==0 upstream
    if n == 1 or dt <= 0.0 or t <= 0.0:
        return 0
    tmax = dt * (n - 1)
    if t >= tmax:
        return n - 1
    return -1


# Used by: ephemeris
@njit(cache=True)
def interp_quat_slerp(t: float, dt: float, q_tab: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Interpolates a quaternion time series using SLERP on a constant-step table.

    Behavior
    --------
    - If the table is empty: returns identity quaternion.
    - If the table has one row: returns that row (normalized for safety).
    - If dt <= 0: returns the first row (normalized; safe fallback).
    - For t <= 0: returns the first row (normalized; clamped).
    - For t >= tmax: returns the last row (normalized; clamped).
    - Otherwise: SLERP between neighbors i and i+1.

    Parameters
    ----------
    t     : float
        Current time [s]
    dt    : float
        Constant time step between rows [s]
    q_tab : ndarray
        (N,4) quaternion table, scalar-first (w,x,y,z)

    Returns
    -------
    (w,x,y,z) : Tuple[float,float,float,float]
        Interpolated unit quaternion.
    """
    n = q_tab.shape[0]

    # Degenerate tables
    if n == 0:
        return 1.0, 0.0, 0.0, 0.0

    # Endpoint
    idx = _table_endpoint_index(t, dt, n)


    if idx >= 0:
        q = q_tab[idx]
        return _quat_normalize_kernel(q[0], q[1], q[2], q[3])

    i, f = _table_index_frac(t, dt, n)

    qA = q_tab[i]
    qB = q_tab[i + 1]

    return _quat_slerp(
        qA[0], qA[1], qA[2], qA[3],
        qB[0], qB[1], qB[2], qB[3],
        f
    )


# Used by: ephemeris
@njit(cache=True)
def interp_vec3_catmull(t: float, dt: float, v_tab: np.ndarray) -> Tuple[float, float, float]:
    """
    Interpolates a 3D vector time series using Catmull-Rom cubic splines (C1 continuous).

    Behavior
    --------
    - If table empty: returns (0,0,0)
    - If table single row: returns that row
    - If dt <= 0: returns first row (safe fallback)
    - Clamps outside [0, tmax] to endpoints
    - Otherwise uses Catmull-Rom on 4 control points with index clamping:
        p0 = i-1, p1 = i, p2 = i+1, p3 = i+2

    Parameters
    ----------
    t     : float
        Current time [s]
    dt    : float
        Constant time step [s]
    v_tab : ndarray
        (N,3) vector table

    Returns
    -------
    (x,y,z) : Tuple[float,float,float]
        Interpolated vector.
    """
    n = v_tab.shape[0]

    # Degenerate tables
    if n == 0:
        return 0.0, 0.0, 0.0

    # Endpoint / fallback selection
    idx = _table_endpoint_index(t, dt, n)


    if idx >= 0:
        v = v_tab[idx]
        return v[0], v[1], v[2]

    # Index + fractional part
    i, f = _table_index_frac(t, dt, n)

    # Control point indices (clamped)
    i0 = i - 1 if i > 0 else 0
    i1 = i
    i2 = i + 1
    i3 = i + 2 if i < n - 2 else n - 1

    p0 = v_tab[i0]
    p1 = v_tab[i1]
    p2 = v_tab[i2]
    p3 = v_tab[i3]

    # Precompute basis weights once (applied to x,y,z)
    f2 = f * f
    f3 = f2 * f

    w0 = -f + 2.0*f2 - f3
    w1 = 2.0 - 5.0*f2 + 3.0*f3
    w2 = f + 4.0*f2 - 3.0*f3
    w3 = -f2 + f3

    # 0.5 factor is part of Catmull-Rom formulation
    x = 0.5 * (p0[0]*w0 + p1[0]*w1 + p2[0]*w2 + p3[0]*w3)
    y = 0.5 * (p0[1]*w0 + p1[1]*w1 + p2[1]*w2 + p3[1]*w3)
    z = 0.5 * (p0[2]*w0 + p1[2]*w1 + p2[2]*w2 + p3[2]*w3)

    return x, y, z



# ============================================================================= 
# 4.                           STEP SIZE & SAMPLING 
# =============================================================================

# Used by: propagator
def nyquist_max_step_s(
    R_ref_m: float,
    mu_m3s2: float,
    degree: int,
    r_min_alt_km: float,
    safety_div: float = 5.0,
    v_margin: float = 1.10,
) -> float:
    """
    Calculates the maximum integrator step size to avoid 'Aliasing' in high-degree
    Spherical Harmonic gravity fields.

    Physics Logic
    -------------
    1. A gravity model of degree N has a minimum spatial wavelength on the surface:
       λ_min ≈ 2π * R / N

    2. A spacecraft moving at max velocity (v_max) encounters these features with
       a frequency:
       f_max ≈ v_max / λ_min  =>  Period T_min ≈ λ_min / v_max

    3. According to Nyquist-Shannon theorem, we must sample at least 2 times per period.
       For numerical integration stability, we usually need 5-10 samples per feature.
       Max Step < T_min / safety_div

    Notes
    -----
    - If the input altitude is negative (or too small), the periapsis radius is clamped
      to at least 1 km above the reference radius to avoid division-by-zero-like issues.

    Returns
    -------
    max_step : float
        Recommended maximum step size in seconds. Returns inf if degree < 2.
    """
    N = int(degree)

    # Monopole (0) and Dipole (1) terms do not have high-frequency ripples.
    if N < 2:
        return math.inf

    if safety_div <= 0.0:
        raise ValueError(f"safety_div must be > 0, got {safety_div}")
    if v_margin <= 0.0:
        raise ValueError(f"v_margin must be > 0, got {v_margin}")

    if R_ref_m <= 0.0:
        raise ValueError(f"R_ref_m must be > 0, got {R_ref_m}")
    if mu_m3s2 <= 0.0:
        raise ValueError(f"mu_m3s2 must be > 0, got {mu_m3s2}")

    # Calculate minimum radius (Periapsis radius)
    r_min_m = R_ref_m + max(1000.0, float(r_min_alt_km) * 1000.0)

    # Estimate Maximum Velocity (Conservative approach)
    v_escape = math.sqrt(2.0 * mu_m3s2 / r_min_m)
    v_max = v_escape * float(v_margin)

    if v_max <= 0.0:  # Pure safety guard (should not happen with valid inputs)
        return math.inf

    # Shortest wavelength on the reference sphere (Eq. Vallado / Montenbruck)
    lam_N = (TWO_PI * R_ref_m) / float(N)

    # Minimum time period to traverse one wavelength
    T_lam = lam_N / v_max

    # Apply safety factor
    return T_lam / float(safety_div)



# ============================================================================= 
# 5.                           ORBITAL FUNCTIONS 
# =============================================================================

# Used by: math_utils
@njit(cache=True)
def _rv_to_coe_kernel(
    rx: float, ry: float, rz: float,
    vx: float, vy: float, vz: float,
    mu: float,
) -> Tuple[float, float, float, float, float, float, float, float, float, float]:
    """
    High-performance kernel: Cartesian state (r,v) -> classical orbital elements (COE).

    References
    ----------
    Standard approach as in Vallado and in Bate/Mueller/White.

    Inputs
    ------
    r = (rx,ry,rz) : position [m]
    v = (vx,vy,vz) : velocity [m/s]
    mu            : gravitational parameter [m^3/s^2] (must be > 0)

    Returns (10)
    ------------
    a     : semi-major axis [m] (inf if "parabolic" by energy test)
    e     : eccentricity magnitude [-]
    inc   : inclination i [rad] in [0, π]
    raan  : RAAN Ω [rad] in [0, 2π)
    argp  : argument of periapsis ω [rad] in [0, 2π)
    nu    : true anomaly ν [rad] in [0, 2π)
    eps   : specific orbital energy [J/kg]
    rnorm : |r| [m]
    vnorm : |v| [m/s]
    hnorm : |h| [m^2/s]

    Singularities handled
    ---------------------
    - Circular (e ~ 0): ω undefined -> 0
        ν becomes:
          * true longitude (equatorial)
          * argument of latitude (inclined)
    - Equatorial (i ~ 0 or π): Ω undefined -> 0
        ω becomes longitude of periapsis via atan2(e_y, e_x) when e>0
    """
    # 1) Magnitudes
    r2 = rx*rx + ry*ry + rz*rz
    v2 = vx*vx + vy*vy + vz*vz
    rnorm = math.sqrt(r2)
    vnorm = math.sqrt(v2)

    # Guard against degenerate states (prevents division by 0 downstream)
    if rnorm < EPS_1E15:
        return (math.inf, 0.0, 0.0, 0.0, 0.0, 0.0, math.nan, rnorm, vnorm, 0.0)

    # Guard against invalid mu (kernel may be called directly)
    if (not math.isfinite(mu)) or (mu <= 0.0):
        return (math.inf, 0.0, 0.0, 0.0, 0.0, 0.0, math.nan, rnorm, vnorm, 0.0)

    # 2) Specific angular momentum h = r x v
    hx = ry*vz - rz*vy
    hy = rz*vx - rx*vz
    hz = rx*vy - ry*vx
    h2 = hx*hx + hy*hy + hz*hz
    hnorm = math.sqrt(h2)

    if hnorm < EPS_1E15:
        # Orbit is ill-defined if h ~ 0 (pure radial motion)
        return (math.inf, 0.0, 0.0, 0.0, 0.0, 0.0, math.nan, rnorm, vnorm, hnorm)

    # 3) Node vector n = k x h = (-hy, hx, 0)
    nx = -hy
    ny = hx
    nnorm = math.sqrt(nx*nx + ny*ny)

    # 4) Eccentricity vector e = (v x h)/mu - r/|r|
    inv_mu = 1.0 / mu
    vxh_x = vy*hz - vz*hy
    vxh_y = vz*hx - vx*hz
    vxh_z = vx*hy - vy*hx

    ex = (vxh_x * inv_mu) - (rx / rnorm)
    ey = (vxh_y * inv_mu) - (ry / rnorm)
    ez = (vxh_z * inv_mu) - (rz / rnorm)
    e2 = ex*ex + ey*ey + ez*ez
    e = math.sqrt(e2)

    # 5) Specific energy eps = v^2/2 - mu/|r|
    eps = 0.5 * v2 - (mu / rnorm)

    # 6) Semi-major axis a
    if abs(eps) < EPS_1E12:
        a = math.inf
    else:
        a = -mu / (2.0 * eps)

    # 7) Inclination inc = acos(hz/|h|) with clamping
    inc = math.acos(clamp(x = (hz / hnorm), lo = -1, hi = 1))

    # ---------------- Singularity detection ----------------
    is_circular = (e < EPS_1E12)
    is_equatorial = (nnorm < EPS_1E12)

    # ---------------- RAAN Ω ----------------
    if is_equatorial:
        raan = 0.0
    else:
        raan = wrap_angle_2pi(math.atan2(ny, nx))

    # ---------------- Argument of periapsis ω ----------------
    if is_circular:
        argp = 0.0
    elif is_equatorial:
        # Longitude of periapsis (since Ω undefined)
        argp = wrap_angle_2pi(math.atan2(ey, ex))
    else:
        # Use atan2(sinω, cosω) for robustness:
        # cosω = (n·e)/(|n||e|)
        # sinω = ((n×e)·h)/(|n||e||h|)
        ndote = nx*ex + ny*ey  # n·e (nz=0)
        cos_w = ndote / (nnorm * e)

        # n×e = (ny*ez, -nx*ez, nx*ey - ny*ex)
        nxe_x = ny * ez
        nxe_y = -nx * ez
        nxe_z = nx*ey - ny*ex
        sin_w = (nxe_x*hx + nxe_y*hy + nxe_z*hz) / (nnorm * e * hnorm)

        argp = wrap_angle_2pi(math.atan2(sin_w, cos_w))

    # ---------------- True anomaly ν ----------------
    if is_circular and is_equatorial:
        # True longitude λ = atan2(y, x)
        nu = wrap_angle_2pi(math.atan2(ry, rx))

    elif is_circular and (not is_equatorial):
        # Argument of latitude u = atan2(sin u, cos u)
        # cos u = (n·r)/(|n||r|)
        # sin u = ((n×r)·h)/(|n||r||h|)
        ndotr = nx*rx + ny*ry  # n·r (nz=0)
        cos_u = ndotr / (nnorm * rnorm)

        # n×r = (ny*rz, -nx*rz, nx*ry - ny*rx)
        nxr_x = ny * rz
        nxr_y = -nx * rz
        nxr_z = nx*ry - ny*rx
        sin_u = (nxr_x*hx + nxr_y*hy + nxr_z*hz) / (nnorm * rnorm * hnorm)

        nu = wrap_angle_2pi(math.atan2(sin_u, cos_u))

    else:
        # Standard ν = atan2(sinν, cosν)
        # cosν = (e·r)/(|e||r|)
        # sinν = ((e×r)·h)/(|e||r||h|)
        edotr = ex*rx + ey*ry + ez*rz
        cos_v = edotr / (e * rnorm)

        # e×r
        exr_x = ey*rz - ez*ry
        exr_y = ez*rx - ex*rz
        exr_z = ex*ry - ey*rx
        sin_v = (exr_x*hx + exr_y*hy + exr_z*hz) / (e * rnorm * hnorm)

        nu = wrap_angle_2pi(math.atan2(sin_v, cos_v))

    return a, e, inc, raan, argp, nu, eps, rnorm, vnorm, hnorm


# Used by: state
def rv_to_coe_select(
    r: np.ndarray,
    v: np.ndarray,
    mu: float,
    *,
    mode: Literal["coe6", "coe10", "kepler5"] = "coe6",
) -> Tuple[float, ...]:
    """
    Public API: single-point RV -> selected element set.

    Parameters
    ----------
    r, v : array-like
        Position and velocity, shape (3,).
    mu : float
        Gravitational parameter (must be > 0).
    mode : str
        - "coe6"    -> (a, e, inc, raan, argp, nu)
        - "coe10"   -> full 10 outputs from kernel
        - "kepler5" -> (a, e, inc, argp, eps)

    Returns
    -------
    Tuple[float, ...] according to mode.
    """
    mu_f = float(mu)
    if not np.isfinite(mu_f) or mu_f <= 0.0:
        raise ValueError("mu must be finite and > 0.")

    r_arr = np.asarray(r, dtype=np.float64).ravel()
    v_arr = np.asarray(v, dtype=np.float64).ravel()
    if r_arr.size != 3 or v_arr.size != 3:
        raise ValueError(f"Expected r and v as 3-vectors; got r.size={r_arr.size}, v.size={v_arr.size}")

    res = _rv_to_coe_kernel(
        float(r_arr[0]), float(r_arr[1]), float(r_arr[2]),
        float(v_arr[0]), float(v_arr[1]), float(v_arr[2]),
        mu_f,
    )

    if mode == "coe6":
        return res[0], res[1], res[2], res[3], res[4], res[5]
    if mode == "coe10":
        return res
    if mode == "kepler5":
        # 0:a, 1:e, 2:inc, 4:argp, 6:eps
        return res[0], res[1], res[2], res[4], res[6]

    # (Literal covers this for type-checkers, but keep runtime safety)
    raise ValueError(f"Unknown mode: {mode!r}. Expected 'coe6', 'coe10', or 'kepler5'.")


# ------------------------------- BATCH KERNEL --------------------------------

MAX_SAFE_SIZE: int = 10_000_000   # safety cap for batch allocations

# Used by: math_utils
@njit(parallel=True, cache=True)
def _batch_y_to_coe_kernel(y: np.ndarray, mu: float) -> Tuple[np.ndarray, ...]:
    """
    Numba parallel kernel: (6,N) state history -> 10 element arrays.

    Input
    -----
    y : (6,N)
        [rx, ry, rz, vx, vy, vz] columns

    Output
    ------
    10 arrays of shape (N,):
    a, e, inc, raan, argp, nu, eps, rnorm, vnorm, hnorm
    """
    N = y.shape[1]
    mu = float(mu)

    a = np.empty(N, dtype=np.float64)
    e = np.empty(N, dtype=np.float64)
    inc = np.empty(N, dtype=np.float64)
    raan = np.empty(N, dtype=np.float64)
    argp = np.empty(N, dtype=np.float64)
    nu = np.empty(N, dtype=np.float64)
    eps = np.empty(N, dtype=np.float64)
    rnorm = np.empty(N, dtype=np.float64)
    vnorm = np.empty(N, dtype=np.float64)
    hnorm = np.empty(N, dtype=np.float64)

    for k in prange(N):
        ak, ek, ik, raan_k, argp_k, nu_k, eps_k, r_k, v_k, h_k = _rv_to_coe_kernel(
            y[0, k], y[1, k], y[2, k],
            y[3, k], y[4, k], y[5, k],
            mu
        )
        a[k] = ak
        e[k] = ek
        inc[k] = ik
        raan[k] = raan_k
        argp[k] = argp_k
        nu[k] = nu_k
        eps[k] = eps_k
        rnorm[k] = r_k
        vnorm[k] = v_k
        hnorm[k] = h_k

    return a, e, inc, raan, argp, nu, eps, rnorm, vnorm, hnorm


# Used by: postprocess
def batch_y_to_elements(
    y: np.ndarray,
    mu: float,
    *,
    mode: Literal["coe10", "coe6", "kepler5"] = "kepler5",
) -> Tuple[np.ndarray, ...]:
    """
    Public API: batch converter for a state history y of shape (6, N).

    Why this exists
    ---------------
    - Python loops over N are slow.
    - This function validates input once, then calls a single Numba-parallel kernel.
    - Output selection is done by slicing the kernel outputs (no extra loops).

    Safety / Stability
    ------------------
    - Enforces y shape = (6, N)
    - Enforces N <= MAX_SAFE_SIZE to avoid accidental huge allocations
    - Forces float64 + contiguous layout (keeps Numba signatures stable)

    Parameters
    ----------
    y : np.ndarray
        State history (6, N) with rows [rx,ry,rz,vx,vy,vz].
    mu : float
        Gravitational parameter (must be > 0).
    mode : str
        - "coe10"   -> returns 10 arrays
        - "coe6"    -> returns 6 arrays (a,e,inc,raan,argp,nu)
        - "kepler5" -> returns 5 arrays (a,e,inc,argp,eps)

    Returns
    -------
    Tuple[np.ndarray, ...] depending on mode.
    """
    mu_f = float(mu)
    if not np.isfinite(mu_f) or mu_f <= 0.0:
        raise ValueError("mu must be finite and > 0.")

    y_arr = np.asarray(y)
    if y_arr.ndim != 2 or y_arr.shape[0] != 6:
        raise ValueError(f"Expected state array shape (6, N), got {y_arr.shape}")

    N = int(y_arr.shape[1])
    if N > MAX_SAFE_SIZE:
        raise ValueError(f"Batch size {N} exceeds safety limit ({MAX_SAFE_SIZE})")

    # Critical for stable Numba typing and speed
    y_contig = np.asarray(y_arr, dtype=np.float64, order="C")

    out = _batch_y_to_coe_kernel(y_contig, mu_f)

    if mode == "coe10":
        return out
    if mode == "coe6":
        return out[0], out[1], out[2], out[3], out[4], out[5]
    if mode == "kepler5":
        return out[0], out[1], out[2], out[4], out[6]

    raise ValueError(f"Unknown mode: {mode!r}. Expected 'coe10', 'coe6', or 'kepler5'.")



# =============================================================================
# 7.                      SCALAR & GEOMETRY UTILITIES
# =============================================================================

# Used by: surface_effects, dynamics
@njit(cache=True)
def wrap_lon_deg(lon_deg: float, west: float = 0.0, east: float = 360.0) -> float:
    """
    Kernel: Wrap longitude in degrees into [west, east).

    Implementation
    --------------
    Uses modulo wrapping:
        wrapped = ((lon - west) % span) + west

    This avoids extra floor() operations and is robust for negative inputs.
    """
    span = east - west
    if span <= EPS_1E12:
        # Degenerate interval; no meaningful wrap possible.
        return lon_deg

    x = lon_deg - west
    # Python/Numba `%` for floats yields a result with the sign of the divisor -> good here.
    return (x % span) + west


# Used by: surface_effects, dynamics
@njit(cache=True)
def latlon_from_xyz_m(
    x_m: float,
    y_m: float,
    z_m: float,
    lon_0_360: bool = True
) -> Tuple[float, float, float]:
    """
    Kernel: Cartesian (body-fixed) -> Spherical (geocentric lat, lon, r).

    Definition
    ----------
    - Geocentric latitude: lat = asin(z / r)
    - Longitude: lon = atan2(y, x)

    Parameters
    ----------
    x_m, y_m, z_m : float
        Cartesian coordinates (meters or any consistent unit).
    lon_0_360 : bool
        If True  -> lon in [0, 360)
        If False -> lon in [-180, 180) (native atan2 output)

    Returns
    -------
    lat_deg : float
        Geocentric latitude in degrees.
    lon_deg : float
        Longitude in degrees.
    r_m : float
        Radius magnitude (same unit as input).
    """
    # Radius squared
    r2 = x_m*x_m + y_m*y_m + z_m*z_m

    # Singularity guard: near-origin => undefined angles; return zeros
    if r2 < EPS_1E18:
        return 0.0, 0.0, 0.0

    r = math.sqrt(r2)

    # Latitude: asin(z/r) with clamp for numerical safety
    sin_lat = clamp(z_m / r, -1.0, 1.0)
    lat_deg = math.asin(sin_lat) * RAD2DEG

    # Longitude in degrees from atan2
    lon_deg = math.atan2(y_m, x_m) * RAD2DEG

    # Optional wrap to [0,360)
    if lon_0_360:
        lon_deg = wrap_lon_deg(lon_deg, 0.0, 360.0)

    return lat_deg, lon_deg, r



# =============================================================================
# 8.                             GRID SAMPLERS
# =============================================================================

# Used by: math_utils
@njit(cache=True)
def _sample_2d_nearest_kernel(
    data: np.ndarray,
    row_f: float,
    col_f: float,
    nlines: int,
    nsamples: int
) -> float:
    """
    Base Kernel: Nearest-neighbor sampling on a 2D grid (latitude/longitude style).

    What it does
    ------------
    Given fractional indices (row_f, col_f), it:
      1) rounds them to the nearest integer (nearest neighbor)
      2) clamps the row index to [0, nlines-1]  (latitude is NOT periodic)
      3) wraps the col index modulo nsamples     (longitude IS periodic)
      4) reads the grid value and returns it as float
    """
    # Safety (assumed handled upstream, but keep kernel robust)
    if nlines <= 0 or nsamples <= 0:
        return math.nan

    # --- 1) nearest integer indices ---
    # NOTE: round() is ties-to-even. If you prefer "half-up", use floor(x+0.5).
    r = int(round(row_f))
    c = int(round(col_f))

    # --- 2) clamp latitude (rows) ---
    if r < 0:
        r = 0
    elif r >= nlines:
        r = nlines - 1

    # --- 3) wrap longitude (cols) ---
    c = c % nsamples

    # --- 4) read from 2D or flat storage ---
    if data.ndim == 2:
        return data[r, c]
    else:
        return data[r * nsamples + c]


# Used bu: math_utils
def _validate_grid_data(
    data: np.ndarray,
    n_rows: int,
    n_cols: int,
    *,
    name: str = "data",
    row_label: str = "n_rows",
    col_label: str = "n_cols",
) -> np.ndarray:
    """Validate and normalize grid storage.

    Accepts either:
    - 2D array with shape (n_rows, n_cols)
    - 1D flat array with size n_rows*n_cols

    Returns a contiguous float64 array view/copy suitable for Numba kernels.
    """
    if n_rows <= 0 or n_cols <= 0:
        raise ValueError(f"Invalid grid dims: {row_label}={n_rows}, {col_label}={n_cols}")

    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim == 2:
        if arr.shape[0] != n_rows or arr.shape[1] != n_cols:
            raise ValueError(f"2D {name} shape {arr.shape} != ({n_rows},{n_cols})")
    elif arr.ndim == 1:
        exp = n_rows * n_cols
        if arr.size != exp:
            raise ValueError(f"Flat {name} size {arr.size} != {row_label}*{col_label}={exp}")
    else:
        raise ValueError(f"{name} must be 1D or 2D, got ndim={arr.ndim}")

    return np.ascontiguousarray(arr)


# Used by: surface_effects
def sample_2d_nearest(
    data: np.ndarray,
    row_f: float,
    col_f: float,
    nlines: int,
    nsamples: int
) -> float:
    """
    Public API: Nearest-neighbor sampler.
    """
    data_f = _validate_grid_data(data, int(nlines), int(nsamples), row_label="nlines", col_label="nsamples")
    return float(_sample_2d_nearest_kernel(
        data_f, float(row_f), float(col_f), int(nlines), int(nsamples)
    ))


# Used by: math_utils
@njit(cache=True)
def _sample_2d_bilinear_kernel(
    data: np.ndarray,
    row_f: float,
    col_f: float,
    nlines: int,
    nsamples: int
) -> float:
    """
    Base Kernel: Bilinear interpolation on a 2D grid (latitude/longitude style).

    Boundary policy
    ---------------
    - Rows (latitude): clamped (non-periodic)
    - Cols (longitude): wrapped modulo nsamples (periodic)
    """
    # Safety (assumed handled upstream, but keep kernel robust)
    if nlines <= 0 or nsamples <= 0:
        return math.nan

    # --- 1) clamp latitude continuously BEFORE forming neighbors ---
    # If outside the grid in latitude, stick to the nearest edge row.
    if row_f <= 0.0:
        r0 = 0
        r1 = 0
        dr = 0.0
    else:
        last = float(nlines - 1)
        if row_f >= last:
            r0 = nlines - 1
            r1 = nlines - 1
            dr = 0.0
        else:
            r0 = int(math.floor(row_f))
            r1 = r0 + 1
            dr = row_f - float(r0)

    # --- 2) longitude cell + fraction (periodic) ---
    c0_unwrapped = int(math.floor(col_f))
    dc = col_f - float(c0_unwrapped)

    c0 = c0_unwrapped % nsamples
    c1 = (c0 + 1) % nsamples

    # --- 3) fetch 4 values (2D or flat) ---
    if data.ndim == 2:
        v00 = data[r0, c0]
        v01 = data[r0, c1]
        v10 = data[r1, c0]
        v11 = data[r1, c1]
    else:
        base0 = r0 * nsamples
        base1 = r1 * nsamples
        v00 = data[base0 + c0]
        v01 = data[base0 + c1]
        v10 = data[base1 + c0]
        v11 = data[base1 + c1]

    # --- 4) bilinear interpolation ---
    top = v00 * (1.0 - dc) + v01 * dc
    bot = v10 * (1.0 - dc) + v11 * dc
    return top * (1.0 - dr) + bot * dr


# Used by: surface_effects
def sample_2d_bilinear(
    data: np.ndarray,
    row_f: float,
    col_f: float,
    nlines: int,
    nsamples: int
) -> float:
    """
    Public API: Bilinear sampler.
    """
    data_f = _validate_grid_data(data, int(nlines), int(nsamples), row_label="nlines", col_label="nsamples")
    return float(_sample_2d_bilinear_kernel(
        data_f, float(row_f), float(col_f), int(nlines), int(nsamples)
    ))


# Used by: math_utils
@njit(cache=True)
def _sample_grid_bilinear_kernel(
    lat_deg: float,
    lon_deg: float,
    data: np.ndarray,
    nlines: int,
    nsamples: int,
    res_deg: float,
    lon0_deg: float,
    lat0_deg: float,
) -> float:
    """
    Composite Kernel: (lat, lon) in degrees -> bilinear sample on a regular grid.
    """
    # --- basic parameter sanity ---
    if nlines <= 0 or nsamples <= 0 or res_deg <= 1e-12:
        return math.nan

    # --- normalize longitude to [0,360) ---
    lon = lon_deg % 360.0

    # --- convert degrees to floating indices ---
    # row: decreases with increasing latitude (north at top)
    line_f = (lat0_deg - lat_deg) / res_deg
    # col: increases with longitude
    samp_f = (lon - lon0_deg) / res_deg

    # --- clamp latitude early ---
    line_f = clamp(line_f, 0.0, float(nlines - 1))

    # --- kernel-to-kernel call (no Python wrappers here!) ---
    return _sample_2d_bilinear_kernel(data, line_f, samp_f, nlines, nsamples)


# Used by: dynamics
def sample_grid_bilinear(
    lat_deg: float,
    lon_deg: float,
    data: np.ndarray,
    nlines: int,
    nsamples: int,
    res_deg: float,
    lon0_deg: float,
    lat0_deg: float,
) -> float:
    """
    Public API: Bilinear sampling using geographic (lat/lon) inputs.
    """
    if nlines <= 0 or nsamples <= 0:
        raise ValueError(f"Invalid grid dims: nlines={nlines}, nsamples={nsamples}")
    if res_deg <= 0.0 or not np.isfinite(res_deg):
        raise ValueError(f"res_deg must be finite and > 0, got {res_deg}")
    data_f = _validate_grid_data(data, int(nlines), int(nsamples), row_label="nlines", col_label="nsamples")
    return float(_sample_grid_bilinear_kernel(
        float(lat_deg), float(lon_deg), data_f,
        int(nlines), int(nsamples), float(res_deg),
        float(lon0_deg), float(lat0_deg)
    ))


# Used by: math_utils
@njit(cache=True)
def _sample_2d_scaled_nearest_kernel(
    data: np.ndarray,
    row_f: float,
    col_f: float,
    n_rows: int,
    n_cols: int,
    scale: float,
    offset: float,
    missing_val: float
) -> float:
    """
    Planetary Kernel: Nearest-neighbor sampling + DN->physical scaling + missing handling.
    """
    if n_rows <= 0 or n_cols <= 0:
        return math.nan

    # Nearest integer indices (half-up)
    r = int(math.floor(row_f + 0.5))
    c = int(math.floor(col_f + 0.5))

    # Clamp latitude
    if r < 0:
        r = 0
    elif r >= n_rows:
        r = n_rows - 1

    # Wrap longitude
    c = c % n_cols

    # Fetch DN
    if data.ndim == 2:
        v = data[r, c]
    else:
        v = data[r * n_cols + c]

    # Missing DN -> NaN
    if (not math.isnan(missing_val)) and (v == missing_val):
        return math.nan

    # Scale + offset to physical units
    return v * scale + offset


# Used by: surface_effects
def sample_2d_scaled_nearest(
    data: np.ndarray,
    row_f: float,
    col_f: float,
    n_rows: int,
    n_cols: int,
    scale: float,
    offset: float,
    missing_val: float
) -> float:
    """
    Public API: Scaled nearest-neighbor sampler.
    """
    data_f = _validate_grid_data(data, int(n_rows), int(n_cols))
    return float(_sample_2d_scaled_nearest_kernel(
        data_f, float(row_f), float(col_f),
        int(n_rows), int(n_cols),
        float(scale), float(offset), float(missing_val)
    ))


# Used by: math_utils
@njit(cache=True)
def _sample_2d_scaled_bilinear_kernel(
    data: np.ndarray,
    row_f: float,
    col_f: float,
    n_rows: int,
    n_cols: int,
    scale: float,
    offset: float,
    missing_val: float
) -> float:
    """
    Planetary Kernel: Bilinear interpolation with "missing DN" robustness.

    Policy
    ------
    If ANY of the 4 bilinear neighbors is missing -> fallback to nearest-neighbor.
    """
    if n_rows <= 0 or n_cols <= 0:
        return math.nan

    # --- clamp latitude continuously BEFORE forming neighbors ---
    # This makes edge behavior robust (n_rows==1 included).
    if row_f <= 0.0:
        r0 = 0
        r1 = 0
        dr = 0.0
    else:
        last = float(n_rows - 1)
        if row_f >= last:
            r0 = n_rows - 1
            r1 = n_rows - 1
            dr = 0.0
        else:
            r0 = int(math.floor(row_f))
            r1 = r0 + 1
            dr = row_f - float(r0)

    # --- longitude cell + fraction (periodic) ---
    c0_unwrapped = int(math.floor(col_f))
    dc = col_f - float(c0_unwrapped)

    c0 = c0_unwrapped % n_cols
    c1 = (c0 + 1) % n_cols

    # Fetch raw DN values
    if data.ndim == 2:
        v00 = data[r0, c0]
        v01 = data[r0, c1]
        v10 = data[r1, c0]
        v11 = data[r1, c1]
    else:
        stride = n_cols
        base0 = r0 * stride
        base1 = r1 * stride
        v00 = data[base0 + c0]
        v01 = data[base0 + c1]
        v10 = data[base1 + c0]
        v11 = data[base1 + c1]

    # Missing DN check -> nearest fallback
    if not math.isnan(missing_val):
        if (v00 == missing_val) or (v01 == missing_val) or (v10 == missing_val) or (v11 == missing_val):
            return _sample_2d_scaled_nearest_kernel(
                data, row_f, col_f, n_rows, n_cols, scale, offset, missing_val
            )

    # Scale DN to physical values before interpolation
    p00 = v00 * scale + offset
    p01 = v01 * scale + offset
    p10 = v10 * scale + offset
    p11 = v11 * scale + offset

    # Bilinear in physical space
    top = p00 * (1.0 - dc) + p01 * dc
    bot = p10 * (1.0 - dc) + p11 * dc
    return top * (1.0 - dr) + bot * dr


# Used by: surface_effects, dynamics
def sample_2d_scaled_bilinear(
    data: np.ndarray,
    row_f: float,
    col_f: float,
    n_rows: int,
    n_cols: int,
    scale: float,
    offset: float,
    missing_val: float
) -> float:
    """
    Public API: Scaled bilinear sampler (with missing-value robustness).
    """
    data_f = _validate_grid_data(data, int(n_rows), int(n_cols))
    return float(_sample_2d_scaled_bilinear_kernel(
        data_f, float(row_f), float(col_f),
        int(n_rows), int(n_cols),
        float(scale), float(offset), float(missing_val)
    ))



# =============================================================================
# 9.                             PUBLIC API
# =============================================================================

__all__ = (

    # Core tiny math (public wrappers)
    "norm3",                     # 3D norm
    "clamp",                     # Public: clamp value into [lo, hi]
    "wrap_angle_2pi",            # Public: wrap angle in radians into [0, 2π)

    # ------------------------------
    # Quaternion Kernels & Wrappers
    # ------------------------------
    "quat_conj",                 # Quaternion conjugate
    "quat_rotate_vec",           # Rotate 3D vector by quaternion

    # NumPy Convenience Wrappers
    "quat_rotate_np",            # Rotate numpy array
    "quat_slerp_np",             # SLERP numpy arrays

    # Interpolation Kernels (JIT) / Public API
    "interp_quat_slerp",         # Time-series quaternion interpolation
    "interp_vec3_catmull",       # Time-series vector cubic spline

    # Physics Helpers
    "nyquist_max_step_s",        # Calculate max safe integrator step size

    # ------------------------------
    # Orbital Mechanics (Public API)
    # ------------------------------
    "rv_to_coe_select",          # Public: single RV->elements selector (coe6/coe10/kepler5)
    "batch_y_to_elements",       # Public: batch RV->elements selector (coe6/coe10/kepler5)

    # ------------------------------
    # Scalar & Geometry Utilities (Public API)
    # ------------------------------
    "wrap_lon_deg",              # Public: longitude wrap in degrees into [west, east)
    "latlon_from_xyz_m",         # Public: Cartesian -> (lat_deg, lon_deg, r_m)

    # ------------------------------
    # Grid Samplers (Public API)
    # ------------------------------
    "sample_2d_nearest",         # Public: nearest-neighbor sampler (wrap lon, clamp lat)
    "sample_2d_bilinear",        # Public: bilinear sampler (wrap lon, clamp lat)
    "sample_grid_bilinear",      # Public: lat/lon -> indices -> bilinear sample
    "sample_2d_scaled_nearest",  # Public: nearest + scale/offset + missing handling
    "sample_2d_scaled_bilinear", # Public: bilinear + scale/offset + missing fallback
)
