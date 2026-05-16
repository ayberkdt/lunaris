# LUNAR_SIMULATION/models/third_body_effects.py
"""
Third-Body Effects (Differential Gravity, Solid Tides, and Optional Earth J2)
============================================================================

This module provides third-body perturbation models in a Moon-centered frame.

1) Differential (tidal) point-mass gravity
------------------------------------------
Because the simulation origin is the Moon center, the physically consistent
third-body term is the *differential* acceleration:

    a_3b = μ * ( (r_body - r_sc)/|r_body - r_sc|^3  -  r_body/|r_body|^3 )

where:
- r_sc   : spacecraft position w.r.t. Moon center [m]
- r_body : third-body position w.r.t. Moon center [m]
- μ      : third-body gravitational parameter [m^3/s^2]

This subtracts the origin (Moon) acceleration due to the third body, yielding
the correct relative equations of motion in a Moon-centered formulation.

2) Solid tides (Moon deformation)  [optional]
---------------------------------------------
Includes a simple degree-2 and optional degree-3 Love-number tide model for
the Moon, raised by an external perturber (e.g., Earth or Sun).

3) Earth J2 differential term  [optional]
-----------------------------------------
Provides an optional differential J2 (oblateness) contribution for Earth in a
Moon-centered inertial frame.

Conventions
-----------
- Frames:
  - All vectors are expressed in the *same* frame per call (typically Moon-centered inertial).
  - This module does not perform rotations; frame transforms belong in your attitude/frames layer.
- Vectors:
  - r_sc   = spacecraft position relative to Moon center.
  - r_body = external body position relative to Moon center (Earth, Sun, etc.).
- Units:
  - Positions in meters [m], μ in [m^3/s^2], accelerations in [m/s^2].
- Sign:
  - Returned accelerations are *applied to the spacecraft* (i.e., the RHS term in your EOM).

Layering and dependencies
-------------------------
- Physical constants are sourced from ``common.constants`` (single source of truth).
- Numba kernels accelerate the inner loops; NumPy wrappers remain available for
  convenience and testing.
"""


# =============================================================================
# 0.                                IMPORTS
# =============================================================================

from __future__ import annotations

import math
import numpy as np

import numpy.typing as npt
from numpy.typing import ArrayLike
from typing import Tuple, Dict, Optional

from dataclasses import dataclass, field

from numba import njit


from common.constants import MU_EARTH, MU_SUN, R_MOON_MEAN, R_EARTH_EQUATORIAL

from common.type_defs import Vec3


# =============================================================================
# 1.                       CORE ACCELERATION LOGIC
# =============================================================================

# A small squared-distance guard to avoid Inf/NaN if vectors are degenerate.
# (1.0 m)^2 is conservative; adjust if you want a different collision/singularity policy.
_MIN_R2 = 1.0


@njit(cache=True)
def accel_third_body_numba(
    rx: float, ry: float, rz: float,
    bx: float, by: float, bz: float,
    mu: float,
) -> Tuple[float, float, float]:
    """
    Third-body differential gravity acceleration (spacecraft relative to central body).

    a_rel = a_sc_tb - a_cb_tb
          = mu * [ (r_tb - r_sc)/|r_tb - r_sc|^3  -  r_tb/|r_tb|^3 ]

    Parameters
    ----------
    (rx, ry, rz)
        Spacecraft position wrt central body [m].
    (bx, by, bz)
        Third-body position wrt central body [m].
    mu
        Third-body gravitational parameter [m^3/s^2].

    Returns
    -------
    (ax, ay, az)
        Differential acceleration [m/s^2].
    """
    # d = r_tb - r_sc
    dx = bx - rx
    dy = by - ry
    dz = bz - rz

    d2 = dx * dx + dy * dy + dz * dz   # |r_tb - r_sc|^2
    b2 = bx * bx + by * by + bz * bz   # |r_tb|^2

    # Singularity/collision guards (policy: return 0-vector)
    if d2 <= _MIN_R2 or b2 <= _MIN_R2:
        return 0.0, 0.0, 0.0

    inv_d3 = 1.0 / (d2 * math.sqrt(d2))
    inv_b3 = 1.0 / (b2 * math.sqrt(b2))

    ax = mu * (dx * inv_d3 - bx * inv_b3)
    ay = mu * (dy * inv_d3 - by * inv_b3)
    az = mu * (dz * inv_d3 - bz * inv_b3)
    return ax, ay, az



# =============================================================================
# 2.                         PYTHON WRAPPERS (API)
# =============================================================================

def _as_vec3(x: npt.ArrayLike, name: str = "vec") -> Vec3:
    v = np.asarray(x, dtype=np.float64)
    if v.shape != (3,):
        # allow (3,1) / (1,3) as common caller mistakes
        if v.size == 3:
            v = v.reshape(3,)
        else:
            raise ValueError(f"{name} must have shape (3,), got {v.shape}.")
    return v


def calc_3rd_body_accel(r_sc: ArrayLike, r_body: ArrayLike, mu: float) -> Vec3:
    """
    Compute third-body differential gravity acceleration (central-body relative).

    This is a Python-level convenience wrapper around `accel_third_body_numba`.

    Parameters
    ----------
    r_sc : ArrayLike, shape (3,)
        Spacecraft position vector relative to the central body [m].
    r_body : ArrayLike, shape (3,)
        Third-body position vector relative to the central body [m].
    mu : float
        Third-body gravitational parameter [m^3/s^2].

    Returns
    -------
    Vec3
        Differential acceleration vector [m/s^2].
    """
    r_sc_v = _as_vec3(r_sc, name="r_sc")
    r_tb_v = _as_vec3(r_body, name="r_body")

    ax, ay, az = accel_third_body_numba(
        float(r_sc_v[0]), float(r_sc_v[1]), float(r_sc_v[2]),
        float(r_tb_v[0]), float(r_tb_v[1]), float(r_tb_v[2]),
        float(mu),
    )
    return np.array((ax, ay, az), dtype=np.float64)


def calc_central_body_accel(r_sc: ArrayLike, mu: float) -> Vec3:
    """
    Compute central-body point-mass (monopole) gravity.

    Model
    -----
        a = -mu * r / |r|^3

    Notes
    -----
    - This is a lightweight baseline (no harmonics).
    - For high-fidelity central-body gravity, use spherical harmonics instead.

    Parameters
    ----------
    r_sc : ArrayLike, shape (3,)
        Spacecraft position vector relative to the central body [m].
    mu : float
        Central-body gravitational parameter [m^3/s^2].

    Returns
    -------
    Vec3
        Acceleration vector [m/s^2].
    """
    r = _as_vec3(r_sc, name="r_sc")

    r2 = float(r[0] * r[0] + r[1] * r[1] + r[2] * r[2])
    if r2 <= 1e-6:
        return np.zeros(3, dtype=np.float64)

    inv_r = 1.0 / math.sqrt(r2)
    inv_r3 = inv_r * inv_r * inv_r
    return (-float(mu) * inv_r3) * r



# =============================================================================
# 3.                      SOLID TIDES (DYNAMICAL GRAVITY)
# =============================================================================

@dataclass(frozen=True, slots=True)
class LoveParams:
    """
    Love numbers and toggles for a simple solid-tide model.

    Notes
    -----
    - k2/k3: dimensionless Love numbers describing the central body's elastic response.
    - apply_earth_tide / apply_sun_tide are orchestration toggles (used by higher-level
      models that combine multiple external perturbers). The kernel below is *generic*
      and does not branch on these toggles.
    - Defaults are commonly used Moon values (often cited around GL1800F-era usage).
      Keep them configurable if you plan to swap geophysical models.
    """
    k2: float = 0.024223
    k3: float = 0.0163
    apply_earth_tide: bool = True
    apply_sun_tide: bool = True


@njit(cache=True, fastmath=True)
def accel_solid_tide(
    rx: float, ry: float, rz: float,
    sx: float, sy: float, sz: float,
    mu_ext: float,
    R_ref: float,
    k2: float,
    k3: float = 0.0,
) -> Tuple[float, float, float]:
    """
    Solid-tide acceleration on a spacecraft due to an external perturber.

    Inputs are central-body centered:
      r = spacecraft position (rx, ry, rz)
      s = external body position (sx, sy, sz)

    The model returns the additional acceleration caused by the tidally deformed
    potential of the central body (degree-2 and optional degree-3).

    Parameters
    ----------
    (rx, ry, rz)
        Spacecraft position wrt central body [m].
    (sx, sy, sz)
        External-body position wrt central body [m].
    mu_ext
        External-body GM [m^3/s^2].
    R_ref
        Central-body reference radius [m]. Must be positive for meaningful results.
    k2, k3
        Degree-2 and degree-3 Love numbers (dimensionless).

    Returns
    -------
    (ax, ay, az)
        Tide acceleration [m/s^2] in the same frame as inputs.

    Notes
    -----
    - The closed-form expression is evaluated for any non-degenerate radius.
      Interior points are saturated to the surface radius so debug/low-altitude
      states do not blow up or silently drop to zero.
    - Degenerate geometry returns (0, 0, 0) by design (robust propagation policy).
    """
    # -------------------------------------------------------------------------
    # Fast reject: nothing to do
    # -------------------------------------------------------------------------
    if mu_ext == 0.0 or (k2 == 0.0 and k3 == 0.0):
        return 0.0, 0.0, 0.0

    # Basic sanity on R_ref (keep kernel robust; policy decisions belong above)
    if R_ref <= 0.0:
        return 0.0, 0.0, 0.0

    # -------------------------------------------------------------------------
    # Geometry
    # -------------------------------------------------------------------------
    r2 = rx * rx + ry * ry + rz * rz
    s2 = sx * sx + sy * sy + sz * sz

    # External body at/near origin is undefined for this model
    if s2 <= 0.0:
        return 0.0, 0.0, 0.0

    # Only reject degenerate near-center states. A broader interior-body guard
    # silently disabled tides for otherwise valid low-altitude/debug scenarios.
    if r2 <= _MIN_R2:
        return 0.0, 0.0, 0.0

    r = math.sqrt(r2)
    s = math.sqrt(s2)

    # Saturate interior states at the reference surface radius. This keeps the
    # response finite and directionally meaningful without silently erasing the
    # perturbation for low-radius test/debug trajectories.
    r_eval = r if r >= R_ref else R_ref

    inv_r = 1.0 / r
    inv_s = 1.0 / s
    inv_r_eval = 1.0 / r_eval
    inv_r2 = inv_r_eval * inv_r_eval

    # Unit vectors u = r/|r|, e = s/|s|
    ux = rx * inv_r
    uy = ry * inv_r
    uz = rz * inv_r

    ex = sx * inv_s
    ey = sy * inv_s
    ez = sz * inv_s

    # cos(theta) = u·e
    c = ux * ex + uy * ey + uz * ez
    # Clamp for numerical safety
    if c > 1.0:
        c = 1.0
    elif c < -1.0:
        c = -1.0

    c2 = c * c

    # -------------------------------------------------------------------------
    # Ratios (R/s) and (R/r)
    # -------------------------------------------------------------------------
    R_over_s = R_ref * inv_s
    R_over_r = R_ref * inv_r_eval

    # Powers (avoid pow() for speed & determinism under Numba)
    R_over_s2 = R_over_s * R_over_s
    R_over_s3 = R_over_s2 * R_over_s
    R_over_s4 = R_over_s2 * R_over_s2

    R_over_r2 = R_over_r * R_over_r
    R_over_r3 = R_over_r2 * R_over_r

    ax = 0.0
    ay = 0.0
    az = 0.0

    # -------------------------------------------------------------------------
    # Degree-2 tide contribution
    # a2 = 1.5*(k2*mu_ext/r^2)*(R/s)^3*(R/r)^2 * [ (5c^2-1) u - 2c e ]
    # -------------------------------------------------------------------------
    if k2 != 0.0:
        fac2 = 1.5 * (k2 * mu_ext * inv_r2) * (R_over_s3 * R_over_r2)
        k_u2 = 5.0 * c2 - 1.0
        k_e2 = 2.0 * c

        ax += fac2 * (k_u2 * ux - k_e2 * ex)
        ay += fac2 * (k_u2 * uy - k_e2 * ey)
        az += fac2 * (k_u2 * uz - k_e2 * ez)

    # -------------------------------------------------------------------------
    # Degree-3 tide contribution
    # a3 = (k3*mu_ext/r^2)*(R/s)^4*(R/r)^3 * [ Tu(c) u - Te(c) e ]
    # Tu = 0.5*(35c^3 - 15c)
    # Te = 1.5*(5c^2 - 1)
    # -------------------------------------------------------------------------
    if k3 != 0.0:
        fac3 = (k3 * mu_ext * inv_r2) * (R_over_s4 * R_over_r3)
        c3 = c2 * c

        Tu = 0.5 * (35.0 * c3 - 15.0 * c)
        Te = 1.5 * (5.0 * c2 - 1.0)

        ax += fac3 * (Tu * ux - Te * ex)
        ay += fac3 * (Tu * uy - Te * ey)
        az += fac3 * (Tu * uz - Te * ez)

    return ax, ay, az



# =============================================================================
# 3.                           EARTHS J2 PERTURBATION
# =============================================================================

@dataclass(frozen=True, slots=True)
class EarthJ2Params:
    """
    Parameters for Earth's J2 (oblateness) contribution to the *differential*
    third-body acceleration in a Moon-centered inertial frame.

    Context
    -------
    In a Moon-centered formulation, the consistent third-body term is:

        a_diff = a_body(Earth -> SC) - a_body(Earth -> Moon-origin)

    This container provides the additional parameters needed to compute the J2
    part of a_body, beyond the standard point-mass third-body model.

    Attributes
    ----------
    j2_coeff
        Earth's J2 coefficient (dimensionless). WGS-84 nominal is ~1.08263e-3.
    r_eq_m
        Earth's equatorial reference radius [m] used by the J2 model.
    spin_axis_i
        Earth's spin axis expressed in the inertial frame (allowed non-unit;
        it will be normalized once per call in the Python wrapper).
    """
    j2_coeff: float = 1.082_626_68e-3
    r_eq_m: float = R_EARTH_EQUATORIAL  # expects your constants SSOT
    spin_axis_i: Tuple[float, float, float] = (0.0, 0.0, 1.0)


@njit(cache=True, nogil=True, fastmath=True)
def _normalize_axis_or_default(
    kx: float, ky: float, kz: float
) -> Tuple[float, float, float]:
    """
    Normalize (kx,ky,kz). If degenerate, return default (0,0,1).
    """
    k2 = kx * kx + ky * ky + kz * kz
    if k2 > 0.0:
        inv_k = 1.0 / math.sqrt(k2)
        return kx * inv_k, ky * inv_k, kz * inv_k
    return 0.0, 0.0, 1.0


@njit(cache=True, nogil=True, fastmath=True)
def _accel_j2_oblate_unit_k(
    x: float, y: float, z: float,
    mu: float, r_ref: float, j2: float,
    kx: float, ky: float, kz: float,
) -> Tuple[float, float, float]:
    """
    J2 acceleration for an oblate body acting on a test particle in the body's
    centered frame. Spin axis k must be UNIT LENGTH.

    Formula
    -------
    a_J2 = (3/2) * J2 * mu * R^2 / r^5 * [ (5 (r·k)^2 / r^2 - 1) r  -  2 (r·k) k ]

    Returns
    -------
    (ax, ay, az) as a tuple (no heap allocations).
    """
    r2 = x * x + y * y + z * z
    # Robust singularity guard (policy: return zero rather than NaN/Inf)
    if r2 < 1.0:
        return 0.0, 0.0, 0.0

    rk = x * kx + y * ky + z * kz  # r·k
    rk2 = rk * rk

    r = math.sqrt(r2)
    inv_r = 1.0 / r
    inv_r2 = inv_r * inv_r
    inv_r5 = inv_r2 * inv_r2 * inv_r  # 1/r^5

    pref = 1.5 * j2 * mu * (r_ref * r_ref) * inv_r5
    term_a = 5.0 * (rk2 * inv_r2) - 1.0
    term_b = 2.0 * rk

    ax = pref * (term_a * x - term_b * kx)
    ay = pref * (term_a * y - term_b * ky)
    az = pref * (term_a * z - term_b * kz)
    return ax, ay, az


@njit(cache=True, nogil=True, fastmath=True)
def accel_j2_oblate_diff_numba(
    rx: float, ry: float, rz: float,
    bx: float, by: float, bz: float,
    mu_body: float,
    r_ref: float,
    j2: float,
    kx: float, ky: float, kz: float,  # UNIT
) -> Tuple[float, float, float]:
    return _accel_j2_oblate_diff_unit_k(
        rx, ry, rz, bx, by, bz, mu_body, r_ref, j2, kx, ky, kz
    )


@njit(cache=True, nogil=True, fastmath=True)
def _accel_j2_oblate_diff_unit_k(
    rx: float, ry: float, rz: float,   # SC position wrt Moon (inertial)
    bx: float, by: float, bz: float,   # Earth position wrt Moon (inertial)
    mu_body: float,
    r_ref: float,
    j2: float,
    kx: float, ky: float, kz: float,   # Earth spin axis (UNIT, inertial)
) -> Tuple[float, float, float]:
    """
    Differential Earth-J2 acceleration in a Moon-centered inertial frame:

        a_diff = a_J2(Earth->SC) - a_J2(Earth->Moon-origin)

    Notes
    -----
    - k must be unit length; normalize once in the Python wrapper (preferred).
    - Returns a tuple (ax, ay, az) to avoid heap allocations in tight RHS loops.
    """
    # Earth -> Spacecraft vector: r_ES = r_SC - r_Earth
    x_es = rx - bx
    y_es = ry - by
    z_es = rz - bz

    # Earth -> Moon-origin vector: r_EM = 0 - r_Earth = -r_Earth
    x_em = -bx
    y_em = -by
    z_em = -bz

    ax_s, ay_s, az_s = _accel_j2_oblate_unit_k(x_es, y_es, z_es, mu_body, r_ref, j2, kx, ky, kz)
    ax_m, ay_m, az_m = _accel_j2_oblate_unit_k(x_em, y_em, z_em, mu_body, r_ref, j2, kx, ky, kz)

    return (ax_s - ax_m), (ay_s - ay_m), (az_s - az_m)


def calc_j2_oblate_diff_accel(
    r_sc: npt.ArrayLike,
    r_body: npt.ArrayLike,
    *,
    mu_body: float,
    params: Optional[EarthJ2Params] = None,
    r_ref: Optional[float] = None,
    j2: Optional[float] = None,
    k_hat: Optional[npt.ArrayLike] = None,
) -> Vec3:
    """
    Compute the differential Earth-J2 acceleration as a (3,) NumPy array.

    Parameter sourcing
    ------------------
    - If `params` is provided: uses (params.r_eq_m, params.j2_coeff, params.spin_axis_i).
    - Otherwise: uses explicit (r_ref, j2, k_hat).

    Performance notes
    -----------------
    - This wrapper allocates a (3,) array.
    - In a tight RHS loop, prefer calling `_accel_j2_oblate_diff_unit_k(...)`
      directly and accumulate into scalar ax/ay/az.

    Parameters
    ----------
    r_sc
        Spacecraft position wrt Moon [m], shape (3,).
    r_body
        Earth position wrt Moon [m], shape (3,).
    mu_body
        Earth's GM [m^3/s^2].
    params
        Optional EarthJ2Params bundle (recommended).
    r_ref, j2, k_hat
        Explicit overrides if `params` is None. `k_hat` can be non-unit; it will
        be normalized once here.

    Returns
    -------
    (3,) np.ndarray
        Differential J2 acceleration [m/s^2].
    """
    if mu_body == 0.0:
        return np.zeros(3, dtype=np.float64)

    r_sc_v = _as_vec3(r_sc, "r_sc")
    r_body_v = _as_vec3(r_body, "r_body")

    if params is not None:
        r_ref_v = float(params.r_eq_m)
        j2_v = float(params.j2_coeff)
        kx, ky, kz = float(params.spin_axis_i[0]), float(params.spin_axis_i[1]), float(params.spin_axis_i[2])
    else:
        if r_ref is None or j2 is None or k_hat is None:
            raise ValueError("Provide either params=EarthJ2Params(...) or (r_ref, j2, k_hat).")
        r_ref_v = float(r_ref)
        j2_v = float(j2)
        kh = _as_vec3(k_hat, "k_hat")
        kx, ky, kz = float(kh[0]), float(kh[1]), float(kh[2])

    # Normalize axis ONCE in Python space (cheaper, avoids per-call branching in kernel)
    kx_u, ky_u, kz_u = _normalize_axis_or_default(kx, ky, kz)

    ax, ay, az = _accel_j2_oblate_diff_unit_k(
        float(r_sc_v[0]), float(r_sc_v[1]), float(r_sc_v[2]),
        float(r_body_v[0]), float(r_body_v[1]), float(r_body_v[2]),
        float(mu_body),
        float(r_ref_v),
        float(j2_v),
        float(kx_u), float(ky_u), float(kz_u),
    )
    return np.array((ax, ay, az), dtype=np.float64)



# =============================================================================
# 4.                            HIGH-LEVEL MANAGER
# =============================================================================

@dataclass(slots=True)
class ThirdBodyModel:
    """
    Facade for third-body differential gravity and Moon solid tides.

    Provides:
      - Differential third-body gravity (Earth/Sun): a_sc - a_cb
      - Optional Moon solid-tide acceleration raised by Earth/Sun (k2/k3)

    Notes
    -----
    - Inputs are Moon-centered inertial (or any consistent CB-centered frame).
    - This class intentionally does not import anything in __post_init__.
      All constants (MU_EARTH, MU_SUN, R_MOON_MEAN) must be module-level SSOT.
    - `calc_3rd_body_accel(...)` returns a freshly allocated (3,) array.
      This is convenient but not the lowest-allocation option for tight RHS loops.
    """

    # User override map, e.g. {"earth": MU_EARTH, "sun": MU_SUN}.
    # Keys are case-insensitive and normalized to lowercase in __post_init__.
    mu_map: Dict[str, float] = field(default_factory=dict)

    # Love numbers + toggles controlling whether Earth/Sun solid-tide terms are applied.
    love: "LoveParams" = field(default_factory=lambda: LoveParams())

    # Reference radius used by the solid-tide model (Moon mean radius by default).
    R_ref: float = float(R_MOON_MEAN)

    def __post_init__(self) -> None:
        # ---------------------------------------------------------------------
        # Normalize and seed mu_map
        # ---------------------------------------------------------------------
        # Base defaults (single source of truth is module-level constants).
        base = {"earth": float(MU_EARTH), "sun": float(MU_SUN)}

        # Normalize user-provided map:
        # - force keys to lowercase strings
        # - force values to float
        merged: Dict[str, float] = {str(k).lower(): float(v) for k, v in self.mu_map.items()}

        # Fill missing keys from base defaults (do not overwrite user overrides).
        for k, v in base.items():
            merged.setdefault(k, v)

        # Assign normalized map back to the instance.
        self.mu_map = merged

        # Ensure reference radius is float (defensive, avoids weird dtypes).
        self.R_ref = float(self.R_ref)

    def compute_earth(self, r_sc: npt.ArrayLike, r_earth: npt.ArrayLike) -> Vec3:
        """
        Earth: differential third-body gravity + optional Earth-raised Moon solid tides.

        Parameters
        ----------
        r_sc
            Spacecraft position wrt Moon [m], shape (3,).
        r_earth
            Earth position wrt Moon [m], shape (3,).

        Returns
        -------
        (3,) np.ndarray
            Total acceleration contribution [m/s^2].
        """
        # Validate/convert to (3,) float64 arrays
        r_sc_v = _as_vec3(r_sc, "r_sc")
        r_e_v = _as_vec3(r_earth, "r_earth")

        # Differential point-mass third-body gravity: a(sc<-Earth) - a(Moon<-Earth)
        mu_e = self.mu_map["earth"]
        a = calc_3rd_body_accel(r_sc_v, r_e_v, mu_e)

        # Optional solid tide (Moon deformed by Earth) contribution
        if self.love.apply_earth_tide:
            tx, ty, tz = accel_solid_tide(
                float(r_sc_v[0]), float(r_sc_v[1]), float(r_sc_v[2]),
                float(r_e_v[0]),  float(r_e_v[1]),  float(r_e_v[2]),
                float(mu_e), float(self.R_ref), float(self.love.k2), float(self.love.k3),
            )
            # In-place add to avoid an extra allocation
            a[0] += tx
            a[1] += ty
            a[2] += tz

        return a

    def compute_sun(self, r_sc: npt.ArrayLike, r_sun: npt.ArrayLike) -> Vec3:
        """
        Sun: differential third-body gravity + optional Sun-raised Moon solid tides.

        Parameters
        ----------
        r_sc
            Spacecraft position wrt Moon [m], shape (3,).
        r_sun
            Sun position wrt Moon [m], shape (3,).

        Returns
        -------
        (3,) np.ndarray
            Total acceleration contribution [m/s^2].
        """
        # Validate/convert to (3,) float64 arrays
        r_sc_v = _as_vec3(r_sc, "r_sc")
        r_s_v = _as_vec3(r_sun, "r_sun")

        # Differential point-mass third-body gravity: a(sc<-Sun) - a(Moon<-Sun)
        mu_s = self.mu_map["sun"]
        a = calc_3rd_body_accel(r_sc_v, r_s_v, mu_s)

        # Optional solid tide (Moon deformed by Sun) contribution
        if self.love.apply_sun_tide:
            tx, ty, tz = accel_solid_tide(
                float(r_sc_v[0]), float(r_sc_v[1]), float(r_sc_v[2]),
                float(r_s_v[0]),  float(r_s_v[1]),  float(r_s_v[2]),
                float(mu_s), float(self.R_ref), float(self.love.k2), float(self.love.k3),
            )
            # In-place add to avoid an extra allocation
            a[0] += tx
            a[1] += ty
            a[2] += tz

        return a

    def compute_generic(self, r_sc: npt.ArrayLike, r_body: npt.ArrayLike, mu: float) -> Vec3:
        """
        Generic differential third-body point-mass gravity (no solid tides).

        Parameters
        ----------
        r_sc
            Spacecraft position wrt Moon [m], shape (3,).
        r_body
            Third-body position wrt Moon [m], shape (3,).
        mu
            Third-body GM [m^3/s^2].

        Returns
        -------
        (3,) np.ndarray
            Differential acceleration [m/s^2].
        """
        # Validate/convert to (3,) float64 arrays, then compute
        return calc_3rd_body_accel(
            _as_vec3(r_sc, "r_sc"),
            _as_vec3(r_body, "r_body"),
            float(mu),
        )



# =============================================================================
# 7.                            PUBLIC API
# =============================================================================

__all__ = (
    # -------------------------------------------------------------------------
    # Configuration dataclasses
    # -------------------------------------------------------------------------
    "LoveParams",        # Moon solid-tide params (if implemented in this module)
    "EarthJ2Params",     # Earth J2 parameters (j2_coeff, r_eq_m, spin_axis_i)

    # -------------------------------------------------------------------------
    # High-level facade (optional, user-facing)
    # -------------------------------------------------------------------------
    "ThirdBodyModel",    # If you keep a class-based manager; otherwise remove.

    # -------------------------------------------------------------------------
    # Core Numba entry points (DynamicsEngine should call THESE)
    # -------------------------------------------------------------------------
    "accel_third_body_numba",        # Differential 3rd-body point-mass acceleration (tuple return)
    "accel_j2_oblate_diff_numba",    # Differential J2 correction (tuple return, UNIT spin axis)

    # -------------------------------------------------------------------------
    # Public NumPy wrappers (debug/tests; allocate arrays)
    # -------------------------------------------------------------------------
    "calc_3rd_body_accel",           # Returns (3,) np.ndarray (wrapper around accel_third_body_numba)
    "calc_j2_oblate_diff_accel",     # Returns (3,) np.ndarray (wrapper; normalizes axis once)
)

