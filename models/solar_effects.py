# ST_LRPS/models/solar_effects.py
"""
Solar Radiation Pressure (SRP) and Eclipse Geometry
===================================================

This module provides fast, reusable physics primitives for solar-driven
perturbations used by the simulation core.

What it contains
----------------
1) Solar Radiation Pressure (SRP) — cannonball model
   - Acceleration magnitude scales with the inverse-square of the Sun–spacecraft distance.
   - Acceleration direction is away from the Sun (along the Sun→spacecraft line).

2) Eclipse / illumination factor (Moon and optionally Earth)
   - Conical umbra/penumbra model using a finite apparent solar radius.
   - Returns a smooth illumination factor ``nu`` in ``[0, 1]``:
       * ``nu = 0`` : full umbra (total eclipse)
       * ``nu = 1`` : full sunlight
       * ``0 < nu < 1`` : penumbra (smoothstep blend)

Coordinate and sign conventions
-------------------------------
- Unless otherwise stated, inputs are Moon-centered vectors in an inertial frame.
- ``r_sc = (rx, ry, rz)``  : spacecraft position w.r.t. the Moon center [m]
- ``r_sun = (sx, sy, sz)`` : Sun position w.r.t. the Moon center [m]
- The Sun→spacecraft direction is ``d_vec = r_sc - r_sun``.
  The SRP acceleration points along ``d_vec`` (i.e., away from the Sun).

Earth eclipse (optional)
------------------------
- Earth-shadow helpers (``earth_shadow_factor_conical`` / ``in_earth_umbra_conical``)
  expect Earth-centered vectors.
- If your ephemeris is Moon-centered, convert via:
  ``r_sc_earth  = r_sc_moon  - r_earth_moon``
  ``r_sun_earth = r_sun_moon - r_earth_moon``

Performance and API layering
----------------------------
- Low-level Numba kernels are allocation-free and suitable for hot loops.
- Python wrappers provide a simple interface; prefer the ``*_out`` wrapper when
  you want to avoid per-call allocations.
- Shared constants are imported from ``common.constants`` (single source of truth).

Notes
-----
- The penumbra transition uses a smoothstep blend for continuity and stability;
  it is not an exact solar-disk overlap area model.
- The conical shadow model is an engineering approximation. For high-fidelity eclipse
  timing, use extended-body occultation with precise ephemerides.
"""



# =============================================================================
# 0.                                IMPORTS
# =============================================================================

from __future__ import annotations

import math
import numpy as np

from typing import Tuple
from dataclasses import dataclass


from numba import njit


from common.constants import (P_SUN_1AU, AU, 
                              R_SUN_MEAN, R_MOON_MEAN, R_EARTH_MEAN,
                              EPS_1E6,EPS_1E12, EPS_1E24)

from common.type_defs import SpacecraftProps



# =============================================================================
# 1.                       SHADOW / ECLIPSE GEOMETRY
# =============================================================================

# -------------------------------------------------------------------------
# Conical eclipse model (finite Sun size) — shared implementation
#
# Returns nu in [0, 1]:
#   nu = 1.0  -> full sunlight
#   nu = 0.0  -> full shadow (umbra)
#   0<nu<1    -> partial shadow (penumbra)
#
# Notes:
# - Penumbra transition uses a smoothstep blend for numerical stability and
#   continuity. It is not an exact solar-disk overlap area model.
# -------------------------------------------------------------------------


@njit(cache=True, nogil=True)
def _shadow_factor_conical(
    rx: float, ry: float, rz: float,
    sx: float, sy: float, sz: float,
    R_occult: float,
) -> float:
    """
    Generic conical shadow factor for an occulting body of radius R_occult.

    Inputs MUST be centered on the occulting body:
      r_sc = (rx, ry, rz)  : spacecraft position w.r.t. occulting body center
      r_sun = (sx, sy, sz) : sun position w.r.t. occulting body center
    """
    # 1) Sun direction (occulting body -> Sun)
    dist_sun_sq = sx * sx + sy * sy + sz * sz
    if dist_sun_sq <= EPS_1E24:
        return 1.0  # degenerate Sun vector -> assume sunlight

    dist_sun = math.sqrt(dist_sun_sq)
    inv_dist_sun = 1.0 / dist_sun
    u_sx = sx * inv_dist_sun
    u_sy = sy * inv_dist_sun
    u_sz = sz * inv_dist_sun

    # 2) Project spacecraft position onto the Sun axis:
    # proj >= 0 => day-side (between occulting body and Sun) => no eclipse
    proj = rx * u_sx + ry * u_sy + rz * u_sz
    if proj >= 0.0:
        return 1.0

    # x = distance behind the occulting body along the shadow axis (x >= 0)
    x = -proj

    # 3) Perpendicular distance from shadow axis:
    # rho^2 = |r|^2 - proj^2
    r_sq = rx * rx + ry * ry + rz * rz
    rho2 = r_sq - proj * proj
    if rho2 < 0.0:
        rho2 = 0.0  # clamp roundoff

    # 4) Cone geometry (similar triangles)
    denom_u = R_SUN_MEAN - R_occult
    if denom_u <= EPS_1E12:
        return 1.0  # degenerate geometry -> assume sunlight

    common_factor = R_occult * dist_sun
    L_u = common_factor / denom_u

    denom_p = R_SUN_MEAN + R_occult
    if denom_p <= EPS_1E12:
        return 1.0
    L_p = common_factor / denom_p

    # 5) Umbra and penumbra radii at distance x
    if x >= L_u:
        r_u = 0.0
    else:
        r_u = R_occult * (1.0 - x / L_u)

    r_p = R_occult * (1.0 + x / L_p)

    # 6) Region classification using rho^2 to avoid sqrt when possible
    r_u2 = r_u * r_u
    r_p2 = r_p * r_p
    if rho2 <= r_u2:
        return 0.0
    if rho2 >= r_p2:
        return 1.0

    # 7) Penumbra transition (smoothstep)
    rho = math.sqrt(rho2)
    width = r_p - r_u
    if width <= EPS_1E12:
        return 0.0

    u = (rho - r_u) / width  # in (0,1)
    return u * u * (3.0 - 2.0 * u)


@njit(cache=True, nogil=True, inline="always")
def _in_umbra_from_nu(nu: float) -> int:
    """Convert a shadow factor nu into a binary umbra flag."""
    return 1 if nu <= EPS_1E6 else 0


# -------------------------------------------------------------------------
# Moon shadow wrappers (Moon-centered vectors)
# -------------------------------------------------------------------------

@njit(cache=True, nogil=True, inline="always")
def moon_shadow_factor_conical(
    rx: float, ry: float, rz: float,
    sx: float, sy: float, sz: float,
    R_moon: float,
) -> float:
    """Moon conical shadow factor (finite Sun size). Inputs must be Moon-centered."""
    return _shadow_factor_conical(rx, ry, rz, sx, sy, sz, R_moon)


@njit(cache=True, nogil=True, inline="always")
def in_moon_umbra_conical(
    rx: float, ry: float, rz: float,
    sx: float, sy: float, sz: float,
    R_moon: float,
) -> int:
    """Return 1 if in Moon umbra (total eclipse), else 0."""
    return _in_umbra_from_nu(moon_shadow_factor_conical(rx, ry, rz, sx, sy, sz, R_moon))


# -------------------------------------------------------------------------
# Earth shadow wrappers (Earth-centered vectors)
# -------------------------------------------------------------------------

@njit(cache=True, nogil=True, inline="always")
def earth_shadow_factor_conical(
    rx: float, ry: float, rz: float,
    sx: float, sy: float, sz: float,
    R_earth: float,
) -> float:
    """Earth conical shadow factor (finite Sun size). Inputs must be Earth-centered."""
    return _shadow_factor_conical(rx, ry, rz, sx, sy, sz, R_earth)


@njit(cache=True, nogil=True, inline="always")
def in_earth_umbra_conical(
    rx: float, ry: float, rz: float,
    sx: float, sy: float, sz: float,
    R_earth: float,
) -> int:
    """Return 1 if in Earth umbra (total eclipse), else 0."""
    return _in_umbra_from_nu(earth_shadow_factor_conical(rx, ry, rz, sx, sy, sz, R_earth))



# =============================================================================
# 2.                      SOLAR RADIATION PRESSURE (SRP)
# =============================================================================

@dataclass(frozen=True, slots=True, kw_only=True)
class SRPConfig:
    """
    Configuration for the SRP algorithm (environment & settings).

    Spacecraft physical properties (mass, area, Cr) belong in SpacecraftProps.
    This config controls SRP environment constants and shadowing options.
    """
    P0: float = P_SUN_1AU
    AU_m: float = AU

    # Shadowing controls
    enable_moon_eclipse: bool = True
    enable_earth_eclipse: bool = False  # turn on if you want Earth shadow too

    shadow_model: str = "conical"       # reserved for future expansion
    R_moon_m: float = R_MOON_MEAN
    R_earth_m: float = R_EARTH_MEAN


@njit(cache=True, nogil=True, inline="always")
def _min_shadow_factor(
    rx: float, ry: float, rz: float,
    sx: float, sy: float, sz: float,
    ex: float, ey: float, ez: float,
    R_moon: float,
    R_earth: float,
    enable_moon_eclipse: bool,
    enable_earth_eclipse: bool,
) -> float:
    """
    Compute combined shadow factor as min(nu_moon, nu_earth) depending on flags.
    Inputs are Moon-centered; Earth eclipse internally converts to Earth-centered.
    """
    shadow = 1.0

    if enable_moon_eclipse:
        shadow = moon_shadow_factor_conical(rx, ry, rz, sx, sy, sz, R_moon)
        if shadow <= EPS_1E6:
            return 0.0

    if enable_earth_eclipse:
        # Convert Moon-centered vectors to Earth-centered vectors
        rsc_ex = rx - ex
        rsc_ey = ry - ey
        rsc_ez = rz - ez

        rsun_ex = sx - ex
        rsun_ey = sy - ey
        rsun_ez = sz - ez

        nu_earth = earth_shadow_factor_conical(
            rsc_ex, rsc_ey, rsc_ez,
            rsun_ex, rsun_ey, rsun_ez,
            R_earth,
        )
        if nu_earth < shadow:
            shadow = nu_earth
            if shadow <= EPS_1E6:
                return 0.0

    return shadow


@njit(cache=True, nogil=True)
def accel_srp(
    rx: float, ry: float, rz: float,     # Spacecraft position (Moon-centered) [m]
    sx: float, sy: float, sz: float,     # Sun position (Moon-centered) [m]
    ex: float, ey: float, ez: float,     # Earth position (Moon-centered) [m]
    R_moon: float,                       # Moon radius [m]
    R_earth: float,                      # Earth radius [m]
    AU_m: float,                         # Astronomical Unit [m]
    P0: float,                           # SRP at 1 AU [N/m^2]
    cr: float,                           # Reflectivity coefficient [-]
    area_m2: float,                      # Cross-sectional area [m^2]
    mass_kg: float,                      # Mass [kg]
    enable_moon_eclipse: bool,
    enable_earth_eclipse: bool,
) -> Tuple[float, float, float]:
    """
    Core SRP kernel (cannonball model), optimized for Numba.

    a_vec = (P0 * Cr * A / m) * (AU / d)^2 * shadow_factor * u_hat(sun->sc)

    Notes
    -----
    - Inputs are Moon-centered (consistent with ephemeris tables).
    - Earth eclipse is computed by converting to Earth-centered vectors:
        r_sc_earth  = r_sc_moon  - r_earth_moon
        r_sun_earth = r_sun_moon - r_earth_moon
      then applying earth_shadow_factor_conical(...).
    - Combined shadow uses shadow = min(nu_moon, nu_earth).
    """
    # 1) Physical validity
    if mass_kg <= 0.0 or area_m2 <= 0.0:
        return 0.0, 0.0, 0.0

    # 2) Sun -> spacecraft vector (points away from the Sun)
    dx = rx - sx
    dy = ry - sy
    dz = rz - sz
    dist_sq = dx * dx + dy * dy + dz * dz
    if dist_sq <= 0.0:
        return 0.0, 0.0, 0.0

    dist = math.sqrt(dist_sq)
    inv_dist3 = 1.0 / (dist_sq * dist)  # 1 / d^3

    # 3) Shadowing (Moon + optional Earth)
    shadow = _min_shadow_factor(
        rx, ry, rz,
        sx, sy, sz,
        ex, ey, ez,
        R_moon, R_earth,
        enable_moon_eclipse,
        enable_earth_eclipse,
    )
    if shadow <= EPS_1E6:
        return 0.0, 0.0, 0.0

    # 4) Acceleration scaling
    # AU^2 * d_vec / d^3 matches (AU/d)^2 * u_hat(sun->sc)
    K = (P0 * cr * area_m2 * (AU_m * AU_m) / mass_kg) * shadow
    scale = K * inv_dist3
    return scale * dx, scale * dy, scale * dz



# =============================================================================
# 3.                       PUBLIC WRAPPERS & EXPORTS
# =============================================================================

def compute_srp_accel_out(
    r_sc: np.ndarray,               # Moon-centered spacecraft position [m]
    r_sun: np.ndarray,              # Moon-centered Sun position [m]
    r_earth: np.ndarray,            # Moon-centered Earth position [m] (needed if Earth eclipse enabled)
    sc_props: SpacecraftProps,
    config: SRPConfig,
    out: np.ndarray,                # shape (3,), float64
) -> None:
    """
    Allocation-free public interface: writes SRP acceleration into `out`.

    Recommended for tight propagation loops.
    """
    ax, ay, az = accel_srp(
        float(r_sc[0]), float(r_sc[1]), float(r_sc[2]),
        float(r_sun[0]), float(r_sun[1]), float(r_sun[2]),
        float(r_earth[0]), float(r_earth[1]), float(r_earth[2]),
        float(config.R_moon_m),
        float(config.R_earth_m),
        float(config.AU_m),
        float(config.P0),
        float(sc_props.cr),
        float(sc_props.area_m2),
        float(sc_props.mass_kg),
        bool(config.enable_moon_eclipse),
        bool(config.enable_earth_eclipse),
    )
    out[0] = ax
    out[1] = ay
    out[2] = az


def compute_srp_accel(
    r_sc: np.ndarray,               # Moon-centered spacecraft position [m]
    r_sun: np.ndarray,              # Moon-centered Sun position [m]
    r_earth: np.ndarray,            # Moon-centered Earth position [m] (needed if Earth eclipse enabled)
    sc_props: SpacecraftProps,
    config: SRPConfig,
) -> np.ndarray:
    """
    Convenience wrapper: returns SRP acceleration as a newly-allocated (3,) float64 array.

    Prefer `compute_srp_accel_out(...)` for hot loops.
    """
    out = np.empty(3, dtype=np.float64)
    compute_srp_accel_out(r_sc, r_sun, r_earth, sc_props, config, out)
    return out



# =============================================================================
# 4.                            PUBLIC API
# =============================================================================

__all__ = (
    # Shadow / Eclipse (Moon-centered)
    "moon_shadow_factor_conical",   # nu in [0,1]
    "in_moon_umbra_conical",        # 1 if umbra else 0

    # Shadow / Eclipse (Earth-centered)
    "earth_shadow_factor_conical",  # nu in [0,1]
    "in_earth_umbra_conical",       # 1 if umbra else 0

    # Solar Radiation Pressure (SRP)
    "SRPConfig",                    # SRP settings/config
    "accel_srp",                    # Numba kernel -> (ax, ay, az)
    "compute_srp_accel_out",        # alloc-free wrapper (writes to out)
    "compute_srp_accel",            # convenience wrapper (allocates)
)
