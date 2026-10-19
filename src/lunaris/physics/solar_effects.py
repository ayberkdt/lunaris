# lunaris.physics.solar_effects
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
   - Returns the visible solar-disk fraction ``nu`` in ``[0, 1]``:
       * ``nu = 0`` : full umbra (total eclipse)
       * ``nu = 1`` : full sunlight
       * ``0 < nu < 1`` : penumbra, or annular beyond the umbra apex
         (apparent-disk overlap area, Montenbruck & Gill 2000, sec. 3.4)

Coordinate and sign conventions
-------------------------------
- Unless otherwise stated, inputs are Moon-centered vectors in an inertial frame.
- ``r_sc = (rx, ry, rz)``  : spacecraft position w.r.t. the Moon center [m]
- ``r_sun = (sx, sy, sz)`` : Sun position w.r.t. the Moon center [m]
- The Sun→spacecraft direction is ``d_vec = r_sc - r_sun``.
  The SRP acceleration points along ``d_vec`` (i.e., away from the Sun).

Earth eclipse (on by default in ``SRPConfig``)
----------------------------------------------
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
- The penumbra is the overlap area of the apparent solar and occulter disks seen
  from the spacecraft (uniform solar disk, spherical occulter, geometric light
  paths). Against an exact spherical-cap integral the planar overlap is within
  3e-4 in ``nu`` for lunar and Earth shadows near 1 AU. Not modelled: solar limb
  darkening, occulter oblateness, atmospheric refraction and light time.
"""



# =============================================================================
# 0.                                IMPORTS
# =============================================================================

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numba import njit

from lunaris.common.constants import (
    AU,
    EPS_1E6,
    EPS_1E24,
    P_SUN_1AU,
    R_EARTH_MEAN,
    R_MOON_MEAN,
    R_SUN_MEAN,
)
from lunaris.common.type_defs import SpacecraftProps

# =============================================================================
# 1.                       SHADOW / ECLIPSE GEOMETRY
# =============================================================================

# -------------------------------------------------------------------------
# Conical eclipse model (finite Sun size) — shared implementation
#
# Returns nu in [0, 1], the fraction of the solar disk visible from the
# spacecraft past one spherical occulter:
#   nu = 1.0  -> full sunlight
#   nu = 0.0  -> full shadow (umbra)
#   0<nu<1    -> partial shadow (penumbra, or annular beyond the umbra apex)
#
# Notes:
# - Apparent-radius overlap of two uniform disks seen from the spacecraft
#   (Montenbruck & Gill 2000, sec. 3.4); no solar limb darkening.
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

    With ``a`` the apparent solar radius, ``b`` the apparent occulter radius
    and ``c`` their angular separation (all seen from the spacecraft), the
    occulted part of the solar disk is the planar lens area of two circles of
    radii ``a`` and ``b`` at centre distance ``c``; ``nu = 1 - area / (pi a^2)``
    (Montenbruck & Gill 2000, sec. 3.4). ``c >= a + b`` is full
    sunlight (exactly 1), ``c <= b - a`` full umbra (exactly 0) and
    ``c <= a - b`` the annular case beyond the umbra apex (``1 - b^2/a^2``).
    Angles use ``atan2`` so the near-collinear geometry keeps full precision.
    """
    # 1) Spacecraft -> Sun vector d = r_sun - r_sc and the spacecraft radius
    dx = sx - rx
    dy = sy - ry
    dz = sz - rz
    d_sq = dx * dx + dy * dy + dz * dz
    r_sq = rx * rx + ry * ry + rz * rz
    if d_sq <= EPS_1E24 or r_sq <= EPS_1E24 or R_occult <= 0.0:
        return 1.0  # degenerate geometry -> assume sunlight

    sin_a = R_SUN_MEAN / math.sqrt(d_sq)
    if sin_a >= 1.0:
        return 1.0  # inside the solar sphere: no meaningful occultation

    # 2) Cheap exit on the sunlit side. dot = (-r_sc) . d = |r| |d| cos(c).
    # For c >= 90 deg an overlap needs b > 90 deg - a, i.e.
    # (R/|r|)^2 > 1 - sin(a)^2. Otherwise the whole disk is visible.
    dot = -(rx * dx + ry * dy + rz * dz)
    R_sq = R_occult * R_occult
    if dot <= 0.0 and r_sq * (1.0 - sin_a * sin_a) >= R_sq:
        return 1.0

    # 3) Apparent radii and separation [rad]
    a = math.asin(sin_a)
    if r_sq <= R_sq:
        b = 0.5 * math.pi  # on/inside the occulter surface: horizon half-space
    else:
        b = math.asin(R_occult / math.sqrt(r_sq))
    # |(-r_sc) x d| = |r_sc x d|
    cx = ry * dz - rz * dy
    cy = rz * dx - rx * dz
    cz = rx * dy - ry * dx
    c = math.atan2(math.sqrt(cx * cx + cy * cy + cz * cz), dot)

    # 4) Region classification
    if c >= a + b:
        return 1.0  # disks apart: full sunlight
    if c <= b - a:
        return 0.0  # solar disk entirely behind the occulter: umbra
    if c <= a - b:
        return 1.0 - (b * b) / (a * a)  # occulter inside the solar disk: annular

    # 5) Partial overlap: lens area of circles (a, b) at centre distance c
    x = (c * c + a * a - b * b) / (2.0 * c)
    y_sq = a * a - x * x
    y = math.sqrt(y_sq) if y_sq > 0.0 else 0.0
    ca = x / a
    if ca > 1.0:
        ca = 1.0
    elif ca < -1.0:
        ca = -1.0
    cb = (c - x) / b
    if cb > 1.0:
        cb = 1.0
    elif cb < -1.0:
        cb = -1.0
    area = a * a * math.acos(ca) + b * b * math.acos(cb) - c * y
    nu = 1.0 - area / (math.pi * a * a)
    if nu < 0.0:
        return 0.0
    if nu > 1.0:
        return 1.0
    return nu


@njit(cache=True, nogil=True)
def conical_shadow_angles(
    rx: float, ry: float, rz: float,
    sx: float, sy: float, sz: float,
    R_occult: float,
) -> tuple[float, float, float]:
    """Apparent solar radius ``a``, occulter radius ``b`` and their separation ``c`` [rad].

    Same frame and inputs as :func:`_shadow_factor_conical` (spacecraft and Sun
    relative to the occulter centre, one set of inertial axes), and the same
    float64 operations, so the contacts below are the points where that
    shadow factor switches region. ``b`` is pi/2 on or inside the occulter.
    Degenerate geometry, where the shadow factor is 1 (spacecraft at the
    occulter centre, inside the solar sphere, ``R_occult <= 0``), returns
    NaN for all three.
    """
    dx = sx - rx
    dy = sy - ry
    dz = sz - rz
    d_sq = dx * dx + dy * dy + dz * dz
    r_sq = rx * rx + ry * ry + rz * rz
    if d_sq <= EPS_1E24 or r_sq <= EPS_1E24 or R_occult <= 0.0:
        return math.nan, math.nan, math.nan
    sin_a = R_SUN_MEAN / math.sqrt(d_sq)
    if sin_a >= 1.0:
        return math.nan, math.nan, math.nan
    a = math.asin(sin_a)
    if r_sq <= R_occult * R_occult:
        b = 0.5 * math.pi
    else:
        b = math.asin(R_occult / math.sqrt(r_sq))
    cx = ry * dz - rz * dy
    cy = rz * dx - rx * dz
    cz = rx * dy - ry * dx
    c = math.atan2(math.sqrt(cx * cx + cy * cy + cz * cz), -(rx * dx + ry * dy + rz * dz))
    return a, b, c


def conical_shadow_boundaries(
    rx: float, ry: float, rz: float,
    sx: float, sy: float, sz: float,
    R_occult: float,
) -> tuple[float, float]:
    """Signed distances [rad] of the geometry from the conical shadow's two contacts.

    With the angles of :func:`conical_shadow_angles` it returns
    ``(c - (a + b), c - |b - a|)``: the first is zero where the penumbra
    begins (the disks touch from outside), the second where the umbra
    (``b > a``) or the annular phase (``a > b``) begins. The shadow factor of
    one occulter is smooth except at these two zeros, so an integrator that
    stops there never steps across a kink of the radiation force. Degenerate
    geometry (shadow factor 1) returns ``(1, 1)``: no contact.

    Ported from Sidera (2ddf461e); here on the shadow kernel's own arithmetic.
    """
    a, b, c = conical_shadow_angles(rx, ry, rz, sx, sy, sz, R_occult)
    if not math.isfinite(c):
        return 1.0, 1.0
    return c - (a + b), c - abs(b - a)


#: Steps across one penumbra passage for an adaptive integrator (see
#: :func:`penumbra_max_step_s`). Sidera's measurement (2026-10-12, 400 km Venus
#: orbit with 30 shadow passages a day, against TudatPy converged to 2.2e-5 m):
#: with restarts at the contacts but no step cap, 2.4e-4 m (rtol 1e-11) and
#: 1.7e-4 m (rtol 1e-12); with 5 steps, 8.2e-6 m and 9.7e-6 m; with 20,
#: 6.4e-6 m and 7.1e-6 m, for 25 % more right-hand-side evaluations than 5.
PENUMBRA_STEPS = 20


def penumbra_max_step_s(
    r_m: np.ndarray, v_m_s: np.ndarray, sun_m: np.ndarray, R_occult: float,
) -> float | None:
    """Largest step [s] that resolves the penumbra the spacecraft is entering or crossing.

    ``r_m``/``v_m_s`` are the spacecraft position and velocity and ``sun_m``
    the Sun, all relative to the occulter on one set of inertial axes. The
    shadow factor grows like a power 3/2 of the time from a contact, so a
    step spanning the whole passage is inaccurate even when no step contains
    the contact itself. The passage lasts about ``2 a / omega`` (``a`` the
    apparent solar radius, ``omega = |r x v| / |r|^2`` the spacecraft's
    angular rate about the occulter; longer at grazing geometry, where this
    estimate is the conservative one), divided into `PENUMBRA_STEPS`.
    Returns ``None`` unless the spacecraft is in the penumbra a moment later
    (a thousandth of that duration ahead along ``v``), so the step stays free
    in sunlight and in the umbra. Ported from Sidera (2ddf461e).
    """
    r = np.asarray(r_m, dtype=np.float64)
    v = np.asarray(v_m_s, dtype=np.float64)
    s = np.asarray(sun_m, dtype=np.float64)
    r_sq = float(r @ r)
    omega = float(np.linalg.norm(np.cross(r, v))) / r_sq if r_sq > 0.0 else 0.0
    distance = float(np.linalg.norm(s - r))
    if omega <= 0.0 or distance <= R_SUN_MEAN:
        return None
    duration = 2.0 * math.asin(R_SUN_MEAN / distance) / omega
    ahead = r + v * (1.0e-3 * duration)
    outer, inner = conical_shadow_boundaries(
        float(ahead[0]), float(ahead[1]), float(ahead[2]), float(s[0]), float(s[1]), float(s[2]), R_occult,
    )
    if outer < 0.0 < inner:
        return duration / PENUMBRA_STEPS
    return None


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
    # Earth's shadow (lunar eclipses) dims SRP like it dims albedo and thermal IR.
    enable_earth_eclipse: bool = True

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

    # A zero/collapsed Earth vector (no ephemeris table) must not place the
    # Earth at the Moon's centre; the dynamics engine applies the same gate.
    if enable_earth_eclipse and (ex * ex + ey * ey + ez * ez) > 1.0e12:
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
) -> tuple[float, float, float]:
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

    # Shadow contacts for adaptive integrators (vectors relative to the occulter)
    "conical_shadow_angles",        # apparent solar/occulter radii and separation [rad]
    "conical_shadow_boundaries",    # signed distances from the two contacts [rad]
    "PENUMBRA_STEPS",               # steps per penumbra passage
    "penumbra_max_step_s",          # step cap while crossing a penumbra [s]

    # Solar Radiation Pressure (SRP)
    "SRPConfig",                    # SRP settings/config
    "accel_srp",                    # Numba kernel -> (ax, ay, az)
    "compute_srp_accel_out",        # alloc-free wrapper (writes to out)
    "compute_srp_accel",            # convenience wrapper (allocates)
)
