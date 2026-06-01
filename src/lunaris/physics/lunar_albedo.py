# lunaris/physics/lunar_albedo.py
"""
Lambertian lunar albedo radiation pressure
==========================================

This module implements a non-gravitational **lunar albedo** perturbation: the
radiation pressure exerted on a spacecraft by *solar radiation reflected from
the lunar surface*. It is the reflected-sunlight counterpart of the thermal IR
model in :mod:`lunaris.physics.thermal_ir` and shares its facet discretization.

Lunar albedo is **not** a gravitational effect. It belongs with SRP and thermal
IR, not with spherical harmonics, surrogate gravity, or tides.

Model
-----
The Moon is discretized into spherical latitude-longitude facets (reused from
:func:`lunaris.physics.thermal_ir.build_latlon_facets`). For each facet ``i``
with center ``r_i``, outward unit normal ``n_i``, area ``dA_i`` and local
bolometric albedo ``A_i``, evaluated in the Moon-fixed frame:

    s_i      = r_sc - r_i                       (facet -> spacecraft)
    u_i      = s_i / |s_i|,        d_i = |s_i|
    h_i      = (r_sun - r_i) / |r_sun - r_i|    (facet -> Sun)
    mu_sun_i = n_i . h_i                         (solar incidence cosine)
    mu_view_i= n_i . u_i                         (spacecraft view cosine)

A facet contributes only when it is both sunlit and visible above the local
horizon, i.e. ``mu_sun_i > 0`` and ``mu_view_i > 0``.

The incident solar irradiance on the facet is ``E_sun_i = S_i * mu_sun_i`` with
``S_i = S_1AU * (AU / |r_sun - r_i|)^2`` (sun-distance scaling optional). For a
Lambertian reflector the reflected radiant exitance is ``M_i = A_i * E_sun_i``
and the irradiance reaching the spacecraft is

    dE_i = (M_i / pi) * mu_view_i * dA_i / d_i^2 .

Under a cannonball spacecraft approximation the acceleration contribution is

    da_i = (C_R_albedo * A_eff / (m_sc * c)) * dE_i * u_i ,

pointing from the reflecting facet toward the spacecraft (the direction of
photon propagation). The total albedo acceleration is the facet sum, which the
dynamics layer rotates from the Moon-fixed frame back to the inertial
integration frame.

An optional lunar-eclipse (Earth-umbra) dimming factor is applied once per call
using the same conical shadow geometry as the SRP eclipse model.

Limitations
-----------
This is a Lambertian (isotropic-reflectance) model. It does **not** include:
bidirectional reflectance distribution functions beyond Lambert, wavelength
dependence, surface roughness, terrain self-shadowing beyond the per-facet
incidence/visibility cutoffs, real-time photometric phase functions, multiple
scattering, or local topography. Per-facet albedo is the only spatial input.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import numpy.typing as npt
from numba import njit

from lunaris.common.constants import AU, C_LIGHT, PI, R_EARTH_MEAN, SOLAR_FLUX_1AU
from lunaris.physics.solar_effects import earth_shadow_factor_conical

# Facet geometry is shared with the thermal IR model (same spherical lat-lon
# discretization with exact band areas). Reusing it keeps the two surface
# radiation models consistent and avoids duplicating the cell-area math.
from lunaris.physics.thermal_ir import build_latlon_facets


# Albedo source modes describe how the per-facet albedo array A_i is built at
# setup time. They are setup/config concerns only: the Numba kernels below are
# source-agnostic and operate on a precomputed facet_albedo array.
ALBEDO_SOURCE_CONSTANT = 0
ALBEDO_SOURCE_GRID = 1
ALBEDO_SOURCE_SCALED_DN = 2


def normalize_albedo_mode(mode: str) -> int:
    """Map an albedo-source mode string to its integer code."""
    m = str(mode).strip().lower()
    if m in {"constant", "constant_albedo"}:
        return ALBEDO_SOURCE_CONSTANT
    if m in {"grid", "albedo_grid"}:
        return ALBEDO_SOURCE_GRID
    if m in {"scaled_dn", "scaled_dn_grid", "dn_grid", "dn"}:
        return ALBEDO_SOURCE_SCALED_DN
    raise ValueError(
        "albedo_mode must be one of 'constant_albedo', 'albedo_grid', "
        f"or 'scaled_dn_grid'. Got {mode!r}."
    )


@njit(cache=True, nogil=True)
def albedo_single_facet_accel_numba(
    rx: float,
    ry: float,
    rz: float,
    sx: float,
    sy: float,
    sz: float,
    fx: float,
    fy: float,
    fz: float,
    nx: float,
    ny: float,
    nz: float,
    area_facet_m2: float,
    facet_albedo: float,
    pressure_coefficient: float,
    spacecraft_area_m2: float,
    spacecraft_mass_kg: float,
    solar_flux_1au_W_m2: float,
    au_m: float,
    c_light_m_s: float,
    include_sun_distance_scaling: bool,
) -> Tuple[float, float, float]:
    """Acceleration contribution from one Lambertian reflecting facet.

    All vectors are Moon-fixed. Returns ``(0, 0, 0)`` unless the facet is both
    sunlit (``mu_sun > 0``) and visible from the spacecraft (``mu_view > 0``).
    """
    if (
        area_facet_m2 <= 0.0
        or facet_albedo <= 0.0
        or pressure_coefficient == 0.0
        or spacecraft_area_m2 <= 0.0
        or spacecraft_mass_kg <= 0.0
        or c_light_m_s <= 0.0
    ):
        return 0.0, 0.0, 0.0

    # Facet -> spacecraft.
    ssx = rx - fx
    ssy = ry - fy
    ssz = rz - fz
    d2 = ssx * ssx + ssy * ssy + ssz * ssz
    if d2 <= 1.0:
        return 0.0, 0.0, 0.0
    inv_d = 1.0 / math.sqrt(d2)
    ux = ssx * inv_d
    uy = ssy * inv_d
    uz = ssz * inv_d

    mu_view = nx * ux + ny * uy + nz * uz
    if mu_view <= 0.0:
        return 0.0, 0.0, 0.0

    # Facet -> Sun (per-facet, exact: includes the small lunar-radius parallax).
    hx = sx - fx
    hy = sy - fy
    hz = sz - fz
    h2 = hx * hx + hy * hy + hz * hz
    if h2 <= 1.0:
        return 0.0, 0.0, 0.0
    inv_h = 1.0 / math.sqrt(h2)
    mu_sun = (nx * hx + ny * hy + nz * hz) * inv_h
    if mu_sun <= 0.0:
        return 0.0, 0.0, 0.0

    solar_flux = solar_flux_1au_W_m2
    if include_sun_distance_scaling:
        solar_flux = solar_flux_1au_W_m2 * (au_m * au_m) / h2
    if solar_flux <= 0.0:
        return 0.0, 0.0, 0.0

    # Reflected Lambertian exitance [W/m^2] and irradiance reaching the SC.
    exitance = facet_albedo * solar_flux * mu_sun
    irradiance = (exitance / PI) * mu_view * area_facet_m2 / d2
    scale = (
        pressure_coefficient
        * (spacecraft_area_m2 / spacecraft_mass_kg)
        * irradiance
        / c_light_m_s
    )
    return scale * ux, scale * uy, scale * uz


@njit(cache=True, nogil=True)
def accel_albedo_facets_numba(
    rx: float,
    ry: float,
    rz: float,
    sunx: float,
    suny: float,
    sunz: float,
    earthx: float,
    earthy: float,
    earthz: float,
    facet_pos_m: np.ndarray,
    facet_normals: np.ndarray,
    facet_areas_m2: np.ndarray,
    facet_albedo: np.ndarray,
    pressure_coefficient: float,
    spacecraft_area_m2: float,
    spacecraft_mass_kg: float,
    solar_flux_1au_W_m2: float,
    au_m: float,
    c_light_m_s: float,
    r_earth_m: float,
    include_sun_distance_scaling: bool,
    enable_eclipse: bool,
) -> Tuple[float, float, float]:
    """Sum Lambertian lunar albedo acceleration over precomputed facets.

    Inputs are Moon-fixed. ``facet_albedo`` is the per-facet bolometric albedo
    A_i in [0, 1]. When ``enable_eclipse`` is set and a valid Earth vector is
    supplied, a single lunar-eclipse (Earth-umbra) dimming factor scales the
    whole sum.
    """
    if (
        pressure_coefficient == 0.0
        or spacecraft_area_m2 <= 0.0
        or spacecraft_mass_kg <= 0.0
        or c_light_m_s <= 0.0
    ):
        return 0.0, 0.0, 0.0

    n = facet_areas_m2.shape[0]
    if n <= 0 or facet_albedo.shape[0] < n:
        return 0.0, 0.0, 0.0

    # Optional lunar eclipse: the Earth shadowing the Moon's dayside. The Moon
    # (<= ~1738 km) is tiny compared with the Earth umbra at lunar distance, so
    # a single factor evaluated at the Moon center is an accurate, cheap proxy
    # shared by every facet. Vectors are converted to Earth-centered to match
    # the conical-shadow contract reused from the SRP eclipse model.
    eclipse_factor = 1.0
    if enable_eclipse:
        earth_norm2 = earthx * earthx + earthy * earthy + earthz * earthz
        if earth_norm2 > 1.0:
            rmoon_ex = -earthx
            rmoon_ey = -earthy
            rmoon_ez = -earthz
            rsun_ex = sunx - earthx
            rsun_ey = suny - earthy
            rsun_ez = sunz - earthz
            eclipse_factor = earth_shadow_factor_conical(
                rmoon_ex, rmoon_ey, rmoon_ez,
                rsun_ex, rsun_ey, rsun_ez,
                r_earth_m,
            )
            if eclipse_factor <= 0.0:
                return 0.0, 0.0, 0.0

    ax = 0.0
    ay = 0.0
    az = 0.0
    for i in range(n):
        a_i = facet_albedo[i]
        if a_i <= 0.0:
            continue
        tx, ty, tz = albedo_single_facet_accel_numba(
            rx,
            ry,
            rz,
            sunx,
            suny,
            sunz,
            facet_pos_m[i, 0],
            facet_pos_m[i, 1],
            facet_pos_m[i, 2],
            facet_normals[i, 0],
            facet_normals[i, 1],
            facet_normals[i, 2],
            facet_areas_m2[i],
            a_i,
            pressure_coefficient,
            spacecraft_area_m2,
            spacecraft_mass_kg,
            solar_flux_1au_W_m2,
            au_m,
            c_light_m_s,
            include_sun_distance_scaling,
        )
        ax += tx
        ay += ty
        az += tz

    if eclipse_factor != 1.0:
        ax *= eclipse_factor
        ay *= eclipse_factor
        az *= eclipse_factor
    return ax, ay, az


def calc_albedo_accel(
    r_sc_fixed: npt.ArrayLike,
    r_sun_fixed: npt.ArrayLike,
    facet_pos_m: np.ndarray,
    facet_normals: np.ndarray,
    facet_areas_m2: np.ndarray,
    facet_albedo: npt.ArrayLike,
    *,
    r_earth_fixed: npt.ArrayLike | None = None,
    pressure_coefficient: float = 1.0,
    spacecraft_area_m2: float = 1.0,
    spacecraft_mass_kg: float = 1000.0,
    solar_flux_1au_W_m2: float = SOLAR_FLUX_1AU,
    au_m: float = AU,
    c_light_m_s: float = C_LIGHT,
    r_earth_m: float = R_EARTH_MEAN,
    include_sun_distance_scaling: bool = True,
    enable_eclipse: bool = False,
) -> np.ndarray:
    """Python convenience wrapper for the facet-summed albedo acceleration.

    All inputs are Moon-fixed. Returns the Moon-fixed acceleration ``(3,)``;
    callers integrating in an inertial frame must rotate it back themselves.
    """
    r = np.asarray(r_sc_fixed, dtype=np.float64).reshape(3)
    s = np.asarray(r_sun_fixed, dtype=np.float64).reshape(3)
    if r_earth_fixed is None:
        e = np.zeros(3, dtype=np.float64)
    else:
        e = np.asarray(r_earth_fixed, dtype=np.float64).reshape(3)
    alb = np.ascontiguousarray(np.asarray(facet_albedo, dtype=np.float64).reshape(-1))

    ax, ay, az = accel_albedo_facets_numba(
        float(r[0]),
        float(r[1]),
        float(r[2]),
        float(s[0]),
        float(s[1]),
        float(s[2]),
        float(e[0]),
        float(e[1]),
        float(e[2]),
        np.ascontiguousarray(facet_pos_m, dtype=np.float64),
        np.ascontiguousarray(facet_normals, dtype=np.float64),
        np.ascontiguousarray(facet_areas_m2, dtype=np.float64),
        alb,
        float(pressure_coefficient),
        float(spacecraft_area_m2),
        float(spacecraft_mass_kg),
        float(solar_flux_1au_W_m2),
        float(au_m),
        float(c_light_m_s),
        float(r_earth_m),
        bool(include_sun_distance_scaling),
        bool(enable_eclipse),
    )
    return np.asarray((ax, ay, az), dtype=np.float64)


__all__ = (
    "ALBEDO_SOURCE_CONSTANT",
    "ALBEDO_SOURCE_GRID",
    "ALBEDO_SOURCE_SCALED_DN",
    "normalize_albedo_mode",
    "build_latlon_facets",
    "albedo_single_facet_accel_numba",
    "accel_albedo_facets_numba",
    "calc_albedo_accel",
)
