# lunaris/physics/surface_radiation_quadrature.py
"""
Visible-disk quadrature for lunar albedo and thermal-IR radiation pressure
==========================================================================

The facet models in :mod:`lunaris.physics.lunar_albedo` and
:mod:`lunaris.physics.thermal_ir` sum a fixed latitude-longitude grid of point
sources. That is accurate only while a facet is small compared with the
spacecraft altitude: on the historical 18 x 36 grid a facet is up to ~300 km
across, so on a sunlit disk at 100 km altitude the facet sum is 0.39x-1.25x of
the converged acceleration depending on where the spacecraft sits over the
grid, and at 30 km 0.04x-1.05x. It diverges like ``dA / h^2`` as the altitude
drops.

This module integrates the same Lambertian physics over the part of the lunar
sphere the spacecraft actually sees, in the spacecraft's own solid angle, with a
node count that does not depend on altitude.

Formulation
-----------
A Lambertian surface point ``p`` with radiant exitance ``M(p)`` has radiance
``L = M / pi``. A cannonball of cross-section ``A`` receives the momentum flux
``L dOmega / c`` from each solid-angle element, directed from ``p`` towards the
spacecraft, so

    a = (C * A / (m c)) * integral_disk  (M(p) / pi) * (-w)  dOmega ,

where ``w`` is the unit line of sight from the spacecraft to the surface and the
disk is the Moon's apparent disk (half-angle ``beta``, ``sin beta = R / r``).
This is the limit of the facet sum: ``dOmega = mu_view dA / d^2`` converts one
into the other. For uniform ``M`` the integral is exactly ``M (R/r)^2`` along
the radial direction (the analytic check the tests pin).

Quadrature
----------
Directions are parameterised by ``x = cos(alpha)`` (``alpha`` measured from
nadir) and azimuth ``phi``, with ``dOmega = dx dphi``. The surface hit point
moves like ``sqrt(x - cos beta)`` near the limb, so ``x = cos beta +
(1 - cos beta) v^2`` makes the integrand smooth in ``v``; ``v`` uses
Gauss-Legendre nodes on ``[0, 1]`` and ``phi`` the periodic midpoint rule. The
day/night terminator makes the integrand continuous but not smooth, which
bounds the convergence rate where it crosses the disk.

Spatial inputs (per-cell albedo or temperature) are read from the same
latitude-longitude cell arrays the facet models use (south-to-north, longitude
``0 -> 2 pi``, row-major, see :func:`lunaris.physics.thermal_ir.build_latlon_facets`),
by the cell containing each hit point.

Everything is Moon-fixed; the dynamics layer rotates the result to inertial.
The optional lunar-eclipse factor is the facet models' Moon-centre Earth-umbra
factor, so both integrations see the same eclipse.

Ported from the Sidera fork (``sidera.physics.radiation.surface_radiation_quadrature``)
and re-verified independently (analytic uniform-sphere limit for thermal IR, a
sub-satellite-centred brute-force surface integral for albedo).
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit

from lunaris.common.constants import PI
from lunaris.physics.solar_effects import earth_shadow_factor_conical
from lunaris.physics.thermal_ir import (
    THERMAL_MODE_CONSTANT,
    THERMAL_MODE_EQUILIBRIUM,
    THERMAL_MODE_TEMPERATURE_GRID,
)

#: Default node counts: Gauss-Legendre nodes in the nadir-angle variable and
#: azimuth nodes (16 x 32 = 512 directions per evaluation).
DEFAULT_RADIAL_NODES: int = 16
DEFAULT_AZIMUTH_NODES: int = 32

# Exitance selector for reflected sunlight (the thermal modes use 0/1/2).
_KIND_ALBEDO = 10


def gauss_legendre_unit_interval(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre nodes and weights on ``[0, 1]`` (float64, C-contiguous).

    Reference: Harris et al. (2020), numpy.polynomial.legendre.leggauss [LUNARIS-ALG-NUM-001].
    """
    count = int(n)
    if count < 1:
        raise ValueError(f"quadrature node count must be >= 1, got {n!r}.")
    nodes, weights = np.polynomial.legendre.leggauss(count)
    return (
        np.ascontiguousarray(0.5 * (nodes + 1.0), dtype=np.float64),
        np.ascontiguousarray(0.5 * weights, dtype=np.float64),
    )


@njit(cache=True, nogil=True)
def _cell_index(px: float, py: float, pz: float, n_lat: int, n_lon: int) -> int:
    """Row-major index of the latitude-longitude cell containing a surface point."""
    radius = math.sqrt(px * px + py * py + pz * pz)
    s = pz / radius
    if s > 1.0:
        s = 1.0
    elif s < -1.0:
        s = -1.0
    lat = math.asin(s)
    lon = math.atan2(py, px)
    if lon < 0.0:
        lon += 2.0 * PI
    i_lat = int((lat + 0.5 * PI) / PI * n_lat)
    if i_lat >= n_lat:
        i_lat = n_lat - 1
    elif i_lat < 0:
        i_lat = 0
    i_lon = int(lon / (2.0 * PI) * n_lon)
    if i_lon >= n_lon:
        i_lon = n_lon - 1
    elif i_lon < 0:
        i_lon = 0
    return i_lat * n_lon + i_lon


@njit(cache=True, nogil=True)
def _moon_centre_eclipse_factor(
    sunx: float, suny: float, sunz: float,
    earthx: float, earthy: float, earthz: float,
    r_earth_m: float,
) -> float:
    """Earth-umbra factor at the Moon centre (Moon-fixed inputs), as the facet models use."""
    earth_norm2 = earthx * earthx + earthy * earthy + earthz * earthz
    if earth_norm2 <= 1.0:
        return 1.0
    return earth_shadow_factor_conical(
        -earthx, -earthy, -earthz,
        sunx - earthx, suny - earthy, sunz - earthz,
        r_earth_m,
    )


@njit(cache=True, nogil=True)
def _visible_disk_quadrature(
    rx: float, ry: float, rz: float,
    sunx: float, suny: float, sunz: float,
    radius_m: float,
    cell_values: np.ndarray,
    n_lat: int,
    n_lon: int,
    kind: int,
    const_value: float,
    solar_flux_1au_W_m2: float,
    au_m: float,
    include_sun_distance_scaling: bool,
    solar_scale: float,
    surface_emissivity: float,
    surface_albedo: float,
    floor_exitance: float,
    sigma_sb: float,
    radial_nodes: np.ndarray,
    radial_weights: np.ndarray,
    n_azimuth: int,
) -> tuple[float, float, float]:
    """Integral of ``(M / pi) * (-w) dOmega`` over the visible disk [W/m^2].

    ``kind`` selects the exitance ``M(p)``:
      * ``_KIND_ALBEDO``: reflected sunlight ``A(p) S_p mu_sun(p)`` with the
        per-point Sun vector, ``A(p)`` from ``cell_values`` (or ``const_value``
        when no cell grid is given);
      * THERMAL_MODE_CONSTANT / _EQUILIBRIUM / _TEMPERATURE_GRID: the thermal
        facet model's exitance for that mode (equilibrium uses the Moon-centre
        Sun direction, like the facet kernel).
    ``solar_scale`` multiplies the solar input (eclipse dimming).
    """
    r2 = rx * rx + ry * ry + rz * rz
    R = radius_m
    if r2 <= R * R or R <= 0.0 or n_azimuth < 1:
        return 0.0, 0.0, 0.0
    r = math.sqrt(r2)
    zx = rx / r
    zy = ry / r
    zz = rz / r

    # Orthonormal e1, e2 perpendicular to the radial direction z.
    if abs(zx) < 0.9:
        tx, ty, tz = 1.0, 0.0, 0.0
    else:
        tx, ty, tz = 0.0, 1.0, 0.0
    dot = tx * zx + ty * zy + tz * zz
    e1x = tx - dot * zx
    e1y = ty - dot * zy
    e1z = tz - dot * zz
    n1 = math.sqrt(e1x * e1x + e1y * e1y + e1z * e1z)
    e1x /= n1
    e1y /= n1
    e1z /= n1
    e2x = zy * e1z - zz * e1y
    e2y = zz * e1x - zx * e1z
    e2z = zx * e1y - zy * e1x

    # cos(beta) = sqrt(r^2 - R^2) / r, without cancellation.
    rr_minus_RR = (r - R) * (r + R)
    cos_beta = math.sqrt(rr_minus_RR) / r
    span = 1.0 - cos_beta

    sun_norm2 = sunx * sunx + suny * suny + sunz * sunz
    shx = 0.0
    shy = 0.0
    shz = 0.0
    centre_flux = solar_flux_1au_W_m2
    if sun_norm2 > 1.0:
        inv_sun = 1.0 / math.sqrt(sun_norm2)
        shx = sunx * inv_sun
        shy = suny * inv_sun
        shz = sunz * inv_sun
        if include_sun_distance_scaling:
            centre_flux = solar_flux_1au_W_m2 * (au_m * au_m / sun_norm2)
    if centre_flux < 0.0:
        centre_flux = 0.0
    centre_flux *= solar_scale

    const_thermal = 0.0
    if kind == THERMAL_MODE_CONSTANT and const_value > 0.0:
        t2 = const_value * const_value
        const_thermal = surface_emissivity * sigma_sb * t2 * t2

    has_cells = n_lat > 0 and n_lon > 0 and cell_values.shape[0] >= n_lat * n_lon
    if kind == THERMAL_MODE_TEMPERATURE_GRID and not has_cells:
        return 0.0, 0.0, 0.0
    d_phi = 2.0 * PI / n_azimuth
    # Azimuth nodes phi_j = (j + 1/2) d_phi by an angle-addition recurrence
    # (two trig calls per evaluation instead of two per node; the rounding
    # drift is ~j ulp, far below the quadrature error).
    cos_half = math.cos(0.5 * d_phi)
    sin_half = math.sin(0.5 * d_phi)
    cos_step = math.cos(d_phi)
    sin_step = math.sin(d_phi)

    sx = 0.0
    sy = 0.0
    sz = 0.0
    for k in range(radial_nodes.shape[0]):
        v = radial_nodes[k]
        x = cos_beta + span * v * v            # cos(alpha)
        weight_x = radial_weights[k] * 2.0 * span * v * d_phi
        sin_alpha_sq = 1.0 - x * x
        sin_alpha = math.sqrt(sin_alpha_sq) if sin_alpha_sq > 0.0 else 0.0
        disc = r2 * x * x - rr_minus_RR
        if disc < 0.0:
            disc = 0.0
        # Near-side range along the ray, rationalised: (r x)^2 - disc = r^2 - R^2.
        t_hit = rr_minus_RR / (r * x + math.sqrt(disc))
        cphi = cos_half
        sphi = sin_half
        for _j in range(n_azimuth):
            if _j > 0:
                cphi, sphi = cphi * cos_step - sphi * sin_step, sphi * cos_step + cphi * sin_step
            # Line of sight w = -x z + sin(alpha) (cos(phi) e1 + sin(phi) e2)
            wx = -x * zx + sin_alpha * (cphi * e1x + sphi * e2x)
            wy = -x * zy + sin_alpha * (cphi * e1y + sphi * e2y)
            wz = -x * zz + sin_alpha * (cphi * e1z + sphi * e2z)
            px = rx + t_hit * wx
            py = ry + t_hit * wy
            pz = rz + t_hit * wz

            exitance = 0.0
            if kind == _KIND_ALBEDO:
                albedo = const_value
                if has_cells:
                    albedo = cell_values[_cell_index(px, py, pz, n_lat, n_lon)]
                if albedo <= 0.0:
                    continue
                hx = sunx - px
                hy = suny - py
                hz = sunz - pz
                h2 = hx * hx + hy * hy + hz * hz
                if h2 <= 1.0:
                    continue
                mu_sun = (px * hx + py * hy + pz * hz) / (R * math.sqrt(h2))
                if mu_sun <= 0.0:
                    continue
                flux = solar_flux_1au_W_m2
                if include_sun_distance_scaling:
                    flux = solar_flux_1au_W_m2 * (au_m * au_m) / h2
                exitance = albedo * flux * solar_scale * mu_sun
            elif kind == THERMAL_MODE_CONSTANT:
                exitance = const_thermal
            elif kind == THERMAL_MODE_EQUILIBRIUM:
                mu_sun = (px * shx + py * shy + pz * shz) / R
                absorbed = 0.0
                if mu_sun > 0.0:
                    absorbed = (1.0 - surface_albedo) * centre_flux * mu_sun
                exitance = absorbed + floor_exitance
            elif kind == THERMAL_MODE_TEMPERATURE_GRID:
                temp = cell_values[_cell_index(px, py, pz, n_lat, n_lon)]
                if temp > 0.0:
                    t2 = temp * temp
                    exitance = surface_emissivity * sigma_sb * t2 * t2
            if exitance <= 0.0:
                continue
            radiance_weight = exitance / PI * weight_x
            sx -= radiance_weight * wx
            sy -= radiance_weight * wy
            sz -= radiance_weight * wz
    return sx, sy, sz


@njit(cache=True, nogil=True)
def accel_albedo_quadrature_numba(
    rx: float, ry: float, rz: float,
    sunx: float, suny: float, sunz: float,
    earthx: float, earthy: float, earthz: float,
    radius_m: float,
    cell_albedo: np.ndarray,
    n_lat: int,
    n_lon: int,
    albedo_const: float,
    pressure_coefficient: float,
    spacecraft_area_m2: float,
    spacecraft_mass_kg: float,
    solar_flux_1au_W_m2: float,
    au_m: float,
    c_light_m_s: float,
    r_earth_m: float,
    include_sun_distance_scaling: bool,
    enable_eclipse: bool,
    radial_nodes: np.ndarray,
    radial_weights: np.ndarray,
    n_azimuth: int,
) -> tuple[float, float, float]:
    """Lambertian lunar albedo acceleration by visible-disk quadrature (Moon-fixed).

    Inputs and the returned acceleration [m/s^2] are Moon-fixed; ``cell_albedo``
    is read per hit point when ``n_lat, n_lon > 0``, otherwise ``albedo_const``
    is used everywhere. Physics and eclipse treatment mirror
    :func:`lunaris.physics.lunar_albedo.accel_albedo_facets_numba`.

    Reference: adapted from Knocke et al. (1988), AIAA Paper 88-4292: Diffuse-body
    shortwave (albedo) radiation force [LUNARIS-MODEL-RAD-002].
    """
    if (
        pressure_coefficient == 0.0
        or spacecraft_area_m2 <= 0.0
        or spacecraft_mass_kg <= 0.0
        or c_light_m_s <= 0.0
    ):
        return 0.0, 0.0, 0.0
    eclipse = 1.0
    if enable_eclipse:
        eclipse = _moon_centre_eclipse_factor(sunx, suny, sunz, earthx, earthy, earthz, r_earth_m)
        if eclipse <= 0.0:
            return 0.0, 0.0, 0.0
    ix, iy, iz = _visible_disk_quadrature(
        rx, ry, rz, sunx, suny, sunz, radius_m,
        cell_albedo, n_lat, n_lon, _KIND_ALBEDO, albedo_const,
        solar_flux_1au_W_m2, au_m, include_sun_distance_scaling, eclipse,
        0.0, 0.0, 0.0, 0.0,
        radial_nodes, radial_weights, n_azimuth,
    )
    scale = pressure_coefficient * (spacecraft_area_m2 / spacecraft_mass_kg) / c_light_m_s
    return scale * ix, scale * iy, scale * iz


@njit(cache=True, nogil=True)
def accel_thermal_ir_quadrature_numba(
    rx: float, ry: float, rz: float,
    sunx: float, suny: float, sunz: float,
    earthx: float, earthy: float, earthz: float,
    radius_m: float,
    cell_temperatures_K: np.ndarray,
    n_lat: int,
    n_lon: int,
    mode: int,
    surface_emissivity: float,
    surface_albedo: float,
    temperature_K: float,
    night_temperature_K: float,
    thermal_floor_flux_W_m2: float,
    ir_pressure_coefficient: float,
    spacecraft_area_m2: float,
    spacecraft_mass_kg: float,
    solar_flux_1au_W_m2: float,
    au_m: float,
    c_light_m_s: float,
    sigma_sb: float,
    include_sun_distance_scaling: bool,
    r_earth_m: float,
    enable_eclipse: bool,
    radial_nodes: np.ndarray,
    radial_weights: np.ndarray,
    n_azimuth: int,
) -> tuple[float, float, float]:
    """Lambertian lunar thermal-IR acceleration by visible-disk quadrature (Moon-fixed).

    Exitance per mode and the eclipse treatment (equilibrium mode only) mirror
    :func:`lunaris.physics.thermal_ir.accel_thermal_ir_facets_numba`;
    ``temperature_grid`` mode reads ``cell_temperatures_K`` per hit point.

    Reference: adapted from Knocke et al. (1988), AIAA Paper 88-4292: Diffuse-body
    longwave (thermal) radiation force [LUNARIS-MODEL-RAD-003].
    """
    if (
        ir_pressure_coefficient == 0.0
        or spacecraft_area_m2 <= 0.0
        or spacecraft_mass_kg <= 0.0
        or surface_emissivity <= 0.0
        or c_light_m_s <= 0.0
        or sigma_sb <= 0.0
    ):
        return 0.0, 0.0, 0.0
    albedo = surface_albedo
    if albedo < 0.0:
        albedo = 0.0
    elif albedo > 1.0:
        albedo = 1.0
    night_exitance = 0.0
    if night_temperature_K > 0.0:
        t2 = night_temperature_K * night_temperature_K
        night_exitance = surface_emissivity * sigma_sb * t2 * t2
    floor_exitance = thermal_floor_flux_W_m2
    if floor_exitance < night_exitance:
        floor_exitance = night_exitance
    if floor_exitance < 0.0:
        floor_exitance = 0.0
    eclipse = 1.0
    if enable_eclipse and mode == THERMAL_MODE_EQUILIBRIUM:
        eclipse = _moon_centre_eclipse_factor(sunx, suny, sunz, earthx, earthy, earthz, r_earth_m)
    ix, iy, iz = _visible_disk_quadrature(
        rx, ry, rz, sunx, suny, sunz, radius_m,
        cell_temperatures_K, n_lat, n_lon, mode, temperature_K,
        solar_flux_1au_W_m2, au_m, include_sun_distance_scaling, eclipse,
        surface_emissivity, albedo, floor_exitance, sigma_sb,
        radial_nodes, radial_weights, n_azimuth,
    )
    scale = ir_pressure_coefficient * (spacecraft_area_m2 / spacecraft_mass_kg) / c_light_m_s
    return scale * ix, scale * iy, scale * iz


__all__ = (
    "DEFAULT_AZIMUTH_NODES",
    "DEFAULT_RADIAL_NODES",
    "accel_albedo_quadrature_numba",
    "accel_thermal_ir_quadrature_numba",
    "gauss_legendre_unit_interval",
)
