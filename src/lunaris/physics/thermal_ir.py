# lunaris/physics/thermal_ir.py
"""
Lambertian lunar thermal IR radiation pressure
==============================================

This module implements a non-gravitational lunar thermal infrared radiation
pressure perturbation. The Moon is discretized into spherical latitude-longitude
facets. Each visible facet emits thermal radiation as a Lambertian surface, and
the spacecraft receives an irradiance contribution

    dE_i = (epsilon_i sigma T_i^4 / pi) * (n_i dot u_i) * dA_i / d_i^2

where ``u_i`` points from the facet to the spacecraft. The acceleration is

    da_i = (C_IR A_eff / (m c)) * dE_i * u_i

The supported temperature modes are:

``constant_temperature``
    Every facet uses one configured temperature. This is a validation and
    smoke-test approximation.

``equilibrium_temperature``
    Facet exitance is computed from instantaneous solar incidence:
    ``M = (1 - A) S max(0, n dot sun_hat) + Q_floor``. This has no thermal
    inertia, regolith conduction, time lag, roughness, or terrain self-shadowing.

``temperature_grid``
    Facet temperatures are supplied as pre-sampled cell values by the caller.
    No synthetic thermal map is invented here.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import numpy.typing as npt
from numba import njit

from lunaris.common.constants import AU, C_LIGHT, PI, R_MOON, SIGMA_SB, SOLAR_FLUX_1AU


THERMAL_MODE_CONSTANT = 0
THERMAL_MODE_EQUILIBRIUM = 1
THERMAL_MODE_TEMPERATURE_GRID = 2


def normalize_thermal_mode(mode: str) -> int:
    m = str(mode).strip().lower()
    if m in {"constant", "constant_temperature"}:
        return THERMAL_MODE_CONSTANT
    if m in {"equilibrium", "equilibrium_temperature", "instantaneous_equilibrium"}:
        return THERMAL_MODE_EQUILIBRIUM
    if m in {"temperature_grid", "grid"}:
        return THERMAL_MODE_TEMPERATURE_GRID
    raise ValueError(
        "thermal_mode must be one of 'constant_temperature', "
        "'equilibrium_temperature', or 'temperature_grid'."
    )


def build_latlon_facets(
    lat_count: int,
    lon_count: int,
    radius_m: float = R_MOON,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build spherical latitude-longitude facet centers, normals, and exact areas.

    Returns
    -------
    positions, normals, areas, lat_centers_rad, lon_centers_rad
        ``positions`` and ``normals`` are ``(N, 3)`` float64 arrays, ``areas`` is
        ``(N,)``. Cell area uses ``R^2 * dlon * (sin(lat2) - sin(lat1))``.
    """
    n_lat = int(lat_count)
    n_lon = int(lon_count)
    r = float(radius_m)
    if n_lat < 1 or n_lon < 1:
        raise ValueError("thermal facet lat/lon counts must be >= 1.")
    if r <= 0.0 or not math.isfinite(r):
        raise ValueError(f"thermal facet radius must be finite and > 0, got {radius_m!r}.")

    lat_edges = np.linspace(-0.5 * math.pi, 0.5 * math.pi, n_lat + 1, dtype=np.float64)
    lon_edges = np.linspace(0.0, 2.0 * math.pi, n_lon + 1, dtype=np.float64)
    dlon = 2.0 * math.pi / float(n_lon)

    n = n_lat * n_lon
    positions = np.empty((n, 3), dtype=np.float64)
    normals = np.empty((n, 3), dtype=np.float64)
    areas = np.empty(n, dtype=np.float64)
    lat_centers = np.empty(n, dtype=np.float64)
    lon_centers = np.empty(n, dtype=np.float64)

    idx = 0
    for ilat in range(n_lat):
        lat1 = float(lat_edges[ilat])
        lat2 = float(lat_edges[ilat + 1])
        # Midpoint in sine(latitude) places the center closer to the area centroid
        # of each spherical band than a plain angular midpoint.
        sin_mid = 0.5 * (math.sin(lat1) + math.sin(lat2))
        sin_mid = max(-1.0, min(1.0, sin_mid))
        lat = math.asin(sin_mid)
        cos_lat = math.cos(lat)
        area_band_cell = r * r * dlon * (math.sin(lat2) - math.sin(lat1))

        for ilon in range(n_lon):
            lon = 0.5 * (float(lon_edges[ilon]) + float(lon_edges[ilon + 1]))
            cos_lon = math.cos(lon)
            sin_lon = math.sin(lon)

            nx = cos_lat * cos_lon
            ny = cos_lat * sin_lon
            nz = sin_mid

            normals[idx, 0] = nx
            normals[idx, 1] = ny
            normals[idx, 2] = nz
            positions[idx, 0] = r * nx
            positions[idx, 1] = r * ny
            positions[idx, 2] = r * nz
            areas[idx] = area_band_cell
            lat_centers[idx] = lat
            lon_centers[idx] = lon
            idx += 1

    return (
        np.ascontiguousarray(positions, dtype=np.float64),
        np.ascontiguousarray(normals, dtype=np.float64),
        np.ascontiguousarray(areas, dtype=np.float64),
        np.ascontiguousarray(lat_centers, dtype=np.float64),
        np.ascontiguousarray(lon_centers, dtype=np.float64),
    )


@njit(cache=True, nogil=True)
def thermal_ir_single_facet_accel_numba(
    rx: float,
    ry: float,
    rz: float,
    fx: float,
    fy: float,
    fz: float,
    nx: float,
    ny: float,
    nz: float,
    area_facet_m2: float,
    exitance_w_m2: float,
    ir_pressure_coefficient: float,
    spacecraft_area_m2: float,
    spacecraft_mass_kg: float,
    c_light_m_s: float,
) -> Tuple[float, float, float]:
    """Acceleration contribution from one Lambertian thermal facet."""
    if (
        area_facet_m2 <= 0.0
        or exitance_w_m2 <= 0.0
        or ir_pressure_coefficient == 0.0
        or spacecraft_area_m2 <= 0.0
        or spacecraft_mass_kg <= 0.0
        or c_light_m_s <= 0.0
    ):
        return 0.0, 0.0, 0.0

    sx = rx - fx
    sy = ry - fy
    sz = rz - fz
    d2 = sx * sx + sy * sy + sz * sz
    if d2 <= 1.0:
        return 0.0, 0.0, 0.0

    inv_d = 1.0 / math.sqrt(d2)
    ux = sx * inv_d
    uy = sy * inv_d
    uz = sz * inv_d

    mu_view = nx * ux + ny * uy + nz * uz
    if mu_view <= 0.0:
        return 0.0, 0.0, 0.0

    irradiance = (exitance_w_m2 / PI) * mu_view * area_facet_m2 / d2
    scale = ir_pressure_coefficient * (spacecraft_area_m2 / spacecraft_mass_kg) * irradiance / c_light_m_s
    return scale * ux, scale * uy, scale * uz


@njit(cache=True, nogil=True)
def accel_thermal_ir_facets_numba(
    rx: float,
    ry: float,
    rz: float,
    sunx: float,
    suny: float,
    sunz: float,
    facet_pos_m: np.ndarray,
    facet_normals: np.ndarray,
    facet_areas_m2: np.ndarray,
    facet_temperatures_K: np.ndarray,
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
) -> Tuple[float, float, float]:
    """Sum Lambertian lunar thermal IR acceleration over precomputed facets."""
    if (
        ir_pressure_coefficient == 0.0
        or spacecraft_area_m2 <= 0.0
        or spacecraft_mass_kg <= 0.0
        or surface_emissivity <= 0.0
        or c_light_m_s <= 0.0
        or sigma_sb <= 0.0
    ):
        return 0.0, 0.0, 0.0

    n = facet_areas_m2.shape[0]
    if n <= 0:
        return 0.0, 0.0, 0.0

    sun_norm2 = sunx * sunx + suny * suny + sunz * sunz
    sun_hat_x = 0.0
    sun_hat_y = 0.0
    sun_hat_z = 0.0
    solar_flux = solar_flux_1au_W_m2
    if sun_norm2 > 1.0:
        inv_sun = 1.0 / math.sqrt(sun_norm2)
        sun_hat_x = sunx * inv_sun
        sun_hat_y = suny * inv_sun
        sun_hat_z = sunz * inv_sun
        if include_sun_distance_scaling:
            solar_flux = solar_flux_1au_W_m2 * (au_m * au_m / sun_norm2)

    if solar_flux < 0.0:
        solar_flux = 0.0

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

    const_exitance = 0.0
    if mode == THERMAL_MODE_CONSTANT and temperature_K > 0.0:
        t2 = temperature_K * temperature_K
        const_exitance = surface_emissivity * sigma_sb * t2 * t2

    ax = 0.0
    ay = 0.0
    az = 0.0

    for i in range(n):
        exitance = 0.0

        if mode == THERMAL_MODE_CONSTANT:
            exitance = const_exitance
        elif mode == THERMAL_MODE_EQUILIBRIUM:
            nx = facet_normals[i, 0]
            ny = facet_normals[i, 1]
            nz = facet_normals[i, 2]
            mu_sun = nx * sun_hat_x + ny * sun_hat_y + nz * sun_hat_z
            absorbed = 0.0
            if mu_sun > 0.0:
                absorbed = (1.0 - albedo) * solar_flux * mu_sun
            exitance = absorbed + floor_exitance
        elif mode == THERMAL_MODE_TEMPERATURE_GRID:
            if facet_temperatures_K.shape[0] <= i:
                return 0.0, 0.0, 0.0
            temp = facet_temperatures_K[i]
            if temp > 0.0:
                t2 = temp * temp
                exitance = surface_emissivity * sigma_sb * t2 * t2

        if exitance <= 0.0:
            continue

        fx = facet_pos_m[i, 0]
        fy = facet_pos_m[i, 1]
        fz = facet_pos_m[i, 2]
        nx = facet_normals[i, 0]
        ny = facet_normals[i, 1]
        nz = facet_normals[i, 2]

        tx, ty, tz = thermal_ir_single_facet_accel_numba(
            rx,
            ry,
            rz,
            fx,
            fy,
            fz,
            nx,
            ny,
            nz,
            facet_areas_m2[i],
            exitance,
            ir_pressure_coefficient,
            spacecraft_area_m2,
            spacecraft_mass_kg,
            c_light_m_s,
        )
        ax += tx
        ay += ty
        az += tz

    return ax, ay, az


def calc_thermal_ir_accel(
    r_sc_fixed: npt.ArrayLike,
    r_sun_fixed: npt.ArrayLike,
    facet_pos_m: np.ndarray,
    facet_normals: np.ndarray,
    facet_areas_m2: np.ndarray,
    *,
    facet_temperatures_K: npt.ArrayLike | None = None,
    mode: str = "constant_temperature",
    surface_emissivity: float = 0.95,
    surface_albedo: float = 0.12,
    temperature_K: float = 250.0,
    night_temperature_K: float = 100.0,
    thermal_floor_flux_W_m2: float = 0.0,
    ir_pressure_coefficient: float = 1.0,
    spacecraft_area_m2: float = 1.0,
    spacecraft_mass_kg: float = 1000.0,
    solar_flux_1au_W_m2: float = SOLAR_FLUX_1AU,
    au_m: float = AU,
    c_light_m_s: float = C_LIGHT,
    sigma_sb: float = SIGMA_SB,
    include_sun_distance_scaling: bool = True,
) -> np.ndarray:
    """Python convenience wrapper for the facet-summed thermal IR acceleration."""
    r = np.asarray(r_sc_fixed, dtype=np.float64).reshape(3)
    s = np.asarray(r_sun_fixed, dtype=np.float64).reshape(3)
    temps = (
        np.zeros(1, dtype=np.float64)
        if facet_temperatures_K is None
        else np.ascontiguousarray(np.asarray(facet_temperatures_K, dtype=np.float64).reshape(-1))
    )
    ax, ay, az = accel_thermal_ir_facets_numba(
        float(r[0]),
        float(r[1]),
        float(r[2]),
        float(s[0]),
        float(s[1]),
        float(s[2]),
        np.ascontiguousarray(facet_pos_m, dtype=np.float64),
        np.ascontiguousarray(facet_normals, dtype=np.float64),
        np.ascontiguousarray(facet_areas_m2, dtype=np.float64),
        temps,
        normalize_thermal_mode(mode),
        float(surface_emissivity),
        float(surface_albedo),
        float(temperature_K),
        float(night_temperature_K),
        float(thermal_floor_flux_W_m2),
        float(ir_pressure_coefficient),
        float(spacecraft_area_m2),
        float(spacecraft_mass_kg),
        float(solar_flux_1au_W_m2),
        float(au_m),
        float(c_light_m_s),
        float(sigma_sb),
        bool(include_sun_distance_scaling),
    )
    return np.asarray((ax, ay, az), dtype=np.float64)


__all__ = (
    "THERMAL_MODE_CONSTANT",
    "THERMAL_MODE_EQUILIBRIUM",
    "THERMAL_MODE_TEMPERATURE_GRID",
    "normalize_thermal_mode",
    "build_latlon_facets",
    "thermal_ir_single_facet_accel_numba",
    "accel_thermal_ir_facets_numba",
    "calc_thermal_ir_accel",
)
