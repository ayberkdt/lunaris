# ST_LRPS/models/surface_effects.py
"""Lunar surface radiative effects (physics layer).

This module owns the **dynamics / force models** that depend on the lunar
surface in an engineering sense:

* Albedo radiation pressure (reflected sunlight)
* Thermal re-radiation recoil (IR)

Everything related to *datasets* (PDS3 label parsing, file discovery, memmap
loading, grid sampling, provider facades) lives in :mod:`models.surface_data`.

Units & frames
--------------
* SI units throughout.
* All vectors are Moon-centered: ``r_sc`` and ``r_sun`` are positions w.r.t.
  the Moon center.
"""

# =============================================================================
# 0.                                IMPORTS
# =============================================================================


from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import numpy.typing as npt
from numba import njit

from lunaris.common.constants import AU, C_LIGHT, P_SUN_1AU, R_MOON_MEAN, SIGMA_SB, SOLAR_FLUX_1AU
from lunaris.common.type_defs import SpacecraftProps, Vec3
from lunaris.physics.solar_effects import moon_shadow_factor_conical
from lunaris.physics.thermal_ir import build_latlon_facets, calc_thermal_ir_accel


# =============================================================================
# 1) Config bundles
# =============================================================================

def _require_range(name: str, x: float, lo: float, hi: float) -> float:
    """Validate x is finite and in [lo, hi]. Returns normalized float."""
    v = float(x)
    if not math.isfinite(v) or not (lo <= v <= hi):
        raise ValueError(f"{name} must be in [{lo}, {hi}], got {v}.")
    return v


def _require_ge0(name: str, x: float) -> float:
    """Validate x is finite and >= 0. Returns normalized float."""
    v = float(x)
    if not math.isfinite(v) or v < 0.0:
        raise ValueError(f"{name} must be >= 0, got {v}.")
    return v


@dataclass(frozen=True, slots=True)
class AlbedoConfig:
    """Settings for lunar albedo radiation-pressure models."""

    A_moon: float = 0.12          # effective lunar albedo in [0,1]
    k_lambert: float = 1.0        # Lambertian scaling (>= 0)
    P0: float = P_SUN_1AU         # SRP at 1 AU [N/m^2]
    AU_m: float = AU              # astronomical unit [m]

    def __post_init__(self) -> None:
        object.__setattr__(self, "A_moon", _require_range("AlbedoConfig.A_moon", self.A_moon, 0.0, 1.0))
        object.__setattr__(self, "k_lambert", _require_ge0("AlbedoConfig.k_lambert", self.k_lambert))
        object.__setattr__(self, "P0", _require_ge0("AlbedoConfig.P0", self.P0))
        object.__setattr__(self, "AU_m", _require_ge0("AlbedoConfig.AU_m", self.AU_m))


@dataclass(frozen=True, slots=True)
class ThermalConfig:
    """
    Settings for Lambertian lunar thermal IR radiation pressure.

    Defaults are configurable engineering values, not a full lunar thermal
    environment. ``k_thermal`` is retained as a compatibility alias for the
    pressure coefficient used by older callers.
    """

    thermal_mode: str = "constant_temperature"
    surface_emissivity: float = 0.95
    surface_albedo: float = 0.12
    temperature_K: float = 250.0
    night_temperature_K: float = 100.0
    thermal_floor_flux_W_m2: float = 0.0
    ir_pressure_coefficient: float = 1.0
    facet_lat_count: int = 18
    facet_lon_count: int = 36
    max_facets: int = 10_000
    use_existing_surface_grid: bool = False
    require_surface_provider: bool = False
    solar_flux_1au_W_m2: float = SOLAR_FLUX_1AU
    c_light_m_s: float = C_LIGHT
    sigma_sb: float = SIGMA_SB
    include_sun_distance_scaling: bool = True

    # Compatibility with the former simple thermal wrapper.
    k_thermal: float | None = None

    # Legacy constants retained for old wrappers.
    P0: float = P_SUN_1AU
    AU_m: float = AU

    def __post_init__(self) -> None:
        mode = str(self.thermal_mode).strip().lower()
        if mode not in {"constant_temperature", "constant", "equilibrium_temperature", "equilibrium", "temperature_grid", "grid"}:
            raise ValueError(
                "ThermalConfig.thermal_mode must be 'constant_temperature', "
                "'equilibrium_temperature', or 'temperature_grid'."
            )
        if mode == "constant":
            mode = "constant_temperature"
        elif mode == "equilibrium":
            mode = "equilibrium_temperature"
        elif mode == "grid":
            mode = "temperature_grid"

        ir_coeff = float(self.ir_pressure_coefficient)
        if self.k_thermal is not None:
            k_alias = float(self.k_thermal)
            # Older callers may pass only k_thermal. Newer dataclass.replace()
            # calls often update ir_pressure_coefficient on an existing config
            # whose compatibility alias is already populated.
            if ir_coeff == 1.0 or ir_coeff == k_alias:
                ir_coeff = k_alias

        object.__setattr__(self, "thermal_mode", mode)
        object.__setattr__(self, "surface_emissivity", _require_range("ThermalConfig.surface_emissivity", self.surface_emissivity, 0.0, 1.0))
        object.__setattr__(self, "surface_albedo", _require_range("ThermalConfig.surface_albedo", self.surface_albedo, 0.0, 1.0))
        object.__setattr__(self, "temperature_K", _require_ge0("ThermalConfig.temperature_K", self.temperature_K))
        object.__setattr__(self, "night_temperature_K", _require_ge0("ThermalConfig.night_temperature_K", self.night_temperature_K))
        object.__setattr__(self, "thermal_floor_flux_W_m2", _require_ge0("ThermalConfig.thermal_floor_flux_W_m2", self.thermal_floor_flux_W_m2))
        object.__setattr__(self, "ir_pressure_coefficient", _require_ge0("ThermalConfig.ir_pressure_coefficient", ir_coeff))
        object.__setattr__(self, "k_thermal", _require_ge0("ThermalConfig.k_thermal", ir_coeff))
        object.__setattr__(self, "facet_lat_count", int(self.facet_lat_count))
        object.__setattr__(self, "facet_lon_count", int(self.facet_lon_count))
        object.__setattr__(self, "max_facets", int(self.max_facets))
        if self.facet_lat_count < 1 or self.facet_lon_count < 1:
            raise ValueError("ThermalConfig facet_lat_count/facet_lon_count must be >= 1.")
        if self.max_facets < 1:
            raise ValueError("ThermalConfig.max_facets must be >= 1.")
        if self.facet_lat_count * self.facet_lon_count > self.max_facets:
            raise ValueError(
                "Thermal facet grid exceeds max_facets: "
                f"{self.facet_lat_count * self.facet_lon_count} > {self.max_facets}."
            )
        object.__setattr__(self, "solar_flux_1au_W_m2", _require_ge0("ThermalConfig.solar_flux_1au_W_m2", self.solar_flux_1au_W_m2))
        object.__setattr__(self, "c_light_m_s", _require_ge0("ThermalConfig.c_light_m_s", self.c_light_m_s))
        object.__setattr__(self, "sigma_sb", _require_ge0("ThermalConfig.sigma_sb", self.sigma_sb))
        object.__setattr__(self, "P0", _require_ge0("ThermalConfig.P0", self.P0))
        object.__setattr__(self, "AU_m", _require_ge0("ThermalConfig.AU_m", self.AU_m))


# =============================================================================
# 2) Physics kernels (Numba)
# =============================================================================

@njit(cache=True, inline="always")
def _valid_area_mass(area_m2: float, mass_kg: float) -> bool:
    return (mass_kg > 0.0) and (area_m2 > 0.0)


@njit(cache=True, inline="always")
def _clamp_pm1(x: float) -> float:
    if x < -1.0:
        return -1.0
    if x > 1.0:
        return 1.0
    return x


@njit(cache=True)
def accel_albedo_simple(
    rx: float, ry: float, rz: float,
    sx: float, sy: float, sz: float,
    R_moon: float,
    AU_m: float, P0: float,
    A_moon: float,
    k_lambert: float,
    Cr: float, area_m2: float, mass_kg: float,
    enable_eclipse: int,
) -> Tuple[float, float, float]:
    """Simple Lambertian albedo (engineering model)."""
    if not _valid_area_mass(area_m2, mass_kg):
        return 0.0, 0.0, 0.0

    # Sun -> Spacecraft vector
    dx = rx - sx
    dy = ry - sy
    dz = rz - sz

    d2 = dx * dx + dy * dy + dz * dz
    if d2 <= 1e-6:  # ~1 mm^2 guard
        return 0.0, 0.0, 0.0

    d = math.sqrt(d2)

    shadow = 1.0
    if enable_eclipse != 0:
        shadow = moon_shadow_factor_conical(rx, ry, rz, sx, sy, sz, R_moon)
        if shadow <= 1e-9:
            return 0.0, 0.0, 0.0

    # Flux scales as 1/d^2
    flux_ratio = (AU_m * AU_m) / d2

    scale = P0 * flux_ratio * shadow
    scale *= (A_moon * k_lambert) * Cr * (area_m2 / mass_kg)

    inv_d = 1.0 / d
    return scale * dx * inv_d, scale * dy * inv_d, scale * dz * inv_d


@njit(cache=True)
def accel_albedo_lommel_seeliger(
    rx: float, ry: float, rz: float,
    sx: float, sy: float, sz: float,
    R_moon: float,
    AU_m: float, P0: float,
    A_moon: float,
    k_lambert: float,
    Cr: float, area_m2: float, mass_kg: float,
    enable_eclipse: int,
) -> Tuple[float, float, float]:
    """Lommel–Seeliger-inspired albedo scaling (engineering)."""
    if not _valid_area_mass(area_m2, mass_kg):
        return 0.0, 0.0, 0.0

    # Sun -> Spacecraft vector
    dx = rx - sx
    dy = ry - sy
    dz = rz - sz

    d2 = dx * dx + dy * dy + dz * dz
    if d2 <= 1e-6:
        return 0.0, 0.0, 0.0

    d = math.sqrt(d2)

    shadow = 1.0
    if enable_eclipse != 0:
        shadow = moon_shadow_factor_conical(rx, ry, rz, sx, sy, sz, R_moon)
        if shadow <= 1e-9:
            return 0.0, 0.0, 0.0

    # Phase weight w in [0,1] (simple proxy)
    r_sc2 = rx * rx + ry * ry + rz * rz
    r_sun2 = sx * sx + sy * sy + sz * sz

    w = 1.0
    if r_sc2 > 0.0 and r_sun2 > 0.0:
        inv_mag = 1.0 / (math.sqrt(r_sc2) * math.sqrt(r_sun2))
        cos_phase = (rx * sx + ry * sy + rz * sz) * inv_mag
        cos_phase = _clamp_pm1(cos_phase)
        w = 0.5 + 0.5 * cos_phase
        if w < 0.0:
            w = 0.0

    flux_ratio = (AU_m * AU_m) / d2

    scale = P0 * flux_ratio * shadow
    scale *= (A_moon * k_lambert * w) * Cr * (area_m2 / mass_kg)

    inv_d = 1.0 / d
    return scale * dx * inv_d, scale * dy * inv_d, scale * dz * inv_d


@njit(cache=True)
def accel_thermal_simple(
    rx: float, ry: float, rz: float,
    sx: float, sy: float, sz: float,
    R_moon: float,
    AU_m: float, P0: float,
    k_thermal: float,
    Cr: float, area_m2: float, mass_kg: float,
    enable_eclipse: int,
) -> Tuple[float, float, float]:
    """Simple lunar thermal IR recoil (engineering)."""
    if not _valid_area_mass(area_m2, mass_kg):
        return 0.0, 0.0, 0.0

    # Moon -> Spacecraft (radial) direction
    r2 = rx * rx + ry * ry + rz * rz
    if r2 <= 1.0:  # avoid singular near origin
        return 0.0, 0.0, 0.0

    r = math.sqrt(r2)
    inv_r = 1.0 / r
    urx = rx * inv_r
    ury = ry * inv_r
    urz = rz * inv_r

    # Sun -> Spacecraft distance (flux proxy)
    dx = rx - sx
    dy = ry - sy
    dz = rz - sz

    d2 = dx * dx + dy * dy + dz * dz
    if d2 <= 1.0:
        return 0.0, 0.0, 0.0

    shadow = 1.0
    if enable_eclipse != 0:
        shadow = moon_shadow_factor_conical(rx, ry, rz, sx, sy, sz, R_moon)
        if shadow <= 1e-9:
            return 0.0, 0.0, 0.0

    flux_ratio = (AU_m * AU_m) / d2

    scale = P0 * flux_ratio * shadow
    scale *= k_thermal * Cr * (area_m2 / mass_kg)

    return scale * urx, scale * ury, scale * urz


# =============================================================================
# 3) Public wrappers (NumPy interface)
# =============================================================================

def _as_vec3(x: npt.ArrayLike, name: str) -> Vec3:
    v = np.asarray(x, dtype=np.float64)
    if v.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {v.shape}.")
    return v


_ALBEDO_KERNELS: Dict[str, object] = {
    "simple": accel_albedo_simple,
    "lommel": accel_albedo_lommel_seeliger,
}


def albedo_accel(
    r_sc: npt.ArrayLike,
    r_sun: npt.ArrayLike,
    sc_props: SpacecraftProps,
    config: AlbedoConfig,
    *,
    model: str = "simple",
    enable_eclipse: bool = True,
    R_moon: float = R_MOON_MEAN,
) -> Vec3:
    """Albedo acceleration in a Moon-centered frame."""
    r_sc_v = _as_vec3(r_sc, "r_sc")
    r_sun_v = _as_vec3(r_sun, "r_sun")

    kernel = _ALBEDO_KERNELS.get(model)
    if kernel is None:
        raise ValueError(f"Unsupported albedo model: {model!r}. Supported: {sorted(_ALBEDO_KERNELS)}")

    eclipse_flag = 1 if enable_eclipse else 0

    ax, ay, az = kernel(  # type: ignore[misc]
        float(r_sc_v[0]), float(r_sc_v[1]), float(r_sc_v[2]),
        float(r_sun_v[0]), float(r_sun_v[1]), float(r_sun_v[2]),
        float(R_moon),
        float(config.AU_m),
        float(config.P0),
        float(config.A_moon),
        float(config.k_lambert),
        float(sc_props.cr),
        float(sc_props.area_m2),
        float(sc_props.mass_kg),
        eclipse_flag,
    )
    return np.array((ax, ay, az), dtype=np.float64)


def thermal_accel(
    r_sc: npt.ArrayLike,
    r_sun: npt.ArrayLike,
    sc_props: SpacecraftProps,
    config: ThermalConfig,
    *,
    enable_eclipse: bool = True,
    R_moon: float = R_MOON_MEAN,
) -> Vec3:
    """Thermal IR acceleration in a Moon-centered frame.

    This wrapper now delegates to the Lambertian facet model in
    :mod:`lunaris.physics.thermal_ir`. ``enable_eclipse`` is accepted for
    compatibility with older callers; thermal emission is controlled by the
    selected thermal mode rather than spacecraft eclipse state.
    """
    r_sc_v = _as_vec3(r_sc, "r_sc")
    r_sun_v = _as_vec3(r_sun, "r_sun")

    facet_pos, facet_normals, facet_areas, _, _ = build_latlon_facets(
        int(config.facet_lat_count),
        int(config.facet_lon_count),
        radius_m=float(R_moon),
    )
    return calc_thermal_ir_accel(
        r_sc_v,
        r_sun_v,
        facet_pos,
        facet_normals,
        facet_areas,
        mode=str(config.thermal_mode),
        surface_emissivity=float(config.surface_emissivity),
        surface_albedo=float(config.surface_albedo),
        temperature_K=float(config.temperature_K),
        night_temperature_K=float(config.night_temperature_K),
        thermal_floor_flux_W_m2=float(config.thermal_floor_flux_W_m2),
        ir_pressure_coefficient=float(config.ir_pressure_coefficient),
        spacecraft_area_m2=float(sc_props.area_m2),
        spacecraft_mass_kg=float(sc_props.mass_kg),
        solar_flux_1au_W_m2=float(config.solar_flux_1au_W_m2),
        au_m=float(config.AU_m),
        c_light_m_s=float(config.c_light_m_s),
        sigma_sb=float(config.sigma_sb),
        include_sun_distance_scaling=bool(config.include_sun_distance_scaling),
    )

__all__ = (
    # Config bundles
    "AlbedoConfig",
    "ThermalConfig",
    # Numba kernels
    "accel_albedo_simple",
    "accel_albedo_lommel_seeliger",
    "accel_thermal_simple",
    # High-level wrappers
    "albedo_accel",
    "thermal_accel",
)
