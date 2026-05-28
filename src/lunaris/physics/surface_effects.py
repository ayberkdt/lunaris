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

from lunaris.common.constants import AU, P_SUN_1AU, R_MOON_MEAN
from lunaris.common.type_defs import SpacecraftProps, Vec3
from lunaris.physics.solar_effects import moon_shadow_factor_conical


# =============================================================================
# 1) Config bundles
# =============================================================================

def _require_range(name: str, x: float, lo: float, hi: float) -> float:
    """Validate x is finite and in [lo, hi]. Returns normalized float."""
    v = float(x)
    if not (lo <= v <= hi):
        raise ValueError(f"{name} must be in [{lo}, {hi}], got {v}.")
    return v


def _require_ge0(name: str, x: float) -> float:
    """Validate x is finite and >= 0. Returns normalized float."""
    v = float(x)
    if v < 0.0:
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
    """Settings for thermal re-radiation (thermal recoil) models."""

    k_thermal: float = 1.0        # tuning factor (>= 0)
    P0: float = P_SUN_1AU         # SRP at 1 AU [N/m^2]
    AU_m: float = AU              # astronomical unit [m]

    def __post_init__(self) -> None:
        object.__setattr__(self, "k_thermal", _require_ge0("ThermalConfig.k_thermal", self.k_thermal))
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
    """Thermal (IR re-radiation) acceleration in a Moon-centered frame."""
    r_sc_v = _as_vec3(r_sc, "r_sc")
    r_sun_v = _as_vec3(r_sun, "r_sun")

    eclipse_flag = 1 if enable_eclipse else 0

    ax, ay, az = accel_thermal_simple(
        float(r_sc_v[0]), float(r_sc_v[1]), float(r_sc_v[2]),
        float(r_sun_v[0]), float(r_sun_v[1]), float(r_sun_v[2]),
        float(R_moon),
        float(config.AU_m),
        float(config.P0),
        float(config.k_thermal),
        float(sc_props.cr),
        float(sc_props.area_m2),
        float(sc_props.mass_kg),
        eclipse_flag,
    )
    return np.array((ax, ay, az), dtype=np.float64)

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
