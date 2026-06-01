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
from typing import Tuple

import numpy as np
import numpy.typing as npt
from numba import njit

from lunaris.common.constants import AU, C_LIGHT, P_SUN_1AU, R_EARTH_MEAN, R_MOON_MEAN, SIGMA_SB, SOLAR_FLUX_1AU
from lunaris.common.type_defs import SpacecraftProps, Vec3
from lunaris.physics.solar_effects import moon_shadow_factor_conical
from lunaris.physics.thermal_ir import build_latlon_facets, calc_thermal_ir_accel
from lunaris.physics.lunar_albedo import calc_albedo_accel, normalize_albedo_mode


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
    """Settings for lunar albedo radiation-pressure models.

    Two backends are available, selected by ``albedo_model``:

    ``lambert_facets`` (default)
        Physically-resolved facet model in :mod:`lunaris.physics.lunar_albedo`.
        The Moon is discretized into sunlit/visible Lambertian facets whose
        reflected-solar contributions are summed (in the Moon-fixed frame) and
        rotated back to inertial by the dynamics layer. It uses the dedicated
        coefficient ``albedo_pressure_coefficient`` (C_R_albedo), **not** the
        spacecraft SRP coefficient ``cr``.

    ``simple``
        Legacy engineering "cannonball" kernel kept for backward compatibility.
        It pushes along the Sun->spacecraft line and reuses the spacecraft SRP
        coefficient ``cr`` scaled by ``k_lambert``. Its behavior is unchanged.

    The per-facet albedo ``A_i`` used by ``lambert_facets`` is built once at
    setup time from ``albedo_mode``:

    ``constant_albedo``
        ``albedo_const`` everywhere (engineering approximation; ~0.07-0.16 for
        the Moon). Requires no surface provider.
    ``albedo_grid``
        Provider-supplied albedo grid already in [0,1].
    ``scaled_dn_grid``
        Provider digital-number grid mapped via ``A = scale*DN + offset`` with
        explicit nodata handling; nodata falls back to ``albedo_const``.

    Grid modes require a surface provider; ``require_surface_provider`` forces a
    hard error if one is missing.
    """

    # --- Legacy / simple-backend fields (kept for backward compatibility) ---
    A_moon: float = 0.12          # constant lunar albedo in [0,1] (legacy alias of albedo_const)
    k_lambert: float = 1.0        # Lambertian scaling for the simple backend (>= 0)
    P0: float = P_SUN_1AU         # SRP at 1 AU [N/m^2] (simple backend only)
    AU_m: float = AU              # astronomical unit [m]

    # --- Backend & albedo-source selection ---
    albedo_model: str = "lambert_facets"   # "lambert_facets" | "simple"
    albedo_mode: str = "constant_albedo"   # "constant_albedo" | "albedo_grid" | "scaled_dn_grid"
    albedo_const: float = 0.12             # constant facet albedo for lambert_facets

    # --- Facet (lambert_facets) model parameters ---
    albedo_pressure_coefficient: float = 1.0   # C_R_albedo (distinct from SRP cr)
    facet_lat_count: int = 18
    facet_lon_count: int = 36
    max_facets: int = 10_000
    solar_flux_1au_W_m2: float = SOLAR_FLUX_1AU
    c_light_m_s: float = C_LIGHT
    include_sun_distance_scaling: bool = True
    enable_eclipse: bool = True
    require_surface_provider: bool = False
    use_existing_surface_grid: bool = False

    def __post_init__(self) -> None:
        model = str(self.albedo_model).strip().lower()
        if model in {"facet", "facets", "lambert", "lambert_facets"}:
            model = "lambert_facets"
        elif model == "simple":
            model = "simple"
        else:
            raise ValueError(
                "AlbedoConfig.albedo_model must be 'lambert_facets' or 'simple'. "
                f"Got {self.albedo_model!r}."
            )

        # Validate and canonicalize the albedo-source mode string.
        mode_code = normalize_albedo_mode(self.albedo_mode)
        mode = {0: "constant_albedo", 1: "albedo_grid", 2: "scaled_dn_grid"}[mode_code]

        # ``albedo_const`` is the canonical constant-albedo field; ``A_moon`` is a
        # legacy alias. Mirror whichever the caller set away from the shared
        # default so both backends honor a single intent.
        albedo_const = float(self.albedo_const)
        a_moon = float(self.A_moon)
        if albedo_const == 0.12 and a_moon != 0.12:
            albedo_const = a_moon
        elif a_moon == 0.12 and albedo_const != 0.12:
            a_moon = albedo_const

        object.__setattr__(self, "albedo_model", model)
        object.__setattr__(self, "albedo_mode", mode)
        object.__setattr__(self, "A_moon", _require_range("AlbedoConfig.A_moon", a_moon, 0.0, 1.0))
        object.__setattr__(self, "albedo_const", _require_range("AlbedoConfig.albedo_const", albedo_const, 0.0, 1.0))
        object.__setattr__(self, "k_lambert", _require_ge0("AlbedoConfig.k_lambert", self.k_lambert))
        object.__setattr__(self, "P0", _require_ge0("AlbedoConfig.P0", self.P0))
        object.__setattr__(self, "AU_m", _require_ge0("AlbedoConfig.AU_m", self.AU_m))
        object.__setattr__(
            self,
            "albedo_pressure_coefficient",
            _require_ge0("AlbedoConfig.albedo_pressure_coefficient", self.albedo_pressure_coefficient),
        )
        object.__setattr__(self, "facet_lat_count", int(self.facet_lat_count))
        object.__setattr__(self, "facet_lon_count", int(self.facet_lon_count))
        object.__setattr__(self, "max_facets", int(self.max_facets))
        if self.facet_lat_count < 1 or self.facet_lon_count < 1:
            raise ValueError("AlbedoConfig facet_lat_count/facet_lon_count must be >= 1.")
        if self.max_facets < 1:
            raise ValueError("AlbedoConfig.max_facets must be >= 1.")
        if self.facet_lat_count * self.facet_lon_count > self.max_facets:
            raise ValueError(
                "Albedo facet grid exceeds max_facets: "
                f"{self.facet_lat_count * self.facet_lon_count} > {self.max_facets}."
            )
        object.__setattr__(
            self,
            "solar_flux_1au_W_m2",
            _require_ge0("AlbedoConfig.solar_flux_1au_W_m2", self.solar_flux_1au_W_m2),
        )
        object.__setattr__(self, "c_light_m_s", _require_ge0("AlbedoConfig.c_light_m_s", self.c_light_m_s))


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


# =============================================================================
# 3) Public wrappers (NumPy interface)
# =============================================================================

def _as_vec3(x: npt.ArrayLike, name: str) -> Vec3:
    v = np.asarray(x, dtype=np.float64)
    if v.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {v.shape}.")
    return v


def albedo_accel(
    r_sc: npt.ArrayLike,
    r_sun: npt.ArrayLike,
    sc_props: SpacecraftProps,
    config: AlbedoConfig,
    *,
    model: str | None = None,
    enable_eclipse: bool = True,
    R_moon: float = R_MOON_MEAN,
) -> Vec3:
    """Albedo acceleration in a Moon-centered frame.

    ``model`` selects the backend; when ``None`` it defaults to
    ``config.albedo_model`` (``"lambert_facets"`` by default). The legacy
    ``"simple"`` cannonball kernel is unchanged and reuses the spacecraft SRP
    coefficient ``cr``; ``"lambert_facets"`` builds a constant-albedo facet
    sphere from the config and uses ``config.albedo_pressure_coefficient``.
    """
    r_sc_v = _as_vec3(r_sc, "r_sc")
    r_sun_v = _as_vec3(r_sun, "r_sun")

    selected = str(model if model is not None else config.albedo_model).strip().lower()

    if selected in {"lambert_facets", "facet", "facets", "lambert"}:
        facet_pos, facet_normals, facet_areas, _, _ = build_latlon_facets(
            int(config.facet_lat_count),
            int(config.facet_lon_count),
            radius_m=float(R_moon),
        )
        facet_albedo = np.full(facet_areas.shape[0], float(config.albedo_const), dtype=np.float64)
        return calc_albedo_accel(
            r_sc_v,
            r_sun_v,
            facet_pos,
            facet_normals,
            facet_areas,
            facet_albedo,
            pressure_coefficient=float(config.albedo_pressure_coefficient),
            spacecraft_area_m2=float(sc_props.area_m2),
            spacecraft_mass_kg=float(sc_props.mass_kg),
            solar_flux_1au_W_m2=float(config.solar_flux_1au_W_m2),
            au_m=float(config.AU_m),
            c_light_m_s=float(config.c_light_m_s),
            r_earth_m=float(R_EARTH_MEAN),
            include_sun_distance_scaling=bool(config.include_sun_distance_scaling),
            enable_eclipse=False,
        )

    if selected != "simple":
        raise ValueError(
            f"Unsupported albedo model: {model!r}. Supported: 'lambert_facets', 'simple'."
        )

    eclipse_flag = 1 if enable_eclipse else 0

    ax, ay, az = accel_albedo_simple(
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
    # Numba kernels (legacy cannonball albedo backend)
    "accel_albedo_simple",
    # High-level wrappers
    "albedo_accel",
    "thermal_accel",
)
