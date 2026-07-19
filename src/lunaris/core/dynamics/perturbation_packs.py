"""Validated perturbation packs for dynamics RHS construction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lunaris.common.constants import AU, C_LIGHT, EPS_1E15, R_EARTH_MEAN, SOLAR_FLUX_1AU
from lunaris.common.type_defs import F64Array
from lunaris.core.dynamics.requirements import _as_f64_c
from lunaris.physics.thermal_ir import (
    THERMAL_MODE_CONSTANT,
    THERMAL_MODE_EQUILIBRIUM,
    THERMAL_MODE_TEMPERATURE_GRID,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class _AlbedoPack:
    """
    Engine-internal albedo configuration.

    ``backend`` selects the reflected-solar model used in the RHS:
      0 = simple legacy cannonball (``accel_albedo_simple``); the acceleration
          points along the Sun->spacecraft line using a single sub-satellite
          albedo sample and the spacecraft SRP coefficient ``cr``.
      1 = lambert_facets (``accel_albedo_facets_numba``); a Lambertian facet sum
          using the precomputed per-facet albedo array and the dedicated
          ``pressure_coefficient`` (C_R_albedo).

    For backend 0, ``mode`` selects the sub-satellite albedo source:
      0 = albedo grid (grid_alb)
      1 = scaled DN grid (dn; albedo = sf*DN + off)
      2 = constant albedo (alb_const)
    """
    backend: int = 1

    # --- legacy simple-backend sub-satellite sampling ---
    mode: int = 2
    alb_const: float = 0.12
    alb_scale: float = 1.0
    k_lambert: float = 1.0

    grid_alb: F64Array | None = None
    dn: F64Array | None = None

    n_lines: int = 0
    n_samples: int = 0
    res_deg: float = 0.0
    lon0_deg: float = 0.0
    lat0_deg: float = 0.0
    sf: float = 1.0
    off: float = 0.0
    missing: float = -9999.0
    flip: int = 0
    latmin: float = -90.0
    latmax: float = 90.0

    # --- lambert_facets backend (precomputed at setup) ---
    facet_pos_m: F64Array | None = None
    facet_normals: F64Array | None = None
    facet_areas_m2: F64Array | None = None
    facet_albedo: F64Array | None = None
    pressure_coefficient: float = 1.0
    solar_flux_1au_W_m2: float = float(SOLAR_FLUX_1AU)
    au_m: float = float(AU)
    c_light_m_s: float = float(C_LIGHT)
    r_earth_m: float = float(R_EARTH_MEAN)
    include_sun_distance_scaling: bool = True
    enable_eclipse: bool = True

    def __post_init__(self) -> None:
        if self.backend not in (0, 1):
            raise ValueError(f"albedo backend must be 0 or 1, got {self.backend}")
        if self.mode not in (0, 1, 2):
            raise ValueError(f"albedo mode must be 0/1/2, got {self.mode}")

        # Facet arrays are always normalized to contiguous float64 with >= 1 row
        # so the closure captures valid arrays even when the facet backend is
        # inactive (the RHS skips the albedo block entirely when disabled).
        pos = (
            np.zeros((1, 3), dtype=np.float64)
            if self.facet_pos_m is None
            else _as_f64_c(self.facet_pos_m, "albedo.facet_pos_m")
        )
        normals = (
            np.zeros((1, 3), dtype=np.float64)
            if self.facet_normals is None
            else _as_f64_c(self.facet_normals, "albedo.facet_normals")
        )
        areas = (
            np.zeros(1, dtype=np.float64)
            if self.facet_areas_m2 is None
            else _as_f64_c(self.facet_areas_m2, "albedo.facet_areas_m2")
        )
        albedo = (
            np.zeros(1, dtype=np.float64)
            if self.facet_albedo is None
            else _as_f64_c(self.facet_albedo, "albedo.facet_albedo")
        )

        if self.backend == 1:
            if pos.ndim != 2 or pos.shape[1] != 3:
                raise ValueError(f"albedo.facet_pos_m must be (N,3), got {pos.shape}")
            if normals.ndim != 2 or normals.shape[1] != 3:
                raise ValueError(f"albedo.facet_normals must be (N,3), got {normals.shape}")
            if areas.ndim != 1:
                raise ValueError(f"albedo.facet_areas_m2 must be 1D, got {areas.shape}")
            if albedo.ndim != 1:
                raise ValueError(f"albedo.facet_albedo must be 1D, got {albedo.shape}")
            n = pos.shape[0]
            if normals.shape[0] != n or areas.shape[0] != n or albedo.shape[0] != n:
                raise ValueError(
                    "albedo facet positions, normals, areas, and albedo must share "
                    f"the same row count (got {pos.shape[0]}, {normals.shape[0]}, "
                    f"{areas.shape[0]}, {albedo.shape[0]})."
                )
            if not np.all(np.isfinite(albedo)):
                raise ValueError("albedo.facet_albedo must be finite.")
            if float(np.min(albedo)) < 0.0 or float(np.max(albedo)) > 1.0:
                raise ValueError("albedo.facet_albedo must lie in [0, 1].")

        object.__setattr__(self, "facet_pos_m", pos)
        object.__setattr__(self, "facet_normals", normals)
        object.__setattr__(self, "facet_areas_m2", areas)
        object.__setattr__(self, "facet_albedo", albedo)

        # Legacy grid validation only matters for the simple backend with a grid.
        if self.backend == 0 and self.mode != 2:
            if self.res_deg <= 0.0 or self.n_lines <= 0 or self.n_samples <= 0:
                raise ValueError("albedo grid params invalid (res_deg, n_lines, n_samples must be positive)")
            if not (self.latmin < self.latmax):
                raise ValueError(f"latmin must be < latmax (latmin={self.latmin}, latmax={self.latmax})")

            if self.mode == 0:
                if self.grid_alb is None:
                    raise ValueError("mode=0 requires grid_alb")
                grid = _as_f64_c(self.grid_alb, "grid_alb")
                if grid.ndim != 2:
                    raise ValueError(f"grid_alb must be 2D, got ndim={grid.ndim}")
                if grid.shape != (self.n_lines, self.n_samples):
                    raise ValueError(
                        f"grid_alb shape mismatch: expected {(self.n_lines, self.n_samples)}, got {grid.shape}"
                    )
                object.__setattr__(self, "grid_alb", grid)

            else:  # mode == 1
                if self.dn is None:
                    raise ValueError("mode=1 requires dn")
                dn = _as_f64_c(self.dn, "dn")
                if dn.ndim != 2:
                    raise ValueError(f"dn must be 2D, got ndim={dn.ndim}")
                if dn.shape != (self.n_lines, self.n_samples):
                    raise ValueError(
                        f"dn shape mismatch: expected {(self.n_lines, self.n_samples)}, got {dn.shape}"
                    )
                object.__setattr__(self, "dn", dn)


@dataclass(frozen=True, slots=True, kw_only=True)
class _EarthJ2Pack:
    """Engine-internal Earth J2 configuration (validated axis and reference radius)."""
    j2: float
    r_ref_m: float
    ax: float
    ay: float
    az: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.j2) or self.j2 < 0.0:
            raise ValueError(f"EarthJ2 j2 must be finite and >= 0, got {self.j2}")
        if not np.isfinite(self.r_ref_m) or self.r_ref_m <= 0.0:
            raise ValueError(f"EarthJ2 r_ref_m must be finite and > 0, got {self.r_ref_m}")
        if not np.all(np.isfinite((self.ax, self.ay, self.az))):
            raise ValueError("EarthJ2 axis vector must contain only finite values.")
        n = (self.ax * self.ax + self.ay * self.ay + self.az * self.az) ** 0.5
        if n <= EPS_1E15:
            raise ValueError("EarthJ2 axis vector is degenerate (norm ~ 0).")
        if abs(n - 1.0) > 1.0e-12:
            raise ValueError(f"EarthJ2 axis vector must be unit length, got norm={n!r}.")


@dataclass(frozen=True, slots=True, kw_only=True)
class _TidePack:
    """Engine-internal solid tide configuration."""
    use_k2: bool
    use_k3: bool
    use_earth: bool
    use_sun: bool
    k2: float
    k3: float
    r_ref_m: float

    def __post_init__(self) -> None:
        if self.r_ref_m <= 0.0 or not np.isfinite(self.r_ref_m):
            raise ValueError(f"solid tide r_ref_m must be finite and > 0, got {self.r_ref_m}")
        if not np.isfinite(self.k2) or self.k2 < 0.0:
            raise ValueError(f"solid tide k2 must be finite and >= 0, got {self.k2}")
        if not np.isfinite(self.k3) or self.k3 < 0.0:
            raise ValueError(f"solid tide k3 must be finite and >= 0, got {self.k3}")
        if (self.use_k2 or self.use_k3) and not (self.use_earth or self.use_sun):
            raise ValueError("solid tides require at least one tide-raising body.")


@dataclass(frozen=True, slots=True, kw_only=True)
class _ThermalPack:
    """Engine-internal Lambertian thermal IR configuration."""
    mode: int
    surface_emissivity: float
    surface_albedo: float
    temperature_K: float
    night_temperature_K: float
    thermal_floor_flux_W_m2: float
    ir_pressure_coefficient: float
    solar_flux_1au_W_m2: float
    au_m: float
    c_light_m_s: float
    sigma_sb: float
    include_sun_distance_scaling: bool
    enable_eclipse: bool
    r_earth_m: float
    facet_pos_m: F64Array
    facet_normals: F64Array
    facet_areas_m2: F64Array
    facet_temperatures_K: F64Array

    def __post_init__(self) -> None:
        if self.mode not in (THERMAL_MODE_CONSTANT, THERMAL_MODE_EQUILIBRIUM, THERMAL_MODE_TEMPERATURE_GRID):
            raise ValueError(f"thermal mode must be 0/1/2, got {self.mode}")
        if self.r_earth_m <= 0.0:
            raise ValueError(f"thermal r_earth_m must be > 0, got {self.r_earth_m}")

        for name in (
            "surface_emissivity",
            "surface_albedo",
            "temperature_K",
            "night_temperature_K",
            "thermal_floor_flux_W_m2",
            "ir_pressure_coefficient",
            "solar_flux_1au_W_m2",
            "au_m",
            "c_light_m_s",
            "sigma_sb",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"thermal {name} must be finite, got {value}")

        if not (0.0 <= self.surface_emissivity <= 1.0):
            raise ValueError(f"thermal surface_emissivity must be in [0,1], got {self.surface_emissivity}")
        if not (0.0 <= self.surface_albedo <= 1.0):
            raise ValueError(f"thermal surface_albedo must be in [0,1], got {self.surface_albedo}")
        if self.temperature_K < 0.0 or self.night_temperature_K < 0.0:
            raise ValueError("thermal temperatures must be >= 0 K")
        if self.thermal_floor_flux_W_m2 < 0.0:
            raise ValueError("thermal_floor_flux_W_m2 must be >= 0")
        if self.ir_pressure_coefficient < 0.0:
            raise ValueError("thermal ir_pressure_coefficient must be >= 0")
        if self.solar_flux_1au_W_m2 < 0.0:
            raise ValueError("thermal solar_flux_1au_W_m2 must be >= 0")
        if self.au_m <= 0.0 or self.c_light_m_s <= 0.0 or self.sigma_sb <= 0.0:
            raise ValueError("thermal au_m, c_light_m_s, and sigma_sb must be > 0")

        pos = _as_f64_c(self.facet_pos_m, "thermal.facet_pos_m")
        normals = _as_f64_c(self.facet_normals, "thermal.facet_normals")
        areas = _as_f64_c(self.facet_areas_m2, "thermal.facet_areas_m2")
        temps = _as_f64_c(self.facet_temperatures_K, "thermal.facet_temperatures_K")

        if pos.ndim != 2 or pos.shape[1] != 3:
            raise ValueError(f"thermal.facet_pos_m must have shape (N,3), got {pos.shape}")
        if normals.ndim != 2 or normals.shape[1] != 3:
            raise ValueError(f"thermal.facet_normals must have shape (N,3), got {normals.shape}")
        if areas.ndim != 1:
            raise ValueError(f"thermal.facet_areas_m2 must be 1D, got {areas.shape}")
        if pos.shape[0] != normals.shape[0] or pos.shape[0] != areas.shape[0]:
            raise ValueError("thermal facet positions, normals, and areas must share the same row count")
        if self.mode == THERMAL_MODE_TEMPERATURE_GRID and temps.shape[0] != areas.shape[0]:
            raise ValueError(
                "temperature_grid mode requires one temperature per thermal facet "
                f"(got temps={temps.shape[0]}, facets={areas.shape[0]})."
            )

        object.__setattr__(self, "facet_pos_m", pos)
        object.__setattr__(self, "facet_normals", normals)
        object.__setattr__(self, "facet_areas_m2", areas)
        object.__setattr__(self, "facet_temperatures_K", temps)

__all__ = ["_AlbedoPack", "_EarthJ2Pack", "_TidePack", "_ThermalPack"]
