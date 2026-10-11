"""Dynamics-engine wiring of the lunar albedo / thermal-IR surface integrations.

* The engine's legacy ``simple`` albedo kernel honours ``AlbedoConfig.enable_eclipse``
  exactly like :func:`lunaris.physics.surface_effects.albedo_accel` (it used to
  hard-code the Moon-shadow test on).
* ``surface_integration`` reaches every engine path (cached RHS kernel,
  acceleration breakdown) and agrees with the standalone wrappers.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from lunaris.common.constants import R_MOON
from lunaris.common.type_defs import PerturbationFlags, SpacecraftProps
from lunaris.core.dynamics import DynamicsEngine
from lunaris.physics.spherical_harmonics import compute_point_mass_acceleration
from lunaris.physics.surface_effects import AlbedoConfig, ThermalConfig, albedo_accel, thermal_accel

AU_M = 1.495978707e11
SC = SpacecraftProps(mass_kg=500.0, area_m2=8.0, cr=1.4)


class _StaticEphem:
    """Sun on +x at 1 AU, Earth fixed, identity Moon attitude (inertial == Moon-fixed)."""

    def __init__(self, sun: tuple[float, float, float] = (AU_M, 0.0, 0.0)) -> None:
        n = 8
        self._sun = np.tile(np.asarray(sun, dtype=np.float64), (n, 1))
        self._earth = np.tile(np.array([-3.0e8, 2.4e8, 1.0e7]), (n, 1))
        self._q = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n, 1))

    def get_data_provider(self) -> dict[str, Any]:
        z = np.zeros_like(self._sun)
        return {
            "dt_s": 60.0,
            "r_sun_tab_m": self._sun,
            "r_earth_tab_m": self._earth,
            "v_sun_tab_m_s": z,
            "v_earth_tab_m_s": z,
            "q_i2f_tab": self._q,
            "ephemeris_schema_version": 2,
        }


def _surface_accel(engine: DynamicsEngine, y: np.ndarray) -> np.ndarray:
    """RHS acceleration minus the point-mass term (the only other force)."""
    a = engine._build_solver_rhs()(0.0, y)[3:6]
    mu = float(engine._prep["grav"].gm_m3s2)
    return a - np.array(compute_point_mass_acceleration(y[0], y[1], y[2], mu))


# A state in the Moon's shadow (anti-Sun side, on the shadow axis).
Y_SHADOW = np.array([-(R_MOON + 100e3), 0.0, 0.0, 0.0, -1600.0, 0.0])
# |a_point_mass| ~ 1.45 m/s^2 at 100 km; 8 ulp of it bounds the subtraction error.
_PM_ATOL = 8.0 * float(np.spacing(1.45))


@pytest.mark.parametrize("enable_eclipse", [True, False])
def test_simple_albedo_honours_enable_eclipse(enable_eclipse: bool) -> None:
    cfg = AlbedoConfig(albedo_model="simple", enable_eclipse=enable_eclipse)
    eng = DynamicsEngine(
        SC, PerturbationFlags(enable_sh=False, enable_albedo=True), ephem_manager=_StaticEphem(), albedo=cfg,
    )
    got = _surface_accel(eng, Y_SHADOW)
    expected = albedo_accel(Y_SHADOW[:3], (AU_M, 0.0, 0.0), SC, cfg, R_moon=R_MOON)
    if enable_eclipse:
        assert np.all(expected == 0.0)
    else:
        assert np.linalg.norm(expected) > 1e-12
    # RHS minus point mass cancels ~1.6 m/s^2: absolute budget a few ulp of that.
    np.testing.assert_allclose(got, expected, rtol=0.0, atol=_PM_ATOL)
    breakdown = eng.get_acceleration_vector_breakdown(0.0, Y_SHADOW)["Albedo"]
    np.testing.assert_allclose(breakdown, expected, rtol=1e-12, atol=1e-30)


# Sunlit state at 100 km, sub-satellite point away from the Sun line.
Y_DAY = np.array([0.6 * (R_MOON + 100e3), 0.8 * (R_MOON + 100e3), 0.0, -1300.0, 980.0, 0.0])
EARTH = (-3.0e8, 2.4e8, 1.0e7)  # _StaticEphem's Earth


def _engine(albedo: AlbedoConfig | None = None, thermal: ThermalConfig | None = None, **kw: Any) -> DynamicsEngine:
    flags = PerturbationFlags(enable_sh=False, enable_albedo=albedo is not None, enable_thermal=thermal is not None)
    return DynamicsEngine(SC, flags, ephem_manager=_StaticEphem(), albedo=albedo, thermal=thermal, **kw)


def test_quadrature_is_the_default_surface_integration() -> None:
    eng = _engine(AlbedoConfig(), ThermalConfig())
    eng._build_solver_rhs()
    assert eng._prep["alb"].integration == 1
    assert eng._prep["thermal"].integration == 1
    eng = _engine(AlbedoConfig(surface_integration="facets"), ThermalConfig(surface_integration="facets"))
    eng._build_solver_rhs()
    assert eng._prep["alb"].integration == 0
    assert eng._prep["thermal"].integration == 0


@pytest.mark.parametrize("integration", ["visible_disk_quadrature", "facets"])
def test_engine_albedo_matches_wrapper(integration: str) -> None:
    cfg = AlbedoConfig(surface_integration=integration, albedo_const=0.15)
    eng = _engine(cfg)
    expected = albedo_accel(Y_DAY[:3], (AU_M, 0.0, 0.0), SC, cfg, R_moon=R_MOON, r_earth=EARTH)
    assert np.linalg.norm(expected) > 1e-10
    breakdown = eng.get_acceleration_vector_breakdown(0.0, Y_DAY)["Albedo"]
    np.testing.assert_allclose(breakdown, expected, rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(_surface_accel(eng, Y_DAY), expected, rtol=0.0, atol=_PM_ATOL)


@pytest.mark.parametrize("integration", ["visible_disk_quadrature", "facets"])
@pytest.mark.parametrize("mode", ["constant_temperature", "equilibrium_temperature"])
def test_engine_thermal_matches_wrapper(integration: str, mode: str) -> None:
    cfg = ThermalConfig(surface_integration=integration, thermal_mode=mode)
    eng = _engine(thermal=cfg)
    expected = thermal_accel(Y_DAY[:3], (AU_M, 0.0, 0.0), SC, cfg, R_moon=R_MOON, r_earth=EARTH)
    assert np.linalg.norm(expected) > 1e-10
    breakdown = eng.get_acceleration_vector_breakdown(0.0, Y_DAY)["Thermal IR"]
    np.testing.assert_allclose(breakdown, expected, rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(_surface_accel(eng, Y_DAY), expected, rtol=0.0, atol=_PM_ATOL)


def test_quadrature_reads_mapped_cells() -> None:
    """A uniform albedo map / temperature grid must reproduce the constant-value quadrature."""
    n_lat, n_lon = 18, 36
    surf = {
        "albedo_grid": np.full((180, 360), 0.15), "n_lines": 180, "n_samples": 360, "res_deg": 1.0,
        "lon0_deg": 0.0, "lat0_deg": 90.0,
        "facet_temperatures_K": np.full(n_lat * n_lon, 250.0),
    }
    mapped = _engine(
        AlbedoConfig(albedo_mode="albedo_grid"), ThermalConfig(thermal_mode="temperature_grid"), surface_provider=surf,
    )
    const = _engine(AlbedoConfig(albedo_const=0.15), ThermalConfig(thermal_mode="constant_temperature"))
    mapped._build_solver_rhs()
    assert (mapped._prep["alb"].cell_lat_count, mapped._prep["alb"].cell_lon_count) == (n_lat, n_lon)
    assert (mapped._prep["thermal"].cell_lat_count, mapped._prep["thermal"].cell_lon_count) == (n_lat, n_lon)
    got = mapped.get_acceleration_vector_breakdown(0.0, Y_DAY)
    ref = const.get_acceleration_vector_breakdown(0.0, Y_DAY)
    np.testing.assert_allclose(got["Albedo"], ref["Albedo"], rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(got["Thermal IR"], ref["Thermal IR"], rtol=1e-12, atol=0.0)
