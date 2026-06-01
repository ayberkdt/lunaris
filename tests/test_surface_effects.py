# -*- coding: utf-8 -*-
"""Unit tests for lunar surface radiation config + wrappers.

Covers :mod:`lunaris.physics.surface_effects`: the `AlbedoConfig` /
`ThermalConfig` validation contracts, the legacy `accel_albedo_simple`
cannonball kernel, and the high-level `albedo_accel` / `thermal_accel` wrappers.

(Grid loading / PDS3 sampling lives in `lunaris.loaders` and is covered by the
loader test modules; this file is physics/config only.)
"""

from __future__ import annotations

import numpy as np
import pytest

from lunaris.common.constants import AU, P_SUN_1AU, R_MOON_MEAN
from lunaris.common.type_defs import SpacecraftProps
from lunaris.physics.surface_effects import (
    AlbedoConfig,
    ThermalConfig,
    accel_albedo_simple,
    albedo_accel,
    thermal_accel,
)


# ---------------------------------------------------------------------------
# AlbedoConfig
# ---------------------------------------------------------------------------

def test_albedo_config_defaults():
    c = AlbedoConfig()
    assert c.albedo_model == "lambert_facets"
    assert c.albedo_mode == "constant_albedo"
    assert c.albedo_const == 0.12
    assert c.albedo_pressure_coefficient == 1.0
    assert c.enable_eclipse is True
    assert c.facet_lat_count >= 1 and c.facet_lon_count >= 1


def test_albedo_config_a_moon_alias_both_directions():
    assert AlbedoConfig(A_moon=0.3).albedo_const == 0.3      # legacy field -> canonical
    assert AlbedoConfig(albedo_const=0.2).A_moon == 0.2      # canonical -> legacy


def test_albedo_config_mode_aliases_normalized():
    assert AlbedoConfig(albedo_mode="constant").albedo_mode == "constant_albedo"
    assert AlbedoConfig(albedo_mode="grid").albedo_mode == "albedo_grid"
    assert AlbedoConfig(albedo_mode="dn").albedo_mode == "scaled_dn_grid"


def test_albedo_config_model_aliases_normalized():
    assert AlbedoConfig(albedo_model="facet").albedo_model == "lambert_facets"
    assert AlbedoConfig(albedo_model="LAMBERT_FACETS").albedo_model == "lambert_facets"
    assert AlbedoConfig(albedo_model="simple").albedo_model == "simple"


@pytest.mark.parametrize("kw", [
    {"albedo_model": "nope"},
    {"albedo_model": "lommel"},                                   # backend removed in cleanup
    {"albedo_mode": "bogus"},
    {"A_moon": 1.5},
    {"albedo_const": -0.1},
    {"albedo_pressure_coefficient": -1.0},
    {"facet_lat_count": 0},
    {"facet_lon_count": 0},
    {"facet_lat_count": 200, "facet_lon_count": 200, "max_facets": 100},  # exceeds max_facets
])
def test_albedo_config_validation_rejects(kw):
    with pytest.raises(ValueError):
        AlbedoConfig(**kw)


def test_albedo_config_lommel_backend_removed():
    # The legacy lommel backend was removed; only lambert_facets / simple remain.
    with pytest.raises(ValueError):
        AlbedoConfig(albedo_model="lommel")


# ---------------------------------------------------------------------------
# ThermalConfig
# ---------------------------------------------------------------------------

def test_thermal_config_mode_normalization():
    assert ThermalConfig(thermal_mode="equilibrium").thermal_mode == "equilibrium_temperature"
    assert ThermalConfig(thermal_mode="grid").thermal_mode == "temperature_grid"
    assert ThermalConfig(thermal_mode="constant").thermal_mode == "constant_temperature"


def test_thermal_config_k_thermal_alias():
    assert ThermalConfig(k_thermal=2.0).ir_pressure_coefficient == 2.0


@pytest.mark.parametrize("kw", [
    {"thermal_mode": "bogus"},
    {"surface_emissivity": 1.5},
    {"surface_albedo": -0.1},
    {"facet_lat_count": 0},
])
def test_thermal_config_validation_rejects(kw):
    with pytest.raises(ValueError):
        ThermalConfig(**kw)


# ---------------------------------------------------------------------------
# accel_albedo_simple (legacy cannonball kernel)
# ---------------------------------------------------------------------------

def _simple(area_m2=2.0, mass_kg=1000.0, sun=(-AU, 0.0, 0.0), eclipse=0):
    # SC on +X; Sun on -X by default so the Sun->SC push is +X (no eclipse).
    return accel_albedo_simple(
        2.0e6, 0.0, 0.0,
        float(sun[0]), float(sun[1]), float(sun[2]),
        R_MOON_MEAN, AU, P_SUN_1AU, 0.12, 1.0, 1.5, area_m2, mass_kg, eclipse,
    )


def test_accel_albedo_simple_zero_guards():
    assert _simple(area_m2=0.0) == (0.0, 0.0, 0.0)
    assert _simple(mass_kg=0.0) == (0.0, 0.0, 0.0)


def test_accel_albedo_simple_direction_is_sun_to_spacecraft():
    ax, ay, az = _simple()
    assert ax > 0.0
    assert abs(ay) < 1e-30 and abs(az) < 1e-30


def test_accel_albedo_simple_scales_with_area_over_mass():
    a1 = _simple(area_m2=1.0, mass_kg=1000.0)[0]
    a2 = _simple(area_m2=2.0, mass_kg=1000.0)[0]
    a3 = _simple(area_m2=2.0, mass_kg=2000.0)[0]
    assert np.isclose(a2 / a1, 2.0, rtol=1e-12)
    assert np.isclose(a3 / a2, 0.5, rtol=1e-12)


# ---------------------------------------------------------------------------
# Wrappers
# ---------------------------------------------------------------------------

def test_albedo_accel_default_backend_is_facet_and_outward():
    r = np.asarray([R_MOON_MEAN + 5.0e5, 0.0, 0.0])
    s = np.asarray([AU, 0.0, 0.0])                       # sub-solar
    sc = SpacecraftProps(mass_kg=1000.0, area_m2=1.0, cr=1.5)
    a = albedo_accel(r, s, sc, AlbedoConfig())
    assert np.all(np.isfinite(a))
    assert a[0] > 0.0


def test_albedo_accel_simple_backend():
    r = np.asarray([R_MOON_MEAN + 5.0e5, 0.0, 0.0])
    s = np.asarray([-AU, 0.0, 0.0])                      # Sun -X -> push +X
    sc = SpacecraftProps(mass_kg=1000.0, area_m2=1.0, cr=1.5)
    a = albedo_accel(r, s, sc, AlbedoConfig(albedo_model="simple"), model="simple", enable_eclipse=False)
    assert a[0] > 0.0


def test_albedo_accel_unsupported_model_raises():
    sc = SpacecraftProps()
    with pytest.raises(ValueError):
        albedo_accel(np.asarray([2.0e6, 0.0, 0.0]), np.asarray([AU, 0.0, 0.0]), sc,
                     AlbedoConfig(), model="lommel")


def test_thermal_accel_returns_finite():
    r = np.asarray([R_MOON_MEAN + 5.0e5, 0.0, 0.0])
    s = np.asarray([AU, 0.0, 0.0])
    sc = SpacecraftProps(mass_kg=1000.0, area_m2=1.0, cr=1.5)
    a = thermal_accel(
        r, s, sc,
        ThermalConfig(thermal_mode="constant_temperature", facet_lat_count=6, facet_lon_count=12),
    )
    assert np.all(np.isfinite(a))
