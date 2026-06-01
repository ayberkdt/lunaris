# -*- coding: utf-8 -*-
"""Regression tests for lunar thermal IR radiation pressure."""

from __future__ import annotations

import math

import numpy as np

from lunaris.common.constants import AU, R_MOON
from lunaris.common.type_defs import PerturbationFlags, SpacecraftProps
from lunaris.core.dynamics import DynamicsEngine
from lunaris.physics.surface_effects import ThermalConfig
from lunaris.physics.thermal_ir import (
    THERMAL_MODE_EQUILIBRIUM,
    build_latlon_facets,
    calc_thermal_ir_accel,
    accel_thermal_ir_facets_numba,
    thermal_ir_single_facet_accel_numba,
)


def test_latlon_facet_areas_sum_to_lunar_surface_area():
    _, _, areas, _, _ = build_latlon_facets(18, 36, radius_m=R_MOON)
    expected = 4.0 * math.pi * R_MOON * R_MOON
    assert np.isclose(float(np.sum(areas)), expected, rtol=1e-12)


def test_single_facet_zero_guards_and_direction():
    zero = thermal_ir_single_facet_accel_numba(
        R_MOON + 1000.0, 0.0, 0.0,
        R_MOON, 0.0, 0.0,
        1.0, 0.0, 0.0,
        0.0, 300.0, 1.0, 1.0, 100.0, 299_792_458.0,
    )
    assert zero == (0.0, 0.0, 0.0)

    ax, ay, az = thermal_ir_single_facet_accel_numba(
        R_MOON + 1000.0, 0.0, 0.0,
        R_MOON, 0.0, 0.0,
        1.0, 0.0, 0.0,
        10.0, 300.0, 1.0, 2.0, 100.0, 299_792_458.0,
    )
    assert ax > 0.0
    assert abs(ay) < 1e-30
    assert abs(az) < 1e-30


def test_single_facet_inverse_square_scaling():
    common = (
        R_MOON, 0.0, 0.0,
        1.0, 0.0, 0.0,
        10.0, 300.0, 1.0, 1.0, 100.0, 299_792_458.0,
    )
    a1 = thermal_ir_single_facet_accel_numba(R_MOON + 1000.0, 0.0, 0.0, *common)[0]
    a2 = thermal_ir_single_facet_accel_numba(R_MOON + 2000.0, 0.0, 0.0, *common)[0]
    assert np.isclose(a1 / a2, 4.0, rtol=1e-12)


def test_equilibrium_mode_respects_day_night_geometry():
    pos = np.asarray([[R_MOON, 0.0, 0.0]], dtype=np.float64)
    normals = np.asarray([[1.0, 0.0, 0.0]], dtype=np.float64)
    areas = np.asarray([100.0], dtype=np.float64)
    temps = np.zeros(1, dtype=np.float64)

    night = accel_thermal_ir_facets_numba(
        R_MOON + 1000.0, 0.0, 0.0,
        -AU, 0.0, 0.0,
        pos, normals, areas, temps,
        THERMAL_MODE_EQUILIBRIUM,
        0.95, 0.12, 250.0, 0.0, 0.0, 1.0, 1.0, 100.0,
        1367.0, AU, 299_792_458.0, 5.670_374_419e-8, True,
    )
    day = accel_thermal_ir_facets_numba(
        R_MOON + 1000.0, 0.0, 0.0,
        AU, 0.0, 0.0,
        pos, normals, areas, temps,
        THERMAL_MODE_EQUILIBRIUM,
        0.95, 0.12, 250.0, 0.0, 0.0, 1.0, 1.0, 100.0,
        1367.0, AU, 299_792_458.0, 5.670_374_419e-8, True,
    )

    assert night == (0.0, 0.0, 0.0)
    assert day[0] > 0.0


def test_constant_mode_rotates_consistently():
    pos = np.asarray([[R_MOON, 0.0, 0.0]], dtype=np.float64)
    normals = np.asarray([[1.0, 0.0, 0.0]], dtype=np.float64)
    areas = np.asarray([100.0], dtype=np.float64)
    r = np.asarray([R_MOON + 1000.0, 0.0, 0.0])
    sun = np.asarray([AU, 0.0, 0.0])

    a = calc_thermal_ir_accel(r, sun, pos, normals, areas, temperature_K=250.0)

    rot = np.asarray(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    a_rot = calc_thermal_ir_accel(
        rot @ r,
        rot @ sun,
        pos @ rot.T,
        normals @ rot.T,
        areas,
        temperature_K=250.0,
    )

    assert np.allclose(a_rot, rot @ a, rtol=1e-12, atol=1e-30)


def test_dynamics_thermal_ir_adds_finite_acceleration_without_surface_provider():
    sc = SpacecraftProps(mass_kg=100.0, area_m2=2.0, cr=1.2)
    y = np.asarray([R_MOON + 1_000_000.0, 0.0, 0.0, 0.0, 1600.0, 0.0], dtype=np.float64)

    engine = DynamicsEngine(
        sc_props=sc,
        flags=PerturbationFlags(enable_sh=False, enable_thermal=True),
        thermal=ThermalConfig(thermal_mode="constant_temperature", facet_lat_count=4, facet_lon_count=8),
        allow_identity_rotation=True,
    )
    rhs = engine.build_rhs(force_rebuild=True)

    dydt = rhs(0.0, y)
    breakdown = engine.get_acceleration_breakdown(0.0, y)
    assert np.all(np.isfinite(dydt[3:6]))
    assert breakdown["Thermal IR"] > 0.0
