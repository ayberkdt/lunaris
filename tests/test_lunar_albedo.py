# -*- coding: utf-8 -*-
"""Regression tests for lunar albedo (reflected-solar) radiation pressure.

Covers the facet Lambertian model in :mod:`lunaris.physics.lunar_albedo` and its
integration through :class:`lunaris.core.dynamics.DynamicsEngine`: facet geometry,
zero/disabled cases, reflection direction, physical scaling, day/night gating,
frame consistency, provider/grid validation, and RHS wiring.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from lunaris.common.constants import AU, C_LIGHT, R_MOON, SOLAR_FLUX_1AU
from lunaris.common.type_defs import PerturbationFlags, SpacecraftProps
from lunaris.core.dynamics import DynamicsEngine, _AlbedoPack
from lunaris.physics.surface_effects import AlbedoConfig, albedo_accel
from lunaris.physics.lunar_albedo import (
    accel_albedo_facets_numba,
    albedo_single_facet_accel_numba,
    build_latlon_facets,
    calc_albedo_accel,
    normalize_albedo_mode,
)


# A facet center at +X with an outward +X normal: directly under the Sun and the
# spacecraft when both sit on the +X axis (mu_sun = mu_view = 1).
_FACET_POS = np.asarray([[R_MOON, 0.0, 0.0]], dtype=np.float64)
_FACET_NRM = np.asarray([[1.0, 0.0, 0.0]], dtype=np.float64)
_FACET_AREA = np.asarray([100.0], dtype=np.float64)
_ALB = np.asarray([0.12], dtype=np.float64)


class _StubEphem:
    """Minimal ephemeris provider: constant Sun/Earth and identity attitude."""

    def __init__(self, sun, earth) -> None:
        self._d = {
            "dt": 1.0,
            "sun_table": np.tile(np.asarray(sun, dtype=np.float64), (2, 1)),
            "earth_table": np.tile(np.asarray(earth, dtype=np.float64), (2, 1)),
            "rot_table": np.tile(np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64), (2, 1)),
        }

    def get_data_provider(self):
        return self._d


def _single_facet(rx, *, area=100.0, albedo=0.12, pcoef=1.0, sc_area=1.0, sc_mass=1000.0,
                  sun=(AU, 0.0, 0.0)):
    return albedo_single_facet_accel_numba(
        rx, 0.0, 0.0,
        float(sun[0]), float(sun[1]), float(sun[2]),
        R_MOON, 0.0, 0.0,
        1.0, 0.0, 0.0,
        area, albedo, pcoef, sc_area, sc_mass,
        SOLAR_FLUX_1AU, AU, C_LIGHT, True,
    )


# ---------------------------------------------------------------------------
# (a) Geometry and area
# ---------------------------------------------------------------------------

def test_facet_areas_sum_to_lunar_surface_area():
    _, _, areas, _, _ = build_latlon_facets(18, 36, radius_m=R_MOON)
    expected = 4.0 * math.pi * R_MOON * R_MOON
    assert np.isclose(float(np.sum(areas)), expected, rtol=1e-12)


def test_facet_normals_are_unit_and_positions_on_sphere():
    pos, normals, _, _, _ = build_latlon_facets(12, 24, radius_m=R_MOON)
    nrm = np.linalg.norm(normals, axis=1)
    assert np.allclose(nrm, 1.0, rtol=1e-12, atol=1e-12)
    radii = np.linalg.norm(pos, axis=1)
    assert np.allclose(radii, R_MOON, rtol=1e-9)


# ---------------------------------------------------------------------------
# (b) Zero / disabled cases
# ---------------------------------------------------------------------------

def test_zero_albedo_gives_zero():
    assert _single_facet(R_MOON + 1000.0, albedo=0.0) == (0.0, 0.0, 0.0)


def test_zero_area_gives_zero():
    assert _single_facet(R_MOON + 1000.0, area=0.0) == (0.0, 0.0, 0.0)


def test_zero_pressure_coefficient_gives_zero():
    assert _single_facet(R_MOON + 1000.0, pcoef=0.0) == (0.0, 0.0, 0.0)


def test_zero_spacecraft_area_gives_zero():
    assert _single_facet(R_MOON + 1000.0, sc_area=0.0) == (0.0, 0.0, 0.0)


def test_no_illuminated_facets_gives_zero():
    # Sun coincident with the Moon center -> every facet normal faces away from
    # the Sun (mu_sun < 0), so the whole sphere contributes nothing.
    pos, normals, areas, _, _ = build_latlon_facets(6, 12, radius_m=R_MOON)
    albedo = np.full(areas.shape[0], 0.12, dtype=np.float64)
    a = accel_albedo_facets_numba(
        R_MOON + 1_000_000.0, 0.0, 0.0,
        0.0, 0.0, 0.0,                       # Sun at Moon center
        0.0, 0.0, 0.0,
        pos, normals, areas, albedo,
        1.0, 1.0, 1000.0, SOLAR_FLUX_1AU, AU, C_LIGHT, 6_371_000.0, True, False,
    )
    assert a == (0.0, 0.0, 0.0)


def test_no_visible_facet_gives_zero():
    # Spacecraft on the -X axis cannot see a +X facet (mu_view < 0).
    assert _single_facet(-(R_MOON + 1000.0)) == (0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# (c) Direction
# ---------------------------------------------------------------------------

def test_single_facet_points_from_facet_to_spacecraft():
    ax, ay, az = _single_facet(R_MOON + 1000.0, sc_area=2.0)
    assert ax > 0.0                      # facet at +X, SC further out at +X
    assert abs(ay) < 1e-30
    assert abs(az) < 1e-30


# ---------------------------------------------------------------------------
# (d) Scaling
# ---------------------------------------------------------------------------

def test_inverse_square_distance_scaling():
    a1 = _single_facet(R_MOON + 1000.0)[0]
    a2 = _single_facet(R_MOON + 2000.0)[0]
    assert np.isclose(a1 / a2, 4.0, rtol=1e-9)


def test_doubling_area_doubles_acceleration():
    a1 = _single_facet(R_MOON + 1000.0, sc_area=1.0)[0]
    a2 = _single_facet(R_MOON + 1000.0, sc_area=2.0)[0]
    assert np.isclose(a2 / a1, 2.0, rtol=1e-12)


def test_doubling_mass_halves_acceleration():
    a1 = _single_facet(R_MOON + 1000.0, sc_mass=500.0)[0]
    a2 = _single_facet(R_MOON + 1000.0, sc_mass=1000.0)[0]
    assert np.isclose(a2 / a1, 0.5, rtol=1e-12)


def test_doubling_albedo_doubles_acceleration():
    a1 = _single_facet(R_MOON + 1000.0, albedo=0.1)[0]
    a2 = _single_facet(R_MOON + 1000.0, albedo=0.2)[0]
    assert np.isclose(a2 / a1, 2.0, rtol=1e-12)


# ---------------------------------------------------------------------------
# (e) Day / night
# ---------------------------------------------------------------------------

def test_nightside_facet_does_not_reflect():
    # Sun on -X: the +X facet is on the lunar nightside (mu_sun < 0).
    ax, ay, az = _single_facet(R_MOON + 1000.0, sun=(-AU, 0.0, 0.0))
    assert (ax, ay, az) == (0.0, 0.0, 0.0)


def test_facets_respect_day_night_geometry():
    pos, normals, areas, _, _ = build_latlon_facets(8, 16, radius_m=R_MOON)
    albedo = np.full(areas.shape[0], 0.12, dtype=np.float64)
    common = (pos, normals, areas, albedo, 1.0, 2.0, 500.0, SOLAR_FLUX_1AU, AU, C_LIGHT, 6_371_000.0, True, False)
    day = accel_albedo_facets_numba(
        R_MOON + 500_000.0, 0.0, 0.0, AU, 0.0, 0.0, 0.0, 0.0, 0.0, *common
    )
    # Spacecraft over the nightside hemisphere, looking back at lit limb facets:
    # net push should still be finite but the dominant sub-solar facet is dark.
    assert day[0] > 0.0


# ---------------------------------------------------------------------------
# (f) Frame consistency
# ---------------------------------------------------------------------------

def test_facet_albedo_rotates_consistently():
    r = np.asarray([R_MOON + 1000.0, 0.0, 0.0])
    sun = np.asarray([AU, 0.0, 0.0])
    a = calc_albedo_accel(
        r, sun, _FACET_POS, _FACET_NRM, _FACET_AREA, _ALB,
        spacecraft_area_m2=2.0, spacecraft_mass_kg=500.0,
    )
    rot = np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    a_rot = calc_albedo_accel(
        rot @ r, rot @ sun, _FACET_POS @ rot.T, _FACET_NRM @ rot.T, _FACET_AREA, _ALB,
        spacecraft_area_m2=2.0, spacecraft_mass_kg=500.0,
    )
    assert np.allclose(a_rot, rot @ a, rtol=1e-12, atol=1e-30)


# ---------------------------------------------------------------------------
# (g) Dynamics integration
# ---------------------------------------------------------------------------

def test_dynamics_albedo_facets_adds_finite_acceleration():
    sc = SpacecraftProps(mass_kg=100.0, area_m2=2.0, cr=1.2)
    y = np.asarray([R_MOON + 1_000_000.0, 0.0, 0.0, 0.0, 1600.0, 0.0], dtype=np.float64)
    engine = DynamicsEngine(
        sc_props=sc,
        flags=PerturbationFlags(enable_sh=False, enable_albedo=True),
        ephem_manager=_StubEphem((AU, 0.0, 0.0), (0.0, 0.0, 0.0)),
        albedo=AlbedoConfig(albedo_model="lambert_facets", facet_lat_count=6, facet_lon_count=12),
    )
    rhs = engine.build_rhs(force_rebuild=True)
    dydt = rhs(0.0, y)
    breakdown = engine.get_acceleration_breakdown(0.0, y)
    assert np.all(np.isfinite(dydt[3:6]))
    assert breakdown["Albedo"] > 0.0


def test_dynamics_albedo_disabled_leaves_rhs_unchanged():
    sc = SpacecraftProps(mass_kg=100.0, area_m2=2.0, cr=1.2)
    y = np.asarray([R_MOON + 1_000_000.0, 0.0, 0.0, 0.0, 1600.0, 0.0], dtype=np.float64)
    base = DynamicsEngine(
        sc_props=sc, flags=PerturbationFlags(enable_sh=False, enable_albedo=False),
        allow_identity_rotation=True,
    )
    alb = DynamicsEngine(
        sc_props=sc, flags=PerturbationFlags(enable_sh=False, enable_albedo=True),
        ephem_manager=_StubEphem((AU, 0.0, 0.0), (0.0, 0.0, 0.0)),
        albedo=AlbedoConfig(facet_lat_count=6, facet_lon_count=12),
    )
    d_base = base.build_rhs(force_rebuild=True)(0.0, y)
    d_alb = alb.build_rhs(force_rebuild=True)(0.0, y)

    # Velocity rows are identical; the acceleration differs only by the albedo
    # term, whose magnitude equals the reported "Albedo" breakdown (albedo is
    # ~1e-9 m/s^2, far below point-mass gravity, so a raw allclose would mask it).
    assert np.allclose(d_base[0:3], d_alb[0:3])
    bk = alb.get_acceleration_breakdown(0.0, y)
    diff = float(np.linalg.norm(d_alb[3:6] - d_base[3:6]))
    assert bk["Albedo"] > 0.0
    assert np.isclose(diff, bk["Albedo"], rtol=1e-6, atol=0.0)
    assert "Albedo" not in base.get_acceleration_breakdown(0.0, y)


def test_dynamics_albedo_constant_mode_needs_no_provider():
    sc = SpacecraftProps(mass_kg=500.0, area_m2=1.0, cr=1.5)
    engine = DynamicsEngine(
        sc_props=sc,
        flags=PerturbationFlags(enable_sh=False, enable_albedo=True),
        ephem_manager=_StubEphem((AU, 0.0, 0.0), (0.0, 0.0, 0.0)),
        albedo=AlbedoConfig(albedo_mode="constant_albedo", facet_lat_count=4, facet_lon_count=8),
    )
    # No surface_provider supplied; constant mode must build without error.
    engine.build_rhs(force_rebuild=True)
    assert engine._prep["alb"].backend == 1


def test_dynamics_albedo_grid_mode_without_provider_raises():
    sc = SpacecraftProps(mass_kg=500.0, area_m2=1.0, cr=1.5)
    with pytest.raises(ValueError):
        DynamicsEngine(
            sc_props=sc,
            flags=PerturbationFlags(enable_sh=False, enable_albedo=True),
            ephem_manager=_StubEphem((AU, 0.0, 0.0), (0.0, 0.0, 0.0)),
            albedo=AlbedoConfig(albedo_mode="albedo_grid", facet_lat_count=4, facet_lon_count=8),
        )


def test_dynamics_albedo_simple_backend_still_available():
    sc = SpacecraftProps(mass_kg=100.0, area_m2=2.0, cr=1.2)
    y = np.asarray([R_MOON + 1_000_000.0, 0.0, 0.0, 0.0, 1600.0, 0.0], dtype=np.float64)
    engine = DynamicsEngine(
        sc_props=sc,
        flags=PerturbationFlags(enable_sh=False, enable_albedo=True),
        ephem_manager=_StubEphem((AU, 0.0, 0.0), (0.0, 0.0, 0.0)),
        albedo=AlbedoConfig(albedo_model="simple"),
    )
    rhs = engine.build_rhs(force_rebuild=True)
    assert engine._prep["alb"].backend == 0
    assert np.all(np.isfinite(rhs(0.0, y)[3:6]))


# ---------------------------------------------------------------------------
# (h) Provider / grid validation
# ---------------------------------------------------------------------------

def test_scaled_dn_facet_albedo_precomputed_from_provider():
    dn = np.full((90, 180), 100.0, dtype=np.float64)
    surf = {
        "dn": dn, "n_lines": 90, "n_samples": 180, "res_deg": 2.0,
        "lon0_deg": 0.0, "lat0_deg": 89.0, "scale_factor": 0.001, "offset": 0.05,
        "missing_dn": -1.0, "flip_lat": 0, "lat_min_deg": -90.0, "lat_max_deg": 90.0,
    }
    engine = DynamicsEngine(
        sc_props=SpacecraftProps(mass_kg=500.0, area_m2=1.0, cr=1.5),
        flags=PerturbationFlags(enable_sh=False, enable_albedo=True),
        ephem_manager=_StubEphem((AU, 0.0, 0.0), (0.0, 0.0, 0.0)),
        surface_provider=surf,
        albedo=AlbedoConfig(albedo_mode="scaled_dn_grid", facet_lat_count=6, facet_lon_count=12),
    )
    engine.build_rhs(force_rebuild=True)
    facet_albedo = engine._prep["alb"].facet_albedo
    assert np.allclose(facet_albedo, 0.001 * 100.0 + 0.05)  # uniform DN -> 0.15


def test_albedo_grid_shape_mismatch_raises():
    surf = {
        "albedo_grid": np.full((4, 8), 0.1, dtype=np.float64),
        "n_lines": 4, "n_samples": 9,                  # deliberately wrong
        "res_deg": 45.0, "lon0_deg": 0.0, "lat0_deg": 67.5,
    }
    engine = DynamicsEngine(
        sc_props=SpacecraftProps(mass_kg=500.0, area_m2=1.0, cr=1.5),
        flags=PerturbationFlags(enable_sh=False, enable_albedo=True),
        ephem_manager=_StubEphem((AU, 0.0, 0.0), (0.0, 0.0, 0.0)),
        surface_provider=surf,
        albedo=AlbedoConfig(albedo_mode="albedo_grid", facet_lat_count=4, facet_lon_count=8),
    )
    with pytest.raises(Exception):
        engine.build_rhs(force_rebuild=True)


def test_albedo_pack_rejects_out_of_range_facet_albedo():
    with pytest.raises(ValueError):
        _AlbedoPack(
            backend=1,
            facet_pos_m=_FACET_POS,
            facet_normals=_FACET_NRM,
            facet_areas_m2=_FACET_AREA,
            facet_albedo=np.asarray([1.5], dtype=np.float64),
        )


def test_albedo_pack_rejects_bad_backend():
    with pytest.raises(ValueError):
        _AlbedoPack(backend=5)


def test_albedo_pack_rejects_facet_shape_mismatch():
    with pytest.raises(ValueError):
        _AlbedoPack(
            backend=1,
            facet_pos_m=np.zeros((3, 3)),
            facet_normals=np.zeros((2, 3)),   # mismatched row count
            facet_areas_m2=np.zeros(3),
            facet_albedo=np.zeros(3),
        )


def test_facet_albedo_grid_captures_spatial_variation():
    # Albedo grid varying with latitude row -> per-facet albedo must not be uniform,
    # and must be clamped to [0, 1].
    nl, ns = 90, 180
    grid = (np.linspace(0.05, 0.20, nl)[:, None] * np.ones((1, ns))).astype(np.float64)
    surf = {
        "albedo_grid": grid, "n_lines": nl, "n_samples": ns,
        "res_deg": 2.0, "lon0_deg": 0.0, "lat0_deg": 89.0,
    }
    engine = DynamicsEngine(
        sc_props=SpacecraftProps(mass_kg=500.0, area_m2=1.0, cr=1.5),
        flags=PerturbationFlags(enable_sh=False, enable_albedo=True),
        ephem_manager=_StubEphem((AU, 0.0, 0.0), (0.0, 0.0, 0.0)),
        surface_provider=surf,
        albedo=AlbedoConfig(albedo_mode="albedo_grid", facet_lat_count=12, facet_lon_count=24),
    )
    engine.build_rhs(force_rebuild=True)
    fa = engine._prep["alb"].facet_albedo
    assert float(fa.min()) < float(fa.max())
    assert float(fa.min()) >= 0.0 and float(fa.max()) <= 1.0


# ---------------------------------------------------------------------------
# (i) Legacy simple backend through the propagator (latent-bug regression)
# ---------------------------------------------------------------------------

def test_dynamics_simple_backend_with_dn_grid_compiles_and_runs():
    # Regression: the legacy DN-grid albedo path called the non-Numba Python
    # sampler from inside @njit code and never compiled. It now uses the @njit
    # kernel, so a `simple` backend with a provider DN grid must build and run.
    dn = np.full((90, 180), 100.0, dtype=np.float64)
    surf = {
        "dn": dn, "n_lines": 90, "n_samples": 180, "res_deg": 2.0,
        "lon0_deg": 0.0, "lat0_deg": 89.0, "scale_factor": 0.001, "offset": 0.05,
        "missing_dn": -1.0, "flip_lat": 0, "lat_min_deg": -90.0, "lat_max_deg": 90.0,
        "albedo_const": 0.12,
    }
    sc = SpacecraftProps(mass_kg=100.0, area_m2=2.0, cr=1.3)
    y = np.asarray([R_MOON + 1_000_000.0, 0.0, 0.0, 0.0, 1600.0, 0.0], dtype=np.float64)
    engine = DynamicsEngine(
        sc_props=sc,
        flags=PerturbationFlags(enable_sh=False, enable_albedo=True),
        ephem_manager=_StubEphem((AU, 0.0, 0.0), (0.0, 0.0, 0.0)),
        surface_provider=surf,
        albedo=AlbedoConfig(albedo_model="simple"),
    )
    rhs = engine.build_rhs(force_rebuild=True)
    assert engine._prep["alb"].backend == 0
    assert engine._prep["alb"].mode == 1            # scaled-DN source
    assert np.all(np.isfinite(rhs(0.0, y)[3:6]))


def test_sample_albedo_dn_scaled_applies_scale_and_offset():
    from lunaris.core.dynamics import _sample_albedo_dn_scaled

    dn = np.full((90, 180), 100.0, dtype=np.float64)
    a = float(_sample_albedo_dn_scaled(0.0, 0.0, dn, 90, 180, 2.0, 0.0, 89.0, 0, 0.001, 0.05, -1.0, -90.0, 90.0))
    assert np.isclose(a, 0.001 * 100.0 + 0.05)


# ---------------------------------------------------------------------------
# (j) Lunar eclipse (Earth-umbra) dimming
# ---------------------------------------------------------------------------

def test_facet_albedo_lunar_eclipse_dims_signal():
    pos, normals, areas, _, _ = build_latlon_facets(8, 16, radius_m=R_MOON)
    albedo = np.full(areas.shape[0], 0.12, dtype=np.float64)
    r = (R_MOON + 500_000.0, 0.0, 0.0)
    sun = (AU, 0.0, 0.0)
    common = (pos, normals, areas, albedo, 1.0, 2.0, 500.0, SOLAR_FLUX_1AU, AU, C_LIGHT, 6_371_000.0, True, True)

    no_earth = accel_albedo_facets_numba(*r, *sun, 0.0, 0.0, 0.0, *common)
    # Earth between Moon (origin) and Sun on +X -> Moon center in Earth's umbra.
    eclipsed = accel_albedo_facets_numba(*r, *sun, 3.8e8, 0.0, 0.0, *common)

    assert np.linalg.norm(no_earth) > 0.0
    assert np.linalg.norm(eclipsed) < np.linalg.norm(no_earth)


# ---------------------------------------------------------------------------
# Config + standalone wrapper
# ---------------------------------------------------------------------------

def test_albedo_config_backward_compat_alias():
    # Legacy callers set A_moon; it must mirror onto albedo_const.
    cfg = AlbedoConfig(A_moon=0.3)
    assert cfg.albedo_const == 0.3
    assert cfg.albedo_model == "lambert_facets"


def test_normalize_albedo_mode_accepts_aliases():
    assert normalize_albedo_mode("constant_albedo") == 0
    assert normalize_albedo_mode("albedo_grid") == 1
    assert normalize_albedo_mode("scaled_dn_grid") == 2
    with pytest.raises(ValueError):
        normalize_albedo_mode("not_a_mode")


def test_standalone_albedo_accel_facet_default():
    r_sc = np.asarray([R_MOON + 500_000.0, 0.0, 0.0])
    r_sun = np.asarray([AU, 0.0, 0.0])
    sc = SpacecraftProps(mass_kg=1000.0, area_m2=1.0, cr=1.5)
    a = albedo_accel(r_sc, r_sun, sc, AlbedoConfig())   # defaults to lambert_facets
    assert np.all(np.isfinite(a))
    assert a[0] > 0.0
