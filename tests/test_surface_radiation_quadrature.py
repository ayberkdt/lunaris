"""Visible-disk quadrature for lunar albedo / thermal-IR radiation pressure.

Independent checks (all Moon-fixed, SI):

* thermal IR, uniform exitance M: the exact cannonball result is
  ``a = C A/(m c) * M (R/r)^2`` along the radial direction (``integral over the
  disk of cos(alpha) dOmega = pi sin^2(beta)``); the quadrature is exact for it;
* albedo: a brute-force midpoint integral over the visible cap in SURFACE
  coordinates centred on the sub-satellite point (a different parameterisation
  from the quadrature's solid-angle one);
* the 18 x 36 point-facet sum is badly off at low altitude (why the default
  changed) and converges to the quadrature at high altitude.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from lunaris.common.constants import AU, C_LIGHT, R_EARTH_MEAN, R_MOON, SIGMA_SB, SOLAR_FLUX_1AU
from lunaris.common.type_defs import SpacecraftProps
from lunaris.physics.lunar_albedo import calc_albedo_accel
from lunaris.physics.surface_effects import AlbedoConfig, ThermalConfig, thermal_accel
from lunaris.physics.surface_radiation_quadrature import (
    accel_albedo_quadrature_numba,
    accel_thermal_ir_quadrature_numba,
    gauss_legendre_unit_interval,
)
from lunaris.physics.thermal_ir import (
    THERMAL_MODE_CONSTANT,
    THERMAL_MODE_EQUILIBRIUM,
    build_latlon_facets,
)

AREA, MASS, ALB, EPS, T_K = 1.0, 1000.0, 0.12, 0.95, 250.0
SCALE = AREA / MASS / C_LIGHT
EMPTY = np.zeros(1)
NO_EARTH = (0.0, 0.0, 0.0)


def _unit(lat_deg: float, lon_deg: float) -> np.ndarray:
    la, lo = math.radians(lat_deg), math.radians(lon_deg)
    return np.array([math.cos(la) * math.cos(lo), math.cos(la) * math.sin(lo), math.sin(la)])


def _basis(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    t = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = t - (t @ z) * z
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(z, e1)


def _sun(z: np.ndarray, gamma_deg: float) -> np.ndarray:
    e1, _ = _basis(z)
    g = math.radians(gamma_deg)
    return AU * (math.cos(g) * z + math.sin(g) * e1)


def _albedo_q(r: np.ndarray, sun: np.ndarray, nr: int = 16, na: int = 32, earth=NO_EARTH, eclipse=False) -> np.ndarray:
    nodes, weights = gauss_legendre_unit_interval(nr)
    return np.array(accel_albedo_quadrature_numba(
        *r, *sun, *earth, R_MOON, EMPTY, 0, 0, ALB, 1.0, AREA, MASS,
        SOLAR_FLUX_1AU, AU, C_LIGHT, R_EARTH_MEAN, True, eclipse, nodes, weights, na,
    ))


def _thermal_q(r, sun, mode, nr=16, na=32, earth=NO_EARTH, eclipse=False, floor=0.0) -> np.ndarray:
    nodes, weights = gauss_legendre_unit_interval(nr)
    return np.array(accel_thermal_ir_quadrature_numba(
        *r, *sun, *earth, R_MOON, EMPTY, 0, 0, mode, EPS, 0.12, T_K, 100.0, floor, 1.0,
        AREA, MASS, SOLAR_FLUX_1AU, AU, C_LIGHT, SIGMA_SB, True, R_EARTH_MEAN, eclipse, nodes, weights, na,
    ))


def _albedo_cap_reference(r_vec: np.ndarray, sun: np.ndarray, n: int) -> np.ndarray:
    """Brute-force midpoint rule over the visible cap (theta from the sub-satellite point)."""
    r = float(np.linalg.norm(r_vec))
    z = r_vec / r
    e1, e2 = _basis(z)
    th_h = math.acos(R_MOON / r)
    dth, dps = th_h / n, math.pi / n
    psi = (np.arange(2 * n) + 0.5) * dps
    ring = np.cos(psi)[:, None] * e1 + np.sin(psi)[:, None] * e2
    acc = np.zeros(3)
    for i in range(n):
        th = (i + 0.5) * dth
        nrm = math.cos(th) * z + math.sin(th) * ring
        p = R_MOON * nrm
        s = r_vec - p
        d2 = np.einsum("ij,ij->i", s, s)
        u = s / np.sqrt(d2)[:, None]
        h = sun - p
        h2 = np.einsum("ij,ij->i", h, h)
        mu_s = np.maximum(np.einsum("ij,ij->i", nrm, h) / np.sqrt(h2), 0.0)
        mu_v = np.maximum(np.einsum("ij,ij->i", nrm, u), 0.0)
        w = ALB * SOLAR_FLUX_1AU * AU**2 / h2 * mu_s / math.pi * mu_v * (R_MOON**2 * math.sin(th) * dth * dps) / d2
        acc += (w[:, None] * u).sum(axis=0)
    return SCALE * acc


def _facets(r, sun, n_lat=18, n_lon=36):
    pos, nor, area, _, _ = build_latlon_facets(n_lat, n_lon, radius_m=R_MOON)
    return calc_albedo_accel(r, sun, pos, nor, area, np.full(area.shape[0], ALB),
                             spacecraft_area_m2=AREA, spacecraft_mass_kg=MASS)


def test_gauss_legendre_unit_interval_is_exact_to_degree_2n_minus_1() -> None:
    for n in (1, 4, 16):
        x, w = gauss_legendre_unit_interval(n)
        assert x.shape == w.shape == (n,)
        assert np.all((x > 0.0) & (x < 1.0))
        for k in range(2 * n):
            assert abs(float(w @ x**k) - 1.0 / (k + 1)) < 1e-14
    with pytest.raises(ValueError):
        gauss_legendre_unit_interval(0)


@pytest.mark.parametrize("alt_km", [30.0, 100.0, 300.0, 1000.0])
@pytest.mark.parametrize("site", [(13.7, 41.3), (0.0, 0.0), (-89.9, 10.0)])
def test_thermal_uniform_exitance_matches_analytic_limit(alt_km: float, site: tuple[float, float]) -> None:
    z = _unit(*site)
    r = (R_MOON + alt_km * 1e3) * z
    exact = SCALE * EPS * SIGMA_SB * T_K**4 * (R_MOON / np.linalg.norm(r)) ** 2 * z
    for nr, na in ((16, 32), (4, 8)):
        got = _thermal_q(r, _sun(z, 30.0), THERMAL_MODE_CONSTANT, nr, na)
        np.testing.assert_allclose(got, exact, rtol=0.0, atol=1e-13 * np.linalg.norm(exact))


@pytest.mark.parametrize("alt_km", [30.0, 100.0, 300.0, 1000.0])
@pytest.mark.parametrize("gamma_deg", [0.0, 45.0])
def test_albedo_matches_brute_force_cap_integral(alt_km: float, gamma_deg: float) -> None:
    """Terminator outside the visible disk: the default 16 x 32 rule is converged.

    n=300 midpoint reference: its own error is <= 1e-4 (measured 7.4e-6 at
    n=1000 at 30 km, decreasing as 1/n^2); the tolerance is 2e-4.
    """
    z = _unit(13.7, 41.3)
    r = (R_MOON + alt_km * 1e3) * z
    sun = _sun(z, gamma_deg)
    ref = _albedo_cap_reference(r, sun, 300)
    got = _albedo_q(r, sun)
    assert np.linalg.norm(got - ref) <= 2e-4 * np.linalg.norm(ref)


def test_facet_sum_is_off_at_low_altitude_and_converges_high() -> None:
    """Why the default changed: 18 x 36 point facets at 100 km are 0.39x over a
    facet-cell corner, while at 1000 km they agree with the quadrature to <1 %."""
    z = _unit(0.0, 0.0)  # sub-satellite point on a facet-cell corner
    low = (R_MOON + 100e3) * z
    q_low = _albedo_q(low, _sun(z, 0.0))
    ratio = np.linalg.norm(_facets(low, _sun(z, 0.0))) / np.linalg.norm(q_low)
    assert ratio < 0.5
    high = (R_MOON + 1000e3) * z
    q_high = _albedo_q(high, _sun(z, 0.0))
    assert np.linalg.norm(_facets(high, _sun(z, 0.0)) - q_high) < 1e-2 * np.linalg.norm(q_high)
    assert np.linalg.norm(_facets(high, _sun(z, 0.0), 72, 144) - q_high) < 1e-3 * np.linalg.norm(q_high)


def test_inside_or_on_the_sphere_returns_zero() -> None:
    sun = (AU, 0.0, 0.0)
    for r in ((R_MOON, 0.0, 0.0), (0.5 * R_MOON, 0.0, 0.0)):
        assert np.all(_albedo_q(np.array(r), np.array(sun)) == 0.0)
        assert np.all(_thermal_q(np.array(r), np.array(sun), THERMAL_MODE_CONSTANT) == 0.0)


def test_lunar_eclipse_matches_facet_model_treatment() -> None:
    """Earth between Sun and Moon: albedo vanishes, equilibrium thermal keeps only
    its night floor, constant-temperature thermal is untouched."""
    z = _unit(0.0, 0.0)
    r = (R_MOON + 100e3) * z
    sun = np.array([AU, 0.0, 0.0])
    earth = (3.844e8, 0.0, 0.0)
    assert np.linalg.norm(_albedo_q(r, sun)) > 0.0
    assert np.all(_albedo_q(r, sun, earth=earth, eclipse=True) == 0.0)
    lit = _thermal_q(r, sun, THERMAL_MODE_EQUILIBRIUM)
    dark = _thermal_q(r, sun, THERMAL_MODE_EQUILIBRIUM, earth=earth, eclipse=True)
    floor_only = SCALE * EPS * SIGMA_SB * 100.0**4 * (R_MOON / np.linalg.norm(r)) ** 2 * z
    assert np.linalg.norm(lit) > 10.0 * np.linalg.norm(dark)
    np.testing.assert_allclose(dark, floor_only, rtol=0.0, atol=1e-13 * np.linalg.norm(floor_only))
    const = _thermal_q(r, sun, THERMAL_MODE_CONSTANT)
    np.testing.assert_array_equal(_thermal_q(r, sun, THERMAL_MODE_CONSTANT, earth=earth, eclipse=True), const)


def test_config_surface_integration_validation() -> None:
    assert AlbedoConfig().surface_integration == "visible_disk_quadrature"
    assert ThermalConfig().surface_integration == "visible_disk_quadrature"
    assert AlbedoConfig(surface_integration="Facets").surface_integration == "facets"
    with pytest.raises(ValueError, match="surface_integration"):
        AlbedoConfig(surface_integration="monte")
    with pytest.raises(ValueError, match="quadrature_radial_nodes"):
        ThermalConfig(quadrature_radial_nodes=0)
    with pytest.raises(ValueError, match="quadrature_azimuth_nodes"):
        AlbedoConfig(quadrature_azimuth_nodes=10_000)


def test_thermal_wrapper_rejects_temperature_grid_without_cells() -> None:
    sc = SpacecraftProps(mass_kg=MASS, area_m2=AREA, cr=1.0)
    with pytest.raises(ValueError, match="temperature_grid"):
        thermal_accel((R_MOON + 1e5, 0.0, 0.0), (AU, 0.0, 0.0), sc, ThermalConfig(thermal_mode="temperature_grid"))
