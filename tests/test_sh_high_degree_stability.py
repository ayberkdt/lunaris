"""Degree-1800 stability of the SH kernels (external-review verification).

An external review noted the Lunaris kernel is "a good normalized recurrence
but not a full Pines implementation". The relevant NASA-level bar is not the
algorithm's name but measurable behaviour: stability at degree 1800, no pole
degradation, and agreement with independent references. These tests pin that
behaviour with a synthetic degree-1800 field (Kaula-like spectrum), so they
run without the real GRAIL file. The measured baselines on the real
jggrx_1800f model (2026-07-14, 30 km altitude, 12 latitudes):

* runtime kernel vs batch potential kernel:   worst rel 3.6e-14
* analytic accel vs central-diff of V (h=0.5): worst rel 6.3e-08
* exact pole: finite, tangential components exactly zero.

The forward-column recurrence with the stable-m order cut is valid for
N <= ~1900 in float64 (Holmes & Featherstone 2002); at N = 1800 the smallest
margin between the underflow cut and the oscillatory-region boundary is
~+50 orders (at u = cos(lat) ~ 0.4), which these latitudes cover.
"""

from __future__ import annotations

import numpy as np
import pytest

from lunaris.physics.spherical_harmonics import (
    GravityModel,
    sh_potential_accel_fixed,
)

DEG = 1800
MU = 4.902800066e12  # m^3/s^2 (Moon-like; exact value irrelevant to stability)
R_REF = 1738000.0    # m
R_EVAL = R_REF + 30e3  # 30 km altitude: high-degree terms strongest

# Latitudes cover the equator, the underflow-margin band (u ~ 0.3-0.6), and
# BOTH pole approaches. The negative branch guards against a north-only pole
# guard: the kernel must be finite and stable approaching the south pole too.
LATS_DEG = (
    0.0, 30.0, 55.0, 60.0, 65.0, 70.0, 80.0, 89.0, 89.9, 89.999,
    -30.0, -60.0, -80.0, -89.9, -89.999,
)
LON_DEG = 37.0
# Longitude sweep: the longitudinal (m-dependent) recomposition is the most
# pole-sensitive part, and longitude is degenerate at the poles. Include the
# ±180 seam and the four cardinal meridians.
LONS_DEG = (0.0, 45.0, 90.0, 179.999, -179.999, 270.0)


@pytest.fixture(scope="module")
def model_1800() -> GravityModel:
    """Synthetic degree-1800 model with a Kaula-like decaying spectrum."""
    rng = np.random.default_rng(20260714)
    n_idx = np.arange(DEG + 1, dtype=np.float64)
    sigma = np.zeros(DEG + 1)
    sigma[2:] = 1.0e-5 / n_idx[2:] ** 2  # Kaula rule of thumb per coefficient
    c = rng.standard_normal((DEG + 1, DEG + 1)) * sigma[:, None]
    s = rng.standard_normal((DEG + 1, DEG + 1)) * sigma[:, None]
    keep = np.tril(np.ones((DEG + 1, DEG + 1), dtype=bool))
    c[~keep] = 0.0
    s[~keep] = 0.0
    s[:, 0] = 0.0            # S_n0 is identically zero
    c[0, 0] = 1.0            # monopole (structural in the kernel, kept for clarity)
    c[1, :2] = 0.0           # center-of-mass frame: no degree-1 terms
    s[1, :2] = 0.0
    return GravityModel.from_arrays(
        degree_max=DEG, r_ref=R_REF, mu=MU, c_coeffs_full=c, s_coeffs_full=s
    )


def _point(lat_deg: float, lon_deg: float = LON_DEG, r: float = R_EVAL) -> np.ndarray:
    phi = np.deg2rad(lat_deg)
    lam = np.deg2rad(lon_deg)
    return np.array(
        [r * np.cos(phi) * np.cos(lam), r * np.cos(phi) * np.sin(lam), r * np.sin(phi)]
    )


def test_deg1800_two_kernels_agree_and_stay_finite(model_1800: GravityModel) -> None:
    """Runtime accel kernel (stable-m path) vs batch potential kernel (full-m path).

    The two implementations handle the high-order underflow region differently,
    so agreement here shows the stable-m cut drops only physically negligible
    columns at degree 1800. Measured baseline on the real model: 3.6e-14.
    """
    c = np.asarray(model_1800.c_coeffs)
    s = np.asarray(model_1800.s_coeffs)
    for lat in LATS_DEG:
        p = _point(lat)
        a_run = model_1800.accel_fixed(p)
        _, a_batch = sh_potential_accel_fixed(p[None, :], c, s, MU, R_REF, DEG, -1)
        assert np.isfinite(a_run).all(), f"lat={lat}: runtime kernel not finite"
        assert np.isfinite(a_batch).all(), f"lat={lat}: batch kernel not finite"
        rel = np.linalg.norm(a_run - a_batch[0]) / np.linalg.norm(a_batch[0])
        assert rel < 1e-12, f"lat={lat}: kernel disagreement rel={rel:.3e}"


def test_deg1800_acceleration_is_gradient_of_potential(model_1800: GravityModel) -> None:
    """a = +grad(V) via central differences at degree 1800.

    h = 0.5 m on a degree-1800 field gives an FD truncation floor around 1e-7
    relative (measured 6.3e-8 worst on the real model); the 1e-6 bar is the
    skill's standard for f64 potential->acceleration paths.
    """
    c = np.asarray(model_1800.c_coeffs)
    s = np.asarray(model_1800.s_coeffs)
    h = 0.5
    for lat in (0.0, 60.0, 85.0):
        p = _point(lat)
        _, a = sh_potential_accel_fixed(p[None, :], c, s, MU, R_REF, DEG, -1)
        fd = np.empty(3)
        for ax in range(3):
            dp = p.copy()
            dp[ax] += h
            dm = p.copy()
            dm[ax] -= h
            vp, _ = sh_potential_accel_fixed(dp[None, :], c, s, MU, R_REF, DEG, -1)
            vm, _ = sh_potential_accel_fixed(dm[None, :], c, s, MU, R_REF, DEG, -1)
            fd[ax] = (vp[0] - vm[0]) / (2.0 * h)
        rel = np.linalg.norm(a[0] - fd) / np.linalg.norm(a[0])
        assert rel < 1e-6, f"lat={lat}: FD mismatch rel={rel:.3e}"


def test_deg1800_exact_pole_matches_limit(model_1800: GravityModel) -> None:
    """The exact pole must be finite and continuous with the lat -> 90 limit."""
    a_pole = model_1800.accel_fixed(np.array([0.0, 0.0, R_EVAL]))
    assert np.isfinite(a_pole).all()
    # At the exact pole the longitude is degenerate; the kernel's pole branch
    # must return a purely axial acceleration.
    assert a_pole[0] == pytest.approx(0.0, abs=1e-15)
    assert a_pole[1] == pytest.approx(0.0, abs=1e-15)

    a_near = model_1800.accel_fixed(_point(89.9999, lon_deg=0.0))
    # The axial component is continuous through the pole (the small tangential
    # part at 89.9999 deg is real field, not a numerical artifact).
    rel_axial = abs(a_pole[2] - a_near[2]) / abs(a_near[2])
    assert rel_axial < 1e-6, f"pole axial discontinuity rel={rel_axial:.3e}"


def test_deg1800_exact_south_pole_matches_limit(model_1800: GravityModel) -> None:
    """The exact SOUTH pole must be finite and continuous with the lat -> -90 limit.

    Mirror of the north-pole test: a pole guard that only special-cases +90 would
    leave the -90 branch divergent or discontinuous.
    """
    a_pole = model_1800.accel_fixed(np.array([0.0, 0.0, -R_EVAL]))
    assert np.isfinite(a_pole).all()
    assert a_pole[0] == pytest.approx(0.0, abs=1e-15)
    assert a_pole[1] == pytest.approx(0.0, abs=1e-15)

    a_near = model_1800.accel_fixed(_point(-89.9999, lon_deg=0.0))
    rel_axial = abs(a_pole[2] - a_near[2]) / abs(a_near[2])
    assert rel_axial < 1e-6, f"south-pole axial discontinuity rel={rel_axial:.3e}"


def test_deg1800_kernels_agree_across_longitudes(model_1800: GravityModel) -> None:
    """Two-kernel agreement must hold across the full longitude sweep near a pole.

    Longitude is degenerate at the poles, so the m-dependent recomposition is
    exercised at a high latitude (89.999 deg) over every meridian, including the
    ±180 seam, to catch a longitude-specific pole artifact.
    """
    c = np.asarray(model_1800.c_coeffs)
    s = np.asarray(model_1800.s_coeffs)
    for lat in (80.0, 89.999, -89.999):
        for lon in LONS_DEG:
            p = _point(lat, lon_deg=lon)
            a_run = model_1800.accel_fixed(p)
            _, a_batch = sh_potential_accel_fixed(p[None, :], c, s, MU, R_REF, DEG, -1)
            assert np.isfinite(a_run).all(), f"lat={lat} lon={lon}: runtime not finite"
            assert np.isfinite(a_batch).all(), f"lat={lat} lon={lon}: batch not finite"
            rel = np.linalg.norm(a_run - a_batch[0]) / np.linalg.norm(a_batch[0])
            assert rel < 1e-12, f"lat={lat} lon={lon}: disagreement rel={rel:.3e}"


@pytest.mark.requires_pyshtools
def test_deg1800_agrees_with_pyshtools(model_1800: GravityModel) -> None:
    """Independent gold standard: pyshtools (Holmes-Featherstone class scaling)."""
    from validation.independent import pyshtools_reference as pysh

    c = np.asarray(model_1800.c_coeffs)
    s = np.asarray(model_1800.s_coeffs)
    for lat in (0.0, 60.0, 85.0, 89.9):
        p = _point(lat)
        a_ref = pysh.acceleration(p, mu=MU, r_ref=R_REF, c_coeffs=c, s_coeffs=s, degree=DEG)
        a_run = model_1800.accel_fixed(p)
        rel = np.linalg.norm(a_run - a_ref) / np.linalg.norm(a_ref)
        assert rel < 1e-9, f"lat={lat}: pyshtools disagreement rel={rel:.3e}"
