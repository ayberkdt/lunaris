# tests/test_independent_sh_validation.py
"""
Independent external-reference validation for Lunaris gravity and ephemeris.

The in-tree gravity validation reuses Lunaris' own SH code, so a bug there can
hide inside both the model and its "validation" (correlated errors). These tests
exercise the ``validation/independent/`` harnesses, whose reference values come
from deliberately independent code paths:

* ``independent_sh`` — geopotential via ``scipy.special.lpmn`` + an explicit 4π
  normalization, with the acceleration taken as the *numerical gradient* of that
  potential. No analytic Legendre-derivative recurrence is shared with the code
  under test.
* ``naif_ephemeris`` — direct ``spkpos`` queries vs the Lunaris EphemerisManager.
* ``pyshtools_reference`` — optional second SH opinion (gated).

Anchors first (closed forms with no Lunaris involvement), then cross-checks
against the Lunaris implementation.
"""

from __future__ import annotations

import numpy as np
import pytest
from validation.independent import independent_sh as iref

# Moon-like constants (values are arbitrary for the math; SI units).
MU = 4.902800066e12      # m^3/s^2
R_REF = 1738000.0        # m


def _point_at(unit_xyz, radius):
    v = np.asarray(unit_xyz, dtype=np.float64)
    return v / np.linalg.norm(v) * radius


# =============================================================================
# Closed-form anchors (no Lunaris code involved)
# =============================================================================

def test_point_mass_matches_newton_closed_form():
    C = np.zeros((3, 3)); S = np.zeros((3, 3)); C[0, 0] = 1.0
    pos = np.array([2.0e6, 5.0e5, 8.0e5])
    a = iref.acceleration(pos, mu=MU, r_ref=R_REF, c_coeffs=C, s_coeffs=S, degree=0)
    r = np.linalg.norm(pos)
    a_newton = -MU * pos / r**3
    assert np.allclose(a, a_newton, rtol=1e-9, atol=0.0)


def test_j2_potential_matches_closed_form():
    j2 = 2.03e-4
    C = np.zeros((3, 3)); S = np.zeros((3, 3))
    C[0, 0] = 1.0
    C[2, 0] = -j2 / np.sqrt(5.0)   # 4π-normalized C20 = -J2/sqrt(5)
    pos = np.array([1.9e6, -7.0e5, 3.0e5])
    r = np.linalg.norm(pos)
    sin_phi = pos[2] / r
    u_ref = iref.geopotential(pos, mu=MU, r_ref=R_REF, c_coeffs=C, s_coeffs=S, degree=2)
    u_cf = MU / r * (1.0 - j2 * (R_REF / r) ** 2 * 0.5 * (3.0 * sin_phi**2 - 1.0))
    assert abs(u_ref - u_cf) / abs(u_cf) < 1e-12


def test_normalization_factor_known_values():
    # N_00 = 1; N_10 = sqrt(3); N_11 = sqrt(3); N_20 = sqrt(5).
    assert iref._normalization_factor(0, 0) == pytest.approx(1.0)
    assert iref._normalization_factor(1, 0) == pytest.approx(np.sqrt(3.0))
    assert iref._normalization_factor(1, 1) == pytest.approx(np.sqrt(3.0))
    assert iref._normalization_factor(2, 0) == pytest.approx(np.sqrt(5.0))


# =============================================================================
# Cross-check vs the Lunaris analytic SH implementation (synthetic models)
# =============================================================================

def _build_synthetic_model(degree: int, seed: int):
    from lunaris.physics.spherical_harmonics import GravityModel

    rng = np.random.default_rng(seed)
    c = np.zeros((degree + 1, degree + 1)); s = np.zeros((degree + 1, degree + 1))
    c[0, 0] = 1.0
    for n in range(2, degree + 1):
        for m in range(0, n + 1):
            c[n, m] = 1e-4 * rng.standard_normal()
            if m > 0:
                s[n, m] = 1e-4 * rng.standard_normal()
    return GravityModel.from_arrays(degree, R_REF, MU, c, s)


def _spread_points():
    pts = []
    for alt in (100e3, 400e3, 2000e3):
        for vec in ([1, 0.2, 0.3], [0.4, -0.9, 0.2], [0.1, 0.1, 1.0], [-0.7, 0.3, -0.5]):
            pts.append(_point_at(vec, R_REF + alt))
    return np.array(pts)


@pytest.mark.parametrize("degree", [2, 4, 6])
def test_independent_matches_lunaris_analytic(degree):
    model = _build_synthetic_model(degree, seed=degree)
    result = iref.crosscheck_gravity_model(model, _spread_points(), degree=degree)
    # The numerical gradient and the analytic recurrence agree to ~1e-11.
    assert result.passes(rel_tol=1e-8), (
        f"degree={degree} max_rel_error={result.max_rel_error:.3e}"
    )


def test_independent_matches_lunaris_with_nonzero_degree1_terms():
    """The Lunaris kernel's degree loop starts at n=1, so stored degree-1
    coefficients ARE applied when non-zero. The independent reference must
    include them too, or the two sides silently diverge on any non-COM-frame
    field (regression: the reference used to start at degree 2)."""
    from lunaris.physics.spherical_harmonics import GravityModel

    degree = 3
    c = np.zeros((degree + 1, degree + 1)); s = np.zeros((degree + 1, degree + 1))
    c[0, 0] = 1.0
    c[1, 0] = 3.0e-5   # deliberate non-zero degree-1 (non-centre-of-mass frame)
    c[1, 1] = -2.0e-5
    s[1, 1] = 1.5e-5
    c[2, 0] = -9.0e-5
    model = GravityModel.from_arrays(degree, R_REF, MU, c, s)

    worst_rel = 0.0
    for pos in _spread_points():
        ref = iref.acceleration(pos, mu=MU, r_ref=R_REF, c_coeffs=c, s_coeffs=s, degree=degree)
        got = np.asarray(model.accel_fixed(pos, degree=degree), dtype=np.float64)
        worst_rel = max(worst_rel, float(np.linalg.norm(got - ref) / np.linalg.norm(ref)))
    assert worst_rel < 1e-8, f"degree-1 parity broken: max rel error {worst_rel:.3e}"


def test_crosscheck_detects_a_corrupted_coefficient():
    # A genuine independent check must FAIL if the model under test is wrong.
    # The reference is computed from the CLEAN coefficients; only the model under
    # test is corrupted, so the two sides genuinely disagree.
    from lunaris.physics.spherical_harmonics import GravityModel

    clean = _build_synthetic_model(4, seed=99)
    c_clean = np.array(clean.c_coeffs, dtype=np.float64)
    s_clean = np.array(clean.s_coeffs, dtype=np.float64)
    bad_c = c_clean.copy()
    bad_c[3, 1] += 5e-4  # corrupt one coefficient
    corrupted = GravityModel.from_arrays(4, R_REF, MU, bad_c, s_clean)

    worst_rel = 0.0
    for pos in _spread_points():
        ref = iref.acceleration(pos, mu=MU, r_ref=R_REF, c_coeffs=c_clean, s_coeffs=s_clean, degree=4)
        got = np.asarray(corrupted.accel_fixed(pos, degree=4), dtype=np.float64)
        worst_rel = max(worst_rel, float(np.linalg.norm(got - ref) / np.linalg.norm(ref)))
    assert worst_rel > 1e-8


# =============================================================================
# Cross-check vs the REAL lunar gravity model (needs the coefficient file)
# =============================================================================

@pytest.mark.requires_data
def test_independent_matches_real_lunar_model():
    from lunaris.physics.spherical_harmonics import GravityModel
    from lunaris.surrogate.st_lrps.data.dataset_parameters import (
        DEFAULT_LUNAR_GRAVITY_PATH,
    )

    degree = 8
    model = GravityModel.from_file(str(DEFAULT_LUNAR_GRAVITY_PATH), requested_degree=degree)
    result = iref.crosscheck_gravity_model(model, _spread_points(), degree=degree)
    assert result.passes(rel_tol=1e-7), (
        f"real-model max_rel_error={result.max_rel_error:.3e} "
        f"(median={result.median_rel_error:.3e})"
    )


# =============================================================================
# Independent NAIF/SPICE ephemeris cross-check (needs SPICE kernels)
# =============================================================================

@pytest.mark.requires_data
def test_ephemeris_manager_matches_direct_spice():
    from pathlib import Path

    from validation.independent.naif_ephemeris import crosscheck_ephemeris_manager

    from lunaris.common.type_defs import TimeConfig
    from lunaris.core.config import KERNEL_DIR
    from lunaris.physics.ephemeris import EphemerisManager, SpiceBuildConfig

    kernel_dir = Path(KERNEL_DIR)
    # Resolve kernels by content, regardless of the exact on-disk filename suffix
    # (the repo ships e.g. ``naif0012.tls.txt``).
    lsk = next(kernel_dir.glob("naif0012.tls*"))
    spk = next(kernel_dir.glob("de440.bsp"))

    manager = EphemerisManager.from_time_and_spice(
        TimeConfig(start_date="2027-03-02T00:00:00", duration_s=3600.0, output_dt_s=300.0),
        SpiceBuildConfig(kernels=(str(lsk), str(spk))),
        need_moon_fixed_rotation=False,
    )

    results = crosscheck_ephemeris_manager(manager, [str(lsk), str(spk)])
    for target, res in results.items():
        # At grid nodes the manager just returns the stored table -> must match a
        # fresh direct spkpos to sub-metre (km->m float64 round-trip only).
        assert res.max_node_error_m < 1.0, f"{target} node error {res.max_node_error_m} m"
        # Interior Catmull-Rom interpolation over a 300 s step tracks the true
        # SPICE curve tightly for the Earth/Sun geometry.
        assert res.max_interp_error_m < 1.0e3, f"{target} interp error {res.max_interp_error_m} m"


# =============================================================================
# Optional pyshtools second opinion (skipped unless pyshtools is installed)
# =============================================================================

@pytest.mark.requires_pyshtools
def test_pyshtools_agrees_with_scipy_reference():
    from validation.independent import pyshtools_reference as pysh

    model = _build_synthetic_model(6, seed=7)
    c = np.array(model.c_coeffs); s = np.array(model.s_coeffs)
    for pos in _spread_points():
        a_scipy = iref.acceleration(pos, mu=MU, r_ref=R_REF, c_coeffs=c, s_coeffs=s, degree=6)
        a_pysh = pysh.acceleration(pos, mu=MU, r_ref=R_REF, c_coeffs=c, s_coeffs=s, degree=6)
        rel = np.linalg.norm(a_pysh - a_scipy) / np.linalg.norm(a_scipy)
        assert rel < 1e-6, f"pyshtools vs scipy rel error {rel:.3e}"


@pytest.mark.requires_tudatpy
def test_tudatpy_agrees_with_scipy_reference():
    """Independent industry-grade (TU Delft Tudat) SH cross-check.

    Skipped cleanly unless tudatpy is installed. A ``NotImplementedError`` here
    means the tudatpy point-gradient API differs on this release and the adapter
    in ``validation/independent/tudatpy_reference.py`` must be confirmed before
    the cross-check is trusted — it never silently passes on a wrong API.
    """
    from validation.independent import tudatpy_reference as tud

    model = _build_synthetic_model(6, seed=7)
    c = np.array(model.c_coeffs); s = np.array(model.s_coeffs)
    for pos in _spread_points():
        a_scipy = iref.acceleration(pos, mu=MU, r_ref=R_REF, c_coeffs=c, s_coeffs=s, degree=6)
        a_tud = tud.acceleration(pos, mu=MU, r_ref=R_REF, c_coeffs=c, s_coeffs=s, degree=6)
        rel = np.linalg.norm(a_tud - a_scipy) / np.linalg.norm(a_scipy)
        assert rel < 1e-6, f"tudatpy vs scipy rel error {rel:.3e}"
