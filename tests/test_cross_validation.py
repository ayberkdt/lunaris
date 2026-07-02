# tests/test_cross_validation.py
"""Tests for the independent trajectory cross-validation harness.

The harness compares a Lunaris/numeric propagation against a *standalone*
analytic two-body Kepler reference (no Lunaris integrator) and reports RIC-frame
agreement. These tests lock the reference's correctness (closure + match to a
tight numeric integration), the RIC error math, the external-ephemeris loader,
and the scenario driver.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from validation.independent import cross_validation as xv

MU = 4.902800066e12      # m^3/s^2 (Moon-like)
R_REF = 1738000.0        # m


def _numeric_two_body(y0, t_eval, mu):
    def rhs(_t, y):
        r = y[0:3]
        rn = float(np.linalg.norm(r))
        return np.concatenate([y[3:6], -mu * r / rn**3])
    return solve_ivp(rhs, (0.0, float(t_eval[-1])), y0, method="DOP853",
                     rtol=1e-13, atol=1e-9, t_eval=t_eval).y.T


# ---------------------------------------------------------------------------
# Analytic Kepler reference
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ecc", [0.0, 0.1, 0.36, 0.6])
def test_kepler_matches_tight_numeric(ecc):
    r_p = R_REF + 100e3
    a = r_p / (1.0 - ecc)
    v_p = math.sqrt(MU * (2.0 / r_p - 1.0 / a))
    y0 = np.array([r_p, 0.0, 0.0, 0.0, v_p * math.cos(0.5), v_p * math.sin(0.5)])
    period = 2.0 * math.pi * math.sqrt(a**3 / MU)
    t_eval = np.linspace(0.0, 2.3 * period, 120)

    ana = xv.propagate_kepler_analytic(y0, t_eval, MU)
    num = _numeric_two_body(y0, t_eval, MU)
    pos_err = np.linalg.norm(ana[:, 0:3] - num[:, 0:3], axis=1)
    assert float(np.max(pos_err)) < 1e-2  # sub-cm agreement over >2 revolutions


def test_kepler_period_closure():
    r0 = R_REF + 250e3
    vc = math.sqrt(MU / r0)
    y0 = np.array([r0, 0.0, 0.0, 0.0, vc * math.cos(0.3), vc * math.sin(0.3)])
    a = r0  # circular
    period = 2.0 * math.pi * math.sqrt(a**3 / MU)
    end = xv.propagate_kepler_analytic(y0, np.array([period]), MU)[0]
    assert float(np.linalg.norm(end[0:3] - y0[0:3])) < 1e-3
    assert float(np.linalg.norm(end[3:6] - y0[3:6])) < 1e-6


def test_kepler_rejects_hyperbolic():
    r0 = R_REF + 100e3
    v_esc = math.sqrt(2.0 * MU / r0)
    y0 = np.array([r0, 0.0, 0.0, 0.0, 1.2 * v_esc, 0.0])  # hyperbolic
    with pytest.raises(ValueError):
        xv.propagate_kepler_analytic(y0, np.array([100.0]), MU)


# ---------------------------------------------------------------------------
# RIC error report
# ---------------------------------------------------------------------------

def test_ric_error_zero_for_identical():
    y = np.tile(np.array([R_REF + 100e3, 0, 0, 0, 1600.0, 0.0]), (10, 1)).astype(float)
    rep = xv.ric_error_report(y, y)
    assert rep["pos_rms_m"] == 0.0
    assert rep["ric_intrack_max_m"] == 0.0


def test_ric_error_pure_radial_offset():
    base = np.array([R_REF + 100e3, 0.0, 0.0, 0.0, 1600.0, 0.0])
    ref = np.tile(base, (5, 1)).astype(float)
    test = ref.copy()
    test[:, 0] += 12.0  # shift +x == radial here (r along +x)
    rep = xv.ric_error_report(test, ref)
    assert rep["ric_radial_rms_m"] == pytest.approx(12.0, abs=1e-9)
    assert rep["ric_intrack_rms_m"] == pytest.approx(0.0, abs=1e-9)
    assert rep["ric_crosstrack_rms_m"] == pytest.approx(0.0, abs=1e-9)


def test_ric_error_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        xv.ric_error_report(np.zeros((4, 6)), np.zeros((5, 6)))


# ---------------------------------------------------------------------------
# External (GMAT) ephemeris loader
# ---------------------------------------------------------------------------

def test_load_external_ephemeris_km_to_m(tmp_path):
    # Synthetic GMAT-style ReportFile: a header line + 3 data rows in km / km/s.
    f = tmp_path / "gmat_report.txt"
    f.write_text(
        "Sat.ElapsedSecs Sat.X Sat.Y Sat.Z Sat.VX Sat.VY Sat.VZ\n"
        "0.0 1838.0 0.0 0.0 0.0 1.6 0.1\n"
        "60.0 1837.0 96.0 6.0 -0.05 1.59 0.099\n"
        "120.0 1834.0 192.0 12.0 -0.10 1.58 0.098\n",
        encoding="utf-8",
    )
    t, y = xv.load_external_ephemeris(f)  # default km -> m
    assert t.shape == (3,)
    assert y.shape == (3, 6)
    assert t[0] == 0.0 and t[1] == 60.0           # seconds from start
    assert y[0, 0] == pytest.approx(1838.0 * 1000.0)  # km -> m
    assert y[1, 4] == pytest.approx(1.59 * 1000.0)    # km/s -> m/s


def test_load_external_ephemeris_meters_passthrough(tmp_path):
    f = tmp_path / "ephem_m.csv"
    f.write_text("0,1.0e6,0,0,0,1600,0\n1,1.0e6,1.0e4,0,0,1600,0\n", encoding="utf-8")
    _t, y = xv.load_external_ephemeris(f, length_unit_m=1.0)
    assert y[0, 0] == pytest.approx(1.0e6)


def test_load_external_ephemeris_a1modjulian_days_to_seconds(tmp_path):
    f = tmp_path / "gmat_a1mjd_report.txt"
    f.write_text(
        "Sat.A1ModJulian Sat.X Sat.Y Sat.Z Sat.VX Sat.VY Sat.VZ\n"
        "30000.000000000000 1838.0 0.0 0.0 0.0 1.6 0.1\n"
        "30000.000694444445 1837.0 96.0 6.0 -0.05 1.59 0.099\n"
        "30000.001388888890 1834.0 192.0 12.0 -0.10 1.58 0.098\n",
        encoding="utf-8",
    )
    t, _y = xv.load_external_ephemeris(f)
    np.testing.assert_allclose(t, [0.0, 60.0, 120.0], rtol=0.0, atol=1.0e-6)


def test_load_external_ephemeris_time_unit_override(tmp_path):
    f = tmp_path / "ephem_minutes.csv"
    f.write_text("0,1,0,0,0,1,0\n1,1,1,0,0,1,0\n", encoding="utf-8")
    t, _y = xv.load_external_ephemeris(f, time_unit_s=60.0)
    np.testing.assert_allclose(t, [0.0, 60.0], rtol=0.0, atol=0.0)


def test_interpolate_states_by_time_aligns_reference_rows():
    t_src = np.asarray([0.0, 60.0, 120.0], dtype=np.float64)
    y_src = np.column_stack([t_src + j for j in range(6)]).astype(np.float64)

    out = xv._interpolate_states_by_time(t_src, y_src, np.asarray([30.0, 90.0]))

    expected = np.column_stack([np.asarray([30.0, 90.0]) + j for j in range(6)])
    np.testing.assert_allclose(out, expected, rtol=0.0, atol=0.0)


def test_interpolate_states_by_time_rejects_out_of_range_targets():
    t_src = np.asarray([0.0, 60.0], dtype=np.float64)
    y_src = np.zeros((2, 6), dtype=np.float64)

    with pytest.raises(ValueError, match="outside external ephemeris range"):
        xv._interpolate_states_by_time(t_src, y_src, np.asarray([120.0]))


# ---------------------------------------------------------------------------
# Scenario driver
# ---------------------------------------------------------------------------

def test_cross_validate_integrator_circular_is_accurate():
    scn = xv.Scenario("llo", alt_km=100.0, ecc=0.0, inc_deg=45.0, duration_s=3 * 3600.0, dt_s=60.0)
    rep = xv.cross_validate_integrator(scn, MU, R_REF, method="DOP853", rtol=1e-10, atol=1e-12)
    assert rep["pos_rms_m"] < 1.0          # DOP853 vs analytic Kepler: well under a meter
    assert rep["atol_mode"] == "scalar"
    assert rep["n_samples"] > 10


def test_cross_validate_integrator_reports_period_from_semi_major_axis():
    """alt_km is the periapsis altitude; the reported period must come from the
    semi-major axis a = r_p / (1 - e), not from r_p (regression: eccentric
    scenarios used to report a too-short period)."""
    scn = xv.Scenario("ell", alt_km=100.0, ecc=0.36, inc_deg=30.0, duration_s=3600.0, dt_s=60.0)
    rep = xv.cross_validate_integrator(scn, MU, R_REF)
    r_p = R_REF + 100e3
    a = r_p / (1.0 - 0.36)
    expected = 2.0 * math.pi * math.sqrt(a**3 / MU)
    assert rep["period_s"] == pytest.approx(expected, rel=1e-12)
    # And the circular case is unchanged (a == r_p).
    scn_c = xv.Scenario("circ", alt_km=100.0, ecc=0.0, inc_deg=0.0, duration_s=3600.0, dt_s=60.0)
    rep_c = xv.cross_validate_integrator(scn_c, MU, R_REF)
    expected_c = 2.0 * math.pi * math.sqrt(r_p**3 / MU)
    assert rep_c["period_s"] == pytest.approx(expected_c, rel=1e-12)


def test_cross_validate_integrator_vector_atol_mode():
    scn = xv.Scenario("llo", alt_km=100.0, ecc=0.0, inc_deg=45.0, duration_s=2 * 3600.0, dt_s=60.0)
    rep = xv.cross_validate_integrator(scn, MU, R_REF, atol_pos=1e-6, atol_vel=1e-9)
    assert rep["atol_mode"] == "vector"
    assert rep["pos_rms_m"] < 1.0
