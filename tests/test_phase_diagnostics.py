"""Tests for the phase-drift diagnostic chain (RIC / tau / Gauss-VE prediction).

Validation strategy: every estimator is checked against configurations whose
answer is known analytically (pure time shift, mean-motion offset, pure RIC
displacements) and against an independent numerical experiment (RK4 point-mass
propagation with a known constant tangential acceleration bias) where the
Gauss-VE prediction must reproduce the measured phase drift.
"""

from __future__ import annotations

import numpy as np
import pytest

from lunaris.analysis.ensemble.statistics import ric_basis_from_state
from lunaris.common.constants import MU_MOON, R_MOON
from lunaris.surrogate.st_lrps.evaluation.phase_diagnostics import (
    PHASE_SCENARIO_COLUMNS,
    compute_covariance_shape_similarity,
    compute_phase_corrected_errors,
    compute_phase_diagnostics,
    compute_ric_error_history,
    compute_uq_alignment,
    diagnose_tangential_bias,
    estimate_phase_lag,
    hermite_resample,
    interpolate_covariance_to_times,
    load_uq_covariance_history,
    osculating_sma,
    phase_scenario_metrics,
    predict_phase_drift_from_tangential_bias,
)

RADIUS_M = R_MOON + 100_000.0                       # 100 km circular orbit
N_RAD_S = float(np.sqrt(MU_MOON / RADIUS_M**3))     # mean motion
PERIOD_S = 2.0 * np.pi / N_RAD_S


def circular_states(t_s: np.ndarray, *, n: float = N_RAD_S) -> tuple[np.ndarray, np.ndarray]:
    """Analytic circular orbit of radius RADIUS_M in the xy plane (h along +z)."""
    th = n * t_s
    cos, sin = np.cos(th), np.sin(th)
    zero = np.zeros_like(th)
    r = RADIUS_M * np.stack([cos, sin, zero], axis=1)
    v = RADIUS_M * n * np.stack([-sin, cos, zero], axis=1)
    return r, v


# =============================================================================
# Analytic-answer tests (layer 1: RIC + phase lag + correction)
# =============================================================================


def test_pure_time_shift_recovers_tau() -> None:
    """r_st(t) = r_gt(t + tau0) must yield tau ≈ tau0 and ~zero aligned residual."""
    tau0 = 0.5
    t = np.arange(0.0, 2.0 * PERIOD_S, 10.0)
    r_gt, v_gt = circular_states(t)
    r_st, v_st = circular_states(t + tau0)

    rep = compute_phase_diagnostics(t, r_st, v_st, r_gt, v_gt)

    assert np.allclose(rep.phase.tau_s, tau0, rtol=1e-4)
    assert rep.phase.tau_final_s == pytest.approx(tau0, rel=1e-4)
    # The error is almost purely along-track ...
    assert rep.ric.ms_fraction[1] > 0.999
    # ... and a <=3-parameter time alignment removes essentially all of it.
    assert rep.corrected.aligned_rms_m < 1e-2
    assert rep.corrected.aligned_rms_m < 1e-4 * rep.corrected.raw_rms_m
    assert rep.corrected.phase_explained_fraction > 0.9999
    # Positive tau pushes the last query epochs past the grid end: they must be
    # masked out, not silently extrapolated.
    assert 0.9 < rep.corrected.aligned_valid_fraction < 1.0
    # No acceleration samples were provided.
    assert rep.bias is None and rep.prediction is None


def test_mean_motion_offset_gives_linear_tau_growth() -> None:
    """Same circle, mean motion n + dn: tau(t) ≈ (dn/n)·t, so the linear fit
    must recover the rate dn/n."""
    dn = 1.0e-8
    t = np.arange(0.0, 2.0 * PERIOD_S, 10.0)
    r_gt, v_gt = circular_states(t)
    r_st, _ = circular_states(t, n=N_RAD_S + dn)

    phase = estimate_phase_lag(t, r_st, r_gt, v_gt, fit_degree=1)

    assert phase.tau_fit_coeffs[1] == pytest.approx(dn / N_RAD_S, rel=1e-3)
    assert phase.tau_final_s == pytest.approx(dn / N_RAD_S * t[-1], rel=1e-3)


def test_ric_decomposition_pure_directions() -> None:
    """Purely radial / cross-track displacements land in the right components."""
    t = np.arange(0.0, 0.5 * PERIOD_S, 10.0)
    r_gt, v_gt = circular_states(t)
    r_hat = r_gt / np.linalg.norm(r_gt, axis=1, keepdims=True)

    ric_r = compute_ric_error_history(t, r_gt + 5.0 * r_hat, r_gt, v_gt)
    assert np.allclose(ric_r.e_ric_m[:, 0], 5.0, atol=1e-9)
    assert np.allclose(ric_r.e_ric_m[:, 1:], 0.0, atol=1e-9)
    assert ric_r.ms_fraction[0] == pytest.approx(1.0)

    z_off = np.array([0.0, 0.0, 7.0])
    ric_c = compute_ric_error_history(t, r_gt + z_off, r_gt, v_gt)
    assert np.allclose(ric_c.e_ric_m[:, 2], 7.0, atol=1e-9)
    assert np.allclose(ric_c.e_ric_m[:, :2], 0.0, atol=1e-9)


def test_pointwise_detrended_never_exceeds_raw_rms() -> None:
    rng = np.random.default_rng(42)
    t = np.arange(0.0, 0.5 * PERIOD_S, 10.0)
    r_gt, v_gt = circular_states(t)
    r_st = r_gt + rng.normal(scale=25.0, size=r_gt.shape)

    _, corrected = compute_phase_corrected_errors(t, r_st, r_gt, v_gt)
    assert corrected.pointwise_detrended_rms_m <= corrected.raw_rms_m


def test_hermite_resample_accuracy_and_mask() -> None:
    t = np.arange(0.0, PERIOD_S, 10.0)
    r, v = circular_states(t)
    tq = t + 3.7
    r_exact, _ = circular_states(tq)

    r_q, valid = hermite_resample(t, r, v, tq)
    assert not valid[-1]                      # last query point is past the grid
    assert np.all(valid[:-1])
    assert np.max(np.linalg.norm(r_q[valid] - r_exact[valid], axis=1)) < 1e-3


def test_osculating_sma_on_circular_orbit() -> None:
    t = np.arange(0.0, PERIOD_S, 10.0)
    r, v = circular_states(t)
    sma = osculating_sma(r, v, MU_MOON)
    assert np.allclose(sma, RADIUS_M, rtol=1e-12)


# =============================================================================
# Numerical causal test (layer 2: bias -> Gauss-VE prediction)
# =============================================================================

EPS_TANGENTIAL = 1.0e-7  # [m/s²] = 0.01 mGal, realistic surrogate-bias scale


def _rk4_propagate(y0: np.ndarray, t: np.ndarray, accel) -> np.ndarray:
    """Fixed-step RK4 on y = (r, v) with accel(r, v) -> a. Test-local integrator."""

    def f(y: np.ndarray) -> np.ndarray:
        return np.concatenate([y[3:], accel(y[:3], y[3:])])

    out = np.empty((t.size, 6))
    y = np.asarray(y0, dtype=np.float64)
    out[0] = y
    for i in range(t.size - 1):
        h = t[i + 1] - t[i]
        k1 = f(y)
        k2 = f(y + 0.5 * h * k1)
        k3 = f(y + 0.5 * h * k2)
        k4 = f(y + h * k3)
        y = y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        out[i + 1] = y
    return out


@pytest.fixture(scope="module")
def biased_run() -> dict[str, np.ndarray]:
    """Truth vs truth-plus-constant-tangential-bias, both RK4-propagated."""

    def accel_pm(r: np.ndarray, _v: np.ndarray) -> np.ndarray:
        return -MU_MOON * r / np.linalg.norm(r) ** 3

    def accel_biased(r: np.ndarray, v: np.ndarray) -> np.ndarray:
        return accel_pm(r, v) + EPS_TANGENTIAL * v / np.linalg.norm(v)

    t = np.arange(0.0, 3.0 * PERIOD_S, 5.0)
    r0, v0 = circular_states(np.array([0.0]))
    y0 = np.concatenate([r0[0], v0[0]])

    y_gt = _rk4_propagate(y0, t, accel_pm)
    y_st = _rk4_propagate(y0, t, accel_biased)

    # Model-minus-truth acceleration along the ground-truth trajectory —
    # exactly what the benchmark-side diagnostic would evaluate.
    v_gt = y_gt[:, 3:]
    da = EPS_TANGENTIAL * v_gt / np.linalg.norm(v_gt, axis=1, keepdims=True)
    return {"t": t, "y_gt": y_gt, "y_st": y_st, "da": da}


def test_gauss_ve_prediction_matches_measured_phase_drift(biased_run) -> None:
    """The causal chain: measured tangential bias must *predict* the measured
    phase drift (sign, magnitude, and time history)."""
    t, y_gt, y_st, da = (biased_run[k] for k in ("t", "y_gt", "y_st", "da"))
    rep = compute_phase_diagnostics(
        t, y_st[:, :3], y_st[:, 3:], y_gt[:, :3], y_gt[:, 3:], da_m_s2=da
    )

    # Sign convention: over-acceleration raises the SMA, lowers the mean
    # motion, so the surrogate falls behind (tau < 0) while its SMA grows.
    assert rep.phase.tau_final_s < 0.0
    assert rep.bias is not None and rep.prediction is not None
    assert rep.bias.delta_sma_measured_m[-1] > 0.0

    # The measured tangential bias is recovered exactly (it was constructed).
    assert rep.bias.mean_da_tangential_m_s2 == pytest.approx(EPS_TANGENTIAL, rel=1e-9)

    # Gauss-VE prediction vs measurement: magnitude within 5%, shape r > 0.999.
    assert rep.tau_pred_final_ratio == pytest.approx(1.0, abs=0.05)
    assert rep.tau_pred_measured_corr is not None
    assert rep.tau_pred_measured_corr > 0.999
    assert rep.prediction.delta_sma_pred_m[-1] == pytest.approx(
        rep.bias.delta_sma_measured_m[-1], rel=0.05
    )

    # Error anatomy of a pure phase drift: along-track dominated, and a
    # degree-2 time alignment explains almost all of the raw RMS.
    assert rep.ric.ms_fraction[1] > 0.95
    assert rep.corrected.phase_explained_fraction > 0.9
    assert rep.corrected.aligned_rms_m < 0.15 * rep.corrected.raw_rms_m

    summary = rep.summary()
    assert all(np.isfinite(v) for v in summary.values())
    assert summary["mean_da_tangential_mGal"] == pytest.approx(0.01, rel=1e-6)
    assert "tau_pred_final_s" in summary and "delta_sma_final_m" in summary


def test_orbit_averaged_bias_is_flat_for_constant_bias(biased_run) -> None:
    t, y_gt, da = (biased_run[k] for k in ("t", "y_gt", "da"))
    diag = diagnose_tangential_bias(t, y_gt[:, :3], y_gt[:, 3:], da, mu_m3_s2=MU_MOON)

    assert diag.orbit_mean_valid.any()
    valid_means = diag.orbit_mean_da_tangential_m_s2[diag.orbit_mean_valid]
    assert np.allclose(valid_means, EPS_TANGENTIAL, rtol=1e-6)
    # Without the surrogate state the SMA-drift channel reports zeros.
    assert np.all(diag.delta_sma_measured_m == 0.0)


def test_prediction_is_zero_for_zero_bias(biased_run) -> None:
    t, y_gt = biased_run["t"], biased_run["y_gt"]
    pred = predict_phase_drift_from_tangential_bias(
        t, y_gt[:, :3], y_gt[:, 3:], np.zeros(t.size), mu_m3_s2=MU_MOON
    )
    assert np.all(pred.tau_pred_s == 0.0)
    assert np.all(pred.delta_sma_pred_m == 0.0)


# =============================================================================
# UQ alignment layer (Faz 3)
# =============================================================================


def _ric_covariance(
    r: np.ndarray,
    v: np.ndarray,
    sigmas_ric_m: tuple[float, float, float],
) -> np.ndarray:
    """Inertial (T, 3, 3) covariance with given 1-sigma values along RIC axes."""
    basis = ric_basis_from_state(np.hstack([r, v]))  # rows [r̂; î; ĉ]
    sig2 = np.square(np.asarray(sigmas_ric_m, dtype=np.float64))
    return np.einsum("tji,j,tjl->til", basis, sig2, basis)


def test_uq_alignment_along_track_error_aligns_with_along_track_covariance() -> None:
    t = np.arange(0.0, 0.5 * PERIOD_S, 10.0)
    r_gt, v_gt = circular_states(t)
    cov = _ric_covariance(r_gt, v_gt, (10.0, 500.0, 5.0))  # along-track dominated
    basis = ric_basis_from_state(np.hstack([r_gt, v_gt]))
    i_hat = basis[:, 1, :]

    uq = compute_uq_alignment(t, 100.0 * i_hat, cov, r_gt, v_gt)
    assert uq.mean_alignment == pytest.approx(1.0, abs=1e-9)
    assert np.allclose(uq.q1_ric, [0.0, 1.0, 0.0], atol=1e-9)
    assert uq.mean_q1_along_abs == pytest.approx(1.0, abs=1e-9)
    assert uq.eigvals_pos_m2[0, 0] == pytest.approx(500.0**2)
    assert uq.leading_eigval_growth == pytest.approx(1.0)

    # A purely radial error must NOT align with the along-track eigenvector.
    r_hat = basis[:, 0, :]
    uq_radial = compute_uq_alignment(t, 100.0 * r_hat, cov, r_gt, v_gt)
    assert uq_radial.mean_alignment == pytest.approx(0.0, abs=1e-9)


def test_uq_alignment_masks_zero_error_epochs() -> None:
    t = np.arange(0.0, 0.5 * PERIOD_S, 10.0)
    r_gt, v_gt = circular_states(t)
    cov = _ric_covariance(r_gt, v_gt, (10.0, 500.0, 5.0))
    e = ric_basis_from_state(np.hstack([r_gt, v_gt]))[:, 1, :].copy()
    e[0] = 0.0  # first epoch has zero error -> alignment undefined there

    uq = compute_uq_alignment(t, e, cov, r_gt, v_gt)
    assert np.isnan(uq.alignment[0])
    assert uq.mean_alignment == pytest.approx(1.0, abs=1e-9)


def test_load_uq_covariance_roundtrip(tmp_path) -> None:
    t = np.arange(0.0, 100.0, 10.0)
    mean_state = np.tile(np.arange(6.0), (t.size, 1))
    cov6 = np.tile(np.diag(np.arange(1.0, 7.0)), (t.size, 1, 1))
    path = tmp_path / "uq_covariance.npz"
    np.savez_compressed(path, t_s=t, mean_state=mean_state, cov=cov6)

    t_out, mean_out, cov_pos = load_uq_covariance_history(path)
    assert np.array_equal(t_out, t)
    assert np.array_equal(mean_out, mean_state)
    assert cov_pos.shape == (t.size, 3, 3)
    assert np.array_equal(cov_pos[0], np.diag([1.0, 2.0, 3.0]))

    bad = tmp_path / "bad.npz"
    np.savez_compressed(bad, t_s=t, mean_state=mean_state)  # no "cov" key
    with pytest.raises(ValueError, match="missing keys"):
        load_uq_covariance_history(bad)


def test_interpolate_covariance_linear_and_bounds() -> None:
    t_uq = np.array([0.0, 10.0])
    cov = np.stack([np.eye(3), 3.0 * np.eye(3)])
    out = interpolate_covariance_to_times(t_uq, cov, np.array([5.0]))
    assert np.allclose(out[0], 2.0 * np.eye(3))
    with pytest.raises(ValueError, match="extrapolation"):
        interpolate_covariance_to_times(t_uq, cov, np.array([11.0]))


def test_covariance_shape_similarity_identity_and_rotation() -> None:
    t = np.arange(0.0, 0.5 * PERIOD_S, 10.0)
    r_gt, v_gt = circular_states(t)
    cov_along = _ric_covariance(r_gt, v_gt, (10.0, 500.0, 5.0))
    cov_radial = _ric_covariance(r_gt, v_gt, (500.0, 10.0, 5.0))

    same = compute_covariance_shape_similarity(t, cov_along, cov_along)
    assert same.mean_leading_angle_deg == pytest.approx(0.0, abs=1e-6)
    assert np.allclose(same.trace_ratio, 1.0)

    rotated = compute_covariance_shape_similarity(t, cov_along, cov_radial)
    assert rotated.mean_leading_angle_deg == pytest.approx(90.0, abs=1e-6)


def test_full_report_with_covariance_input(biased_run) -> None:
    t, y_gt, y_st, da = (biased_run[k] for k in ("t", "y_gt", "y_st", "da"))
    cov = _ric_covariance(y_gt[:, :3], y_gt[:, 3:], (10.0, 500.0, 5.0))
    rep = compute_phase_diagnostics(
        t, y_st[:, :3], y_st[:, 3:], y_gt[:, :3], y_gt[:, 3:],
        da_m_s2=da, cov_pos_m2=cov,
    )
    assert rep.uq is not None
    # Phase-drift error is along-track dominated, so it must align with the
    # along-track-dominated reference covariance direction.
    assert rep.uq.mean_alignment > 0.9
    summary = rep.summary()
    for key in ("uq_alignment_mean", "uq_alignment_final",
                "uq_q1_along_abs_mean", "uq_leading_eigval_growth"):
        assert key in summary and np.isfinite(summary[key])


# =============================================================================
# Benchmark-harness adapter (Faz 4)
# =============================================================================


def test_phase_scenario_metrics_adapter(biased_run) -> None:
    t, y_gt, y_st = (biased_run[k] for k in ("t", "y_gt", "y_st"))
    row = phase_scenario_metrics(t, y_st[:, :3], y_gt[:, :3], y_gt[:, 3:])

    assert tuple(row.keys()) == PHASE_SCENARIO_COLUMNS
    assert all(np.isfinite(v) for v in row.values())
    assert row["phase_lag_final_s"] < 0.0            # falls behind (documented sign)
    assert row["phase_lag_slope_s_per_day"] < 0.0
    assert 0.0 <= row["phase_explained_fraction"] <= 1.0
    assert row["phase_explained_fraction"] > 0.9
    # km convention of the scenario CSV.
    raw_rms_km = float(np.sqrt(np.mean(
        np.sum((y_st[:, :3] - y_gt[:, :3]) ** 2, axis=1)))) / 1_000.0
    assert row["phase_corrected_rms_km"] < 0.15 * raw_rms_km


def test_phase_scenario_metrics_degenerate_returns_nan() -> None:
    t = np.arange(0.0, 200.0, 10.0)
    r = np.tile([1.0e6, 0.0, 0.0], (t.size, 1))
    v = np.zeros_like(r)  # zero velocity -> RIC/phase undefined
    row = phase_scenario_metrics(t, r, r, v)
    assert tuple(row.keys()) == PHASE_SCENARIO_COLUMNS
    assert all(np.isnan(v) for v in row.values())


# =============================================================================
# Input validation
# =============================================================================


def test_input_validation_errors() -> None:
    t = np.arange(0.0, 200.0, 10.0)
    r, v = circular_states(t)

    with pytest.raises(ValueError, match="strictly increasing"):
        compute_ric_error_history(t[::-1], r, r, v)
    with pytest.raises(ValueError, match=r"must be \(T, 3\)"):
        compute_ric_error_history(t, r[:, :2], r, v)
    with pytest.raises(ValueError, match="epochs"):
        compute_ric_error_history(t, r[:-1], r, v)
    with pytest.raises(ValueError, match="at least 4"):
        compute_ric_error_history(t[:3], r[:3], r[:3], v[:3])
    with pytest.raises(ValueError, match="fit_degree"):
        estimate_phase_lag(t, r, r, v, fit_degree=5)
    with pytest.raises(ValueError, match="zero-velocity"):
        estimate_phase_lag(t, r, r, np.zeros_like(v))
    with pytest.raises(ValueError, match="non-finite"):
        compute_ric_error_history(t, np.full_like(r, np.nan), r, v)
    with pytest.raises(ValueError, match="non-elliptic"):
        osculating_sma(r, 10.0 * v, MU_MOON)
