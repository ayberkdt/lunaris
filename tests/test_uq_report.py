"""UQ report: covariance validity, RIC decomposition, provenance, linear check.

Guards the scientific core of the UQ MVP (docs/UQ_COVARIANCE.md): the sample
covariance must be symmetric PSD, the RIC decomposition must match the
benchmark convention, zero input dispersion must yield zero covariance, the
report must be seed-reproducible with a recorded content hash, and the Monte
Carlo covariance must agree with linear (STM) propagation in the linear regime.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from lunaris.analysis.ensemble.linear_check import (
    compare_covariance_histories,
    finite_difference_stm,
    finite_difference_stm_with_quality,
    linear_covariance_history,
    resolve_fd_steps,
    stm_quality,
)
from lunaris.analysis.ensemble.statistics import (
    compute_ensemble_statistics,
    compute_ric_uncertainty,
    ric_basis_from_state,
)
from lunaris.analysis.ensemble.uq_report import build_uq_report, ensemble_content_sha256
from lunaris.common.batch_defs import BatchPropagationResult

MU_MOON = 4.9048695e12  # [m^3/s^2]
R_ORBIT = 1.9e6  # [m] ~163 km altitude circular orbit


def _circular_state() -> np.ndarray:
    v = np.sqrt(MU_MOON / R_ORBIT)
    return np.array([R_ORBIT, 0.0, 0.0, 0.0, v, 0.0], dtype=np.float64)


def _make_result(
    seed: int,
    *,
    n_samples: int = 64,
    n_epochs: int = 25,
    sigma_r_m: float = 100.0,
    sigma_v_ms: float = 0.1,
) -> BatchPropagationResult:
    """Synthetic ensemble: dispersed initial states under exact linear drift.

    The per-sample trajectory is ``r(t) = r0 + v0 t, v(t) = v0`` — not orbital
    dynamics, but a deterministic map that exercises every statistics/report
    path with a closed-form covariance.
    """
    rng = np.random.default_rng(seed)
    y0 = _circular_state()
    dy = np.concatenate(
        [
            rng.standard_normal((n_samples, 3)) * sigma_r_m,
            rng.standard_normal((n_samples, 3)) * sigma_v_ms,
        ],
        axis=1,
    )
    y_init = y0[None, :] + dy  # (N, 6)
    t = np.linspace(0.0, 600.0, n_epochs)  # (T,)
    Y = np.empty((n_epochs, n_samples, 6), dtype=np.float64)
    for k, tk in enumerate(t):
        Y[k, :, :3] = y_init[:, :3] + y_init[:, 3:] * tk
        Y[k, :, 3:] = y_init[:, 3:]
    return BatchPropagationResult(
        t=t,
        Y=Y,
        sc_samples=np.tile([1000.0, 2.0, 2.2, 1.3], (n_samples, 1)),
        impact_mask=np.zeros(n_samples),
        t_impact=np.full(n_samples, np.nan),
        diagnostics={"seed": seed, "backend": "synthetic_test"},
    )


# ---------------------------------------------------------------------------
# Covariance validity
# ---------------------------------------------------------------------------


def test_covariance_is_symmetric_psd_at_every_epoch():
    ens = compute_ensemble_statistics(_make_result(11))
    for k in range(ens.cov.shape[0]):
        P = ens.cov[k]
        assert np.allclose(P, P.T, rtol=0.0, atol=1e-9 * max(1.0, np.abs(P).max()))
        eigvals = np.linalg.eigvalsh(0.5 * (P + P.T))
        assert eigvals.min() >= -1e-6 * max(1.0, eigvals.max())


def test_zero_dispersion_yields_zero_covariance():
    result = _make_result(3, sigma_r_m=0.0, sigma_v_ms=0.0)
    ens = compute_ensemble_statistics(result)
    scale = float(np.max(np.abs(ens.mean))) ** 2
    assert float(np.max(np.abs(ens.cov))) <= 1e-18 * scale


# ---------------------------------------------------------------------------
# RIC decomposition
# ---------------------------------------------------------------------------


def test_ric_basis_hand_case():
    # r along +x, v along +y  =>  R=+x, C=+z (r x v), I=+y.
    state = np.array([[R_ORBIT, 0, 0, 0, 1000.0, 0]], dtype=np.float64)
    B = ric_basis_from_state(state)[0]
    np.testing.assert_allclose(B[0], [1, 0, 0], atol=1e-12)  # radial
    np.testing.assert_allclose(B[1], [0, 1, 0], atol=1e-12)  # along
    np.testing.assert_allclose(B[2], [0, 0, 1], atol=1e-12)  # cross


def test_ric_basis_rejects_degenerate_angular_momentum():
    # Purely radial motion has no orbit normal, so the RIC frame is undefined.
    state = np.array([[R_ORBIT, 0, 0, 1000.0, 0, 0]], dtype=np.float64)
    with pytest.raises(ValueError, match="RIC frame is undefined"):
        ric_basis_from_state(state)


def test_ric_covariance_diagonal_hand_case():
    ens = compute_ensemble_statistics(_make_result(5))
    # Overwrite with a controlled case: mean state along axes, diagonal P_pos.
    ens.mean[:] = np.array([R_ORBIT, 0, 0, 0, 1000.0, 0])
    ens.cov[:] = 0.0
    ens.cov[:, 0, 0] = 4.0  # sigma_x^2 -> radial
    ens.cov[:, 1, 1] = 9.0  # sigma_y^2 -> along
    ens.cov[:, 2, 2] = 16.0  # sigma_z^2 -> cross
    ric = compute_ric_uncertainty(ens)
    np.testing.assert_allclose(ric.sigma_ric_m[:, 0], 2.0, atol=1e-12)
    np.testing.assert_allclose(ric.sigma_ric_m[:, 1], 3.0, atol=1e-12)
    np.testing.assert_allclose(ric.sigma_ric_m[:, 2], 4.0, atol=1e-12)


def test_ric_matches_benchmark_decomposition():
    # The analysis-layer RIC projection must equal the benchmark convention.
    from lunaris.surrogate.st_lrps.evaluation._gravity_benchmark.types import (
        decompose_vector_ric,
    )

    rng = np.random.default_rng(7)
    r = rng.normal(scale=2e6, size=(20, 3))
    v = rng.normal(scale=1e3, size=(20, 3))
    vec = rng.normal(scale=100.0, size=(20, 3))
    B = ric_basis_from_state(np.concatenate([r, v], axis=1))
    ours = np.einsum("tij,tj->ti", B, vec)
    theirs = decompose_vector_ric(vec, r, v)
    np.testing.assert_allclose(ours, theirs, rtol=1e-12, atol=1e-9)


# ---------------------------------------------------------------------------
# Report bundle + provenance
# ---------------------------------------------------------------------------


def test_uq_report_same_seed_identical_content_hash(tmp_path):
    m1 = build_uq_report(_make_result(42), tmp_path / "a", make_figures=False)
    m2 = build_uq_report(_make_result(42), tmp_path / "b", make_figures=False)
    m3 = build_uq_report(_make_result(43), tmp_path / "c", make_figures=False)
    assert m1["covariance_content_sha256"] == m2["covariance_content_sha256"]
    assert m1["covariance_content_sha256"] != m3["covariance_content_sha256"]


def test_uq_manifest_completeness_and_files(tmp_path):
    out = tmp_path / "report"
    result = _make_result(42)
    manifest = build_uq_report(
        result,
        out,
        run_config={"seed": 42, "sampling_method": "random", "n_samples": 64},
        make_figures=False,
    )
    on_disk = json.loads((out / "uq_manifest.json").read_text(encoding="utf-8"))
    assert on_disk["covariance_content_sha256"] == manifest["covariance_content_sha256"]
    assert on_disk["covariance_estimator_kind"] == "sample_covariance_estimator"
    assert on_disk["sampling_method"] == "random"
    assert "Sample covariance estimator" in on_disk["covariance_definition"]
    assert "orbit-determination" in on_disk["covariance_definition"]
    assert on_disk["schema_version"] == 2
    assert on_disk["ensemble"]["n_samples"] == 64
    assert on_disk["run_config"]["seed"] == 42
    assert on_disk["run_config_hash"]
    assert on_disk["archive_metadata"]["backend"] == "synthetic_test"
    assert "repo" in on_disk and "environment" in on_disk
    for name in ("uq_covariance.npz", "uq_summary.csv"):
        assert (out / name).is_file()
        assert on_disk["files"][name]["sha256"]

    # NPZ arrays hash back to the manifest content hash (round-trip integrity).
    with np.load(out / "uq_covariance.npz") as data:
        arrays = {name: data[name] for name in data.files}
    assert ensemble_content_sha256(arrays) == on_disk["covariance_content_sha256"]

    # Summary CSV has the promised columns and one row per epoch.
    header = (out / "uq_summary.csv").read_text(encoding="utf-8").splitlines()
    assert header[0].split(",")[:6] == [
        "t_s",
        "sigma_pos_m",
        "sigma_vel_ms",
        "sigma_radial_m",
        "sigma_along_m",
        "sigma_cross_m",
    ]
    assert len(header) == 1 + int(result.t.shape[0])


def test_uq_manifest_labels_qmc_covariance_as_empirical(tmp_path):
    manifest = build_uq_report(
        _make_result(42),
        tmp_path / "report_qmc",
        run_config={"sampling_method": "sobol", "n_samples": 64},
        make_figures=False,
    )

    assert manifest["sampling_method"] == "sobol"
    assert manifest["covariance_estimator_kind"] == "empirical_ensemble_covariance"
    assert "non-IID sobol" in manifest["covariance_definition"]
    assert "not make this an unbiased" in manifest["covariance_definition"]


def test_uq_report_writes_figures(tmp_path):
    pytest.importorskip("matplotlib")
    out = tmp_path / "report"
    manifest = build_uq_report(_make_result(1, n_samples=16, n_epochs=10), out)
    assert manifest["figures_skipped_reason"] is None
    for rel in (
        "figures/position_covariance_history.png",
        "figures/ric_sigma_history.png",
        "figures/covariance_eigenvalues.png",
        "figures/covariance_tubes_3d.png",
        "figures/altitude_envelope.png",
    ):
        assert (out / rel).is_file(), rel
        assert manifest["files"][rel]["sha256"]


def test_uq_report_records_plot_build_importerror(tmp_path, monkeypatch):
    import lunaris.analysis.ensemble.plotting as plotting

    def _raise_import_error(_stats):
        raise ImportError("synthetic matplotlib unavailable")

    monkeypatch.setattr(plotting, "plot_position_covariance_history", _raise_import_error)

    manifest = build_uq_report(_make_result(2, n_samples=16, n_epochs=10), tmp_path / "report")
    assert "matplotlib unavailable" in str(manifest["figures_skipped_reason"])
    assert not any(name.startswith("figures/") for name in manifest["files"])


# ---------------------------------------------------------------------------
# Linear (STM) cross-check
# ---------------------------------------------------------------------------


def _propagate_drift(y0: np.ndarray) -> np.ndarray:
    """Exact linear drift dynamics on a fixed grid: r += v t."""
    t = np.linspace(0.0, 600.0, 25)
    out = np.empty((t.size, 6), dtype=np.float64)
    out[:, :3] = y0[None, :3] + np.outer(t, y0[3:])
    out[:, 3:] = y0[None, 3:]
    return out


def test_fd_stm_exact_for_linear_dynamics():
    Phi = finite_difference_stm(_propagate_drift, _circular_state())
    t = np.linspace(0.0, 600.0, 25)
    for k, tk in enumerate(t):
        expected = np.eye(6)
        expected[:3, 3:] = tk * np.eye(3)
        # atol budget: differencing ~1.9e6-m positions leaves ~4e-10 m round-off,
        # amplified by the 1e-3 m/s velocity step to ~1e-7 in the STM entries.
        np.testing.assert_allclose(Phi[k], expected, rtol=0.0, atol=1e-6)


def test_relative_fd_steps_are_blockwise_and_rotation_neutral():
    y0 = np.array([2.0e6, 0.0, 0.0, 0.0, 1.6e3, 0.0])
    steps = resolve_fd_steps(y0, eps_mode="relative", rel_step=1.0e-6)
    np.testing.assert_array_equal(steps[:3], np.full(3, 2.0))
    np.testing.assert_allclose(steps[3:], np.full(3, 1.6e-3), rtol=1e-15, atol=0.0)


def test_stm_quality_requires_dimensionless_scales_and_handles_not_applicable():
    Phi = finite_difference_stm(_propagate_drift, _circular_state())
    not_applicable = stm_quality(Phi, symplectic_applicable=False)
    assert not_applicable["symplecticity_status"] == "not_applicable"
    assert not_applicable["symplecticity_error"] is None
    with pytest.raises(ValueError, match="state_scales"):
        stm_quality(Phi, symplectic_applicable=True)


def test_two_body_fd_stm_quality_and_eps_halving_are_reported():
    y0 = _circular_state()
    Phi, quality = finite_difference_stm_with_quality(
        _propagate_two_body,
        y0,
        eps_mode="relative",
        rel_step=1.0e-6,
        symplectic_applicable=True,
        check_eps_halving=True,
    )
    assert Phi.shape[1:] == (6, 6)
    assert quality["symplecticity_status"] == "evaluated_dimensionless"
    assert quality["symplecticity_error"] < 1.0e-4
    assert quality["det_deviation"] < 1.0e-4
    assert quality["eps_halving_rel_diff"] < 1.0e-4


def test_covariance_comparison_fans_in_stm_quality():
    P = np.repeat(np.eye(6)[None, :, :], 2, axis=0)
    report = compare_covariance_histories(
        P,
        P,
        stm_quality_metrics={"symplecticity_status": "not_applicable"},
    )
    assert report["stm_quality"]["symplecticity_status"] == "not_applicable"


def test_linear_vs_batch_agree_for_linear_dynamics():
    sigma_r, sigma_v = 100.0, 0.1
    P0 = np.diag([sigma_r**2] * 3 + [sigma_v**2] * 3)
    P_lin = linear_covariance_history(_propagate_drift, _circular_state(), P0)
    ens = compute_ensemble_statistics(_make_result(21, n_samples=4096))
    report = compare_covariance_histories(P_lin, ens.cov)
    # Linear dynamics: only sampling error separates the two (N=4096 => ~few %).
    assert report["max_frobenius_rel_diff"] < 0.15
    lo, hi = report["pos_eig_ratio_range"]
    assert 0.8 < lo <= hi < 1.2


def _propagate_two_body(y0: np.ndarray) -> np.ndarray:
    """Vectorized fixed-step RK4 point-mass propagator on a fixed grid."""
    y0 = np.atleast_2d(np.asarray(y0, dtype=np.float64))  # (B, 6)
    dt, n_steps, record_every = 10.0, 400, 20

    def rhs(y: np.ndarray) -> np.ndarray:
        r = y[:, :3]
        rn = np.linalg.norm(r, axis=1, keepdims=True)
        out = np.empty_like(y)
        out[:, :3] = y[:, 3:]
        out[:, 3:] = -MU_MOON * r / rn**3
        return out

    states = [y0.copy()]
    y = y0.copy()
    for step in range(1, n_steps + 1):
        k1 = rhs(y)
        k2 = rhs(y + 0.5 * dt * k1)
        k3 = rhs(y + 0.5 * dt * k2)
        k4 = rhs(y + dt * k3)
        y = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        if step % record_every == 0:
            states.append(y.copy())
    out = np.stack(states, axis=0)  # (T, B, 6)
    return out[:, 0, :] if out.shape[1] == 1 else out


def test_linear_vs_batch_two_body():
    """Ensemble covariance matches STM propagation on a short point-mass arc."""
    y0 = _circular_state()
    sigma_r, sigma_v = 50.0, 0.05
    P0 = np.diag([sigma_r**2] * 3 + [sigma_v**2] * 3)
    P_lin = linear_covariance_history(_propagate_two_body, y0, P0)

    rng = np.random.default_rng(99)
    n_samples = 2048
    dy = np.concatenate(
        [
            rng.standard_normal((n_samples, 3)) * sigma_r,
            rng.standard_normal((n_samples, 3)) * sigma_v,
        ],
        axis=1,
    )
    Y_batch = _propagate_two_body(y0[None, :] + dy)  # (T, N, 6)
    T = Y_batch.shape[0]
    P_ens = np.empty((T, 6, 6), dtype=np.float64)
    for k in range(T):
        P_ens[k] = np.cov(Y_batch[k].T, ddof=1)

    report = compare_covariance_histories(P_lin, P_ens)
    # Small dispersion + short arc: near-linear regime, budget = sampling error
    # (N=2048 => eigenvalue s.e. ~3%) plus mild nonlinearity growth.
    assert report["max_frobenius_rel_diff"] < 0.25
    lo, hi = report["pos_eig_ratio_range"]
    assert 0.7 < lo <= hi < 1.3
