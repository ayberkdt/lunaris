"""Evaluation leakage scope and streaming-statistics regressions (ST-LRPS eval CLI)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")
torch = pytest.importorskip("torch")

from lunaris.surrogate.st_lrps.data.dataset_parameters import MU_MOON_SI, R_MOON_SI  # noqa: E402
from lunaris.surrogate.st_lrps.evaluation.cli import (  # noqa: E402
    _StreamingMetrics,
    evaluate,
)

_CONTENT_SHA = "ab" * 32


def _make_h5(path: Path, n: int = 120, *, content_sha256: str | None = _CONTENT_SHA) -> None:
    rng = np.random.default_rng(11)
    r = R_MOON_SI + rng.uniform(200e3, 500e3, n)
    theta = rng.uniform(0.1, np.pi - 0.1, n)
    phi = rng.uniform(0, 2 * np.pi, n)
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    # Residual-scale labels (the fixture model predicts a residual field).
    u = 1e2 * np.cos(theta)
    a = 1e-4 * np.stack([np.cos(phi), np.sin(phi), np.cos(theta)], axis=1)
    data = np.column_stack([x, y, z, u, a]).astype(np.float64)
    with h5py.File(str(path), "w") as hf:
        hf.create_dataset("data", data=data)
        hf.attrs["unit_system"] = "SI"
        hf.attrs["mu_si"] = float(MU_MOON_SI)
        hf.attrs["r_ref_m"] = float(R_MOON_SI)
        hf.attrs["central_body"] = "moon"
        hf.attrs["target_mode"] = "residual"
        hf.attrs["degree_min"] = 20
        hf.attrs["degree_max"] = 60
        hf.attrs["requested_degree"] = 60
        hf.attrs["alt_min_km"] = 200.0
        hf.attrs["alt_max_km"] = 500.0
        hf.attrs["columns"] = "[x,y,z,U,ax,ay,az]"
        if content_sha256 is not None:
            hf.attrs["content_sha256"] = content_sha256


def _make_run_dir(root: Path) -> Path:
    from st_lrps_contract_test_utils import make_contract_run

    return Path(make_contract_run(root, alt_min_km=100.0, alt_max_km=1000.0)["run_dir"])


def _write_split(run_dir: Path, *, policy: str, dataset_sha: str | None, train: np.ndarray | None) -> None:
    prov = run_dir / "provenance"
    prov.mkdir(parents=True, exist_ok=True)
    manifest = {"split_policy": policy, "dataset_content_sha256": dataset_sha}
    (prov / "split_manifest.json").write_text(json.dumps(manifest))
    if train is not None:
        np.savez(prov / "split_indices.npz", train=train, val=np.array([], dtype=np.int64))


def _evaluate(run_dir: Path, data: Path, out: Path, **kw):
    return evaluate(
        model_dir=run_dir, data_path=data, out_dir=out, device=torch.device("cpu"),
        batch_size=32, a_sign=1.0, r_ref_m=R_MOON_SI, alt_bin_km=5.0, **kw,
    )


# ----------------------------------------------------------------------------
# Streaming statistics
# ----------------------------------------------------------------------------

def test_streaming_linf_propagates_nan() -> None:
    sm = _StreamingMetrics(n_alt_bins=4, alt_min_km=0.0, alt_max_km=100.0)
    x = np.array([[R_MOON_SI + 10e3, 0.0, 0.0], [R_MOON_SI + 20e3, 0.0, 0.0]])
    a_true = np.ones((2, 3))
    a_pred = np.array([[1.0, 1.0, 1.0], [np.nan, 1.0, 1.0]])
    sm.update(x, a_true, a_pred, np.ones(2), np.array([1.0, np.nan]), R_MOON_SI)
    # A later clean batch must not wash the NaN out again.
    sm.update(x, a_true, a_true + 0.5, np.ones(2), np.full(2, 3.0), R_MOON_SI)
    res = sm.finalize()
    assert math.isnan(res["max_abs_u"])
    assert math.isnan(res["max_abs_a_vec"])
    assert math.isnan(res["max_abs_a_mag"])


def test_streaming_linf_keeps_inf_and_finite_max() -> None:
    sm = _StreamingMetrics(n_alt_bins=4, alt_min_km=0.0, alt_max_km=100.0)
    x = np.array([[R_MOON_SI + 10e3, 0.0, 0.0]])
    sm.update(x, np.ones((1, 3)), np.ones((1, 3)), np.ones(1), np.array([4.0]), R_MOON_SI)
    sm.update(x, np.ones((1, 3)), np.ones((1, 3)), np.ones(1), np.array([2.0]), R_MOON_SI)
    assert sm.finalize()["max_abs_u"] == pytest.approx(3.0)
    sm.update(x, np.ones((1, 3)), np.ones((1, 3)), np.ones(1), np.array([np.inf]), R_MOON_SI)
    assert sm.finalize()["max_abs_u"] == math.inf


def test_streaming_altitude_under_and_overflow_are_counted() -> None:
    sm = _StreamingMetrics(n_alt_bins=4, alt_min_km=100.0, alt_max_km=200.0)
    alts_km = np.array([50.0, 150.0, 199.0, 250.0, 300.0])
    x = np.column_stack([R_MOON_SI + alts_km * 1e3, np.zeros(5), np.zeros(5)])
    sm.update(x, np.ones((5, 3)), np.ones((5, 3)), np.ones(5), np.ones(5), R_MOON_SI)
    res = sm.finalize()
    assert res["alt_underflow_count"] == 1
    assert res["alt_overflow_count"] == 2
    assert res["alt_bin_range_km"] == [100.0, 200.0]


def test_streaming_bins_span_dataset_altitude_and_ratio_key_is_distinct(tmp_path: Path) -> None:
    data = tmp_path / "data.h5"
    _make_h5(data, content_sha256=None)
    run_dir = _make_run_dir(tmp_path / "run")
    metrics = _evaluate(run_dir, data, tmp_path / "out", streaming=True)
    # Data are 200-500 km; the old range was [0, 20 * alt_bin_km] = [0, 100] km.
    assert metrics["altitude_bin_range_km"] == [200.0, 500.0]
    assert metrics["altitude_underflow_count"] == 0
    assert metrics["altitude_overflow_count"] == 0
    for block in ("U", "|a|", "residual_vector_metrics", "residual_magnitude_metrics"):
        assert "rel_ratio_of_sums_pct" in metrics[block]
        assert "rel_mean_pct" not in metrics[block], block


# ----------------------------------------------------------------------------
# Evaluation scope / leakage
# ----------------------------------------------------------------------------

def test_eval_on_training_file_excludes_train_rows_automatically(tmp_path: Path) -> None:
    n = 120
    data = tmp_path / "data.h5"
    _make_h5(data, n=n)
    run_dir = _make_run_dir(tmp_path / "run")
    train = np.arange(0, n, 2, dtype=np.int64)
    _write_split(run_dir, policy="seeded_random", dataset_sha=_CONTENT_SHA, train=train)
    metrics = _evaluate(run_dir, data, tmp_path / "out", streaming=True)
    assert metrics["n_samples"] == n - train.size
    assert metrics["eval_scope"] == "same_file_train_rows_excluded"
    assert metrics["excluded_train_rows"] == train.size


def test_auto_exclusion_counts_max_samples_as_retained_rows(tmp_path: Path) -> None:
    n = 120
    data = tmp_path / "data.h5"
    _make_h5(data, n=n)
    run_dir = _make_run_dir(tmp_path / "run")
    _write_split(run_dir, policy="seeded_random", dataset_sha=_CONTENT_SHA,
                 train=np.arange(0, n, 2, dtype=np.int64))
    metrics = _evaluate(run_dir, data, tmp_path / "out", streaming=True, max_samples=40)
    assert metrics["n_samples"] == 40


def test_training_file_without_indices_fails_closed(tmp_path: Path) -> None:
    data = tmp_path / "data.h5"
    _make_h5(data)
    run_dir = _make_run_dir(tmp_path / "run")
    _write_split(run_dir, policy="seeded_random", dataset_sha=_CONTENT_SHA, train=None)
    with pytest.raises(ValueError, match="training rows cannot be excluded"):
        _evaluate(run_dir, data, tmp_path / "out", streaming=True)


def test_heldout_file_is_not_filtered_and_scope_is_stated(tmp_path: Path) -> None:
    n = 120
    data = tmp_path / "data.h5"
    _make_h5(data, n=n)
    run_dir = _make_run_dir(tmp_path / "run")
    _write_split(run_dir, policy="seeded_random", dataset_sha="cd" * 32,
                 train=np.arange(0, n, 2, dtype=np.int64))
    metrics = _evaluate(run_dir, data, tmp_path / "out", streaming=True)
    assert metrics["n_samples"] == n
    assert metrics["eval_scope"] == "heldout_file"


def test_independent_run_on_its_training_file_is_labelled_training_file(tmp_path: Path) -> None:
    data = tmp_path / "data.h5"
    _make_h5(data)
    run_dir = _make_run_dir(tmp_path / "run")
    prov = run_dir / "provenance"
    prov.mkdir(parents=True, exist_ok=True)
    (prov / "split_manifest.json").write_text(
        json.dumps({"split_policy": "independent_files", "train_file_sha256": _CONTENT_SHA})
    )
    metrics = _evaluate(run_dir, data, tmp_path / "out", streaming=True)
    assert metrics["eval_scope"] == "training_file"
    assert metrics["n_samples"] == 120  # nothing can be excluded: every row was trained on


def test_unexcluded_scope_falls_back_to_scaler_provenance(tmp_path: Path) -> None:
    from lunaris.surrogate.st_lrps.evaluation.cli import _unexcluded_eval_scope

    prov = tmp_path / "provenance"
    prov.mkdir()
    (prov / "split_manifest.json").write_text(json.dumps({"split_policy": "independent_files"}))
    scaler = tmp_path / "scaler.json"
    scaler.write_text(json.dumps({"provenance": {"dataset_content_sha256": _CONTENT_SHA}}))
    assert _unexcluded_eval_scope(prov, scaler, {"content_sha256": _CONTENT_SHA}) == "training_file"
    assert _unexcluded_eval_scope(prov, scaler, {"content_sha256": "ef" * 32}) == "heldout_file"
    assert _unexcluded_eval_scope(prov, scaler, {}) == "unverified"
    assert _unexcluded_eval_scope(prov, tmp_path / "missing.json", {"content_sha256": _CONTENT_SHA}) == "unverified"


def test_scope_is_unverified_without_hashes(tmp_path: Path) -> None:
    data = tmp_path / "data.h5"
    _make_h5(data, content_sha256=None)
    run_dir = _make_run_dir(tmp_path / "run")
    _evaluate(run_dir, data, tmp_path / "out", streaming=False)
    metrics = json.loads((tmp_path / "out" / "eval_report.json").read_text())["metrics"]
    assert metrics["eval_scope"] == "unverified"
    assert metrics["excluded_train_rows"] == 0
