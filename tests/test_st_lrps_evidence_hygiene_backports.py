"""Field-validation, paper-evidence and benchmark-output honesty regressions."""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from lunaris.surrogate.st_lrps.data.dataset_parameters import R_MOON_SI  # noqa: E402


def _write_h5(path: Path, n: int = 16, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    data = rng.normal(size=(n, 7))
    data[:, 0] += R_MOON_SI + 1e5
    with h5py.File(path, "w") as hf:
        hf.create_dataset("data", data=data)
    return path


# ----------------------------------------------------------------------------
# validation_suite
# ----------------------------------------------------------------------------

def test_field_metrics_are_nan_when_any_prediction_is_non_finite() -> None:
    from lunaris.surrogate.st_lrps.evaluation.validation_suite import compute_field_metrics

    n = 50
    xyz = np.zeros((n, 3))
    xyz[:, 0] = R_MOON_SI + np.linspace(100e3, 400e3, n)
    a_true = np.tile([1.0, 0.0, 0.0], (n, 1))
    a_pred = a_true + 1e-3
    a_pred[7, 1] = np.nan
    u_true = np.zeros(n)
    u_pred = u_true + 2.0
    m = compute_field_metrics(xyz, u_true, a_true, u_pred, a_pred, r_ref_m=R_MOON_SI)
    assert m["non_finite_prediction_count"] == 1
    for key in ("residual_accel_rmse_m_s2", "residual_accel_mae_m_s2", "accel_error_p50_m_s2",
                "accel_error_p99_m_s2", "radial_error_rms_m_s2", "relative_accel_error_pct_mean",
                "residual_potential_rmse_m2_s2"):
        assert math.isnan(m[key]), key


def test_label_split_kind_downgrades_mismatched_training_policy() -> None:
    from lunaris.surrogate.st_lrps.evaluation.validation_suite import label_split_kind

    assert label_split_kind("seeded_random", "spatial_block") == ("interpolation", None)
    assert label_split_kind("spatial_block", "spatial_block") == ("spatial_generalization", None)
    kind, reason = label_split_kind("spatial_block", "seeded_random")
    assert kind == "interpolation" and "seeded_random" in str(reason)
    kind, reason = label_split_kind("ood_high_altitude", None)
    assert kind == "interpolation" and "unknown" in str(reason)


def _independent_run(tmp_path: Path, *, manifest_extra: dict, scaler_sha: str | None) -> Path:
    run = tmp_path / "run"
    (run / "provenance").mkdir(parents=True)
    manifest = {"split_policy": "independent_files", "train_count": 10, "val_count": 4, **manifest_extra}
    (run / "provenance" / "split_manifest.json").write_text(json.dumps(manifest))
    prov = {"fit_scope": "train_only"}
    if scaler_sha is not None:
        prov["dataset_content_sha256"] = scaler_sha
    (run / "scaler.json").write_text(json.dumps({"provenance": prov}))
    return run


def test_independent_files_guard_checks_the_training_file_hash(tmp_path: Path) -> None:
    from lunaris.surrogate.st_lrps.data.dataset_contract import content_sha256_for_hdf5_dataset
    from lunaris.surrogate.st_lrps.evaluation.validation_suite import (
        _resolve_training_heldout_guard,
    )

    data = _write_h5(tmp_path / "eval.h5")
    sha = content_sha256_for_hdf5_dataset(data)
    xyz = np.zeros((16, 3))
    alt = np.zeros(16)

    # Old manifests carry no file hash: the train-only scaler's provenance does.
    run = _independent_run(tmp_path / "a", manifest_extra={}, scaler_sha=sha)
    g = _resolve_training_heldout_guard(run, 16, xyz, alt, dataset_path=data)
    assert g["status"] == "unverified" and g["dataset_identity"] == "training_file"

    run = _independent_run(tmp_path / "b", manifest_extra={"train_file_sha256": "f" * 64}, scaler_sha=None)
    g = _resolve_training_heldout_guard(run, 16, xyz, alt, dataset_path=data)
    assert g["status"] == "independent_files" and g["dataset_identity"] == "verified"

    run = _independent_run(tmp_path / "c", manifest_extra={}, scaler_sha=None)
    g = _resolve_training_heldout_guard(run, 16, xyz, alt, dataset_path=data)
    assert g["status"] == "unknown"


# ----------------------------------------------------------------------------
# paper_evidence
# ----------------------------------------------------------------------------

def _manifest_for(splits: dict[str, np.ndarray]) -> dict:
    from lunaris.surrogate.st_lrps.data.splits import hash_indices

    return {
        "split_policy": "seeded_random",
        **{f"{k}_count": int(v.size) for k, v in splits.items()},
        "index_hashes": {k: hash_indices(v) for k, v in splits.items()},
    }


def test_split_check_catches_overlapping_indices_with_distinct_hashes(tmp_path: Path) -> None:
    from lunaris.surrogate.st_lrps.paper_evidence.runner import (
        PaperEvidenceError,
        _verify_split_disjointness,
    )

    splits = {"train": np.arange(0, 100), "val": np.arange(50, 60), "test": np.arange(0), "ood": np.arange(0)}
    manifest = _manifest_for(splits)
    np.savez(tmp_path / "split_indices.npz", **splits)
    _verify_split_disjointness(manifest)  # hash-only view cannot see it
    with pytest.raises(PaperEvidenceError, match="overlap"):
        _verify_split_disjointness(manifest, provenance_dir=tmp_path)

    clean = {"train": np.arange(0, 100), "val": np.arange(100, 110), "test": np.arange(0), "ood": np.arange(0)}
    np.savez(tmp_path / "split_indices.npz", **clean)
    _verify_split_disjointness(_manifest_for(clean), provenance_dir=tmp_path)

    # A sidecar that does not match the manifest hashes is refused.
    with pytest.raises(PaperEvidenceError, match="does not match"):
        _verify_split_disjointness(manifest, provenance_dir=tmp_path)


def test_independent_files_split_needs_distinct_file_hashes() -> None:
    from lunaris.surrogate.st_lrps.paper_evidence.runner import (
        PaperEvidenceError,
        _verify_split_disjointness,
    )

    base = {"split_policy": "independent_files", "train_count": 10, "val_count": 4, "index_hashes": {}}
    with pytest.raises(PaperEvidenceError, match="train_file_sha256"):
        _verify_split_disjointness(base)
    with pytest.raises(PaperEvidenceError, match="same file"):
        _verify_split_disjointness({**base, "train_file_sha256": "a" * 64, "val_file_sha256": "a" * 64})
    _verify_split_disjointness({**base, "train_file_sha256": "a" * 64, "val_file_sha256": "b" * 64})


def test_multi_seed_std_is_the_sample_stdev() -> None:
    from lunaris.surrogate.st_lrps.paper_evidence.multi_seed import aggregate_multi_seed

    entries = [{"seed": s, "bench1day_median_rms_km": v} for s, v in ((1, 1.0), (2, 2.0), (3, 3.0))]
    stats = aggregate_multi_seed(entries)["statistics"]["bench1day_median_rms_km"]
    assert stats["std"] == pytest.approx(statistics.stdev([1.0, 2.0, 3.0]))  # 1.0, not pstdev 0.816
    single = aggregate_multi_seed(entries[:1])["statistics"]["bench1day_median_rms_km"]
    assert single["std"] is None


def test_evidence_manifest_refuses_to_reset_an_unreadable_file(tmp_path: Path) -> None:
    from lunaris.surrogate.st_lrps.paper_evidence.evidence_manifest import write_evidence_manifest

    path = tmp_path / "evidence_manifest.json"
    path.write_text("{ truncated", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        write_evidence_manifest(path, run_key="a", entry={"x": 1})
    assert path.read_text(encoding="utf-8") == "{ truncated"

    path.write_text(json.dumps([1, 2]), encoding="utf-8")
    with pytest.raises(ValueError, match="not an evidence manifest"):
        write_evidence_manifest(path, run_key="a", entry={"x": 1})

    path.unlink()
    write_evidence_manifest(path, run_key="a", entry={"x": 1})
    write_evidence_manifest(path, run_key="b", entry={"x": 2})
    runs = json.loads(path.read_text(encoding="utf-8"))["runs"]
    assert set(runs) == {"a", "b"}
    assert not list(tmp_path.glob("*.tmp"))


def test_training_argv_parses_string_booleans() -> None:
    from lunaris.surrogate.st_lrps.paper_evidence.training_argv import _bool_flag

    for value, expected in (("false", "--off"), ("False", "--off"), ("0", "--off"), (False, "--off"),
                            ("true", "--on"), ("yes", "--on"), (True, "--on"), (1, "--on")):
        argv: list[str] = []
        _bool_flag(argv, value, on="--on", off="--off")
        assert argv == [expected], value
    with pytest.raises(ValueError, match="boolean"):
        _bool_flag([], "flase", on="--on", off="--off")


# ----------------------------------------------------------------------------
# benchmark output honesty
# ----------------------------------------------------------------------------

def test_scenario_rows_do_not_copy_one_statistic_into_another() -> None:
    from lunaris.surrogate.st_lrps.evaluation.benchmark_pipeline import _standardize_scenario_rows

    (row,) = _standardize_scenario_rows([{"model": "m", "rms_pos_err_km": 1.5}], synthetic=False)
    assert row["trajectory_rms_km"] == 1.5  # same statistic under its standardized name
    for key in ("final_pos_err_km", "max_pos_err_km", "p95_pos_err_km", "rms_vel_err_ms", "final_vel_err_ms"):
        assert row[key] == "", key


def test_runtime_rows_do_not_fabricate_cold_warm_jit() -> None:
    from lunaris.surrogate.st_lrps.evaluation.benchmark_pipeline import _standardize_runtime_rows

    cfg = {"scenario": {"count": 2}, "propagation": {"duration_days": 1.0}}
    (row,) = _standardize_runtime_rows([{"model": "m", "total_runtime_s": 3.0, "n_steps": 10}], cfg, synthetic=False)
    assert row["cold_time_s"] == row["warm_time_s"] == row["jit_compile_time_s"] == ""
    (syn,) = _standardize_runtime_rows([{"model": "m", "total_runtime_s": 3.0}], cfg, synthetic=True)
    assert syn["warm_time_s"] == 3.0


def test_runtime_validator_accepts_unmeasured_timings_only_as_blank() -> None:
    from lunaris.surrogate.st_lrps.evaluation.benchmark_validation import (
        _check_runtime_timing_columns,
    )

    base = {"propagation_time_s": "3.0", "acceleration_evaluations_per_second": "10",
            "propagated_seconds_per_wall_second": "5"}
    errors: list[str] = []
    checked: list[str] = []
    _check_runtime_timing_columns(
        [{**base, "cold_time_s": "", "warm_time_s": "", "jit_compile_time_s": ""}], None, errors, checked
    )
    assert errors == []
    assert "runtime_timing:cold_time_s:not_measured" in checked
    errors = []
    _check_runtime_timing_columns(
        [{**base, "cold_time_s": "abc", "warm_time_s": "", "jit_compile_time_s": ""}], None, errors, []
    )
    assert any("cold_time_s is required" in e for e in errors)


def test_cached_runtime_rows_skip_scenarios_without_measured_runtime(tmp_path: Path, monkeypatch) -> None:
    pytest.importorskip("matplotlib")
    from types import SimpleNamespace

    from lunaris.surrogate.st_lrps.evaluation._gravity_benchmark import metrics as gm
    from lunaris.surrogate.st_lrps.evaluation._gravity_benchmark.types import CachedTrajectory

    runtimes = {0: 2.0, 1: float("nan"), 2: 4.0}
    monkeypatch.setattr(gm, "_cached_model_path", lambda cache_dir, model, sid: sid)
    monkeypatch.setattr(
        gm, "_load_cached_trajectory",
        lambda sid: CachedTrajectory(t=np.arange(11.0), y=np.zeros((11, 6)), runtime_s=runtimes[sid], metadata={}),
    )
    scenarios = [SimpleNamespace(scenario_id=i) for i in range(3)]
    truth = SimpleNamespace(total_runtime_s=1.0, mean_runtime_s=1.0)
    (row,) = gm._cached_gpu_runtime_rows(SimpleNamespace(), ["st_lrps"], scenarios, tmp_path, truth)
    assert row["n_scenarios"] == 2
    assert row["n_steps"] == 20
    assert row["runtime_per_scenario_s"] == pytest.approx(3.0)  # (2 + 4) / 2, not / 3


def test_provenance_does_not_glob_an_arbitrary_checkpoint(tmp_path: Path) -> None:
    from lunaris.surrogate.st_lrps.evaluation.provenance import _find_checkpoint

    (tmp_path / "optim").mkdir()
    (tmp_path / "optim" / "adam_state.pt").write_bytes(b"x")
    assert _find_checkpoint(tmp_path) is None
    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "checkpoints" / "ckpt_last.pt").write_bytes(b"x")
    assert _find_checkpoint(tmp_path) == tmp_path / "checkpoints" / "ckpt_last.pt"


def test_phase_explained_fraction_uses_the_same_epochs_on_both_sides() -> None:
    from lunaris.common.constants import MU_MOON
    from lunaris.surrogate.st_lrps.evaluation.phase_diagnostics import (
        compute_phase_corrected_errors,
    )

    a = R_MOON_SI + 100e3
    n_mean = math.sqrt(MU_MOON / a**3)
    t = np.arange(0.0, 86400.0, 60.0)

    def circ(tt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        th = n_mean * tt
        r = a * np.column_stack([np.cos(th), np.sin(th), np.zeros_like(th)])
        v = a * n_mean * np.column_stack([-np.sin(th), np.cos(th), np.zeros_like(th)])
        return r, v

    r_gt, v_gt = circ(t)
    lag = 2.0 + 4e-4 * t  # growing along-track lag (s)
    r_st, _ = circ(t + lag)
    r_st = r_st + np.column_stack([np.zeros_like(t), np.zeros_like(t), 5.0 * np.sin(1e-4 * t)])
    _phase, out = compute_phase_corrected_errors(t, r_st, r_gt, v_gt)
    valid = np.all(np.isfinite(out.e_aligned_m), axis=1)
    assert 0 < np.count_nonzero(valid) < t.size  # some end epochs drop out
    e = r_st - r_gt
    aligned_ms = np.mean(np.sum(out.e_aligned_m[valid] ** 2, axis=1))
    expected = float(np.clip(1.0 - aligned_ms / np.mean(np.sum(e[valid] ** 2, axis=1)), 0.0, 1.0))
    assert out.phase_explained_fraction == pytest.approx(expected, rel=1e-12)


# ----------------------------------------------------------------------------
# dataset generator / validation
# ----------------------------------------------------------------------------

def test_chunk_seeds_of_neighbouring_base_seeds_do_not_collide() -> None:
    pytest.importorskip("numba")
    from lunaris.surrogate.st_lrps.data.spatial_cloud_generator import _chunk_seed

    seeds = {(b, i): _chunk_seed(b, i) for b in range(40) for i in range(40)}
    assert len(set(seeds.values())) == len(seeds)
    # The old scheme made chunk 1 of seed 100 equal to chunk 0 of seed 101.
    assert _chunk_seed(100, 1) != _chunk_seed(101, 0)
    assert _chunk_seed(7, 3) == _chunk_seed(7, 3)  # deterministic


def test_suite_writer_leaves_no_file_under_the_final_name_on_failure(tmp_path: Path, monkeypatch) -> None:
    pytest.importorskip("numba")
    from lunaris.surrogate.st_lrps.data import spatial_cloud_generator as gen

    data = np.zeros((4, 7))
    data[:, 0] = R_MOON_SI + 2e5
    from lunaris.surrogate.st_lrps.data.dataset_parameters import MU_MOON_SI

    attrs = {"degree_min": "2", "degree_max": "4", "target_mode": "residual", "alt_min_km": "100",
             "alt_max_km": "300", "n_samples": "4", "seed": "1", "random_seed": "1",
             "mu_si": str(MU_MOON_SI), "r_ref_m": str(R_MOON_SI),
             "source_gravity_model": "toy.gfc", "source_gravity_file_path": "toy.gfc",
             "source_gravity_file_sha256": "a" * 64,
             "spherical_harmonic_convention": "4pi_geodesy_no_condon_shortley_v1"}
    out = tmp_path / "suite.h5"

    def boom(*_a, **_k):
        raise RuntimeError("interrupted")

    monkeypatch.setattr(gen, "stamp_hdf5_content_hash", boom)
    with pytest.raises(RuntimeError):
        gen._write_suite_h5(out, data, attrs, 4, np.dtype(np.float32))
    assert not out.exists()
    monkeypatch.undo()
    gen._write_suite_h5(out, data, attrs, 4, np.dtype(np.float32))
    assert out.exists() and not (tmp_path / "suite.h5.partial").exists()


def test_default_recompute_tolerance_accepts_float32_storage_and_catches_corruption(tmp_path: Path) -> None:
    from dataset_pipeline_test_utils import (
        make_toy_residual_rows,
        toy_baseline_fn,
        toy_truth_fn,
        write_toy_contract_h5,
    )
    from lunaris.surrogate.st_lrps.data.dataset_validation import validate_dataset_file

    rows = make_toy_residual_rows(n=64)  # float32 storage
    ok = validate_dataset_file(write_toy_contract_h5(tmp_path / "ok.h5", rows=rows), n_check=64,
                               truth_fn=toy_truth_fn, baseline_fn=toy_baseline_fn)
    assert ok["passed"] is True, ok["errors"]

    bad = rows.copy()
    bad[:, 4:7] *= -1.0  # sign error in the acceleration labels
    rep = validate_dataset_file(write_toy_contract_h5(tmp_path / "bad.h5", rows=bad), n_check=64,
                                truth_fn=toy_truth_fn, baseline_fn=toy_baseline_fn)
    assert rep["passed"] is False
    assert any("acceleration label mismatch" in e for e in rep["errors"])


def test_content_hash_is_recomputed_not_trusted(tmp_path: Path) -> None:
    from dataset_pipeline_test_utils import write_toy_contract_h5
    from lunaris.surrogate.st_lrps.data.dataset_validation import validate_dataset_file

    path = write_toy_contract_h5(tmp_path / "d.h5", n=24)
    clean = validate_dataset_file(path, n_check=24)
    assert clean["passed"] is True, clean["errors"]
    assert clean["content_sha256_observed"] == clean["content_sha256_recorded"]
    with h5py.File(path, "a") as hf:
        hf["data"][3, 3] = hf["data"][3, 3] + 1.0
    tampered = validate_dataset_file(path, n_check=24)
    assert tampered["passed"] is False
    assert any("content_sha256 mismatch" in e for e in tampered["errors"])


def test_worst_tail_takes_exactly_the_requested_fraction():
    import numpy as np

    from lunaris.surrogate.st_lrps.evaluation.validation_suite import _worst_tail

    err = np.arange(1.0, 101.0)  # ascending, n = 100
    assert _worst_tail(err, 0.01).tolist() == [100.0]
    assert _worst_tail(err, 0.05).tolist() == [96.0, 97.0, 98.0, 99.0, 100.0]
    assert _worst_tail(np.arange(1.0, 11.0), 0.01).tolist() == [10.0]  # at least one
    assert _worst_tail(np.array([]), 0.01).size == 0
