"""Task 6 — ST-LRPS validation suite (field metrics across split policies)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from lunaris.surrogate.st_lrps.data.dataset_parameters import R_MOON_SI
from lunaris.surrogate.st_lrps.evaluation.validation_suite import (
    SPLIT_KIND,
    attach_orbit_validation,
    compute_field_metrics,
    write_validation_report,
)


def test_compute_field_metrics_known_error():
    n = 200
    # Points spread along +x at lunar radius so r_hat = +x.
    xyz = np.zeros((n, 3))
    xyz[:, 0] = R_MOON_SI + np.linspace(100e3, 400e3, n)
    a_true = np.tile([1.0, 0.0, 0.0], (n, 1))
    # Constant error purely cross-radial (y), magnitude 1e-3.
    a_pred = a_true + np.tile([0.0, 1.0e-3, 0.0], (n, 1))
    u_true = np.zeros(n)
    u_pred = u_true + 2.0  # constant potential error -> RMSE = 2.0

    m = compute_field_metrics(xyz, u_true, a_true, u_pred, a_pred, r_ref_m=R_MOON_SI)

    assert m["count"] == n
    assert m["residual_potential_rmse_m2_s2"] == pytest.approx(2.0, rel=1e-9)
    assert m["residual_accel_rmse_m_s2"] == pytest.approx(1.0e-3, rel=1e-6)
    assert m["relative_accel_error_pct_median"] == pytest.approx(0.1, rel=1e-6)
    # Error is perpendicular to the radial direction.
    assert m["radial_error_rms_m_s2"] < 1e-12
    assert m["cross_radial_error_rms_m_s2"] == pytest.approx(1.0e-3, rel=1e-6)
    # angle = atan(1e-3 / 1) in degrees.
    assert m["angular_accel_error_deg_median"] == pytest.approx(np.degrees(np.arctan(1e-3)), rel=1e-3)
    assert m["accel_error_p99_m_s2"] == pytest.approx(1.0e-3, rel=1e-6)
    assert isinstance(m["altitude_binned_error"], list) and m["altitude_binned_error"]
    assert isinstance(m["latitude_binned_error"], list)
    assert isinstance(m["longitude_binned_error"], list)


def test_compute_field_metrics_empty():
    empty = np.zeros((0, 3))
    m = compute_field_metrics(empty, np.zeros(0), empty, np.zeros(0), empty)
    assert m["count"] == 0


def test_write_report_separates_generalization_kinds(tmp_path):
    # Hand-built report mixing the split kinds; the markdown must label each
    # section so interpolation is never conflated with generalization.
    report = {
        "schema_version": 1,
        "model_dir": "run",
        "dataset_path": "data.h5",
        "split_seed": 1,
        "n_rows": 100,
        "field_validation": {
            "seeded_random": {"kind": "interpolation", "residual_accel_rmse_m_s2": 1e-6, "count": 10},
            "spatial_block": {"kind": "spatial_generalization", "residual_accel_rmse_m_s2": 2e-6, "count": 10},
            "ood_low_altitude": {"kind": "altitude_extrapolation", "residual_accel_rmse_m_s2": 5e-6, "count": 10},
        },
        "orbit_validation": None,
    }
    paths = write_validation_report(report, tmp_path)
    md = paths["md"].read_text(encoding="utf-8")
    assert "Interpolation" in md
    assert "Spatial generalization" in md
    assert "Altitude extrapolation" in md
    assert "Trajectory propagation" in md
    loaded = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert loaded["field_validation"]["spatial_block"]["kind"] == "spatial_generalization"


def test_attach_orbit_validation():
    report = {"orbit_validation": None}
    attach_orbit_validation(report, {"final_pos_err_km": 1.2, "rms_pos_err_km": 0.8})
    assert report["orbit_validation"]["final_pos_err_km"] == 1.2


def test_split_kind_labels_cover_default_policies():
    for policy in ("seeded_random", "spatial_block", "ood_low_altitude", "ood_high_altitude"):
        assert policy in SPLIT_KIND


# --- Integration smoke (needs torch + a real run dir) ----------------------

torch = pytest.importorskip("torch")

from dataset_pipeline_test_utils import write_toy_contract_h5
from lunaris.surrogate.st_lrps.evaluation.validation_suite import run_field_validation
from st_lrps_contract_test_utils import make_contract_run


def test_run_field_validation_smoke(tmp_path):
    run = make_contract_run(tmp_path, degree_min=20, degree_max=60, alt_min_km=100.0, alt_max_km=500.0)
    dataset = write_toy_contract_h5(tmp_path / "cloud.h5", n=400, alt_min_km=100.0, alt_max_km=500.0)

    report = run_field_validation(
        run["run_dir"], dataset, split_seed=7, val_fraction=0.2, device="cpu"
    )
    field = report["field_validation"]
    for policy in ("seeded_random", "altitude_stratified", "spatial_block", "ood_low_altitude", "ood_high_altitude"):
        assert policy in field, policy
    # Random + spatial + OOD-low should all evaluate without error.
    for policy in ("seeded_random", "spatial_block", "ood_low_altitude"):
        assert "error" not in field[policy], field[policy]
        assert field[policy]["count"] > 0
        assert field[policy]["kind"] == SPLIT_KIND[policy]
        assert np.isfinite(field[policy]["residual_accel_rmse_m_s2"])

    paths = write_validation_report(report, tmp_path / "report")
    assert paths["json"].exists() and paths["md"].exists()


def test_run_field_validation_excludes_training_rows(tmp_path):
    """A verified split_manifest must force a held-out evaluation even when the
    evaluation re-split seed differs from the training seed (Risk 1)."""
    from lunaris.surrogate.st_lrps.data.splits import _hash_indices, make_seeded_random_split

    run = make_contract_run(tmp_path, degree_min=20, degree_max=60, alt_min_km=100.0, alt_max_km=500.0)
    n = 400
    dataset = write_toy_contract_h5(tmp_path / "cloud.h5", n=n, alt_min_km=100.0, alt_max_km=500.0)

    # Training split the model "saw": seed 123. Write a matching manifest.
    train_seed = 123
    splits = make_seeded_random_split(n, val_fraction=0.2, test_fraction=0.0, seed=train_seed)
    manifest = {
        "schema_version": 1,
        "split_policy": "seeded_random",
        "split_seed": train_seed,
        "train_count": int(splits["train"].size),
        "val_count": int(splits["val"].size),
        "test_count": int(splits["test"].size),
        "ood_count": 0,
        "index_hashes": {name: _hash_indices(idx) for name, idx in splits.items()},
    }
    prov = run["run_dir"] / "provenance"
    prov.mkdir(parents=True, exist_ok=True)
    (prov / "split_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    # Evaluate with a DIFFERENT seed so the naive val set overlaps training rows.
    report = run_field_validation(
        run["run_dir"], dataset, policies=["seeded_random"], split_seed=999,
        val_fraction=0.2, device="cpu", enforce_heldout=True,
    )
    guard = report["leakage_guard"]
    assert guard["status"] == "verified", guard
    m = report["field_validation"]["seeded_random"]
    assert "error" not in m, m
    assert m["leakage_guard_status"] == "verified"
    hf = m["heldout_filtering"]
    # A differently-seeded random val set must overlap the training train-set, so
    # some rows are excluded; the evaluated count is what remains.
    assert hf["train_rows_excluded"] > 0, hf
    assert hf["evaluated"] == hf["val_before"] - hf["train_rows_excluded"]
    assert m["count"] == hf["evaluated"]


def test_run_field_validation_unverified_when_no_manifest(tmp_path):
    """No split_manifest -> guard is flagged, not silently treated as clean."""
    run = make_contract_run(tmp_path, degree_min=20, degree_max=60, alt_min_km=100.0, alt_max_km=500.0)
    dataset = write_toy_contract_h5(tmp_path / "cloud.h5", n=300, alt_min_km=100.0, alt_max_km=500.0)
    report = run_field_validation(
        run["run_dir"], dataset, policies=["seeded_random"], split_seed=7,
        val_fraction=0.2, device="cpu", enforce_heldout=True,
    )
    assert report["leakage_guard"]["status"] == "manifest_missing"
    # Non-strict still produces metrics, but flagged.
    assert report["field_validation"]["seeded_random"]["leakage_guard_status"] == "manifest_missing"
    md = write_validation_report(report, tmp_path / "report")["md"].read_text(encoding="utf-8")
    assert "Leakage guard not verified" in md

    with pytest.raises(RuntimeError):
        run_field_validation(
            run["run_dir"], dataset, policies=["seeded_random"], split_seed=7,
            val_fraction=0.2, device="cpu", enforce_heldout=True, strict_leakage=True,
        )


def _write_manifest(prov, manifest):
    prov.mkdir(parents=True, exist_ok=True)
    (prov / "split_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_persisted_indices_bypass_fragile_reconstruction(tmp_path):
    """split_indices.npz makes held-out filtering deterministic even when the
    policy/seed reconstruction would NOT reproduce the same split (Risk: rounding
    fragility for altitude_stratified etc.)."""
    from lunaris.surrogate.st_lrps.data.splits import _hash_indices, write_split_indices

    run = make_contract_run(tmp_path, degree_min=20, degree_max=60, alt_min_km=100.0, alt_max_km=500.0)
    n = 400
    dataset = write_toy_contract_h5(tmp_path / "cloud.h5", n=n, alt_min_km=100.0, alt_max_km=500.0)

    # An ARBITRARY train set that no seeded_random reconstruction would produce.
    train = np.arange(0, 300, dtype=np.int64)
    val = np.arange(300, 400, dtype=np.int64)
    splits = {"train": train, "val": val, "test": np.asarray([], np.int64), "ood": np.asarray([], np.int64)}
    prov = run["run_dir"] / "provenance"
    _write_manifest(prov, {
        "schema_version": 1, "split_policy": "seeded_random", "split_seed": 999,
        "train_count": 300, "val_count": 100, "test_count": 0, "ood_count": 0,
        "index_hashes": {name: _hash_indices(idx) for name, idx in splits.items()},
    })
    write_split_indices(prov / "split_indices.npz", splits)

    report = run_field_validation(
        run["run_dir"], dataset, policies=["seeded_random"], split_seed=7,
        val_fraction=0.2, device="cpu", enforce_heldout=True,
    )
    guard = report["leakage_guard"]
    assert guard["status"] == "verified", guard
    assert guard["method"] == "persisted_indices", guard
    # Every evaluated row must be outside the persisted training set [0, 300).
    hf = report["field_validation"]["seeded_random"].get("heldout_filtering")
    assert hf is not None and hf["evaluated"] > 0


def test_dataset_identity_mismatch_is_unverified(tmp_path):
    """A manifest whose dataset_content_sha256 does not match the eval dataset
    must be flagged (and hard-fail under strict): index values alone cannot catch
    a wrong/modified dataset with the same row count."""
    from lunaris.surrogate.st_lrps.data.splits import _hash_indices, write_split_indices

    run = make_contract_run(tmp_path, degree_min=20, degree_max=60, alt_min_km=100.0, alt_max_km=500.0)
    n = 400
    dataset = write_toy_contract_h5(tmp_path / "cloud.h5", n=n, alt_min_km=100.0, alt_max_km=500.0)
    train = np.arange(0, 320, dtype=np.int64)
    val = np.arange(320, 400, dtype=np.int64)
    splits = {"train": train, "val": val, "test": np.asarray([], np.int64), "ood": np.asarray([], np.int64)}
    prov = run["run_dir"] / "provenance"
    _write_manifest(prov, {
        "schema_version": 1, "split_policy": "seeded_random", "split_seed": 1,
        "train_count": 320, "val_count": 80, "test_count": 0, "ood_count": 0,
        "index_hashes": {name: _hash_indices(idx) for name, idx in splits.items()},
        "dataset_content_sha256": "0" * 64,  # deliberately wrong
    })
    write_split_indices(prov / "split_indices.npz", splits)

    report = run_field_validation(
        run["run_dir"], dataset, policies=["seeded_random"], split_seed=7,
        val_fraction=0.2, device="cpu", enforce_heldout=True,
    )
    guard = report["leakage_guard"]
    assert guard["status"] == "unverified"
    assert guard["dataset_identity"] == "mismatch"

    with pytest.raises(RuntimeError):
        run_field_validation(
            run["run_dir"], dataset, policies=["seeded_random"], split_seed=7,
            val_fraction=0.2, device="cpu", enforce_heldout=True, strict_leakage=True,
        )


def test_dataset_identity_verified_with_matching_hash(tmp_path):
    """A matching dataset_content_sha256 yields dataset_identity='verified'."""
    from lunaris.surrogate.st_lrps.data.dataset_contract import content_sha256_for_hdf5_dataset
    from lunaris.surrogate.st_lrps.data.splits import _hash_indices, write_split_indices

    run = make_contract_run(tmp_path, degree_min=20, degree_max=60, alt_min_km=100.0, alt_max_km=500.0)
    n = 400
    dataset = write_toy_contract_h5(tmp_path / "cloud.h5", n=n, alt_min_km=100.0, alt_max_km=500.0)
    real_sha = content_sha256_for_hdf5_dataset(dataset, dataset_name="data")
    train = np.arange(0, 320, dtype=np.int64)
    val = np.arange(320, 400, dtype=np.int64)
    splits = {"train": train, "val": val, "test": np.asarray([], np.int64), "ood": np.asarray([], np.int64)}
    prov = run["run_dir"] / "provenance"
    _write_manifest(prov, {
        "schema_version": 1, "split_policy": "seeded_random", "split_seed": 1,
        "train_count": 320, "val_count": 80, "test_count": 0, "ood_count": 0,
        "index_hashes": {name: _hash_indices(idx) for name, idx in splits.items()},
        "dataset_content_sha256": real_sha,
    })
    write_split_indices(prov / "split_indices.npz", splits)

    report = run_field_validation(
        run["run_dir"], dataset, policies=["seeded_random"], split_seed=7,
        val_fraction=0.2, device="cpu", enforce_heldout=True,
    )
    guard = report["leakage_guard"]
    assert guard["status"] == "verified"
    assert guard["dataset_identity"] == "verified"
    assert guard["method"] == "persisted_indices"


def test_curl_diagnostics_flag_non_conservative_field(tmp_path):
    """force_direct conservativeness surfacing (Risk 2): a rotational field must
    report nonconservative_ratio > 0, a gradient field ~0."""
    from lunaris.surrogate.st_lrps.evaluation.validation_suite import _maybe_curl_diagnostics

    rng = np.random.default_rng(0)
    xyz = R_MOON_SI * np.ones((256, 1)) * np.array([1.0, 0.0, 0.0]) + rng.normal(0, 50e3, (256, 3))

    class _Rotational:
        runtime_model_kind = "force_direct"
        def predict_residual_accel_fixed(self, pts):
            pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
            out = np.zeros_like(pts)
            out[:, 0] = -pts[:, 1]
            out[:, 1] = pts[:, 0]
            return out

    class _Conservative:
        runtime_model_kind = "force_direct"
        def predict_residual_accel_fixed(self, pts):
            return -np.asarray(pts, dtype=np.float64).reshape(-1, 3)

    rot = _maybe_curl_diagnostics(_Rotational(), xyz, xyz.shape[0], seed=1, max_points=256, step_m=1.0)
    cons = _maybe_curl_diagnostics(_Conservative(), xyz, xyz.shape[0], seed=1, max_points=256, step_m=1.0)
    assert rot["nonconservative_ratio"] > 0.5, rot
    assert cons["nonconservative_ratio"] < 1e-6, cons
    assert "warning" in rot
