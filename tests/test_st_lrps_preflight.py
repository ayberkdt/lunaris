"""P2 regression: the single ST-LRPS pre-training preflight go/no-go gate.

Validates each check in isolation and the aggregate ``run_preflight`` decision,
including the leakage-style blockers (split overlap, non-train-only scaler, OOD
band overlap) that must refuse to start, and the graceful ``skip`` of checks
whose inputs are absent (independent train/val files).
"""

from __future__ import annotations
import pytest
try:
    import torch
    import torch.nn
    _ = torch.cuda
    _ = getattr(torch, "compile", None)
except (ImportError, AttributeError, ModuleNotFoundError):
    pytest.skip("PyTorch not installed or missing required attributes like cuda/compile", allow_module_level=True)



import json

import numpy as np

from lunaris.surrogate.st_lrps.training.preflight import (
    FAIL,
    PASS,
    SKIP,
    WARN,
    check_determinism_consistency,
    check_ood_band_ordering,
    check_resource_headroom,
    check_scaler_train_only,
    check_split_integrity,
    run_preflight,
    write_preflight_report,
)


def test_split_integrity_disjoint_passes() -> None:
    splits = {
        "train": np.array([0, 1, 2, 3]),
        "val": np.array([4, 5]),
        "test": np.array([], dtype=np.int64),
        "ood": np.array([], dtype=np.int64),
    }
    assert check_split_integrity(splits).status == PASS


def test_split_integrity_overlap_fails() -> None:
    splits = {"train": np.array([0, 1, 2]), "val": np.array([2, 3])}
    check = check_split_integrity(splits)
    assert check.status == FAIL
    assert "leakage" in check.detail


def test_split_integrity_empty_train_fails() -> None:
    splits = {"train": np.array([], dtype=np.int64), "val": np.array([0, 1])}
    assert check_split_integrity(splits).status == FAIL


def test_split_integrity_none_skips() -> None:
    assert check_split_integrity(None).status == SKIP


def test_ood_low_band_below_train_passes() -> None:
    manifest = {
        "altitude_range_per_split": {
            "train": {"min": 50.0, "max": 100.0},
            "val": {"min": 10.0, "max": 40.0},
            "ood": {"min": 5.0, "max": 30.0},
        }
    }
    assert check_ood_band_ordering("ood_low_altitude", manifest).status == PASS


def test_ood_low_band_overlapping_train_fails() -> None:
    manifest = {
        "altitude_range_per_split": {
            "train": {"min": 50.0, "max": 100.0},
            "val": {"min": 10.0, "max": 60.0},  # 60 > train.min 50 -> overlap
        }
    }
    check = check_ood_band_ordering("ood_low_altitude", manifest)
    assert check.status == FAIL
    assert "overlaps train" in check.detail


def test_ood_high_band_above_train_passes() -> None:
    manifest = {
        "altitude_range_per_split": {
            "train": {"min": 50.0, "max": 100.0},
            "ood": {"min": 120.0, "max": 200.0},
        }
    }
    assert check_ood_band_ordering("ood_high_altitude", manifest).status == PASS


def test_ood_ordering_skipped_for_non_ood_policy() -> None:
    assert check_ood_band_ordering("seeded_random", {}).status == SKIP


def test_scaler_train_only_passes_and_full_dataset_fails() -> None:
    assert check_scaler_train_only({"fit_scope": "train_only"}).status == PASS
    leaky = check_scaler_train_only({"fit_scope": "full_dataset"})
    assert leaky.status == FAIL
    assert "leak" in leaky.detail
    assert check_scaler_train_only(None).status == SKIP


def test_determinism_consistency_warns_on_benchmark_with_deterministic() -> None:
    assert check_determinism_consistency(deterministic=True, benchmark_cudnn=True).status == WARN
    assert check_determinism_consistency(deterministic=True, benchmark_cudnn=False).status == PASS
    warned = check_determinism_consistency(
        deterministic=True,
        benchmark_cudnn=False,
        determinism_flags={"use_deterministic_algorithms": False},
    )
    assert warned.status == WARN


def test_resource_headroom_passes_on_real_tmp_dir(tmp_path) -> None:
    check = check_resource_headroom(out_dir=tmp_path, device_type="cpu")
    assert check.status == PASS
    assert "free disk" in check.detail


def test_run_preflight_all_good_is_go(tmp_path) -> None:
    report = run_preflight(
        out_dir=tmp_path,
        deterministic=True,
        benchmark_cudnn=False,
        validation_report={"passed": True, "warnings": []},
        splits={"train": np.array([0, 1, 2]), "val": np.array([3, 4])},
        split_policy="seeded_random",
        split_manifest={},
        scaler_provenance={"fit_scope": "train_only"},
    )
    assert report.go is True
    assert all(c.status in (PASS, SKIP) for c in report.checks)


def test_run_preflight_blocks_on_scaler_leakage(tmp_path) -> None:
    report = run_preflight(
        out_dir=tmp_path,
        validation_report={"passed": True},
        splits={"train": np.array([0, 1]), "val": np.array([2, 3])},
        scaler_provenance={"fit_scope": "full_dataset"},
    )
    assert report.go is False
    assert "scaler_train_only" in report.summary()


def test_run_preflight_blocks_on_failed_dataset_validation(tmp_path) -> None:
    report = run_preflight(
        out_dir=tmp_path,
        validation_report={"passed": False, "errors": ["NaN values present"]},
        scaler_provenance={"fit_scope": "train_only"},
    )
    assert report.go is False


def test_run_preflight_independent_files_skip_split_check(tmp_path) -> None:
    report = run_preflight(
        out_dir=tmp_path,
        validation_report={"passed": True},
        splits=None,  # independent train/val files
        scaler_provenance={"fit_scope": "train_only"},
    )
    assert report.go is True
    by_name = {c.name: c for c in report.checks}
    assert by_name["split_integrity"].status == SKIP


def test_write_preflight_report_round_trips(tmp_path) -> None:
    report = run_preflight(
        out_dir=tmp_path,
        validation_report={"passed": True},
        scaler_provenance={"fit_scope": "train_only"},
    )
    path = write_preflight_report(tmp_path, report)
    assert path.name == "preflight_report.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["go"] is True
    assert payload["schema_version"] == 1
    assert "summary_counts" in payload
    assert isinstance(payload["checks"], list) and payload["checks"]
