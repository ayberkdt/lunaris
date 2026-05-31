from __future__ import annotations

import json

import numpy as np

from lunaris.surrogate.st_lrps.data.dataset_validation import validate_dataset_file

from dataset_pipeline_test_utils import (
    make_toy_residual_rows,
    toy_baseline_fn,
    toy_truth_fn,
    write_toy_contract_h5,
)


def test_dataset_validation_passes_and_writes_report(tmp_path):
    data_path = write_toy_contract_h5(tmp_path / "toy.h5", n=24)
    out_dir = tmp_path / "reports"

    report = validate_dataset_file(
        data_path,
        out_dir=out_dir,
        n_check=24,
        truth_fn=toy_truth_fn,
        baseline_fn=toy_baseline_fn,
        potential_atol=128.0,
        accel_atol=1e-2,
    )

    assert report["passed"] is True
    assert report["n_samples_total"] == 24
    assert "dataset_contract" in report["checked"]
    written = json.loads((out_dir / "dataset_validation_report.json").read_text(encoding="utf-8"))
    assert written["passed"] is True


def test_dataset_validation_rejects_label_mismatch(tmp_path):
    rows = make_toy_residual_rows(n=16)
    rows[3, 4] += 10.0
    data_path = write_toy_contract_h5(tmp_path / "bad_labels.h5", rows=rows)

    report = validate_dataset_file(
        data_path,
        n_check=16,
        truth_fn=toy_truth_fn,
        baseline_fn=toy_baseline_fn,
        potential_atol=128.0,
        accel_atol=1e-4,
    )

    assert report["passed"] is False
    assert any("residual acceleration label mismatch" in msg for msg in report["errors"])


def test_dataset_validation_detects_non_finite_values(tmp_path):
    rows = make_toy_residual_rows(n=12)
    rows[0, 3] = np.nan
    data_path = write_toy_contract_h5(tmp_path / "nan.h5", rows=rows)

    report = validate_dataset_file(data_path, n_check=12)

    assert report["passed"] is False
    assert report["nan_count"] == 1
