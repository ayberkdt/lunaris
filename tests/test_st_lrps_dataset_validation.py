"""Unit coverage for the ST-LRPS dataset-validation gate."""

from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip('torch')

from dataset_pipeline_test_utils import (
    make_toy_residual_rows,
    toy_baseline_fn,
    toy_truth_fn,
    write_toy_contract_h5,
)
from lunaris.surrogate.st_lrps.data.dataset_contract import DatasetContractError
from lunaris.surrogate.st_lrps.data.dataset_validation import (
    require_dataset_valid,
    validate_dataset_file,
)


def test_validate_clean_dataset_passes_and_writes_report(tmp_path):
    data = write_toy_contract_h5(tmp_path / "clean.h5", n=48, alt_min_km=100.0, alt_max_km=500.0)
    out = tmp_path / "report"
    report = validate_dataset_file(data, out_dir=out, n_check=48)

    assert report["passed"] is True, report["errors"]
    assert report["errors"] == []
    assert {"shape", "finite_values", "altitude_envelope", "duplicate_points", "outliers"}.issubset(
        set(report["checked"])
    )
    assert report["n_samples_total"] == 48
    assert report["nan_count"] == 0 and report["inf_count"] == 0
    assert 90.0 <= report["altitude_min_km"] <= 110.0
    assert 490.0 <= report["altitude_max_km"] <= 510.0
    written = json.loads((out / "dataset_validation_report.json").read_text(encoding="utf-8"))
    assert written["passed"] is True


def test_validate_detects_nan(tmp_path):
    rows = make_toy_residual_rows(n=16)
    rows[0, 3] = np.nan
    data = write_toy_contract_h5(tmp_path / "nan.h5", rows=rows)
    report = validate_dataset_file(data, n_check=16)
    assert report["passed"] is False
    assert report["nan_count"] >= 1
    assert any("NaN" in e for e in report["errors"])


def test_validate_detects_wrong_shape(tmp_path):
    rows = np.zeros((8, 5), dtype=np.float32)
    data = write_toy_contract_h5(tmp_path / "wrong_shape.h5", rows=rows)
    report = validate_dataset_file(data, n_check=8)
    assert report["passed"] is False
    assert any("shape (N, 7)" in e for e in report["errors"])


def test_validate_altitude_envelope_strict_errors_lenient_warns(tmp_path):
    # Rows reach 2000 km but the contract envelope is [100, 500] km.
    rows = make_toy_residual_rows(n=24, alt_min_km=100.0, alt_max_km=2000.0)
    data = write_toy_contract_h5(tmp_path / "envelope.h5", rows=rows, alt_min_km=100.0, alt_max_km=500.0)

    strict = validate_dataset_file(data, n_check=24, strict=True)
    assert strict["passed"] is False
    assert any("altitude envelope" in e for e in strict["errors"])

    lenient = validate_dataset_file(data, n_check=24, strict=False)
    assert lenient["passed"] is True
    assert any("altitude envelope" in w for w in lenient["warnings"])


def test_residual_label_recompute_pass_and_mismatch(tmp_path):
    rows = make_toy_residual_rows(n=20)
    data = write_toy_contract_h5(tmp_path / "labels.h5", rows=rows)

    # Matching labels (generous atol absorbs the float32 storage rounding).
    ok = validate_dataset_file(
        data, n_check=20, truth_fn=toy_truth_fn, baseline_fn=toy_baseline_fn,
        potential_atol=1.0, accel_atol=1e-3,
    )
    assert ok["passed"] is True, ok["errors"]
    assert ok["residual_potential_max_abs_error"] is not None
    assert "residual_label_recompute" in ok["checked"]

    # Corrupt the potential label -> recompute mismatch fails.
    bad_rows = rows.copy()
    bad_rows[:, 3] = bad_rows[:, 3] + 1000.0
    bad = write_toy_contract_h5(tmp_path / "labels_bad.h5", rows=bad_rows)
    report = validate_dataset_file(
        bad, n_check=20, truth_fn=toy_truth_fn, baseline_fn=toy_baseline_fn,
    )
    assert report["passed"] is False
    assert any("potential label mismatch" in e for e in report["errors"])


def test_require_dataset_valid_raises_on_invalid(tmp_path):
    rows = make_toy_residual_rows(n=12)
    rows[1, 5] = np.inf
    data = write_toy_contract_h5(tmp_path / "inf.h5", rows=rows)
    with pytest.raises(DatasetContractError):
        require_dataset_valid(data, n_check=12)


def test_require_dataset_valid_returns_report_when_clean(tmp_path):
    data = write_toy_contract_h5(tmp_path / "clean2.h5", n=16)
    report = require_dataset_valid(data, n_check=16)
    assert report["passed"] is True
