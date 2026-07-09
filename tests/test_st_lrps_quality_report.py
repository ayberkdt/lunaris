"""Unit coverage for the ST-LRPS dataset quality report."""

from __future__ import annotations

import json

import numpy as np

from dataset_pipeline_test_utils import make_toy_residual_rows, write_toy_contract_h5
from lunaris.surrogate.st_lrps.data.quality_report import build_dataset_quality_report


def test_quality_report_basic_stats_and_files(tmp_path):
    data = write_toy_contract_h5(tmp_path / "cloud.h5", n=40, alt_min_km=100.0, alt_max_km=500.0)
    out = tmp_path / "q"
    report = build_dataset_quality_report(data, out_dir=out, bins=8)

    assert report["n_samples"] == 40
    assert report["finite_fraction"] == 1.0
    assert report["nan_count"] == 0 and report["inf_count"] == 0
    assert report["warnings"] == []
    # Altitude stats inside the generated shell.
    assert 90.0 <= report["altitude_km"]["min"] <= 110.0
    assert 490.0 <= report["altitude_km"]["max"] <= 510.0
    # Histogram covers (nearly) every sample; boundary points at the exact
    # envelope edge can fall outside the fixed [min, max] range under float32.
    assert 38 <= sum(report["altitude_histogram"]["counts"]) <= 40
    assert len(report["altitude_histogram"]["edges_km"]) == 9  # bins + 1
    # Latitude/longitude in valid ranges.
    assert -90.0 <= report["latitude_deg"]["min"] <= report["latitude_deg"]["max"] <= 90.0
    assert -180.0 <= report["longitude_deg"]["min"] <= report["longitude_deg"]["max"] <= 180.0

    assert (out / "dataset_quality_report.json").exists()
    md = (out / "dataset_quality_summary.md").read_text(encoding="utf-8")
    assert "Dataset Quality Summary" in md
    written = json.loads((out / "dataset_quality_report.json").read_text(encoding="utf-8"))
    assert written["n_samples"] == 40


def test_quality_report_flags_nonfinite_and_duplicates(tmp_path):
    rows = make_toy_residual_rows(n=16)
    rows[0, 3] = np.nan
    rows = np.vstack([rows, rows[1:2]])  # exact duplicate position
    data = write_toy_contract_h5(tmp_path / "dirty.h5", rows=rows)
    report = build_dataset_quality_report(data)

    assert report["nan_count"] >= 1
    assert report["finite_fraction"] < 1.0
    assert report["duplicate_fraction"] > 0.0
    assert any("non-finite" in w for w in report["warnings"])
    assert any("duplicate" in w for w in report["warnings"])


def test_quality_report_split_counts_from_manifest(tmp_path):
    data = write_toy_contract_h5(tmp_path / "cloud2.h5", n=20)
    manifest = {"train_count": 14, "val_count": 4, "test_count": 2, "ood_count": 0}
    report = build_dataset_quality_report(data, split_manifest=manifest)
    assert report["split_counts"] == {"train": 14, "val": 4, "test": 2, "ood": 0}

    # No manifest -> empty mapping.
    report_none = build_dataset_quality_report(data, split_manifest=None)
    assert report_none["split_counts"] == {}
