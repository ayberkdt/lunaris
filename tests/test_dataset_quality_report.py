from __future__ import annotations

import json

from lunaris.surrogate.st_lrps.data.quality_report import build_dataset_quality_report

from dataset_pipeline_test_utils import write_toy_contract_h5


def test_dataset_quality_report_writes_json_and_markdown(tmp_path):
    data_path = write_toy_contract_h5(tmp_path / "toy.h5", n=30)
    out_dir = tmp_path / "quality"

    report = build_dataset_quality_report(
        data_path,
        out_dir=out_dir,
        bins=5,
        split_manifest={"train_count": 20, "val_count": 10},
    )

    assert report["n_samples"] == 30
    assert report["finite_fraction"] == 1.0
    assert len(report["altitude_histogram"]["counts"]) == 5
    assert report["split_counts"] == {"train": 20, "val": 10, "test": 0, "ood": 0}
    assert report["source_gravity_file_sha256"] == "a" * 64

    written = json.loads((out_dir / "dataset_quality_report.json").read_text(encoding="utf-8"))
    assert written["n_samples"] == 30
    summary = (out_dir / "dataset_quality_summary.md").read_text(encoding="utf-8")
    assert "Dataset Quality Summary" in summary
