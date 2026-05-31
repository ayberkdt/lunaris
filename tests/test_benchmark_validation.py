from __future__ import annotations

import csv
import json
from pathlib import Path

from lunaris.surrogate.st_lrps.evaluation.benchmark_validation import validate_benchmark_outputs


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _valid_dir(tmp_path: Path) -> Path:
    out = tmp_path / "run"
    out.mkdir()
    (out / "figures").mkdir()
    (out / "benchmark_manifest.json").write_text("{}", encoding="utf-8")
    (out / "resolved_config.json").write_text("{}", encoding="utf-8")
    (out / "report.md").write_text("# report\n", encoding="utf-8")
    _write_csv(
        out / "metrics_summary.csv",
        [
            {
                "model": "SH20",
                "median_rms_pos_err_km": 1.0,
                "p95_rms_pos_err_km": 2.0,
                "max_rms_pos_err_km": 3.0,
            }
        ],
    )
    _write_csv(
        out / "scenario_results.csv",
        [
            {
                "scenario_id": 0,
                "model": "SH20",
                "rms_pos_err_km": 1.0,
                "radial_rms_km": 0.1,
                "along_rms_km": 0.2,
                "cross_rms_km": 0.3,
                "status": "ok",
                "domain_warning": "",
            }
        ],
    )
    _write_csv(
        out / "runtime_summary.csv",
        [
            {
                "model": "SH20",
                "total_runtime_s": 1.0,
                "n_steps": 10,
            }
        ],
    )
    (out / "metrics_summary.json").write_text(
        json.dumps({"units": {"distance": "km", "time": "s"}, "rows": []}),
        encoding="utf-8",
    )
    return out


def test_valid_metrics_pass(tmp_path):
    report = validate_benchmark_outputs(_valid_dir(tmp_path), expected_count=1)
    assert report["passed"] is True


def test_nan_metric_fails(tmp_path):
    out = _valid_dir(tmp_path)
    _write_csv(out / "metrics_summary.csv", [{"model": "SH20", "median_rms_pos_err_km": "nan", "p95_rms_pos_err_km": 2, "max_rms_pos_err_km": 3}])
    report = validate_benchmark_outputs(out, expected_count=1)
    assert report["passed"] is False
    assert any("not finite" in e for e in report["errors"])


def test_p95_greater_than_max_fails(tmp_path):
    out = _valid_dir(tmp_path)
    _write_csv(out / "metrics_summary.csv", [{"model": "SH20", "median_rms_pos_err_km": 1, "p95_rms_pos_err_km": 4, "max_rms_pos_err_km": 3}])
    report = validate_benchmark_outputs(out, expected_count=1)
    assert report["passed"] is False
    assert any("metric order failed" in e for e in report["errors"])


def test_missing_file_fails(tmp_path):
    out = _valid_dir(tmp_path)
    (out / "runtime_summary.csv").unlink()
    report = validate_benchmark_outputs(out, expected_count=1)
    assert report["passed"] is False
    assert any("missing required output file" in e for e in report["errors"])


def test_negative_runtime_fails(tmp_path):
    out = _valid_dir(tmp_path)
    _write_csv(out / "runtime_summary.csv", [{"model": "SH20", "total_runtime_s": -1, "n_steps": 10}])
    report = validate_benchmark_outputs(out, expected_count=1)
    assert report["passed"] is False
    assert any("runtime_summary.csv" in e for e in report["errors"])


def test_duplicate_model_name_fails(tmp_path):
    out = _valid_dir(tmp_path)
    _write_csv(
        out / "metrics_summary.csv",
        [
            {"model": "SH20", "median_rms_pos_err_km": 1, "p95_rms_pos_err_km": 2, "max_rms_pos_err_km": 3},
            {"model": "SH20", "median_rms_pos_err_km": 1, "p95_rms_pos_err_km": 2, "max_rms_pos_err_km": 3},
        ],
    )
    report = validate_benchmark_outputs(out, expected_count=1)
    assert report["passed"] is False
    assert any("duplicate model names" in e for e in report["errors"])


def test_warning_only_cases_remain_pass(tmp_path):
    out = _valid_dir(tmp_path)
    _write_csv(
        out / "scenario_results.csv",
        [
            {
                "scenario_id": 0,
                "model": "SH20",
                "rms_pos_err_km": 1,
                "radial_rms_km": 0.1,
                "along_rms_km": 0.2,
                "cross_rms_km": 0.3,
                "status": "ok",
                "domain_warning": "outside training envelope",
            }
        ],
    )
    report = validate_benchmark_outputs(out, expected_count=1)
    assert report["passed"] is True
    assert report["warnings"]
