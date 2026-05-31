from __future__ import annotations

import json
from pathlib import Path

from lunaris.surrogate.st_lrps.evaluation import compare_gravity_models as cgm
from lunaris.surrogate.st_lrps.evaluation.benchmark_validation import validate_benchmark_outputs


def test_config_cli_quick_synthetic_pipeline_writes_standard_outputs(tmp_path):
    config = Path("configs/benchmarks/st_lrps_1day_high_degree.json").resolve()
    out = tmp_path / "benchmark_run"
    rc = cgm.main(["--config", str(config), "--out", str(out), "--quick"])
    assert rc == 0
    for name in [
        "benchmark_manifest.json",
        "resolved_config.json",
        "metrics_summary.csv",
        "metrics_summary.json",
        "scenario_results.csv",
        "runtime_summary.csv",
        "validation_report.json",
        "report.md",
    ]:
        assert (out / name).exists(), name
    report = json.loads((out / "validation_report.json").read_text(encoding="utf-8"))
    assert report["passed"] is True


def test_validator_fails_for_corrupted_synthetic_output(tmp_path):
    config = Path("configs/benchmarks/st_lrps_1day_high_degree.json").resolve()
    out = tmp_path / "benchmark_run"
    assert cgm.main(["--config", str(config), "--out", str(out), "--quick"]) == 0
    text = (out / "metrics_summary.csv").read_text(encoding="utf-8")
    text = text.replace("max_rms_pos_err_km", "max_rms_pos_err_km", 1)
    lines = text.splitlines()
    header = lines[0].split(",")
    max_index = header.index("max_rms_pos_err_km")
    first = lines[1].split(",")
    first[max_index] = "0.0"
    lines[1] = ",".join(first)
    (out / "metrics_summary.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    report = validate_benchmark_outputs(out, expected_count=3)
    assert report["passed"] is False
    assert any("metric order failed" in e for e in report["errors"])
