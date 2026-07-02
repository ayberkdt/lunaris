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
                "n_scenarios": 1,
                "total_runtime_s": 1.0,
                "runtime_per_scenario_s": 1.0,
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


def test_evidence_block_marks_synthetic_run(tmp_path):
    out = _valid_dir(tmp_path)
    (out / "resolved_config.json").write_text(
        json.dumps({"name": "smoke", "run_options": {"synthetic": True, "quick": True}}),
        encoding="utf-8",
    )
    report = validate_benchmark_outputs(out, expected_count=1)
    assert report["passed"] is True  # structurally valid, but not evidence
    evidence = report["evidence"]
    assert evidence["benchmark_name"] == "smoke"
    assert evidence["synthetic"] is True
    assert evidence["quick"] is True
    assert evidence["scientific_evidence"] is False
    assert "NOT A SCIENTIFIC BENCHMARK" in str(evidence["banner"])
    assert any("not scientific benchmark evidence" in w for w in report["warnings"])


def test_evidence_block_marks_real_run_as_evidence_candidate(tmp_path):
    out = _valid_dir(tmp_path)
    (out / "resolved_config.json").write_text(
        json.dumps({"name": "real", "run_options": {"synthetic": False, "quick": False}}),
        encoding="utf-8",
    )
    report = validate_benchmark_outputs(out, expected_count=1)
    assert report["passed"] is True
    evidence = report["evidence"]
    assert evidence["scientific_evidence"] is True
    assert evidence["banner"] is None


def test_evidence_block_fails_closed_without_resolved_config(tmp_path):
    out = _valid_dir(tmp_path)
    (out / "resolved_config.json").unlink()
    report = validate_benchmark_outputs(out, expected_count=1)
    assert report["passed"] is False  # required file missing
    evidence = report["evidence"]
    assert evidence["scientific_evidence"] is False
    assert "resolved config unavailable" in evidence["reason"]


def test_paper_safe_with_synthetic_run_options_is_error(tmp_path):
    out = _valid_dir(tmp_path)
    (out / "resolved_config.json").write_text(
        json.dumps({"name": "bad", "paper_safe": True, "run_options": {"synthetic": True}}),
        encoding="utf-8",
    )
    report = validate_benchmark_outputs(out, expected_count=1)
    assert report["passed"] is False
    assert any("paper_safe" in e and "synthetic" in e for e in report["errors"])


def test_evidence_block_prefers_explicit_resolved_config_arg(tmp_path):
    out = _valid_dir(tmp_path)  # file on disk says "{}"
    report = validate_benchmark_outputs(
        out,
        resolved_config={"name": "arg_wins", "run_options": {"synthetic": True}},
        expected_count=1,
    )
    assert report["evidence"]["benchmark_name"] == "arg_wins"
    assert report["evidence"]["synthetic"] is True
    assert report["evidence"]["scientific_evidence"] is False


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
    _write_csv(
        out / "runtime_summary.csv",
        [{"model": "SH20", "n_scenarios": 1, "total_runtime_s": -1, "runtime_per_scenario_s": -1, "n_steps": 10}],
    )
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


# --- Task 5: scenario-count / metadata validation -------------------------

def _scenario_row(sid, model="SH20"):
    return {
        "scenario_id": sid,
        "model": model,
        "rms_pos_err_km": 1.0,
        "radial_rms_km": 0.1,
        "along_rms_km": 0.2,
        "cross_rms_km": 0.3,
        "status": "ok",
        "domain_warning": "",
    }


def _build_dir(
    tmp_path,
    *,
    scenario_ids,
    models=("SH20",),
    runtime_n=None,
    total_runtime=1.0,
    runtime_per=None,
    report_count=None,
):
    out = _valid_dir(tmp_path)
    rn = runtime_n if runtime_n is not None else len(set(scenario_ids))
    rp = runtime_per if runtime_per is not None else (total_runtime / max(rn, 1))
    _write_csv(out / "scenario_results.csv", [_scenario_row(s, m) for m in models for s in scenario_ids])
    _write_csv(
        out / "metrics_summary.csv",
        [{"model": m, "median_rms_pos_err_km": 1, "p95_rms_pos_err_km": 2, "max_rms_pos_err_km": 3} for m in models],
    )
    _write_csv(
        out / "runtime_summary.csv",
        [
            {"model": m, "n_scenarios": rn, "total_runtime_s": total_runtime, "runtime_per_scenario_s": rp, "n_steps": 10}
            for m in models
        ],
    )
    if report_count is not None:
        (out / "report.md").write_text(f"# report\n- Scenario count: {report_count}\n", encoding="utf-8")
    return out


def test_contiguous_ids_pass(tmp_path):
    out = _build_dir(tmp_path, scenario_ids=[0, 1, 2])
    report = validate_benchmark_outputs(out, expected_count=3)
    assert report["passed"] is True, report["errors"]


def test_noncontiguous_external_ids_without_mapping_fail(tmp_path):
    # IDs 300..302 with expected_count=3 but no mapping -> must fail.
    out = _build_dir(tmp_path, scenario_ids=[300, 301, 302])
    report = validate_benchmark_outputs(out, expected_count=3)
    assert report["passed"] is False
    assert any("scenario_id min must be 0" in e or "contiguous" in e for e in report["errors"])


def test_noncontiguous_ids_allowed_with_policy_and_mapping(tmp_path):
    out = _build_dir(tmp_path, scenario_ids=[300, 301, 302])
    (out / "scenario_id_mapping.json").write_text(json.dumps({"300": 0, "301": 1, "302": 2}), encoding="utf-8")
    config = {"scenario_id_policy": "external_noncontiguous"}
    report = validate_benchmark_outputs(out, resolved_config=config, expected_count=3)
    assert report["passed"] is True, report["errors"]


def test_external_policy_without_mapping_fails(tmp_path):
    out = _build_dir(tmp_path, scenario_ids=[300, 301, 302])
    config = {"scenario_id_policy": "external_noncontiguous"}
    report = validate_benchmark_outputs(out, resolved_config=config, expected_count=3)
    assert report["passed"] is False
    assert any("scenario_id_mapping.json" in e for e in report["errors"])


def test_runtime_n_scenarios_mismatch_fails(tmp_path):
    out = _build_dir(tmp_path, scenario_ids=[0, 1, 2], runtime_n=5)
    report = validate_benchmark_outputs(out, expected_count=3)
    assert report["passed"] is False
    assert any("n_scenarios=5" in e for e in report["errors"])


def test_missing_n_scenarios_column_fails(tmp_path):
    out = _build_dir(tmp_path, scenario_ids=[0, 1, 2])
    _write_csv(out / "runtime_summary.csv", [{"model": "SH20", "total_runtime_s": 1.0, "n_steps": 10}])
    report = validate_benchmark_outputs(out, expected_count=3)
    assert report["passed"] is False
    assert any("n_scenarios column" in e for e in report["errors"])


def test_report_scenario_count_mismatch_fails(tmp_path):
    # report says 100, scenario_results has 3 -> must be caught.
    out = _build_dir(tmp_path, scenario_ids=[0, 1, 2], report_count=100)
    report = validate_benchmark_outputs(out, expected_count=3)
    assert report["passed"] is False
    assert any("report.md scenario count" in e for e in report["errors"])


def test_model_name_inconsistency_fails(tmp_path):
    out = _build_dir(tmp_path, scenario_ids=[0, 1, 2], models=("SH20",))
    # metrics references an extra model that scenario_results does not contain.
    _write_csv(
        out / "metrics_summary.csv",
        [
            {"model": "SH20", "median_rms_pos_err_km": 1, "p95_rms_pos_err_km": 2, "max_rms_pos_err_km": 3},
            {"model": "SH50", "median_rms_pos_err_km": 1, "p95_rms_pos_err_km": 2, "max_rms_pos_err_km": 3},
        ],
    )
    report = validate_benchmark_outputs(out, expected_count=3)
    assert report["passed"] is False
    assert any("model names differ" in e for e in report["errors"])


def test_per_model_scenario_count_mismatch_fails(tmp_path):
    # Two models but one only has 2 of the 3 expected scenarios.
    out = _valid_dir(tmp_path)
    rows = [_scenario_row(s, "SH20") for s in (0, 1, 2)] + [_scenario_row(s, "SH50") for s in (0, 1)]
    _write_csv(out / "scenario_results.csv", rows)
    _write_csv(
        out / "metrics_summary.csv",
        [
            {"model": "SH20", "median_rms_pos_err_km": 1, "p95_rms_pos_err_km": 2, "max_rms_pos_err_km": 3},
            {"model": "SH50", "median_rms_pos_err_km": 1, "p95_rms_pos_err_km": 2, "max_rms_pos_err_km": 3},
        ],
    )
    _write_csv(
        out / "runtime_summary.csv",
        [
            {"model": "SH20", "n_scenarios": 3, "total_runtime_s": 3.0, "runtime_per_scenario_s": 1.0, "n_steps": 10},
            {"model": "SH50", "n_scenarios": 3, "total_runtime_s": 3.0, "runtime_per_scenario_s": 1.0, "n_steps": 10},
        ],
    )
    report = validate_benchmark_outputs(out, expected_count=3)
    assert report["passed"] is False
    assert any("SH50 has 2 scenario rows" in e for e in report["errors"])


def test_runtime_per_scenario_inconsistency_fails(tmp_path):
    # total/n = 10/2 = 5.0 but runtime_per_scenario_s says 1.0.
    out = _build_dir(tmp_path, scenario_ids=[0, 1], total_runtime=10.0, runtime_per=1.0)
    report = validate_benchmark_outputs(out, expected_count=2)
    assert report["passed"] is False
    assert any("runtime_per_scenario_s" in e for e in report["errors"])
