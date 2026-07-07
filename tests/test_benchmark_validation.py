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


def _scenario_contract_fields() -> dict:
    return {
        "trajectory_rms_km": 1.0,
        "final_pos_err_km": 1.1,
        "max_pos_err_km": 1.5,
        "p95_pos_err_km": 1.3,
        "rms_vel_err_ms": 0.1,
        "final_vel_err_ms": 0.12,
        "energy_drift_rel": 1.0e-8,
        "accel_max_error_m_s2": 1.0e-12,
        "potential_error_m2_s2": 1.0e-6,
        "impact_count": 0,
        "domain_exit_count": 0,
    }


def _runtime_contract_fields(total_runtime: float = 1.0) -> dict:
    return {
        "cold_time_s": total_runtime + 0.1,
        "warm_time_s": total_runtime,
        "jit_compile_time_s": 0.1,
        "propagation_time_s": total_runtime,
        "acceleration_evaluations_per_second": 40.0,
        "propagated_seconds_per_wall_second": 864.0,
    }


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
                "phase_lag_final_s": -0.4,
                "phase_lag_slope_s_per_day": -0.08,
                "phase_corrected_rms_km": 0.05,
                "phase_explained_fraction": 0.9,
                **_scenario_contract_fields(),
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
                **_runtime_contract_fields(1.0),
            }
        ],
    )
    (out / "metrics_summary.json").write_text(
        json.dumps(
            {
                "units": {
                    "distance": "km",
                    "time": "s",
                    "acceleration": "m/s^2",
                    "potential": "m^2/s^2",
                    "energy_drift": "relative",
                },
                "rows": [],
            }
        ),
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


def test_paper_safe_with_identity_frame_mode_is_error(tmp_path):
    """A paper-safe run whose manifest records an identity/inertial frame mode
    (Moon-fixed gravity evaluated in inertial coordinates) fails closed."""
    out = _valid_dir(tmp_path)
    (out / "resolved_config.json").write_text(
        json.dumps({"name": "bad_frame", "paper_safe": True}),
        encoding="utf-8",
    )
    (out / "benchmark_manifest.json").write_text(
        json.dumps({"numerics": {"frame_mode": "identity_diagnostic"}}),
        encoding="utf-8",
    )
    report = validate_benchmark_outputs(out, expected_count=1)
    assert report["passed"] is False
    assert any("frame_mode" in e and "identity rotation" in e for e in report["errors"])


def test_paper_safe_with_rotating_frame_mode_passes(tmp_path):
    out = _valid_dir(tmp_path)
    (out / "resolved_config.json").write_text(
        json.dumps({"name": "good_frame", "paper_safe": True}),
        encoding="utf-8",
    )
    (out / "benchmark_manifest.json").write_text(
        json.dumps({"numerics": {"frame_mode": "moon_fixed_ephemeris"}}),
        encoding="utf-8",
    )
    report = validate_benchmark_outputs(out, expected_count=1)
    assert not any("frame_mode" in e for e in report["errors"])
    assert "paper_safe_frame_mode" in report["checked_metrics"]


def test_validation_report_includes_evidence_taxonomy(tmp_path):
    """The report self-describes its metric columns by evidence category so a
    trajectory-only run cannot be read as ST-LRPS field accuracy."""
    out = _valid_dir(tmp_path)  # metrics are trajectory error only
    report = validate_benchmark_outputs(out, expected_count=1)
    tax = report["evidence_taxonomy"]
    assert tax["has_field_level_evidence"] is False
    assert tax["trajectory_error_only"] is True
    assert any("trajectory error only" in w for w in report["warnings"])
    assert "evidence_taxonomy_field_vs_trajectory" in report["checked_metrics"]


def _paper_safe_manifest(frame_mode: str = "moon_fixed_ephemeris") -> str:
    """A manifest with the numerics provenance a real paper-safe run records."""
    return json.dumps(
        {
            "numerics": {
                "frame_mode": frame_mode,
                "dtype": "float64",
                "dtype_provenance": {
                    "requested": "float64",
                    "gpu_st_lrps_potential": {
                        "requested": "float64",
                        "effective": "float64",
                        "downgraded": False,
                    },
                },
            }
        }
    )


def test_paper_safe_explicit_trajectory_only_run_passes(tmp_path):
    out = _valid_dir(tmp_path)  # metrics are trajectory error only
    (out / "resolved_config.json").write_text(
        json.dumps(
            {
                "name": "traj_paper",
                "paper_safe": True,
                "paper_safe_claim_type": "trajectory_only",
            }
        ),
        encoding="utf-8",
    )
    (out / "benchmark_manifest.json").write_text(_paper_safe_manifest(), encoding="utf-8")
    report = validate_benchmark_outputs(out, expected_count=1)
    # An explicit trajectory-only paper-safe run is labeled + warned, not failed.
    assert not any("field" in e.lower() for e in report["errors"]), report["errors"]
    assert report["evidence_taxonomy"]["trajectory_error_only"] is True
    assert report["passed"] is True


def test_paper_safe_default_claim_fails_trajectory_only_run(tmp_path):
    """No claim_type -> strict default (full_surrogate_validation): a trajectory-
    only paper-safe run is rejected because it carries no field evidence."""
    out = _valid_dir(tmp_path)
    (out / "resolved_config.json").write_text(
        json.dumps({"name": "traj_default", "paper_safe": True}),
        encoding="utf-8",
    )
    (out / "benchmark_manifest.json").write_text(_paper_safe_manifest(), encoding="utf-8")
    report = validate_benchmark_outputs(out, expected_count=1)
    assert report["passed"] is False
    assert any("model_error_field" in e or "field accuracy" in e for e in report["errors"])
    assert report["evidence"]["paper_safe_claim_type"] == "full_surrogate_validation"


def test_error_decomposition_block_present_in_report(tmp_path):
    out = _valid_dir(tmp_path)
    (out / "benchmark_manifest.json").write_text(_paper_safe_manifest(), encoding="utf-8")
    report = validate_benchmark_outputs(out, expected_count=1)
    block = report["error_decomposition"]
    assert block["schema_version"] == "st_lrps_error_decomposition_v1"
    for name in ("field_error", "orbit_error", "integrator_error",
                 "phase_corrected_error", "runtime", "provenance"):
        assert name in block
    # Trajectory columns are present; field columns absent for this fixture.
    assert block["orbit_error"]["present"] is True
    assert block["runtime"]["present"] is True
    assert block["field_error"]["present"] is False
    assert block["provenance"]["frame_mode"] == "moon_fixed_ephemeris"
    assert block["provenance"]["effective_dtype"] == "float64"


def test_paper_safe_error_decomposition_requires_provenance(tmp_path):
    """A paper-safe run whose manifest lacks backend/dtype provenance fails the
    error-decomposition contract even for a trajectory_only claim."""
    out = _valid_dir(tmp_path)
    (out / "resolved_config.json").write_text(
        json.dumps(
            {
                "name": "traj_incomplete",
                "paper_safe": True,
                "paper_safe_claim_type": "trajectory_only",
            }
        ),
        encoding="utf-8",
    )
    (out / "benchmark_manifest.json").write_text(
        json.dumps({"numerics": {"frame_mode": "moon_fixed_ephemeris"}}),
        encoding="utf-8",
    )
    report = validate_benchmark_outputs(out, expected_count=1)
    assert report["passed"] is False
    assert any(
        "error_decomposition" in e and "dtype" in e for e in report["errors"]
    ), report["errors"]


def test_paper_safe_error_decomposition_rejects_identity_frame(tmp_path):
    out = _valid_dir(tmp_path)
    (out / "resolved_config.json").write_text(
        json.dumps(
            {
                "name": "identity_ed",
                "paper_safe": True,
                "paper_safe_claim_type": "trajectory_only",
            }
        ),
        encoding="utf-8",
    )
    (out / "benchmark_manifest.json").write_text(
        _paper_safe_manifest(frame_mode="identity_diagnostic"), encoding="utf-8"
    )
    report = validate_benchmark_outputs(out, expected_count=1)
    assert report["passed"] is False
    assert any(
        "error_decomposition" in e and "identity" in e for e in report["errors"]
    ), report["errors"]


def test_non_paper_safe_identity_frame_mode_is_not_error(tmp_path):
    """Identity frame mode is a legitimate explicit diagnostic mode; it is only
    forbidden under paper_safe, not in general."""
    out = _valid_dir(tmp_path)
    (out / "resolved_config.json").write_text(
        json.dumps({"name": "diag", "run_options": {"synthetic": False}}),
        encoding="utf-8",
    )
    (out / "benchmark_manifest.json").write_text(
        json.dumps({"numerics": {"frame_mode": "identity_diagnostic"}}),
        encoding="utf-8",
    )
    report = validate_benchmark_outputs(out, expected_count=1)
    assert not any("frame_mode" in e for e in report["errors"])


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


def test_missing_phase_columns_fail(tmp_path):
    out = _valid_dir(tmp_path)
    row = _scenario_row(0)
    for col in ("phase_lag_final_s", "phase_lag_slope_s_per_day",
                "phase_corrected_rms_km", "phase_explained_fraction"):
        row.pop(col)
    _write_csv(out / "scenario_results.csv", [row])
    report = validate_benchmark_outputs(out, expected_count=1)
    assert report["passed"] is False
    assert any("phase-drift metrics requested but missing columns" in e for e in report["errors"])


def test_missing_phase_columns_allowed_when_config_disables_them(tmp_path):
    # Pre-phase-diagnostics outputs stay validatable via an explicit opt-out.
    out = _valid_dir(tmp_path)
    row = _scenario_row(0)
    for col in ("phase_lag_final_s", "phase_lag_slope_s_per_day",
                "phase_corrected_rms_km", "phase_explained_fraction"):
        row.pop(col)
    _write_csv(out / "scenario_results.csv", [row])
    config = {"metrics": {"phase": False}}
    report = validate_benchmark_outputs(out, resolved_config=config, expected_count=1)
    assert report["passed"] is True, report["errors"]


def test_missing_extended_validation_columns_fail(tmp_path):
    out = _valid_dir(tmp_path)
    row = _scenario_row(0)
    row.pop("energy_drift_rel")
    _write_csv(out / "scenario_results.csv", [row])
    report = validate_benchmark_outputs(out, expected_count=1)
    assert report["passed"] is False
    assert any("extended validation metrics requested" in e for e in report["errors"])


def test_missing_runtime_timing_columns_fail(tmp_path):
    out = _valid_dir(tmp_path)
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
    report = validate_benchmark_outputs(out, expected_count=1)
    assert report["passed"] is False
    assert any("runtime timing protocol requested" in e for e in report["errors"])


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
                "phase_lag_final_s": -0.4,
                "phase_lag_slope_s_per_day": -0.08,
                "phase_corrected_rms_km": 0.05,
                "phase_explained_fraction": 0.9,
                **_scenario_contract_fields(),
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
        "phase_lag_final_s": -0.4,
        "phase_lag_slope_s_per_day": -0.08,
        "phase_corrected_rms_km": 0.05,
        "phase_explained_fraction": 0.9,
        **_scenario_contract_fields(),
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
            {
                "model": m,
                "n_scenarios": rn,
                "total_runtime_s": total_runtime,
                "runtime_per_scenario_s": rp,
                "n_steps": 10,
                **_runtime_contract_fields(total_runtime),
            }
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
