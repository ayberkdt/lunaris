"""Schema-only tests for the benchmark evidence taxonomy (Phase 5).

These pin the taxonomy contract: which error category a metric name maps to,
and that a benchmark artifact cannot present orbit-level trajectory error as
ST-LRPS gravity-field accuracy.
"""

from __future__ import annotations

import pytest

from lunaris.surrogate.st_lrps.evaluation.benchmark_evidence_taxonomy import (
    INTEGRATOR_ERROR,
    MODEL_ERROR_FIELD,
    PHASE_CORRECTED_ERROR,
    RUNTIME_METRICS,
    TRAJECTORY_ERROR,
    classify_metric,
    field_evidence_error_for_paper_safe,
    summarize_evidence_taxonomy,
)


@pytest.mark.parametrize(
    "name,expected",
    [
        ("median_rms_pos_err_km", TRAJECTORY_ERROR),
        ("p95_rms_vel_err_ms", TRAJECTORY_ERROR),
        ("radial_rms_km", TRAJECTORY_ERROR),
        ("along_rms_km", TRAJECTORY_ERROR),
        ("model_error_rms_mgal", MODEL_ERROR_FIELD),
        ("field_error_max_mgal", MODEL_ERROR_FIELD),
        ("accel_error_rms", MODEL_ERROR_FIELD),
        ("integrator_error_rms_km", INTEGRATOR_ERROR),
        ("sh200_rk4_vs_dop853_rms_km", INTEGRATOR_ERROR),
        ("phase_corrected_rms_pos_err_km", PHASE_CORRECTED_ERROR),
        ("total_runtime_s", RUNTIME_METRICS),
        ("traj_steps_per_second", RUNTIME_METRICS),
        ("acceleration_evals_per_second", RUNTIME_METRICS),
    ],
)
def test_classify_metric_buckets_names(name, expected):
    assert classify_metric(name) == expected


def test_classify_metric_returns_none_for_non_metric():
    assert classify_metric("scenario_id") is None
    assert classify_metric("model") is None


def test_field_marker_wins_over_trajectory_marker():
    # A field metric that also mentions position must classify as field, not
    # trajectory (order of the classification rules).
    assert classify_metric("model_error_pos_field_rms") == MODEL_ERROR_FIELD


def test_summarize_flags_trajectory_only_run():
    tax = summarize_evidence_taxonomy(
        ["median_rms_pos_err_km", "radial_rms_km", "total_runtime_s"],
    )
    assert tax["has_field_level_evidence"] is False
    assert tax["trajectory_error_only"] is True
    assert tax["scientific_evidence"] is True
    assert tax["categories"][TRAJECTORY_ERROR]["metrics"]
    assert tax["categories"][MODEL_ERROR_FIELD]["metrics"] == []
    assert tax["categories"][MODEL_ERROR_FIELD]["proves_field_accuracy"] is True


def test_summarize_flags_field_evidence_present():
    tax = summarize_evidence_taxonomy(
        ["model_error_rms_mgal", "median_rms_pos_err_km", "total_runtime_s"],
    )
    assert tax["has_field_level_evidence"] is True
    assert tax["trajectory_error_only"] is False


def test_summarize_synthetic_is_not_scientific_evidence():
    tax = summarize_evidence_taxonomy(["model_error_rms_mgal"], synthetic=True)
    assert tax["scientific_evidence"] is False


def test_paper_safe_synthetic_is_error():
    tax = summarize_evidence_taxonomy(["median_rms_pos_err_km"], synthetic=True)
    err = field_evidence_error_for_paper_safe(tax, paper_safe=True)
    assert err is not None and "synthetic" in err


def test_paper_safe_trajectory_only_is_not_a_hard_error():
    # An honest trajectory benchmark is legitimate trajectory evidence; the
    # taxonomy labels it, it does not forbid it.
    tax = summarize_evidence_taxonomy(["median_rms_pos_err_km"], synthetic=False)
    assert field_evidence_error_for_paper_safe(tax, paper_safe=True) is None


def test_non_paper_safe_never_errors():
    tax = summarize_evidence_taxonomy(["median_rms_pos_err_km"], synthetic=True)
    assert field_evidence_error_for_paper_safe(tax, paper_safe=False) is None
