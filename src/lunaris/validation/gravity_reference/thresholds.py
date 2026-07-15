"""Acceptance-threshold classification."""

from __future__ import annotations

from typing import Any

PASS = "PASS"
FAIL = "FAIL"
REPORT_ONLY = "REPORT_ONLY"
INCOMPLETE_CONTRACT = "INCOMPLETE_CONTRACT"
SOURCE_HASH_MISMATCH = "SOURCE_HASH_MISMATCH"
REFERENCE_GENERATION_REQUIRED = "REFERENCE_GENERATION_REQUIRED"
UNSUPPORTED_FRAME = "UNSUPPORTED_FRAME"
UNSUPPORTED_TIME_SCALE = "UNSUPPORTED_TIME_SCALE"


def classify_field_metrics(metrics: dict[str, Any], comparison: dict[str, Any]) -> dict[str, Any]:
    """Classify field metrics using manifest-declared thresholds."""
    origin = str(comparison.get("threshold_origin", "report_only"))
    abs_tol = comparison.get("absolute_tolerances") or {}
    rel_tol = comparison.get("relative_tolerances") or {}
    checks: list[dict[str, Any]] = []

    def add_check(name: str, value: float, limit: float | None) -> None:
        if limit is None:
            return
        checks.append({
            "name": name,
            "value": float(value),
            "limit": float(limit),
            "passed": bool(float(value) <= float(limit)),
        })

    add_check(
        "potential_abs_error_m2_s2.max",
        metrics["potential_abs_error_m2_s2"]["max"],
        abs_tol.get("potential_m2_s2"),
    )
    add_check(
        "potential_rel_error.max",
        metrics["potential_rel_error"]["max"],
        rel_tol.get("potential"),
    )
    add_check(
        "acceleration_norm_error_m_s2.max",
        metrics["acceleration_norm_error_m_s2"]["max"],
        abs_tol.get("acceleration_norm_m_s2"),
    )
    add_check(
        "acceleration_rel_norm_error.max",
        metrics["acceleration_rel_norm_error"]["max"],
        rel_tol.get("acceleration_norm"),
    )
    add_check(
        "acceleration_component_abs_error_m_s2.max_any_component",
        metrics["acceleration_component_abs_error_m_s2"]["max_any_component"],
        abs_tol.get("acceleration_component_m_s2"),
    )

    if origin == "report_only" or not checks:
        status = REPORT_ONLY
    elif all(check["passed"] for check in checks):
        status = PASS
    else:
        status = FAIL

    return {
        "status": status,
        "threshold_origin": origin,
        "checks": checks,
    }


def classify_trajectory_metrics(metrics: dict[str, Any], comparison: dict[str, Any]) -> dict[str, Any]:
    """Classify trajectory metrics using manifest-declared thresholds.

    Checks are only added when the manifest declares the corresponding limit,
    so a manifest may legitimately gate on a subset (e.g. position + energy).
    """
    origin = str(comparison.get("threshold_origin", "report_only"))
    abs_tol = comparison.get("absolute_tolerances") or {}
    rel_tol = comparison.get("relative_tolerances") or {}
    checks: list[dict[str, Any]] = []

    def add_check(name: str, value: float, limit: float | None) -> None:
        if limit is None:
            return
        checks.append({
            "name": name,
            "value": float(value),
            "limit": float(limit),
            "passed": bool(float(value) <= float(limit)),
        })

    add_check("position_error_m.max", metrics["position_error_m"]["max"], abs_tol.get("position_m"))
    add_check("position_error_m.rms", metrics["position_error_m"]["rms"], abs_tol.get("position_rms_m"))
    add_check("velocity_error_m_s.max", metrics["velocity_error_m_s"]["max"], abs_tol.get("velocity_m_s"))
    add_check(
        "normalized_position_error.max_over_traveled_distance",
        metrics["normalized_position_error"]["max_over_traveled_distance"] or 0.0,
        rel_tol.get("position_over_traveled_distance"),
    )
    if "lunaris_energy_drift" in metrics:
        add_check(
            "lunaris_energy_drift.max_abs_relative_drift",
            metrics["lunaris_energy_drift"]["max_abs_relative_drift"],
            rel_tol.get("energy_relative_drift"),
        )

    if origin == "report_only" or not checks:
        status = REPORT_ONLY
    elif all(check["passed"] for check in checks):
        status = PASS
    else:
        status = FAIL

    return {
        "status": status,
        "threshold_origin": origin,
        "checks": checks,
    }

