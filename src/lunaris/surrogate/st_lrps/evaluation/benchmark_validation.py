# -*- coding: utf-8 -*-
"""Validation checks for benchmark output artifacts."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping

from .benchmark_config import canonical_json_text


REQUIRED_OUTPUT_FILES = (
    "benchmark_manifest.json",
    "resolved_config.json",
    "metrics_summary.csv",
    "metrics_summary.json",
    "scenario_results.csv",
    "runtime_summary.csv",
    "report.md",
)

RIC_COLUMNS = ("radial_rms_km", "along_rms_km", "cross_rms_km")


def validate_benchmark_outputs(
    out_dir: str | Path,
    *,
    resolved_config: Mapping[str, Any] | None = None,
    expected_count: int | None = None,
    write_report: bool = True,
) -> dict[str, Any]:
    """Validate benchmark outputs and optionally write validation_report.json."""

    root = Path(out_dir)
    errors: list[str] = []
    warnings: list[str] = []
    checked_files: list[str] = []
    checked_metrics: list[str] = []

    for rel in REQUIRED_OUTPUT_FILES:
        path = root / rel
        checked_files.append(str(path))
        if not path.exists():
            errors.append(f"missing required output file: {rel}")
        elif path.is_file() and path.stat().st_size <= 0:
            errors.append(f"required output file is empty: {rel}")

    figures_dir = root / "figures"
    checked_files.append(str(figures_dir))
    if not figures_dir.exists() or not figures_dir.is_dir():
        errors.append("missing required figures/ directory")

    metrics_rows = _read_csv(root / "metrics_summary.csv", errors)
    scenario_rows = _read_csv(root / "scenario_results.csv", errors)
    runtime_rows = _read_csv(root / "runtime_summary.csv", errors)
    metrics_json = _read_json(root / "metrics_summary.json", errors)
    manifest_json = _read_json(root / "benchmark_manifest.json", errors)

    _check_no_nan_inf("metrics_summary.csv", metrics_rows, errors, checked_metrics)
    _check_no_nan_inf("scenario_results.csv", scenario_rows, errors, checked_metrics)
    _check_no_nan_inf("runtime_summary.csv", runtime_rows, errors, checked_metrics)
    _check_metric_order(metrics_rows, errors, checked_metrics)
    _check_scenario_count(scenario_rows, expected_count, errors, checked_metrics)
    _check_positive_runtime(runtime_rows, errors, checked_metrics)
    _check_positive_steps(runtime_rows, errors, checked_metrics)
    _check_unique_model_names(metrics_rows, errors, checked_metrics)
    _check_truth_baseline_duplication(resolved_config, errors, checked_metrics)
    _check_domain_warnings(scenario_rows, warnings)
    _check_ric_columns(scenario_rows, resolved_config, errors, checked_metrics)
    _check_units(metrics_json, errors, checked_metrics)
    _include_manifest_contract_findings(manifest_json, errors, warnings, checked_metrics)

    report = {
        "passed": not errors,
        "errors": errors,
        "warnings": warnings,
        "checked_files": checked_files,
        "checked_metrics": sorted(set(checked_metrics)),
    }
    if write_report:
        path = root / "validation_report.json"
        path.write_text(canonical_json_text(report), encoding="utf-8")
    return report


def _read_csv(path: Path, errors: list[str]) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size <= 0:
        return []
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    except Exception as exc:
        errors.append(f"could not read CSV {path.name}: {exc}")
        return []


def _read_json(path: Path, errors: list[str]) -> Any:
    if not path.exists() or path.stat().st_size <= 0:
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        errors.append(f"could not read JSON {path.name}: {exc}")
        return None


def _check_no_nan_inf(
    label: str,
    rows: list[dict[str, str]],
    errors: list[str],
    checked: list[str],
) -> None:
    for row_index, row in enumerate(rows):
        for key, value in row.items():
            number = _to_float(value)
            if number is None:
                continue
            checked.append(f"{label}:{key}:finite")
            if not math.isfinite(number):
                errors.append(f"{label} row {row_index} column {key} is not finite")


def _check_metric_order(
    rows: list[dict[str, str]],
    errors: list[str],
    checked: list[str],
) -> None:
    prefixes: set[str] = set()
    for row in rows:
        for key in row:
            if key.startswith("median_"):
                prefixes.add(key[len("median_") :])
    for row in rows:
        model = row.get("model", "<unknown>")
        for suffix in prefixes:
            med = _to_float(row.get(f"median_{suffix}"))
            p95 = _to_float(row.get(f"p95_{suffix}"))
            maxv = _to_float(row.get(f"max_{suffix}"))
            if med is None or p95 is None or maxv is None:
                continue
            checked.append(f"order:{suffix}")
            if not (maxv >= p95 >= med):
                errors.append(
                    f"metric order failed for {model} {suffix}: max={maxv}, p95={p95}, median={med}"
                )


def _check_scenario_count(
    rows: list[dict[str, str]],
    expected_count: int | None,
    errors: list[str],
    checked: list[str],
) -> None:
    if expected_count is None:
        return
    ids = {row.get("scenario_id") for row in rows if row.get("scenario_id") not in {None, ""}}
    checked.append("scenario_count")
    if len(ids) != int(expected_count):
        errors.append(f"scenario count mismatch: expected {expected_count}, observed {len(ids)}")


def _check_positive_runtime(
    rows: list[dict[str, str]],
    errors: list[str],
    checked: list[str],
) -> None:
    for index, row in enumerate(rows):
        for key, value in row.items():
            if "runtime" not in key.lower() or not key.lower().endswith("_s"):
                continue
            number = _to_float(value)
            if number is None:
                continue
            checked.append(f"runtime_positive:{key}")
            if number <= 0:
                errors.append(f"runtime_summary.csv row {index} column {key} must be positive")


def _check_positive_steps(
    rows: list[dict[str, str]],
    errors: list[str],
    checked: list[str],
) -> None:
    for index, row in enumerate(rows):
        for key, value in row.items():
            key_lower = key.lower()
            if key_lower not in {"n_steps", "step_count", "steps"}:
                continue
            number = _to_float(value)
            if number is None:
                continue
            checked.append(f"steps_positive:{key}")
            if number <= 0:
                errors.append(f"runtime_summary.csv row {index} column {key} must be positive")


def _check_unique_model_names(
    rows: list[dict[str, str]],
    errors: list[str],
    checked: list[str],
) -> None:
    names = [row.get("model", "") for row in rows if row.get("model")]
    checked.append("unique_model_names")
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        errors.append(f"duplicate model names in metrics_summary.csv: {', '.join(duplicates)}")


def _check_truth_baseline_duplication(
    config: Mapping[str, Any] | None,
    errors: list[str],
    checked: list[str],
) -> None:
    if not config:
        return
    checked.append("truth_not_duplicated_as_baseline")
    truth = config.get("truth", {}) if isinstance(config.get("truth"), Mapping) else {}
    truth_key = (truth.get("model"), truth.get("degree"))
    allow = bool(config.get("allow_truth_baseline", False))
    validation = config.get("validation")
    if isinstance(validation, Mapping):
        allow = allow or bool(validation.get("allow_truth_baseline", False))
    for baseline in config.get("baselines", []):
        if not isinstance(baseline, Mapping):
            continue
        if (baseline.get("model"), baseline.get("degree")) == truth_key and not (
            allow or bool(baseline.get("allow_truth_duplicate", False))
        ):
            errors.append(
                f"baseline {baseline.get('name')} duplicates the truth model without explicit allowance"
            )


def _check_domain_warnings(rows: list[dict[str, str]], warnings: list[str]) -> None:
    for row in rows:
        warning = (row.get("domain_warning") or row.get("surrogate_domain_warning") or "").strip()
        if warning:
            warnings.append(f"scenario {row.get('scenario_id', '?')} {row.get('model', '?')}: {warning}")


def _check_ric_columns(
    rows: list[dict[str, str]],
    config: Mapping[str, Any] | None,
    errors: list[str],
    checked: list[str],
) -> None:
    require_ric = True
    if config:
        metrics = config.get("metrics")
        if isinstance(metrics, Mapping) and metrics.get("ric") is False:
            require_ric = False
    if not require_ric or not rows:
        return
    checked.append("ric_columns_present")
    missing = [col for col in RIC_COLUMNS if col not in rows[0]]
    if missing:
        errors.append(f"RIC metrics requested but missing columns: {', '.join(missing)}")


def _check_units(metrics_json: Any, errors: list[str], checked: list[str]) -> None:
    checked.append("metric_units_present")
    if not isinstance(metrics_json, Mapping):
        errors.append("metrics_summary.json must be a JSON object with units")
        return
    units = metrics_json.get("units")
    if not isinstance(units, Mapping):
        errors.append("metrics_summary.json missing units mapping")
        return
    required_units = {
        "distance": {"km", "m"},
        "time": {"s", "seconds"},
    }
    for key, allowed in required_units.items():
        value = str(units.get(key, "")).lower()
        if value not in allowed:
            errors.append(f"metrics_summary.json units.{key} must be one of {sorted(allowed)}")


def _include_manifest_contract_findings(
    manifest_json: Any,
    errors: list[str],
    warnings: list[str],
    checked: list[str],
) -> None:
    if not isinstance(manifest_json, Mapping):
        return
    report = manifest_json.get("contract_compatibility")
    if not isinstance(report, Mapping):
        return
    checked.append("artifact_contract_compatibility")
    for message in report.get("warnings", []) or []:
        warnings.append(str(message))
    for message in report.get("errors", []) or []:
        errors.append("contract compatibility: " + str(message))


def _to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
