# -*- coding: utf-8 -*-
"""Validation checks for benchmark output artifacts."""

from __future__ import annotations

import csv
import json
import math
import re
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
    _check_scenario_ids(root, scenario_rows, expected_count, resolved_config, errors, checked_metrics)
    _check_per_model_scenario_counts(scenario_rows, expected_count, errors, checked_metrics)
    _check_runtime_scenarios(runtime_rows, expected_count, errors, checked_metrics)
    _check_model_name_consistency(metrics_rows, scenario_rows, runtime_rows, errors, checked_metrics)
    _check_report_scenario_count(root, expected_count, errors, checked_metrics)
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


def _to_int(value: Any) -> int | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return int(f) if float(f).is_integer() else None


def _scenario_id_policy(config: Mapping[str, Any] | None) -> str:
    if not config:
        return "contiguous"
    policy = config.get("scenario_id_policy")
    if not policy and isinstance(config.get("scenario"), Mapping):
        policy = config["scenario"].get("scenario_id_policy")
    return str(policy or "contiguous").strip().lower()


def _check_scenario_ids(
    root: Path,
    rows: list[dict[str, str]],
    expected_count: int | None,
    config: Mapping[str, Any] | None,
    errors: list[str],
    checked: list[str],
) -> None:
    """Scenario IDs must be integers and (for generated benchmarks) the exact
    contiguous range ``0..expected_count-1``. Non-contiguous external IDs are
    only allowed with an explicit policy + mapping file."""
    if expected_count is None or not rows:
        return
    checked.append("scenario_ids")
    raw_ids = [row.get("scenario_id") for row in rows if row.get("scenario_id") not in {None, ""}]
    int_ids = [_to_int(v) for v in raw_ids]
    if any(v is None for v in int_ids):
        bad = [raw for raw, parsed in zip(raw_ids, int_ids) if parsed is None]
        errors.append(f"scenario_results.csv has non-integer scenario_id values, e.g. {bad[:5]}")
        return
    unique = sorted(set(int_ids))

    policy = _scenario_id_policy(config)
    if policy == "external_noncontiguous":
        checked.append("scenario_id_mapping")
        if not (root / "scenario_id_mapping.json").exists():
            errors.append(
                "scenario_id_policy='external_noncontiguous' requires scenario_id_mapping.json"
            )
        if len(unique) != int(expected_count):
            errors.append(
                f"scenario count mismatch: expected {expected_count}, observed {len(unique)} unique ids"
            )
        return

    # Standard generated benchmark: IDs must be exactly 0..expected_count-1.
    if len(unique) != int(expected_count):
        errors.append(
            f"scenario count mismatch: expected {expected_count}, observed {len(unique)} unique ids"
        )
    if unique and unique[0] != 0:
        errors.append(f"scenario_id min must be 0, got {unique[0]}")
    if unique and unique[-1] != int(expected_count) - 1:
        errors.append(f"scenario_id max must be {int(expected_count) - 1}, got {unique[-1]}")
    expected_set = set(range(int(expected_count)))
    missing = sorted(expected_set - set(unique))
    extra = sorted(set(unique) - expected_set)
    if missing:
        errors.append(
            f"scenario_ids are not contiguous; missing {missing[:10]}"
            + (" ..." if len(missing) > 10 else "")
            + ". Use scenario_id_policy='external_noncontiguous' with a mapping if intentional."
        )
    if extra:
        errors.append(f"scenario_ids outside 0..{int(expected_count) - 1}: {extra[:10]}")


def _check_per_model_scenario_counts(
    rows: list[dict[str, str]],
    expected_count: int | None,
    errors: list[str],
    checked: list[str],
) -> None:
    if expected_count is None or not rows:
        return
    checked.append("per_model_scenario_count")
    by_model: dict[str, set[Any]] = {}
    for row in rows:
        model = row.get("model")
        if model:
            by_model.setdefault(model, set()).add(row.get("scenario_id"))
    for model, ids in sorted(by_model.items()):
        if len(ids) != int(expected_count):
            errors.append(
                f"model {model} has {len(ids)} scenario rows in scenario_results.csv, "
                f"expected {expected_count}"
            )


def _check_runtime_scenarios(
    rows: list[dict[str, str]],
    expected_count: int | None,
    errors: list[str],
    checked: list[str],
) -> None:
    """runtime_summary.csv must carry n_scenarios per model, equal to the expected
    count, and total_runtime_s/n_scenarios must match runtime_per_scenario_s."""
    if not rows:
        return
    checked.append("runtime_n_scenarios")
    if "n_scenarios" not in rows[0]:
        errors.append("runtime_summary.csv is missing the required n_scenarios column")
        return
    for index, row in enumerate(rows):
        model = row.get("model", f"<row {index}>")
        n = _to_int(row.get("n_scenarios"))
        if n is None:
            errors.append(f"runtime_summary.csv row {index} ({model}) n_scenarios is not an integer")
            continue
        if expected_count is not None and n != int(expected_count):
            errors.append(
                f"runtime_summary.csv row {index} ({model}) n_scenarios={n} != expected {expected_count}"
            )
        total = _to_float(row.get("total_runtime_s"))
        per = _to_float(row.get("runtime_per_scenario_s"))
        if total is not None and per is not None and n and n > 0:
            expected_per = total / n
            tol = max(1e-9, 1e-2 * abs(expected_per))
            if abs(expected_per - per) > tol:
                errors.append(
                    f"runtime_summary.csv row {index} ({model}): total_runtime_s/n_scenarios="
                    f"{expected_per:.6g} != runtime_per_scenario_s={per:.6g}"
                )


def _check_model_name_consistency(
    metrics_rows: list[dict[str, str]],
    scenario_rows: list[dict[str, str]],
    runtime_rows: list[dict[str, str]],
    errors: list[str],
    checked: list[str],
) -> None:
    if not metrics_rows or not scenario_rows:
        return
    checked.append("model_name_consistency")
    metrics_models = {r.get("model") for r in metrics_rows if r.get("model")}
    scenario_models = {r.get("model") for r in scenario_rows if r.get("model")}
    if metrics_models != scenario_models:
        errors.append(
            "model names differ between metrics_summary.csv and scenario_results.csv "
            f"(metrics-only={sorted(metrics_models - scenario_models)}, "
            f"scenario-only={sorted(scenario_models - metrics_models)})"
        )
    if runtime_rows:
        runtime_models = {r.get("model") for r in runtime_rows if r.get("model")}
        if runtime_models and runtime_models != metrics_models:
            errors.append(
                "model names differ between runtime_summary.csv and metrics_summary.csv "
                f"(runtime-only={sorted(runtime_models - metrics_models)}, "
                f"metrics-only={sorted(metrics_models - runtime_models)})"
            )


def _check_report_scenario_count(
    root: Path,
    expected_count: int | None,
    errors: list[str],
    checked: list[str],
) -> None:
    """If report.md states a scenario count, it must match the validated count
    (guards against stale config text in the report header)."""
    if expected_count is None:
        return
    report = root / "report.md"
    if not report.exists():
        return
    match = re.search(r"Scenario count:\s*(\d+)", report.read_text(encoding="utf-8"))
    if match is None:
        return
    checked.append("report_scenario_count")
    if int(match.group(1)) != int(expected_count):
        errors.append(
            f"report.md scenario count {match.group(1)} does not match validated count {expected_count}"
        )


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
