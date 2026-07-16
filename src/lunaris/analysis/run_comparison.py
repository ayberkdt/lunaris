"""Typed, persisted comparison of two canonical mission-analysis packages."""

from __future__ import annotations

import csv
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from lunaris.analysis.artifacts import load_analysis_artifacts
from lunaris.analysis.contracts import MetricValue, OrbitAnalysisResult
from lunaris.common.hashing import canonical_json_text
from lunaris.common.provenance import utc_now_iso

COMPARISON_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class MetricComparison:
    metric_id: str
    label: str
    unit: str | None
    baseline_value: float | int | None
    candidate_value: float | int | None
    delta: float | None
    relative_delta: float | None
    comparable: bool
    reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RunComparison:
    baseline_run_id: str
    candidate_run_id: str
    generated_at_utc: str
    metrics: tuple[MetricComparison, ...]
    warnings: tuple[str, ...]
    baseline_config_sha256: str | None
    candidate_config_sha256: str | None
    schema_version: int = COMPARISON_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "comparison_schema_version": self.schema_version,
            "baseline_run_id": self.baseline_run_id,
            "candidate_run_id": self.candidate_run_id,
            "generated_at_utc": self.generated_at_utc,
            "baseline_config_sha256": self.baseline_config_sha256,
            "candidate_config_sha256": self.candidate_config_sha256,
            "metrics": [metric.to_dict() for metric in self.metrics],
            "warnings": list(self.warnings),
        }


def _numeric(metric: MetricValue | None) -> float | int | None:
    if metric is None or metric.value is None or isinstance(metric.value, bool):
        return None
    if not isinstance(metric.value, int | float):
        return None
    value = float(metric.value)
    return metric.value if math.isfinite(value) else None


def _compare_metric(
    metric_id: str,
    baseline: MetricValue | None,
    candidate: MetricValue | None,
) -> MetricComparison:
    label = candidate.label if candidate is not None else baseline.label if baseline else metric_id
    unit = candidate.unit if candidate is not None else baseline.unit if baseline else None
    baseline_value = _numeric(baseline)
    candidate_value = _numeric(candidate)
    reason: str | None = None
    comparable = True
    if baseline is None or candidate is None:
        comparable = False
        reason = "Metric is absent from one run."
    elif baseline.unit != candidate.unit:
        comparable = False
        reason = "Metric units differ between runs."
    elif baseline_value is None or candidate_value is None:
        comparable = False
        reason = "Metric is unavailable or non-numeric in one run."

    delta: float | None = None
    relative_delta: float | None = None
    if comparable and baseline_value is not None and candidate_value is not None:
        delta = float(candidate_value) - float(baseline_value)
        if float(baseline_value) != 0.0:
            relative_delta = delta / abs(float(baseline_value))
    return MetricComparison(
        metric_id=metric_id,
        label=label,
        unit=unit,
        baseline_value=baseline_value,
        candidate_value=candidate_value,
        delta=delta,
        relative_delta=relative_delta,
        comparable=comparable,
        reason=reason,
    )


def build_run_comparison(
    baseline: OrbitAnalysisResult,
    candidate: OrbitAnalysisResult,
) -> RunComparison:
    """Compare shared typed metrics without rerunning propagation."""
    baseline_metrics = baseline.metric_map
    candidate_metrics = candidate.metric_map
    metric_ids = sorted(set(baseline_metrics) | set(candidate_metrics))
    metrics = tuple(
        _compare_metric(
            metric_id,
            baseline_metrics.get(metric_id),
            candidate_metrics.get(metric_id),
        )
        for metric_id in metric_ids
    )
    warnings = list(baseline.warnings) + list(candidate.warnings)
    if baseline.frame != candidate.frame:
        warnings.append(
            f"Frame mismatch: baseline={baseline.frame}, candidate={candidate.frame}."
        )
    if baseline.time_system != candidate.time_system:
        warnings.append("Time-system mismatch between runs.")
    return RunComparison(
        baseline_run_id=baseline.run_id,
        candidate_run_id=candidate.run_id,
        generated_at_utc=utc_now_iso(),
        metrics=metrics,
        warnings=tuple(dict.fromkeys(warnings)),
        baseline_config_sha256=str(baseline.provenance.get("config_sha256") or "") or None,
        candidate_config_sha256=str(candidate.provenance.get("config_sha256") or "") or None,
    )


def _slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", text).strip("_") or "run"


def _markdown(comparison: RunComparison) -> str:
    lines = [
        "# Lunaris Run Comparison",
        "",
        f"- Baseline: `{comparison.baseline_run_id}`",
        f"- Candidate: `{comparison.candidate_run_id}`",
        f"- Generated: `{comparison.generated_at_utc}`",
        "",
        "> Deltas are descriptive candidate-minus-baseline values; they are not statistical significance claims.",
        "",
        "| Metric | Baseline | Candidate | Delta | Relative delta | Unit | Status |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for item in comparison.metrics:
        def render(value: float | int | None) -> str:
            if value is None:
                return "Unavailable"
            return f"{float(value):.8g}"

        relative = "Unavailable" if item.relative_delta is None else f"{item.relative_delta:.5%}"
        status = "comparable" if item.comparable else (item.reason or "unavailable")
        lines.append(
            f"| {item.label} | {render(item.baseline_value)} | {render(item.candidate_value)} | "
            f"{render(item.delta)} | {relative} | {item.unit or ''} | {status} |"
        )
    if comparison.warnings:
        lines.extend(["", "## Warnings", "", *(f"- {warning}" for warning in comparison.warnings)])
    return "\n".join(lines) + "\n"


def compare_run_packages(
    baseline_run_dir: str | Path,
    candidate_run_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
) -> dict[str, str]:
    """Load two canonical packages and persist JSON/CSV/Markdown comparison artifacts."""
    baseline_path = Path(baseline_run_dir)
    candidate_path = Path(candidate_run_dir)
    comparison = build_run_comparison(
        load_analysis_artifacts(baseline_path),
        load_analysis_artifacts(candidate_path),
    )
    root = (
        Path(output_dir)
        if output_dir is not None
        else baseline_path / "comparisons" / _slug(comparison.candidate_run_id)
    )
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / "comparison.json"
    csv_path = root / "comparison.csv"
    markdown_path = root / "comparison.md"
    json_path.write_text(canonical_json_text(comparison.to_dict()), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(MetricComparison.__dataclass_fields__)
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metric.to_dict() for metric in comparison.metrics)
    markdown_path.write_text(_markdown(comparison), encoding="utf-8")
    return {
        "comparison_json": str(json_path),
        "comparison_csv": str(csv_path),
        "comparison_markdown": str(markdown_path),
    }


__all__ = [
    "COMPARISON_SCHEMA_VERSION",
    "MetricComparison",
    "RunComparison",
    "build_run_comparison",
    "compare_run_packages",
]
