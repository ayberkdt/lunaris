"""Multi-seed comparison for ST-LRPS final candidates.

Aggregates per-seed field-validation and orbit-benchmark numbers into a summary
table with mean / std / best / median / worst, so the paper does not rely on a
single lucky seed. When only one seed is available the limitation is stated
explicitly rather than hidden.
"""

from __future__ import annotations

import csv
import json
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

# Numeric per-seed metrics aggregated across seeds (lower is better for errors).
AGGREGATED_METRICS = (
    "best_val_epoch",
    "field_random_accel_rmse",
    "field_spatial_accel_rmse",
    "field_ood_low_accel_rmse",
    "field_ood_high_accel_rmse",
    "bench1day_median_rms_km",
    "bench1day_p95_rms_km",
    "bench1day_max_rms_km",
    "bench5day_median_rms_km",
    "bench5day_p95_rms_km",
    "bench5day_max_rms_km",
    "runtime_s",
)


def _to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    p = Path(path)
    if not p.exists() or p.stat().st_size <= 0:
        return []
    with p.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def collect_seed_entry(
    seed: int,
    *,
    field_metrics_csv: str | Path | None = None,
    benchmark_metrics: Mapping[str, str | Path] | None = None,
    artifact_hash: str | None = None,
    best_val_epoch: int | None = None,
    surrogate_model: str = "ST-LRPS",
) -> dict[str, Any]:
    """Build one per-seed entry from a seed's field + benchmark CSV outputs.

    ``benchmark_metrics`` maps a case key (``"1day"``/``"5day"``) to that case's
    ``metrics_summary.csv`` path. Missing inputs become ``None`` (not dropped).
    """
    entry: dict[str, Any] = {"seed": int(seed), "artifact_hash": artifact_hash, "best_val_epoch": best_val_epoch}

    # Field validation accel RMSE per split policy.
    policy_to_key = {
        "seeded_random": "field_random_accel_rmse",
        "spatial_block": "field_spatial_accel_rmse",
        "ood_low_altitude": "field_ood_low_accel_rmse",
        "ood_high_altitude": "field_ood_high_accel_rmse",
    }
    for key in policy_to_key.values():
        entry[key] = None
    if field_metrics_csv:
        for row in _read_csv(field_metrics_csv):
            key = policy_to_key.get(str(row.get("policy")))
            if key:
                entry[key] = _to_float(row.get("residual_accel_rmse_m_s2"))

    # Orbit benchmark position-error percentiles for the surrogate.
    for case in ("1day", "5day"):
        for stat in ("median", "p95", "max"):
            entry[f"bench{case}_{stat}_rms_km"] = None
    runtime = None
    for case, path in (benchmark_metrics or {}).items():
        rows = _read_csv(path)
        for row in rows:
            if str(row.get("model")) != surrogate_model:
                continue
            entry[f"bench{case}_median_rms_km"] = _to_float(row.get("median_rms_pos_err_km"))
            entry[f"bench{case}_p95_rms_km"] = _to_float(row.get("p95_rms_pos_err_km"))
            entry[f"bench{case}_max_rms_km"] = _to_float(row.get("max_rms_pos_err_km"))
            runtime = runtime or _to_float(row.get("runtime_s") or row.get("total_runtime_s"))
    entry["runtime_s"] = runtime
    return entry


def aggregate_multi_seed(
    entries: Sequence[Mapping[str, Any]],
    *,
    headline: str = "bench1day_median_rms_km",
) -> dict[str, Any]:
    """Aggregate per-seed entries into mean/std/best/median/worst statistics."""
    entries = [dict(e) for e in entries]
    n = len(entries)
    stats: dict[str, dict[str, float | None]] = {}
    for metric in AGGREGATED_METRICS:
        values = [v for v in (_to_float(e.get(metric)) for e in entries) if v is not None]
        if values:
            stats[metric] = {
                "mean": float(statistics.fmean(values)),
                "std": float(statistics.pstdev(values)) if len(values) > 1 else 0.0,
                "min": float(min(values)),
                "max": float(max(values)),
                "median": float(statistics.median(values)),
                "n": len(values),
            }
        else:
            stats[metric] = {"mean": None, "std": None, "min": None, "max": None, "median": None, "n": 0}

    # Best / median / worst seed by the headline metric (lower = better).
    scored = [(e, _to_float(e.get(headline))) for e in entries]
    scored = [(e, v) for e, v in scored if v is not None]
    scored.sort(key=lambda ev: ev[1])
    best_seed = scored[0][0].get("seed") if scored else None
    worst_seed = scored[-1][0].get("seed") if scored else None
    median_seed = scored[len(scored) // 2][0].get("seed") if scored else None

    single_seed = n <= 1
    return {
        "schema_version": 1,
        "n_seeds": n,
        "headline_metric": headline,
        "single_seed_limitation": single_seed,
        "single_seed_note": (
            "Only one seed is available; mean/std are not meaningful and results may "
            "reflect a single lucky (or unlucky) initialization. Train additional "
            "seeds before drawing seed-robust conclusions."
            if single_seed else None
        ),
        "best_seed": best_seed,
        "median_seed": median_seed,
        "worst_seed": worst_seed,
        "statistics": stats,
        "entries": entries,
    }


def write_multi_seed_outputs(summary: Mapping[str, Any], out_dir: str | Path) -> dict[str, Path]:
    """Write ``multi_seed_summary.csv`` + ``multi_seed_summary.md`` (+ JSON)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    entries = summary.get("entries", [])

    csv_path = out / "multi_seed_summary.csv"
    columns = ["seed", "artifact_hash", *AGGREGATED_METRICS]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for e in entries:
            writer.writerow([e.get(c) for c in columns])
        # mean / std rows over seeds.
        stats = summary.get("statistics", {})
        writer.writerow(["MEAN", ""] + [_fmt_stat(stats.get(m, {}).get("mean")) for m in AGGREGATED_METRICS])
        writer.writerow(["STD", ""] + [_fmt_stat(stats.get(m, {}).get("std")) for m in AGGREGATED_METRICS])

    json_path = out / "multi_seed_summary.json"
    json_path.write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")

    md_path = out / "multi_seed_summary.md"
    md_path.write_text(_render_summary(summary), encoding="utf-8")
    return {"csv": csv_path, "json": json_path, "summary": md_path}


def _fmt_stat(value: Any) -> str:
    return "" if value is None else f"{float(value):.6g}"


def _render_summary(summary: Mapping[str, Any]) -> str:
    lines = [
        "# ST-LRPS Multi-Seed Summary",
        "",
        f"- Seeds: {summary.get('n_seeds')}  |  headline metric: {summary.get('headline_metric')}",
        f"- Best seed: {summary.get('best_seed')}  |  median seed: {summary.get('median_seed')}  |  worst seed: {summary.get('worst_seed')}",
        "",
    ]
    if summary.get("single_seed_limitation"):
        lines += [f"> **Single-seed limitation:** {summary.get('single_seed_note')}", ""]
    lines += ["| metric | mean | std | min | median | max | n |", "|---|---|---|---|---|---|---|"]
    stats = summary.get("statistics", {})
    for metric in AGGREGATED_METRICS:
        s = stats.get(metric, {})
        lines.append(
            f"| {metric} | {_fmt_stat(s.get('mean'))} | {_fmt_stat(s.get('std'))} | "
            f"{_fmt_stat(s.get('min'))} | {_fmt_stat(s.get('median'))} | {_fmt_stat(s.get('max'))} | {s.get('n', 0)} |"
        )
    lines.append("")
    return "\n".join(lines)


__all__ = [
    "AGGREGATED_METRICS",
    "aggregate_multi_seed",
    "collect_seed_entry",
    "write_multi_seed_outputs",
]
