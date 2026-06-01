"""Run orchestration and file reporting for Perturbation Budget Analysis."""

from __future__ import annotations

import csv
import json
import math
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np

from .ablation import propagation_ablation_not_run
from .acceleration_budget import (
    compute_acceleration_budget,
    load_gravity_model_for_budget,
)
from .config import PerturbationBudgetConfig
from .gravity_degree_sensitivity import compute_gravity_degree_sensitivity
from .sampling import generate_sample_states
from .uncertainty_budget import compute_uncertainty_budget, recommend_gravity_degree_by_altitude


@dataclass(frozen=True, slots=True)
class PerturbationBudgetResult:
    output_dir: Path
    perturbation_budget_csv: Path
    gravity_degree_sensitivity_csv: Path
    force_model_uncertainty_budget_csv: Path
    recommended_gravity_degree_by_altitude_csv: Path
    propagation_ablation_csv: Path
    runtime_budget_csv: Path
    summary_md: Path
    warnings: List[str]


def _git_commit() -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if proc.returncode == 0:
            return proc.stdout.strip()
    except Exception:
        pass
    return "unavailable"


def _csv_value(value: object) -> object:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return ""
    if isinstance(value, np.floating):
        value_f = float(value)
        if math.isnan(value_f) or math.isinf(value_f):
            return ""
        return value_f
    return value


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(str(key))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key, "")) for key in fieldnames})


def _median_force_ranking(rows: Iterable[Mapping[str, object]]) -> List[tuple[str, float]]:
    values: Dict[str, List[float]] = {}
    for row in rows:
        name = str(row.get("force_name", ""))
        if name.startswith("Gravity SH"):
            continue
        try:
            val = float(row["acceleration_norm_m_s2"])
        except Exception:
            continue
        if np.isfinite(val):
            values.setdefault(name, []).append(val)
    ranking = [(name, float(np.median(vals))) for name, vals in values.items() if vals]
    return sorted(ranking, key=lambda item: item[1], reverse=True)


def _collect_warnings(
    budget_rows: Sequence[Mapping[str, object]],
    sh_rows: Sequence[Mapping[str, object]],
    gravity_warning: str,
) -> List[str]:
    warnings: List[str] = []
    if gravity_warning:
        warnings.append(gravity_warning)

    by_sample: Dict[str, Dict[str, float]] = {}
    for row in budget_rows:
        sid = str(row.get("sample_id", ""))
        name = str(row.get("force_name", ""))
        try:
            val = float(row.get("acceleration_norm_m_s2", 0.0))
        except Exception:
            continue
        if not np.isfinite(val):
            warnings.append(f"Non-finite acceleration for {name} in {sid}.")
            continue
        by_sample.setdefault(sid, {})[name] = val

    for sid, vals in by_sample.items():
        srp = vals.get("SRP", 0.0)
        central = vals.get("Central Lunar Gravity", 0.0)
        if srp > 0.0:
            if vals.get("Lunar Albedo", 0.0) > 5.0 * srp:
                warnings.append(f"Lunar albedo exceeds 5 x SRP in {sid}.")
            if vals.get("Thermal IR", 0.0) > 5.0 * srp:
                warnings.append(f"Thermal IR exceeds 5 x SRP in {sid}.")
            if vals.get("1PN Relativity", 0.0) > srp:
                warnings.append(f"1PN relativity exceeds SRP in {sid}; inspect geometry/config.")
        if central > 0.0 and vals.get("Solid Tides", 0.0) > 1.0e-3 * central:
            warnings.append(f"Solid tides exceed 1e-3 x central gravity in {sid}.")

    # Broad altitude trend: high-degree increment should generally decrease with altitude.
    band_alt: Dict[str, Dict[float, List[float]]] = {}
    for row in sh_rows:
        band = str(row.get("band", ""))
        alt = float(row.get("altitude_km", 0.0))
        val = float(row.get("increment_norm_m_s2", 0.0))
        if np.isfinite(val):
            band_alt.setdefault(band, {}).setdefault(alt, []).append(val)
    for band, alt_map in band_alt.items():
        ordered = sorted((alt, float(np.median(vals))) for alt, vals in alt_map.items() if vals)
        if len(ordered) >= 2 and ordered[-1][1] > 1.5 * ordered[0][1]:
            warnings.append(f"{band} median increases with altitude in broad trend; inspect gravity model/config.")

    return warnings


def _sh_uncertainty_comparison_rows(
    sh_rows: Sequence[Mapping[str, object]],
    uncertainty_rows: Sequence[Mapping[str, object]],
) -> List[Dict[str, object]]:
    combined_by_sample: Dict[str, float] = {}
    for row in uncertainty_rows:
        if row.get("model") == "Combined Non-Gravitational RSS":
            combined_by_sample[str(row["sample_id"])] = float(row["uncertainty_norm_m_s2"])
    rows: List[Dict[str, object]] = []
    for row in sh_rows:
        sample_id = str(row["sample_id"])
        combined = combined_by_sample.get(sample_id, float("nan"))
        inc = float(row["increment_norm_m_s2"])
        ratio = inc / combined if np.isfinite(combined) and combined > 0.0 else float("nan")
        rows.append(
            {
                "sample_id": sample_id,
                "altitude_km": row["altitude_km"],
                "inclination_deg": row["inclination_deg"],
                "true_anomaly_deg": row["true_anomaly_deg"],
                "epoch_utc": row["epoch_utc"],
                "model": str(row["band"]),
                "acceleration_norm_m_s2": inc,
                "relative_uncertainty": "",
                "uncertainty_norm_m_s2": combined,
                "combination": "sh_increment_vs_combined_non_grav_rss",
                "ratio_to_combined_non_grav_uncertainty": ratio,
            }
        )
    return rows


def _summary_markdown(
    config: PerturbationBudgetConfig,
    budget_rows: Sequence[Mapping[str, object]],
    recommendation_rows: Sequence[Mapping[str, object]],
    warnings: Sequence[str],
    *,
    gravity_source: str,
    runtime_s: float,
) -> str:
    ranking = _median_force_ranking(budget_rows)
    lines: List[str] = [
        "# Perturbation Budget Analysis",
        "",
        "This report compares acceleration contributions and first-order force-model uncertainties.",
        "It is not an electrical power analysis and it does not define a universal gravity-degree rule.",
        "",
        "## Configuration",
        "",
        f"- Gravity source: `{gravity_source}`",
        f"- Geometry source: `{'synthetic_geometry' if config.use_synthetic_geometry_fallback else 'ephemeris_requested'}`",
        f"- Git commit: `{_git_commit()}`",
        f"- Runtime: {runtime_s:.3f} s",
        f"- Altitudes km: {', '.join(str(v) for v in config.altitudes_km)}",
        f"- Inclinations deg: {', '.join(str(v) for v in config.inclinations_deg)}",
        f"- True anomalies deg: {', '.join(str(v) for v in config.true_anomalies_deg)}",
        f"- SH degrees: {', '.join(str(v) for v in config.sh_degrees)}",
        "",
        "## Force Magnitude Ranking",
        "",
    ]
    for name, median in ranking[:12]:
        lines.append(f"- {name}: median `{median:.3e}` m/s^2")

    lines.extend(["", "## Recommended Gravity Degree By Altitude", ""])
    if recommendation_rows:
        lines.append("| Altitude km | Quick | Medium | High | Reason |")
        lines.append("|---:|---:|---:|---:|---|")
        for row in recommendation_rows:
            reason = str(row.get("reason", "")).replace("|", ";")
            lines.append(
                "| {altitude_km:g} | {recommended_degree_quick} | {recommended_degree_medium} | "
                "{recommended_degree_high} | {reason} |".format(**{**row, "reason": reason})
            )

    lines.extend(
        [
            "",
            "## Interpretation Notes",
            "",
            "- Incremental SH bands are vector differences, e.g. `a_SH100 - a_SH60`.",
            "- Force magnitude and force-model uncertainty are different quantities.",
            "- A small instantaneous acceleration can still accumulate during propagation.",
            "- Recommendations are for this configuration only: altitude, geometry, gravity file, spacecraft area-to-mass ratio, force models, and thresholds all matter.",
            "",
            "## Limitations",
            "",
            "- Default runs use deterministic synthetic Sun/Earth geometry unless ephemeris-backed integration is added/configured.",
            "- If no gravity model file is supplied, synthetic coefficients are used only for smoke testing and workflow validation.",
            "- Propagation-level ablation and detailed runtime benchmarking are scaffolded but not run by the MVP.",
            "",
            "## Warnings",
            "",
        ]
    )
    if warnings:
        lines.extend(f"- {w}" for w in warnings)
    else:
        lines.append("- None.")
    lines.append("")
    return "\n".join(lines)


def run_perturbation_budget(config: PerturbationBudgetConfig) -> PerturbationBudgetResult:
    t0 = time.perf_counter()
    out = config.output_path
    out.mkdir(parents=True, exist_ok=True)

    samples = generate_sample_states(config)
    gravity_info = load_gravity_model_for_budget(config)
    budget_rows, sh_by_sample, forces_by_sample = compute_acceleration_budget(config, samples, gravity_info)
    sh_rows = compute_gravity_degree_sensitivity(config, samples, sh_by_sample, forces_by_sample)
    uncertainty_rows = compute_uncertainty_budget(config, samples, forces_by_sample)
    uncertainty_rows_with_comparisons = uncertainty_rows + _sh_uncertainty_comparison_rows(sh_rows, uncertainty_rows)
    recommendation_rows = recommend_gravity_degree_by_altitude(config, sh_rows, uncertainty_rows)
    runtime_s = time.perf_counter() - t0

    warnings = _collect_warnings(budget_rows, sh_rows, gravity_info.warning)

    paths = {
        "perturbation_budget": out / "perturbation_budget.csv",
        "gravity_degree_sensitivity": out / "gravity_degree_sensitivity.csv",
        "force_model_uncertainty_budget": out / "force_model_uncertainty_budget.csv",
        "recommended_gravity_degree_by_altitude": out / "recommended_gravity_degree_by_altitude.csv",
        "propagation_ablation": out / "propagation_ablation.csv",
        "runtime_budget": out / "runtime_budget.csv",
        "summary": out / "perturbation_budget_summary.md",
    }
    _write_csv(paths["perturbation_budget"], budget_rows)
    _write_csv(paths["gravity_degree_sensitivity"], sh_rows)
    _write_csv(paths["force_model_uncertainty_budget"], uncertainty_rows_with_comparisons)
    _write_csv(paths["recommended_gravity_degree_by_altitude"], recommendation_rows)
    _write_csv(paths["propagation_ablation"], propagation_ablation_not_run())
    _write_csv(
        paths["runtime_budget"],
        [
            {
                "stage": "instantaneous_budget",
                "sample_count": len(samples),
                "wall_time_s": runtime_s,
                "status": "completed",
            }
        ],
    )
    (out / "config.json").write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
    paths["summary"].write_text(
        _summary_markdown(
            config,
            budget_rows,
            recommendation_rows,
            warnings,
            gravity_source=gravity_info.source,
            runtime_s=runtime_s,
        ),
        encoding="utf-8",
    )

    return PerturbationBudgetResult(
        output_dir=out,
        perturbation_budget_csv=paths["perturbation_budget"],
        gravity_degree_sensitivity_csv=paths["gravity_degree_sensitivity"],
        force_model_uncertainty_budget_csv=paths["force_model_uncertainty_budget"],
        recommended_gravity_degree_by_altitude_csv=paths["recommended_gravity_degree_by_altitude"],
        propagation_ablation_csv=paths["propagation_ablation"],
        runtime_budget_csv=paths["runtime_budget"],
        summary_md=paths["summary"],
        warnings=warnings,
    )
