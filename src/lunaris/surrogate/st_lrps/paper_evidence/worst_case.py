"""Worst-case ST-LRPS orbit-scenario analysis.

Reads scenario-level orbit benchmark results (``scenario_results.csv``) and,
when available, the scenario definitions (``scenarios.csv``), and surfaces the
worst scenarios for the ST-LRPS candidate. Failures are documented honestly:
phase drift is distinguished from radial instability, and out-of-domain (OOD)
failures are labelled as such rather than hidden.

Trajectory-level RIC-vs-time plots require per-step error histories. When those
are not present in the benchmark output, the aggregate analysis is still
produced and the time-history plots are skipped with a note.
"""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

# Lunar GM for the orbital-period estimate (SI). Imported lazily-safe.
try:
    from lunaris.surrogate.st_lrps.data.dataset_parameters import MU_MOON_SI, R_MOON_SI
except Exception:  # pragma: no cover - defensive
    MU_MOON_SI = 4.902800066e12
    R_MOON_SI = 1737400.0

# Rankings: logical name -> scenario_results.csv column (lower is better).
WORST_BY = {
    "max_pos_err_km": "max_pos_err_km",
    "rms_pos_err_km": "rms_pos_err_km",
    "final_pos_err_km": "final_pos_err_km",
    "along_track_err_km": "along_rms_km",
    "radial_err_km": "radial_rms_km",
}


def _to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size <= 0:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _orbital_period_s(a_km: float | None) -> float | None:
    if a_km is None or a_km <= 0:
        return None
    a_m = a_km * 1000.0
    return float(2.0 * math.pi * math.sqrt(a_m ** 3 / MU_MOON_SI))


def _scenario_def_index(scenario_defs: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for row in scenario_defs:
        sid = str(row.get("scenario_id"))
        if sid:
            index[sid] = dict(row)
    return index


def analyze_worst_cases(
    scenario_rows: Sequence[Mapping[str, Any]],
    *,
    model: str = "ST-LRPS",
    scenario_defs: Sequence[Mapping[str, Any]] | None = None,
    train_alt_min_km: float | None = None,
    train_alt_max_km: float | None = None,
    top_n: int = 5,
) -> dict[str, Any]:
    """Rank the worst scenarios for ``model`` and flag their failure modes."""
    rows = [dict(r) for r in scenario_rows if str(r.get("model")) == model]
    defs = _scenario_def_index(scenario_defs or [])

    rankings: dict[str, list[dict[str, Any]]] = {}
    seen_scenarios: dict[str, dict[str, Any]] = {}
    for name, column in WORST_BY.items():
        scored = [(r, _to_float(r.get(column))) for r in rows]
        scored = [(r, v) for r, v in scored if v is not None]
        scored.sort(key=lambda rv: rv[1], reverse=True)
        ranked: list[dict[str, Any]] = []
        for r, value in scored[: int(top_n)]:
            record = _scenario_record(r, defs, train_alt_min_km, train_alt_max_km)
            record["ranked_by"] = name
            record["ranked_value"] = value
            ranked.append(record)
            seen_scenarios.setdefault(str(r.get("scenario_id")), record)
        rankings[name] = ranked

    return {
        "schema_version": 1,
        "model": model,
        "n_scenarios": len(rows),
        "top_n": int(top_n),
        "train_altitude_envelope_km": [train_alt_min_km, train_alt_max_km],
        "rankings": rankings,
        "worst_scenarios": list(seen_scenarios.values()),
        "has_scenario_defs": bool(defs),
    }


def _scenario_record(
    row: Mapping[str, Any],
    defs: Mapping[str, Mapping[str, Any]],
    train_alt_min_km: float | None,
    train_alt_max_km: float | None,
) -> dict[str, Any]:
    sid = str(row.get("scenario_id"))
    sdef = defs.get(sid, {})
    hp_km = _to_float(sdef.get("hp_km"))
    ha_km = _to_float(sdef.get("ha_km"))
    a_km = _to_float(sdef.get("a_km"))
    if a_km is None and hp_km is not None and ha_km is not None:
        a_km = float(R_MOON_SI) / 1000.0 + 0.5 * (hp_km + ha_km)

    radial = _to_float(row.get("radial_rms_km"))
    along = _to_float(row.get("along_rms_km"))
    cross = _to_float(row.get("cross_rms_km"))

    # Phase drift vs radial instability (from the RIC decomposition we have).
    phase_drift_dominated = bool(
        along is not None
        and along > 0
        and (radial is None or along >= 2.0 * radial)
        and (cross is None or along >= 2.0 * cross)
    )
    radial_dominated = bool(
        radial is not None and along is not None and radial > along and (cross is None or radial > cross)
    )

    domain_warning = str(row.get("domain_warning") or row.get("surrogate_domain_warning") or "").strip()
    leaves_envelope = None
    if train_alt_min_km is not None and train_alt_max_km is not None and (hp_km is not None or ha_km is not None):
        lo = hp_km if hp_km is not None else ha_km
        hi = ha_km if ha_km is not None else hp_km
        leaves_envelope = bool(lo < float(train_alt_min_km) - 1.0 or hi > float(train_alt_max_km) + 1.0)
    is_ood = bool(domain_warning) or bool(leaves_envelope)

    return {
        "scenario_id": _to_int(sid),
        "periapsis_altitude_km": hp_km,
        "apoapsis_altitude_km": ha_km,
        "altitude_range_km": [hp_km, ha_km],
        "semi_major_axis_km": a_km,
        "eccentricity": _to_float(sdef.get("e")),
        "inclination_deg": _to_float(sdef.get("inc_deg") or sdef.get("inclination_deg")),
        "orbital_period_s": _orbital_period_s(a_km),
        "rms_pos_err_km": _to_float(row.get("rms_pos_err_km")),
        "max_pos_err_km": _to_float(row.get("max_pos_err_km")),
        "final_pos_err_km": _to_float(row.get("final_pos_err_km")),
        "radial_rms_km": radial,
        "along_rms_km": along,
        "cross_rms_km": cross,
        "rms_vel_err_ms": _to_float(row.get("rms_vel_err_ms")),
        "phase_drift_dominated": phase_drift_dominated,
        "radial_dominated": radial_dominated,
        "radial_unbounded": None,  # requires the RIC time history; see note
        "domain_warning": domain_warning or None,
        "leaves_training_envelope": leaves_envelope,
        "is_ood": is_ood,
    }


def _to_int(value: Any) -> Any:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return value


_OUTPUT_COLUMNS = (
    "ranked_by", "ranked_value", "scenario_id", "periapsis_altitude_km", "apoapsis_altitude_km",
    "semi_major_axis_km", "eccentricity", "inclination_deg", "orbital_period_s",
    "rms_pos_err_km", "max_pos_err_km", "final_pos_err_km", "radial_rms_km", "along_rms_km",
    "cross_rms_km", "rms_vel_err_ms", "phase_drift_dominated", "radial_dominated", "radial_unbounded",
    "domain_warning", "leaves_training_envelope", "is_ood",
)


def write_worst_case_outputs(analysis: Mapping[str, Any], out_dir: str | Path) -> dict[str, Path]:
    """Write ``worst_case_scenarios.csv`` + ``worst_case_summary.md`` (+ JSON)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    csv_path = out / "worst_case_scenarios.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(_OUTPUT_COLUMNS)
        for _name, ranked in analysis.get("rankings", {}).items():
            for rec in ranked:
                writer.writerow([rec.get(col) for col in _OUTPUT_COLUMNS])

    json_path = out / "worst_case_analysis.json"
    json_path.write_text(json.dumps(analysis, indent=2, default=str) + "\n", encoding="utf-8")

    md_path = out / "worst_case_summary.md"
    md_path.write_text(_render_summary(analysis), encoding="utf-8")
    return {"csv": csv_path, "json": json_path, "summary": md_path}


def _render_summary(analysis: Mapping[str, Any]) -> str:
    model = analysis.get("model")
    lines = [
        f"# ST-LRPS Worst-Case Scenario Analysis — {model}",
        "",
        f"- Scenarios analyzed: {analysis.get('n_scenarios')}  |  top-N: {analysis.get('top_n')}",
        f"- Training altitude envelope (km): {analysis.get('train_altitude_envelope_km')}",
        "",
        "> Worst-case behavior is reported honestly. Phase drift (along-track) is "
        "distinguished from radial instability. Failures outside the training "
        "altitude envelope are labelled OOD; failures inside the envelope indicate "
        "a model/dataset improvement target.",
        "",
    ]
    if not analysis.get("has_scenario_defs"):
        lines += [
            "_Scenario definitions (altitude/inclination/eccentricity) were not "
            "available; geometric fields are blank. Provide scenarios.csv for full detail._",
            "",
        ]
    for name, ranked in analysis.get("rankings", {}).items():
        lines.append(f"## Worst {len(ranked)} by {name}")
        lines.append("")
        lines.append("| scenario | value | peri/apo alt km | RIC (r/a/c) km | mode | OOD |")
        lines.append("|---|---|---|---|---|---|")
        for rec in ranked:
            mode = "phase-drift" if rec.get("phase_drift_dominated") else ("radial" if rec.get("radial_dominated") else "mixed")
            ric = f"{_fmt(rec.get('radial_rms_km'))}/{_fmt(rec.get('along_rms_km'))}/{_fmt(rec.get('cross_rms_km'))}"
            alt = f"{_fmt(rec.get('periapsis_altitude_km'))}/{_fmt(rec.get('apoapsis_altitude_km'))}"
            lines.append(
                f"| {rec.get('scenario_id')} | {_fmt(rec.get('ranked_value'))} | {alt} | {ric} | {mode} | "
                f"{'yes' if rec.get('is_ood') else 'no'} |"
            )
        lines.append("")
    lines.append(
        "_radial_unbounded is left undetermined here because it requires the per-step "
        "RIC time history; run with trajectory histories to resolve it._"
    )
    lines.append("")
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.4g}"
    except (TypeError, ValueError):
        return str(value)


def run_worst_case_from_benchmark_dir(
    benchmark_dir: str | Path,
    out_dir: str | Path,
    *,
    model: str = "ST-LRPS",
    train_alt_min_km: float | None = None,
    train_alt_max_km: float | None = None,
    top_n: int = 5,
) -> dict[str, Path]:
    """Load a benchmark output dir and write the worst-case analysis."""
    bdir = Path(benchmark_dir)
    scenario_rows = _read_csv(bdir / "scenario_results.csv")
    if not scenario_rows:
        raise FileNotFoundError(f"no scenario_results.csv under {bdir}")
    scenario_defs = (
        _read_csv(bdir / "scenarios.csv")
        or _read_csv(bdir / "metrics" / "scenarios.csv")
    )
    analysis = analyze_worst_cases(
        scenario_rows,
        model=model,
        scenario_defs=scenario_defs,
        train_alt_min_km=train_alt_min_km,
        train_alt_max_km=train_alt_max_km,
        top_n=top_n,
    )
    return write_worst_case_outputs(analysis, out_dir)


__all__ = [
    "WORST_BY",
    "analyze_worst_cases",
    "run_worst_case_from_benchmark_dir",
    "write_worst_case_outputs",
]
