"""Machine-readable and Markdown artifacts for the canonical run directory."""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from lunaris.analysis.contracts import (
    AnalysisEvent,
    ForceContribution,
    MetricValue,
    OrbitAnalysisResult,
    OrbitSeries,
)
from lunaris.common.hashing import canonical_json_text
from lunaris.common.provenance import sha256_file, utc_now_iso


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json_text(payload), encoding="utf-8")


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: "" if isinstance(value, float) and not math.isfinite(value) else value
                    for key, value in row.items()
                }
            )


def _event_row(event: AnalysisEvent) -> dict[str, Any]:
    state: list[float | None] = list(event.state_m_mps or ())
    state += [None] * (6 - len(state))
    return {
        "event_id": event.event_id,
        "event_type": event.event_type,
        "simulation_time_s": event.simulation_time_s,
        "epoch_utc": event.epoch_utc,
        "x_m": state[0],
        "y_m": state[1],
        "z_m": state[2],
        "vx_m_s": state[3],
        "vy_m_s": state[4],
        "vz_m_s": state[5],
        "altitude_m": event.altitude_m,
        "frame": event.frame,
        "source": event.source,
        "severity": event.severity,
        "note": event.note,
    }


def _orbital_rows(result: OrbitAnalysisResult) -> Iterable[dict[str, Any]]:
    s = result.series
    for index in range(int(s.t_s.size)):
        eccentricity = float(s.eccentricity[index])
        inclination = float(s.inclination_rad[index])
        circular = eccentricity < 1.0e-8
        equatorial = abs(math.sin(inclination)) < 1.0e-8
        state = s.state_m_mps[index]
        yield {
            "simulation_time_s": float(s.t_s[index]),
            "x_m": float(state[0]),
            "y_m": float(state[1]),
            "z_m": float(state[2]),
            "vx_m_s": float(state[3]),
            "vy_m_s": float(state[4]),
            "vz_m_s": float(state[5]),
            "semi_major_axis_m": float(s.semi_major_axis_m[index]),
            "eccentricity": eccentricity,
            "inclination_rad": inclination,
            "raan_rad": "" if equatorial else float(s.raan_rad[index]),
            "raan_status": "undefined_equatorial" if equatorial else "ok",
            "argument_of_periapsis_rad": (
                "" if circular else float(s.argument_of_periapsis_rad[index])
            ),
            "argument_of_periapsis_status": "undefined_circular" if circular else "ok",
            "true_anomaly_rad": float(s.true_anomaly_rad[index]),
            "altitude_m": float(s.altitude_m[index]),
            "radius_m": float(s.radius_m[index]),
            "speed_m_s": float(s.speed_m_s[index]),
            "specific_energy_j_kg": float(s.specific_energy_j_kg[index]),
            "angular_momentum_m2_s": float(s.angular_momentum_m2_s[index]),
            "eccentricity_vector_norm": float(s.eccentricity_vector_norm[index]),
            "eclipse": (
                "" if s.eclipse_mask is None else bool(s.eclipse_mask[index])
            ),
            "latitude_rad": (
                "" if s.latitude_rad is None else float(s.latitude_rad[index])
            ),
            "longitude_rad": (
                "" if s.longitude_rad is None else float(s.longitude_rad[index])
            ),
            "frame": result.frame,
            "time_system": "simulation elapsed time",
        }


def _force_row(item: ForceContribution) -> dict[str, Any]:
    return item.to_dict()


def _force_time_rows(result: OrbitAnalysisResult) -> Iterable[dict[str, Any]]:
    if result.force_time_s is None:
        return ()
    rows: list[dict[str, Any]] = []
    names = sorted(result.force_magnitudes_m_s2)
    for index, time_value in enumerate(np.asarray(result.force_time_s, dtype=np.float64)):
        row: dict[str, Any] = {"simulation_time_s": float(time_value)}
        for name in names:
            value = float(result.force_magnitudes_m_s2[name][index])
            row[name] = value if math.isfinite(value) else ""
        for name, values in sorted(result.force_vectors_m_s2.items()):
            for axis, suffix in enumerate(("x", "y", "z")):
                value = float(values[index, axis])
                row[f"{name}::inertial_{suffix}_m_s2"] = value if math.isfinite(value) else ""
        for name, values in sorted(result.force_ric_m_s2.items()):
            for axis, suffix in enumerate(("r", "i", "c")):
                value = float(values[index, axis])
                row[f"{name}::ric_{suffix}_m_s2"] = value if math.isfinite(value) else ""
        rows.append(row)
    return rows


def _flatten(value: Any, parent: str = "") -> Iterable[tuple[str, Any]]:
    if isinstance(value, Mapping):
        for key, item in value.items():
            path = f"{parent}.{key}" if parent else str(key)
            yield from _flatten(item, path)
    elif isinstance(value, list | tuple):
        yield parent, json.dumps(value, ensure_ascii=True, default=str)
    else:
        yield parent, value


def _display_metric(metric: MetricValue) -> str:
    if metric.value is None:
        return f"Unavailable ({metric.availability_reason})"
    value = metric.value
    if not isinstance(value, int | float) or isinstance(value, bool):
        return str(value)
    number = float(value)
    if metric.unit == "m" and any(token in metric.metric_id for token in ("altitude", "orbit.a")):
        return f"{number / 1000.0:,.3f} km"
    if metric.unit == "rad":
        return f"{math.degrees(number):,.5f} deg"
    if metric.unit == "s" and metric.metric_id in {"mission.duration", "orbit.period"}:
        if number >= 86_400.0:
            return f"{number / 86_400.0:,.4f} d"
        if number >= 3600.0:
            return f"{number / 3600.0:,.3f} h"
        if number >= 60.0:
            return f"{number / 60.0:,.3f} min"
    if metric.unit == "1" and isinstance(value, int):
        return f"{value:,}"
    if abs(number) != 0.0 and (abs(number) >= 1.0e5 or abs(number) < 1.0e-3):
        rendered = f"{number:.5e}"
    else:
        rendered = f"{number:,.6g}"
    unit = "" if metric.unit == "1" else (metric.unit or "")
    return f"{rendered} {unit}".strip()


def _metric_table(metrics: Sequence[MetricValue]) -> list[str]:
    lines = ["| Metric | Value | Status | Source |", "|---|---:|---|---|"]
    for metric in metrics:
        source = metric.source.replace("|", ";")
        value = _display_metric(metric).replace("|", ";")
        lines.append(f"| {metric.label} | {value} | {metric.status} | {source} |")
    return lines


def build_report_markdown(result: OrbitAnalysisResult) -> str:
    """Build the human-readable report from the same typed metric objects."""

    metrics = result.metric_map

    def selected(prefixes: tuple[str, ...]) -> list[MetricValue]:
        return [metric for metric in result.metrics if metric.metric_id.startswith(prefixes)]

    lines = [
        f"# Lunaris Mission Analysis - {result.run_id}",
        "",
        f"- Generated: `{result.generated_at_utc}`",
        f"- Preset: `{result.preset}`",
        f"- Frame: `{result.frame}`",
        f"- Time system: `{result.time_system}`",
        "",
        "## 1. Executive Mission Summary",
        "",
    ]
    summary_ids = (
        "run.status",
        "mission.start_epoch",
        "mission.duration",
        "orbit.altitude.minimum",
        "orbit.altitude.maximum",
        "orbit.period",
        "orbit.completed_count",
        "numerical.integrator",
        "numerical.integration_backend",
        "physics.rhs_path.effective",
        "physics.gravity_degree",
        "physics.active_force_models",
        "numerical.wall_time",
        "numerical.termination_reason",
    )
    lines.extend(_metric_table([metrics[item] for item in summary_ids if item in metrics]))
    lines.extend(
        [
            "",
            "### Initial / final orbit",
            "",
            "| Quantity | Initial | Final |",
            "|---|---:|---:|",
        ]
    )
    for prefix, label in (
        ("orbit.altitude", "Altitude"),
        ("orbit.a", "Semi-major axis"),
        ("orbit.e", "Eccentricity"),
        ("orbit.i", "Inclination"),
        ("orbit.raan", "RAAN"),
        ("orbit.argp", "Argument of periapsis"),
    ):
        initial = metrics.get(f"{prefix}.initial")
        final = metrics.get(f"{prefix}.final")
        if initial and final:
            lines.append(f"| {label} | {_display_metric(initial)} | {_display_metric(final)} |")

    lines.extend(["", "## 2. Mission Configuration", ""])
    lines.extend(_metric_table(selected(("physics.",))))
    lines.extend(["", "Configuration fields and asset hashes are preserved in `config.json` and `provenance.json`."])

    lines.extend(["", "## 3. Orbit Geometry and Evolution", ""])
    lines.extend(_metric_table(selected(("orbit.",))))
    lines.extend(
        [
            "",
            "Figures: [orbit overview](figures/orbit_overview.png), "
            "[altitude history](figures/altitude_history.png), "
            "[orbital elements](figures/orbital_elements.png), and "
            "[ground track](figures/groundtrack.png).",
        ]
    )

    lines.extend(["", "## 4. Critical Extrema and Events", ""])
    lines.extend(
        [
            "| Time [s] | Epoch [UTC] | Event | State summary | Altitude [km] | Frame | Severity | Source |",
            "|---:|---|---|---|---:|---|---|---|",
        ]
    )
    for event in result.events:
        altitude = "Unavailable" if event.altitude_m is None else f"{event.altitude_m / 1000.0:.3f}"
        if event.state_m_mps is None:
            state_summary = "Unavailable"
        else:
            state = np.asarray(event.state_m_mps, dtype=np.float64)
            state_summary = (
                f"norm(r)={np.linalg.norm(state[:3]) / 1000.0:.3f} km; "
                f"norm(v)={np.linalg.norm(state[3:6]) / 1000.0:.6f} km/s"
            )
        lines.append(
            f"| {event.simulation_time_s:.3f} | {event.epoch_utc or 'Unavailable'} | "
            f"{event.event_type} | {state_summary} | {altitude} | {event.frame} | "
            f"{event.severity} | {event.source.replace('|', ';')} |"
        )

    lines.extend(["", "## 5. Numerical Health", ""])
    lines.extend(_metric_table(selected(("numerical.",))))

    lines.extend(["", "## 6. Physical and Invariant Diagnostics", ""])
    lines.extend(_metric_table(selected(("diagnostic.",))))
    diagnostic_notes = sorted(
        {
            metric.interpretation
            for metric in result.metrics
            if metric.metric_id.startswith("diagnostic.") and metric.interpretation
        }
    )
    lines.extend(["", *(f"> {note}" for note in diagnostic_notes)])

    lines.extend(["", "## 7. Perturbation Budget", ""])
    lines.extend(
        [
            "| Component | Active | Minimum [m/s^2] | Median [m/s^2] | P95 [m/s^2] | Maximum [m/s^2] | Interpretation |",
            "|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for item in result.force_contributions:
        if item.available:
            minimum = item.minimum_m_s2
            median = item.median_m_s2
            p95 = item.p95_m_s2
            maximum = item.maximum_m_s2
            assert minimum is not None and median is not None and p95 is not None and maximum is not None
            values = [
                f"{minimum:.5e}",
                f"{median:.5e}",
                f"{p95:.5e}",
                f"{maximum:.5e}",
            ]
        else:
            values = ["Unavailable"] * 4
        note = (item.interpretation or item.availability_reason or "").replace("|", ";")
        lines.append(
            f"| {item.label} | {'yes' if item.active else 'no'} | "
            f"{values[0]} | {values[1]} | {values[2]} | {values[3]} | {note} |"
        )
    lines.append("")
    total_ric = result.force_ric_m_s2.get("Total non-central acceleration")
    if total_ric is not None:
        lines.extend(
            [
                "Signed total non-central acceleration in the local RIC frame:",
                "",
                "| Axis | Minimum [m/s^2] | Median [m/s^2] | Maximum [m/s^2] |",
                "|---|---:|---:|---:|",
            ]
        )
        for axis, label in enumerate(("Radial", "In-track", "Cross-track")):
            finite = total_ric[:, axis][np.isfinite(total_ric[:, axis])]
            if finite.size:
                lines.append(
                    f"| {label} | {np.min(finite):.5e} | {np.median(finite):.5e} | "
                    f"{np.max(finite):.5e} |"
                )
    else:
        lines.append(
            "RIC component statistics require engine force vectors. Magnitudes are never "
            "subtracted or summed as vectors."
        )

    lines.extend(["", "## 8. Provenance and Reproducibility", ""])
    lines.extend(["| Field | Value |", "|---|---|"])
    for path, value in _flatten(result.provenance):
        lines.append(f"| `{path}` | `{str(value).replace('|', ';')}` |")

    lines.extend(["", "## 9. Warnings and Limitations", ""])
    if result.warnings:
        lines.extend(f"- {warning}" for warning in result.warnings)
    else:
        lines.append("- No analysis warnings were recorded.")
    lines.extend(
        [
            "",
            "## Appendix A. Metric Definitions",
            "",
            "| Metric ID | Kind | Unit | Frame | Interpretation / availability |",
            "|---|---|---|---|---|",
        ]
    )
    for metric in result.metrics:
        note = metric.interpretation or metric.availability_reason or ""
        lines.append(
            f"| `{metric.metric_id}` | {metric.kind} | {metric.unit or '-'} | "
            f"{metric.frame or '-'} | {note.replace('|', ';')} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_analysis_artifacts(result: OrbitAnalysisResult, run_dir: str | Path) -> dict[str, Path]:
    """Write canonical JSON/CSV/Markdown artifacts (PDF and figures are separate)."""

    root = Path(run_dir)
    root.mkdir(parents=True, exist_ok=True)
    tables = root / "tables"
    figures = root / "figures"
    tables.mkdir(exist_ok=True)
    figures.mkdir(exist_ok=True)

    paths = {
        "metrics": root / "metrics.json",
        "provenance": root / "provenance.json",
        "config": root / "config.json",
        "diagnostics": root / "diagnostics.json",
        "events": root / "events.csv",
        "orbital_elements": root / "orbital_elements.csv",
        "force_budget": root / "force_budget.csv",
        "force_budget_timeseries": root / "force_budget_timeseries.csv",
        "report_markdown": root / "report.md",
        "summary_table": tables / "summary.csv",
        "extrema_table": tables / "extrema.csv",
        "integrator_table": tables / "integrator.csv",
        "provenance_table": tables / "provenance.csv",
    }
    _write_json(paths["metrics"], result.metrics_payload())
    _write_json(paths["provenance"], result.provenance)
    _write_json(paths["config"], result.config_snapshot)
    _write_json(paths["diagnostics"], result.diagnostics_snapshot)

    event_fields = [
        "event_id",
        "event_type",
        "simulation_time_s",
        "epoch_utc",
        "x_m",
        "y_m",
        "z_m",
        "vx_m_s",
        "vy_m_s",
        "vz_m_s",
        "altitude_m",
        "frame",
        "source",
        "severity",
        "note",
    ]
    _write_csv(paths["events"], event_fields, (_event_row(event) for event in result.events))
    orbital_fields = [
        "simulation_time_s",
        "x_m",
        "y_m",
        "z_m",
        "vx_m_s",
        "vy_m_s",
        "vz_m_s",
        "semi_major_axis_m",
        "eccentricity",
        "inclination_rad",
        "raan_rad",
        "raan_status",
        "argument_of_periapsis_rad",
        "argument_of_periapsis_status",
        "true_anomaly_rad",
        "altitude_m",
        "radius_m",
        "speed_m_s",
        "specific_energy_j_kg",
        "angular_momentum_m2_s",
        "eccentricity_vector_norm",
        "eclipse",
        "latitude_rad",
        "longitude_rad",
        "frame",
        "time_system",
    ]
    _write_csv(paths["orbital_elements"], orbital_fields, _orbital_rows(result))
    force_fields = list(ForceContribution.__dataclass_fields__)
    _write_csv(paths["force_budget"], force_fields, (_force_row(item) for item in result.force_contributions))
    vector_fields = [
        f"{name}::inertial_{axis}_m_s2"
        for name in sorted(result.force_vectors_m_s2)
        for axis in ("x", "y", "z")
    ]
    ric_fields = [
        f"{name}::ric_{axis}_m_s2"
        for name in sorted(result.force_ric_m_s2)
        for axis in ("r", "i", "c")
    ]
    force_series_fields = [
        "simulation_time_s",
        *sorted(result.force_magnitudes_m_s2),
        *vector_fields,
        *ric_fields,
    ]
    _write_csv(paths["force_budget_timeseries"], force_series_fields, _force_time_rows(result))

    summary_metrics = [
        metric
        for metric in result.metrics
        if metric.metric_id.startswith(("run.", "mission.", "orbit."))
    ]
    numerical_metrics = [metric for metric in result.metrics if metric.metric_id.startswith("numerical.")]
    extrema_metrics = [
        metric
        for metric in result.metrics
        if metric.metric_id in {"orbit.altitude.minimum", "orbit.altitude.maximum"}
    ]
    table_fields = [
        "metric_id",
        "label",
        "value",
        "unit",
        "status",
        "source",
        "kind",
        "frame",
        "time_system",
        "interpretation",
        "availability_reason",
    ]
    _write_csv(paths["summary_table"], table_fields, (metric.to_dict() for metric in summary_metrics))
    _write_csv(paths["extrema_table"], table_fields, (metric.to_dict() for metric in extrema_metrics))
    _write_csv(paths["integrator_table"], table_fields, (metric.to_dict() for metric in numerical_metrics))
    _write_csv(
        paths["provenance_table"],
        ["field", "value"],
        ({"field": path, "value": value} for path, value in _flatten(result.provenance)),
    )
    paths["report_markdown"].write_text(build_report_markdown(result), encoding="utf-8")
    return paths


def write_artifact_manifest(run_dir: str | Path) -> Path:
    """Hash the completed canonical package after PDF/figures are written."""

    root = Path(run_dir)
    manifest_path = root / "artifact_manifest.json"
    records = [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path, suppress_errors=True),
        }
        for path in sorted(
            item for item in root.rglob("*") if item.is_file() and item != manifest_path
        )
    ]
    _write_json(
        manifest_path,
        {
            "artifact_manifest_schema_version": 1,
            "generated_at_utc": utc_now_iso(),
            "artifacts": records,
        },
    )
    return manifest_path


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return payload


def _optional_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("non-finite numeric value in analysis artifact")
    return number


def _series_float(value: str | None) -> float:
    number = _optional_float(value)
    return float("nan") if number is None else number


def load_analysis_artifacts(run_dir: str | Path) -> OrbitAnalysisResult:
    """Reconstruct a canonical analysis result without rerunning propagation."""

    root = Path(run_dir)
    payload = _read_json(root / "metrics.json")
    metrics = tuple(MetricValue(**item) for item in payload.get("metrics", ()))
    forces = tuple(
        ForceContribution(**item) for item in payload.get("force_contributions", ())
    )

    with (root / "orbital_elements.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("orbital_elements.csv contains no state history")

    def column(name: str) -> np.ndarray:
        return np.asarray([_series_float(row.get(name)) for row in rows], dtype=np.float64)

    state = np.column_stack(
        [column(name) for name in ("x_m", "y_m", "z_m", "vx_m_s", "vy_m_s", "vz_m_s")]
    )

    def optional_column(name: str) -> np.ndarray | None:
        values = column(name)
        return None if np.all(np.isnan(values)) else values

    eclipse_values = [row.get("eclipse", "").strip().lower() for row in rows]
    eclipse = None
    if any(value for value in eclipse_values):
        eclipse = np.asarray([value in {"true", "1", "yes"} for value in eclipse_values])

    series = OrbitSeries(
        t_s=column("simulation_time_s"),
        state_m_mps=state,
        semi_major_axis_m=column("semi_major_axis_m"),
        eccentricity=column("eccentricity"),
        inclination_rad=column("inclination_rad"),
        raan_rad=column("raan_rad"),
        argument_of_periapsis_rad=column("argument_of_periapsis_rad"),
        true_anomaly_rad=column("true_anomaly_rad"),
        altitude_m=column("altitude_m"),
        radius_m=column("radius_m"),
        speed_m_s=column("speed_m_s"),
        specific_energy_j_kg=column("specific_energy_j_kg"),
        angular_momentum_m2_s=column("angular_momentum_m2_s"),
        eccentricity_vector_norm=column("eccentricity_vector_norm"),
        eclipse_mask=eclipse,
        latitude_rad=optional_column("latitude_rad"),
        longitude_rad=optional_column("longitude_rad"),
    )

    events: list[AnalysisEvent] = []
    with (root / "events.csv").open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            state_values = tuple(
                float(row[name]) for name in ("x_m", "y_m", "z_m", "vx_m_s", "vy_m_s", "vz_m_s")
                if row.get(name) not in {None, ""}
            )
            events.append(
                AnalysisEvent(
                    event_id=row["event_id"],
                    event_type=row["event_type"],
                    simulation_time_s=float(row["simulation_time_s"]),
                    epoch_utc=row.get("epoch_utc") or None,
                    state_m_mps=state_values if len(state_values) == 6 else None,
                    altitude_m=_optional_float(row.get("altitude_m")),
                    frame=row.get("frame") or str(payload.get("frame", "unknown")),
                    source=row.get("source") or "events.csv",
                    severity=row.get("severity") or "normal",
                    note=row.get("note") or None,
                )
            )

    force_time: np.ndarray | None = None
    force_series: dict[str, np.ndarray] = {}
    force_vectors: dict[str, np.ndarray] = {}
    force_ric: dict[str, np.ndarray] = {}
    force_path = root / "force_budget_timeseries.csv"
    if force_path.is_file():
        with force_path.open("r", encoding="utf-8", newline="") as handle:
            force_rows = list(csv.DictReader(handle))
        if force_rows:
            force_time = np.asarray(
                [float(row["simulation_time_s"]) for row in force_rows], dtype=np.float64
            )
            scalar_columns: dict[str, np.ndarray] = {
                name: np.asarray(
                    [_series_float(row.get(name)) for row in force_rows],
                    dtype=np.float64,
                )
                for name in force_rows[0]
                if name != "simulation_time_s"
            }
            vector_suffixes = tuple(f"::inertial_{axis}_m_s2" for axis in ("x", "y", "z"))
            ric_suffixes = tuple(f"::ric_{axis}_m_s2" for axis in ("r", "i", "c"))
            for name, values in scalar_columns.items():
                matched = False
                for suffixes, target in (
                    (vector_suffixes, force_vectors),
                    (ric_suffixes, force_ric),
                ):
                    for axis, suffix in enumerate(suffixes):
                        if not name.endswith(suffix):
                            continue
                        component = name[: -len(suffix)]
                        target.setdefault(
                            component,
                            np.full((len(force_rows), 3), np.nan, dtype=np.float64),
                        )[:, axis] = values
                        matched = True
                        break
                    if matched:
                        break
                if not matched:
                    force_series[name] = values

    return OrbitAnalysisResult(
        run_id=str(payload.get("run_id", root.name)),
        preset=str(payload.get("preset", "standard")),
        generated_at_utc=str(payload.get("generated_at_utc", "unknown")),
        frame=str(payload.get("frame", "unknown")),
        time_system=str(payload.get("time_system", "simulation elapsed time")),
        metrics=metrics,
        events=tuple(sorted(events, key=lambda event: (event.simulation_time_s, event.event_id))),
        series=series,
        force_contributions=forces,
        force_time_s=force_time,
        force_magnitudes_m_s2=force_series,
        force_vectors_m_s2=force_vectors,
        force_ric_m_s2=force_ric,
        provenance=_read_json(root / "provenance.json"),
        config_snapshot=_read_json(root / "config.json"),
        diagnostics_snapshot=_read_json(root / "diagnostics.json"),
        warnings=tuple(str(item) for item in payload.get("warnings", ())),
        schema_version=int(payload.get("analysis_schema_version", 1)),
    )


__all__ = [
    "build_report_markdown",
    "load_analysis_artifacts",
    "write_analysis_artifacts",
    "write_artifact_manifest",
]
