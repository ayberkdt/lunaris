# -*- coding: utf-8 -*-
"""Paper tables and figures for ST-LRPS, generated from CSV evidence outputs.

Every table and figure is derived from a produced CSV — numbers are never
hardcoded into the documents, so the whole set can be regenerated when the
evidence changes. Figures are only rendered when matplotlib is available;
otherwise the tables are still produced and the figures are skipped with a note.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    p = Path(path)
    if not p.exists() or p.stat().st_size <= 0:
        return []
    with p.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def csv_to_markdown_table(
    path: str | Path,
    *,
    columns: Optional[Sequence[str]] = None,
    max_rows: Optional[int] = None,
) -> str:
    """Render a CSV as a Markdown table (no values are altered or hardcoded)."""
    rows = read_csv_rows(path)
    if not rows:
        return "_(no data)_\n"
    cols = list(columns) if columns else list(rows[0].keys())
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for row in rows[: max_rows if max_rows is not None else len(rows)]:
        out.append("| " + " | ".join(str(row.get(c, "")) for c in cols) + " |")
    return "\n".join(out) + "\n"


# Logical table name -> (source CSV filename, title). Sources are searched for
# under the evidence tree; missing sources are reported, not invented.
PAPER_TABLE_SPECS = {
    "dataset_split_summary": ("split_manifest_summary.csv", "Dataset and split summary"),
    "field_validation_summary": ("field_validation_metrics.csv", "Field validation summary"),
    "orbit_benchmark_summary": ("metrics_summary.csv", "Orbit benchmark summary"),
    "runtime_speedup_summary": ("runtime_summary.csv", "Runtime / speedup summary"),
    "worst_case_summary": ("worst_case_scenarios.csv", "Worst-case scenario summary"),
    "multi_seed_summary": ("multi_seed_summary.csv", "Multi-seed summary"),
    "ablation_summary": ("st_lrps_ablation_summary.csv", "Ablation summary (optional)"),
}


def _find_csv(evidence_root: Path, filename: str) -> Optional[Path]:
    direct = evidence_root / filename
    if direct.exists():
        return direct
    matches = sorted(evidence_root.rglob(filename))
    return matches[0] if matches else None


def generate_paper_tables(evidence_root: str | Path, out_dir: str | Path) -> dict[str, Any]:
    """Write one Markdown table per available source CSV under ``tables/``."""
    evidence_root = Path(evidence_root)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}
    missing: list[str] = []
    for name, (filename, title) in PAPER_TABLE_SPECS.items():
        source = _find_csv(evidence_root, filename)
        if source is None:
            missing.append(name)
            continue
        table_md = csv_to_markdown_table(source)
        text = f"# {title}\n\n_Source: `{source}`_\n\n{table_md}"
        path = out / f"table_{name}.md"
        path.write_text(text, encoding="utf-8")
        written[name] = str(path)
    index = out / "TABLES_INDEX.md"
    index.write_text(_render_index(written, missing), encoding="utf-8")
    return {"written": written, "missing": missing, "index": str(index)}


def _render_index(written: Mapping[str, str], missing: Sequence[str]) -> str:
    lines = ["# ST-LRPS Paper Tables", "", "All tables are generated from CSV evidence outputs.", ""]
    for name, path in written.items():
        lines.append(f"- `{Path(path).name}` ({PAPER_TABLE_SPECS[name][1]})")
    if missing:
        lines += ["", "## Missing sources (not yet produced)"]
        lines += [f"- {name} (expected `{PAPER_TABLE_SPECS[name][0]}`)" for name in missing]
    lines.append("")
    return "\n".join(lines)


def generate_paper_figures(evidence_root: str | Path, out_dir: str | Path) -> dict[str, Any]:
    """Render figures from the benchmark CSVs. Skipped (with a note) without matplotlib."""
    evidence_root = Path(evidence_root)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        note = out / "FIGURES_SKIPPED.txt"
        note.write_text("matplotlib not available; figures skipped. Tables (CSV/MD) are still produced.\n", encoding="utf-8")
        return {"rendered": {}, "skipped": True, "note": str(note)}

    rendered: dict[str, str] = {}
    metrics = _find_csv(evidence_root, "metrics_summary.csv")
    rows = read_csv_rows(metrics) if metrics else []

    def _bar(models, values, *, ylabel, title, fname, logy=False):
        pairs = [(m, v) for m, v in zip(models, values) if v is not None]
        if not pairs:
            return
        m2, v2 = zip(*pairs)
        fig, ax = plt.subplots(figsize=(7.5, 4.2), constrained_layout=True)
        ax.bar(range(len(m2)), v2)
        ax.set_xticks(range(len(m2)))
        ax.set_xticklabels(m2, rotation=30, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if logy:
            ax.set_yscale("log")
        path = out / fname
        fig.savefig(path, dpi=160)
        plt.close(fig)
        rendered[fname] = str(path)

    if rows:
        models = [r.get("model") for r in rows]
        med = [_f(r.get("median_rms_pos_err_km")) for r in rows]
        _bar(models, med, ylabel="median RMS pos err [km]", title="RMS position error vs model",
             fname="rms_position_error_vs_model.png", logy=True)
        # RIC decomposition.
        radial = [_f(r.get("median_radial_rms_km")) for r in rows]
        along = [_f(r.get("median_along_rms_km")) for r in rows]
        cross = [_f(r.get("median_cross_rms_km")) for r in rows]
        if any(v is not None for v in radial + along + cross):
            fig, ax = plt.subplots(figsize=(8.0, 4.2), constrained_layout=True)
            import numpy as np

            x = np.arange(len(models))
            ax.bar(x - 0.25, [v or 0 for v in radial], width=0.25, label="radial")
            ax.bar(x, [v or 0 for v in along], width=0.25, label="along")
            ax.bar(x + 0.25, [v or 0 for v in cross], width=0.25, label="cross")
            ax.set_xticks(x); ax.set_xticklabels(models, rotation=30, ha="right")
            ax.set_ylabel("median RIC RMS [km]"); ax.set_title("RIC error decomposition"); ax.legend()
            path = out / "ric_error_decomposition.png"
            fig.savefig(path, dpi=160); plt.close(fig)
            rendered["ric_error_decomposition.png"] = str(path)

    runtime = _find_csv(evidence_root, "runtime_summary.csv")
    rt_rows = read_csv_rows(runtime) if runtime else []
    if rt_rows:
        _bar([r.get("model") for r in rt_rows], [_f(r.get("total_runtime_s")) for r in rt_rows],
             ylabel="runtime [s]", title="Runtime vs model", fname="runtime_vs_model.png", logy=True)
        # Error-runtime trade-off (join on model).
        rt_by_model = {r.get("model"): _f(r.get("total_runtime_s")) for r in rt_rows}
        err_by_model = {r.get("model"): _f(r.get("median_rms_pos_err_km")) for r in rows}
        common = [m for m in rt_by_model if m in err_by_model and rt_by_model[m] and err_by_model[m]]
        if common:
            fig, ax = plt.subplots(figsize=(6.8, 5.0), constrained_layout=True)
            for m in common:
                ax.scatter(rt_by_model[m], err_by_model[m])
                ax.annotate(m, (rt_by_model[m], err_by_model[m]))
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_xlabel("runtime [s]"); ax.set_ylabel("median RMS pos err [km]")
            ax.set_title("Error vs runtime trade-off")
            path = out / "error_runtime_tradeoff.png"
            fig.savefig(path, dpi=160); plt.close(fig)
            rendered["error_runtime_tradeoff.png"] = str(path)

    # Field acceleration error vs altitude.
    alt = _find_csv(evidence_root, "field_validation_by_altitude.csv")
    alt_rows = read_csv_rows(alt) if alt else []
    if alt_rows:
        import numpy as np

        fig, ax = plt.subplots(figsize=(7.5, 4.6), constrained_layout=True)
        by_policy: dict[str, list[tuple[float, float]]] = {}
        for r in alt_rows:
            lo, hi, val = _f(r.get("altitude_km_min")), _f(r.get("altitude_km_max")), _f(r.get("accel_rmse"))
            if lo is None or hi is None or val is None:
                continue
            by_policy.setdefault(str(r.get("policy")), []).append((0.5 * (lo + hi), val))
        for policy, pts in by_policy.items():
            pts.sort()
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", label=policy)
        if by_policy:
            ax.set_yscale("log"); ax.set_xlabel("altitude [km]"); ax.set_ylabel("accel RMSE [m/s²]")
            ax.set_title("Field acceleration error vs altitude"); ax.legend()
            path = out / "field_acceleration_error_vs_altitude.png"
            fig.savefig(path, dpi=160); plt.close(fig)
            rendered["field_acceleration_error_vs_altitude.png"] = str(path)

    return {"rendered": rendered, "skipped": False}


def _f(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


__all__ = [
    "PAPER_TABLE_SPECS",
    "csv_to_markdown_table",
    "generate_paper_figures",
    "generate_paper_tables",
    "read_csv_rows",
]
