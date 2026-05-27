#!/usr/bin/env python3
# st_lrps/evaluation/publication_plots.py
# -*- coding: utf-8 -*-
"""
Publication-quality figures for ST-LRPS lunar-orbit validation.

This is the relocated, parameterized version of the former
``AIAA SciTech/generate_publication_plots_v2.py`` (paths are now CLI arguments
instead of hardcoded absolutes). It renders AIAA-journal-style figures + a
LaTeX summary table from one or more ``compare_gravity_models`` GPU-batch run
directories.

Inputs (per run directory) — produced by ``st_lrps.evaluation.compare_gravity_models``:
  metrics/gpu_batch_per_scenario_metrics.csv
  metrics/gpu_batch_aggregate_metrics.csv
  metrics/gpu_batch_runtime_metrics.csv
  metrics/stlrps_selected_scenarios.json   (ST-LRPS run only)
  scenarios.csv

Usage
-----
  python -m st_lrps.evaluation.publication_plots \\
      --stlrps-run outputs/gravity_benchmark/stlrps_100 \\
      --multi-run  outputs/gravity_benchmark/sh20_sh60_stlrps_100 \\
      --out-dir    outputs/aiaa_scitech/publication_plots

If a single run directory holds every model (SH20, SH60, ST-LRPS), pass it once
with ``--run`` and it is used for both roles.
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from matplotlib.colors import LogNorm
import numpy as np

try:
    import pandas as pd
except ImportError as exc:  # pragma: no cover - pandas is required here
    raise SystemExit("publication_plots requires pandas. Install it with `pip install pandas`.") from exc

KM2M = 1000.0

# Colorblind-safe palette (Wong 2011, Nature Methods).
CLR = {"SH20": "#D55E00", "STLRPS": "#0072B2", "SH60": "#009E73", "light": "#E5E5E5"}
MRK = {"SH20": "s", "STLRPS": "o", "SH60": "D"}
LS = {"SH20": "--", "STLRPS": "-", "SH60": "-."}

_RCPARAMS = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "axes.titleweight": "normal",
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "legend.framealpha": 0.85,
    "legend.edgecolor": "0.7",
    "legend.fancybox": False,
    "figure.dpi": 300,
    "figure.facecolor": "white",
    "figure.figsize": (6.5, 3.5),
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.04,
    "axes.facecolor": "white",
    "axes.grid": True,
    "axes.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.alpha": 0.30,
    "grid.linewidth": 0.35,
    "grid.color": "#cccccc",
    "lines.linewidth": 1.0,
    "lines.markersize": 5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def _require(path: Path) -> Path:
    if not path.exists():
        raise SystemExit(f"Required input not found: {path}")
    return path


def _load_run(run_dir: Path) -> Dict[str, Any]:
    """Load the per-scenario / aggregate / runtime frames for one run dir."""
    m = run_dir / "metrics"
    return {
        "per": pd.read_csv(_require(m / "gpu_batch_per_scenario_metrics.csv")),
        "agg": pd.read_csv(_require(m / "gpu_batch_aggregate_metrics.csv")),
        "rt": pd.read_csv(_require(m / "gpu_batch_runtime_metrics.csv")),
        "scenarios": pd.read_csv(_require(run_dir / "scenarios.csv")),
        "selected_path": m / "stlrps_selected_scenarios.json",
    }


def _model_frame(per: "pd.DataFrame", display: str) -> "pd.DataFrame":
    sub = per[per["model"] == display].copy().reset_index(drop=True)
    if sub.empty:
        raise SystemExit(
            f"Model '{display}' not found in per-scenario metrics. "
            f"Available: {sorted(per['model'].unique())}"
        )
    return sub


def _save(fig, out: Path, stem: str) -> None:
    fig.savefig(out / f"{stem}.pdf", facecolor="white")
    fig.savefig(out / f"{stem}.png", facecolor="white")
    plt.close(fig)
    print(f"  [ok] {stem}")


def generate_figures(stlrps_run: Path, multi_run: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(_RCPARAMS)

    st = _load_run(stlrps_run)
    multi = _load_run(multi_run)

    st_per = st["per"]
    st_agg = st["agg"]
    st_rt = st["rt"]
    scenarios = st["scenarios"]
    selected: Dict[str, Any] = {}
    if st["selected_path"].exists():
        selected = json.loads(st["selected_path"].read_text(encoding="utf-8"))

    multi_per = multi["per"]
    multi_agg = multi["agg"]
    multi_rt = multi["rt"]

    sh20_per = _model_frame(multi_per, "GPU_SH20_RK4")
    sh60_per = _model_frame(multi_per, "GPU_SH60_RK4")
    sh20_agg = multi_agg[multi_agg["model"] == "GPU_SH20_RK4"].iloc[0]
    sh60_agg = multi_agg[multi_agg["model"] == "GPU_SH60_RK4"].iloc[0]
    st_agg_row = st_agg[st_agg["model"] == "GPU_ST_LRPS_RK4"]
    st_agg_row = st_agg_row.iloc[0] if not st_agg_row.empty else st_agg.iloc[0]

    # Merge orbital elements into ST-LRPS per-scenario frame.
    st_per = st_per.merge(scenarios[["scenario_id", "hp_km", "ha_km", "inc_deg"]],
                          on="scenario_id", how="left", suffixes=("", "_scen"))
    for col in ("hp_km", "ha_km", "inc_deg"):
        if f"{col}_scen" in st_per.columns:
            st_per[col] = st_per[f"{col}_scen"]
    st_per_model = _model_frame(st_per, "GPU_ST_LRPS_RK4") if "GPU_ST_LRPS_RK4" in set(st_per["model"]) else st_per

    # ---- Figure 1: speed-accuracy tradeoff ----------------------------
    print("Fig 1: Speed-accuracy tradeoff")
    models = {
        "SH20": {"t": multi_rt[multi_rt["model"] == "GPU_SH20_RK4"]["total_runtime_s"].iloc[0],
                 "err": sh20_agg["median_rms_pos_err_km"] * KM2M},
        "ST-LRPS": {"t": st_rt["total_runtime_s"].iloc[0],
                    "err": st_agg_row["median_rms_pos_err_km"] * KM2M},
        "SH60": {"t": multi_rt[multi_rt["model"] == "GPU_SH60_RK4"]["total_runtime_s"].iloc[0],
                 "err": sh60_agg["median_rms_pos_err_km"] * KM2M},
    }
    fig1, ax = plt.subplots(figsize=(3.5, 2.8))
    for lbl, d in models.items():
        key = lbl.replace("-", "").replace(" ", "")
        ms = 8 if lbl == "ST-LRPS" else 6.5
        zord = 6 if lbl == "ST-LRPS" else 5
        ax.plot(d["t"], d["err"], marker=MRK[key], color=CLR[key], ms=ms, mec="k",
                mew=0.5, ls="none", zorder=zord, label=lbl)
    ax.axhline(1.0, color="0.65", lw=0.8, ls=":", zorder=1)
    ax.set_yscale("log")
    ax.grid(True, which="minor", axis="y", alpha=0.15, ls=":")
    ax.set_xlabel("Total runtime for the scenario set [s]")
    ax.set_ylabel("Median RMS position error [m]")
    ax.legend(loc="upper right", frameon=True, fontsize=8)
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.yaxis.get_major_formatter().set_scientific(False)
    _save(fig1, out_dir, "fig1_speed_accuracy")

    # ---- Figure 2: RMS error distribution (box + ECDF) ----------------
    print("Fig 2: RMS error distribution")
    model_errs = {
        "SH20": sh20_per["rms_pos_err_km"].values * KM2M,
        "ST-LRPS": st_per_model["rms_pos_err_km"].values * KM2M,
        "SH60": sh60_per["rms_pos_err_km"].values * KM2M,
    }
    keys_ordered = ["SH20", "ST-LRPS", "SH60"]
    colors_ordered = [CLR["SH20"], CLR["STLRPS"], CLR["SH60"]]
    positions = [1, 2, 3]
    fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(6.5, 2.8),
                                      gridspec_kw={"width_ratios": [1, 1.1], "wspace": 0.38})
    bp = ax2a.boxplot([model_errs[k] for k in keys_ordered], positions=positions,
                      widths=0.4, patch_artist=True, showfliers=False,
                      medianprops=dict(color="k", lw=1.2),
                      whiskerprops=dict(lw=0.8), capprops=dict(lw=0.8))
    for patch, col in zip(bp["boxes"], colors_ordered):
        patch.set_facecolor(col); patch.set_alpha(0.3)
        patch.set_edgecolor(col); patch.set_linewidth(1.0)
    rng = np.random.default_rng(42)
    for pos, k, col in zip(positions, keys_ordered, colors_ordered):
        jitter = rng.uniform(-0.15, 0.15, size=len(model_errs[k]))
        ax2a.scatter(pos + jitter, model_errs[k], s=8, color=col, alpha=0.6,
                     edgecolors="none", zorder=4)
    ax2a.set_yscale("log")
    ax2a.set_xticks(positions); ax2a.set_xticklabels(keys_ordered, fontsize=8)
    ax2a.set_ylabel("RMS position error [m]")
    ax2a.set_title("(a)", loc="left", fontsize=9, pad=4)
    for k, col, ls in zip(keys_ordered, colors_ordered, [LS["SH20"], LS["STLRPS"], LS["SH60"]]):
        vals = np.sort(model_errs[k])
        ecdf = np.arange(1, len(vals) + 1) / len(vals)
        ax2b.step(vals, ecdf, where="post", color=col, lw=1.5, ls=ls, label=k)
    ax2b.set_xscale("log")
    ax2b.set_xlabel("RMS position error [m]"); ax2b.set_ylabel("ECDF")
    ax2b.set_ylim(-0.02, 1.04)
    ax2b.set_title("(b)", loc="left", fontsize=9, pad=4)
    ax2b.legend(loc="lower right", frameon=True)
    ax2b.axvline(1.0, color="0.65", lw=0.5, ls=":", zorder=1)
    _save(fig2, out_dir, "fig2_rms_distribution")

    # ---- Figure 3: error sensitivity (altitude vs inclination) --------
    print("Fig 3: Error sensitivity")
    fig3, axes3 = plt.subplots(1, 3, figsize=(6.5, 2.5), gridspec_kw={"wspace": 0.55})
    panel_list = [("(a) SH20", sh20_per, "SH20"),
                  ("(b) ST-LRPS", st_per_model, "STLRPS"),
                  ("(c) SH60", sh60_per, "SH60")]
    for ax, (title, df, mkey) in zip(axes3, panel_list):
        err_m = df["rms_pos_err_km"].values * KM2M
        finite = err_m[np.isfinite(err_m) & (err_m > 0)]
        vmin = float(np.percentile(finite, 2)) if finite.size else 1e-3
        vmax = float(np.percentile(finite, 98)) if finite.size else 1.0
        vmin = max(vmin, 1e-6); vmax = max(vmax, vmin * 10)
        if {"hp_km", "inc_deg"}.issubset(df.columns):
            sc = ax.scatter(df["hp_km"], df["inc_deg"], c=err_m, cmap="RdYlGn_r",
                            norm=LogNorm(vmin=vmin, vmax=vmax), s=14,
                            edgecolors="0.4", linewidths=0.15, zorder=3)
            cb = plt.colorbar(sc, ax=ax, pad=0.04, shrink=0.88, aspect=25)
            cb.set_label("[m]", fontsize=7, labelpad=2)
            cb.ax.tick_params(labelsize=6.5)
        ax.set_xlabel("Periapsis altitude [km]")
        ax.set_ylabel("Inclination [deg]" if ax is axes3[0] else "")
        ax.set_title(title, loc="left", fontsize=8, pad=3)
        ax.tick_params(labelsize=7)
        if mkey == "STLRPS" and selected:
            case_markers = {"best": ("v", "#1a9641"), "representative": ("D", "#2166AC"),
                            "worst": ("X", "#d7191c")}
            handles = []
            for case_key, (mk, mc) in case_markers.items():
                item = selected.get(case_key)
                if not item:
                    continue
                row = df[df["scenario_id"] == item["scenario_id"]]
                if row.empty:
                    continue
                r = row.iloc[0]
                ax.scatter(r["hp_km"], r["inc_deg"], marker=mk, s=40, facecolors="none",
                           edgecolors=mc, linewidths=1.2, zorder=7)
                handles.append(Line2D([0], [0], marker=mk, color="w", markerfacecolor="none",
                                      markeredgecolor=mc, markeredgewidth=1.0, markersize=5,
                                      label=case_key.capitalize()))
            if handles:
                ax.legend(handles=handles, loc="lower right", fontsize=6, frameon=True,
                          borderpad=0.3, handletextpad=0.3)
    _save(fig3, out_dir, "fig3_error_sensitivity")

    # ---- Figure 4: RIC decomposition ----------------------------------
    print("Fig 4: RIC decomposition")
    ric_cols = {"Radial": "radial_rms_km", "Along-track": "along_rms_km", "Cross-track": "cross_rms_km"}
    components = list(ric_cols.keys())
    y_pos = np.arange(len(components))
    model_specs = [("SH20", sh20_per, CLR["SH20"], MRK["SH20"]),
                   ("ST-LRPS", st_per_model, CLR["STLRPS"], MRK["STLRPS"]),
                   ("SH60", sh60_per, CLR["SH60"], MRK["SH60"])]
    fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(6.5, 2.2), gridspec_kw={"wspace": 0.40})
    y_offsets = [-0.15, 0.0, 0.15]
    for panel_ax, stat_func, panel_title in [(ax4a, np.median, "(a) Median"),
                                             (ax4b, lambda x: np.percentile(x, 95), "(b) 95th percentile")]:
        for (name, df, col, mk), dy in zip(model_specs, y_offsets):
            vals = [stat_func(df[v].values) * KM2M for v in ric_cols.values()]
            yy = y_pos + dy
            for yi, vi in zip(yy, vals):
                panel_ax.plot([0, vi], [yi, yi], color=col, lw=0.4, alpha=0.4, zorder=2)
            panel_ax.scatter(vals, yy, marker=mk, s=28, color=col, edgecolors="k",
                             linewidths=0.2, zorder=5, label=name)
        panel_ax.set_xscale("log")
        panel_ax.set_yticks(y_pos); panel_ax.set_yticklabels(components)
        panel_ax.set_xlabel("RMS error [m]")
        panel_ax.set_title(panel_title, loc="left", fontsize=8, pad=3)
        panel_ax.invert_yaxis(); panel_ax.tick_params(axis="y", length=0)
    ax4a.legend(loc="lower right", fontsize=7, frameon=True, borderpad=0.3)
    _save(fig4, out_dir, "fig4_ric_decomposition")

    # ---- LaTeX summary table ------------------------------------------
    print("Table: validation summary (LaTeX)")
    sh20_rt = multi_rt[multi_rt["model"] == "GPU_SH20_RK4"].iloc[0]
    sh60_rt = multi_rt[multi_rt["model"] == "GPU_SH60_RK4"].iloc[0]
    st_rt_row = st_rt.iloc[0]
    denom = float(sh60_rt["total_runtime_s"]) or 1.0
    rows = [
        ("SH20 RK4", sh20_agg, sh20_rt["total_runtime_s"], sh20_rt["total_runtime_s"] / denom),
        ("ST-LRPS RK4", st_agg_row, st_rt_row["total_runtime_s"], st_rt_row["total_runtime_s"] / denom),
        ("SH60 RK4", sh60_agg, sh60_rt["total_runtime_s"], 1.0),
    ]
    latex = (
        "\\begin{table}[htbp]\n\\centering\n"
        "\\caption{Validation summary for the lunar orbit propagation cases. "
        "All errors are relative to the high-degree DOP853 reference solution.}\n"
        "\\label{tab:validation_summary}\n"
        "\\begin{tabular}{l r r r r r}\n\\toprule\n"
        "Model & Median RMS & P95 RMS & Max RMS & Runtime & Rel.\\ Time \\\\\n"
        "      & {[m]}      & {[m]}   & {[m]}   & {[s]}   & {vs.\\ SH60} \\\\\n\\midrule\n"
    )
    for name, agg, rt, rel in rows:
        med = float(agg["median_rms_pos_err_km"]) * KM2M
        p95 = float(agg["p95_rms_pos_err_km"]) * KM2M
        mx = float(agg["max_rms_pos_err_km"]) * KM2M
        label = "\\textbf{" + name + "}" if name.startswith("ST-LRPS") else name
        latex += f"{label} & {med:.3f} & {p95:.3f} & {mx:.3f} & {rt:.1f} & {rel:.2f}$\\times$ \\\\\n"
    latex += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    (out_dir / "table_validation_summary.tex").write_text(latex, encoding="utf-8")
    print("  [ok] table_validation_summary.tex")

    captions = textwrap.dedent(
        """\
        # Figure Captions — ST-LRPS Validation

        ## Figure 1
        Speed–accuracy tradeoff. Markers denote SH20 RK4, ST-LRPS RK4, and SH60 RK4;
        position errors are relative to the high-degree DOP853 reference. The dashed
        horizontal line marks the 1 m threshold.

        ## Figure 2
        Distribution of trajectory-level RMS position error. (a) Box plots with jittered
        per-scenario points; (b) empirical CDFs on a logarithmic abscissa.

        ## Figure 3
        Dependence of RMS position error on initial periapsis altitude and inclination.
        Each panel uses an individual logarithmic color scale; selected ST-LRPS cases are
        marked on the ST-LRPS panel.

        ## Figure 4
        RIC-frame decomposition of RMS position error: (a) median and (b) 95th percentile.
        """
    )
    (out_dir / "figure_captions.md").write_text(captions, encoding="utf-8")
    print("  [ok] figure_captions.md")
    print(f"\nAll outputs -> {out_dir}")


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Render ST-LRPS publication figures from compare_gravity_models GPU-batch runs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--run", default=None,
                    help="Single GPU-batch run dir containing SH20, SH60 and ST-LRPS "
                         "(used for both roles when --stlrps-run/--multi-run are omitted).")
    ap.add_argument("--stlrps-run", default=None,
                    help="Run dir with the ST-LRPS GPU-batch metrics and selected scenarios.")
    ap.add_argument("--multi-run", default=None,
                    help="Run dir with the SH20/SH60 (and ST-LRPS) GPU-batch metrics.")
    ap.add_argument("--out-dir", default=None,
                    help="Output directory for figures (default: <run>/publication_plots).")
    return ap


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    single = Path(args.run).expanduser().resolve() if args.run else None
    stlrps_run = Path(args.stlrps_run).expanduser().resolve() if args.stlrps_run else single
    multi_run = Path(args.multi_run).expanduser().resolve() if args.multi_run else single
    if stlrps_run is None or multi_run is None:
        raise SystemExit("Provide --run, or both --stlrps-run and --multi-run.")
    out_dir = (Path(args.out_dir).expanduser().resolve() if args.out_dir
               else (stlrps_run / "publication_plots"))
    generate_figures(stlrps_run, multi_run, out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
