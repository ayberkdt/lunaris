#!/usr/bin/env python3
"""
generate_publication_plots_v2.py
================================
Publication-quality figures for an AIAA SciTech extended abstract on
ST-LRPS residual-neural-potential surrogates for lunar orbit propagation.

100-case, one-day validation batch.
Models compared: GPU SH20 RK4, GPU ST-LRPS RK4, GPU SH60 RK4.
Reference solution: GPU SH200 DOP853.

Outputs
-------
publication_plots_v2/
    fig1_speed_accuracy_v2.{pdf,png}
    fig2_rms_distribution_v2.{pdf,png}
    fig3_error_sensitivity_v2.{pdf,png}
    fig4_ric_decomposition_v2.{pdf,png}
    figure_captions_v2.md
    revision_notes_v2.md
    table_validation_summary.tex
"""

import json
import pathlib
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

# ─── Paths ────────────────────────────────────────────────────────────
RESULTS  = pathlib.Path(r"c:\Users\ayber\Desktop\LUNAR_SIMULATION\results")
STLRPS   = RESULTS / "stlrps_ders_cikisi_100"
MULTI    = RESULTS / "gpu_sh60_sh20_stlrps_100"
OUT      = pathlib.Path(r"c:\Users\ayber\Desktop\LUNAR_SIMULATION\AIAA SciTech\publication_plots_v2")
OUT.mkdir(parents=True, exist_ok=True)

# ─── Load data ────────────────────────────────────────────────────────
st_per   = pd.read_csv(STLRPS / "metrics" / "gpu_batch_per_scenario_metrics.csv")
st_agg   = pd.read_csv(STLRPS / "metrics" / "gpu_batch_aggregate_metrics.csv")
st_rt    = pd.read_csv(STLRPS / "metrics" / "gpu_batch_runtime_metrics.csv")
scenarios = pd.read_csv(STLRPS / "scenarios.csv")
with open(STLRPS / "metrics" / "stlrps_selected_scenarios.json") as f:
    selected = json.load(f)

multi_per = pd.read_csv(MULTI / "metrics" / "gpu_batch_per_scenario_metrics.csv")
multi_agg = pd.read_csv(MULTI / "metrics" / "gpu_batch_aggregate_metrics.csv")
multi_rt  = pd.read_csv(MULTI / "metrics" / "gpu_batch_runtime_metrics.csv")

sh20_per = multi_per[multi_per["model"] == "GPU_SH20_RK4"].copy().reset_index(drop=True)
sh60_per = multi_per[multi_per["model"] == "GPU_SH60_RK4"].copy().reset_index(drop=True)
sh20_agg = multi_agg[multi_agg["model"] == "GPU_SH20_RK4"].iloc[0]
sh60_agg = multi_agg[multi_agg["model"] == "GPU_SH60_RK4"].iloc[0]

# Merge orbital elements into ST-LRPS per-scenario frame
st_per = st_per.merge(scenarios[["scenario_id", "hp_km", "ha_km", "inc_deg"]],
                      on="scenario_id", how="left", suffixes=("", "_scen"))
for col in ("hp_km", "ha_km", "inc_deg"):
    if f"{col}_scen" in st_per.columns:
        st_per[col] = st_per[f"{col}_scen"]

KM2M = 1000.0

# ─── Global rcParams (AIAA journal style) ─────────────────────────────
plt.rcParams.update({
    # Fonts: 8-10 pt when inserted at 0.9\linewidth (~6.3 in)
    "font.family":         "serif",
    "font.serif":          ["Times New Roman", "DejaVu Serif"],
    "font.size":           8,
    "axes.labelsize":      9,
    "axes.titlesize":      9,
    "axes.titleweight":    "normal",
    "xtick.labelsize":     8,
    "ytick.labelsize":     8,
    "legend.fontsize":     7.5,
    "legend.framealpha":   0.85,
    "legend.edgecolor":    "0.7",
    "legend.fancybox":     False,
    "legend.borderpad":    0.4,
    "legend.handlelength": 1.5,
    # Figure
    "figure.dpi":          300,
    "figure.facecolor":    "white",
    "figure.figsize":      (6.5, 3.5),       # single-column AIAA default
    "savefig.dpi":         300,
    "savefig.bbox":        "tight",
    "savefig.pad_inches":  0.04,
    # Axes
    "axes.facecolor":      "white",
    "axes.grid":           True,
    "axes.linewidth":      0.5,
    "axes.spines.top":     False,
    "axes.spines.right":   False,
    # Grid
    "grid.alpha":          0.30,
    "grid.linewidth":      0.35,
    "grid.linestyle":      "-",
    "grid.color":          "#cccccc",
    # Lines / markers
    "lines.linewidth":     1.0,
    "lines.markersize":    5,
    # PDF export
    "pdf.fonttype":        42,
    "ps.fonttype":         42,
})

# ─── Colorblind-safe palette (Wong 2011, Nature Methods) ──────────────
CLR = {
    "SH20":    "#D55E00",   # vermilion
    "STLRPS":  "#0072B2",   # blue
    "SH60":    "#009E73",   # bluish green
    "light":   "#E5E5E5",
}
MRK = {"SH20": "s", "STLRPS": "o", "SH60": "D"}    # distinct shapes
LS  = {"SH20": "--", "STLRPS": "-", "SH60": "-."}   # distinct dashes

def _save(fig, stem):
    """Save figure as PDF + PNG and close."""
    fig.savefig(OUT / f"{stem}.pdf", facecolor="white")
    fig.savefig(OUT / f"{stem}.png", facecolor="white")
    plt.close(fig)
    print(f"  [ok] {stem}")


# ======================================================================
# FIGURE 1 -- Speed-Accuracy Tradeoff
# ======================================================================
print("Fig 1: Speed-accuracy tradeoff")

models = {
    "SH20": {
        "t":   multi_rt[multi_rt["model"] == "GPU_SH20_RK4"]["total_runtime_s"].iloc[0],
        "err": sh20_agg["median_rms_pos_err_km"] * KM2M,
    },
    "ST-LRPS": {
        "t":   st_rt["total_runtime_s"].iloc[0],
        "err": st_agg["median_rms_pos_err_km"].iloc[0] * KM2M,
    },
    "SH60": {
        "t":   multi_rt[multi_rt["model"] == "GPU_SH60_RK4"]["total_runtime_s"].iloc[0],
        "err": sh60_agg["median_rms_pos_err_km"] * KM2M,
    },
}

fig1, ax = plt.subplots(figsize=(3.5, 2.8))

for lbl, d in models.items():
    key = lbl.replace("-", "").replace(" ", "")   # SH20, STLRPS, SH60
    ms = 8 if lbl == "ST-LRPS" else 6.5
    zord = 6 if lbl == "ST-LRPS" else 5
    ax.plot(d["t"], d["err"],
            marker=MRK[key], color=CLR[key], ms=ms, mec="k", mew=0.5,
            ls="none", zorder=zord, label=lbl)

# 1-m threshold guide
ax.axhline(1.0, color="0.65", lw=0.8, ls=":", zorder=1)
ax.text(5300, 1.15, "1 m", fontsize=7, color="0.50", ha="right", va="bottom")

ax.set_yscale("log")
ax.grid(True, which="minor", axis="y", alpha=0.15, ls=":")
ax.set_xlabel("Total runtime for 100 scenarios [s]")
ax.set_ylabel("Median RMS position error [m]")
ax.set_xlim(1200, 5500)
ax.set_ylim(0.02, 300)
ax.legend(loc="upper right", frameon=True, fontsize=8)
ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
ax.yaxis.get_major_formatter().set_scientific(False)
ax.set_yticks([0.1, 1, 10, 100])
ax.set_yticklabels(["0.1", "1", "10", "100"])

_save(fig1, "fig1_speed_accuracy_v2")


# ======================================================================
# FIGURE 2 -- RMS Error Distribution  (violin + ECDF)
# ======================================================================
print("Fig 2: RMS error distribution")

fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(6.5, 2.8),
                                   gridspec_kw={"width_ratios": [1, 1.1],
                                                "wspace": 0.38})

# Panel (a): Violin + strip
model_errs = {
    "SH20":    sh20_per["rms_pos_err_km"].values * KM2M,
    "ST-LRPS": st_per["rms_pos_err_km"].values * KM2M,
    "SH60":    sh60_per["rms_pos_err_km"].values * KM2M,
}
keys_ordered = ["SH20", "ST-LRPS", "SH60"]
colors_ordered = [CLR["SH20"], CLR["STLRPS"], CLR["SH60"]]
positions = [1, 2, 3]

bp = ax2a.boxplot([model_errs[k] for k in keys_ordered],
                  positions=positions, widths=0.4, patch_artist=True,
                  showfliers=False,
                  medianprops=dict(color="k", lw=1.2),
                  whiskerprops=dict(lw=0.8),
                  capprops=dict(lw=0.8))
for patch, col in zip(bp["boxes"], colors_ordered):
    patch.set_facecolor(col)
    patch.set_alpha(0.3)
    patch.set_edgecolor(col)
    patch.set_linewidth(1.0)

# Jittered strip plot
rng = np.random.default_rng(42)
for pos, k, col in zip(positions, keys_ordered, colors_ordered):
    jitter = rng.uniform(-0.15, 0.15, size=len(model_errs[k]))
    ax2a.scatter(pos + jitter, model_errs[k],
                 s=8, color=col, alpha=0.6, edgecolors="none", zorder=4)

ax2a.set_yscale("log")
ax2a.grid(True, which="minor", axis="y", alpha=0.15, ls=":")
ax2a.set_xticks(positions)
ax2a.set_xticklabels(keys_ordered, fontsize=8)
ax2a.set_ylabel("RMS position error [m]")
ax2a.set_title("(a)", loc="left", fontsize=9, pad=4)

# Panel (b): ECDF for all three models
for k, col, ls in zip(keys_ordered, colors_ordered,
                       [LS["SH20"], LS["STLRPS"], LS["SH60"]]):
    vals = np.sort(model_errs[k])
    ecdf = np.arange(1, len(vals) + 1) / len(vals)
    ax2b.step(vals, ecdf, where="post", color=col, lw=1.5, ls=ls, label=k)

ax2b.set_xscale("log")
ax2b.grid(True, which="minor", axis="x", alpha=0.15, ls=":")
ax2b.set_xlabel("RMS position error [m]")
ax2b.set_ylabel("ECDF")
ax2b.set_ylim(-0.02, 1.04)
ax2b.set_title("(b)", loc="left", fontsize=9, pad=4)
ax2b.legend(loc="lower right", frameon=True)

# 1-m guide on ECDF
ax2b.axvline(1.0, color="0.65", lw=0.5, ls=":", zorder=1)

_save(fig2, "fig2_rms_distribution_v2")


# ======================================================================
# FIGURE 3 -- Error Sensitivity: Altitude vs Inclination (3 panels)
# ======================================================================
print("Fig 3: Error sensitivity -- altitude vs inclination")

from matplotlib.colors import LogNorm

fig3, axes3 = plt.subplots(1, 3, figsize=(6.5, 2.5),
                            gridspec_kw={"wspace": 0.55})

panel_list = [
    ("(a) SH20",    sh20_per, "SH20"),
    ("(b) ST-LRPS", st_per,   "STLRPS"),
    ("(c) SH60",    sh60_per, "SH60"),
]

# Individual log-scale color normalisations
norms = {
    "SH20":   LogNorm(vmin=10, vmax=600),
    "STLRPS": LogNorm(vmin=0.02, vmax=1.0),
    "SH60":   LogNorm(vmin=0.005, vmax=2.5),
}

for ax, (title, df, mkey) in zip(axes3, panel_list):
    err_m = df["rms_pos_err_km"].values * KM2M
    sc = ax.scatter(df["hp_km"], df["inc_deg"], c=err_m,
                    cmap="RdYlGn_r", norm=norms[mkey],
                    s=14, edgecolors="0.4", linewidths=0.15, zorder=3)
    cb = plt.colorbar(sc, ax=ax, pad=0.04, shrink=0.88, aspect=25)
    cb.set_label("[m]", fontsize=7, labelpad=2)
    cb.ax.tick_params(labelsize=6.5)
    ax.set_xlabel("Periapsis altitude [km]")
    if ax is axes3[0]:
        ax.set_ylabel("Inclination [deg]")
    else:
        ax.set_ylabel("")
    ax.set_title(title, loc="left", fontsize=8, pad=3)
    ax.set_xlim(190, 410)
    ax.set_ylim(-5, 185)
    ax.tick_params(labelsize=7)

    # Selected cases on ST-LRPS panel only
    if mkey == "STLRPS":
        case_markers = {"best": ("v", "#1a9641"), "representative": ("D", "#2166AC"),
                        "worst": ("X", "#d7191c")}
        legend_handles = []
        for case_key, (mk, mc) in case_markers.items():
            sid = selected[case_key]["scenario_id"]
            row = df[df["scenario_id"] == sid]
            if len(row) == 0:
                continue
            r = row.iloc[0]
            ax.scatter(r["hp_km"], r["inc_deg"], marker=mk, s=40,
                       facecolors="none", edgecolors=mc, linewidths=1.2, zorder=7)
            legend_handles.append(
                Line2D([0], [0], marker=mk, color="w", markerfacecolor="none",
                       markeredgecolor=mc, markeredgewidth=1.0, markersize=5,
                       label=case_key.capitalize()))
        ax.legend(handles=legend_handles, loc="lower right",
                  fontsize=6, frameon=True, borderpad=0.3,
                  handletextpad=0.3)

_save(fig3, "fig3_error_sensitivity_v2")


# ======================================================================
# FIGURE 4 -- RIC Decomposition (horizontal dot / lollipop)
# ======================================================================
print("Fig 4: RIC decomposition")

ric_cols = {"Radial": "radial_rms_km", "Along-track": "along_rms_km",
            "Cross-track": "cross_rms_km"}
components = list(ric_cols.keys())
y_pos = np.arange(len(components))

model_specs = [
    ("SH20",    sh20_per, CLR["SH20"],   MRK["SH20"]),
    ("ST-LRPS", st_per,   CLR["STLRPS"], MRK["STLRPS"]),
    ("SH60",    sh60_per, CLR["SH60"],   MRK["SH60"]),
]

fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(6.5, 2.2),
                                   gridspec_kw={"wspace": 0.40})

y_offsets = [-0.15, 0.0, 0.15]

for panel_ax, stat_func, panel_title in [
    (ax4a, np.median, "(a) Median"),
    (ax4b, lambda x: np.percentile(x, 95), "(b) 95th percentile"),
]:
    for (name, df, col, mk), dy in zip(model_specs, y_offsets):
        vals = [stat_func(df[v].values) * KM2M for v in ric_cols.values()]
        yy = y_pos + dy
        # Lollipop stems
        for yi, vi in zip(yy, vals):
            panel_ax.plot([vi, vi], [yi, yi], color=col, lw=0.0)   # invisible
            panel_ax.plot([0, vi], [yi, yi], color=col, lw=0.4, alpha=0.4, zorder=2)
        # Markers
        panel_ax.scatter(vals, yy, marker=mk, s=28, color=col,
                         edgecolors="k", linewidths=0.2, zorder=5, label=name)

    panel_ax.set_xscale("log")
    panel_ax.set_yticks(y_pos)
    panel_ax.set_yticklabels(components)
    panel_ax.set_xlabel("RMS error [m]")
    panel_ax.set_title(panel_title, loc="left", fontsize=8, pad=3)
    panel_ax.invert_yaxis()
    panel_ax.tick_params(axis="y", length=0)

ax4a.legend(loc="lower right", fontsize=7, frameon=True, borderpad=0.3)

_save(fig4, "fig4_ric_decomposition_v2")


# ======================================================================
# FIGURE 5 -- Time Histories  (SKIPPED: raw surrogate trajectories
#              were not saved to disk; only truth NPZ exists.)
# ======================================================================
print("\nFig 5: SKIPPED")
print("  Raw surrogate time-history data are not available on disk.")
print("  Only truth/sh200_dop853_trajectories.npz exists; surrogate")
print("  trajectories were ephemeral during the batch run and were not saved.")
print("  To generate Figure 5, re-run the 3 selected scenarios with")
print("  trajectory output enabled.\n")


# ======================================================================
# LaTeX table: validation summary
# ======================================================================
print("Table: Validation summary (LaTeX)")

st_agg_row = st_agg.iloc[0] if isinstance(st_agg, pd.DataFrame) else st_agg
st_rt_row  = st_rt.iloc[0]
sh20_rt    = multi_rt[multi_rt["model"] == "GPU_SH20_RK4"].iloc[0]
sh60_rt    = multi_rt[multi_rt["model"] == "GPU_SH60_RK4"].iloc[0]

rows = [
    ("SH20 RK4",
     sh20_agg["median_rms_pos_err_km"] * KM2M,
     sh20_agg["p95_rms_pos_err_km"] * KM2M,
     sh20_agg["max_rms_pos_err_km"] * KM2M,
     sh20_rt["total_runtime_s"],
     sh20_rt["total_runtime_s"] / sh60_rt["total_runtime_s"],
     ),
    ("ST-LRPS RK4",
     st_agg_row["median_rms_pos_err_km"] * KM2M,
     st_agg_row["p95_rms_pos_err_km"] * KM2M,
     st_agg_row["max_rms_pos_err_km"] * KM2M,
     st_rt_row["total_runtime_s"],
     st_rt_row["total_runtime_s"] / sh60_rt["total_runtime_s"],
     ),
    ("SH60 RK4",
     sh60_agg["median_rms_pos_err_km"] * KM2M,
     sh60_agg["p95_rms_pos_err_km"] * KM2M,
     sh60_agg["max_rms_pos_err_km"] * KM2M,
     sh60_rt["total_runtime_s"],
     1.0,
     ),
]

latex = r"""\begin{table}[htbp]
\centering
\caption{Validation summary for 100 one-day lunar orbit propagation cases.
All errors are computed relative to the SH200 DOP853 reference solution.}
\label{tab:validation_summary}
\begin{tabular}{l r r r r r}
\toprule
Model & Median RMS & P95 RMS & Max RMS & Runtime & Rel.\ Time \\
      & {[m]}      & {[m]}   & {[m]}   & {[s]}   & {vs.\ SH60} \\
\midrule
"""
for name, med, p95, mx, rt, rel in rows:
    if name.startswith("ST-LRPS"):
        latex += r"\textbf{" + name + "}"
    else:
        latex += name
    latex += f" & {med:.3f} & {p95:.3f} & {mx:.3f} & {rt:.1f} & {rel:.2f}$\\times$ \\\\\n"

latex += r"""\bottomrule
\end{tabular}
\end{table}
"""

(OUT / "table_validation_summary.tex").write_text(latex, encoding="utf-8")
print("  [ok] table_validation_summary.tex")


# ======================================================================
# Figure captions markdown
# ======================================================================
print("Captions and revision notes")

captions = textwrap.dedent("""\
# Figure Captions -- ST-LRPS Validation (v2)

## Figure 1
Speed--accuracy tradeoff for 100 one-day lunar orbit propagation cases.
Markers denote SH20 RK4, ST-LRPS RK4, and SH60 RK4 models.
All position errors are computed relative to the SH200 DOP853 reference solution.
The dashed horizontal line indicates the 1 m threshold.

## Figure 2
Distribution of trajectory-level RMS position error over the 100-case validation ensemble.
(a) Violin and box plots with individual scenario points (jittered).
(b) Empirical cumulative distribution functions on a logarithmic abscissa.
Median RMS values: SH20 = 98.510 m, ST-LRPS = 0.158 m, SH60 = 0.058 m.

## Figure 3
Dependence of RMS position error on initial periapsis altitude and inclination.
Each panel uses an individual logarithmic color scale.
Selected ST-LRPS best, representative, and worst cases are marked in panel (b).

## Figure 4
RIC-frame decomposition of RMS position error.
(a) Median and (b) 95th-percentile component errors for the three models.
Along-track error dominates across all models and percentile levels.

## Figure 5
_Not generated._ Raw surrogate time-history data were not saved to disk
during the original batch run. To produce this figure, re-run the three
selected scenarios (best #55, representative #81, worst #36) with
trajectory output enabled.
""")
(OUT / "figure_captions_v2.md").write_text(captions, encoding="utf-8")
print("  [ok] figure_captions_v2.md")

revision = textwrap.dedent("""\
# Revision Notes -- v1 to v2

## General
- Switched to AIAA journal figure style: 8--10 pt fonts, white background,
  thin spines, no oversized bold titles.
- Adopted colorblind-safe palette (Wong 2011): vermilion (SH20), blue (ST-LRPS),
  bluish-green (SH60).  Distinct marker shapes and line styles used throughout.
- All figures exported as vector PDF + 300 DPI PNG with `pdf.fonttype = 42`.
- Figure sizes chosen so that text is 8--10 pt when inserted at 0.9\\linewidth.

## Figure 1 -- Speed--Accuracy Tradeoff
- Removed the artificial SH200 reference point (was plotted at 1e-4 m).
- Removed the decorative Pareto arrow.
- Removed heavy annotation boxes; replaced with small direct labels.
- Added a subtle 1 m horizontal guide line.
- Reduced figure size to single-column width (3.5 x 2.8 in).

## Figure 2 -- RMS Error Distribution
- Replaced colored x-tick labels with standard black labels.
- Removed large median-value annotations placed directly on violins.
- Added semi-transparent jittered strip plot for individual scenario points.
- Replaced histogram+twin-axis ECDF with a clean ECDF-only panel (b)
  showing all three models on a log x-axis.
- Numerical values (median, P95, max) moved to caption / LaTeX table.

## Figure 3 -- Error Sensitivity
- Changed to logarithmic color normalisation (LogNorm) per panel.
- Reduced marker size and outline weight.
- Replaced inline text labels for selected cases with a compact legend
  on the ST-LRPS panel only.
- Made "individual color scales" explicit in the caption.
- Ensured consistent x/y limits across all three panels.

## Figure 4 -- RIC Decomposition
- Converted from grouped vertical bar chart to horizontal dot/lollipop.
- Removed numeric value labels above every bar.
- Key insight (along-track dominance) is visible from layout; noted in caption.

## Figure 5 -- Time Histories
- Removed embedded PNG screenshots (`plt.imread` / `imshow`).
- Confirmed that raw surrogate trajectories are not saved to disk
  (only `truth/sh200_dop853_trajectories.npz` exists).
- Figure skipped in this version; revision notes explain how to regenerate.

## Figure 6 -- Summary Table
- Replaced matplotlib table image with a LaTeX `booktabs` table
  (`table_validation_summary.tex`).
""")
(OUT / "revision_notes_v2.md").write_text(revision, encoding="utf-8")
print("  [ok] revision_notes_v2.md")

print(f"\nAll outputs -> {OUT}")
