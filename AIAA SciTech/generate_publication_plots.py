#!/usr/bin/env python3
"""
Publication-quality figures for AIAA SciTech extended abstract.

ST-LRPS surrogate model validation – 100-case one-day lunar orbit batch.
Compares GPU SH20 RK4, GPU SH60 RK4, and GPU ST-LRPS RK4 against
SH200 DOP853 reference.

Data sources:
  - stlrps_ders_cikisi_100/   → ST-LRPS per-scenario metrics (correct model)
  - gpu_sh60_sh20_stlrps_100/ → SH20 & SH60 per-scenario metrics (same scenarios)
"""

import json, pathlib, textwrap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────
RESULTS = pathlib.Path(r"c:\Users\ayber\Desktop\LUNAR_SIMULATION\results")
STLRPS  = RESULTS / "stlrps_ders_cikisi_100"        # correct ST-LRPS
MULTI   = RESULTS / "gpu_sh60_sh20_stlrps_100"       # SH20 + SH60
AIAA    = pathlib.Path(r"c:\Users\ayber\Desktop\LUNAR_SIMULATION\AIAA SciTech")
OUT     = AIAA / "publication_plots"
OUT.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────
# Load data
# ──────────────────────────────────────────────────────────────────────
# ST-LRPS (correct checkpoint)
st_per  = pd.read_csv(STLRPS / "metrics" / "gpu_batch_per_scenario_metrics.csv")
st_agg  = pd.read_csv(STLRPS / "metrics" / "gpu_batch_aggregate_metrics.csv")
st_rt   = pd.read_csv(STLRPS / "metrics" / "gpu_batch_runtime_metrics.csv")
scenarios = pd.read_csv(STLRPS / "scenarios.csv")
with open(STLRPS / "metrics" / "stlrps_selected_scenarios.json") as f:
    selected = json.load(f)

# SH20 + SH60 per-scenario (same 100 scenarios, seed=42)
multi_per = pd.read_csv(MULTI / "metrics" / "gpu_batch_per_scenario_metrics.csv")
multi_agg = pd.read_csv(MULTI / "metrics" / "gpu_batch_aggregate_metrics.csv")
multi_rt  = pd.read_csv(MULTI / "metrics" / "gpu_batch_runtime_metrics.csv")

# Extract SH20 and SH60 subsets
sh20_per = multi_per[multi_per["model"] == "GPU_SH20_RK4"].copy()
sh60_per = multi_per[multi_per["model"] == "GPU_SH60_RK4"].copy()
sh20_agg = multi_agg[multi_agg["model"] == "GPU_SH20_RK4"].iloc[0]
sh60_agg = multi_agg[multi_agg["model"] == "GPU_SH60_RK4"].iloc[0]

# Merge orbital elements
st_per = st_per.merge(scenarios[["scenario_id", "hp_km", "ha_km", "inc_deg"]],
                      on="scenario_id", how="left", suffixes=("", "_scen"))
if "hp_km_scen" in st_per.columns:
    st_per["hp_km"] = st_per["hp_km_scen"]
    st_per["ha_km"] = st_per["ha_km_scen"]
    st_per["inc_deg"] = st_per["inc_deg_scen"]

KM2M = 1000.0

# ──────────────────────────────────────────────────────────────────────
# Global style
# ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "DejaVu Serif"],
    "font.size":          11,
    "axes.labelsize":     12,
    "axes.titlesize":     13,
    "axes.titleweight":   "bold",
    "xtick.labelsize":    10,
    "ytick.labelsize":    10,
    "legend.fontsize":    10,
    "legend.framealpha":  0.92,
    "legend.edgecolor":   "#cccccc",
    "figure.dpi":         300,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.12,
    "axes.grid":          True,
    "grid.alpha":         0.25,
    "grid.linewidth":     0.5,
    "grid.linestyle":     "--",
    "axes.linewidth":     0.9,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "lines.linewidth":    1.4,
    "lines.markersize":   7,
    "pdf.fonttype":       42,
    "ps.fonttype":        42,
    "figure.facecolor":   "white",
    "axes.facecolor":     "#fafafa",
})

# Color palette
C = {
    "ST-LRPS":  "#2166AC",   # deep blue
    "SH20":     "#B2182B",   # dark red
    "SH60":     "#E66101",   # orange
    "SH200":    "#4DAF4A",   # green (reference)
    "median":   "#E66101",
    "p95":      "#5E3C99",
    "max":      "#1B7837",
    "radial":   "#4393C3",
    "along":    "#D6604D",
    "cross":    "#92C5DE",
    "best":     "#1a9641",
    "repr":     "#2166AC",
    "worst":    "#d7191c",
}


def save(fig, name):
    """Save figure as both PNG and PDF."""
    fig.savefig(OUT / f"{name}.png", facecolor="white")
    fig.savefig(OUT / f"{name}.pdf", facecolor="white")
    plt.close(fig)
    print(f"  [OK] {name}.png / .pdf")


# ══════════════════════════════════════════════════════════════════════
# FIGURE 1 - Speed-Accuracy Tradeoff (all models)
# ══════════════════════════════════════════════════════════════════════
print("Figure 1: Speed-accuracy tradeoff")

models_data = {
    "SH20": {
        "runtime": multi_rt[multi_rt["model"] == "GPU_SH20_RK4"]["total_runtime_s"].iloc[0],
        "median_m": sh20_agg["median_rms_pos_err_km"] * KM2M,
        "color": C["SH20"], "marker": "s", "size": 130,
    },
    "SH60": {
        "runtime": multi_rt[multi_rt["model"] == "GPU_SH60_RK4"]["total_runtime_s"].iloc[0],
        "median_m": sh60_agg["median_rms_pos_err_km"] * KM2M,
        "color": C["SH60"], "marker": "D", "size": 130,
    },
    "ST-LRPS": {
        "runtime": st_rt["total_runtime_s"].iloc[0],
        "median_m": st_agg["median_rms_pos_err_km"].iloc[0] * KM2M,
        "color": C["ST-LRPS"], "marker": "o", "size": 180,
    },
    "SH200\n(reference)": {
        "runtime": st_rt["truth_total_runtime_s"].iloc[0],
        "median_m": 1e-4,
        "color": C["SH200"], "marker": "^", "size": 130,
    },
}

fig1, ax1 = plt.subplots(figsize=(7, 5))

for label, d in models_data.items():
    ax1.scatter(d["runtime"], d["median_m"],
                color=d["color"], marker=d["marker"], s=d["size"],
                edgecolors="k", linewidths=0.6, zorder=5)

# Annotations with smart offset
offsets = {
    "SH20":             (12, 10),
    "SH60":             (-15, -18),
    "ST-LRPS":          (12, -18),
    "SH200\n(reference)": (-15, 10),
}
ha_map = {"SH20": "left", "SH60": "right", "ST-LRPS": "left", "SH200\n(reference)": "right"}

for label, d in models_data.items():
    xoff, yoff = offsets[label]
    ax1.annotate(label, (d["runtime"], d["median_m"]),
                 textcoords="offset points", xytext=(xoff, yoff),
                 ha=ha_map[label], va="center", fontsize=10, fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#bbbbbb",
                           alpha=0.9, lw=0.6))

ax1.set_yscale("log")
ax1.set_xlabel("Total runtime for 100 scenarios [s]")
ax1.set_ylabel("Median RMS position error [m]")
ax1.set_xlim(0, 5500)
ax1.set_ylim(5e-5, 500)
ax1.set_yticks([1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100])
ax1.set_yticklabels(["0.0001", "0.001", "0.01", "0.1", "1", "10", "100"])

# Pareto arrow
ax1.annotate("", xy=(200, 5e-5), xytext=(1200, 5e-3),
             arrowprops=dict(arrowstyle="-|>", color="#999999", lw=1.2, ls="--"))
ax1.text(300, 1.5e-4, "Pareto ideal", fontsize=9, color="#999999", style="italic")

fig1.tight_layout()
save(fig1, "fig1_speed_accuracy_tradeoff")


# ══════════════════════════════════════════════════════════════════════
# FIGURE 2 - RMS Error Distribution: Boxplot + Histogram (3 models)
# ══════════════════════════════════════════════════════════════════════
print("Figure 2: RMS error distribution comparison")

fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(10, 4.5),
                                   gridspec_kw={"width_ratios": [1.2, 1], "wspace": 0.35})

# --- Left panel: side-by-side violin/box ---
data_box = {
    "SH20":    sh20_per["rms_pos_err_km"].values * KM2M,
    "ST-LRPS": st_per["rms_pos_err_km"].values * KM2M,
    "SH60":    sh60_per["rms_pos_err_km"].values * KM2M,
}
labels_box = ["SH20", "ST-LRPS", "SH60"]
colors_box = [C["SH20"], C["ST-LRPS"], C["SH60"]]
positions_box = [1, 2, 3]

vp = ax2a.violinplot([data_box[l] for l in labels_box], positions=positions_box,
                     showmedians=False, showextrema=False, widths=0.7)
for body, col in zip(vp["bodies"], colors_box):
    body.set_facecolor(col)
    body.set_alpha(0.25)
    body.set_edgecolor(col)
    body.set_linewidth(0.8)

bp = ax2a.boxplot([data_box[l] for l in labels_box], positions=positions_box,
                  widths=0.3, patch_artist=True, showfliers=True,
                  flierprops=dict(marker=".", markersize=3, alpha=0.5))
for patch, col in zip(bp["boxes"], colors_box):
    patch.set_facecolor(col)
    patch.set_alpha(0.65)
for med in bp["medians"]:
    med.set_color("white")
    med.set_linewidth(2)

# Add median annotations
for i, l in enumerate(labels_box):
    med_val = np.median(data_box[l])
    ax2a.text(positions_box[i], med_val, f"  {med_val:.3f} m",
              va="center", ha="left", fontsize=8, fontweight="bold",
              color=colors_box[i])

ax2a.set_yscale("log")
ax2a.set_xticks(positions_box)
ax2a.set_xticklabels(labels_box, fontweight="bold")
ax2a.set_ylabel("RMS position error [m]")
ax2a.set_title("(a) Error Distribution", loc="left")

# Color x-tick labels
for tick_label, col in zip(ax2a.get_xticklabels(), colors_box):
    tick_label.set_color(col)

# --- Right panel: ST-LRPS histogram + ECDF ---
rms_st = st_per["rms_pos_err_km"].values * KM2M
med_val  = np.median(rms_st)
p95_val  = np.percentile(rms_st, 95)
max_val  = np.max(rms_st)

ax2b_twin = ax2b.twinx()

bins = np.linspace(0, 1.1, 30)
ax2b.hist(rms_st, bins=bins, color=C["ST-LRPS"], alpha=0.55, edgecolor="white",
          linewidth=0.5, zorder=3, label="Histogram")

# ECDF on twin axis
sorted_err = np.sort(rms_st)
ecdf_y = np.arange(1, len(sorted_err) + 1) / len(sorted_err)
ax2b_twin.step(sorted_err, ecdf_y, where="post", color=C["ST-LRPS"], lw=2.0,
               ls="-", alpha=0.9, zorder=4, label="ECDF")
ax2b_twin.set_ylabel("ECDF", color=C["ST-LRPS"])
ax2b_twin.set_ylim(0, 1.08)
ax2b_twin.spines["right"].set_visible(True)
ax2b_twin.tick_params(axis="y", colors=C["ST-LRPS"])

# Vertical markers
for val, lbl, col, ls in [
    (med_val, f"Median = {med_val:.3f} m", C["median"], "-"),
    (p95_val, f"P95 = {p95_val:.3f} m",    C["p95"],   "--"),
    (max_val, f"Max = {max_val:.3f} m",     C["max"],   ":"),
]:
    ax2b.axvline(val, color=col, ls=ls, lw=1.5, zorder=5, label=lbl)

ax2b.set_xlabel("RMS position error [m]")
ax2b.set_ylabel("Number of scenarios")
ax2b.set_title("(b) ST-LRPS Detail", loc="left")
ax2b.legend(loc="upper right", fontsize=8.5, framealpha=0.9)
ax2b.set_xlim(0, 1.1)

fig2.tight_layout()
save(fig2, "fig2_rms_error_distribution")


# ══════════════════════════════════════════════════════════════════════
# FIGURE 3 - Error Sensitivity: Altitude vs Inclination (3 models)
# ══════════════════════════════════════════════════════════════════════
print("Figure 3: Error sensitivity - altitude vs inclination")

fig3, axes3 = plt.subplots(1, 3, figsize=(12, 4.2),
                            gridspec_kw={"wspace": 0.40})

panel_data = [
    ("(a) SH20 RK4",    sh20_per, C["SH20"],   (0, 600)),
    ("(b) ST-LRPS RK4", st_per,   C["ST-LRPS"], (0, 1.0)),
    ("(c) SH60 RK4",    sh60_per, C["SH60"],    (0, 2.5)),
]

for ax, (title, df, color, vlims) in zip(axes3, panel_data):
    err_m = df["rms_pos_err_km"].values * KM2M
    sc = ax.scatter(df["hp_km"], df["inc_deg"], c=err_m,
                    cmap="RdYlGn_r", s=35, edgecolors="k", linewidths=0.25,
                    vmin=vlims[0], vmax=vlims[1], zorder=3)
    cbar = plt.colorbar(sc, ax=ax, pad=0.03, shrink=0.85)
    cbar.set_label("RMS error [m]", fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    ax.set_xlabel("Periapsis altitude [km]", fontsize=10)
    ax.set_ylabel("Inclination [deg]", fontsize=10)
    ax.set_title(title, loc="left", fontsize=11)
    ax.set_xlim(190, 410)
    ax.set_ylim(-5, 185)
    ax.tick_params(labelsize=9)

    # Annotate selected scenarios on ST-LRPS panel
    if "ST-LRPS" in title:
        sel_markers = {"best": "v", "representative": "D", "worst": "X"}
        sel_colors  = {"best": C["best"], "representative": C["repr"], "worst": C["worst"]}
        for key in ["best", "representative", "worst"]:
            sid = selected[key]["scenario_id"]
            row = df[df["scenario_id"] == sid]
            if len(row) == 0:
                continue
            row = row.iloc[0]
            ax.scatter(row["hp_km"], row["inc_deg"],
                       marker=sel_markers[key], s=100, facecolors="none",
                       edgecolors=sel_colors[key], linewidths=2.0, zorder=6)
            ax.annotate(f" {key.capitalize()}", (row["hp_km"], row["inc_deg"]),
                        fontsize=7, color=sel_colors[key], fontweight="bold",
                        ha="left", va="center")

fig3.tight_layout()
save(fig3, "fig3_error_sensitivity_alt_inc")


# ══════════════════════════════════════════════════════════════════════
# FIGURE 4 - RIC Error Decomposition (3 models grouped)
# ══════════════════════════════════════════════════════════════════════
print("Figure 4: RIC error decomposition (3-model comparison)")

ric_cols = {"Radial": "radial_rms_km", "Along-track": "along_rms_km", "Cross-track": "cross_rms_km"}
model_list = [
    ("SH20",    sh20_per, C["SH20"]),
    ("ST-LRPS", st_per,   C["ST-LRPS"]),
    ("SH60",    sh60_per, C["SH60"]),
]

fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(10, 4.5),
                                   gridspec_kw={"width_ratios": [1, 1], "wspace": 0.30})

# --- Left: median RIC ---
labels_ric = list(ric_cols.keys())
x = np.arange(len(labels_ric))
n_models = len(model_list)
width = 0.22
offsets_ric = np.linspace(-(n_models-1)*width/2, (n_models-1)*width/2, n_models)

for i, (name, df, col) in enumerate(model_list):
    vals = [np.median(df[v]) * KM2M for v in ric_cols.values()]
    bars = ax4a.bar(x + offsets_ric[i], vals, width, label=name,
                    color=col, alpha=0.75, edgecolor="white", linewidth=0.5, zorder=3)
    for bar in bars:
        h = bar.get_height()
        if h < 10:
            ax4a.text(bar.get_x() + bar.get_width()/2, h + 0.002,
                      f"{h:.3f}", ha="center", va="bottom", fontsize=6.5, rotation=0)
        else:
            ax4a.text(bar.get_x() + bar.get_width()/2, h + 1,
                      f"{h:.1f}", ha="center", va="bottom", fontsize=6.5, rotation=0)

ax4a.set_xticks(x)
ax4a.set_xticklabels(labels_ric)
ax4a.set_ylabel("Median RMS error [m]")
ax4a.set_title("(a) Median RIC Error", loc="left")
ax4a.legend(loc="upper left", fontsize=9)
ax4a.set_yscale("log")

# --- Right: P95 RIC ---
for i, (name, df, col) in enumerate(model_list):
    vals = [np.percentile(df[v], 95) * KM2M for v in ric_cols.values()]
    bars = ax4b.bar(x + offsets_ric[i], vals, width, label=name,
                    color=col, alpha=0.75, edgecolor="white", linewidth=0.5, zorder=3)
    for bar in bars:
        h = bar.get_height()
        if h < 10:
            ax4b.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                      f"{h:.3f}", ha="center", va="bottom", fontsize=6.5, rotation=0)
        else:
            ax4b.text(bar.get_x() + bar.get_width()/2, h + 3,
                      f"{h:.1f}", ha="center", va="bottom", fontsize=6.5, rotation=0)

ax4b.set_xticks(x)
ax4b.set_xticklabels(labels_ric)
ax4b.set_ylabel("P95 RMS error [m]")
ax4b.set_title("(b) 95th Percentile RIC Error", loc="left")
ax4b.legend(loc="upper left", fontsize=9)
ax4b.set_yscale("log")

fig4.tight_layout()
save(fig4, "fig4_ric_error_decomposition")


# ══════════════════════════════════════════════════════════════════════
# FIGURE 5 - Selected Case Time Histories (3 panels, actual data)
# ══════════════════════════════════════════════════════════════════════
print("Figure 5: Selected case time histories")

PLOTS_ST = STLRPS / "plots"
cases = [
    ("best",           "(a) Best Case",           C["best"]),
    ("representative", "(b) Representative Case",  C["repr"]),
    ("worst",          "(c) Worst Case",            C["worst"]),
]

# Use existing per-timestep plots from the correct ST-LRPS run
# Vertical stack: 3 rows x 1 column for readability
fig5, axes5 = plt.subplots(3, 1, figsize=(9, 12),
                            gridspec_kw={"hspace": 0.35})

for ax, (key, panel_title, col) in zip(axes5, cases):
    png_path = PLOTS_ST / f"selected_{key}_position_error_all_models.png"
    if png_path.exists():
        img = plt.imread(str(png_path))
        ax.imshow(img, aspect="auto")
    ax.axis("off")
    meta = selected[key]
    sid  = meta["scenario_id"]
    rms  = meta["rms_pos_err_km"] * KM2M
    maxe = meta["max_pos_err_km"] * KM2M
    hp   = meta["hp_km"]
    inc  = meta["inc_deg"]
    ax.set_title(f"{panel_title}:  Scenario #{sid}  |  hp = {hp:.0f} km,  i = {inc:.0f} deg"
                 f"  |  RMS = {rms:.2f} m,  max = {maxe:.2f} m",
                 fontsize=10, pad=8, color=col, fontweight="bold", loc="left")

fig5.tight_layout()
save(fig5, "fig5_selected_case_time_history")


# ══════════════════════════════════════════════════════════════════════
# FIGURE 6 - Summary comparison table (as a figure)
# ══════════════════════════════════════════════════════════════════════
print("Figure 6: Summary comparison table")

# Build table data
table_data = []
for name, agg_row, rt_df, per_df, col in [
    ("GPU SH20 RK4",
     sh20_agg,
     multi_rt[multi_rt["model"] == "GPU_SH20_RK4"].iloc[0],
     sh20_per, C["SH20"]),
    ("GPU ST-LRPS RK4",
     st_agg.iloc[0] if isinstance(st_agg, pd.DataFrame) else st_agg,
     st_rt.iloc[0],
     st_per, C["ST-LRPS"]),
    ("GPU SH60 RK4",
     sh60_agg,
     multi_rt[multi_rt["model"] == "GPU_SH60_RK4"].iloc[0],
     sh60_per, C["SH60"]),
]:
    med_m  = agg_row["median_rms_pos_err_km"] * KM2M
    p95_m  = agg_row["p95_rms_pos_err_km"] * KM2M
    max_m  = agg_row["max_rms_pos_err_km"] * KM2M
    rt_s   = rt_df["total_runtime_s"]
    rt_per = rt_df["runtime_per_scenario_s"]
    speedup = rt_df["speedup_vs_truth_total"]
    table_data.append([
        name,
        f"{med_m:.3f}",
        f"{p95_m:.3f}",
        f"{max_m:.3f}",
        f"{rt_s:.1f}",
        f"{rt_per:.2f}",
        f"{speedup:.1f}x",
    ])

col_labels = ["Model", "Median\nRMS [m]", "P95\nRMS [m]", "Max\nRMS [m]",
              "Total\nRuntime [s]", "Per-Scenario\nRuntime [s]", "Speedup\nvs Truth"]

fig6, ax6 = plt.subplots(figsize=(9, 2.8))
ax6.axis("off")
ax6.set_title("Validation Summary: 100 One-Day Lunar Orbit Propagation Cases",
              fontsize=13, fontweight="bold", pad=20)

tbl = ax6.table(cellText=table_data, colLabels=col_labels,
                loc="center", cellLoc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1.0, 1.8)

# Style header
for j in range(len(col_labels)):
    cell = tbl[0, j]
    cell.set_facecolor("#2166AC")
    cell.set_text_props(color="white", fontweight="bold", fontsize=9)

# Color model name cells
model_colors = [C["SH20"], C["ST-LRPS"], C["SH60"]]
for i in range(3):
    cell = tbl[i + 1, 0]
    cell.set_text_props(fontweight="bold", color=model_colors[i])
    # Highlight ST-LRPS row
    if i == 1:
        for j in range(len(col_labels)):
            tbl[i + 1, j].set_facecolor("#e8f0fe")

fig6.tight_layout()
save(fig6, "fig6_summary_table")


# ══════════════════════════════════════════════════════════════════════
# Generate markdown listing
# ══════════════════════════════════════════════════════════════════════
print("\nGenerating figure listing markdown...")

rms_st_m = st_per["rms_pos_err_km"].values * KM2M
med_val = np.median(rms_st_m)
p95_val = np.percentile(rms_st_m, 95)
max_val = np.max(rms_st_m)

markdown = f"""# Publication Figures - ST-LRPS Validation (AIAA SciTech)

Generated from 100-case one-day lunar orbit validation batch.
Three-model comparison: GPU SH20 RK4, GPU ST-LRPS RK4, GPU SH60 RK4
vs SH200 DOP853 reference.

---

## Data Sources

| Source | Description |
|--------|-------------|
| `stlrps_ders_cikisi_100/` | ST-LRPS per-scenario metrics (correct model checkpoint) |
| `gpu_sh60_sh20_stlrps_100/` | SH20 & SH60 per-scenario metrics (same 100 scenarios, seed=42) |

---

## Figure 1: Speed-Accuracy Tradeoff
`fig1_speed_accuracy_tradeoff.{{png,pdf}}`

Four models compared on total runtime vs median RMS position error.
ST-LRPS achieves sub-meter accuracy ({med_val:.3f} m median) at runtime
comparable to the lowest-fidelity SH20 model.

---

## Figure 2: RMS Error Distribution
`fig2_rms_error_distribution.{{png,pdf}}`

(a) Violin+box plot comparing SH20, ST-LRPS, SH60 error distributions.
(b) ST-LRPS detailed histogram + ECDF.
Median = {med_val:.3f} m, P95 = {p95_val:.3f} m, Max = {max_val:.3f} m.

---

## Figure 3: Error Sensitivity
`fig3_error_sensitivity_alt_inc.{{png,pdf}}`

Three-panel scatter: periapsis altitude vs inclination colored by RMS error.
Shows SH20 errors ~100 m, ST-LRPS ~0.16 m, SH60 ~0.06 m.

---

## Figure 4: RIC Error Decomposition
`fig4_ric_error_decomposition.{{png,pdf}}`

Grouped bar chart comparing Radial/Along-track/Cross-track error
for all three models. Along-track dominates for all models.

---

## Figure 5: Selected Case Time Histories
`fig5_selected_case_time_history.{{png,pdf}}`

Three panels from actual simulation data:
- (a) Best: Scenario #{selected['best']['scenario_id']}
- (b) Representative: Scenario #{selected['representative']['scenario_id']}
- (c) Worst: Scenario #{selected['worst']['scenario_id']}

---

## Figure 6: Summary Comparison Table
`fig6_summary_table.{{png,pdf}}`

Publication-ready table with key metrics for all three models.

---

## Notes

- All position errors in meters (converted from km).
- 300 DPI, TrueType fonts (PDF type 42).
- SH20 and SH60 data from `gpu_sh60_sh20_stlrps_100` run
  (same 100 scenarios, seed=42, same SH200 DOP853 reference).
"""

(OUT / "figure_listing.md").write_text(markdown, encoding="utf-8")
print("  [OK] figure_listing.md")
print(f"\nAll outputs saved to: {OUT}")
