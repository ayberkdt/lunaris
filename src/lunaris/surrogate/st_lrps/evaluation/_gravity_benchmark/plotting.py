# -*- coding: utf-8 -*-
"""Internal module of the lunar gravity-model benchmark harness.

Part of :mod:`lunaris.surrogate.st_lrps.evaluation.compare_gravity_models`;
this is an implementation detail, not a public API. See that module's
docstring for CLI usage.
"""
from __future__ import annotations

import argparse
import math
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from lunaris.common.constants import R_MOON

# --- intra-package wiring (auto-generated split) ---
from .types import (
    BatchModelResult,
    Scenario,
    TruthTrajectorySet,
    compute_ric_errors,
    interpolate_state_to_times,
)
from .compute import (
    _gpu_rk4_dt_values,
)
from .results_io import (
    _ensure_dir,
)

# =============================================================================
# Plotting helpers
# =============================================================================

# =============================================================================
# Publication-grade plotting style (visualization only; no numeric impact)
# =============================================================================
# Consistent, professional styling shared by every Orbit-Level Benchmark figure:
#   * ST-LRPS uses a single distinctive accent colour + star marker and a heavier
#     line so it always stands out.
#   * Spherical-harmonic baselines share a degree-ordered colour family that runs
#     warm (low degree) -> cool/dark (high degree), so SH20 and SH200 are easy to
#     tell apart and the ordering reads naturally.
#   * Helpers pick a sensible display unit (km/m/cm) and never leave a blank plot.

_ST_LRPS_COLOR = "#8E2DC4"   # deep violet accent — ST-LRPS stands out
_TRUTH_COLOR = "#15202B"     # near-black reference
_FALLBACK_COLOR = "#7A8699"

# Degree -> colour anchors (interpolated in RGB for arbitrary degrees).
_SH_DEGREE_ANCHORS = [
    (20,  "#D1495B"),  # muted red
    (30,  "#E8833A"),  # warm amber
    (60,  "#C9A227"),  # gold
    (80,  "#6C8EBF"),  # blue-gray
    (100, "#3D5A80"),  # slate blue
    (120, "#33518A"),  # deeper slate
    (160, "#23386B"),  # dark blue
    (200, "#1B2A41"),  # charcoal navy
]
_KNOWN_DEGREE_ORDER = [20, 30, 60, 80, 100, 120, 160, 200]
_SH_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "<", ">"]
_SH_DEGREE_RE = re.compile(r"SH(\d+)")

# Legacy override table (kept for the older CPU-mode plots / batch-rk4 panels).
MODEL_COLORS = {
    "st_lrps_batch_rk4": _ST_LRPS_COLOR, "sh200_rk4": "#1B2A41",
}


def _hex_to_rgb(h: str) -> Tuple[float, float, float]:
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))  # type: ignore[return-value]


def _rgb_to_hex(rgb: Tuple[float, float, float]) -> str:
    return "#%02x%02x%02x" % tuple(int(round(max(0.0, min(1.0, c)) * 255)) for c in rgb)


def _model_degree(model: str) -> Optional[int]:
    m = str(model).upper()
    if "ST_LRPS" in m or "ST-LRPS" in m:
        return None
    found = _SH_DEGREE_RE.search(m)
    return int(found.group(1)) if found else None


def _is_stlrps(model: str) -> bool:
    m = str(model).upper()
    return "ST_LRPS" in m or "ST-LRPS" in m


def _sh_degree_color(deg: int) -> str:
    anchors = _SH_DEGREE_ANCHORS
    if deg <= anchors[0][0]:
        return anchors[0][1]
    if deg >= anchors[-1][0]:
        return anchors[-1][1]
    for (d0, c0), (d1, c1) in zip(anchors, anchors[1:]):
        if d0 <= deg <= d1:
            f = (deg - d0) / (d1 - d0) if d1 > d0 else 0.0
            r0, r1 = _hex_to_rgb(c0), _hex_to_rgb(c1)
            return _rgb_to_hex(tuple(a + (b - a) * f for a, b in zip(r0, r1)))
    return anchors[-1][1]


def model_color(model: str) -> str:
    """Consistent colour for a model across every figure."""
    if _is_stlrps(model):
        return _ST_LRPS_COLOR
    deg = _model_degree(model)
    if deg is not None:
        return _sh_degree_color(deg)
    return MODEL_COLORS.get(str(model), MODEL_COLORS.get(str(model).lower(), _FALLBACK_COLOR))


def _color(m: str) -> str:  # backwards-compatible alias
    return model_color(m)


def model_marker(model: str) -> str:
    if _is_stlrps(model):
        return "*"
    deg = _model_degree(model)
    if deg is None:
        return "o"
    if deg in _KNOWN_DEGREE_ORDER:
        return _SH_MARKERS[_KNOWN_DEGREE_ORDER.index(deg) % len(_SH_MARKERS)]
    return _SH_MARKERS[deg % len(_SH_MARKERS)]


def model_linewidth(model: str) -> float:
    return 2.8 if _is_stlrps(model) else 1.6


def model_marker_size(model: str) -> float:
    return 210.0 if _is_stlrps(model) else 90.0


def model_zorder(model: str) -> int:
    return 7 if _is_stlrps(model) else 3


def display_label(model: str) -> str:
    """Human label, e.g. GPU_SH20_RK4 -> SH20, GPU_ST_LRPS_RK4 -> ST-LRPS."""
    m = str(model)
    dt_label: Optional[str] = None
    if "_DT" in m.upper():
        head, tail = re.split(r"_DT", m, maxsplit=1, flags=re.IGNORECASE)
        m = head
        dt_label = tail
    if _is_stlrps(m):
        label = "ST-LRPS"
    else:
        label = m.replace("GPU_", "").replace("_RK4", "").upper()
    return f"{label} dt{dt_label}" if dt_label else label


def select_length_unit(max_km: float) -> Tuple[str, float]:
    """Pick a readable display unit for a length given the largest value in km.

    Returns ``(unit_label, multiplier)`` where ``display = value_km * multiplier``.
    CSV/metric units are never changed — this only affects plotting.
    """
    try:
        v = float(max_km)
    except (TypeError, ValueError):
        return ("km", 1.0)
    if not math.isfinite(v) or v <= 0.0:
        return ("km", 1.0)
    if v < 1.0e-3:
        return ("cm", 1.0e5)   # 1 km = 1e5 cm
    if v < 1.0e-2:
        return ("m", 1.0e3)    # 1 km = 1e3 m
    return ("km", 1.0)


def _fmt_km(value: Any) -> str:
    """Format a kilometre value for tables/console without collapsing to zero.

    Fixed ``%.4f`` formatting renders any error below 0.5 m as ``"0.0000"``,
    which misleadingly reads as an exact-zero metric on short/accurate runs.
    Sub-metre values are shown in scientific notation instead (still in km).
    """
    try:
        x = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(x):
        return "n/a"
    a = abs(x)
    if a == 0.0:
        return "0"
    if a < 1.0e-3:        # below ~1 m: %.4f would print "0.0000"
        return f"{x:.3e}"
    return f"{x:.4f}"


def _finite_positive(values: Sequence[float]) -> List[float]:
    out = []
    for v in values:
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(f) and f > 0.0:
            out.append(f)
    return out


def _should_log(values: Sequence[float], ratio: float = 50.0) -> bool:
    """True when positive values span more than ``ratio`` orders-of-magnitude."""
    pos = _finite_positive(values)
    return len(pos) >= 2 and (max(pos) / min(pos)) > ratio


# Theme palettes (used by figure styling helpers + rcParams).
_PLOT_THEMES = {
    "report_light": dict(bg="#FFFFFF", ax_bg="#FFFFFF", text="#1A1F29",
                         grid="#D7DEE8", edge="#3A4452", muted="#5A6675", accent="#2A9D8F"),
    "technical_dark": dict(bg="#0E1116", ax_bg="#11151C", text="#E8ECF8",
                           grid="#2A3340", edge="#8A98AD", muted="#9AA7C7", accent="#35D0FF"),
}
_ACTIVE_PLOT_THEME = _PLOT_THEMES["report_light"]


def apply_plot_theme(theme: str) -> None:
    """Central publication-grade plotting style for validation figures."""
    global _ACTIVE_PLOT_THEME
    th = _PLOT_THEMES.get(str(theme), _PLOT_THEMES["report_light"])
    _ACTIVE_PLOT_THEME = th
    plt.style.use("dark_background" if theme == "technical_dark" else "default")
    plt.rcParams.update({
        "figure.facecolor": th["bg"],
        "axes.facecolor": th["ax_bg"],
        "savefig.facecolor": th["bg"],
        "text.color": th["text"],
        "axes.labelcolor": th["text"],
        "axes.edgecolor": th["edge"],
        "axes.linewidth": 0.9,
        "xtick.color": th["text"],
        "ytick.color": th["text"],
        "figure.dpi": 120,
        "savefig.dpi": 220,
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "legend.fontsize": 9.5,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.grid": True,
        "grid.color": th["grid"],
        "grid.alpha": 0.55 if theme == "technical_dark" else 0.9,
        "grid.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 1.8,
        "legend.frameon": False,
    })


def _style_ax(ax: Any, *, title: Optional[str] = None, xlabel: Optional[str] = None,
              ylabel: Optional[str] = None, subtitle: Optional[str] = None) -> None:
    th = _ACTIVE_PLOT_THEME
    if title:
        ax.set_title(title, color=th["text"], pad=30 if subtitle else 8)
    if subtitle:
        ax.text(0.0, 1.015, subtitle, transform=ax.transAxes, fontsize=9,
                color=th["muted"], va="bottom", ha="left")
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.5, linewidth=0.7)
    for spine in ("top", "right"):
        if spine in ax.spines:
            ax.spines[spine].set_visible(False)


def _legend(ax: Any, *, outside: bool = False, loc: str = "best", ncol: int = 1) -> Any:
    th = _ACTIVE_PLOT_THEME
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return None
    if outside:
        return ax.legend(handles, labels, loc="center left", bbox_to_anchor=(1.02, 0.5),
                         frameon=False, ncol=ncol)
    leg = ax.legend(handles, labels, loc=loc, frameon=True, framealpha=0.9, ncol=ncol)
    if leg is not None:
        leg.get_frame().set_edgecolor(th["grid"])
        leg.get_frame().set_facecolor(th["ax_bg"])
        leg.get_frame().set_linewidth(0.6)
    return leg


def _legend_outside(ax: Any) -> None:  # backwards-compatible alias
    _legend(ax, outside=True)


def _empty_note(ax: Any, text: str) -> None:
    """Stamp an explanatory note on an otherwise-blank plot."""
    ax.text(0.5, 0.5, text, ha="center", va="center", transform=ax.transAxes,
            fontsize=11, color=_ACTIVE_PLOT_THEME["muted"], style="italic", wrap=True)


def _highlight_ticklabels(ax: Any, labels: Sequence[str], axis: str = "y") -> None:
    """Bold + accent the ST-LRPS tick label on a category axis."""
    th = _ACTIVE_PLOT_THEME
    ticklabels = ax.get_yticklabels() if axis == "y" else ax.get_xticklabels()
    for lbl, text in zip(ticklabels, labels):
        if _is_stlrps(text):
            lbl.set_color(_ST_LRPS_COLOR)
            lbl.set_fontweight("bold")
        else:
            lbl.set_color(th["text"])


def _model_sort_key(model: str) -> Tuple[int, int]:
    m = str(model).upper()
    if "SH200" in m:
        return (0, 200)
    if "SH160" in m:
        return (1, 160)
    if "SH120" in m:
        return (2, 120)
    if "SH60" in m:
        return (3, 60)
    if "SH20" in m:
        return (4, 20)
    if "ST_LRPS" in m:
        return (5, 0)
    return (9, 0)


def plot_selected_scenario(
    scenario: Scenario,
    truth_model: str,
    model_trajectories: Dict[str, Any],
    out_dir: Path,
    prefix: str = "selected",
) -> List[Path]:
    saved = []
    plt.style.use("dark_background")

    truth_res = model_trajectories.get(truth_model)
    if truth_res is None:
        return saved

    t_ref  = truth_res.t / 86400.0
    r_ref  = truth_res.y[:, :3]
    v_ref  = truth_res.y[:, 3:6]
    other_models = [m for m in model_trajectories if m != truth_model]

    # 3D orbit
    fig = plt.figure(figsize=(9, 7))
    ax  = fig.add_subplot(111, projection="3d")
    r_km = r_ref / 1_000.0
    ax.plot(r_km[:, 0], r_km[:, 1], r_km[:, 2],
            color=_color(truth_model), lw=2, label=truth_model.upper(), zorder=5)
    for m in other_models:
        res = model_trajectories[m]
        rk  = interpolate_state_to_times(res.t, res.y, truth_res.t)[:, :3] / 1_000.0
        ax.plot(rk[:, 0], rk[:, 1], rk[:, 2], color=_color(m), lw=1,
                alpha=0.8, label=m.upper())
    ax.set_title(f"3D Orbit — scenario {scenario.scenario_id}\n"
                 f"hp={scenario.hp_km:.0f} km  ha={scenario.ha_km:.0f} km  "
                 f"i={scenario.inc_deg:.1f} deg")
    ax.set_xlabel("X [km]"); ax.set_ylabel("Y [km]"); ax.set_zlabel("Z [km]")
    ax.legend(fontsize=8)
    p = out_dir / f"{prefix}_orbit_3d.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig); saved.append(p)

    # Altitude
    fig, ax = plt.subplots(figsize=(10, 5))
    alt_ref = (np.linalg.norm(r_ref, axis=1) - R_MOON) / 1_000.0
    ax.plot(t_ref, alt_ref, color=_color(truth_model), lw=2, label=truth_model.upper(), zorder=5)
    for m in other_models:
        res = model_trajectories[m]
        y_m = interpolate_state_to_times(res.t, res.y, truth_res.t)
        alt = (np.linalg.norm(y_m[:, :3], axis=1) - R_MOON) / 1_000.0
        ax.plot(t_ref, alt, color=_color(m), lw=1, alpha=0.85, label=m.upper())
    ax.set_title(f"Altitude — scenario {scenario.scenario_id}")
    ax.set_xlabel("Time [days]"); ax.set_ylabel("Altitude [km]")
    ax.grid(True, alpha=0.25); ax.legend()
    p = out_dir / f"{prefix}_altitude.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig); saved.append(p)

    # Position error
    fig, ax = plt.subplots(figsize=(10, 5))
    for m in other_models:
        res = model_trajectories[m]
        y_m = interpolate_state_to_times(res.t, res.y, truth_res.t)
        dr  = np.linalg.norm(y_m[:, :3] - r_ref, axis=1) / 1_000.0
        ax.semilogy(t_ref, np.maximum(dr, 1e-9), color=_color(m), lw=1.2, label=m.upper())
    ax.set_title(f"Position Error vs {truth_model.upper()} — scenario {scenario.scenario_id}")
    ax.set_xlabel("Time [days]"); ax.set_ylabel("Position Error [km]")
    ax.grid(True, alpha=0.25, which="both"); ax.legend()
    p = out_dir / f"{prefix}_position_error.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig); saved.append(p)

    # Velocity error
    fig, ax = plt.subplots(figsize=(10, 5))
    for m in other_models:
        res = model_trajectories[m]
        y_m = interpolate_state_to_times(res.t, res.y, truth_res.t)
        dv  = np.linalg.norm(y_m[:, 3:] - v_ref, axis=1)
        ax.semilogy(t_ref, np.maximum(dv, 1e-9), color=_color(m), lw=1.2, label=m.upper())
    ax.set_title(f"Velocity Error vs {truth_model.upper()} — scenario {scenario.scenario_id}")
    ax.set_xlabel("Time [days]"); ax.set_ylabel("Velocity Error [m/s]")
    ax.grid(True, alpha=0.25, which="both"); ax.legend()
    p = out_dir / f"{prefix}_velocity_error.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig); saved.append(p)

    # RIC error
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    labels_ric = ["Radial", "In-track (Along)", "Cross-track"]
    for m in other_models:
        res = model_trajectories[m]
        y_m = interpolate_state_to_times(res.t, res.y, truth_res.t)
        ric = compute_ric_errors(r_ref, v_ref, y_m[:, :3]) / 1_000.0
        for k in range(3):
            axes[k].plot(t_ref, ric[:, k], color=_color(m), lw=1, label=m.upper())
    for k, lbl in enumerate(labels_ric):
        axes[k].set_ylabel(f"{lbl} [km]")
        axes[k].grid(True, alpha=0.25)
        axes[k].legend(fontsize=7)
    axes[0].set_title(f"RIC Position Error — scenario {scenario.scenario_id}")
    axes[2].set_xlabel("Time [days]")
    fig.tight_layout()
    p = out_dir / f"{prefix}_ric_error.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig); saved.append(p)

    return saved


def plot_aggregate_stats(
    all_metrics: List[Dict],
    agg: Dict[str, Dict],
    rankings: List[Dict],
    out_dir: Path,
) -> List[Path]:
    saved = []
    plt.style.use("dark_background")

    from collections import defaultdict
    grouped: Dict[str, List[float]] = defaultdict(list)
    for m in all_metrics:
        if m.get("status") == "ok" and m.get("rms_pos_err_km") is not None:
            grouped[m["model"]].append(m["rms_pos_err_km"])

    if not grouped:
        return saved

    models_sorted = [r["model"] for r in rankings if r["model"] in grouped]

    # Boxplot
    fig, ax = plt.subplots(figsize=(max(6, len(models_sorted) * 1.5), 6))
    data   = [grouped[m] for m in models_sorted]
    bp = ax.boxplot(data, patch_artist=True, notch=False,
                    medianprops=dict(color="white", lw=2))
    for patch, m in zip(bp["boxes"], models_sorted):
        patch.set_facecolor(_color(m))
        patch.set_alpha(0.7)
    ax.set_xticks(range(1, len(models_sorted) + 1))
    ax.set_xticklabels([m.upper() for m in models_sorted])
    ax.set_ylabel("RMS Position Error [km]")
    ax.set_title("RMS Position Error Distribution vs Truth")
    ax.grid(True, alpha=0.2, axis="y")
    p = out_dir / "aggregate_boxplot_rms_error.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig); saved.append(p)

    # P95 bar
    p95_vals = [agg[m].get("rms_pos_err_km__p95", 0) for m in models_sorted]
    fig, ax  = plt.subplots(figsize=(max(6, len(models_sorted) * 1.5), 5))
    bars = ax.bar(range(len(models_sorted)), p95_vals,
                  color=[_color(m) for m in models_sorted], alpha=0.8)
    ax.set_xticks(range(len(models_sorted)))
    ax.set_xticklabels([m.upper() for m in models_sorted])
    ax.set_ylabel("P95 RMS Position Error [km]")
    ax.set_title("P95 RMS Position Error vs Truth")
    ax.grid(True, alpha=0.2, axis="y")
    for bar, val in zip(bars, p95_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.4f}", ha="center", va="bottom", fontsize=8)
    p = out_dir / "aggregate_p95_error_bar.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig); saved.append(p)

    # Runtime vs accuracy
    fig, ax = plt.subplots(figsize=(8, 6))
    for m in models_sorted:
        rt  = agg[m].get("runtime_s__mean", np.nan)
        err = agg[m].get("rms_pos_err_km__median", np.nan)
        ax.scatter(rt, err, color=_color(m), s=120, zorder=5, label=m.upper())
        ax.annotate(m.upper(), (rt, err), textcoords="offset points",
                    xytext=(6, 4), fontsize=8)
    ax.set_xlabel("Mean Runtime per Scenario [s]")
    ax.set_ylabel("Median RMS Position Error [km]")
    ax.set_title("Runtime vs Accuracy (DOP853)")
    ax.grid(True, alpha=0.2); ax.legend()
    p = out_dir / "runtime_vs_accuracy.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig); saved.append(p)

    return saved


def plot_batch_rk4_results(
    total_rows: List[Dict],
    model_rows: List[Dict],
    integr_rows: List[Dict],
    batch_meta: Dict[str, Any],
    out_dir: Path,
) -> List[Path]:
    saved = []
    plt.style.use("dark_background")

    ok_total = [r for r in total_rows if r.get("status") == "ok"]
    if not ok_total:
        return saved

    rms_total = np.array([r["rms_pos_err_km"] for r in ok_total])

    # Runtime vs accuracy single-point panel.  It looks simple, but it makes the
    # GPU batch result visually comparable to the CPU DOP853 runtime plots.
    fig, ax = plt.subplots(figsize=(7, 5))
    runtime_s = float(batch_meta.get("runtime_s", np.nan))
    ax.scatter(runtime_s, float(np.median(rms_total)), color=_color("st_lrps"), s=140)
    ax.annotate("ST-LRPS RK4", (runtime_s, float(np.median(rms_total))),
                textcoords="offset points", xytext=(8, 6), fontsize=9)
    ax.set_xlabel("Total Batch Runtime [s]")
    ax.set_ylabel("Median RMS Position Error [km]")
    ax.set_title("Batch Runtime vs Accuracy")
    ax.grid(True, alpha=0.2)
    p = out_dir / "batch_runtime_vs_accuracy.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig); saved.append(p)

    # RMS distribution histogram
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(rms_total, bins=20, color=_color("st_lrps"), alpha=0.8, edgecolor="white", lw=0.5)
    ax.axvline(np.median(rms_total), color="yellow", lw=1.5, label=f"Median {np.median(rms_total):.3f} km")
    ax.axvline(np.percentile(rms_total, 95), color="orange", lw=1.5,
               label=f"P95 {np.percentile(rms_total, 95):.3f} km")
    ax.set_xlabel("RMS Position Error [km]")
    ax.set_ylabel("Count")
    ax.set_title("ST-LRPS Batch RK4 vs SH200 DOP853 — RMS Error Distribution")
    ax.legend(); ax.grid(True, alpha=0.2)
    p = out_dir / "batch_rms_error_distribution.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig); saved.append(p)

    # Error decomposition bar chart (if decomposition available)
    if model_rows and integr_rows:
        ok_model  = [r for r in model_rows if r.get("status") == "ok"]
        ok_integr = [r for r in integr_rows if r.get("status") == "ok"]

        rms_model  = np.median([r["rms_pos_err_km"] for r in ok_model]) if ok_model else 0
        rms_integr = np.median([r["rms_pos_err_km"] for r in ok_integr]) if ok_integr else 0
        rms_total_med = float(np.median(rms_total))

        labels = ["ST-LRPS RK4\nvs SH200 DOP853\n(total)", "ST-LRPS RK4\nvs SH200 RK4\n(model error)", "SH200 RK4\nvs SH200 DOP853\n(integrator error)"]
        vals   = [rms_total_med, rms_model, rms_integr]
        colors = [_color("st_lrps"), _color("st_lrps_batch_rk4"), _color("sh200_rk4")]

        fig, ax = plt.subplots(figsize=(9, 5))
        bars = ax.bar(range(3), vals, color=colors, alpha=0.85)
        ax.set_xticks(range(3))
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("Median RMS Position Error [km]")
        ax.set_title("Error Decomposition (Batch RK4)")
        ax.grid(True, alpha=0.2, axis="y")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.4f}", ha="center", va="bottom", fontsize=9)
        p = out_dir / "batch_error_decomposition_bar.png"
        fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig); saved.append(p)

    # RMS error vs inclination
    inc_vals = np.array([r["inc_deg"] for r in ok_total])
    fig, ax = plt.subplots(figsize=(8, 5))
    sc = ax.scatter(inc_vals, rms_total, c=rms_total, cmap="plasma", s=30, alpha=0.8)
    plt.colorbar(sc, ax=ax, label="RMS Error [km]")
    ax.set_xlabel("Inclination [deg]")
    ax.set_ylabel("RMS Position Error [km]")
    ax.set_title("ST-LRPS Batch RK4 Error vs Inclination")
    ax.grid(True, alpha=0.2)
    p = out_dir / "batch_error_vs_inclination.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig); saved.append(p)

    return saved


def plot_batch_selected_scenario(
    total_rows: List[Dict],
    batch_result: Dict[str, Any],
    truth_results: List[Optional[Any]],
    scenarios: List[Scenario],
    out_dir: Path,
) -> List[Path]:
    """Plot the median-error batch RK4 scenario against SH200 DOP853."""

    ok_rows = [
        r for r in total_rows
        if r.get("status") == "ok" and np.isfinite(r.get("rms_pos_err_km", np.nan))
    ]
    if not ok_rows:
        return []

    median_rms = float(np.median([r["rms_pos_err_km"] for r in ok_rows]))
    selected = min(ok_rows, key=lambda r: abs(float(r["rms_pos_err_km"]) - median_rms))
    sid = int(selected["scenario_id"])
    idx_by_sid = {sc.scenario_id: i for i, sc in enumerate(scenarios)}
    i = idx_by_sid.get(sid)
    if i is None or i >= len(truth_results) or truth_results[i] is None:
        return []

    truth = truth_results[i]
    assert truth is not None
    t_batch = np.asarray(batch_result["t"], dtype=np.float64)
    y_st = np.asarray(batch_result["Y"][:, i, :], dtype=np.float64)
    y_truth = interpolate_state_to_times(truth.t, truth.y, t_batch)
    t_days = t_batch / 86400.0

    saved: List[Path] = []
    plt.style.use("dark_background")

    pos_err_km = np.linalg.norm(y_st[:, :3] - y_truth[:, :3], axis=1) / 1_000.0
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.semilogy(t_days, np.maximum(pos_err_km, 1e-9), color=_color("st_lrps"), lw=1.4)
    ax.set_title(f"Batch Selected Position Error - scenario {sid}")
    ax.set_xlabel("Time [days]")
    ax.set_ylabel("Position Error [km]")
    ax.grid(True, alpha=0.25, which="both")
    p = out_dir / "batch_selected_position_error.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig); saved.append(p)

    ric_km = compute_ric_errors(y_truth[:, :3], y_truth[:, 3:], y_st[:, :3]) / 1_000.0
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    labels_ric = ["Radial", "In-track", "Cross-track"]
    for k, label in enumerate(labels_ric):
        axes[k].plot(t_days, ric_km[:, k], color=_color("st_lrps"), lw=1.1)
        axes[k].set_ylabel(f"{label} [km]")
        axes[k].grid(True, alpha=0.25)
    axes[0].set_title(f"Batch Selected RIC Error - scenario {sid}")
    axes[-1].set_xlabel("Time [days]")
    fig.tight_layout()
    p = out_dir / "batch_selected_ric_error.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig); saved.append(p)

    return saved


def estimate_stlrps_equivalent_sh_degree(aggregate_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Estimate which classical SH degree ST-LRPS resembles by error level."""

    by_model = {str(r["model"]).upper(): r for r in aggregate_rows}
    st = by_model.get("GPU_ST_LRPS_RK4")
    if not st:
        return {"status": "missing_st_lrps"}

    def _metric(metric_key: str) -> Dict[str, Any]:
        sh_points = []
        for model, row in by_model.items():
            deg = _model_degree(model)
            if model.startswith("GPU_SH") and deg is not None:
                try:
                    sh_points.append((deg, float(row[metric_key]), model))
                except Exception:
                    pass
        sh_points.sort()
        st_err = float(st[metric_key])
        if not sh_points or not np.isfinite(st_err):
            return {"status": "insufficient_data", "st_lrps_error": st_err}
        errs = np.array([p[1] for p in sh_points], dtype=np.float64)
        degrees = np.array([p[0] for p in sh_points], dtype=np.float64)
        monotonic = bool(np.all(np.diff(errs) <= 1e-12))
        closest = min(sh_points, key=lambda p: abs(p[1] - st_err))
        out = {
            "status": "ok" if monotonic else "non_monotonic_unreliable",
            "st_lrps_error": st_err,
            "closest_model": closest[2],
            "closest_degree": closest[0],
            "closest_error": closest[1],
            "monotonic": monotonic,
        }
        if not monotonic:
            return out
        if st_err > errs[0]:
            out["equivalent_degree_status"] = "worse_than_sh20"
            return out
        if st_err < errs[-1]:
            out["equivalent_degree_status"] = f"better_than_sh{int(degrees[-1])}"
            return out
        # errors decrease with degree; find enclosing interval
        for i in range(len(degrees) - 1):
            e_lo, e_hi = errs[i], errs[i + 1]
            if e_lo >= st_err >= e_hi:
                x0, x1 = degrees[i], degrees[i + 1]
                y0, y1 = math.log(max(e_lo, 1e-30)), math.log(max(e_hi, 1e-30))
                ys = math.log(max(st_err, 1e-30))
                frac = 0.0 if abs(y1 - y0) < 1e-30 else (ys - y0) / (y1 - y0)
                out["equivalent_degree_status"] = "interpolated"
                out["equivalent_degree"] = float(x0 + frac * (x1 - x0))
                return out
        return out

    return {
        "median_rms": _metric("median_rms_pos_err_km"),
        "p95_rms": _metric("p95_rms_pos_err_km"),
    }


def select_stlrps_scenarios(rows: List[Dict[str, Any]], scenarios_by_id: Dict[int, Scenario],
                            args: argparse.Namespace) -> Dict[str, Any]:
    st_rows = [
        r for r in rows
        if str(r.get("model", "")).upper().startswith("GPU_ST_LRPS_RK4")
        and r.get("status") in {"ok", "warning_negative_altitude"}
        and np.isfinite(float(r.get("rms_pos_err_km", np.nan)))
    ]
    source_label = "ST-LRPS"
    if not st_rows:
        from collections import defaultdict
        vals_by_sid: Dict[int, List[float]] = defaultdict(list)
        base_by_sid: Dict[int, Dict[str, Any]] = {}
        for r in rows:
            if r.get("status") not in {"ok", "warning_negative_altitude"}:
                continue
            try:
                sid = int(r["scenario_id"])
                val = float(r.get("rms_pos_err_km", np.nan))
            except (TypeError, ValueError):
                continue
            if not np.isfinite(val):
                continue
            vals_by_sid[sid].append(val)
            base_by_sid.setdefault(sid, dict(r))
        if not vals_by_sid:
            return {}
        st_rows = []
        for sid, vals in vals_by_sid.items():
            row = dict(base_by_sid[sid])
            row["model"] = "ALL_GPU_MODELS"
            row["rms_pos_err_km"] = float(np.mean(vals))
            st_rows.append(row)
        source_label = "comparison set"

    by_id = {int(r["scenario_id"]): r for r in st_rows}

    def _pick(label: str, override: Optional[int], key_fn: Any) -> Dict[str, Any]:
        if override is not None and override in by_id:
            row = by_id[override]
        else:
            row = key_fn(st_rows)
        sid = int(row["scenario_id"])
        sc = scenarios_by_id.get(sid)
        payload = dict(row)
        if sc is not None:
            payload.update({
                "hp_km": sc.hp_km, "ha_km": sc.ha_km, "a_km": sc.a_km,
                "e": sc.e, "inc_deg": sc.inc_deg, "raan_deg": sc.raan_deg,
                "argp_deg": sc.argp_deg, "ta_deg": sc.ta_deg,
            })
        payload["selection"] = label
        payload["selection_source"] = source_label
        return payload

    vals = np.array([float(r["rms_pos_err_km"]) for r in st_rows], dtype=np.float64)
    median = float(np.median(vals))
    mean = float(np.mean(vals))
    selected = {
        "best": _pick("best", args.plot_best_scenario_id,
                      lambda rr: min(rr, key=lambda r: float(r["rms_pos_err_km"]))),
        "worst": _pick("worst", args.plot_worst_scenario_id,
                       lambda rr: max(rr, key=lambda r: float(r["rms_pos_err_km"]))),
        "representative": _pick("representative", args.plot_representative_scenario_id,
                                lambda rr: min(rr, key=lambda r: abs(float(r["rms_pos_err_km"]) - median))),
        "mean_error": _pick("mean_error", None,
                            lambda rr: min(rr, key=lambda r: abs(float(r["rms_pos_err_km"]) - mean))),
    }
    selected["_selection_source"] = source_label
    return selected


def plot_gpu_batch_report_figures(
    aggregate_rows: List[Dict[str, Any]],
    runtime_rows: List[Dict[str, Any]],
    metrics_rows: List[Dict[str, Any]],
    results: List[BatchModelResult],
    truth: TruthTrajectorySet,
    scenarios: List[Scenario],
    selected: Dict[str, Any],
    equivalent: Dict[str, Any],
    plots_dir: Path,
    args: argparse.Namespace,
) -> List[Path]:
    """Create publication-grade report figures for the GPU batch comparison.

    Visualization only — no metric value is recomputed here. CSV units stay in
    km; figures pick a readable display unit (km/m/cm) per axis.
    """

    plots_dir.mkdir(parents=True, exist_ok=True)
    apply_plot_theme(getattr(args, "plot_theme", "report_light"))
    th = _ACTIVE_PLOT_THEME
    saved: List[Path] = []
    agg_by_model = {r["model"]: r for r in aggregate_rows}
    runtime_by_model = {r["model"]: r for r in runtime_rows}

    truth_integrator = str(getattr(args, "truth_integrator", "DOP853"))
    truth_label = f"{str(args.truth).upper()} {truth_integrator}"
    n_scn = len(scenarios)
    duration = float(getattr(args, "duration_days", 0.0) or 0.0)
    ctx = f"N = {n_scn} scenarios  ·  {duration:g} d  ·  errors vs {truth_label}"

    def _safe(x: Any, default: float = float("inf")) -> float:
        try:
            v = float(x)
        except (TypeError, ValueError):
            return default
        return v if math.isfinite(v) else default

    def _fmt(v: float) -> str:
        return f"{v:.3g}"

    def _model_vals(m: str, key: str = "rms_pos_err_km") -> List[float]:
        out = []
        for r in metrics_rows:
            if r.get("model") != m or r.get("status") not in {"ok", "warning_negative_altitude"}:
                continue
            v = _safe(r.get(key), default=float("nan"))
            if math.isfinite(v):
                out.append(v)
        return out

    # Best (lowest median RMS) first.
    models = sorted(agg_by_model.keys(),
                    key=lambda m: _safe(agg_by_model[m].get("median_rms_pos_err_km")))
    counts = [len(_model_vals(m)) for m in models]
    n_dist = max(counts) if counts else 0
    small_n = 0 < n_dist < 8

    # ----- 1. Accuracy ranking (horizontal lollipop) ---------------------
    if models:
        med_km = [_safe(agg_by_model[m].get("median_rms_pos_err_km"), 0.0) for m in models]
        p95_km = [_safe(agg_by_model[m].get("p95_rms_pos_err_km"), 0.0) for m in models]
        unit, mult = select_length_unit(max(_finite_positive(med_km + p95_km) or [0.0]))
        med = [v * mult for v in med_km]
        p95 = [v * mult for v in p95_km]
        labels = [display_label(m) for m in models]
        logx = _should_log(med_km + p95_km)
        y = np.arange(len(models))

        fig, ax = plt.subplots(figsize=(9.5, max(3.2, 0.62 * len(models) + 1.6)))
        x0 = min(_finite_positive(med + p95) or [0.0]) * 0.5 if logx else 0.0
        for yi, m, mv, pv in zip(y, models, med, p95):
            c = model_color(m)
            ax.hlines(yi, x0, mv, color=c, lw=2.6, alpha=0.45, zorder=2)
            ax.scatter(mv, yi, color=c, marker=model_marker(m),
                       s=150 if _is_stlrps(m) else 80,
                       edgecolor=th["edge"], linewidth=0.6, zorder=model_zorder(m))
            ax.scatter(pv, yi, facecolors="none", edgecolors=c, marker="D",
                       s=46, linewidth=1.3, zorder=4)
            anchor = max(mv, pv)
            xt = anchor * 1.10 if logx else anchor + 0.02 * max(med + p95 + [1e-9])
            ax.text(xt, yi, _fmt(mv), va="center", ha="left", fontsize=9, color=th["text"])
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()  # best at top
        if logx:
            ax.set_xscale("log")
        ax.margins(x=0.16)  # headroom for end-of-bar value labels
        from matplotlib.lines import Line2D
        proxies = [
            Line2D([0], [0], marker="o", color=th["muted"], ls="none", label="Median RMS"),
            Line2D([0], [0], marker="D", markerfacecolor="none", markeredgecolor=th["muted"],
                   color=th["muted"], ls="none", label="P95 RMS"),
        ]
        ax.legend(handles=proxies, loc="upper right", frameon=False)
        _style_ax(ax, title="GPU RK4 Accuracy Ranking",
                  xlabel=f"RMS Position Error [{unit}]",
                  subtitle=f"Lower is better.  {ctx}")
        ax.grid(True, axis="y", alpha=0.0)
        _highlight_ticklabels(ax, labels, axis="y")
        fig.tight_layout()
        p = plots_dir / "gpu_accuracy_ranking_bar.png"
        fig.savefig(p); plt.close(fig); saved.append(p)

    # ----- 2. Runtime vs accuracy ----------------------------------------
    if models:
        pts = []
        for m in models:
            x = _safe(runtime_by_model.get(m, {}).get("total_runtime_s"), default=float("nan"))
            yk = _safe(agg_by_model[m].get("median_rms_pos_err_km"), default=float("nan"))
            if math.isfinite(x) and math.isfinite(yk):
                pts.append((x, yk, m))
        unit, mult = select_length_unit(max(_finite_positive([p[1] for p in pts]) or [0.0]))
        logy = _should_log([p[1] for p in pts])
        fig, ax = plt.subplots(figsize=(8.6, 5.8))
        if pts:
            # Pareto frontier (lower-left): cheapest run achieving each new best error.
            front = []
            best = float("inf")
            for x, yk, m in sorted(pts, key=lambda t: t[0]):
                if yk < best - 1e-30:
                    best = yk
                    front.append((x, yk * mult))
            if len(front) >= 2:
                ax.step([f[0] for f in front], [f[1] for f in front], where="post",
                        ls="--", lw=1.2, color=th["muted"], alpha=0.7, zorder=1,
                        label="Pareto front")
            for x, yk, m in pts:
                ax.scatter(x, yk * mult, color=model_color(m), marker=model_marker(m),
                           s=model_marker_size(m), edgecolor=th["edge"],
                           linewidth=1.0 if _is_stlrps(m) else 0.6, zorder=model_zorder(m))
                ax.annotate(display_label(m), (x, yk * mult), xytext=(8, 5),
                            textcoords="offset points", fontsize=9.5 if _is_stlrps(m) else 9,
                            fontweight="bold" if _is_stlrps(m) else "normal",
                            color=model_color(m) if _is_stlrps(m) else th["text"])
            if logy:
                ax.set_yscale("log")
            _legend(ax, loc="upper right")
        else:
            _empty_note(ax, "No runtime/accuracy data available.")
        _style_ax(ax, title="Runtime vs Accuracy",
                  xlabel="Total GPU Runtime [s]",
                  ylabel=f"Median RMS Position Error [{unit}]",
                  subtitle=f"Lower-left is better (faster + more accurate).  {ctx}")
        fig.tight_layout()
        p = plots_dir / "gpu_runtime_vs_accuracy.png"
        fig.savefig(p); plt.close(fig); saved.append(p)

    # ----- 3. RMS error distribution -------------------------------------
    data = [_model_vals(m) for m in models]
    all_vals = [v for d in data for v in d]
    unit, mult = select_length_unit(max(_finite_positive(all_vals) or [0.0]))
    fig, ax = plt.subplots(figsize=(9.5, max(3.2, 0.62 * len(models) + 1.8)))
    if any(data):
        y = np.arange(len(models))
        logx = _should_log(all_vals)
        if small_n:
            rng = np.random.default_rng(0)
            for yi, m, vals in zip(y, models, data):
                if not vals:
                    continue
                vv = np.asarray(vals) * mult
                jitter = (rng.random(len(vv)) - 0.5) * 0.28
                ax.scatter(vv, np.full_like(vv, yi) + jitter, color=model_color(m),
                           marker=model_marker(m), s=70 if _is_stlrps(m) else 42,
                           alpha=0.85, edgecolor=th["edge"], linewidth=0.4,
                           zorder=model_zorder(m))
                ax.scatter(np.median(vv), yi, color=th["text"], marker="|", s=420,
                           linewidth=2.2, zorder=6)
            subtitle = f"N={n_dist} is small — strip plot is diagnostic, not statistical.  {ctx}"
        else:
            box = ax.boxplot(
                [np.asarray(d) * mult for d in data], vert=False, patch_artist=True,
                showfliers=False, widths=0.6, positions=y,
                medianprops=dict(color=th["text"], lw=1.8),
            )
            for patch, m in zip(box["boxes"], models):
                patch.set_facecolor(model_color(m))
                patch.set_alpha(0.45 if not _is_stlrps(m) else 0.65)
                patch.set_edgecolor(model_color(m))
            subtitle = ctx
        if logx:
            ax.set_xscale("log")
        ax.set_yticks(y)
        ax.set_yticklabels([display_label(m) for m in models])
        ax.invert_yaxis()
        _style_ax(ax, title="RMS Position Error Distribution",
                  xlabel=f"RMS Position Error [{unit}]", subtitle=subtitle)
        ax.grid(True, axis="y", alpha=0.0)
        _highlight_ticklabels(ax, [display_label(m) for m in models], axis="y")
    else:
        _empty_note(ax, "Errors are below plotting threshold for this short run.")
        _style_ax(ax, title="RMS Position Error Distribution", subtitle=ctx)
    fig.tight_layout()
    p = plots_dir / "gpu_rms_error_distribution_boxplot.png"
    fig.savefig(p); plt.close(fig); saved.append(p)

    # ----- 4. Histograms --------------------------------------------------
    if models:
        fig, axes = plt.subplots(len(models), 1,
                                 figsize=(9, max(4.2, 1.4 * len(models))), sharex=True)
        if len(models) == 1:
            axes = [axes]
        for ax, m, vals in zip(axes, models, data):
            if vals:
                ax.hist(np.asarray(vals) * mult, bins=min(24, max(6, n_dist)),
                        color=model_color(m), alpha=0.85, edgecolor=th["bg"], linewidth=0.4)
            else:
                _empty_note(ax, "no data")
            ax.set_ylabel(display_label(m), rotation=0, ha="right", va="center",
                          fontsize=9, color=(_ST_LRPS_COLOR if _is_stlrps(m) else th["text"]),
                          fontweight="bold" if _is_stlrps(m) else "normal")
            ax.grid(True, alpha=0.4)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
        axes[-1].set_xlabel(f"RMS Position Error [{unit}]")
        axes[0].set_title("RMS Error Histograms per Model")
        fig.tight_layout()
        p = plots_dir / "gpu_rms_error_histograms.png"
        fig.savefig(p); plt.close(fig); saved.append(p)

    # ----- 5. ST-LRPS equivalent SH degree -------------------------------
    sh_points = []
    for m in models:
        deg = _model_degree(m)
        if deg is not None and m.upper().startswith("GPU_SH"):
            sh_points.append((deg, _safe(agg_by_model[m].get("median_rms_pos_err_km")),
                              _safe(agg_by_model[m].get("p95_rms_pos_err_km"))))
    sh_points.sort()
    st = agg_by_model.get("GPU_ST_LRPS_RK4")
    st_med_km = _safe(st.get("median_rms_pos_err_km")) if st else float("nan")
    st_p95_km = _safe(st.get("p95_rms_pos_err_km")) if st else float("nan")
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    span_km = [p[1] for p in sh_points] + [p[2] for p in sh_points]
    if math.isfinite(st_med_km):
        span_km.append(st_med_km)
    unit, mult = select_length_unit(max(_finite_positive(span_km) or [0.0]))
    if sh_points:
        degs = [p[0] for p in sh_points]
        ax.plot(degs, [p[1] * mult for p in sh_points], marker="o", color="#3D5A80",
                lw=2.0, label="SH median RMS", zorder=3)
        ax.plot(degs, [p[2] * mult for p in sh_points], marker="s", ls="--", color="#6C8EBF",
                lw=1.6, label="SH P95 RMS", zorder=3)
        if math.isfinite(st_med_km):
            ax.axhline(st_med_km * mult, color=_ST_LRPS_COLOR, lw=2.6, ls="-",
                       label="ST-LRPS median RMS", zorder=4)
        if math.isfinite(st_p95_km):
            ax.axhline(st_p95_km * mult, color=_ST_LRPS_COLOR, lw=1.6, ls=":",
                       alpha=0.8, label="ST-LRPS P95 RMS", zorder=4)
        # Annotate the equivalent-degree estimate, if available.
        med_eq = equivalent.get("median_rms", {}) if isinstance(equivalent, dict) else {}
        eq_txt = None
        if med_eq.get("equivalent_degree") is not None:
            eq_deg = float(med_eq["equivalent_degree"])
            eq_txt = f"≈ SH{eq_deg:.0f}"
            ax.axvline(eq_deg, color=_ST_LRPS_COLOR, lw=1.0, ls=":", alpha=0.6)
        else:
            status_map = {
                "worse_than_sh20": "below SH20",
                "better_than_sh200": "above SH200",
            }
            raw = str(med_eq.get("equivalent_degree_status", ""))
            eq_txt = status_map.get(raw)
            if eq_txt is None and raw.startswith("better_than_sh"):
                eq_txt = f"above {raw.replace('better_than_', '').upper()}"
        if eq_txt and math.isfinite(st_med_km):
            ax.annotate(f"ST-LRPS {eq_txt}", xy=(degs[len(degs) // 2], st_med_km * mult),
                        xytext=(0, 8), textcoords="offset points", color=_ST_LRPS_COLOR,
                        fontweight="bold", fontsize=10, ha="center")
        if _should_log(span_km):
            ax.set_yscale("log")
        _legend(ax, loc="best")
    else:
        _empty_note(ax, "No spherical-harmonic baselines available for comparison.")
    _style_ax(ax, title="ST-LRPS Equivalent Spherical-Harmonic Degree",
              xlabel="Spherical Harmonic Degree",
              ylabel=f"RMS Position Error [{unit}]",
              subtitle=f"Where ST-LRPS sits on the SH error ladder.  {ctx}")
    fig.tight_layout()
    p = plots_dir / "stlrps_equivalent_sh_degree.png"
    fig.savefig(p); plt.close(fig); saved.append(p)

    # ----- 6-7. Error vs inclination / altitude --------------------------
    for xkey, xlabel, fname in [
        ("inc_deg", "Inclination [deg]", "gpu_error_vs_inclination_all_models.png"),
        ("hp_km", "Periselene Altitude [km]", "gpu_error_vs_altitude_all_models.png"),
    ]:
        all_y = [v for m in models for v in _model_vals(m)]
        unit, mult = select_length_unit(max(_finite_positive(all_y) or [0.0]))
        fig, ax = plt.subplots(figsize=(9, 5.4))
        plotted = False
        for m in models:
            rows = [r for r in metrics_rows
                    if r.get("model") == m and r.get("status") in {"ok", "warning_negative_altitude"}]
            xs = [_safe(r.get(xkey), float("nan")) for r in rows]
            ys = [_safe(r.get("rms_pos_err_km"), float("nan")) * mult for r in rows]
            if rows:
                ax.scatter(xs, ys, color=model_color(m), marker=model_marker(m),
                           s=70 if _is_stlrps(m) else 26, alpha=0.85,
                           edgecolor=th["edge"], linewidth=0.3,
                           zorder=model_zorder(m), label=display_label(m))
                plotted = True
        if plotted:
            if _should_log(all_y):
                ax.set_yscale("log")
            _legend(ax, outside=True)
        else:
            _empty_note(ax, "Errors are below plotting threshold for this short run.")
        _style_ax(ax, title=f"Error vs {xlabel.split()[0]}", xlabel=xlabel,
                  ylabel=f"RMS Position Error [{unit}]", subtitle=ctx)
        fig.tight_layout()
        p = plots_dir / fname
        fig.savefig(p); plt.close(fig); saved.append(p)

    # ----- Ensemble time-series ------------------------------------------
    if scenarios and truth.t_by_scenario:
        common_t = next(iter(truth.t_by_scenario.values()))
        t_days = np.asarray(common_t) / 86400.0
        med_by_model: Dict[str, np.ndarray] = {}
        band_by_model: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        ric_by_model: Dict[str, np.ndarray] = {}
        for result in results:
            if result.status != "ok":
                continue
            pos_err, ric_err = [], []
            for i, sc in enumerate(scenarios):
                if sc.scenario_id not in truth.t_by_scenario:
                    continue
                y_model = interpolate_state_to_times(result.t, result.y[:, i, :], common_t)
                y_truth = truth.y_by_scenario[sc.scenario_id]
                pos_err.append(np.linalg.norm(y_model[:, :3] - y_truth[:, :3], axis=1) / 1000.0)
                ric_err.append(compute_ric_errors(y_truth[:, :3], y_truth[:, 3:], y_model[:, :3]) / 1000.0)
            if not pos_err:
                continue
            pos_arr = np.asarray(pos_err)
            med_by_model[result.display_name] = np.median(pos_arr, axis=0)
            band_by_model[result.display_name] = (np.percentile(pos_arr, 25, axis=0),
                                                  np.percentile(pos_arr, 75, axis=0))
            ric_by_model[result.display_name] = np.asarray(ric_err)

        pos_max_km = max(_finite_positive([float(np.max(v)) for v in med_by_model.values()]) or [0.0])
        unit, mult = select_length_unit(pos_max_km)

        fig, ax = plt.subplots(figsize=(10, 5.6))
        if med_by_model and pos_max_km > 1e-12:
            for name, curve in med_by_model.items():
                lo, hi = band_by_model[name]
                ax.plot(t_days, curve * mult, color=model_color(name),
                        lw=model_linewidth(name), label=display_label(name),
                        zorder=model_zorder(name))
                ax.fill_between(t_days, lo * mult, hi * mult, color=model_color(name), alpha=0.10)
            _legend(ax, outside=True)
        else:
            _empty_note(ax, "Errors are below plotting threshold for this short run.")
        _style_ax(ax, title="Ensemble Position Error vs Time",
                  xlabel="Time [days]", ylabel=f"Median Position Error [{unit}]",
                  subtitle=f"Median across scenarios; shaded band = 25–75%.  {ctx}")
        fig.tight_layout()
        p = plots_dir / "ensemble_mean_position_error_vs_time.png"
        fig.savefig(p); plt.close(fig); saved.append(p)

        ric_curves = {name: np.sqrt(np.mean(arr ** 2, axis=0)) for name, arr in ric_by_model.items()}
        ric_max_km = max(_finite_positive(
            [float(np.max(c)) for c in ric_curves.values()]) or [0.0])
        runit, rmult = select_length_unit(ric_max_km)
        fig_ric, axes_ric = plt.subplots(3, 1, figsize=(10, 8.2), sharex=True)
        for k, lbl in enumerate(["Radial", "Along-track", "Cross-track"]):
            if ric_curves and ric_max_km > 1e-12:
                for name, c in ric_curves.items():
                    axes_ric[k].plot(t_days, c[:, k] * rmult, color=model_color(name),
                                     lw=model_linewidth(name), label=display_label(name),
                                     zorder=model_zorder(name))
            else:
                _empty_note(axes_ric[k], "below plotting threshold")
            axes_ric[k].set_ylabel(f"{lbl} RMS [{runit}]")
            axes_ric[k].grid(True, alpha=0.45)
            for spine in ("top", "right"):
                axes_ric[k].spines[spine].set_visible(False)
        axes_ric[-1].set_xlabel("Time [days]")
        axes_ric[0].set_title("Ensemble RIC RMS Error vs Time")
        if ric_curves and ric_max_km > 1e-12:
            _legend(axes_ric[0], outside=True)
        fig_ric.tight_layout()
        p = plots_dir / "ensemble_ric_rms_vs_time.png"
        fig_ric.savefig(p); plt.close(fig_ric); saved.append(p)

    # ----- Selected ST-LRPS scenarios ------------------------------------
    scenario_by_id = {s.scenario_id: s for s in scenarios}
    selection_source = str(selected.get("_selection_source", "ST-LRPS"))
    for label in ("best", "representative", "worst"):
        item = selected.get(label)
        if not item:
            continue
        sid = int(item["scenario_id"])
        sc = scenario_by_id.get(sid)
        if sc is None or sid not in truth.t_by_scenario:
            continue
        idx = scenarios.index(sc)
        t_truth = truth.t_by_scenario[sid]
        y_truth = truth.y_by_scenario[sid]
        t_days = np.asarray(t_truth) / 86400.0

        pos_by_model: Dict[str, np.ndarray] = {}
        alt_by_model: Dict[str, np.ndarray] = {}
        ric_by_model = {}
        for result in results:
            if result.status != "ok":
                continue
            y_model = interpolate_state_to_times(result.t, result.y[:, idx, :], t_truth)
            pos_by_model[result.display_name] = (
                np.linalg.norm(y_model[:, :3] - y_truth[:, :3], axis=1) / 1000.0)
            alt_by_model[result.display_name] = (
                np.linalg.norm(y_model[:, :3], axis=1) - np.linalg.norm(y_truth[:, :3], axis=1)) / 1000.0
            ric_by_model[result.display_name] = (
                compute_ric_errors(y_truth[:, :3], y_truth[:, 3:], y_model[:, :3]) / 1000.0)

        pos_max_km = max(_finite_positive([float(np.max(v)) for v in pos_by_model.values()]) or [0.0])
        unit, mult = select_length_unit(pos_max_km)
        sub = f"Scenario {sid}: hp={sc.hp_km:.0f} km, i={sc.inc_deg:.1f}°.  vs {truth_label}"

        # Position error
        fig_pos, ax_pos = plt.subplots(figsize=(10, 5.4))
        if pos_by_model and pos_max_km > 1e-12:
            use_log = bool(getattr(args, "plot_error_logscale", False)) or _should_log(
                [float(np.max(v)) for v in pos_by_model.values()])
            for name, curve in pos_by_model.items():
                ax_pos.plot(t_days, np.maximum(curve * mult, 1e-12 if use_log else 0.0),
                            color=model_color(name), lw=model_linewidth(name),
                            label=display_label(name), zorder=model_zorder(name))
            if use_log:
                ax_pos.set_yscale("log")
            _legend(ax_pos, outside=True)
        else:
            _empty_note(ax_pos, "Errors are below plotting threshold for this short run.")
        _style_ax(ax_pos, title=f"{label.title()} {selection_source} Scenario: Position Error",
                  xlabel="Time [days]", ylabel=f"Position Error [{unit}]", subtitle=sub)
        fig_pos.tight_layout()
        p = plots_dir / f"selected_{label}_position_error_all_models.png"
        fig_pos.savefig(p); plt.close(fig_pos); saved.append(p)

        # Altitude error
        alt_max_km = max(_finite_positive(
            [float(np.max(np.abs(v))) for v in alt_by_model.values()]) or [0.0])
        aunit, amult = select_length_unit(alt_max_km)
        fig_alt, ax_alt = plt.subplots(figsize=(10, 5.0))
        if alt_by_model and alt_max_km > 1e-12:
            for name, curve in alt_by_model.items():
                ax_alt.plot(t_days, curve * amult, color=model_color(name),
                            lw=model_linewidth(name), label=display_label(name),
                            zorder=model_zorder(name))
            _legend(ax_alt, outside=True)
        else:
            _empty_note(ax_alt, "Errors are below plotting threshold for this short run.")
        _style_ax(ax_alt, title=f"{label.title()} {selection_source} Scenario: Altitude Error",
                  xlabel="Time [days]", ylabel=f"Altitude Error [{aunit}]", subtitle=sub)
        fig_alt.tight_layout()
        p = plots_dir / f"selected_{label}_altitude_error_all_models.png"
        fig_alt.savefig(p); plt.close(fig_alt); saved.append(p)

        # RIC error
        ric_max_km = max(_finite_positive(
            [float(np.max(np.abs(v))) for v in ric_by_model.values()]) or [0.0])
        runit, rmult = select_length_unit(ric_max_km)
        fig_ric_sel, axes_sel = plt.subplots(3, 1, figsize=(10, 8.2), sharex=True)
        for k, lbl in enumerate(["Radial", "Along-track", "Cross-track"]):
            if ric_by_model and ric_max_km > 1e-12:
                for name, curve in ric_by_model.items():
                    axes_sel[k].plot(t_days, curve[:, k] * rmult, color=model_color(name),
                                     lw=model_linewidth(name), label=display_label(name),
                                     zorder=model_zorder(name))
            else:
                _empty_note(axes_sel[k], "below plotting threshold")
            axes_sel[k].set_ylabel(f"{lbl} [{runit}]")
            axes_sel[k].grid(True, alpha=0.45)
            for spine in ("top", "right"):
                axes_sel[k].spines[spine].set_visible(False)
        axes_sel[0].set_title(f"{label.title()} {selection_source} Scenario {sid}: RIC Error")
        axes_sel[-1].set_xlabel("Time [days]")
        if ric_by_model and ric_max_km > 1e-12:
            _legend(axes_sel[0], outside=True)
        fig_ric_sel.tight_layout()
        p = plots_dir / f"selected_{label}_ric_error_all_models.png"
        fig_ric_sel.savefig(p); plt.close(fig_ric_sel); saved.append(p)

        # 3D trajectory (optional)
        if getattr(args, "plot_3d", False):
            fig_3d = plt.figure(figsize=(8, 7))
            ax_3d = fig_3d.add_subplot(111, projection="3d")
            rk = y_truth[:, :3] / 1000.0
            ax_3d.plot(rk[:, 0], rk[:, 1], rk[:, 2], color=_TRUTH_COLOR, lw=2.5, label=truth_label)
            for result in results:
                if result.status != "ok":
                    continue
                y_model = interpolate_state_to_times(result.t, result.y[:, idx, :], t_truth)
                rk = y_model[:, :3] / 1000.0
                ax_3d.plot(rk[:, 0], rk[:, 1], rk[:, 2], color=model_color(result.display_name),
                           lw=model_linewidth(result.display_name), label=display_label(result.display_name))
            ax_3d.set_title(f"{label.title()} ST-LRPS Scenario {sid}: 3D Trajectory")
            ax_3d.set_xlabel("X [km]"); ax_3d.set_ylabel("Y [km]"); ax_3d.set_zlabel("Z [km]")
            ax_3d.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
            p = plots_dir / f"selected_{label}_trajectory_3d_all_models.png"
            fig_3d.savefig(p, bbox_inches="tight"); plt.close(fig_3d); saved.append(p)

    return saved
# =============================================================================
# PDF report — professional template
# =============================================================================
# A small, dependency-free (matplotlib-only) report toolkit that gives the
# generated PDFs a consistent, publication-grade look: a title cover page, a
# navy header band + accent rule, a footer with page numbers and timestamp,
# cleanly styled tables, and captioned figure pages. The numeric content is
# unchanged — only the presentation is upgraded.

_REPORT_THEME = {
    "navy": "#16314F",
    "navy_soft": "#1F4068",
    "accent": "#2A9D8F",
    "ink": "#1A1F29",
    "muted": "#5A6675",
    "rule": "#C9D2DE",
    "row_alt": "#EEF2F7",
    "highlight": "#E3F2EE",
    "page": "#FFFFFF",
}
_REPORT_PAGE_SIZE = (8.5, 11.0)  # US Letter, portrait


class _ReportPager:
    """Builds a multi-page PDF with a consistent header/footer and styled pages."""

    def __init__(self, pdf: PdfPages, title: str, subtitle: str) -> None:
        self.pdf = pdf
        self.title = title
        self.subtitle = subtitle
        self.page_no = 0
        self.generated = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())

    # -- low-level page scaffolding ------------------------------------
    def _blank(self):
        fig = plt.figure(figsize=_REPORT_PAGE_SIZE)
        fig.patch.set_facecolor(_REPORT_THEME["page"])
        return fig

    def _chrome(self, fig, heading: Optional[str]) -> Any:
        """Draw header band + footer; return a content axes (0..1)."""
        from matplotlib.patches import Rectangle
        from matplotlib.lines import Line2D
        self.page_no += 1
        t = _REPORT_THEME
        # Header band
        fig.add_artist(Rectangle((0, 0.945), 1, 0.055, transform=fig.transFigure,
                                 facecolor=t["navy"], edgecolor="none", zorder=0))
        fig.add_artist(Rectangle((0, 0.941), 1, 0.004, transform=fig.transFigure,
                                 facecolor=t["accent"], edgecolor="none", zorder=0))
        fig.text(0.06, 0.973, self.title, color="white", fontsize=13,
                 fontweight="bold", va="center")
        fig.text(0.06, 0.954, self.subtitle, color="#A9C0D6", fontsize=8.5, va="center")
        # Footer
        fig.add_artist(Line2D([0.06, 0.94], [0.052, 0.052], color=t["rule"],
                              lw=0.8, transform=fig.transFigure))
        fig.text(0.06, 0.034, "ST-LRPS · Lunar Gravity Model Validation",
                 color=t["muted"], fontsize=8, va="center")
        fig.text(0.50, 0.034, self.generated, color=t["muted"], fontsize=8,
                 ha="center", va="center")
        fig.text(0.94, 0.034, f"Page {self.page_no}", color=t["muted"], fontsize=8,
                 ha="right", va="center")
        ax = fig.add_axes([0.06, 0.075, 0.88, 0.85])
        ax.axis("off")
        if heading:
            ax.text(0.0, 1.0, heading, transform=ax.transAxes, fontsize=15,
                    fontweight="bold", color=t["navy"], va="top")
        return ax

    def _save(self, fig) -> None:
        self.pdf.savefig(fig, facecolor=_REPORT_THEME["page"])
        plt.close(fig)

    # -- public page builders ------------------------------------------
    def cover(self, meta: List[Tuple[str, str]], note: str) -> None:
        from matplotlib.patches import Rectangle
        from matplotlib.lines import Line2D
        t = _REPORT_THEME
        self.page_no += 1
        fig = self._blank()
        # Full-bleed navy banner
        fig.add_artist(Rectangle((0, 0.62), 1, 0.38, transform=fig.transFigure,
                                 facecolor=t["navy"], edgecolor="none", zorder=0))
        fig.add_artist(Rectangle((0, 0.612), 1, 0.008, transform=fig.transFigure,
                                 facecolor=t["accent"], edgecolor="none", zorder=0))
        fig.text(0.08, 0.86, self.title, color="white", fontsize=26, fontweight="bold", va="center")
        fig.text(0.08, 0.80, self.subtitle, color="#A9C0D6", fontsize=13, va="center")
        fig.text(0.08, 0.665, self.generated, color="#7FA8C9", fontsize=10, va="center")
        # Metadata table area
        ax = fig.add_axes([0.08, 0.16, 0.84, 0.40])
        ax.axis("off")
        rows = [[k, v] for k, v in meta]
        if rows:
            tbl = ax.table(cellText=rows, colLabels=["Parameter", "Value"],
                           cellLoc="left", loc="upper left", bbox=[0, 0, 1, 1])
            _style_table(tbl, n_body=len(rows), first_col_left=True)
        # Disclaimer note
        fig.add_artist(Line2D([0.08, 0.92], [0.12, 0.12], color=t["rule"], lw=0.8,
                              transform=fig.transFigure))
        fig.text(0.08, 0.085, note, color=t["muted"], fontsize=9, va="center", wrap=True)
        fig.text(0.08, 0.045, "Generated by st_lrps.evaluation.compare_gravity_models",
                 color=t["muted"], fontsize=8, va="center")
        self._save(fig)

    def table_page(self, heading: str, col_labels: List[str], rows: List[List[str]],
                   *, highlight_row: Optional[int] = None, intro: Optional[str] = None) -> None:
        fig = self._blank()
        ax = self._chrome(fig, heading)  # content axes, axis off, coords 0..1
        top = 0.90
        if intro:
            ax.text(0.0, top, intro, transform=ax.transAxes, fontsize=9.5,
                    color=_REPORT_THEME["muted"], va="top", wrap=True)
            top -= 0.06
        # Bound the table to a sensible region under the heading (axes-relative).
        n = max(1, len(rows))
        tbl_h = min(top - 0.02, 0.05 * (n + 1))
        sub = ax.inset_axes([0.0, max(0.0, top - tbl_h), 1.0, tbl_h])
        sub.axis("off")
        tbl = sub.table(cellText=rows, colLabels=col_labels, cellLoc="center",
                        loc="upper center", bbox=[0, 0, 1, 1])
        _style_table(tbl, n_body=n, first_col_left=True, highlight_row=highlight_row)
        self._save(fig)

    def figure_page(self, heading: str, image_path: Path, caption: str = "") -> bool:
        if not Path(image_path).exists():
            return False
        fig = self._blank()
        self._chrome(fig, heading)
        img = plt.imread(str(image_path))
        h = int(img.shape[0]) or 1
        w = int(img.shape[1]) or 1
        img_aspect = w / h
        page_w, page_h = _REPORT_PAGE_SIZE
        # Size the image box so the figure fills the page width (preserving its
        # aspect) and sits just under the heading — no tiny plot floating in a
        # large blank page.
        bw, x0, y_top = 0.88, 0.06, 0.90
        bh = bw * page_w / (img_aspect * page_h)
        max_bh = 0.74
        if bh > max_bh:
            bh = max_bh
            bw = min(0.88, bh * img_aspect * page_h / page_w)
            x0 = (1.0 - bw) / 2.0
        y0 = y_top - bh
        img_ax = fig.add_axes([x0, y0, bw, bh])
        img_ax.imshow(img)
        img_ax.axis("off")
        if caption:
            fig.text(0.06, max(0.07, y0 - 0.022), caption, color=_REPORT_THEME["muted"],
                     fontsize=9, va="top", wrap=True)
        self._save(fig)
        return True

    def text_page(self, heading: str, paragraphs: List[str]) -> None:
        fig = self._blank()
        ax = self._chrome(fig, heading)
        y = 0.92
        for para in paragraphs:
            ax.text(0.0, y, para, transform=ax.transAxes, fontsize=10,
                    color=_REPORT_THEME["ink"], va="top", wrap=True)
            y -= 0.035 + 0.02 * para.count("\n")
        self._save(fig)


def _style_table(tbl, *, n_body: int, first_col_left: bool = False,
                 highlight_row: Optional[int] = None) -> None:
    """Apply the professional table style to a matplotlib table in place."""
    t = _REPORT_THEME
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(t["rule"])
        cell.set_linewidth(0.6)
        if r == 0:
            cell.set_facecolor(t["navy"])
            cell.set_text_props(color="white", fontweight="bold")
            cell.set_height(cell.get_height() * 1.5)
        else:
            if highlight_row is not None and (r - 1) == highlight_row:
                cell.set_facecolor(t["highlight"])
                cell.set_text_props(color=t["ink"], fontweight="bold")
            else:
                cell.set_facecolor("#FFFFFF" if (r % 2 == 1) else t["row_alt"])
                cell.set_text_props(color=t["ink"])
            cell.set_height(cell.get_height() * 1.25)
        if first_col_left and c == 0:
            cell.get_text().set_horizontalalignment("left")
            cell.PAD = 0.04


def write_report_pdf(
    args: argparse.Namespace,
    scenarios: List[Scenario],
    agg: Dict[str, Dict],
    rankings: List[Dict],
    worst_cases: List[Dict],
    plots_dir: Path,
    out_dir: Path,
) -> None:
    pdf_path = out_dir / "gravity_random_validation_report.pdf"
    _ensure_dir(pdf_path)
    truth_integ = str(getattr(args, "truth_integrator", args.integrator))
    with PdfPages(pdf_path) as pdf:
        pager = _ReportPager(
            pdf,
            title="Lunar Gravity Model Validation",
            subtitle="Orbit-level comparison vs a high-degree reference",
        )
        pager.cover(
            meta=[
                ("Scenarios", str(args.random_scenarios)),
                ("Sampling method", str(getattr(args, "sampling_method", "random"))),
                ("Scenario seed", str(args.scenario_seed)),
                ("Scenario mode", str(args.scenario_mode)),
                ("Inclination sampling", str(getattr(args, "inclination_sampling", "uniform_deg"))),
                ("Altitude range", f"{args.altitude_min_km:g} - {args.altitude_max_km:g} km"),
                ("Duration", f"{args.duration_days:g} days"),
                ("Truth model", f"{args.truth.upper()} ({truth_integ})"),
                ("Compared models", str(args.models)),
                ("Integrator", str(args.integrator)),
                ("CPU workers", str(getattr(args, "workers", 1))),
            ],
            note=("Reference note: the truth model is a high-accuracy numerical "
                  "reference, not physical ground truth. Reported errors are "
                  "relative to that reference."),
        )

        if rankings:
            best_idx = 0
            for i, r in enumerate(rankings):
                if r.get("rank_median_rms") in (1, "1"):
                    best_idx = i
                    break
            rows = [[
                r.get("model", "").upper(),
                str(r.get("rank_median_rms", "")),
                f"{r.get('median_rms_pos_err_km', 0):.4f}",
                f"{r.get('p95_rms_pos_err_km', 0):.4f}",
                f"{r.get('max_pos_err_km__mean', 0):.4f}",
                f"{r.get('runtime_s__mean', 0):.2f}",
            ] for r in rankings]
            pager.table_page(
                "Model Accuracy Ranking",
                ["Model", "Rank", "Median RMS [km]", "P95 RMS [km]",
                 "Mean Max [km]", "Mean Runtime [s]"],
                rows,
                highlight_row=best_idx,
                intro="Ranked by median RMS position error across all scenarios "
                      "(lower is better).",
            )

        figure_specs = [
            ("aggregate_boxplot_rms_error.png", "RMS position-error distribution across scenarios."),
            ("aggregate_p95_error_bar.png", "Per-model 95th-percentile RMS position error."),
            ("runtime_vs_accuracy.png", "Runtime vs accuracy tradeoff."),
            ("selected_position_error.png", "Selected scenario: position error vs time."),
            ("selected_ric_error.png", "Selected scenario: radial/in-track/cross-track error."),
            ("selected_orbit_3d.png", "Selected scenario: 3-D trajectory overlay."),
        ]
        for png_name, caption in figure_specs:
            for search_dir in (plots_dir, out_dir):
                if pager.figure_page("Figure", search_dir / png_name, caption):
                    break

    print(f"  [report] PDF saved: {pdf_path}", flush=True)


def write_gpu_batch_report_pdf(
    args: argparse.Namespace,
    aggregate_rows: List[Dict[str, Any]],
    runtime_rows: List[Dict[str, Any]],
    equivalent: Dict[str, Any],
    selected: Dict[str, Any],
    plots_dir: Path,
    reports_dir: Path,
) -> None:
    """Write the GPU batch validation report PDF (professional template)."""

    reports_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = reports_dir / "gpu_batch_validation_report.pdf"
    rk4_dt_values = _gpu_rk4_dt_values(args)
    rk4_dt_text = ", ".join(f"{v:g}" for v in rk4_dt_values)
    truth_integ = str(getattr(args, "truth_integrator", "DOP853"))
    gpu_integ = str(getattr(args, "gpu_integrator", "medium"))
    selection_source = str(selected.get("_selection_source", "ST-LRPS"))

    with PdfPages(pdf_path) as pdf:
        pager = _ReportPager(
            pdf,
            title="GPU Batch Lunar Gravity Validation",
            subtitle="Fixed-step GPU propagation vs an adaptive reference",
        )
        pager.cover(
            meta=[
                ("Scenarios", str(args.random_scenarios)),
                ("Sampling method", str(getattr(args, "sampling_method", "random"))),
                ("Scenario seed", str(args.scenario_seed)),
                ("Scenario mode", str(getattr(args, "scenario_mode", "near_circular_altitude"))),
                ("Inclination sampling", str(getattr(args, "inclination_sampling", "uniform_deg"))),
                ("Altitude range", f"{args.altitude_min_km:.0f} - {args.altitude_max_km:.0f} km"),
                ("Duration", f"{args.duration_days:g} days"),
                ("GPU integrator", f"{gpu_integ} (fixed step {rk4_dt_text} s)"),
                ("Truth workers", str(getattr(args, "workers", 1))),
                ("Output cadence", f"{args.dt_out:g} s"),
                ("Precision", str(args.torch_dtype)),
                ("Truth", f"{args.truth.upper()} {truth_integ}"),
                ("GPU models", str(args.gpu_models)),
                ("Frame mode", str(args.batch_frame_mode)),
            ],
            note=("Reference note: the truth trajectories are a high-accuracy "
                  "adaptive numerical reference, not physical ground truth. GPU "
                  "fixed-step models carry both model and integration error."),
        )

        # Executive summary as a styled key/value table.
        med_eq = equivalent.get("median_rms", {}) if isinstance(equivalent, dict) else {}
        p95_eq = equivalent.get("p95_rms", {}) if isinstance(equivalent, dict) else {}
        fastest = min(runtime_rows, key=lambda r: r.get("total_runtime_s", np.inf)) if runtime_rows else {}
        most_acc = min(aggregate_rows, key=lambda r: r.get("median_rms_pos_err_km", np.inf)) if aggregate_rows else {}
        pager.table_page(
            "Executive Summary",
            ["Metric", "Value"],
            [
                ["Most accurate GPU model", str(most_acc.get("model", "n/a"))],
                ["Fastest GPU model", str(fastest.get("model", "n/a"))],
                ["ST-LRPS closest by median RMS", str(med_eq.get("closest_model", "n/a"))],
                ["ST-LRPS median-equivalent status",
                 str(med_eq.get("equivalent_degree_status", med_eq.get("status", "n/a")))],
                ["ST-LRPS closest by P95 RMS", str(p95_eq.get("closest_model", "n/a"))],
                [f"Best {selection_source} scenario", str(selected.get("best", {}).get("scenario_id", "n/a"))],
                [f"Representative {selection_source} scenario",
                 str(selected.get("representative", {}).get("scenario_id", "n/a"))],
                [f"Worst {selection_source} scenario", str(selected.get("worst", {}).get("scenario_id", "n/a"))],
            ],
        )

        if aggregate_rows:
            acc_rows = [[
                str(r.get("model", "")),
                _fmt_km(r.get("median_rms_pos_err_km", float("nan"))),
                _fmt_km(r.get("p95_rms_pos_err_km", float("nan"))),
                _fmt_km(r.get("max_rms_pos_err_km", float("nan"))),
                _fmt_km(r.get("median_along_rms_km", float("nan"))),
            ] for r in aggregate_rows]
            pager.table_page(
                "Accuracy Ranking",
                ["Model", "Median RMS [km]", "P95 RMS [km]", "Max RMS [km]", "Median Along [km]"],
                acc_rows,
                highlight_row=0,
                intro="Sorted best-to-worst by median RMS position error.",
            )

        if runtime_rows:
            run_rows = [[
                str(r.get("model", "")),
                f"{r.get('total_runtime_s', float('nan')):.3f}",
                f"{r.get('runtime_per_scenario_s', float('nan')):.5f}",
                f"{r.get('trajectory_steps_per_second', float('nan')):.1f}",
                f"{r.get('speedup_vs_truth_total', float('nan')):.2f}",
            ] for r in runtime_rows]
            pager.table_page(
                "Runtime",
                ["Model", "Total [s]", "Per scenario [s]", "Steps/s", "Speedup vs truth"],
                run_rows,
                intro="Wall-clock runtime and throughput for the GPU fixed-step propagation.",
            )

        # Shared context appended to each caption (N, duration, truth, unit note).
        ctx = (f"N = {args.random_scenarios} scenarios over {args.duration_days:g} day(s); "
               f"errors are relative to the {args.truth.upper()} {truth_integ} reference. "
               f"Axes auto-select display units (km/m/cm); CSV metrics remain in km.")
        small_n_note = (" For small N, distribution panels are diagnostic rather than "
                        "statistical." if int(args.random_scenarios) < 8 else "")
        figure_specs = [
            ("gpu_runtime_vs_accuracy.png",
             "Runtime–accuracy tradeoff across GPU models. Lower-left is better "
             "(faster and more accurate); the dashed line marks the Pareto front. " + ctx),
            ("gpu_accuracy_ranking_bar.png",
             "Per-model accuracy ranking (lollipops = median RMS, open diamonds = P95 RMS), "
             "sorted best-to-worst; ST-LRPS is highlighted. Lower is better. " + ctx),
            ("stlrps_equivalent_sh_degree.png",
             "ST-LRPS equivalent SH-degree estimate by interpolating median RMS error "
             "across the spherical-harmonic baselines. " + ctx),
            ("gpu_rms_error_distribution_boxplot.png",
             "Distribution of per-scenario RMS position error per model." + small_n_note + " " + ctx),
            ("ensemble_mean_position_error_vs_time.png",
             "Ensemble median position error vs time (shaded band = 25–75% across scenarios). " + ctx),
            ("ensemble_ric_rms_vs_time.png",
             "Ensemble radial/along-track/cross-track RMS error vs time. " + ctx),
            ("selected_representative_position_error_all_models.png",
             "Representative scenario: position error vs time. " + ctx),
            ("selected_representative_ric_error_all_models.png",
             "Representative scenario: radial/along-track/cross-track error vs time. " + ctx),
            ("selected_worst_position_error_all_models.png",
             "Worst-case scenario: position error vs time. " + ctx),
            ("selected_worst_ric_error_all_models.png",
             "Worst-case scenario: radial/along-track/cross-track error vs time. " + ctx),
        ]
        for png_name, caption in figure_specs:
            pager.figure_page("Figure", plots_dir / png_name, caption)

        notes = [
            "- The truth trajectories are a high-accuracy adaptive numerical reference, "
            "not physical ground truth.",
            "- GPU fixed-step SH models include both spherical-harmonic truncation error "
            "and integration error.",
            "- ST-LRPS includes surrogate-model error plus integration error.",
            f"- Frame mode: {args.batch_frame_mode}.",
        ]
        if args.batch_frame_mode == "inertial_fixed_legacy":
            notes.append("- Legacy frame mode is approximate and should not be used for final claims.")
        pager.text_page("Notes & Caveats", notes)

    print(f"  [report] GPU batch PDF saved: {pdf_path}", flush=True)
