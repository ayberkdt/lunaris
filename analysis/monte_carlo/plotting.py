# ST_LRPS/analysis/mc_plotting.py
# -*- coding: utf-8 -*-
"""
Monte Carlo Visualization
==========================

Matplotlib-based figures for Monte Carlo ensemble analysis results.

Available plots
---------------
plot_altitude_envelope
    σ-band altitude vs time with individual trajectories (thin lines).

plot_covariance_tubes_3d
    3-D orbit ensemble with 3-σ position error ellipsoids rendered as
    wireframe ellipsoids at selected epochs.

plot_position_covariance_history
    Position covariance diagonal (σ_x, σ_y, σ_z) and cross-terms vs time.

plot_impact_map
    Mollweide lunar surface projection of impact site distribution.

plot_impact_time_histogram
    Distribution of impact times across the ensemble.

plot_oe_dispersion
    Semi-major axis, eccentricity, and inclination spread vs time.

plot_mc_report
    Master function: all MC plots on a single multi-page PDF.

Design conventions
------------------
- Functions return ``matplotlib.figure.Figure`` objects; callers save/show.
- Default colour scheme and font sizes respect ``analysis.styling`` when
  available (graceful import fallback).
- No side effects beyond the returned figure.
"""

from __future__ import annotations

import math
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from analysis.formatting import safe_float, format_percent, format_days, format_km

try:
    import matplotlib
    matplotlib.use("Agg")   # non-interactive backend for headless runs
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    _MPL_OK = True
except ImportError:
    _MPL_OK = False
    plt = None  # type: ignore[assignment]

from analysis.monte_carlo.statistics import (
    EnsembleStatistics,
    ErrorEllipsoids,
    ImpactStatistics,
    MCStatistics,
    OEDispersion,
)
from common.constants import DAY_S, R_MOON_MEAN
from common.montecarlo_defs import MCRunResult


# =============================================================================
# 0.              INTERNAL HELPERS
# =============================================================================

def _require_mpl() -> None:
    if not _MPL_OK:
        raise ImportError(
            "matplotlib is required for MC plotting.  "
            "Install via:  pip install matplotlib"
        )


def _days(t_s: np.ndarray) -> np.ndarray:
    return t_s / DAY_S


def _km(arr: np.ndarray) -> np.ndarray:
    return arr / 1_000.0


def _default_figsize(landscape: bool = True) -> Tuple[float, float]:
    return (12.0, 7.0) if landscape else (8.0, 10.0)











def _style_ax(ax: Any, xlabel: str = "", ylabel: str = "", title: str = "") -> None:
    """Apply minimal consistent style to an Axes."""
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10)
    if title:
        ax.set_title(title, fontsize=11, fontweight="bold")
    ax.tick_params(labelsize=9)
    ax.grid(True, linewidth=0.5, alpha=0.4)


def _ellipsoid_wireframe(
    centre: np.ndarray,        # (3,) [m]
    semi_axes: np.ndarray,     # (3,) [m]
    eigvecs: np.ndarray,       # (3,3) columns = principal axes
    n_u: int = 16,
    n_v: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute (X, Y, Z) wireframe coordinates of an ellipsoid.

    Returns arrays of shape (n_u, n_v) in the original Cartesian frame.
    """
    u = np.linspace(0.0, 2.0 * math.pi, n_u)
    v = np.linspace(0.0, math.pi, n_v)
    # Parametric sphere
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))
    # Scale to ellipsoid in principal frame
    a, b, c = semi_axes
    pts = np.stack([a * xs, b * ys, c * zs], axis=-1)   # (n_u, n_v, 3)
    # Rotate back to Cartesian frame
    pts_rot = pts @ eigvecs.T                             # (n_u, n_v, 3)
    # Translate to centre
    cx, cy, cz = _km(centre)
    Xw = pts_rot[..., 0] / 1_000.0 + cx
    Yw = pts_rot[..., 1] / 1_000.0 + cy
    Zw = pts_rot[..., 2] / 1_000.0 + cz
    return Xw, Yw, Zw


def plot_mc_summary(
    result: MCRunResult,
    mc_stats: MCStatistics,
    *,
    figsize: Optional[Tuple[float, float]] = None,
    title: str = "Monte Carlo Executive Summary",
) -> Any:
    """
    Build a compact first page that surfaces the run's key risk and dispersion metrics.

    The detailed pages remain important for engineering inspection, but a concise
    summary page helps the operator understand the ensemble outcome immediately
    before diving into the heavier plots.
    """

    _require_mpl()

    fig = plt.figure(figsize=figsize or (11.0, 8.5))
    fig.patch.set_facecolor("white")

    impacts = mc_stats.impacts
    ensemble = mc_stats.ensemble
    ellipsoids = mc_stats.ellipsoids

    total_samples = int(result.n_samples)
    n_impacts = int(impacts.n_impacts)
    n_survivors = max(0, total_samples - n_impacts)
    t_start = float(result.t[0]) if len(result.t) else 0.0
    t_end = float(result.t[-1]) if len(result.t) else 0.0
    duration_s = max(0.0, t_end - t_start)
    tube_km = np.asarray(ellipsoids.tube_radii(), dtype=np.float64) / 1_000.0
    peak_tube_km = float(np.nanmax(tube_km)) if tube_km.size else math.nan
    final_alt_mean_km = float(ensemble.alt_mean[-1]) if ensemble.alt_mean.size else math.nan
    final_alt_std_km = float(ensemble.alt_std[-1]) if ensemble.alt_std.size else math.nan
    initial_alt_mean_km = float(ensemble.alt_mean[0]) if ensemble.alt_mean.size else math.nan
    initial_alt_std_km = float(ensemble.alt_std[0]) if ensemble.alt_std.size else math.nan

    fig.text(0.06, 0.94, title, fontsize=22, fontweight="bold", color="#10233F")
    fig.text(
        0.06,
        0.905,
        "Risk, ensemble spread, and endpoint metrics for the current Monte Carlo archive.",
        fontsize=10.5,
        color="#4A5C7A",
    )

    kpi_specs = [
        ("Impact Probability", format_percent(impacts.p_impact), "#C65151"),
        ("Survival Rate", format_percent(n_survivors / max(1, total_samples)), "#1E7A57"),
        ("Peak 3-sigma Tube", format_km(peak_tube_km), "#355CBE"),
    ]
    for idx, (label, value, accent) in enumerate(kpi_specs):
        left = 0.06 + idx * 0.30
        ax = fig.add_axes([left, 0.73, 0.26, 0.13])
        ax.axis("off")
        ax.add_patch(
            plt.Rectangle(
                (0.0, 0.0),
                1.0,
                1.0,
                transform=ax.transAxes,
                facecolor="#F7FAFF",
                edgecolor="#D7E1F0",
                linewidth=1.2,
            )
        )
        ax.text(0.06, 0.72, label, fontsize=9.5, color="#5D6E88", fontweight="bold", transform=ax.transAxes)
        ax.text(0.06, 0.28, value, fontsize=22, color=accent, fontweight="bold", transform=ax.transAxes)

    left_metrics = [
        ("Scenarios", f"{total_samples:,}"),
        ("Impacted Samples", f"{n_impacts:,} ({format_percent(impacts.p_impact)})"),
        ("95% Wilson CI", f"{format_percent(impacts.p_impact_ci95[0])} to {format_percent(impacts.p_impact_ci95[1])}"),
        ("Mean Impact Epoch", "No impacts" if not math.isfinite(safe_float(impacts.t_impact_mean)) else format_days(impacts.t_impact_mean)),
        ("Impact Time 1-sigma", format_days(impacts.t_impact_std)),
        ("Output Epochs", f"{int(len(result.t)):,}"),
    ]
    right_metrics = [
        ("Trajectory Span", format_days(duration_s)),
        ("Initial Mean Altitude", format_km(initial_alt_mean_km)),
        ("Initial Altitude 1-sigma", format_km(initial_alt_std_km)),
        ("Final Mean Altitude", format_km(final_alt_mean_km)),
        ("Final Altitude 1-sigma", format_km(final_alt_std_km)),
        ("OE Dispersion", "Included" if mc_stats.oe_disp is not None else "Not requested"),
    ]

    def _draw_metric_column(bounds: Sequence[float], heading: str, rows: Sequence[Tuple[str, str]]) -> None:
        ax = fig.add_axes(bounds)
        ax.axis("off")
        ax.text(0.0, 1.0, heading, fontsize=11.5, fontweight="bold", color="#233754", va="top")
        y = 0.88
        for label, value in rows:
            ax.add_patch(
                plt.Rectangle(
                    (0.0, y - 0.10),
                    1.0,
                    0.11,
                    transform=ax.transAxes,
                    facecolor="#FBFCFE",
                    edgecolor="#E3EAF4",
                    linewidth=0.9,
                )
            )
            ax.text(0.03, y - 0.028, label, fontsize=9.4, color="#5D6E88", transform=ax.transAxes)
            ax.text(
                0.97,
                y - 0.028,
                value,
                fontsize=10.4,
                color="#10233F",
                fontweight="bold",
                ha="right",
                transform=ax.transAxes,
            )
            y -= 0.135

    _draw_metric_column([0.06, 0.16, 0.41, 0.47], "Risk Metrics", left_metrics)
    _draw_metric_column([0.53, 0.16, 0.41, 0.47], "Dispersion Metrics", right_metrics)

    fig.text(
        0.06,
        0.06,
        "Interpretation: use the detailed pages for time-history structure, covariance growth, and impact geography.",
        fontsize=9,
        color="#5D6E88",
    )
    fig.text(
        0.94,
        0.06,
        "ST_LRPS Monte Carlo Report",
        fontsize=9,
        color="#8A97AC",
        ha="right",
    )

    return fig


# =============================================================================
# 1.              ALTITUDE ENVELOPE
# =============================================================================

def plot_altitude_envelope(
    result: MCRunResult,
    stats: EnsembleStatistics,
    *,
    sigma_levels: Sequence[float] = (1.0, 2.0, 3.0),
    max_traj: int = 50,
    r_ref_m: float = R_MOON_MEAN,
    figsize: Optional[Tuple[float, float]] = None,
    title: str = "Altitude Envelope – Monte Carlo Ensemble",
) -> Any:
    """
    Plot altitude vs time with σ-bands and individual sample trajectories.

    Parameters
    ----------
    result : MCRunResult
    stats  : EnsembleStatistics
    sigma_levels : σ-bands to shade (default 1σ, 2σ, 3σ)
    max_traj : max number of individual trajectories to overlay (thin lines)

    Returns
    -------
    matplotlib Figure
    """
    _require_mpl()
    fig, ax = plt.subplots(figsize=figsize or _default_figsize())

    t_days = _days(stats.t)

    # Individual trajectories (random subset)
    N = result.n_samples
    n_show = min(max_traj, N)
    rng = np.random.default_rng(1)
    indices = rng.choice(N, size=n_show, replace=False)

    for i in indices:
        alt_i = (_km(np.linalg.norm(result.Y[:, i, :3], axis=1))
                 - r_ref_m / 1_000.0)
        imp = result.impact_mask[i] > 0.5
        color = "#e06060" if imp else "#6090c0"
        ax.plot(t_days, alt_i, color=color, alpha=0.15, linewidth=0.7)

    # σ-bands (shaded)
    mu_alt = stats.alt_mean
    s_alt  = stats.alt_std
    alphas = [0.30, 0.20, 0.12]
    band_colors = ["#2060a0", "#2060a0", "#2060a0"]

    for k, sig in enumerate(sigma_levels):
        lo = mu_alt - sig * s_alt
        hi = mu_alt + sig * s_alt
        alpha_fill = alphas[k] if k < len(alphas) else 0.10
        ax.fill_between(t_days, lo, hi, alpha=alpha_fill,
                        color=band_colors[k % len(band_colors)],
                        label=f"±{sig:.0f}σ band")

    # Mean altitude
    ax.plot(t_days, mu_alt, color="#1a3a6a", linewidth=1.8, label="Mean alt.", zorder=5)

    # Lunar surface line
    ax.axhline(0.0, color="#a05020", linewidth=1.2, linestyle="--", label="Surface", zorder=4)

    ax.set_xlim(t_days[0], t_days[-1])
    _style_ax(ax, xlabel="Time [days]", ylabel="Altitude [km]", title=title)
    ax.legend(fontsize=9, loc="upper right")

    fig.tight_layout()
    return fig


# =============================================================================
# 2.              3-D ORBIT ENSEMBLE WITH ERROR ELLIPSOIDS
# =============================================================================

def plot_covariance_tubes_3d(
    result: MCRunResult,
    ellipsoids: ErrorEllipsoids,
    *,
    max_traj: int = 30,
    ellipsoid_epochs: Optional[Sequence[int]] = None,
    r_moon_km: float = R_MOON_MEAN / 1_000.0,
    figsize: Optional[Tuple[float, float]] = None,
    title: str = "3-D Orbit Ensemble with 3σ Error Ellipsoids",
) -> Any:
    """
    3-D view of the trajectory ensemble with 3-σ error ellipsoids at selected epochs.

    Parameters
    ----------
    result     : MCRunResult
    ellipsoids : ErrorEllipsoids from compute_error_ellipsoids()
    ellipsoid_epochs : list of integer epoch indices for ellipsoid rendering.
                       Defaults to 5 evenly spaced indices.
    max_traj   : max individual trajectories to show

    Returns
    -------
    matplotlib Figure
    """
    _require_mpl()
    fig = plt.figure(figsize=figsize or (12, 9))
    ax  = fig.add_subplot(111, projection="3d")

    T = int(result.t.shape[0])
    N = result.n_samples

    # Default epoch selection
    if ellipsoid_epochs is None:
        step = max(1, T // 5)
        ellipsoid_epochs = list(range(0, T, step))[:6]

    # Individual trajectories
    n_show = min(max_traj, N)
    rng = np.random.default_rng(2)
    idx_show = rng.choice(N, size=n_show, replace=False)

    for i in idx_show:
        pos_km = _km(result.Y[:, i, :3])
        imp = result.impact_mask[i] > 0.5
        c = "#d04040" if imp else "#4070b0"
        ax.plot(pos_km[:, 0], pos_km[:, 1], pos_km[:, 2],
                color=c, alpha=0.15, linewidth=0.6)

    # Mean trajectory
    mean_km = _km(np.mean(result.Y[:, :, :3], axis=1))
    ax.plot(mean_km[:, 0], mean_km[:, 1], mean_km[:, 2],
            color="#1a3a6a", linewidth=2.0, label="Mean trajectory", zorder=5)

    # Error ellipsoids at selected epochs
    ell_colors = ["#e08000", "#20a040", "#8020c0", "#c04080", "#0080a0", "#a04000"]
    for j, ep_idx in enumerate(ellipsoid_epochs):
        if ep_idx >= T:
            continue
        Xw, Yw, Zw = _ellipsoid_wireframe(
            ellipsoids.centres[ep_idx],
            ellipsoids.semi_axes[ep_idx],
            ellipsoids.eigvecs[ep_idx],
        )
        c = ell_colors[j % len(ell_colors)]
        ax.plot_wireframe(Xw, Yw, Zw, color=c, alpha=0.55, linewidth=0.8,
                          rstride=2, cstride=2)

    # Lunar sphere (low-res)
    u_s = np.linspace(0, 2 * math.pi, 30)
    v_s = np.linspace(0, math.pi, 20)
    xs = r_moon_km * np.outer(np.cos(u_s), np.sin(v_s))
    ys = r_moon_km * np.outer(np.sin(u_s), np.sin(v_s))
    zs = r_moon_km * np.outer(np.ones_like(u_s), np.cos(v_s))
    ax.plot_surface(xs, ys, zs, color="#d0c0a0", alpha=0.20, linewidth=0)

    ax.set_xlabel("X [km]", fontsize=9); ax.set_ylabel("Y [km]", fontsize=9)
    ax.set_zlabel("Z [km]", fontsize=9); ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


# =============================================================================
# 3.              POSITION COVARIANCE HISTORY
# =============================================================================

def plot_position_covariance_history(
    stats: EnsembleStatistics,
    *,
    figsize: Optional[Tuple[float, float]] = None,
    title: str = "Position Covariance History",
) -> Any:
    """
    Plot σ_x, σ_y, σ_z (diagonal) and cross-correlations ρ_xy, ρ_xz, ρ_yz vs time.

    Returns
    -------
    matplotlib Figure
    """
    _require_mpl()
    fig, axes = plt.subplots(2, 1, figsize=figsize or (11, 8), sharex=True)
    t_days = _days(stats.t)

    # Top: σ in each Cartesian direction [km]
    ax0 = axes[0]
    for j, lbl in enumerate(("σ_x", "σ_y", "σ_z")):
        ax0.plot(t_days, _km(stats.std[:, j]), label=lbl, linewidth=1.5)
    _style_ax(ax0, ylabel="Position 1-σ [km]", title=title)
    ax0.legend(fontsize=9)

    # Bottom: off-diagonal correlations ρ = C_ij / (σ_i σ_j)
    ax1 = axes[1]
    pairs = [(0, 1, "ρ_xy"), (0, 2, "ρ_xz"), (1, 2, "ρ_yz")]
    for i, j, lbl in pairs:
        si = stats.std[:, i]; sj = stats.std[:, j]
        denom = si * sj
        denom = np.where(denom > 1e-30, denom, np.nan)
        rho   = stats.cov[:, i, j] / denom
        ax1.plot(t_days, rho, label=lbl, linewidth=1.5)

    ax1.axhline(0.0, color="k", linewidth=0.5, linestyle=":")
    ax1.set_ylim(-1.1, 1.1)
    _style_ax(ax1, xlabel="Time [days]", ylabel="Correlation coefficient", title="")
    ax1.legend(fontsize=9)

    fig.tight_layout()
    return fig


# =============================================================================
# 4.              IMPACT MAP (MOLLWEIDE PROJECTION)
# =============================================================================

def plot_impact_map(
    impacts: ImpactStatistics,
    *,
    figsize: Optional[Tuple[float, float]] = None,
    title: str = "Monte Carlo Impact Site Distribution",
) -> Any:
    """
    Mollweide projection of lunar surface impact sites.

    Returns
    -------
    matplotlib Figure
    """
    _require_mpl()

    if impacts.n_impacts == 0:
        fig, ax = plt.subplots(figsize=figsize or _default_figsize())
        ax.text(0.5, 0.5, "No impacts detected", ha="center", va="center",
                transform=ax.transAxes, fontsize=13)
        ax.set_title(title, fontsize=11, fontweight="bold")
        return fig

    fig = plt.figure(figsize=figsize or _default_figsize())
    ax  = fig.add_subplot(111, projection="mollweide")

    lon_rad = np.radians(impacts.lon_deg)
    lat_rad = np.radians(impacts.lat_deg)

    sc = ax.scatter(
        lon_rad, lat_rad,
        c=np.arange(impacts.n_impacts),
        cmap="plasma",
        s=20, alpha=0.75,
        zorder=4,
    )
    plt.colorbar(sc, ax=ax, orientation="horizontal", fraction=0.04,
                 pad=0.06, label="Sample index")

    ax.set_title(
        f"{title}\n"
        f"N_hits = {impacts.n_impacts} / {impacts.n_total}  "
        f"  P = {impacts.p_impact:.3f}  "
        f"[{impacts.p_impact_ci95[0]:.3f}, {impacts.p_impact_ci95[1]:.3f}] 95%CI",
        fontsize=10,
    )
    ax.grid(True, linewidth=0.4, alpha=0.5)
    fig.tight_layout()
    return fig


# =============================================================================
# 5.              IMPACT TIME HISTOGRAM
# =============================================================================

def plot_impact_time_histogram(
    impacts: ImpactStatistics,
    result: MCRunResult,
    *,
    n_bins: int = 30,
    figsize: Optional[Tuple[float, float]] = None,
    title: str = "Impact Time Distribution",
) -> Any:
    """
    Histogram of impact times across the ensemble.

    Returns
    -------
    matplotlib Figure
    """
    _require_mpl()
    fig, ax = plt.subplots(figsize=figsize or _default_figsize())

    t_hit = result.t_impact[result.impact_mask > 0.5]
    t_hit = t_hit[np.isfinite(t_hit)] / DAY_S    # convert to days

    if len(t_hit) == 0:
        ax.text(0.5, 0.5, "No impacts detected", ha="center", va="center",
                transform=ax.transAxes, fontsize=13)
    else:
        ax.hist(t_hit, bins=n_bins, edgecolor="white", linewidth=0.4,
                color="#2060a0", alpha=0.85)
        if math.isfinite(impacts.t_impact_mean):
            ax.axvline(impacts.t_impact_mean / DAY_S, color="r", linewidth=1.5,
                       linestyle="--", label=f"Mean = {impacts.t_impact_mean / DAY_S:.2f} d")
            ax.legend(fontsize=9)

    _style_ax(ax, xlabel="Impact time [days]", ylabel="Count", title=title)
    fig.tight_layout()
    return fig


# =============================================================================
# 6.              ORBITAL ELEMENT DISPERSION
# =============================================================================

def plot_oe_dispersion(
    oe: OEDispersion,
    *,
    figsize: Optional[Tuple[float, float]] = None,
    title: str = "Orbital Element Dispersion",
) -> Any:
    """
    Plot mean ± 1-σ bands for a, e, and inc vs time.

    Returns
    -------
    matplotlib Figure
    """
    _require_mpl()
    fig, axes = plt.subplots(3, 1, figsize=figsize or (11, 10), sharex=True)
    t_days = _days(oe.t)

    def _panel(ax, mu, s, ylabel, title_):
        ax.plot(t_days, mu, color="#1a3a6a", linewidth=1.5)
        ax.fill_between(t_days, mu - s, mu + s, alpha=0.30, color="#2060a0", label="±1σ")
        _style_ax(ax, ylabel=ylabel, title=title_)
        ax.legend(fontsize=9)

    _panel(axes[0], oe.a_mean_km,    oe.a_std_km,    "a [km]",   "Semi-major axis")
    _panel(axes[1], oe.e_mean,       oe.e_std,       "e [-]",    "Eccentricity")
    _panel(axes[2], oe.inc_mean_deg, oe.inc_std_deg, "i [deg]",  "Inclination")
    axes[2].set_xlabel("Time [days]", fontsize=10)

    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


# =============================================================================
# 7.              MASTER MC REPORT (multi-figure PDF)
# =============================================================================

def plot_mc_report(
    result: MCRunResult,
    mc_stats: MCStatistics,
    *,
    output_path: Optional[str] = None,
    show: bool = False,
) -> List[Any]:
    """
    Generate the full MC report bundle and optionally save it to a multi-page PDF.

    Parameters
    ----------
    result     : MCRunResult
    mc_stats   : MCStatistics (from compute_mc_statistics)
    output_path: if set, saves all figures to this PDF path
    show       : if True, calls plt.show() after all figures are created

    Returns
    -------
    figs : list of matplotlib Figure objects
        Starts with an executive-summary page followed by the detailed plots.
    """
    _require_mpl()

    figs: List[Any] = []

    # Figure 0: Executive summary
    figs.append(
        plot_mc_summary(result, mc_stats)
    )

    # Figure 1: Altitude envelope
    figs.append(
        plot_altitude_envelope(result, mc_stats.ensemble)
    )

    # Figure 2: Covariance tubes 3-D
    figs.append(
        plot_covariance_tubes_3d(result, mc_stats.ellipsoids)
    )

    # Figure 3: Position covariance history
    figs.append(
        plot_position_covariance_history(mc_stats.ensemble)
    )

    # Figure 4: Impact map
    figs.append(
        plot_impact_map(mc_stats.impacts)
    )

    # Figure 5: Impact time histogram
    figs.append(
        plot_impact_time_histogram(mc_stats.impacts, result)
    )

    # Figure 6: OE dispersion (if available)
    if mc_stats.oe_disp is not None:
        figs.append(
            plot_oe_dispersion(mc_stats.oe_disp)
        )

    # Save to PDF
    if output_path is not None:
        from matplotlib.backends.backend_pdf import PdfPages
        from pathlib import Path
        p = Path(output_path).expanduser().resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        with PdfPages(str(p)) as pdf:
            info = pdf.infodict()
            info["Title"] = "ST_LRPS Monte Carlo Analysis Report"
            info["Subject"] = "Monte Carlo uncertainty, dispersion, and impact-risk summary"
            for fig in figs:
                pdf.savefig(fig, bbox_inches="tight")
        print(f"[MC] Report saved → {p}", flush=True)

    if show:
        plt.show()

    return figs


# =============================================================================
# 8.                        PUBLIC API
# =============================================================================

__all__ = [
    "plot_mc_summary",
    "plot_altitude_envelope",
    "plot_covariance_tubes_3d",
    "plot_position_covariance_history",
    "plot_impact_map",
    "plot_impact_time_histogram",
    "plot_oe_dispersion",
    "plot_mc_report",
]
