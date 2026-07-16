"""Editorial figures and PDF pages driven by :mod:`lunaris.analysis.contracts`."""

from __future__ import annotations

import logging
import math
import textwrap
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from lunaris.analysis.artifacts import (
    load_analysis_artifacts,
    write_analysis_artifacts,
    write_artifact_manifest,
)
from lunaris.analysis.contracts import MetricValue, OrbitAnalysisResult

from .styling import (
    REPORT_LAYOUT,
    REPORT_PALETTE,
    REPORT_SERIES_COLORS,
    REPORT_STATUS_COLORS,
    REPORT_TYPOGRAPHY,
    apply_rcparams,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ReportPreset:
    name: str
    dpi: int
    max_plot_points: int
    vector_figures: bool
    page_sections: tuple[str, ...]


REPORT_PRESETS: Mapping[str, ReportPreset] = {
    "quick": ReportPreset(
        name="quick",
        dpi=180,
        max_plot_points=2500,
        vector_figures=False,
        page_sections=("cover", "executive", "orbit", "events", "numerical"),
    ),
    "standard": ReportPreset(
        name="standard",
        dpi=240,
        max_plot_points=6000,
        vector_figures=False,
        page_sections=(
            "cover",
            "executive",
            "configuration",
            "orbit",
            "spatial",
            "elements",
            "events",
            "numerical",
            "diagnostics",
            "force_budget",
            "force_dynamics",
            "provenance",
            "assets",
        ),
    ),
    "paper": ReportPreset(
        name="paper",
        dpi=360,
        max_plot_points=12_000,
        vector_figures=True,
        page_sections=(
            "cover",
            "executive",
            "configuration",
            "orbit",
            "spatial",
            "elements",
            "events",
            "numerical",
            "diagnostics",
            "force_budget",
            "force_dynamics",
            "provenance",
            "assets",
        ),
    ),
}


_LINE_STYLES = ("-", "--", "-.", ":")
_MARKERS = (None, "o", "s", "^")


def _preset(name: str) -> ReportPreset:
    try:
        return REPORT_PRESETS[str(name).strip().lower()]
    except KeyError as exc:
        raise ValueError("unknown report preset; expected quick, standard, or paper") from exc


def _figure(*, landscape: bool = False) -> plt.Figure:
    apply_rcparams()
    size = (
        (REPORT_LAYOUT["page_height_in"], REPORT_LAYOUT["page_width_in"])
        if landscape
        else (REPORT_LAYOUT["page_width_in"], REPORT_LAYOUT["page_height_in"])
    )
    fig = plt.figure(figsize=size, facecolor=REPORT_PALETTE["paper"])
    return fig


def _plot_indices(count: int, cap: int) -> np.ndarray:
    if count <= cap:
        return np.arange(count, dtype=np.int64)
    return np.linspace(0, count - 1, cap, dtype=np.int64)


def _metric(result: OrbitAnalysisResult, metric_id: str) -> MetricValue | None:
    return result.metric_map.get(metric_id)


def _display(metric: MetricValue | None, *, compact: bool = False) -> str:
    if metric is None:
        return "Unavailable"
    if metric.value is None:
        return "Unavailable"
    value = metric.value
    if not isinstance(value, int | float) or isinstance(value, bool):
        return str(value)
    number = float(value)
    if metric.unit == "m" and any(token in metric.metric_id for token in ("altitude", "orbit.a")):
        return f"{number / 1000.0:,.2f} km" if compact else f"{number / 1000.0:,.3f} km"
    if metric.unit == "rad":
        return f"{math.degrees(number):,.3f} deg"
    if metric.unit == "s" and metric.metric_id in {"mission.duration", "orbit.period"}:
        if number >= 86_400.0:
            return f"{number / 86_400.0:,.3f} d"
        if number >= 3600.0:
            return f"{number / 3600.0:,.2f} h"
        if number >= 60.0:
            return f"{number / 60.0:,.2f} min"
    if metric.unit == "1" and float(number).is_integer():
        return f"{int(number):,}"
    if abs(number) != 0.0 and (abs(number) >= 1.0e5 or abs(number) < 1.0e-3):
        rendered = f"{number:.3e}"
    else:
        rendered = f"{number:,.5g}"
    unit = "" if metric.unit == "1" else (metric.unit or "")
    return f"{rendered} {unit}".strip()


def _style_axes(ax: plt.Axes, *, grid: bool = True) -> None:
    ax.set_facecolor(REPORT_PALETTE["white"])
    ax.tick_params(colors=REPORT_PALETTE["muted"], labelsize=7.5, length=3.5)
    ax.xaxis.label.set_color(REPORT_PALETTE["graphite_mid"])
    ax.yaxis.label.set_color(REPORT_PALETTE["graphite_mid"])
    ax.xaxis.label.set_size(8.2)
    ax.yaxis.label.set_size(8.2)
    for name, spine in ax.spines.items():
        spine.set_visible(name in {"left", "bottom"})
        spine.set_color(REPORT_PALETTE["rule"])
        spine.set_linewidth(0.8)
    ax.grid(grid, color=REPORT_PALETTE["grid"], linewidth=0.55, alpha=0.75)
    ax.set_axisbelow(True)


@dataclass(frozen=True, slots=True)
class _ForceModelStatus:
    label: str
    active: bool
    measured: bool
    color_key: str


_FORCE_MODEL_SPECS: tuple[tuple[str, tuple[str, ...], str, tuple[str, ...]], ...] = (
    ("Lunar gravity", (), "central_gravity", ("Gravity (PM)", "Gravity (SH)", "Gravity ST-LRPS")),
    ("Spherical harmonics", ("enable_sh",), "spherical_harmonics", ("Gravity (SH)", "Gravity ST-LRPS")),
    ("Earth third-body", ("enable_3rd_body_earth",), "third_body_earth", ("3rd Body (Earth)",)),
    ("Sun third-body", ("enable_3rd_body_sun",), "third_body_sun", ("3rd Body (Sun)",)),
    ("Earth J2", ("enable_earth_j2",), "third_body_earth", ("3rd Body (Earth J2)",)),
    ("Solar radiation", ("enable_srp",), "srp", ("SRP",)),
    ("Solid tides", ("enable_tides_k2", "enable_tides_k3"), "solid_tides", ("Solid Tides",)),
    ("Relativity 1PN", ("enable_relativity_1pn",), "relativity", ("Relativity (1PN)",)),
    ("Lunar albedo", ("enable_albedo",), "albedo", ("Lunar albedo pressure",)),
    ("Thermal IR", ("enable_thermal",), "thermal_ir", ("Lunar thermal IR pressure",)),
)


def _force_model_statuses(result: OrbitAnalysisResult) -> tuple[_ForceModelStatus, ...]:
    flags = result.config_snapshot.get("flags", {})
    flag_map = flags if isinstance(flags, Mapping) else {}
    contribution_map = {item.label: item for item in result.force_contributions}
    statuses: list[_ForceModelStatus] = []
    for label, flag_names, color_key, contribution_names in _FORCE_MODEL_SPECS:
        active = True if not flag_names else any(bool(flag_map.get(name, False)) for name in flag_names)
        measured = any(
            bool(contribution_map.get(name) and contribution_map[name].available)
            for name in contribution_names
        )
        statuses.append(
            _ForceModelStatus(
                label=label,
                active=active,
                measured=active and measured,
                color_key=color_key,
            )
        )
    return tuple(statuses)


def _draw_force_status_matrix(
    ax: plt.Axes,
    result: OrbitAnalysisResult,
    *,
    columns: int,
) -> None:
    ax.axis("off")
    statuses = _force_model_statuses(result)
    rows = int(math.ceil(len(statuses) / columns))
    cell_w = 1.0 / columns
    cell_h = 1.0 / rows
    for index, status in enumerate(statuses):
        column = index % columns
        row = index // columns
        x = column * cell_w + 0.008
        y = 1.0 - (row + 1) * cell_h + 0.07 * cell_h
        width = cell_w - 0.016
        height = cell_h * 0.84
        color = REPORT_SERIES_COLORS.get(status.color_key, REPORT_PALETTE["muted"])
        edge = color if status.active else REPORT_PALETTE["rule"]
        face = REPORT_PALETTE["white"] if status.active else REPORT_PALETTE["paper_alt"]
        ax.add_patch(
            plt.Rectangle(
                (x, y),
                width,
                height,
                facecolor=face,
                edgecolor=edge,
                linewidth=0.75 if status.active else 0.5,
            )
        )
        ax.add_patch(
            plt.Rectangle(
                (x, y),
                0.009,
                height,
                facecolor=color if status.active else REPORT_PALETTE["rule"],
                linewidth=0,
            )
        )
        ax.text(
            x + 0.022,
            y + height * 0.62,
            status.label,
            fontsize=5.8,
            fontweight="bold" if status.active else "normal",
            color=REPORT_PALETTE["graphite"] if status.active else REPORT_PALETTE["muted"],
            va="center",
        )
        state = "ON / sampled" if status.measured else ("ON / configured" if status.active else "OFF")
        ax.text(
            x + 0.022,
            y + height * 0.25,
            state,
            fontsize=5.1,
            color=color if status.active else REPORT_PALETTE["muted"],
            va="center",
        )


def _ranked_unique_forces(
    result: OrbitAnalysisResult,
    *,
    limit: int | None = None,
) -> list[Any]:
    ranked = [
        item
        for item in result.force_contributions
        if item.active
        and item.available
        and item.included_in_noncentral_ranking
        and item.median_m_s2 is not None
        and item.median_m_s2 > 0.0
    ]
    ranked.sort(key=lambda item: float(item.median_m_s2 or 0.0), reverse=True)
    return ranked if limit is None else ranked[:limit]


def _force_display_label(label: str) -> str:
    """Keep report-facing labels compact without changing artifact identifiers."""
    replacements = {
        "Relativity (Moon Schwarzschild)": "Moon 1PN",
        "Relativity (External 1PN)": "External 1PN",
        "3rd Body (Earth J2)": "Earth J2",
    }
    return replacements.get(label, label)


def _osculating_envelope(result: OrbitAnalysisResult) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a = np.asarray(result.series.semi_major_axis_m, dtype=np.float64)
    e = np.asarray(result.series.eccentricity, dtype=np.float64)
    body_radius = float(np.nanmedian(result.series.radius_m - result.series.altitude_m))
    periselene_km = (a * (1.0 - e) - body_radius) / 1000.0
    aposelene_km = (a * (1.0 + e) - body_radius) / 1000.0
    energy = np.asarray(result.series.specific_energy_j_kg, dtype=np.float64)
    mu_samples = -2.0 * energy * a
    finite_mu = mu_samples[np.isfinite(mu_samples) & (mu_samples > 0.0)]
    period_h = np.full_like(a, np.nan)
    if finite_mu.size:
        mu = float(np.median(finite_mu))
        valid = np.isfinite(a) & (a > 0.0)
        period_h[valid] = 2.0 * np.pi * np.sqrt(a[valid] ** 3 / mu) / 3600.0
    return periselene_km, aposelene_km, period_h


def _plot_osculating_envelope(ax: plt.Axes, result: OrbitAnalysisResult, idx: np.ndarray) -> None:
    time_days = result.series.t_s[idx] / 86_400.0
    periselene_km, aposelene_km, _ = _osculating_envelope(result)
    ax.plot(
        time_days,
        periselene_km[idx],
        color=REPORT_SERIES_COLORS["periselene"],
        linewidth=1.2,
        label="Periselene bound",
    )
    ax.plot(
        time_days,
        aposelene_km[idx],
        color=REPORT_SERIES_COLORS["aposelene"],
        linewidth=1.2,
        linestyle="--",
        label="Aposelene bound",
    )
    ax.set_xlabel("Simulation time [days]")
    ax.set_ylabel("Osculating altitude [km]")
    ax.set_title("Osculating altitude bounds", loc="left", fontsize=8.8, fontweight="bold")
    _style_axes(ax)
    ax.legend(frameon=False, fontsize=6.2, ncol=2, loc="best")


def _plot_period_history(ax: plt.Axes, result: OrbitAnalysisResult, idx: np.ndarray) -> None:
    time_days = result.series.t_s[idx] / 86_400.0
    _, _, period_h = _osculating_envelope(result)
    finite = np.isfinite(period_h[idx])
    if np.any(finite):
        ax.plot(
            time_days[finite],
            period_h[idx][finite],
            color=REPORT_SERIES_COLORS["orbital_period"],
            linewidth=1.25,
        )
        ax.scatter(
            [time_days[finite][0], time_days[finite][-1]],
            [period_h[idx][finite][0], period_h[idx][finite][-1]],
            color=REPORT_SERIES_COLORS["orbital_period"],
            s=13,
            zorder=4,
        )
    else:
        ax.text(0.5, 0.5, "Bound-orbit period unavailable", ha="center", va="center", fontsize=7.0)
    ax.set_xlabel("Simulation time [days]")
    ax.set_ylabel("Osculating period [h]")
    ax.set_title("Period evolution", loc="left", fontsize=8.8, fontweight="bold")
    _style_axes(ax)


def _plot_force_range(ax: plt.Axes, result: OrbitAnalysisResult, *, limit: int = 8) -> None:
    ranked = _ranked_unique_forces(result, limit=limit)
    if not ranked:
        ax.axis("off")
        ax.text(0.5, 0.55, "Ranked perturbation statistics unavailable", ha="center", fontsize=9.0)
        return
    labels = [_force_display_label(item.label) for item in ranked][::-1]
    y = np.arange(len(ranked), dtype=float)
    for row, item in enumerate(ranked[::-1]):
        minimum = max(float(item.minimum_m_s2 or 0.0), 1.0e-300)
        median = max(float(item.median_m_s2 or 0.0), 1.0e-300)
        p95 = max(float(item.p95_m_s2 or 0.0), 1.0e-300)
        maximum = max(float(item.maximum_m_s2 or 0.0), 1.0e-300)
        color = REPORT_SERIES_COLORS.get(_force_color_key(item.label), REPORT_PALETTE["muted"])
        ax.hlines(row, minimum, maximum, color=color, linewidth=1.4, alpha=0.85)
        ax.scatter(median, row, color=color, s=28, marker="o", edgecolor=REPORT_PALETTE["white"], linewidth=0.5, zorder=4)
        ax.scatter(p95, row, color=color, s=24, marker="D", facecolor=REPORT_PALETTE["white"], edgecolor=color, linewidth=0.9, zorder=4)
    ax.set_yticks(y, labels)
    ax.set_xscale("log")
    ax.set_xlabel("Acceleration magnitude [m/s^2]")
    ax.set_title("Unique non-central force scale", loc="left", fontsize=9.2, fontweight="bold")
    _style_axes(ax)
    ax.grid(False, axis="y")
    ax.plot([], [], marker="o", color=REPORT_PALETTE["graphite"], linestyle="none", label="Median")
    ax.plot([], [], marker="D", markerfacecolor=REPORT_PALETTE["white"], markeredgecolor=REPORT_PALETTE["graphite"], linestyle="none", label="P95")
    ax.legend(frameon=False, fontsize=6.2, ncol=2, loc="lower right")


def _plot_force_history(ax: plt.Axes, result: OrbitAnalysisResult, *, limit: int = 6) -> None:
    if result.force_time_s is None or not result.force_magnitudes_m_s2:
        ax.axis("off")
        ax.text(0.5, 0.55, "Force time history unavailable", ha="center", fontsize=9.0)
        return
    selected = [item.label for item in _ranked_unique_forces(result, limit=limit)]
    total_name = "Total non-central acceleration"
    if total_name in result.force_magnitudes_m_s2:
        selected.append(total_name)
    time_days = np.asarray(result.force_time_s, dtype=np.float64) / 86_400.0
    for index, name in enumerate(selected):
        values = result.force_magnitudes_m_s2.get(name)
        if values is None:
            continue
        arr = np.asarray(values, dtype=np.float64)
        color_key = "total_noncentral" if name == total_name else _force_color_key(name)
        ax.plot(
            time_days,
            arr,
            label=name,
            color=REPORT_SERIES_COLORS.get(color_key, REPORT_PALETTE["muted"]),
            linestyle=":" if name == total_name else _LINE_STYLES[index % len(_LINE_STYLES)],
            linewidth=1.45 if name == total_name else 1.05,
        )
    ax.set_yscale("log")
    ax.set_xlabel("Simulation time [days]")
    ax.set_ylabel("Acceleration magnitude [m/s^2]")
    ax.set_title("Dominant non-central time histories", loc="left", fontsize=9.2, fontweight="bold")
    _style_axes(ax)
    ax.legend(frameon=False, fontsize=5.9, ncol=2, loc="best")


def _plot_total_ric(ax: plt.Axes, result: OrbitAnalysisResult) -> None:
    name = "Total non-central acceleration"
    ric = result.force_ric_m_s2.get(name)
    if result.force_time_s is None or ric is None:
        ax.axis("off")
        ax.text(0.5, 0.56, "Signed RIC acceleration unavailable", ha="center", fontsize=9.0)
        ax.text(
            0.5,
            0.43,
            "The runtime force hook must expose acceleration vectors.",
            ha="center",
            fontsize=6.6,
            color=REPORT_PALETTE["muted"],
        )
        return
    time_days = np.asarray(result.force_time_s, dtype=np.float64) / 86_400.0
    values = np.asarray(ric, dtype=np.float64)
    specs = (
        (0, "R - radial", "ric_radial", "-"),
        (1, "I - in-track", "ric_intrack", "--"),
        (2, "C - cross-track", "ric_crosstrack", "-."),
    )
    for axis, label, color_key, style in specs:
        ax.plot(
            time_days,
            values[:, axis],
            label=label,
            color=REPORT_SERIES_COLORS[color_key],
            linestyle=style,
            linewidth=1.15,
        )
    ax.axhline(0.0, color=REPORT_PALETTE["rule"], linewidth=0.75)
    ax.set_xlabel("Simulation time [days]")
    ax.set_ylabel("Signed acceleration [m/s^2]")
    ax.set_title("Total non-central acceleration in instantaneous RIC", loc="left", fontsize=9.2, fontweight="bold")
    _style_axes(ax)
    ax.legend(frameon=False, fontsize=6.3, ncol=3, loc="best")


def _plot_orbit_3d(ax: plt.Axes, result: OrbitAnalysisResult, preset: ReportPreset) -> None:
    idx = _plot_indices(result.series.t_s.size, preset.max_plot_points)
    xyz = result.series.state_m_mps[idx, :3] / 1000.0
    radius_km = float(np.nanmedian(result.series.radius_m - result.series.altitude_m)) / 1000.0
    u = np.linspace(0.0, 2.0 * np.pi, 36)
    v = np.linspace(0.0, np.pi, 18)
    ax.plot_surface(
        radius_km * np.outer(np.cos(u), np.sin(v)),
        radius_km * np.outer(np.sin(u), np.sin(v)),
        radius_km * np.outer(np.ones_like(u), np.cos(v)),
        color=REPORT_PALETTE["graphite_mid"],
        alpha=0.16,
        linewidth=0.0,
        shade=True,
    )
    ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], color=REPORT_SERIES_COLORS["altitude"], linewidth=1.35)
    ax.scatter(*xyz[0], color=REPORT_STATUS_COLORS["ok"], s=24, marker="o", label="Start")
    ax.scatter(*xyz[-1], color=REPORT_PALETTE["indigo"], s=25, marker="s", label="End")
    ax.set_xlabel("X [km]", labelpad=2)
    ax.set_ylabel("Y [km]", labelpad=2)
    ax.set_zlabel("Z [km]", labelpad=2)
    ax.set_title("Moon-centered inertial trajectory", loc="left", fontsize=9.2, fontweight="bold")
    ax.tick_params(labelsize=5.8, pad=1)
    ax.legend(frameon=False, fontsize=6.0, loc="upper right")
    ax.grid(True, color=REPORT_PALETTE["grid"], linewidth=0.45)


def _plot_projection(ax: plt.Axes, result: OrbitAnalysisResult, preset: ReportPreset) -> None:
    idx = _plot_indices(result.series.t_s.size, preset.max_plot_points)
    xy = result.series.state_m_mps[idx, :2] / 1000.0
    radius_km = float(np.nanmedian(result.series.radius_m - result.series.altitude_m)) / 1000.0
    theta = np.linspace(0.0, 2.0 * np.pi, 180)
    ax.fill(radius_km * np.cos(theta), radius_km * np.sin(theta), color=REPORT_PALETTE["graphite"], alpha=0.13)
    ax.plot(xy[:, 0], xy[:, 1], color=REPORT_SERIES_COLORS["altitude"], linewidth=1.1)
    ax.scatter(*xy[0], color=REPORT_STATUS_COLORS["ok"], s=18, zorder=4)
    ax.scatter(*xy[-1], color=REPORT_PALETTE["indigo"], marker="s", s=18, zorder=4)
    ax.set_xlabel("Inertial X [km]")
    ax.set_ylabel("Inertial Y [km]")
    ax.set_title("Inertial XY projection", loc="left", fontsize=8.8, fontweight="bold")
    ax.set_aspect("equal", adjustable="datalim")
    _style_axes(ax)


def _plot_groundtrack_or_status(ax: plt.Axes, result: OrbitAnalysisResult) -> None:
    lat_values = result.series.latitude_rad
    lon_values = result.series.longitude_rad
    if lat_values is None or lon_values is None or not np.asarray(lat_values).size:
        ax.axis("off")
        ax.add_patch(
            plt.Rectangle(
                (0.0, 0.12),
                1.0,
                0.76,
                facecolor=REPORT_PALETTE["paper_alt"],
                edgecolor=REPORT_PALETTE["rule"],
                linewidth=0.7,
            )
        )
        ax.text(0.05, 0.70, "BODY-FIXED COVERAGE", fontsize=6.3, fontweight="bold", color=REPORT_PALETTE["muted"])
        ax.text(0.05, 0.54, "Unavailable", fontsize=11.0, fontweight="bold", color=REPORT_STATUS_COLORS["unavailable"])
        ax.text(
            0.05,
            0.39,
            "No Moon-fixed position history was persisted.\nGround track is not inferred from inertial longitude.",
            fontsize=6.5,
            color=REPORT_PALETTE["graphite_mid"],
            va="top",
            linespacing=1.35,
        )
        return
    lon = np.degrees(np.asarray(lon_values, dtype=np.float64))
    lat = np.degrees(np.asarray(lat_values, dtype=np.float64))
    breaks = np.abs(np.diff(lon)) > 180.0
    start = 0
    for stop in [*(np.flatnonzero(breaks) + 1), lon.size]:
        ax.plot(lon[start:stop], lat[start:stop], color=REPORT_SERIES_COLORS["altitude"], linewidth=1.0)
        start = int(stop)
    ax.scatter(lon[0], lat[0], color=REPORT_STATUS_COLORS["ok"], s=18, zorder=4)
    ax.scatter(lon[-1], lat[-1], color=REPORT_PALETTE["indigo"], marker="s", s=18, zorder=4)
    ax.set_xlim(-180.0, 180.0)
    ax.set_ylim(-90.0, 90.0)
    ax.set_xticks(np.arange(-180.0, 181.0, 60.0))
    ax.set_yticks(np.arange(-90.0, 91.0, 30.0))
    ax.set_xlabel("Body-fixed longitude [deg]")
    ax.set_ylabel("Body-fixed latitude [deg]")
    ax.set_title("Lunar ground track", loc="left", fontsize=8.8, fontweight="bold")
    _style_axes(ax)


def _page_chrome(
    fig: plt.Figure,
    result: OrbitAnalysisResult,
    *,
    section: str,
    page_number: int,
    page_count: int,
    cover: bool = False,
) -> None:
    if cover:
        return
    commit = str(result.provenance.get("git", {}).get("commit") or "unavailable")[:10]
    chrome = fig.add_axes([0.0, 0.0, 1.0, 1.0], zorder=1_000, frameon=False)
    chrome.set_axis_off()
    chrome.patch.set_alpha(0.0)
    chrome.text(
        REPORT_LAYOUT["margin_left"],
        0.965,
        "LUNARIS / MISSION ANALYSIS",
        color=REPORT_PALETTE["muted"],
        fontsize=6.8,
        fontweight="bold",
        va="top",
        transform=chrome.transAxes,
    )
    chrome.text(
        REPORT_LAYOUT["margin_right"],
        0.965,
        section.upper(),
        color=REPORT_PALETTE["muted"],
        fontsize=6.8,
        ha="right",
        va="top",
        transform=chrome.transAxes,
    )
    chrome.plot(
        [REPORT_LAYOUT["margin_left"], REPORT_LAYOUT["margin_right"]],
        [0.948, 0.948],
        transform=chrome.transAxes,
        color=REPORT_PALETTE["rule"],
        linewidth=0.7,
    )
    chrome.plot(
        [REPORT_LAYOUT["margin_left"], REPORT_LAYOUT["margin_right"]],
        [0.052, 0.052],
        transform=chrome.transAxes,
        color=REPORT_PALETTE["rule"],
        linewidth=0.7,
    )
    chrome.text(
        REPORT_LAYOUT["margin_left"],
        0.032,
        f"{result.run_id}  /  commit {commit}",
        color=REPORT_PALETTE["muted"],
        fontsize=6.4,
        family="monospace",
        transform=chrome.transAxes,
    )
    chrome.text(
        REPORT_LAYOUT["margin_right"],
        0.032,
        f"{page_number} / {page_count}",
        color=REPORT_PALETTE["graphite_mid"],
        fontsize=6.8,
        ha="right",
        transform=chrome.transAxes,
    )


def _title(fig: plt.Figure, number: str, heading: str, subtitle: str) -> None:
    fig.text(
        REPORT_LAYOUT["margin_left"],
        0.91,
        number,
        color=REPORT_PALETTE["lunar_blue"],
        fontsize=REPORT_TYPOGRAPHY["subsection_title"],
        fontweight="bold",
        va="top",
    )
    fig.text(
        REPORT_LAYOUT["margin_left"] + 0.055,
        0.91,
        heading,
        color=REPORT_PALETTE["graphite"],
        fontsize=REPORT_TYPOGRAPHY["section_title"],
        fontweight="bold",
        va="top",
    )
    fig.text(
        REPORT_LAYOUT["margin_left"] + 0.055,
        0.882,
        subtitle,
        color=REPORT_PALETTE["muted"],
        fontsize=REPORT_TYPOGRAPHY["caption"],
        va="top",
    )


def _draw_table(
    ax: plt.Axes,
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    *,
    widths: Sequence[float] | None = None,
    font_size: float | None = None,
) -> None:
    ax.axis("off")
    if not rows:
        ax.text(
            0.0,
            0.9,
            "Unavailable for this run.",
            color=REPORT_PALETTE["muted"],
            fontsize=REPORT_TYPOGRAPHY["body"],
            transform=ax.transAxes,
        )
        return
    table = ax.table(
        cellText=rows,
        colLabels=headers,
        colWidths=widths,
        cellLoc="left",
        loc="upper left",
        bbox=[0.0, 0.0, 1.0, 1.0],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(font_size or REPORT_TYPOGRAPHY["table_header"])
    for (row, _col), cell in table.get_celld().items():
        cell.visible_edges = "horizontal"
        cell.set_edgecolor(REPORT_PALETTE["rule"])
        cell.set_linewidth(0.65 if row == 0 else 0.35)
        cell.PAD = 0.06
        if row == 0:
            cell.set_facecolor(REPORT_PALETTE["paper_alt"])
            cell.set_text_props(color=REPORT_PALETTE["graphite"], weight="bold")
        else:
            cell.set_facecolor("none")
            cell.set_text_props(color=REPORT_PALETTE["graphite_mid"])


def _metric_strip(
    fig: plt.Figure,
    result: OrbitAnalysisResult,
    specs: Sequence[tuple[str, str]],
    *,
    rect: tuple[float, float, float, float],
) -> None:
    ax = fig.add_axes(rect)
    ax.axis("off")
    count = max(1, len(specs))
    for index, (metric_id, label) in enumerate(specs):
        metric = _metric(result, metric_id)
        x0 = index / count
        if index:
            ax.plot(
                [x0, x0],
                [0.08, 0.92],
                color=REPORT_PALETTE["rule"],
                linewidth=0.7,
                transform=ax.transAxes,
            )
        ax.text(
            x0 + 0.025,
            0.72,
            label.upper(),
            color=REPORT_PALETTE["muted"],
            fontsize=6.5,
            fontweight="bold",
            transform=ax.transAxes,
        )
        ax.text(
            x0 + 0.025,
            0.23,
            _display(metric, compact=True),
            color=REPORT_PALETTE["graphite"],
            fontsize=REPORT_TYPOGRAPHY["metric_headline"],
            fontweight="bold",
            transform=ax.transAxes,
        )


def figure_orbit_overview(result: OrbitAnalysisResult, preset: ReportPreset) -> plt.Figure:
    fig = plt.figure(figsize=(9.6, 7.2), facecolor=REPORT_PALETTE["paper"])
    ax = fig.add_subplot(111, projection="3d")
    idx = _plot_indices(result.series.t_s.size, preset.max_plot_points)
    xyz = result.series.state_m_mps[idx, :3] / 1000.0
    radius_km = float(np.median(result.series.radius_m - result.series.altitude_m)) / 1000.0
    u = np.linspace(0.0, 2.0 * np.pi, 48)
    v = np.linspace(0.0, np.pi, 24)
    sphere_x = radius_km * np.outer(np.cos(u), np.sin(v))
    sphere_y = radius_km * np.outer(np.sin(u), np.sin(v))
    sphere_z = radius_km * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(
        sphere_x,
        sphere_y,
        sphere_z,
        color=REPORT_PALETTE["graphite_mid"],
        alpha=0.18,
        linewidth=0.0,
        shade=True,
    )
    ax.plot(
        xyz[:, 0],
        xyz[:, 1],
        xyz[:, 2],
        color=REPORT_SERIES_COLORS["altitude"],
        linewidth=1.25,
    )
    ax.scatter(*xyz[0], color=REPORT_STATUS_COLORS["ok"], s=26, marker="o", label="Start")
    ax.scatter(*xyz[-1], color=REPORT_PALETTE["indigo"], s=28, marker="s", label="End")
    ax.set_xlabel("X [km]")
    ax.set_ylabel("Y [km]")
    ax.set_zlabel("Z [km]")
    ax.set_title("Moon-centered inertial trajectory", loc="left", fontsize=12, fontweight="bold")
    ax.set_facecolor(REPORT_PALETTE["white"])
    fig.patch.set_facecolor(REPORT_PALETTE["paper"])
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    ax.grid(True, color=REPORT_PALETTE["grid"], linewidth=0.5)
    fig.tight_layout(pad=1.4)
    return fig


def figure_altitude_history(result: OrbitAnalysisResult, preset: ReportPreset) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10.5, 4.8), facecolor=REPORT_PALETTE["paper"])
    idx = _plot_indices(result.series.t_s.size, preset.max_plot_points)
    t_days = result.series.t_s[idx] / 86_400.0
    altitude_km = result.series.altitude_m[idx] / 1000.0
    ax.plot(t_days, altitude_km, color=REPORT_SERIES_COLORS["altitude"], linewidth=1.45)
    for event_type, marker, color in (
        ("minimum_altitude", "v", REPORT_STATUS_COLORS["warning"]),
        ("maximum_altitude", "^", REPORT_PALETTE["indigo"]),
        ("impact", "X", REPORT_STATUS_COLORS["critical"]),
    ):
        for event in result.events:
            if event.event_type == event_type and event.altitude_m is not None:
                ax.scatter(
                    event.simulation_time_s / 86_400.0,
                    event.altitude_m / 1000.0,
                    marker=marker,
                    color=color,
                    s=44,
                    zorder=5,
                    label=event_type.replace("_", " ").title(),
                )
    ax.set_xlabel("Simulation time [days]")
    ax.set_ylabel("Altitude [km]")
    ax.set_title("Altitude history and critical extrema", loc="left", fontsize=12, fontweight="bold")
    _style_axes(ax)
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        unique = dict(zip(labels, handles, strict=True))
        ax.legend(unique.values(), unique.keys(), frameon=False, fontsize=7.5, ncol=3)
    fig.tight_layout(pad=1.3)
    return fig


def figure_orbital_elements(result: OrbitAnalysisResult, preset: ReportPreset) -> plt.Figure:
    fig, axes = plt.subplots(3, 2, figsize=(10.5, 9.2), facecolor=REPORT_PALETTE["paper"], sharex=True)
    idx = _plot_indices(result.series.t_s.size, preset.max_plot_points)
    t_days = result.series.t_s[idx] / 86_400.0
    eccentricity = result.series.eccentricity[idx]
    inclination = result.series.inclination_rad[idx]
    raan = np.degrees(result.series.raan_rad[idx]).astype(float)
    argp = np.degrees(result.series.argument_of_periapsis_rad[idx]).astype(float)
    raan[np.abs(np.sin(inclination)) < 1.0e-8] = np.nan
    argp[eccentricity < 1.0e-8] = np.nan
    payloads = (
        (result.series.semi_major_axis_m[idx] / 1000.0, "Semi-major axis [km]", "semi_major_axis"),
        (eccentricity, "Eccentricity [-]", "eccentricity"),
        (np.degrees(inclination), "Inclination [deg]", "inclination"),
        (raan, "RAAN [deg]", "raan"),
        (argp, "Argument of periapsis [deg]", "argument_of_periapsis"),
        (np.degrees(result.series.true_anomaly_rad[idx]) % 360.0, "True anomaly [deg]", "true_anomaly"),
    )
    for plot_index, (ax, (values, label, color_key)) in enumerate(zip(axes.ravel(), payloads, strict=True)):
        ax.plot(
            t_days,
            values,
            color=REPORT_SERIES_COLORS[color_key],
            linestyle=_LINE_STYLES[plot_index % len(_LINE_STYLES)],
            linewidth=1.25,
        )
        ax.set_ylabel(label)
        _style_axes(ax)
    axes[-1, 0].set_xlabel("Simulation time [days]")
    axes[-1, 1].set_xlabel("Simulation time [days]")
    fig.suptitle(
        "Osculating orbital elements",
        x=0.08,
        y=0.985,
        ha="left",
        fontsize=12,
        fontweight="bold",
        color=REPORT_PALETTE["graphite"],
    )
    fig.text(
        0.08,
        0.075,
        "Gaps in RAAN or argument of periapsis are intentional singularity markers, not zeros.",
        fontsize=7.2,
        color=REPORT_PALETTE["muted"],
    )
    fig.tight_layout(rect=(0.03, 0.035, 0.98, 0.96), pad=1.2)
    return fig


def figure_numerical_health(result: OrbitAnalysisResult, preset: ReportPreset) -> plt.Figure:
    del preset
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), facecolor=REPORT_PALETTE["paper"])
    t_days = result.series.t_s / 86_400.0
    energy = result.series.specific_energy_j_kg
    angmom = result.series.angular_momentum_m2_s
    rel_energy = (energy - energy[0]) / max(abs(float(energy[0])), 1.0e-30)
    rel_h = (angmom - angmom[0]) / max(abs(float(angmom[0])), 1.0e-30)
    axes[0].plot(t_days, rel_energy, color=REPORT_SERIES_COLORS["energy"], linewidth=1.2, label="Energy")
    axes[0].plot(
        t_days,
        rel_h,
        color=REPORT_SERIES_COLORS["angular_momentum"],
        linewidth=1.2,
        linestyle="--",
        label="|h|",
    )
    axes[0].set_xlabel("Simulation time [days]")
    axes[0].set_ylabel("Relative change [-]")
    axes[0].set_title("Physical diagnostic drift", loc="left", fontsize=10, fontweight="bold")
    axes[0].legend(frameon=False, fontsize=7.5)
    _style_axes(axes[0])

    dt = np.diff(result.series.t_s)
    if dt.size:
        axes[1].hist(
            dt,
            bins=min(30, max(5, int(np.sqrt(dt.size)))),
            color=REPORT_PALETTE["lunar_cyan"],
            alpha=0.82,
            edgecolor=REPORT_PALETTE["white"],
        )
        axes[1].set_xlabel("Output cadence [s]")
        axes[1].set_ylabel("Output intervals [count]")
    else:
        axes[1].text(0.5, 0.5, "Output cadence unavailable", ha="center", va="center")
    axes[1].set_title("Recorded output cadence", loc="left", fontsize=10, fontweight="bold")
    _style_axes(axes[1])
    fig.tight_layout(pad=1.3)
    return fig


def figure_force_budget(result: OrbitAnalysisResult, preset: ReportPreset) -> plt.Figure:
    del preset
    fig = plt.figure(figsize=(10.5, 6.2), facecolor=REPORT_PALETTE["paper"])
    ax_range = fig.add_axes([0.08, 0.14, 0.39, 0.76])
    _plot_force_range(ax_range, result, limit=8)
    ax_history = fig.add_axes([0.55, 0.14, 0.40, 0.76])
    _plot_force_history(ax_history, result, limit=6)
    fig.text(
        0.08,
        0.035,
        "Range: minimum to maximum; circle: median; diamond: P95. Aggregate rows and central gravity are excluded from the unique-force ranking.",
        fontsize=7.0,
        color=REPORT_PALETTE["muted"],
    )
    return fig


def figure_force_ric(result: OrbitAnalysisResult, preset: ReportPreset) -> plt.Figure:
    del preset
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.0), facecolor=REPORT_PALETTE["paper"])
    _plot_force_history(axes[0], result, limit=6)
    _plot_total_ric(axes[1], result)
    fig.tight_layout(rect=(0.04, 0.05, 0.98, 0.98), h_pad=2.2)
    fig.text(
        0.07,
        0.015,
        "RIC is evaluated in the instantaneous orbital frame. The total is a vector sum of unique non-central components, not a sum of magnitudes.",
        fontsize=7.0,
        color=REPORT_PALETTE["muted"],
    )
    return fig


def figure_orbit_envelope(result: OrbitAnalysisResult, preset: ReportPreset) -> plt.Figure:
    fig = plt.figure(figsize=(10.5, 6.7), facecolor=REPORT_PALETTE["paper"])
    idx = _plot_indices(result.series.t_s.size, preset.max_plot_points)
    time_days = result.series.t_s[idx] / 86_400.0
    ax_alt = fig.add_axes([0.08, 0.57, 0.87, 0.35])
    ax_alt.plot(time_days, result.series.altitude_m[idx] / 1000.0, color=REPORT_SERIES_COLORS["altitude"], linewidth=1.3)
    ax_alt.set_xlabel("Simulation time [days]")
    ax_alt.set_ylabel("Observed altitude [km]")
    ax_alt.set_title("Observed altitude history", loc="left", fontsize=10.0, fontweight="bold")
    _style_axes(ax_alt)
    ax_bounds = fig.add_axes([0.08, 0.12, 0.41, 0.32])
    _plot_osculating_envelope(ax_bounds, result, idx)
    ax_period = fig.add_axes([0.56, 0.12, 0.39, 0.32])
    _plot_period_history(ax_period, result, idx)
    return fig


def figure_spatial_context(result: OrbitAnalysisResult, preset: ReportPreset) -> plt.Figure:
    fig = plt.figure(figsize=(10.5, 5.8), facecolor=REPORT_PALETTE["paper"])
    ax_3d = fig.add_subplot(1, 2, 1, projection="3d")
    _plot_orbit_3d(ax_3d, result, preset)
    ax_ground = fig.add_subplot(1, 2, 2)
    _plot_groundtrack_or_status(ax_ground, result)
    fig.tight_layout(pad=1.2, w_pad=2.0)
    return fig


def _force_color_key(name: str) -> str:
    normalized = name.lower()
    if "total non-central" in normalized:
        return "total_noncentral"
    if "gravity (pm)" in normalized:
        return "central_gravity"
    if "gravity" in normalized and ("sh" in normalized or "st-lrps" in normalized):
        return "spherical_harmonics"
    if "sun" in normalized and "3rd" in normalized:
        return "third_body_sun"
    if "earth" in normalized and "3rd" in normalized:
        return "third_body_earth"
    if "srp" in normalized:
        return "srp"
    if "albedo" in normalized:
        return "albedo"
    if "thermal" in normalized:
        return "thermal_ir"
    if "tide" in normalized:
        return "solid_tides"
    if "relativ" in normalized or "1pn" in normalized:
        return "relativity"
    return "central_gravity"


def figure_event_timeline(result: OrbitAnalysisResult, preset: ReportPreset) -> plt.Figure:
    del preset
    fig, ax = plt.subplots(figsize=(10.5, 4.8), facecolor=REPORT_PALETTE["paper"])
    event_types = list(dict.fromkeys(event.event_type for event in result.events))
    positions = {name: index for index, name in enumerate(event_types)}
    for event in result.events:
        ax.scatter(
            event.simulation_time_s / 86_400.0,
            positions[event.event_type],
            s=32 if event.severity == "normal" else 48,
            marker="X" if event.severity == "critical" else ("^" if event.severity == "warning" else "o"),
            color=REPORT_STATUS_COLORS.get(event.severity, REPORT_PALETTE["lunar_blue"]),
            edgecolor=REPORT_PALETTE["white"],
            linewidth=0.5,
            zorder=4,
        )
    ax.set_yticks(range(len(event_types)), [name.replace("_", " ").title() for name in event_types])
    ax.set_xlabel("Simulation time [days]")
    ax.set_title("Chronological event timeline", loc="left", fontsize=12, fontweight="bold")
    _style_axes(ax)
    fig.tight_layout(pad=1.3)
    return fig


def figure_groundtrack(result: OrbitAnalysisResult, preset: ReportPreset) -> plt.Figure:
    del preset
    fig, ax = plt.subplots(figsize=(10.5, 5.2), facecolor=REPORT_PALETTE["paper"])
    if result.series.latitude_rad is None or result.series.longitude_rad is None:
        ax.axis("off")
        ax.text(
            0.5,
            0.55,
            "Ground track unavailable",
            ha="center",
            fontsize=13,
            fontweight="bold",
            color=REPORT_PALETTE["graphite"],
        )
        ax.text(
            0.5,
            0.44,
            "A body-fixed attitude/ephemeris table was not available for this run.",
            ha="center",
            fontsize=8.5,
            color=REPORT_PALETTE["muted"],
        )
        return fig
    lon = np.degrees(result.series.longitude_rad)
    lat = np.degrees(result.series.latitude_rad)
    breaks = np.abs(np.diff(lon)) > 180.0
    start = 0
    for stop in [*(np.flatnonzero(breaks) + 1), lon.size]:
        ax.plot(lon[start:stop], lat[start:stop], color=REPORT_SERIES_COLORS["altitude"], linewidth=0.9)
        start = int(stop)
    ax.set_xlim(-180.0, 180.0)
    ax.set_ylim(-90.0, 90.0)
    ax.set_xticks(np.arange(-180.0, 181.0, 60.0))
    ax.set_yticks(np.arange(-90.0, 91.0, 30.0))
    ax.set_xlabel("Body-fixed longitude [deg]")
    ax.set_ylabel("Body-fixed latitude [deg]")
    ax.set_title("Lunar ground track", loc="left", fontsize=12, fontweight="bold")
    _style_axes(ax)
    fig.tight_layout(pad=1.3)
    return fig


def _cover_page(result: OrbitAnalysisResult, preset: ReportPreset) -> plt.Figure:
    fig = _figure()
    ax = fig.add_axes([0.53, 0.50, 0.38, 0.36])
    ax.axis("off")
    idx = _plot_indices(result.series.t_s.size, min(preset.max_plot_points, 1800))
    xy = result.series.state_m_mps[idx, :2] / 1000.0
    body_radius = float(np.median(result.series.radius_m - result.series.altitude_m)) / 1000.0
    theta = np.linspace(0.0, 2.0 * np.pi, 180)
    ax.fill(
        body_radius * np.cos(theta),
        body_radius * np.sin(theta),
        color=REPORT_PALETTE["graphite"],
        alpha=0.14,
        linewidth=0.0,
    )
    ax.plot(xy[:, 0], xy[:, 1], color=REPORT_PALETTE["lunar_blue"], linewidth=1.3)
    ax.scatter(*xy[0], color=REPORT_STATUS_COLORS["ok"], s=20)
    ax.scatter(*xy[-1], color=REPORT_PALETTE["indigo"], s=22, marker="s")
    ax.set_aspect("equal", adjustable="datalim")

    fig.text(0.08, 0.89, "LUNARIS", fontsize=10, fontweight="bold", color=REPORT_PALETTE["lunar_blue"])
    fig.text(
        0.08,
        0.80,
        "Mission Analysis\nReport",
        fontsize=REPORT_TYPOGRAPHY["cover_title"],
        fontweight="bold",
        color=REPORT_PALETTE["graphite"],
        linespacing=1.05,
    )
    fig.text(0.08, 0.68, result.run_id, fontsize=10, family="monospace", color=REPORT_PALETTE["graphite_mid"])
    status = _metric(result, "run.status")
    status_text = _display(status).upper()
    status_color = REPORT_STATUS_COLORS.get(status.status if status else "unavailable", REPORT_PALETTE["muted"])
    fig.text(0.08, 0.625, status_text, fontsize=7.4, fontweight="bold", color=status_color)
    fig.add_artist(
        plt.Line2D([0.08, 0.45], [0.606, 0.606], transform=fig.transFigure, color=status_color, linewidth=2.0)
    )
    cover_rows = (
        ("Epoch", _display(_metric(result, "mission.start_epoch"))),
        ("Duration", _display(_metric(result, "mission.duration"))),
        ("Gravity", _display(_metric(result, "physics.gravity_backend.requested"))),
        ("SH degree", _display(_metric(result, "physics.gravity_degree"))),
        ("Generated", result.generated_at_utc),
        ("Preset", result.preset.title()),
    )
    y = 0.54
    for label, value in cover_rows:
        fig.text(0.08, y, label.upper(), fontsize=6.4, fontweight="bold", color=REPORT_PALETTE["muted"])
        fig.text(0.205, y, value, fontsize=8.1, color=REPORT_PALETTE["graphite_mid"])
        y -= 0.046
    git = result.provenance.get("git", {})
    commit = str(git.get("commit") or "unavailable")[:12]
    dirty = "dirty" if git.get("dirty") else "clean"
    fig.text(0.08, 0.105, f"commit {commit} / {dirty}", family="monospace", fontsize=6.7, color=REPORT_PALETTE["muted"])
    fig.text(0.92, 0.105, "RESEARCH / BETA", ha="right", fontsize=6.7, fontweight="bold", color=REPORT_PALETTE["indigo"])
    return fig


def _executive_page(result: OrbitAnalysisResult, preset: ReportPreset) -> plt.Figure:
    del preset
    fig = _figure()
    _title(fig, "01", "Executive Mission Summary", "Outcome, orbit envelope, active physics, and reliability at a glance.")
    status = _metric(result, "run.status")
    status_key = status.status if status else "unavailable"
    status_color = REPORT_STATUS_COLORS.get(status_key, REPORT_PALETTE["muted"])
    ax_banner = fig.add_axes([0.075, 0.79, 0.85, 0.065])
    ax_banner.axis("off")
    ax_banner.add_patch(
        plt.Rectangle((0, 0), 1, 1, facecolor=REPORT_PALETTE["white"], edgecolor=REPORT_PALETTE["rule"], linewidth=0.8)
    )
    ax_banner.add_patch(plt.Rectangle((0, 0), 0.012, 1, facecolor=status_color, linewidth=0))
    ax_banner.text(0.035, 0.62, "MISSION STATUS", fontsize=6.5, fontweight="bold", color=REPORT_PALETTE["muted"])
    ax_banner.text(0.035, 0.22, _display(status).upper(), fontsize=14.5, fontweight="bold", color=status_color)
    ax_banner.text(
        0.98,
        0.5,
        _display(_metric(result, "numerical.termination_reason")),
        ha="right",
        va="center",
        fontsize=8.5,
        color=REPORT_PALETTE["graphite_mid"],
    )
    _metric_strip(
        fig,
        result,
        (
            ("orbit.altitude.minimum", "Minimum altitude"),
            ("orbit.altitude.maximum", "Maximum altitude"),
            ("orbit.period", "Orbit period"),
            ("orbit.completed_count", "Orbit count"),
        ),
        rect=(0.075, 0.68, 0.85, 0.085),
    )

    ax_alt = fig.add_axes([0.075, 0.40, 0.53, 0.23])
    idx = _plot_indices(result.series.t_s.size, 2400)
    ax_alt.plot(
        result.series.t_s[idx] / 86_400.0,
        result.series.altitude_m[idx] / 1000.0,
        color=REPORT_SERIES_COLORS["altitude"],
        linewidth=1.25,
    )
    ax_alt.set_xlabel("Simulation time [days]")
    ax_alt.set_ylabel("Altitude [km]")
    ax_alt.set_title("Orbit envelope", loc="left", fontsize=9.5, fontweight="bold")
    _style_axes(ax_alt)

    initial_final_rows = []
    for prefix, label in (
        ("orbit.altitude", "Altitude"),
        ("orbit.a", "Semi-major axis"),
        ("orbit.e", "Eccentricity"),
        ("orbit.i", "Inclination"),
    ):
        initial_final_rows.append(
            [label, _display(_metric(result, f"{prefix}.initial")), _display(_metric(result, f"{prefix}.final"))]
        )
    ax_compare = fig.add_axes([0.64, 0.40, 0.285, 0.23])
    _draw_table(ax_compare, ["Orbit", "Initial", "Final"], initial_final_rows, widths=(0.35, 0.33, 0.32), font_size=7.0)

    ax_models = fig.add_axes([0.075, 0.185, 0.53, 0.17])
    ax_models.axis("off")
    ax_models.text(0, 1.06, "FORCE MODEL STATUS", fontsize=7.2, fontweight="bold", color=REPORT_PALETTE["graphite"])
    _draw_force_status_matrix(ax_models, result, columns=2)
    health_rows = [
        ["Integrator", _display(_metric(result, "numerical.integrator"))],
        ["Backend", _display(_metric(result, "numerical.integration_backend"))],
        ["RHS path", _display(_metric(result, "physics.rhs_path.effective"))],
        ["Runtime", _display(_metric(result, "numerical.wall_time"))],
    ]
    ax_health = fig.add_axes([0.64, 0.185, 0.285, 0.17])
    _draw_table(ax_health, ["Numerical health", "Value"], health_rows, widths=(0.42, 0.58), font_size=7.0)

    ax_warning = fig.add_axes([0.075, 0.075, 0.85, 0.075])
    ax_warning.axis("off")
    warnings = list(result.warnings[:3])
    if warnings:
        ax_warning.add_patch(
            plt.Rectangle(
                (0, 0),
                1,
                1,
                facecolor=REPORT_PALETTE["warning_bg"],
                edgecolor=REPORT_PALETTE["amber"],
                linewidth=0.7,
            )
        )
        ax_warning.text(0.02, 0.72, "WARNINGS / LIMITATIONS", fontsize=6.6, fontweight="bold", color=REPORT_PALETTE["amber"])
        ax_warning.text(
            0.02,
            0.46,
            "\n".join(f"- {textwrap.shorten(item, 130)}" for item in warnings),
            fontsize=6.9,
            color=REPORT_PALETTE["graphite_mid"],
            va="top",
            linespacing=1.35,
        )
    else:
        ax_warning.text(0.0, 0.5, "No analysis warnings recorded.", color=REPORT_STATUS_COLORS["ok"], fontsize=7.5)
    return fig


def _configuration_page(result: OrbitAnalysisResult, preset: ReportPreset) -> plt.Figure:
    del preset
    fig = _figure()
    _title(fig, "02", "Mission Configuration", "Requested settings and effective runtime configuration remain distinct.")
    flattened = dict(_flatten_mapping(result.config_snapshot))
    selected_fields = (
        "effective_initial_state.source", "effective_initial_state.frame",
        "effective_initial_state.x_m", "effective_initial_state.y_m", "effective_initial_state.z_m",
        "effective_initial_state.vx_m_s", "effective_initial_state.vy_m_s", "effective_initial_state.vz_m_s",
        "time.start_date", "time.duration_s", "time.output_dt_s", "spice.inertial_frame", "spice.fixed_frame",
        "spacecraft.mass_kg", "spacecraft.area_m2", "spacecraft.cd", "spacecraft.cr",
        "propagator.method", "propagator.rtol", "propagator.atol", "propagator.atol_pos", "propagator.atol_vel",
        "propagator.user_max_step_s", "propagator.use_nyquist_max_step", "propagator.checkpoint_path",
        "gravity.backend", "gravity.file_path", "gravity.degree", "gravity.adaptive.enabled", "gravity.adaptive.min_degree",
        "flags.enable_sh", "flags.enable_3rd_body_sun", "flags.enable_3rd_body_earth", "flags.enable_earth_j2",
        "flags.enable_srp", "flags.enable_albedo", "flags.enable_thermal", "flags.enable_tides_k2",
        "flags.enable_tides_k3", "flags.enable_relativity_1pn",
    )
    rows = [
        [path, _short_value(flattened[path])]
        for path in selected_fields
        if path in flattened
    ]
    mid = (len(rows) + 1) // 2
    ax_left = fig.add_axes([0.075, 0.11, 0.405, 0.72])
    ax_right = fig.add_axes([0.52, 0.11, 0.405, 0.72])
    _draw_table(ax_left, ["Configuration field", "Value"], rows[:mid], widths=(0.55, 0.45), font_size=6.6)
    _draw_table(ax_right, ["Configuration field", "Value"], rows[mid:], widths=(0.55, 0.45), font_size=6.6)
    return fig


def _flatten_mapping(value: Any, parent: str = "") -> Iterable[tuple[str, Any]]:
    if isinstance(value, Mapping):
        for key, item in value.items():
            path = f"{parent}.{key}" if parent else str(key)
            yield from _flatten_mapping(item, path)
    else:
        yield parent, value


def _short_value(value: Any, limit: int = 45) -> str:
    if isinstance(value, list | tuple):
        text = ", ".join(str(item) for item in value)
    else:
        text = str(value)
    return textwrap.shorten(text, width=limit, placeholder="...")


def _orbit_page(result: OrbitAnalysisResult, preset: ReportPreset) -> plt.Figure:
    fig = _figure()
    _title(fig, "03", "Orbit Geometry and Evolution", "Full-resolution extrema with presentation-only downsampling.")
    ax = fig.add_axes([0.075, 0.52, 0.85, 0.29])
    idx = _plot_indices(result.series.t_s.size, preset.max_plot_points)
    t_days = result.series.t_s[idx] / 86_400.0
    ax.plot(t_days, result.series.altitude_m[idx] / 1000.0, color=REPORT_SERIES_COLORS["altitude"], linewidth=1.35)
    for event in result.events:
        if event.event_type in {"minimum_altitude", "maximum_altitude", "impact"} and event.altitude_m is not None:
            ax.scatter(
                event.simulation_time_s / 86_400.0,
                event.altitude_m / 1000.0,
                color=(REPORT_STATUS_COLORS["critical"] if event.event_type == "impact" else REPORT_PALETTE["indigo"]),
                marker="X" if event.event_type == "impact" else "o",
                s=38,
                zorder=5,
            )
    ax.set_xlabel("Simulation time [days]")
    ax.set_ylabel("Altitude [km]")
    ax.set_title("Altitude envelope and critical extrema", loc="left", fontsize=10.5, fontweight="bold")
    _style_axes(ax)
    ax_bounds = fig.add_axes([0.075, 0.255, 0.40, 0.185])
    _plot_osculating_envelope(ax_bounds, result, idx)
    ax_period = fig.add_axes([0.525, 0.255, 0.40, 0.185])
    _plot_period_history(ax_period, result, idx)
    fig.text(
        0.075,
        0.195,
        "Osculating bounds describe the instantaneous two-body ellipse; they are not observed local extrema.",
        fontsize=6.4,
        color=REPORT_PALETTE["muted"],
    )
    _metric_strip(
        fig,
        result,
        (
            ("orbit.altitude.minimum", "Observed minimum"),
            ("orbit.altitude.maximum", "Observed maximum"),
            ("orbit.period", "Reference period"),
            ("orbit.completed_count", "Completed orbits"),
        ),
        rect=(0.075, 0.09, 0.85, 0.085),
    )
    return fig


def _spatial_page(result: OrbitAnalysisResult, preset: ReportPreset) -> plt.Figure:
    fig = _figure()
    _title(
        fig,
        "04",
        "Spatial Context and Coverage",
        "Inertial geometry is separated from body-fixed surface coverage.",
    )
    ax_3d = fig.add_axes([0.075, 0.46, 0.50, 0.36], projection="3d")
    _plot_orbit_3d(ax_3d, result, preset)
    ax_xy = fig.add_axes([0.63, 0.52, 0.295, 0.27])
    _plot_projection(ax_xy, result, preset)
    ax_ground = fig.add_axes([0.075, 0.13, 0.85, 0.25])
    _plot_groundtrack_or_status(ax_ground, result)
    fig.text(
        0.075,
        0.09,
        f"Trajectory frame: {result.frame}. A ground track is shown only when Moon-fixed positions are persisted.",
        fontsize=6.6,
        color=REPORT_PALETTE["muted"],
    )
    return fig


def _elements_page(result: OrbitAnalysisResult, preset: ReportPreset) -> plt.Figure:
    fig = figure_orbital_elements(result, preset)
    fig.set_size_inches(REPORT_LAYOUT["page_width_in"], REPORT_LAYOUT["page_height_in"])
    fig.subplots_adjust(left=0.11, right=0.94, bottom=0.12, top=0.88, hspace=0.30, wspace=0.28)
    fig.suptitle("05  Orbital Element Evolution", x=0.075, y=0.925, ha="left", fontsize=17, fontweight="bold")
    return fig


def _events_page(result: OrbitAnalysisResult, preset: ReportPreset) -> plt.Figure:
    del preset
    fig = _figure()
    _title(fig, "06", "Critical Extrema and Events", "Chronological state-linked events from full-resolution analysis.")
    ax_timeline = fig.add_axes([0.16, 0.58, 0.765, 0.22])
    types = list(dict.fromkeys(event.event_type for event in result.events))
    positions = {name: index for index, name in enumerate(types)}
    for event in result.events:
        ax_timeline.scatter(
            event.simulation_time_s / 86_400.0,
            positions[event.event_type],
            color=REPORT_STATUS_COLORS.get(event.severity, REPORT_PALETTE["lunar_blue"]),
            marker="X" if event.severity == "critical" else "o",
            s=28,
        )
    ax_timeline.set_yticks(range(len(types)), [name.replace("_", " ").title() for name in types])
    ax_timeline.set_xlabel("Simulation time [days]")
    _style_axes(ax_timeline)
    table_events = [
        event
        for event in result.events
        if event.event_type in {
            "minimum_altitude",
            "maximum_altitude",
            "impact",
            "terminal_event",
            "completed",
            "eclipse_enter",
            "eclipse_exit",
            "backend_fallback",
            "numerical_warning",
        }
    ][:14]
    rows = [
        [
            f"{event.simulation_time_s:.3f}",
            event.epoch_utc or "Unavailable",
            event.event_type.replace("_", " "),
            (
                "Unavailable"
                if event.state_m_mps is None
                else f"r {np.linalg.norm(event.state_m_mps[:3]) / 1000.0:.1f} km / "
                f"v {np.linalg.norm(event.state_m_mps[3:6]) / 1000.0:.4f} km/s"
            ),
            "Unavailable" if event.altitude_m is None else f"{event.altitude_m / 1000.0:.3f}",
            event.frame,
            event.source,
        ]
        for event in table_events
    ]
    ax_table = fig.add_axes([0.075, 0.11, 0.85, 0.40])
    _draw_table(
        ax_table,
        ["Time [s]", "Epoch [UTC]", "Event", "State summary", "Altitude [km]", "Frame", "Source"],
        rows,
        widths=(0.08, 0.18, 0.12, 0.20, 0.10, 0.08, 0.24),
        font_size=5.8,
    )
    return fig


def _numerical_page(result: OrbitAnalysisResult, preset: ReportPreset) -> plt.Figure:
    del preset
    fig = _figure()
    _title(fig, "07", "Numerical Health", "Integrator-reported values stay separate from output-grid diagnostics.")
    metric_ids = (
        "numerical.integrator",
        "numerical.integration_backend",
        "numerical.rhs_evaluations",
        "numerical.accepted_steps",
        "numerical.rejected_steps",
        "numerical.internal_step.minimum",
        "numerical.internal_step.median",
        "numerical.internal_step.maximum",
        "numerical.output_step.minimum",
        "numerical.output_step.median",
        "numerical.output_step.maximum",
        "numerical.tolerance.relative",
        "numerical.tolerance.absolute",
        "numerical.wall_time",
        "numerical.throughput",
        "numerical.finite_validation",
        "numerical.event_location_quality",
        "numerical.checkpoint",
        "numerical.warning_count",
        "numerical.termination_reason",
    )
    rows = []
    for metric_id in metric_ids:
        metric = _metric(result, metric_id)
        if metric:
            rows.append([metric.label, _short_value(_display(metric), 24), metric.status])
    midpoint = (len(rows) + 1) // 2
    for left, subset in ((0.075, rows[:midpoint]), (0.515, rows[midpoint:])):
        ax_table = fig.add_axes([left, 0.45, 0.41, 0.37])
        _draw_table(
            ax_table,
            ["Diagnostic", "Value", "Status"],
            subset,
            widths=(0.48, 0.32, 0.20),
            font_size=6.5,
        )
    fig.text(
        0.075,
        0.425,
        "Internal-step fields remain Unavailable when the integrator exposes only output-grid samples; output cadence is never substituted.",
        fontsize=6.5,
        color=REPORT_PALETTE["muted"],
    )
    ax_drift = fig.add_axes([0.075, 0.14, 0.52, 0.27])
    t_days = result.series.t_s / 86_400.0
    energy = result.series.specific_energy_j_kg
    h_norm = result.series.angular_momentum_m2_s
    rel_energy = (energy - energy[0]) / max(abs(float(energy[0])), 1.0e-30)
    rel_h = (h_norm - h_norm[0]) / max(abs(float(h_norm[0])), 1.0e-30)
    ax_drift.plot(t_days, rel_energy, color=REPORT_SERIES_COLORS["energy"], linewidth=1.15, label="Energy")
    ax_drift.plot(t_days, rel_h, color=REPORT_SERIES_COLORS["angular_momentum"], linewidth=1.15, linestyle="--", label="|h|")
    ax_drift.set_xlabel("Simulation time [days]")
    ax_drift.set_ylabel("Relative change [-]")
    ax_drift.legend(frameon=False, fontsize=7)
    _style_axes(ax_drift)
    ax_note = fig.add_axes([0.64, 0.14, 0.285, 0.27])
    ax_note.axis("off")
    note = _metric(result, "diagnostic.energy.max_relative_drift")
    ax_note.text(0, 0.95, "INTERPRETATION", fontsize=7.2, fontweight="bold", color=REPORT_PALETTE["graphite"])
    ax_note.text(
        0,
        0.82,
        textwrap.fill(note.interpretation if note and note.interpretation else "Unavailable", 40),
        fontsize=7.6,
        color=REPORT_PALETTE["graphite_mid"],
        va="top",
        linespacing=1.45,
    )
    return fig


def _diagnostics_page(result: OrbitAnalysisResult, preset: ReportPreset) -> plt.Figure:
    del preset
    fig = _figure()
    _title(fig, "08", "Physical and Invariant Diagnostics", "Conservation expectations follow the active force model.")
    rows = [
        [metric.label, _display(metric), metric.kind]
        for metric in result.metrics
        if metric.metric_id.startswith("diagnostic.")
    ]
    ax_table = fig.add_axes([0.075, 0.48, 0.85, 0.34])
    _draw_table(ax_table, ["Quantity", "Value", "Role"], rows, widths=(0.48, 0.32, 0.20), font_size=6.8)
    fig.text(
        0.075,
        0.445,
        "Diagnostic only - energy and angular momentum are not expected invariants under the active time-dependent/non-central force model.",
        fontsize=6.7,
        color=REPORT_PALETTE["muted"],
    )
    ax = fig.add_axes([0.075, 0.14, 0.85, 0.25])
    t_days = result.series.t_s / 86_400.0
    ax.plot(t_days, result.series.specific_energy_j_kg, color=REPORT_SERIES_COLORS["energy"], linewidth=1.2, label="Specific energy [J/kg]")
    ax2 = ax.twinx()
    ax2.plot(t_days, result.series.angular_momentum_m2_s, color=REPORT_SERIES_COLORS["angular_momentum"], linewidth=1.1, linestyle="--", label="|h| [m^2/s]")
    ax.set_xlabel("Simulation time [days]")
    ax.set_ylabel("Specific energy [J/kg]")
    ax2.set_ylabel("Angular momentum [m^2/s]", color=REPORT_SERIES_COLORS["angular_momentum"])
    _style_axes(ax)
    ax2.grid(False)
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    ax2.legend(frameon=False, fontsize=7, loc="upper right")
    return fig


def _force_page(result: OrbitAnalysisResult, preset: ReportPreset) -> plt.Figure:
    del preset
    fig = _figure()
    _title(fig, "09", "Perturbation Overview", "Configured physics, measured availability, and unique force scales.")
    fig.text(0.075, 0.825, "FORCE MODEL STATUS", fontsize=7.0, fontweight="bold", color=REPORT_PALETTE["graphite"])
    ax_status = fig.add_axes([0.075, 0.675, 0.85, 0.135])
    _draw_force_status_matrix(ax_status, result, columns=5)

    ax_range = fig.add_axes([0.15, 0.355, 0.75, 0.255])
    _plot_force_range(ax_range, result, limit=8)

    ranked = _ranked_unique_forces(result, limit=5)
    rows: list[list[str]] = []
    for item in ranked:
        assert item.median_m_s2 is not None and item.p95_m_s2 is not None and item.maximum_m_s2 is not None
        rows.append(
            [
                item.label,
                f"{item.median_m_s2:.3e}",
                f"{item.p95_m_s2:.3e}",
                f"{item.maximum_m_s2:.3e}",
            ]
        )
    ax_table = fig.add_axes([0.075, 0.12, 0.57, 0.17])
    _draw_table(
        ax_table,
        ["Unique component", "Median", "P95", "Maximum"],
        rows,
        widths=(0.48, 0.17, 0.17, 0.18),
        font_size=6.1,
    )
    ax_insight = fig.add_axes([0.69, 0.12, 0.235, 0.17])
    ax_insight.axis("off")
    ax_insight.add_patch(
        plt.Rectangle((0, 0), 1, 1, facecolor=REPORT_PALETTE["paper_alt"], edgecolor=REPORT_PALETTE["rule"], linewidth=0.7)
    )
    ax_insight.text(0.06, 0.82, "DOMINANT SCALE", fontsize=6.2, fontweight="bold", color=REPORT_PALETTE["muted"])
    if ranked:
        strongest = ranked[0]
        ax_insight.text(0.06, 0.60, strongest.label, fontsize=8.3, fontweight="bold", color=REPORT_PALETTE["graphite"])
        ax_insight.text(0.06, 0.42, f"median {float(strongest.median_m_s2 or 0.0):.3e} m/s^2", fontsize=6.4, color=REPORT_PALETTE["graphite_mid"])
        if len(ranked) > 1 and ranked[1].median_m_s2:
            ratio = float(strongest.median_m_s2 or 0.0) / float(ranked[1].median_m_s2)
            ax_insight.text(0.06, 0.22, f"{ratio:.1f}x next measured component", fontsize=6.3, color=REPORT_PALETTE["lunar_blue"])
    else:
        ax_insight.text(0.06, 0.52, "Unavailable", fontsize=8.3, color=REPORT_PALETTE["muted"])
    fig.text(
        0.075,
        0.085,
        "Central gravity and aggregate rows remain in force_budget.csv but are excluded here to prevent double counting and scale compression.",
        fontsize=6.5,
        color=REPORT_PALETTE["muted"],
    )
    return fig


def _force_dynamics_page(result: OrbitAnalysisResult, preset: ReportPreset) -> plt.Figure:
    del preset
    fig = _figure()
    _title(fig, "10", "Perturbation Time History and RIC", "Dominant magnitudes and signed orbital-frame directionality.")
    ax_history = fig.add_axes([0.075, 0.51, 0.85, 0.30])
    _plot_force_history(ax_history, result, limit=6)
    ax_ric = fig.add_axes([0.075, 0.18, 0.85, 0.235])
    _plot_total_ric(ax_ric, result)
    total_ric = result.force_ric_m_s2.get("Total non-central acceleration")
    spans: list[str] = []
    if total_ric is not None:
        for axis, label in enumerate(("R", "I", "C")):
            finite = np.asarray(total_ric[:, axis], dtype=np.float64)
            finite = finite[np.isfinite(finite)]
            if finite.size:
                spans.append(f"{label} [{np.min(finite):.2e}, {np.max(finite):.2e}]")
    note = (
        "Signed ranges [m/s^2]: " + "; ".join(spans)
        if spans
        else "Signed RIC ranges are unavailable because vector force telemetry was not persisted."
    )
    fig.text(0.075, 0.125, textwrap.fill(note, 125), fontsize=6.6, color=REPORT_PALETTE["graphite_mid"])
    fig.text(
        0.075,
        0.09,
        "RIC uses the instantaneous radial, in-track, and cross-track basis. Total non-central acceleration is a vector sum of unique components.",
        fontsize=6.5,
        color=REPORT_PALETTE["muted"],
    )
    return fig


def _provenance_page(result: OrbitAnalysisResult, preset: ReportPreset) -> plt.Figure:
    del preset
    fig = _figure()
    _title(fig, "11", "Provenance and Reproducibility", "Hashes, software environment, assets, backend, and generation contract.")
    core = {
        key: value
        for key, value in result.provenance.items()
        if key not in {"gravity_model", "spice_kernels", "surface_assets"}
    }
    rows = [[path, _short_value(value, 78)] for path, value in _flatten_mapping(core)]
    rows = rows[:44]
    ax = fig.add_axes([0.075, 0.11, 0.85, 0.72])
    _draw_table(ax, ["Provenance field", "Value"], rows, widths=(0.34, 0.66), font_size=6.25)
    return fig


def _assets_page(result: OrbitAnalysisResult, preset: ReportPreset) -> plt.Figure:
    del preset
    fig = _figure()
    _title(
        fig,
        "A",
        "Appendix A / Data Assets",
        "Gravity, ephemeris, and optional surface inputs with recorded integrity hashes.",
    )
    rows: list[tuple[str, str, str, str]] = []
    gravity = result.provenance.get("gravity_model", {})
    if isinstance(gravity, Mapping):
        gravity_path = str(gravity.get("path") or gravity.get("name") or "unavailable")
        rows.append(
            (
                "Gravity model",
                gravity_path,
                str(gravity.get("sha256") or "unavailable"),
                str(gravity.get("status") or "recorded"),
            )
        )
    kernels = result.provenance.get("spice_kernels", [])
    if isinstance(kernels, list | tuple):
        for index, item in enumerate(kernels):
            if not isinstance(item, Mapping):
                continue
            path = str(item.get("path") or "unavailable")
            default_label = Path(path).name if path != "unavailable" else f"SPICE kernel {index + 1}"
            rows.append(
                (
                    str(item.get("label") or default_label),
                    path,
                    str(item.get("sha256") or "unavailable"),
                    str(item.get("status") or "unknown"),
                )
            )
    assets = result.provenance.get("surface_assets", [])
    if isinstance(assets, list | tuple):
        for index, item in enumerate(assets):
            if not isinstance(item, Mapping):
                continue
            path = str(item.get("path") or "unavailable")
            default_label = Path(path).name if path != "unavailable" else f"Surface asset {index + 1}"
            rows.append(
                (
                    str(item.get("label") or default_label),
                    path,
                    str(item.get("sha256") or "unavailable"),
                    str(item.get("status") or "unknown"),
                )
            )
    ax = fig.add_axes([0.075, 0.19, 0.85, 0.64])
    ax.axis("off")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_autoscale_on(False)
    if not rows:
        ax.text(0.0, 0.95, "No data assets were recorded for this run.", fontsize=8, color=REPORT_PALETTE["muted"])
    else:
        block_height = 1.0 / len(rows)
        for index, (label, path, sha256, status) in enumerate(rows):
            top = 1.0 - index * block_height
            ax.text(0.0, top - 0.13 * block_height, label, fontsize=7.1, fontweight="bold", va="top")
            ax.text(
                1.0,
                top - 0.13 * block_height,
                status.upper(),
                fontsize=6.2,
                fontweight="bold",
                color=REPORT_STATUS_COLORS.get(status, REPORT_PALETTE["muted"]),
                ha="right",
                va="top",
            )
            ax.text(0.0, top - 0.40 * block_height, "PATH", fontsize=5.5, fontweight="bold", color=REPORT_PALETTE["muted"], va="top")
            ax.text(0.08, top - 0.40 * block_height, path, fontsize=5.8, family="monospace", va="top")
            ax.text(0.0, top - 0.66 * block_height, "SHA-256", fontsize=5.5, fontweight="bold", color=REPORT_PALETTE["muted"], va="top")
            ax.text(0.08, top - 0.66 * block_height, sha256, fontsize=5.8, family="monospace", va="top")
            ax.plot([0.0, 1.0], [top - 0.94 * block_height, top - 0.94 * block_height], color=REPORT_PALETTE["rule"], linewidth=0.45)
    fig.text(
        0.075,
        0.13,
        "Full, machine-readable values are preserved in provenance.json; unavailable hashes remain explicit and are never inferred.",
        fontsize=6.8,
        color=REPORT_PALETTE["muted"],
    )
    return fig


_PAGE_FACTORIES: Mapping[str, Callable[[OrbitAnalysisResult, ReportPreset], plt.Figure]] = {
    "cover": _cover_page,
    "executive": _executive_page,
    "configuration": _configuration_page,
    "orbit": _orbit_page,
    "spatial": _spatial_page,
    "elements": _elements_page,
    "events": _events_page,
    "numerical": _numerical_page,
    "diagnostics": _diagnostics_page,
    "force_budget": _force_page,
    "force_dynamics": _force_dynamics_page,
    "provenance": _provenance_page,
    "assets": _assets_page,
}


_PAGE_LABELS = {
    "cover": "Cover",
    "executive": "Executive Mission Summary",
    "configuration": "Mission Configuration",
    "orbit": "Orbit Geometry and Evolution",
    "spatial": "Spatial Context and Coverage",
    "elements": "Orbital Element Evolution",
    "events": "Critical Extrema and Events",
    "numerical": "Numerical Health",
    "diagnostics": "Physical Diagnostics",
    "force_budget": "Perturbation Overview",
    "force_dynamics": "Perturbation Time History and RIC",
    "provenance": "Provenance",
    "assets": "Appendix / Data Assets",
}


def _failure_page(section: str, exc: Exception) -> plt.Figure:
    fig = _figure()
    fig.text(0.08, 0.88, section, fontsize=18, fontweight="bold", color=REPORT_PALETTE["graphite"])
    fig.text(0.08, 0.80, "Optional section unavailable", fontsize=11, color=REPORT_STATUS_COLORS["warning"])
    fig.text(
        0.08,
        0.74,
        textwrap.fill(str(exc), 95),
        fontsize=8,
        color=REPORT_PALETTE["muted"],
    )
    return fig


def write_report_pdf(result: OrbitAnalysisResult, path: str | Path) -> Path:
    """Write the editorial multi-page PDF without letting one page abort it."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    preset = _preset(result.preset)
    sections = preset.page_sections
    with PdfPages(output) as pdf:
        info = pdf.infodict()
        info["Title"] = f"Lunaris Mission Analysis - {result.run_id}"
        info["Author"] = "Lunaris"
        info["Subject"] = "Post-propagation orbit analysis and reproducibility package"
        info["Keywords"] = "astrodynamics, lunar orbit, numerical diagnostics, provenance"
        try:
            timestamp = datetime.fromisoformat(result.generated_at_utc.replace("Z", "+00:00"))
            info["CreationDate"] = timestamp
            info["ModDate"] = timestamp
        except ValueError:
            pass
        for page_number, section in enumerate(sections, start=1):
            fig: plt.Figure | None = None
            try:
                fig = _PAGE_FACTORIES[section](result, preset)
            except Exception as exc:
                logger.exception("Report page %s failed", section)
                fig = _failure_page(_PAGE_LABELS[section], exc)
            _page_chrome(
                fig,
                result,
                section=_PAGE_LABELS[section],
                page_number=page_number,
                page_count=len(sections),
                cover=section == "cover",
            )
            with matplotlib.rc_context({"savefig.bbox": None}):
                pdf.savefig(fig, facecolor=REPORT_PALETTE["paper"], bbox_inches=None)
            plt.close(fig)
    return output


def write_report_figures(result: OrbitAnalysisResult, run_dir: str | Path) -> dict[str, Path]:
    """Write stable PNG figures (and SVGs for Paper) from typed result series."""

    root = Path(run_dir) / "figures"
    root.mkdir(parents=True, exist_ok=True)
    preset = _preset(result.preset)
    factories: Mapping[str, Callable[[OrbitAnalysisResult, ReportPreset], plt.Figure]] = {
        "orbit_overview": figure_orbit_overview,
        "spatial_context": figure_spatial_context,
        "altitude_history": figure_altitude_history,
        "orbit_envelope": figure_orbit_envelope,
        "orbital_elements": figure_orbital_elements,
        "numerical_health": figure_numerical_health,
        "force_budget": figure_force_budget,
        "force_ric": figure_force_ric,
        "event_timeline": figure_event_timeline,
        "groundtrack": figure_groundtrack,
    }
    outputs: dict[str, Path] = {}
    for name, factory in factories.items():
        fig: plt.Figure | None = None
        try:
            fig = factory(result, preset)
            png = root / f"{name}.png"
            fig.savefig(png, dpi=preset.dpi, facecolor=REPORT_PALETTE["paper"], bbox_inches="tight")
            outputs[name] = png
            if preset.vector_figures:
                svg = root / f"{name}.svg"
                fig.savefig(svg, facecolor=REPORT_PALETTE["paper"], bbox_inches="tight")
                outputs[f"{name}_svg"] = svg
        except Exception:
            logger.exception("Optional report figure %s failed", name)
        finally:
            if fig is not None:
                plt.close(fig)
    return outputs


def generate_analysis_package(result: OrbitAnalysisResult, run_dir: str | Path) -> dict[str, Any]:
    """Generate all canonical artifacts in the existing run directory."""

    root = Path(run_dir)
    root.mkdir(parents=True, exist_ok=True)
    core = write_analysis_artifacts(result, root)
    figures = write_report_figures(result, root)
    pdf = write_report_pdf(result, root / "report.pdf")
    manifest = write_artifact_manifest(root)
    return {
        "status": "success",
        "out_dir": str(root),
        "report_markdown": str(core["report_markdown"]),
        "pdf": str(pdf),
        "metrics": str(core["metrics"]),
        "events": str(core["events"]),
        "orbital_elements": str(core["orbital_elements"]),
        "force_budget": str(core["force_budget"]),
        "figures": {name: str(path) for name, path in figures.items()},
        "manifest": str(manifest),
    }


def regenerate_analysis_package(
    run_dir: str | Path,
    *,
    preset: str | None = None,
) -> dict[str, Any]:
    """Regenerate presentation artifacts from persisted canonical analysis."""

    result = load_analysis_artifacts(run_dir)
    if preset is not None and preset != result.preset:
        if preset not in REPORT_PRESETS:
            raise ValueError(f"unknown report preset: {preset!r}")
        result = replace(result, preset=preset)
    return generate_analysis_package(result, run_dir)


__all__ = [
    "REPORT_PRESETS",
    "ReportPreset",
    "figure_altitude_history",
    "figure_event_timeline",
    "figure_force_budget",
    "figure_force_ric",
    "figure_groundtrack",
    "figure_numerical_health",
    "figure_orbit_envelope",
    "figure_orbit_overview",
    "figure_orbital_elements",
    "figure_spatial_context",
    "generate_analysis_package",
    "regenerate_analysis_package",
    "write_report_figures",
    "write_report_pdf",
]
