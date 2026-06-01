# ST_LRPS/analysis/plotting.py
# -*- coding: utf-8 -*-
"""
Plotting and report-figure utilities for ST_LRPS.

This module generates Matplotlib figures used by the PDF/HTML reporting layer and
provides small helpers for table rendering and metadata discovery. It is designed
to be robust against partial histories (missing fields, mismatched array lengths)
and to fail softly by emitting placeholder figures instead of raising.

Key features
------------
- Figure builders for orbital elements, altitude/events, invariants, relative drift,
  ground track maps, 3D trajectory views, phase-space plots, and acceleration budgets.
- Report layout helpers for striped metric tables and key-value summary blocks.
- Conservative auto-detection of enabled physics effects from metadata/history and,
  when available, recovery/merging of a nearby JSON config from the output directory.
- Consistent styling via the project's styling helpers (theme, legends, colorbars),
  with safe fallbacks where appropriate.

Notes
-----
- This module avoids interactive backends by default (suitable for batch runs).
- Effect detection is intended for labeling/reporting only; it prefers explicit
  boolean toggles and avoids false positives.
"""

# =============================================================================
# 0.                              IMPORTS
# =============================================================================

from __future__ import annotations

import os
import math
import json
import re
from typing import Any, Dict, Mapping, Optional, Tuple, List, Iterable

import numpy as np

import matplotlib

# --- Matplotlib backend selection ---
# Must run BEFORE importing pyplot.
if os.environ.get("STLRPS_INTERACTIVE", "0").strip().lower() not in ("1", "true", "yes", "y"):
    try:
        matplotlib.use("Agg")
    except Exception:
        pass

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import matplotlib.legend



# Optional: acceleration reconstruction plots
try:
    from lunaris.core.dynamics import make_accel  # type: ignore
except ImportError:
    make_accel = None

from lunaris.analysis.postprocess import (
    extract_time_days,
    extract_invariants,
    _extract_rv_vectors,
    _first_present,
)

from .styling import (
    DEFAULT_STYLE,
    THEME,
    apply_rcparams,
    add_lunar_background,
    format_scientific_axis,
    format_log_axis_sci,
    get_accel_color,
    get_series_color,
    apply_legend_style,
    apply_standard_colorbar,
)

from lunaris.common.math_utils import wrap_lon_deg



# =============================================================================
# 2.                            CORE PLOT PRIMITIVES
# =============================================================================

def _legend(
    ax: plt.Axes,
    *,
    dedupe: bool = True,
    ignore_labels: Tuple[str, ...] = ("_nolegend_", ""),
    **kwargs,
) -> Optional[matplotlib.legend.Legend]:
    """
    Apply a consistent legend style (prefer project styling; fall back to Matplotlib).

    Features
    --------
    - Filters empty / special labels (e.g. "_nolegend_")
    - Optionally deduplicates repeated labels while preserving first occurrence
    - Fail-soft: returns None if a legend cannot be created
    """
    try:
        handles, labels = ax.get_legend_handles_labels()
    except Exception:
        return None

    if not handles or not labels:
        return None

    # 1) Filter labels
    kept_h: list = []
    kept_l: list[str] = []
    for h, l in zip(handles, labels):
        if l is None:
            continue
        s = str(l).strip()
        if s in ignore_labels:
            continue
        kept_h.append(h)
        kept_l.append(s)

    if not kept_h:
        return None

    # 2) Deduplicate by label (stable)
    if dedupe:
        seen: set[str] = set()
        dh: list = []
        dl: list[str] = []
        for h, l in zip(kept_h, kept_l):
            if l in seen:
                continue
            seen.add(l)
            dh.append(h)
            dl.append(l)
        kept_h, kept_l = dh, dl

    # Reasonable default if user didn't specify location
    kwargs.setdefault("loc", "best")

    # 3) Prefer project legend styling if available
    try:
        # Many helper styles accept handles/labels; if not, we fall back.
        try:
            return apply_legend_style(ax, handles=kept_h, labels=kept_l, **kwargs)
        except TypeError:
            leg = apply_legend_style(ax, **kwargs)
            return leg
    except Exception:
        pass

    # 4) Matplotlib fallback
    try:
        return ax.legend(handles=kept_h, labels=kept_l, **kwargs)
    except Exception:
        return None


def shade_boolean_intervals(
    ax: plt.Axes,
    x: np.ndarray,
    mask: np.ndarray,
    *,
    alpha: float = 0.12,
    label: str = "Eclipse",
    color: Optional[str] = None,
    ymin: float = 0.0,
    ymax: float = 1.0,
    min_width: float = 0.0,
    extend_to_next: bool = True,
    zorder: int = 0,
) -> None:
    """
    Shade vertical spans where `mask` is True.

    Parameters
    ----------
    ax
        Target axes.
    x
        X-axis samples (typically time in seconds or days).
    mask
        Boolean mask aligned with `x`. True segments are shaded.
    alpha
        Span transparency.
    label
        Legend label (applied to the first span only).
    color
        Span color. If None, uses THEME["grid"] when available, else a neutral gray.
    ymin, ymax
        Relative y-limits in axes coordinates for the span (0..1).
    min_width
        Skip spans narrower than this (in x units).
    extend_to_next
        If True, extends each segment's right edge to the next sample (when possible).
    zorder
        Drawing order (lower = further back).
    """
    x = np.asarray(x, dtype=float).ravel()
    mask = np.asarray(mask, dtype=bool).ravel()

    n = min(x.size, mask.size)
    if n < 2:
        return
    x = x[:n]
    mask = mask[:n]

    # Treat non-finite x as False in mask
    finite_x = np.isfinite(x)
    if not np.all(finite_x):
        mask = mask & finite_x

    if not np.any(mask):
        return

    if color is None:
        try:
            color = THEME["grid"]
        except Exception:
            color = "0.5"

    # Find [start, end) index pairs of True runs
    padded = np.concatenate(([False], mask, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    if edges.size < 2:
        return
    runs = edges.reshape(-1, 2)

    first = True
    for start, end in runs:
        x0 = x[start]
        x1 = x[end - 1]

        if extend_to_next and end < n and np.isfinite(x[end]):
            x1 = x[end]

        if not (np.isfinite(x0) and np.isfinite(x1)):
            continue

        xa, xb = (x0, x1) if x0 <= x1 else (x1, x0)
        if float(min_width) > 0.0 and (xb - xa) < float(min_width):
            continue

        ax.axvspan(
            xa,
            xb,
            ymin=float(ymin),
            ymax=float(ymax),
            alpha=float(alpha),
            color=color,
            lw=0.0,
            label=(label if first else None),
            zorder=int(zorder),
        )
        first = False


def time_colored_path(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    t: Optional[np.ndarray] = None,
    *,
    cmap: str = "plasma",
    linewidth: float = 1.8,
    alpha: float = 0.95,
    jump_deg: float = 180.0,
    add_colorbar: bool = True,
    cbar_label: str = "Time [days]",
    cbar_orientation: str = "horizontal",
    cbar_pad: float = 0.08,
    cbar_fraction: float = 0.05,
    autoscale: bool = True,
    zorder: int = 3,
) -> Tuple[Optional[LineCollection], Optional[matplotlib.colorbar.Colorbar]]:
    """
    Draw a 2D path colored by time, while breaking discontinuities (e.g., longitude wrap).

    Implementation details
    ----------------------
    - Drops non-finite samples in (x, y, t)
    - Splits segments where abs(diff(x)) exceeds `jump_deg`
    - Renders a single LineCollection for performance

    Returns
    -------
    (LineCollection, Colorbar) or (None, None)
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()

    n = min(x.size, y.size)
    if n < 2:
        return None, None
    x = x[:n]
    y = y[:n]

    if t is None:
        t = np.linspace(0.0, 1.0, n, dtype=float)
    else:
        t = np.asarray(t, dtype=float).ravel()[:n]

    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(t)
    if not np.all(finite):
        x = x[finite]
        y = y[finite]
        t = t[finite]
        n = int(x.size)
        if n < 2:
            return None, None

    tmin = float(np.nanmin(t))
    tmax = float(np.nanmax(t))
    if (not np.isfinite(tmin)) or (not np.isfinite(tmax)) or (tmax <= tmin):
        tmin, tmax = 0.0, 1.0
    norm = Normalize(vmin=tmin, vmax=tmax)

    jump_deg = float(jump_deg)
    jumps = np.where(np.abs(np.diff(x)) > jump_deg)[0]
    cuts = np.concatenate(([0], jumps + 1, [n])).astype(np.int64)

    seg_list: list[np.ndarray] = []
    c_list: list[np.ndarray] = []

    for a, b in zip(cuts[:-1], cuts[1:]):
        if (b - a) < 2:
            continue

        xs = x[a:b]
        ys = y[a:b]
        ts = t[a:b]

        pts = np.column_stack((xs, ys)).reshape(-1, 1, 2)
        segs = np.concatenate((pts[:-1], pts[1:]), axis=1)

        # Color each segment by midpoint time for smoother gradients
        cvals = 0.5 * (ts[:-1] + ts[1:])

        seg_list.append(segs)
        c_list.append(cvals)

    if not seg_list:
        return None, None

    seg_all = np.vstack(seg_list)
    c_all = np.concatenate(c_list)

    lc = LineCollection(
        seg_all,
        cmap=str(cmap),
        norm=norm,
        linewidths=float(linewidth),
        alpha=float(alpha),
        zorder=int(zorder),
    )
    lc.set_array(c_all)
    ax.add_collection(lc)

    if autoscale:
        try:
            ax.update_datalim(np.column_stack((x, y)))
            ax.autoscale_view()
        except Exception:
            pass

    cbar = None
    if add_colorbar:
        try:
            cbar = ax.figure.colorbar(
                lc,
                ax=ax,
                orientation=str(cbar_orientation),
                pad=float(cbar_pad),
                fraction=float(cbar_fraction),
            )
            cbar.set_label(str(cbar_label))
            try:
                apply_standard_colorbar(cbar)
            except Exception:
                pass
        except Exception:
            cbar = None

    return lc, cbar



# =============================================================================
# 3.                    INTERNAL DATA & FORMATTING HELPERS
# =============================================================================

def _as_np(
    x: Any,
    dtype: Any = None,
    *,
    copy: bool = False,
    atleast_1d: bool = False,
    ravel: bool = False,
) -> np.ndarray:
    """
    Convert input to a NumPy array safely.

    - Supports optional dtype casting (e.g., _as_np(x, float)).
    - Returns an empty array for None.
    - Optional shape normalization via atleast_1d and ravel.
    """
    if x is None:
        return np.array([], dtype=(dtype if dtype is not None else float))

    # Fast-path for numpy arrays
    if isinstance(x, np.ndarray):
        arr = x.astype(dtype, copy=copy) if dtype is not None else (x.copy() if copy else x)
    else:
        try:
            arr = np.asarray(x, dtype=dtype)
            if copy:
                arr = arr.copy()
        except Exception:
            # Fallback: treat as a single scalar-like value
            arr = np.asarray([x], dtype=(dtype if dtype is not None else object))

    if atleast_1d:
        arr = np.atleast_1d(arr)
    if ravel:
        arr = np.ravel(arr)
    return arr


def _aligned_xy(t: Any, y: Any) -> Tuple[np.ndarray, np.ndarray]:
    """
    Align two 1D series by truncating both to the shorter length.

    This prevents plotting crashes when time and data arrays have mismatched lengths.
    """
    t_arr = _as_np(t, float, atleast_1d=True, ravel=True)
    y_arr = _as_np(y, float, atleast_1d=True, ravel=True)

    if t_arr.size == 0 or y_arr.size == 0:
        return t_arr, y_arr

    n = int(min(t_arr.size, y_arr.size))
    return t_arr[:n], y_arr[:n]


def _get_body_radius_km(meta: Mapping[str, Any], history: Mapping[str, Any]) -> float:
    """
    Retrieve body radius in kilometers from meta/history.

    Falls back to the Moon mean radius (1737.4 km) if not found or invalid.
    """
    def _valid_positive_float(v: Any) -> Optional[float]:
        try:
            f = float(v)
            if math.isfinite(f) and f > 0.0:
                return f
        except Exception:
            return None
        return None

    # Flat keys (km)
    for container in (meta, history):
        if not isinstance(container, Mapping):
            continue

        for k in ("body_radius_km", "R_km", "moon_radius_km", "radius_km"):
            if k in container and container[k] is not None:
                out = _valid_positive_float(container[k])
                if out is not None:
                    return out

        # Flat keys (m -> km)
        for k in ("body_radius_m", "R_m", "moon_radius_m", "radius_m"):
            if k in container and container[k] is not None:
                out = _valid_positive_float(container[k])
                if out is not None:
                    return out / 1000.0

        # Common nested pattern: container["body"]["radius_m"/"radius_km"]
        body = container.get("body", None)
        if isinstance(body, Mapping):
            for k in ("radius_km", "R_km"):
                if k in body and body[k] is not None:
                    out = _valid_positive_float(body[k])
                    if out is not None:
                        return out
            for k in ("radius_m", "R_m"):
                if k in body and body[k] is not None:
                    out = _valid_positive_float(body[k])
                    if out is not None:
                        return out / 1000.0

    return 1737.4


def _as_bool(v: Any) -> bool:
    """
    Convert mixed types (bool/int/float/str) into a boolean flag.

    - Treats NaN as False.
    - Recognizes common string tokens.
    - Defaults to False for unknown inputs (conservative).
    """
    if v is None:
        return False
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return bool(int(v))
    if isinstance(v, (float, np.floating)):
        fv = float(v)
        return bool(fv) if math.isfinite(fv) else False

    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "y", "on", "enabled", "enable"):
        return True
    if s in ("0", "false", "no", "n", "off", "disabled", "disable", ""):
        return False
    return False


def _wrap(text: str, width: int = 50) -> str:
    """
    Wrap text to a fixed width for cleaner titles/annotations.

    Preserves existing newline-separated paragraphs.
    """
    if not text:
        return ""
    width_i = int(max(10, width))
    try:
        import textwrap
        parts = str(text).splitlines() or [str(text)]
        out: list[str] = []
        for p in parts:
            p = p.strip()
            out.append("" if not p else textwrap.fill(p, width=width_i))
        return "\n".join(out)
    except Exception:
        # Minimal fallback without importing textwrap
        words = str(text).split()
        lines: list[str] = []
        line: list[str] = []
        cur = 0
        for w in words:
            if line and (cur + len(w) + 1 > width_i):
                lines.append(" ".join(line))
                line = [w]
                cur = len(w)
            else:
                line.append(w)
                cur += len(w) + 1
        if line:
            lines.append(" ".join(line))
        return "\n".join(lines)


def _nan_stats(x: np.ndarray) -> Dict[str, float]:
    """
    Compute start/end/min/max while ignoring NaN and Inf.

    Start/end are taken as the first/last finite values (not simply x[0]/x[-1]).
    """
    arr = _as_np(x, float, atleast_1d=True, ravel=True)
    nan = float("nan")
    if arr.size == 0:
        return {"start": nan, "end": nan, "min": nan, "max": nan}

    finite = np.isfinite(arr)
    if not np.any(finite):
        return {"start": nan, "end": nan, "min": nan, "max": nan}

    idx = np.flatnonzero(finite)
    vals = arr[finite]
    return {
        "start": float(arr[idx[0]]),
        "end": float(arr[idx[-1]]),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
    }


def _fmt_fixed(x: Any, nd: int = 2) -> str:
    """
    Format a number with fixed decimals.

    Returns an em dash for invalid or non-finite inputs.
    """
    try:
        val = float(x)
        if not math.isfinite(val):
            return "—"
        nd_i = int(max(0, min(12, nd)))
        return f"{val:.{nd_i}f}"
    except (TypeError, ValueError):
        return "—"


def _fmt_sci(x: Any, nd: int = 2) -> str:
    """
    Format a number in scientific notation.

    Returns an em dash for invalid or non-finite inputs.
    """
    try:
        val = float(x)
        if not math.isfinite(val):
            return "—"
        nd_i = int(max(0, min(12, nd)))
        return f"{val:.{nd_i}e}"
    except (TypeError, ValueError):
        return "—"



# =============================================================================
# 4.                        FIGURE BUILDERS (figure_*)
# =============================================================================

def figure_elements_timeseries(t_days: np.ndarray, elems: Dict[str, np.ndarray]) -> plt.Figure:
    """
    Plot a 3x2 grid showing the evolution of Keplerian orbital elements.

    Panels:
      1) Semi-major axis a [km]
      2) Eccentricity e [-]
      3) Inclination i [deg]
      4) RAAN Ω [deg]
      5) Argument of periapsis ω [deg]
      6) Apsidal radii rp, ra [km] derived from a and e

    Parameters
    ----------
    t_days : np.ndarray
        Time array in days.
    elems : Dict[str, np.ndarray]
        Dictionary with keys: 'a_km', 'e', 'i_deg', 'raan_deg', 'argp_deg'.

    Returns
    -------
    matplotlib.figure.Figure
        Figure ready for saving.
    """
    apply_rcparams(DEFAULT_STYLE)

    t_days = _as_np(t_days, float, atleast_1d=True, ravel=True)
    fig, axes = plt.subplots(3, 2, figsize=(11.7, 8.3))
    fig.suptitle("Orbital Elements Evolution", fontsize=16, fontweight="bold", y=0.96)
    axs = np.asarray(axes).ravel()

    # Plot specifications: key, title, y-label, subplot index
    specs = [
        ("a_km",     r"Semi-major Axis $a$",              "Distance [km]", 0),
        ("e",        r"Eccentricity $e$",                 "[-]",           1),
        ("i_deg",    r"Inclination $i$",                  "Angle [deg]",   2),
        ("raan_deg", r"RAAN $\Omega$",                    "Angle [deg]",   3),
        ("argp_deg", r"Argument of Periapsis $\omega$",   "Angle [deg]",   4),
    ]

    def _plot_1d(ax: plt.Axes, key: str, title: str, ylabel: str) -> None:
        data = _as_np(elems.get(key, None), float, atleast_1d=True, ravel=True)
        if t_days.size < 2 or data.size < 2:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, color="0.45")
            ax.set_title(title, loc="left", fontsize=11)
            return

        tt, yy = _aligned_xy(t_days, data)
        if tt.size < 2:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, color="0.45")
            ax.set_title(title, loc="left", fontsize=11)
            return

        color = None
        try:
            color = get_series_color(key)
        except Exception:
            color = None

        ax.plot(tt, yy, linewidth=1.6, color=color, zorder=2)
        ax.set_title(title, loc="left", fontsize=11)
        ax.set_ylabel(ylabel, fontweight="bold")
        ax.grid(True, which="major", alpha=0.30)
        ax.grid(True, which="minor", alpha=0.12, linestyle=":")
        ax.minorticks_on()

        if key == "a_km":
            format_scientific_axis(ax, "y")

    for key, title, ylabel, idx in specs:
        _plot_1d(axs[idx], key, title, ylabel)

    # Apsidal radii panel
    ax_r = axs[5]
    a = _as_np(elems.get("a_km", None), float, atleast_1d=True, ravel=True)
    e = _as_np(elems.get("e", None), float, atleast_1d=True, ravel=True)

    if t_days.size >= 2 and a.size >= 2 and e.size >= 2:
        n = int(min(t_days.size, a.size, e.size))
        tt = t_days[:n]
        aa = a[:n]
        ee = e[:n]

        rp = aa * (1.0 - ee)
        ra = aa * (1.0 + ee)

        c_rp = None
        c_ra = None
        try:
            c_rp = get_series_color("periapsis")
        except Exception:
            c_rp = None
        try:
            c_ra = get_series_color("apoapsis")
        except Exception:
            c_ra = None

        ax_r.plot(tt, rp, label=r"$r_p=a(1-e)$", linewidth=1.6, color=c_rp, zorder=2)
        ax_r.plot(tt, ra, label=r"$r_a=a(1+e)$", linewidth=1.6, linestyle="--", color=c_ra, zorder=2)

        ax_r.set_title(r"Apsidal Radii ($r_p, r_a$)", loc="left", fontsize=11)
        ax_r.set_ylabel("Radius [km]", fontweight="bold")
        ax_r.grid(True, which="major", alpha=0.30)
        ax_r.grid(True, which="minor", alpha=0.12, linestyle=":")
        ax_r.minorticks_on()
        _legend(ax_r, loc="best")
        format_scientific_axis(ax_r, "y")
    else:
        ax_r.text(0.5, 0.5, "Insufficient data", ha="center", va="center", transform=ax_r.transAxes, color="0.45")
        ax_r.set_title(r"Apsidal Radii ($r_p, r_a$)", loc="left", fontsize=11)

    # Common x-label on bottom row
    for ax in axs[-2:]:
        ax.set_xlabel("Time [days]", fontweight="bold")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def figure_invariants(t_days: np.ndarray, inv: Dict[str, np.ndarray]) -> plt.Figure:
    """
    Plot state norms and conservation checks (invariants).

    Shows the following when available:
      - ||r|| [km]
      - ||v|| [km/s]
      - Energy drift (epsilon - epsilon0) [J/kg]
      - Angular momentum drift (h - h0) [m^2/s]

    Missing series are omitted and the layout is computed dynamically.
    """
    apply_rcparams(DEFAULT_STYLE)

    t_days = _as_np(t_days, float, atleast_1d=True, ravel=True)
    r = _as_np(inv.get("r_norm_km", None), float, atleast_1d=True, ravel=True)
    v = _as_np(inv.get("v_norm_kmps", None), float, atleast_1d=True, ravel=True)
    E = _as_np(inv.get("energy_Jkg", None), float, atleast_1d=True, ravel=True)
    h = _as_np(inv.get("h_norm_m2s", None), float, atleast_1d=True, ravel=True)

    specs: list[tuple[str, str, np.ndarray, str]] = []
    if r.size >= 2:
        specs.append(("r_norm", r"$\|r\|$", r, "Position [km]"))
    if v.size >= 2:
        specs.append(("v_norm", r"$\|v\|$", v, "Velocity [km/s]"))
    if E.size >= 2:
        specs.append(("energy", r"$\Delta \epsilon$", E - float(E[0]), "Energy Drift [J/kg]"))
    if h.size >= 2:
        specs.append(("h_norm", r"$\Delta h$", h - float(h[0]), r"Ang. Mom. Drift [m$^2$/s]"))

    fig = plt.figure(figsize=(11.7, 8.3))
    fig.suptitle("State Norms & Conservation Checks", fontsize=16, fontweight="bold", y=0.96)

    if (t_days.size < 2) or (not specs):
        ax = fig.add_subplot(1, 1, 1)
        ax.axis("off")
        ax.text(
            0.5, 0.5,
            "Invariant data unavailable.\nEnsure postprocess exports norms and invariants.",
            ha="center", va="center", transform=ax.transAxes, color="0.45"
        )
        return fig

    n_plots = len(specs)
    n_cols = 2 if n_plots > 1 else 1
    n_rows = int(math.ceil(n_plots / n_cols))
    axs = fig.subplots(n_rows, n_cols)
    axs = np.atleast_1d(axs).ravel()

    for i, (key, label, data, ylabel) in enumerate(specs):
        ax = axs[i]
        tt, yy = _aligned_xy(t_days, data)
        if tt.size < 2:
            ax.axis("off")
            continue

        color = None
        try:
            color = get_series_color(key)
        except Exception:
            color = None

        ax.plot(tt, yy, linewidth=1.7, label=label, color=color, zorder=2)
        ax.set_ylabel(ylabel, fontweight="bold")
        ax.set_xlabel("Time [days]")
        ax.grid(True, which="major", alpha=0.30)
        ax.grid(True, which="minor", alpha=0.12, linestyle=":")
        ax.minorticks_on()
        format_scientific_axis(ax, "y")
        _legend(ax, loc="best")

    for j in range(n_plots, len(axs)):
        axs[j].axis("off")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def figure_altitude_with_events(
    t_days: np.ndarray,
    alt_km: np.ndarray,
    events: Mapping[str, Any],
    eclipse: Optional[np.ndarray] = None,
) -> plt.Figure:
    """
    Plot altitude over time and annotate key orbital events.

    Features:
      - Altitude profile [km]
      - Periapsis markers
      - Apoapsis markers
      - Impact marker (if present)
      - Eclipse intervals as shaded spans (if mask provided)
    """
    apply_rcparams(DEFAULT_STYLE)

    tt, aa = _aligned_xy(t_days, alt_km)
    if tt.size < 2:
        return _placeholder_figure(
            "Altitude & Events",
            "Altitude time series is missing or too short."
        )

    fig = plt.figure(figsize=(11.7, 8.3))
    fig.suptitle("Altitude & Events Profile", fontsize=16, fontweight="bold", y=0.96)
    ax = fig.add_subplot(1, 1, 1)

    c_alt = None
    try:
        c_alt = get_series_color("altitude")
    except Exception:
        c_alt = None

    ax.plot(tt, aa, linewidth=1.6, label="Altitude", color=c_alt, zorder=2)

    # Eclipse shading (optional)
    if eclipse is not None:
        ecl = _as_np(eclipse, bool, atleast_1d=True, ravel=True)
        if ecl.size >= 2:
            ecl = ecl[:tt.size]
            shade_boolean_intervals(
                ax, tt, ecl,
                label="Eclipse",
                color=None,
                alpha=0.18,
                ymin=0.0,
                ymax=1.0,
                min_width=0.0,
                extend_to_next=True,
            )

    # Event indices
    peri_idx = _as_np(events.get("periapsis", events.get("peri_idx", None)), int, atleast_1d=True, ravel=True)
    apo_idx = _as_np(events.get("apoapsis", events.get("apo_idx", None)), int, atleast_1d=True, ravel=True)
    impact_idx = events.get("impact_index", events.get("impact_idx", None))

    def _valid_idx(idx: np.ndarray, n: int) -> np.ndarray:
        if idx.size == 0:
            return idx
        idx = idx.astype(np.int64, copy=False)
        return idx[(idx >= 0) & (idx < int(n))]

    n = int(tt.size)
    peri_v = _valid_idx(peri_idx, n)
    apo_v = _valid_idx(apo_idx, n)

    if peri_v.size:
        c = None
        try:
            c = get_series_color("periapsis")
        except Exception:
            c = None
        ax.scatter(tt[peri_v], aa[peri_v], s=28, marker="o", color=c, edgecolors="black",
                   linewidths=0.5, label="Periapsis", zorder=4)

    if apo_v.size:
        c = None
        try:
            c = get_series_color("apoapsis")
        except Exception:
            c = None
        ax.scatter(tt[apo_v], aa[apo_v], s=28, marker="o", color=c, edgecolors="black",
                   linewidths=0.5, label="Apoapsis", zorder=4)

    if impact_idx is not None:
        try:
            ii = int(impact_idx)
            if 0 <= ii < n:
                c = None
                try:
                    c = get_series_color("impact")
                except Exception:
                    c = None
                ax.scatter(tt[ii], aa[ii], s=90, marker="x", color=c, linewidths=2.2,
                           label="Impact", zorder=6)
        except Exception:
            pass

    ax.set_xlabel("Time [days]", fontweight="bold")
    ax.set_ylabel("Altitude [km]", fontweight="bold")
    ax.grid(True, which="major", alpha=0.30)
    ax.grid(True, which="minor", alpha=0.12, linestyle=":")
    ax.minorticks_on()
    format_scientific_axis(ax, "y")
    _legend(ax, loc="best")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def figure_relative_drift(t_days: np.ndarray, inv: Dict[str, np.ndarray]) -> plt.Figure:
    """
    Plot relative drift of conserved quantities vs time.

    Relative drift is computed as:
      (x - x0) / max(|x0|, eps)

    If precomputed series exist (rel_energy_drift, rel_h_drift), they are used.
    Otherwise drift is derived from absolute series.
    """
    apply_rcparams(DEFAULT_STYLE)

    t_days = _as_np(t_days, float, atleast_1d=True, ravel=True)

    def _drift(rel_key: str, abs_key: str) -> np.ndarray:
        rel = _as_np(inv.get(rel_key, None), float, atleast_1d=True, ravel=True)
        if rel.size >= 2:
            return rel
        abs_v = _as_np(inv.get(abs_key, None), float, atleast_1d=True, ravel=True)
        if abs_v.size >= 2:
            v0 = float(abs_v[0])
            denom = max(abs(v0), 1e-30)
            return (abs_v - v0) / denom
        return np.array([], dtype=float)

    rel_E = _drift("rel_energy_drift", "energy_Jkg")
    rel_h = _drift("rel_h_drift", "h_norm_m2s")

    if t_days.size < 2 or (rel_E.size < 2 and rel_h.size < 2):
        return _placeholder_figure(
            "Relative Drift",
            "Energy/angular momentum series missing; cannot compute drift."
        )

    fig = plt.figure(figsize=(11.7, 8.3))
    fig.suptitle("Integrator Accuracy: Relative Drift", fontsize=16, fontweight="bold", y=0.96)
    ax = fig.add_subplot(1, 1, 1)

    if rel_E.size >= 2:
        tt, yy = _aligned_xy(t_days, rel_E)
        c = None
        try:
            c = get_series_color("energy")
        except Exception:
            c = None
        ax.plot(tt, yy, linewidth=1.7, label=r"Energy $\Delta \epsilon/|\epsilon_0|$", color=c)

    if rel_h.size >= 2:
        tt, yy = _aligned_xy(t_days, rel_h)
        c = None
        try:
            c = get_series_color("h_norm")
        except Exception:
            c = None
        ax.plot(tt, yy, linewidth=1.7, label=r"Ang. Mom. $\Delta h/|h_0|$", color=c)

    ax.axhline(0.0, linewidth=0.9, alpha=0.5, linestyle="--", color="black", zorder=1)
    ax.set_xlabel("Time [days]", fontweight="bold")
    ax.set_ylabel("Relative error [-]", fontweight="bold")
    ax.grid(True, which="major", alpha=0.30)
    ax.grid(True, which="minor", alpha=0.12, linestyle=":")
    ax.minorticks_on()
    format_scientific_axis(ax, "y")
    _legend(ax, loc="best")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def figure_ground_track(
    history: Mapping[str, Any],
    meta: Optional[Mapping[str, Any]] = None,
    ctx: Any = None,
) -> plt.Figure:
    """
    Plot body-fixed ground track (latitude vs longitude) over a lunar texture map.

    Expects:
      - history["groundtrack"] with keys {"lat_deg","lon_deg"}
        OR
      - if missing, attempts to compute groundtrack using ctx via postprocess helper.
    """
    apply_rcparams(DEFAULT_STYLE)

    meta = meta or {}

    gt = _first_present(history, ["groundtrack", "ground_track", "gt"])
    if not isinstance(gt, Mapping) and ctx is not None:
        try:
            from lunaris.analysis.postprocess import _groundtrack_if_available
            t_s = _as_np(_first_present(history, ["t_s", "t", "time_s", "time"]), float, atleast_1d=True, ravel=True)
            y = _as_np(_first_present(history, ["y", "y_ns", "state", "states", "Y"]), float)
            if t_s.size and y.size:
                # compute_history stores y as (n_state, n_steps); pass through
                gt = _groundtrack_if_available(ctx, t_s, y)
        except Exception:
            gt = None

    if not isinstance(gt, Mapping):
        return _placeholder_figure(
            "Ground Track",
            "Ground track unavailable. Ensure history contains 'groundtrack' or provide a valid ctx."
        )

    lat = _as_np(gt.get("lat_deg", None), float, atleast_1d=True, ravel=True)
    lon = _as_np(gt.get("lon_deg", None), float, atleast_1d=True, ravel=True)
    n = int(min(lat.size, lon.size))
    if n < 2:
        return _placeholder_figure("Ground Track", "Insufficient ground track samples.")

    lat = lat[:n]
    lon = wrap_lon_deg(lon[:n])

    t_s = _as_np(_first_present(history, ["t_s", "t", "time_s", "time"]), float, atleast_1d=True, ravel=True)
    if t_s.size >= n:
        t_days = (t_s[:n] / 86400.0).astype(float)
    else:
        t_days = np.linspace(0.0, 1.0, n, dtype=float)

    fig = plt.figure(figsize=(12.0, 6.7))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title("Lunar Ground Track (Body-Fixed)", fontsize=14, fontweight="bold", pad=12)

    map_path = None
    for k in ("lunar_map_path", "lunar_texture_path", "texture_path"):
        v = meta.get(k, None)
        if v:
            map_path = str(v)
            break

    add_lunar_background(ax, map_path=map_path, alpha=0.6)

    time_colored_path(
        ax, lon, lat, t_days,
        linewidth=1.6,
        alpha=0.92,
        jump_deg=180.0,
        add_colorbar=True,
        cbar_label="Time [days]",
        cbar_orientation="horizontal",
    )

    c0 = None
    c1 = None
    try:
        c0 = get_series_color("start")
    except Exception:
        c0 = None
    try:
        c1 = get_series_color("end")
    except Exception:
        c1 = None

    ax.scatter(lon[0],  lat[0],  marker="^", s=85, color=c0, edgecolors="black", linewidths=0.6,
               zorder=10, label="Start")
    ax.scatter(lon[-1], lat[-1], marker="s", s=85, color=c1, edgecolors="black", linewidths=0.6,
               zorder=10, label="End")

    ax.set_xlabel("Longitude [deg]", fontweight="bold")
    ax.set_ylabel("Latitude [deg]", fontweight="bold")
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xticks(np.arange(-180, 181, 60))
    ax.set_yticks(np.arange(-90,  91, 30))
    ax.grid(True, linestyle=":", alpha=0.40)
    _legend(ax, loc="upper right")

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    return fig


def figure_orbit_3d(history: Mapping[str, Any], meta: Optional[Mapping[str, Any]] = None) -> plt.Figure:
    """
    Render a 3D trajectory view with a reference body wireframe (in kilometers).

    The plot:
      - Converts meters to km for readability
      - Draws a body wireframe (transparent) to avoid occluding the trajectory
      - Colors the trajectory by time
      - Enforces equal aspect ratio when supported
    """
    apply_rcparams(DEFAULT_STYLE)

    meta = meta or {}
    if not _as_bool(meta.get("make_3d_plots", True)):
        return _placeholder_figure("3D Orbit View", "3D plotting disabled by config.")

    r_vec_m, _ = _extract_rv_vectors(history)
    if r_vec_m.size == 0:
        return _placeholder_figure("3D Orbit View", "Position history missing.")

    r_km = np.asarray(r_vec_m, dtype=float) / 1000.0
    R_km = float(_get_body_radius_km(meta, history))

    down = int(meta.get("downsample_3d", 6) or 6)
    down = max(1, down)
    r = r_km[::down, :]

    t_s = _as_np(_first_present(history, ["t_s", "t", "time_s", "time"]), float, atleast_1d=True, ravel=True)
    if t_s.size >= r_km.shape[0]:
        t_days = (t_s[:r_km.shape[0]] / 86400.0)[::down]
    else:
        t_days = np.linspace(0.0, 1.0, r.shape[0], dtype=float)

    fig = plt.figure(figsize=(10, 8))
    fig.suptitle("3D Trajectory Visualization", y=0.95, fontsize=16, fontweight="bold")

    try:
        ax = fig.add_subplot(1, 1, 1, projection="3d")
        from mpl_toolkits.mplot3d.art3d import Line3DCollection
    except Exception:
        return _placeholder_figure("3D Orbit View", "Matplotlib 3D toolkit not available.")

    # Body wireframe
    u = np.linspace(0.0, 2.0 * np.pi, 28)
    v = np.linspace(0.0, np.pi, 14)
    xs = R_km * np.outer(np.cos(u), np.sin(v))
    ys = R_km * np.outer(np.sin(u), np.sin(v))
    zs = R_km * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(xs, ys, zs, alpha=0.14, linewidth=0.5)

    # Equator and prime meridian
    th = np.linspace(0.0, 2.0 * np.pi, 200)
    ax.plot(R_km * np.cos(th), R_km * np.sin(th), np.zeros_like(th), linewidth=1.0, linestyle="--", alpha=0.45)
    ph = np.linspace(0.0, np.pi, 200)
    ax.plot(R_km * np.sin(ph), np.zeros_like(ph), R_km * np.cos(ph), linewidth=0.9, linestyle=":", alpha=0.40)

    # Time-colored 3D line
    if r.shape[0] >= 2:
        pts = r.reshape(-1, 1, 3)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)

        tmin = float(np.nanmin(t_days))
        tmax = float(np.nanmax(t_days))
        if not np.isfinite(tmin) or not np.isfinite(tmax) or tmax <= tmin:
            tmin, tmax = 0.0, 1.0
        norm = plt.Normalize(tmin, tmax)

        lc = Line3DCollection(segs, cmap="plasma", norm=norm)
        lc.set_array(t_days[:-1])
        lc.set_linewidth(1.8)
        lc.set_alpha(0.92)
        ax.add_collection3d(lc)

        cbar = fig.colorbar(lc, ax=ax, fraction=0.03, pad=0.05, shrink=0.75)
        cbar.set_label("Time [days]")
        try:
            apply_standard_colorbar(cbar)
        except Exception:
            pass

        c0 = None
        c1 = None
        try:
            c0 = get_series_color("start")
        except Exception:
            c0 = None
        try:
            c1 = get_series_color("end")
        except Exception:
            c1 = None

        ax.scatter(r[0, 0], r[0, 1], r[0, 2], s=55, marker="^", color=c0, edgecolors="black", linewidths=0.6,
                   label="Start", zorder=10)
        ax.scatter(r[-1, 0], r[-1, 1], r[-1, 2], s=55, marker="s", color=c1, edgecolors="black", linewidths=0.6,
                   label="End", zorder=10)

    # Clean panes and set labels
    try:
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
    except Exception:
        pass

    ax.grid(True, linestyle=":", alpha=0.30)
    ax.set_xlabel("x [km]", fontweight="bold")
    ax.set_ylabel("y [km]", fontweight="bold")
    ax.set_zlabel("z [km]", fontweight="bold")

    # Robust equal-aspect handling
    lim = float(max(np.nanmax(np.abs(r)), R_km))
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass

    _legend(ax, loc="upper right")
    fig.tight_layout()
    return fig


def figure_eomega(t_days: np.ndarray, elems: Dict[str, np.ndarray]) -> plt.Figure:
    """
    Plot eccentricity vs argument of periapsis in polar phase space (e-ω),
    colored by time.

    Radius: e
    Angle: ω (radians)
    """
    apply_rcparams(DEFAULT_STYLE)

    t = _as_np(t_days, float, atleast_1d=True, ravel=True)
    e = _as_np(elems.get("e", None), float, atleast_1d=True, ravel=True)
    w_deg = _as_np(elems.get("argp_deg", None), float, atleast_1d=True, ravel=True)

    n = int(min(t.size, e.size, w_deg.size))
    if n < 2:
        return _placeholder_figure("Phase Space (e-ω)", "Insufficient data for phase plot.")

    t = t[:n]
    e = e[:n]
    w = np.radians(w_deg[:n])

    fig = plt.figure(figsize=(10, 8.5))
    ax = fig.add_subplot(1, 1, 1, projection="polar")
    ax.set_title(r"Eccentricity Phase Space ($e$-$\omega$)", fontsize=15, fontweight="bold", pad=18)

    # Build polar segments (theta, r)
    pts = np.column_stack([w, e]).reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)

    tmin = float(np.nanmin(t))
    tmax = float(np.nanmax(t))
    if not np.isfinite(tmin) or not np.isfinite(tmax) or tmax <= tmin:
        tmin, tmax = 0.0, 1.0
    norm = Normalize(vmin=tmin, vmax=tmax)

    lc = LineCollection(segs, cmap="plasma", norm=norm, linewidths=1.8, alpha=0.92)
    lc.set_array(0.5 * (t[:-1] + t[1:]))
    ax.add_collection(lc)

    ax.set_ylim(0.0, float(np.nanmax(e) * 1.08) if np.isfinite(np.nanmax(e)) else 1.0)

    cbar = fig.colorbar(lc, ax=ax, pad=0.10, fraction=0.045, shrink=0.85)
    cbar.set_label("Time [days]")
    try:
        apply_standard_colorbar(cbar)
    except Exception:
        pass

    c0 = None
    c1 = None
    try:
        c0 = get_series_color("start")
    except Exception:
        c0 = None
    try:
        c1 = get_series_color("end")
    except Exception:
        c1 = None

    ax.scatter(w[0],  e[0],  marker="^", s=60, color=c0, edgecolors="black", linewidths=0.6, label="Start", zorder=10)
    ax.scatter(w[-1], e[-1], marker="s", s=60, color=c1, edgecolors="black", linewidths=0.6, label="End", zorder=10)

    # Use consistent polar conventions
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.grid(True, alpha=0.30, linestyle=":")
    ax.set_rlabel_position(45)

    _legend(ax, loc="upper right", bbox_to_anchor=(1.25, 1.12))
    fig.tight_layout()
    return fig


def figure_perturbation_magnitude(
    history: Dict[str, Any],
    ctx: Any,
    *,
    max_points: int = 4000,
) -> plt.Figure:
    """
    Plot acceleration magnitude budget for active perturbations.

    Notes
    -----
    - This routine reconstructs magnitudes by re-building the acceleration function
      with different flag combinations, then differencing against a gravity baseline.
    - The implementation intentionally avoids legacy alias webs; it only toggles
      attributes that exist on ctx.flags. Unknown names are ignored safely.

    Requirements
    ------------
    - make_accel must be importable.
    - ctx must contain a 'flags' object with boolean attributes.
    - history must contain time and a state history (either y or r/v).

    Parameters
    ----------
    history : Dict[str, Any]
        Simulation history containing time and state vectors.
    ctx : Any
        Dynamics context used during propagation.
    max_points : int
        Downsampling cap for responsiveness.

    Returns
    -------
    matplotlib.figure.Figure
        Log-scale magnitude plot.
    """
    apply_rcparams(DEFAULT_STYLE)

    # Time and state extraction
    t_days = _as_np(extract_time_days(history), float, atleast_1d=True, ravel=True)

    y = _first_present(history, ["y", "y_ns", "state", "states", "Y"])
    if y is None:
        r = _as_np(_first_present(history, ["r", "r_m", "r_vec", "r_meters"]), float)
        v = _as_np(_first_present(history, ["v", "v_mps", "v_vec", "v_m_s"]), float)
        if r.size and v.size:
            r = np.asarray(r, dtype=float)
            v = np.asarray(v, dtype=float)
            if r.ndim == 2 and v.ndim == 2:
                y = np.hstack([r, v])
            else:
                y = None

    if y is None:
        return _placeholder_figure("Acceleration Budget", "State history not found (y or r/v).")

    y = np.asarray(y, dtype=float)

    # Accept either (n_steps, n_state) or (n_state, n_steps)
    if y.ndim != 2:
        return _placeholder_figure("Acceleration Budget", "Invalid state array shape.")
    if y.shape[1] < 6 and y.shape[0] >= 6:
        y = y.T
    if y.shape[1] < 6:
        return _placeholder_figure("Acceleration Budget", "State vectors must include at least 6 components.")

    # Work with row-wise states for evaluation: (n_steps, 6)
    if y.shape[0] == t_days.size:
        states = y[:, :6]
    elif y.shape[1] == t_days.size:
        states = y[:6, :].T
    else:
        n = int(min(t_days.size, y.shape[0]))
        states = y[:n, :6]
        t_days = t_days[:n]

    n = int(min(t_days.size, states.shape[0]))
    t_days = t_days[:n]
    states = states[:n, :6]

    if n < 2:
        return _placeholder_figure("Acceleration Budget", "Insufficient samples for budget plot.")

    # Preferred path: use the live DynamicsEngine debug/reporting hook. This is
    # the same source the runtime engine itself trusts, so it stays aligned with
    # the real active-force implementation and does not depend on legacy helper
    # factories that may not exist in newer builds.
    if hasattr(ctx, "get_acceleration_breakdown"):
        series_map: Dict[str, list[float]] = {}
        valid_t: list[float] = []
        for i in range(n):
            try:
                breakdown = ctx.get_acceleration_breakdown(float(t_days[i] * 86400.0), np.asarray(states[i, :], dtype=float))
            except Exception:
                breakdown = None

            if not isinstance(breakdown, dict) or not breakdown:
                continue

            valid_t.append(float(t_days[i]))
            for key, value in breakdown.items():
                series_map.setdefault(str(key), []).append(float(value))

            # Back-fill missing components so every series stays aligned with time.
            for key in list(series_map.keys()):
                if len(series_map[key]) < len(valid_t):
                    series_map[key].append(float("nan"))

        if valid_t and series_map:
            fig = plt.figure(figsize=(11.7, 7.0))
            ax = fig.add_subplot(1, 1, 1)
            ax.set_title("Perturbation Acceleration Budget", fontsize=14, fontweight="bold", pad=12)

            t_plot = np.asarray(valid_t, dtype=float)
            for label, values in series_map.items():
                color = None
                try:
                    color = get_accel_color(label)
                except Exception:
                    color = None
                ax.plot(t_plot, np.asarray(values, dtype=float), label=label, linewidth=1.7, color=color)

            ax.set_yscale("log")
            ax.set_ylabel(r"Acceleration magnitude [m/s$^2$]", fontweight="bold")
            ax.set_xlabel("Time [days]", fontweight="bold")
            format_log_axis_sci(ax, axis="y")
            ax.grid(True, which="major", alpha=0.35)
            ax.grid(True, which="minor", alpha=0.12, linestyle=":")
            _legend(ax, loc="upper right", ncol=2)
            fig.tight_layout()
            return fig

    if make_accel is None:
        return _placeholder_figure(
            "Acceleration Budget",
            "Dynamics kernel unavailable; cannot reconstruct acceleration components."
        )

    # Downsample for speed
    cap = int(max(2, max_points))
    if n > cap:
        idx = np.linspace(0, n - 1, cap, dtype=np.int64)
        t_days = t_days[idx]
        states = states[idx, :]

    flags = getattr(ctx, "flags", None)
    if flags is None:
        return _placeholder_figure("Acceleration Budget", "ctx.flags not available; cannot toggle perturbations.")

    def _clone_ctx_with(overrides: Dict[str, bool]):
        import copy
        new_ctx = copy.copy(ctx)
        new_flags = copy.copy(flags)
        for k, v in overrides.items():
            if hasattr(new_flags, k):
                try:
                    setattr(new_flags, k, bool(v))
                except Exception:
                    pass
        new_ctx.flags = new_flags
        return new_ctx

    def _eval_accel(accel_fn) -> np.ndarray:
        # Evaluate at each state. The time argument is typically unused by static models.
        out = np.empty((states.shape[0], 3), dtype=float)
        for i in range(states.shape[0]):
            out[i, :] = accel_fn(0.0, states[i, :])
        return out

    def _mag(a_xyz: np.ndarray) -> np.ndarray:
        return np.linalg.norm(a_xyz, axis=1)

    # Canonical flags expected in strict builds (adjust here to match your flags dataclass)
    # Unknown attributes are ignored. Keep this list short and explicit.
    groups: Dict[str, tuple[str, ...]] = {
        "spherical_harmonics": ("enable_sh",),
        "third_body_sun": ("enable_3rd_body_sun",),
        "third_body_earth": ("enable_3rd_body_earth",),
        "earth_j2": ("enable_earth_j2",),
        "srp": ("enable_srp",),
        "albedo": ("enable_albedo",),
        "thermal": ("enable_thermal", "enable_thermal_ir"),
        "tides_k2": ("enable_tides_k2",),
        "tides_k3": ("enable_tides_k3",),
        "relativity": ("enable_relativity_1pn", "enable_relativity"),
    }

    def _is_active(names: tuple[str, ...]) -> bool:
        for nm in names:
            if hasattr(flags, nm):
                try:
                    if bool(getattr(flags, nm)):
                        return True
                except Exception:
                    continue
        return False

    active = {k: _is_active(v) for k, v in groups.items()}

    # Baseline: central gravity only (all known toggles off)
    off = {nm: False for names in groups.values() for nm in names}
    a0 = _eval_accel(make_accel(_clone_ctx_with(off)))
    series: list[tuple[str, np.ndarray]] = [("Central Gravity", _mag(a0))]

    # Build an incremental gravity stack
    current = a0.copy()

    def _enable_group(base_overrides: Dict[str, bool], key: str) -> Dict[str, bool]:
        ov = dict(base_overrides)
        for nm in groups.get(key, ()):
            ov[nm] = True
        return ov

    # Spherical harmonics
    if active["spherical_harmonics"]:
        ov = _enable_group(off, "spherical_harmonics")
        a = _eval_accel(make_accel(_clone_ctx_with(ov)))
        series.append(("Spherical Harmonics", _mag(a - current)))
        current = a

    # Third bodies (Sun/Earth)
    if active["third_body_sun"] or active["third_body_earth"]:
        ov = dict(off)
        if active["spherical_harmonics"]:
            ov = _enable_group(ov, "spherical_harmonics")
        if active["third_body_sun"]:
            ov = _enable_group(ov, "third_body_sun")
        if active["third_body_earth"]:
            ov = _enable_group(ov, "third_body_earth")
        a = _eval_accel(make_accel(_clone_ctx_with(ov)))
        series.append(("Third-Body Gravity", _mag(a - current)))
        current = a

    # Earth J2
    if active["earth_j2"]:
        ov = dict(off)
        if active["spherical_harmonics"]:
            ov = _enable_group(ov, "spherical_harmonics")
        if active["third_body_sun"]:
            ov = _enable_group(ov, "third_body_sun")
        if active["third_body_earth"]:
            ov = _enable_group(ov, "third_body_earth")
        ov = _enable_group(ov, "earth_j2")
        a = _eval_accel(make_accel(_clone_ctx_with(ov)))
        series.append(("Earth J2", _mag(a - current)))
        current = a

    # Gravity reference (for non-grav differencing)
    grav_ref = current.copy()

    def _add_relative(name: str, key: str) -> None:
        ov = dict(off)
        # Keep gravity stack consistent with what was active above
        if active["spherical_harmonics"]:
            ov = _enable_group(ov, "spherical_harmonics")
        if active["third_body_sun"]:
            ov = _enable_group(ov, "third_body_sun")
        if active["third_body_earth"]:
            ov = _enable_group(ov, "third_body_earth")
        if active["earth_j2"]:
            ov = _enable_group(ov, "earth_j2")

        ov = _enable_group(ov, key)
        a = _eval_accel(make_accel(_clone_ctx_with(ov)))
        series.append((name, _mag(a - grav_ref)))

    if active["srp"]:
        _add_relative("SRP", "srp")
    if active["albedo"]:
        _add_relative("Albedo", "albedo")
    if active["thermal"]:
        # Prefer thermal_ir if present and active; otherwise enable thermal
        _add_relative("Thermal Radiation", "thermal")
    if active["tides_k2"] or active["tides_k3"]:
        # If both are active, show them separately for clarity
        if active["tides_k2"]:
            _add_relative("Solid Tides (k2)", "tides_k2")
        if active["tides_k3"]:
            _add_relative("Solid Tides (k3)", "tides_k3")
    if active["relativity"]:
        _add_relative("Relativity (1PN)", "relativity")

    # Plot
    fig = plt.figure(figsize=(11.7, 7.0))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title("Perturbation Acceleration Budget", fontsize=14, fontweight="bold", pad=12)

    for label, data in series:
        color = None
        try:
            color = get_accel_color(label)
        except Exception:
            color = None
        ax.plot(t_days, data, label=label, linewidth=1.7, color=color)

    ax.set_yscale("log")
    ax.set_ylabel(r"Acceleration magnitude [m/s$^2$]", fontweight="bold")
    ax.set_xlabel("Time [days]", fontweight="bold")
    format_log_axis_sci(ax, axis="y")

    ax.grid(True, which="major", alpha=0.35)
    ax.grid(True, which="minor", alpha=0.12, linestyle=":")

    _legend(ax, loc="upper right", ncol=2)
    fig.tight_layout()
    return fig


# =============================================================================
# 5.                  REPORT LAYOUT HELPERS (Tables & Text Blocks)
# =============================================================================

def draw_table(
    ax: plt.Axes,
    title: str,
    rows: List[List[str]],
    col_widths: Optional[List[float]] = None,
) -> None:
    """
    Render a polished striped statistics table for PDF reports.

    This later definition replaces the legacy renderer above with slightly
    better typography, ASCII-safe column headers, and stronger emphasis on the
    metric-name column.
    """

    ax.axis("off")
    ax.text(
        0.0, 1.02, title,
        transform=ax.transAxes, ha="left", va="bottom",
        fontsize=12, fontweight="bold", color=THEME["text"]
    )

    if not rows:
        ax.text(0.5, 0.5, "No data available", transform=ax.transAxes, ha="center", color="gray")
        return

    cols = ["Metric", "Start", "End", "Delta", "Min", "Max", "Unit"]
    if not col_widths or len(col_widths) != len(cols):
        col_widths = [0.24, 0.12, 0.12, 0.12, 0.12, 0.12, 0.10]

    tbl = ax.table(
        cellText=rows,
        colLabels=cols,
        cellLoc="center",
        colLoc="center",
        colWidths=col_widths,
        loc="center",
        bbox=[0.0, 0.0, 1.0, 0.94],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.2)
    tbl.scale(1.0, 1.34)

    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(THEME["grid"])
        cell.set_linewidth(0.5)
        if r == 0:
            cell.set_facecolor(THEME["table_header"])
            cell.set_text_props(fontweight="bold", color=THEME["text"])
            cell.set_height(cell.get_height() * 1.18)
        else:
            bg = THEME["table_row_even"] if (r % 2 == 0) else THEME["table_row_odd"]
            cell.set_facecolor(bg)
            if c == 0:
                cell.set_text_props(color=THEME["text"], fontweight="bold", ha="left")
            else:
                cell.set_text_props(color=THEME["text"])


def draw_kv_block(
    ax: plt.Axes,
    title: str,
    items: List[Tuple[str, str]],
    *,
    ncols: int = 1,
) -> None:
    """
    Render a clean narrative key-value block for short notes and summaries.
    """

    ax.axis("off")
    ax.text(
        0.0, 1.02, title,
        transform=ax.transAxes, ha="left", va="bottom",
        fontsize=12, fontweight="bold", color=THEME["text"]
    )

    if not items:
        return

    ncols = max(1, int(ncols))
    n_items = len(items)
    n_rows = math.ceil(n_items / ncols)
    row_height = 0.84 / max(1, n_rows)
    col_width = 1.0 / ncols
    top_margin = 0.94

    for idx, (key, val) in enumerate(items):
        row = idx % n_rows
        col = idx // n_rows
        x = col * col_width
        y = top_margin - row * row_height

        ax.text(
            x,
            y,
            f"{key}:",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9.8,
            fontweight="bold",
            color=THEME["text_dim"],
        )
        ax.text(
            x + min(0.16, col_width * 0.30),
            y,
            _wrap(str(val), width=52 if ncols == 1 else 28),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9.6,
            color=THEME["text"],
        )


def draw_kv_table(
    ax: plt.Axes,
    title: str,
    rows: List[Tuple[str, str]],
    *,
    col_widths: Tuple[float, float] = (0.47, 0.53),
) -> None:
    """
    Render a two-column metric table with report-friendly typography.
    """

    ax.axis("off")
    ax.text(
        0.0, 1.02, title,
        transform=ax.transAxes, ha="left", va="bottom",
        fontsize=12, fontweight="bold", color=THEME["text"]
    )

    if not rows:
        ax.text(0.5, 0.5, "No data available", transform=ax.transAxes, ha="center", color="gray")
        return

    normalized_rows = [(str(k), _wrap(str(v), width=46)) for k, v in rows]
    tbl = ax.table(
        cellText=normalized_rows,
        colLabels=["Metric", "Value"],
        cellLoc="left",
        colLoc="left",
        colWidths=col_widths,
        loc="upper left",
        bbox=[0.0, 0.0, 1.0, 0.94],
    )

    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.4)
    tbl.scale(1.0, 1.46)

    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(THEME["grid"])
        cell.set_linewidth(0.6)
        if r == 0:
            cell.set_facecolor(THEME["table_header"])
            cell.set_text_props(fontweight="bold", color=THEME["text"])
        else:
            bg = THEME["table_row_even"] if (r % 2 == 0) else THEME["table_row_odd"]
            cell.set_facecolor(bg)
            if c == 0:
                cell.set_text_props(color=THEME["text"], fontweight="bold")
            else:
                cell.set_text_props(color=THEME["text"])



# =============================================================================
# 6.                      CONFIG & PHYSICS DISCOVERY HELPERS
# =============================================================================

def _normalize_key(k: str) -> str:
    """Normalize a string key for matching: lowercase + remove non-alphanumerics."""
    if not k:
        return ""
    return re.sub(r"[^a-z0-9]", "", str(k).lower())


def _flatten_to_lookup(data: Any, parent: str = "", sep: str = ".") -> Dict[str, Any]:
    """
    Flatten a nested mapping into a lookup table keyed by normalized full paths.

    Only flattens mappings; lists/tuples are not expanded to avoid noisy keys.

    Example:
        {"physics": {"flags": {"enable_srp": True}}}
    ->  {"physics.flags.enable_srp": True}  (stored normalized internally)
    """
    out: Dict[str, Any] = {}
    if not isinstance(data, Mapping):
        return out

    stack: list[tuple[str, Mapping[str, Any]]] = [(parent, data)]
    seen = 0
    max_nodes = 4000  # hard cap: prevents pathological configs from blowing up

    while stack and seen < max_nodes:
        cur_parent, cur = stack.pop()
        seen += 1

        for k, v in cur.items():
            k_str = str(k)
            full = f"{cur_parent}{sep}{k_str}" if cur_parent else k_str
            if isinstance(v, Mapping):
                stack.append((full, v))
            else:
                out[_normalize_key(full)] = v

    return out


def effects_from_meta_history(
    meta: Mapping[str, Any],
    history: Mapping[str, Any],
    *,
    ctx: Any = None,
) -> Tuple[Dict[str, bool], str]:
    """
    Determine which physical effects were active (for report labeling only).

    Strict strategy (preferred):
      1) Find a dedicated flags mapping (meta has priority over history).
      2) Optionally inspect ctx.flags if provided (strict attribute names only).
      3) Conservative fallback: flattened lookup restricted to enable/flags keys.

    This function is intentionally conservative: false positives are worse than false negatives.
    """
    effects: Dict[str, bool] = {
        "Spherical Harmonics": False,
        "Third-Body Sun": False,
        "Third-Body Earth": False,
        "Earth J2": False,
        "Solar Radiation Pressure": False,
        "Lunar Albedo": False,
        "Lunar Thermal IR": False,
        "Solid Tides (k2)": False,
        "Solid Tides (k3)": False,
        "General Relativity": False,
    }

    # ----------------------------
    # Canonical strict keys
    # ----------------------------
    # Keep these aligned with your strict Flags dataclass field names.
    CANON = {
        "sh": ("enable_sh",),
        "third_sun": ("enable_3rd_body_sun",),
        "third_earth": ("enable_3rd_body_earth",),
        "earth_j2": ("enable_earth_j2",),
        "srp": ("enable_srp",),
        "albedo": ("enable_albedo",),
        "thermal": ("enable_thermal", "enable_thermal_ir"),
        "tides": ("enable_tides_k2", "enable_tides_k3"),
        "gr": ("enable_relativity_1pn", "enable_relativity"),
    }

    def _iter_flag_items(flags: Mapping[str, Any]) -> Dict[str, Any]:
        # Normalize keys once for stable matching.
        return {_normalize_key(str(k)): v for k, v in flags.items()}

    def _bool_on_in_dict(norm_flags: Mapping[str, Any], keys: Iterable[str]) -> bool:
        for k in keys:
            nk = _normalize_key(k)
            if nk in norm_flags and _as_bool(norm_flags[nk]):
                return True
        return False

    def _find_flags_dict(obj: Any) -> Tuple[Optional[Mapping[str, Any]], Optional[str]]:
        """
        Breadth-first search for a mapping named 'flags'.
        Returns (flags_mapping, path) if found, else (None, None).
        """
        if not isinstance(obj, Mapping):
            return None, None

        queue: list[tuple[str, Any]] = [("", obj)]
        visited = 0
        max_visits = 2500

        while queue and visited < max_visits:
            path, cur = queue.pop(0)
            visited += 1

            if not isinstance(cur, Mapping):
                continue

            for k, v in cur.items():
                k_str = str(k)
                new_path = f"{path}.{k_str}" if path else k_str

                if k_str.lower() == "flags" and isinstance(v, Mapping):
                    return v, new_path

                if isinstance(v, Mapping):
                    queue.append((new_path, v))

        return None, None

    def _apply_from_flags_dict(flags_dict: Mapping[str, Any]) -> None:
        norm_flags = _iter_flag_items(flags_dict)

        effects["Spherical Harmonics"] = _bool_on_in_dict(norm_flags, CANON["sh"])
        effects["Third-Body Sun"] = _bool_on_in_dict(norm_flags, CANON["third_sun"])
        effects["Third-Body Earth"] = _bool_on_in_dict(norm_flags, CANON["third_earth"])
        effects["Earth J2"] = _bool_on_in_dict(norm_flags, CANON["earth_j2"])
        effects["Solar Radiation Pressure"] = _bool_on_in_dict(norm_flags, CANON["srp"])
        effects["Lunar Albedo"] = _bool_on_in_dict(norm_flags, CANON["albedo"])
        effects["Lunar Thermal IR"] = _bool_on_in_dict(norm_flags, CANON["thermal"])

        effects["Solid Tides (k2)"] = _bool_on_in_dict(norm_flags, ("enable_tides_k2",))
        effects["Solid Tides (k3)"] = _bool_on_in_dict(norm_flags, ("enable_tides_k3",))
        effects["General Relativity"] = _bool_on_in_dict(norm_flags, CANON["gr"])

    # ----------------------------
    # 1) Direct flags dict (best)
    # ----------------------------
    meta = meta or {}
    history = history or {}

    flags_meta, p_meta = _find_flags_dict(meta)
    if flags_meta:
        _apply_from_flags_dict(flags_meta)
        return effects, f"meta:{p_meta or 'flags'}"

    # Common history nesting places to try (strict list, not fuzzy).
    for root_key in ("config", "cfg", "settings", "options"):
        if isinstance(history, Mapping) and isinstance(history.get(root_key), Mapping):
            flags_hist, p_hist = _find_flags_dict(history.get(root_key))
            if flags_hist:
                _apply_from_flags_dict(flags_hist)
                return effects, f"history.{root_key}:{p_hist or 'flags'}"

    # Direct history.flags (if the structure is {flags: {...}})
    if isinstance(history, Mapping) and isinstance(history.get("flags"), Mapping):
        _apply_from_flags_dict(history["flags"])
        return effects, "history:flags"

    # ----------------------------
    # 2) Optional ctx.flags (strict attributes only)
    # ----------------------------
    if ctx is not None:
        flags_obj = getattr(ctx, "flags", None)
        if flags_obj is not None:
            def _attr_on(*names: str) -> bool:
                for name in names:
                    if hasattr(flags_obj, name):
                        try:
                            return bool(getattr(flags_obj, name))
                        except Exception:
                            continue
                return False

            effects["Spherical Harmonics"] = _attr_on(*CANON["sh"])
            effects["Third-Body Sun"] = _attr_on(*CANON["third_sun"])
            effects["Third-Body Earth"] = _attr_on(*CANON["third_earth"])
            effects["Earth J2"] = _attr_on(*CANON["earth_j2"])
            effects["Solar Radiation Pressure"] = _attr_on(*CANON["srp"])
            effects["Lunar Albedo"] = _attr_on(*CANON["albedo"])
            effects["Lunar Thermal IR"] = _attr_on(*CANON["thermal"])
            effects["Solid Tides (k2)"] = _attr_on("enable_tides_k2")
            effects["Solid Tides (k3)"] = _attr_on("enable_tides_k3")
            effects["General Relativity"] = _attr_on(*CANON["gr"])

            return effects, "ctx.flags"

    # ----------------------------
    # 3) Conservative flattened search (last resort)
    # ----------------------------
    lookup: Dict[str, Any] = {}

    # Only flatten well-scoped roots to reduce noise.
    for root_key in ("config", "cfg", "settings", "options", "flags"):
        if isinstance(history, Mapping) and isinstance(history.get(root_key), Mapping):
            lookup.update(_flatten_to_lookup(history[root_key], parent=root_key))

    lookup.update(_flatten_to_lookup(meta, parent="meta"))

    def _lookup_on(*needles: str) -> bool:
        needles_n = tuple(_normalize_key(n) for n in needles if n)
        if not needles_n:
            return False

        for flat_k, v in lookup.items():
            if not _as_bool(v):
                continue

            # Strict gating: key must look like a boolean toggle path.
            # This prevents accidental matches like "groundtrack" -> "gr".
            if ("enable" not in flat_k) and ("flags" not in flat_k):
                continue

            if any(n in flat_k for n in needles_n):
                return True

        return False

    # Match using strict canonical names only (no generic sun/earth/gr tokens).
    effects["Spherical Harmonics"] = _lookup_on("enable_sh")
    effects["Third-Body Sun"] = _lookup_on("enable_3rd_body_sun")
    effects["Third-Body Earth"] = _lookup_on("enable_3rd_body_earth")
    effects["Earth J2"] = _lookup_on("enable_earth_j2")
    effects["Solar Radiation Pressure"] = _lookup_on("enable_srp")
    effects["Lunar Albedo"] = _lookup_on("enable_albedo")
    effects["Lunar Thermal IR"] = _lookup_on("enable_thermal", "enable_thermal_ir")
    effects["Solid Tides (k2)"] = _lookup_on("enable_tides_k2")
    effects["Solid Tides (k3)"] = _lookup_on("enable_tides_k3")
    effects["General Relativity"] = _lookup_on("enable_relativity_1pn", "enable_relativity")

    return effects, "Auto-Detected"



# =============================================================================
# 7.                     CONFIG AUTO-DISCOVERY & MERGING
# =============================================================================

def _try_load_nearby_config(search_dir: str) -> Tuple[Optional[dict], Optional[str]]:
    """
    Scan a directory for a likely ST_LRPS run configuration (JSON).

    The scoring is conservative: false positives are worse than false negatives.
    We prioritize small JSON files whose *structure* resembles a run config
    (flags/time/propagator/spacecraft), and de-prioritize obvious data dumps.
    """
    if not search_dir or not os.path.isdir(search_dir):
        return None, None

    # Candidate file collection (fast I/O; avoid large JSON dumps)
    candidates: List[str] = []
    try:
        for fn in os.listdir(search_dir):
            if not fn.lower().endswith(".json"):
                continue
            path = os.path.join(search_dir, fn)
            try:
                if os.path.getsize(path) <= 1_000_000:  # configs are typically tiny
                    candidates.append(path)
            except Exception:
                continue
    except Exception:
        return None, None

    if not candidates:
        return None, None

    # Name-based priors
    preferred_names = (
        "config.json",
        "run_config.json",
        "run_settings.json",
        "settings.json",
        "options.json",
        "preset.json",
    )
    reject_tokens = (
        "history",
        "metrics",
        "results",
        "trajectory",
        "states",
        "telemetry",
        "log",
        "report",
        "cache",
    )

    def _name_score(path: str) -> int:
        fn = os.path.basename(path).lower()
        if fn in preferred_names:
            return 200
        s = 0
        # Soft preference: contains config-like tokens
        if any(t in fn for t in ("config", "settings", "options", "preset", "run")):
            s += 60
        # Hard penalty: looks like a dump
        if any(t in fn for t in reject_tokens):
            s -= 250
        return s

    def _content_score(cfg: Mapping[str, Any]) -> int:
        """
        Score a dict based on config-shaped markers. Conservative by design.
        """
        flat = set(_flatten_to_lookup(cfg).keys())

        score = 0

        # Strong signals: flags + propagator/time/spacecraft are typical SSOT blocks
        if any("flags" in k for k in flat):
            score += 80
        if any("spacecraft" in k for k in flat):
            score += 60
        if any("propagator" in k for k in flat):
            score += 60
        if any("time" in k for k in flat):
            score += 40
        if any("gravity" in k for k in flat):
            score += 25

        # Additional hints (lower weight)
        if any("integrator" in k for k in flat) or any("method" in k for k in flat):
            score += 15

        # Negative signals: looks like data rather than config
        if any("history" in k for k in flat) or any("telemetry" in k for k in flat):
            score -= 80
        if any("states" in k for k in flat) or any("trajectory" in k for k in flat):
            score -= 80

        return score

    best_cfg: Optional[dict] = None
    best_path: Optional[str] = None
    best_score = -10**9

    # Deterministic iteration: sort file names so ties resolve consistently
    for path in sorted(candidates, key=lambda p: os.path.basename(p).lower()):
        ns = _name_score(path)

        # Quick reject based on filename alone
        if ns < -200:
            continue

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        if not isinstance(data, dict) or not data:
            continue

        cs = _content_score(data)
        score = ns + cs

        # Conservative threshold: require at least one strong marker
        # i.e., score must be meaningfully positive.
        if score > best_score and score >= 80:
            best_score = score
            best_cfg = data
            best_path = path

    return best_cfg, best_path


def merge_meta_with_auto_config(
    meta: Mapping[str, Any],
    history: Mapping[str, Any],
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Merge explicit runtime metadata with an auto-discovered on-disk config.

    Merge rules (deterministic):
      1) Start from disk config (if found).
      2) Overlay explicit meta (explicit always wins).
      3) Fill *missing* spacecraft fields from history (as a last resort).
    """
    merged: Dict[str, Any] = {}

    # 1) Disk config base (optional)
    cfg_path = None
    if output_dir:
        cfg_disk, cfg_path = _try_load_nearby_config(output_dir)
        if isinstance(cfg_disk, dict) and cfg_disk:
            merged.update(cfg_disk)

    # 2) Explicit meta overrides
    if isinstance(meta, Mapping) and meta:
        merged.update(dict(meta))

    if cfg_path:
        try:
            merged["_source_file"] = os.path.basename(cfg_path)
        except Exception:
            merged["_source_file"] = str(cfg_path)

    # 3) Normalize spacecraft container
    sc = merged.get("spacecraft")
    if not isinstance(sc, Mapping):
        sc = {}
        merged["spacecraft"] = sc
    else:
        sc = dict(sc)
        merged["spacecraft"] = sc

    # 4) Fill missing spacecraft properties from history (strictly optional)
    # Only set if missing; never override explicit meta/disk values.
    if "mass_kg" not in sc:
        val = _first_present(history, ["mass_kg", "mass", "m"])
        if val is not None:
            try:
                sc["mass_kg"] = float(_as_np(val).ravel()[0])
            except Exception:
                pass

    if "area_m2" not in sc:
        val = _first_present(history, ["area_m2", "area", "cross_section"])
        if val is not None:
            try:
                sc["area_m2"] = float(_as_np(val).ravel()[0])
            except Exception:
                pass

    return merged



# =============================================================================
# 8.                    FINAL UTILITIES & METRICS GENERATORS
# =============================================================================

def _placeholder_figure(title: str, message: str) -> plt.Figure:
    """
    Create a consistent placeholder figure when a plot cannot be generated.
    """
    fig = plt.figure(figsize=(11.7, 8.3))
    ax = fig.add_subplot(1, 1, 1)
    ax.axis("off")

    # Keep styling simple and robust even if theme helpers are not available.
    title_color = "#EF4444"
    text_color = "#7F1D1D"
    face = "#FEE2E2"
    edge = "#EF4444"

    fig.suptitle(title, y=0.96, fontsize=16, fontweight="bold", color=title_color)

    bbox_props = dict(boxstyle="round,pad=0.8", fc=face, ec=edge, lw=1.4)
    ax.text(
        0.5, 0.5, str(message),
        ha="center", va="center",
        transform=ax.transAxes,
        fontsize=12,
        color=text_color,
        bbox=bbox_props,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def _make_row(name: str, arr: np.ndarray, unit: str, nd: int, sci: bool = False) -> List[str]:
    """
    Compute (start, end, delta, min, max) summary for a series and return formatted row cells.
    """
    x = _as_np(arr).astype(float).ravel()
    st = _nan_stats(x)

    start = st["start"]
    end = st["end"]
    delta = (end - start) if (np.isfinite(start) and np.isfinite(end)) else float("nan")

    fmt = _fmt_sci if bool(sci) else _fmt_fixed
    return [
        str(name),
        fmt(start, nd),
        fmt(end, nd),
        fmt(delta, nd),
        fmt(st["min"], nd),
        fmt(st["max"], nd),
        str(unit),
    ]


# =============================================================================
# 9.                           EVENTS SUMMARY TABLE
# =============================================================================

def metrics_rows(
    history: Mapping[str, Any],
    elems: Dict[str, np.ndarray],
) -> Tuple[List[List[str]], List[List[str]]]:
    """
    Build report-table rows using ASCII-safe, reader-friendly metric labels.

    This later definition intentionally replaces the legacy labels above so PDF
    exports remain robust even when special-glyph font coverage is limited.
    """

    a = _as_np(elems.get("a_km", [])).astype(float)
    e = _as_np(elems.get("e", [])).astype(float)
    inc = _as_np(elems.get("i_deg", [])).astype(float)
    raan = _as_np(elems.get("raan_deg", [])).astype(float)
    argp = _as_np(elems.get("argp_deg", [])).astype(float)

    orbital_rows: List[List[str]] = []
    if a.size:
        orbital_rows.append(_make_row("Semi-major axis a", a, "km", 3))
    if e.size:
        orbital_rows.append(_make_row("Eccentricity e", e, "-", 6))
    if inc.size:
        orbital_rows.append(_make_row("Inclination i", inc, "deg", 4))
    if raan.size:
        orbital_rows.append(_make_row("RAAN", raan, "deg", 4))
    if argp.size:
        orbital_rows.append(_make_row("Argument of periapsis", argp, "deg", 4))

    if a.size and e.size:
        n = int(min(a.size, e.size))
        if n >= 1:
            aa = a[:n]
            ee = e[:n]
            orbital_rows.append(_make_row("Periapsis radius rp", aa * (1.0 - ee), "km", 3))
            orbital_rows.append(_make_row("Apoapsis radius ra", aa * (1.0 + ee), "km", 3))

    inv = extract_invariants(history)
    inv_rows: List[List[str]] = []
    r = _as_np(inv.get("r_norm_km", []))
    if r.size:
        inv_rows.append(_make_row("Position norm |r|", r, "km", 3))

    v = _as_np(inv.get("v_norm_kmps", []))
    if v.size:
        inv_rows.append(_make_row("Velocity norm |v|", v, "km/s", 5))

    energy = _as_np(inv.get("energy_Jkg", []))
    if energy.size:
        inv_rows.append(_make_row("Specific energy", energy, "J/kg", 4, sci=True))

    h_norm = _as_np(inv.get("h_norm_m2s", []))
    if h_norm.size:
        inv_rows.append(_make_row("Angular momentum |h|", h_norm, "m^2/s", 4, sci=True))

    return orbital_rows, inv_rows


def figure_events_table(events: Mapping[str, Any], t_days: np.ndarray) -> plt.Figure:
    """
    Build a compact summary table for detected orbital events (periapsis, apoapsis, impact).

    The function is intentionally defensive: it tolerates missing keys, mixed index types
    (list/array/scalar), and out-of-range indices without crashing the report.
    """
    apply_rcparams(DEFAULT_STYLE)

    t_days = _as_np(t_days).astype(float).ravel()

    def _sanitize_indices(x: Any) -> np.ndarray:
        """Return sorted, unique, in-range integer indices."""
        idx = _as_np(x).astype(int).ravel() if x is not None else np.array([], dtype=int)
        if idx.size == 0 or t_days.size == 0:
            return np.array([], dtype=int)

        # Remove obviously invalid values and keep in bounds
        idx = idx[np.isfinite(idx)]
        idx = idx[(idx >= 0) & (idx < t_days.size)]
        if idx.size == 0:
            return np.array([], dtype=int)

        # Sort and unique (stable summary and consistent first/last)
        idx = np.unique(np.sort(idx))
        return idx

    def _sanitize_impact(x: Any) -> Optional[int]:
        """Return a single valid impact index if present."""
        if x is None:
            return None
        try:
            # Support scalar, 0-d array, or 1-element list/array
            arr = _as_np(x).ravel()
            if arr.size == 0:
                return None
            ii = int(arr[0])
        except Exception:
            try:
                ii = int(x)
            except Exception:
                return None

        if t_days.size == 0 or not (0 <= ii < t_days.size):
            return None
        return ii

    peri = _sanitize_indices(events.get("peri_idx", None))
    apo = _sanitize_indices(events.get("apo_idx", None))
    impact_idx = _sanitize_impact(events.get("impact_idx", None))

    def _fmt_day(x: float) -> str:
        if not np.isfinite(x):
            return "—"
        return f"{x:.6f}"

    def _fmt_sec_from_day(x: float) -> str:
        if not np.isfinite(x):
            return "—"
        sec = float(x) * 86400.0
        if not np.isfinite(sec):
            return "—"
        # Use fixed notation unless very large
        return _fmt_fixed(sec, 2) if abs(sec) < 1e7 else _fmt_sci(sec, 3)

    def _time_cells(idx: int) -> Tuple[str, str]:
        """(days, seconds) strings for a given index."""
        if not (0 <= idx < t_days.size):
            return "—", "—"
        d = float(t_days[idx])
        return _fmt_day(d), _fmt_sec_from_day(d)

    def _span_days(indices: np.ndarray) -> str:
        if indices.size < 2:
            return "—"
        d0 = float(t_days[indices[0]])
        d1 = float(t_days[indices[-1]])
        if not (np.isfinite(d0) and np.isfinite(d1)):
            return "—"
        return _fmt_fixed(d1 - d0, 6)

    rows: List[Tuple[str, str]] = []

    # Metadata / detection flags (if provided)
    detect_peri_apo = events.get("detect_peri_apo", None)
    detect_impact = events.get("detect_impact", None)
    impact_alt_km = events.get("impact_alt_km", None)

    if detect_peri_apo is not None:
        rows.append(("Peri/Apo Detection", "ON" if _as_bool(detect_peri_apo) else "OFF"))
    if detect_impact is not None:
        rows.append(("Impact Detection", "ON" if _as_bool(detect_impact) else "OFF"))
    if impact_alt_km is not None:
        try:
            rows.append(("Impact Altitude Threshold [km]", _fmt_fixed(float(impact_alt_km), 3)))
        except Exception:
            rows.append(("Impact Altitude Threshold [km]", "—"))

    # Periapsis
    rows.append(("Periapsis Count", str(int(peri.size))))
    if peri.size > 0:
        d_first, s_first = _time_cells(int(peri[0]))
        d_last, s_last = _time_cells(int(peri[-1]))
        rows.append(("First Periapsis Time [days]", d_first))
        rows.append(("First Periapsis Time [s]", s_first))
        rows.append(("Last Periapsis Time [days]", d_last))
        rows.append(("Last Periapsis Time [s]", s_last))
        rows.append(("Periapsis Time Span [days]", _span_days(peri)))

    # Apoapsis
    rows.append(("Apoapsis Count", str(int(apo.size))))
    if apo.size > 0:
        d_first, s_first = _time_cells(int(apo[0]))
        d_last, s_last = _time_cells(int(apo[-1]))
        rows.append(("First Apoapsis Time [days]", d_first))
        rows.append(("First Apoapsis Time [s]", s_first))
        rows.append(("Last Apoapsis Time [days]", d_last))
        rows.append(("Last Apoapsis Time [s]", s_last))
        rows.append(("Apoapsis Time Span [days]", _span_days(apo)))

    # Impact
    if impact_idx is not None:
        d_imp, s_imp = _time_cells(int(impact_idx))
        rows.append(("Impact Detected", "YES"))
        rows.append(("Impact Time [days]", d_imp))
        rows.append(("Impact Time [s]", s_imp))
    else:
        rows.append(("Impact Detected", "NO"))

    fig = plt.figure(figsize=(11.7, 8.3))
    ax = fig.add_subplot(1, 1, 1)

    draw_kv_table(
        ax,
        title="Orbital Events Summary",
        rows=rows,
        col_widths=(0.55, 0.45),
    )

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


# =============================================================================
# 10.                           PUBLIC API
# =============================================================================

__all__ = [
    "figure_elements_timeseries",
    "figure_invariants",
    "figure_altitude_with_events",
    "figure_relative_drift",
    "figure_ground_track",
    "figure_orbit_3d",
    "figure_eomega",
    "figure_perturbation_magnitude",
    "figure_events_table",
    "draw_table",
    "draw_kv_table",
    "draw_kv_block",
    "metrics_rows",
    "effects_from_meta_history",
    "merge_meta_with_auto_config",
]
