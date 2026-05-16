# LUNAR_SIMULATION/analysis/styling.py
"""
LunarSim Plot Styling (analysis.styling)
======================================

This module is the single, project-wide source of truth for *all* plot aesthetics in LunarSim.
It is designed for three goals:

1) Consistency
   Every figure produced by the project should look like it belongs to the same report/paper:
   same typography, same grid/spine treatment, same legend framing, same semantic colors.

2) Publication readiness
   Defaults are tuned for PDF export and white backgrounds:
   subtle grids, non-neon colors, readable text, and reliable layout during savefig.

3) Maintainability
   Styling logic is centralized and intentionally small:
   - palette definitions are descriptive (no magic numeric indices)
   - semantic mapping is stable (a physical meaning always has the same color)
   - helpers are lightweight and backend-safe (works in notebooks, scripts, CI)

Quick start
-----------
In any analysis script:

    import matplotlib.pyplot as plt
    from analysis.styling import apply_rcparams, apply_axes_style, apply_standard_legend

    apply_rcparams()  # call once at program start

    fig, ax = plt.subplots()
    apply_axes_style(ax, title="Altitude vs Time")

    ax.plot(t, alt_km, label="Altitude")
    apply_standard_legend(ax)

Core concepts
-------------
1) Palette primitives (COLORS)
   - Neutrals: paper-like background, ink, muted text, subtle grid, table fills.
   - Accents: a categorical palette for line series (print-friendly).

2) Theme (THEME)
   - High-level UI/layout colors derived from neutrals (axes bg, page bg, grid, text).

3) Semantic mapping (SEMANTIC)
   - Domain concepts -> colors (orbit / spherical harmonics / SRP / third-body / albedo / relativity).
   - These should be stable across the entire project. If you change a semantic color,
     you are changing the "visual language" of LunarSim.

4) Series colors (SERIES_COLORS)
   - Canonical time series keys (altitude, rp/ra, drift metrics, trajectories, events) -> colors.
   - The public helper `get_series_color()` uses this mapping first, and falls back to a
     deterministic color choice for unknown series names.

5) Matplotlib integration
   - `apply_rcparams()` configures global rcParams (fonts, grids, spines, legend style, line styles).
   - `apply_axes_style()` provides per-Axes finishing touches (minor grid, spine styling, ticks).
   - Legend, colorbar, and numeric formatter helpers standardize "small details".

Assets and backgrounds
----------------------
The module optionally supports drawing a lunar texture behind plots (e.g., ground tracks).
If a texture is not available, background helpers are safe no-ops.

Canonical asset location:
    LUNAR_SIMULATION/data/assets/

Environment overrides (optional):
    LUNARSIM_LUNAR_MAP   -> explicit image file path
    LUNARSIM_ASSETS_DIR  -> directory containing textures

Public API overview
-------------------
- Palettes / theme:
    COLORS, COLOR_CYCLE, THEME, SEMANTIC, SERIES_COLORS, FIGURE_SIZES, LABEL_UNIT

- Global styling:
    PlotStyle, DEFAULT_STYLE, setup_global_style(), apply_rcparams()

- Axes styling & formatters:
    apply_axes_style(), format_scientific_axis(), format_log_axis_sci(), apply_standard_colorbar()

- Legend helpers:
    apply_legend_style(), apply_standard_legend()

- Assets:
    load_lunar_map(), add_lunar_background()

- Label helpers:
    get_unit_label(), get_pretty_label(), make_axis_label()

- Color helpers:
    get_accel_color(), get_series_color()

Guidelines for contributors
---------------------------
- Prefer semantic keys over raw hex codes:
      use SEMANTIC["spherical_harmonics"] rather than "#B07D28"

- Keep synonyms/aliases out of this module:
  normalize series names at the call site (strict, predictable keys).

- If you add new plots:
  (a) add a canonical key to SERIES_COLORS if it will be reused
  (b) add a LABEL_UNIT entry if it will appear as an axis label
  (c) prefer get_series_color(...) over choosing colors ad-hoc

- Keep helpers lightweight:
  avoid side effects beyond styling; avoid importing heavy project modules here.

"""

# The rest of the module defines:
# - Palette primitives (COLORS, COLOR_CYCLE)
# - Theme maps (THEME, SEMANTIC, SERIES_COLORS)
# - Matplotlib rcParams helpers and axes-level styling
# - Legend utilities and formatters
# - Optional background texture helpers
# - Labels/units + deterministic color selection



# =============================================================================
# 0.                                 IMPORTS
# =============================================================================

from __future__ import annotations

import os
import hashlib
from pathlib import Path
from typing import Optional, Tuple, Any, Mapping
from types import MappingProxyType
import numpy as np
import matplotlib.axes
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.ticker import ScalarFormatter, LogLocator, LogFormatterSciNotation
from dataclasses import dataclass

from loaders.io_helpers import (
    find_lunar_map_path as _find_lunar_map_path_from_loader,
    iter_lunar_map_candidates as _iter_lunar_map_candidates_from_loader,
    project_root_from_path as _project_root_from_path_loader,
)



# =============================================================================
# 1.                          CONSTANTS & PALETTES
# =============================================================================

# ----------------------------
# Core color primitives
# ----------------------------
# Notes:
# - Keep hex codes centralized.
# - Use read-only mapping proxies to prevent accidental mutation at runtime.
_COLORS = {
    # Neutrals (paper-like)
    "neutral": {
        "page":        "#F7F7F5",  # Warm off-white page background
        "axes":        "#FFFFFF",  # Plot interior
        "grid":        "#D9D9D6",  # Subtle gridlines
        "ink":         "#111827",  # Primary text (near-black)
        "ink_muted":   "#4B5563",  # Secondary text (units, minor labels)
        "table_head":  "#F2F2F0",  # Table header fill
        "row_odd":     "#FAFAF8",  # Table zebra stripe (odd rows)
        "row_even":    "#FFFFFF",  # Table zebra stripe (even rows)
    },

    # Categorical accents (tab10-like, readable on white)
    "accent": {
        "blue":   "#1F77B4",
        "red":    "#D62728",
        "ochre":  "#B07D28",
        "green":  "#2CA02C",
        "gray":   "#7F7F7F",
        "navy":   "#0F2E4D",
        "orange": "#FF7F0E",
        "teal":   "#17A2A4",
        "purple": "#9467BD",
        "brown":  "#8C564B",

        # Event helpers (distinct from base red/orange families)
        "magenta": "#CC79A7",  # colorblind-friendlier than rose for "end"
        "maroon":  "#7A1E3A",  # high-salience impact marker
    },
}
COLORS: Mapping[str, Mapping[str, str]] = MappingProxyType(
    {k: MappingProxyType(v) for k, v in _COLORS.items()}
)

# Ordered accent cycle for Matplotlib (stable and readable on white)
# Example: ax.set_prop_cycle(color=COLOR_CYCLE)
COLOR_CYCLE: Tuple[str, ...] = (
    COLORS["accent"]["blue"],
    COLORS["accent"]["red"],
    COLORS["accent"]["ochre"],
    COLORS["accent"]["green"],
    COLORS["accent"]["gray"],
    COLORS["accent"]["navy"],
    COLORS["accent"]["orange"],
    COLORS["accent"]["teal"],
    COLORS["accent"]["purple"],
    COLORS["accent"]["brown"],
)


# ----------------------------
# Theme-level styling (what the UI/figures use)
# ----------------------------
_THEME = {
    # Backgrounds / layout
    "bg_page":   COLORS["neutral"]["page"],
    "bg_axes":   COLORS["neutral"]["axes"],
    "grid":      COLORS["neutral"]["grid"],
    "text":      COLORS["neutral"]["ink"],
    "text_dim":  COLORS["neutral"]["ink_muted"],

    # Tables
    "table_header":    COLORS["neutral"]["table_head"],
    "table_row_odd":   COLORS["neutral"]["row_odd"],
    "table_row_even":  COLORS["neutral"]["row_even"],
    "table_text":      COLORS["neutral"]["ink"],
}
THEME: Mapping[str, str] = MappingProxyType(_THEME)


# ----------------------------
# Semantic colors (domain concepts)
# Keep these stable: figures become instantly interpretable.
# ----------------------------
# Changes vs your original:
# - event_end moved away from the red family (to magenta) to avoid confusion with SRP/apoapsis.
# - event_impact uses explicit maroon accent for maximum salience.
_SEMANTIC = {
    # Physics / model components
    "orbit":                    COLORS["accent"]["blue"],
    "spherical_harmonics":      COLORS["accent"]["ochre"],   # gravity anomalies / SH terms
    "solar_radiation_pressure": COLORS["accent"]["red"],
    "third_body_gravity":       COLORS["accent"]["brown"],
    "albedo":                   COLORS["accent"]["green"],
    "relativity":               COLORS["accent"]["purple"],

    # Events (markers should be salient but not neon)
    "event_start":              COLORS["accent"]["teal"],
    "event_end":                COLORS["accent"]["magenta"],
    "event_impact":             COLORS["accent"]["maroon"],
}
SEMANTIC: Mapping[str, str] = MappingProxyType(_SEMANTIC)


# ----------------------------
# Series -> colors (canonical keys only)
# If you want multiple names for the same concept, normalize input keys before lookup.
# ----------------------------
_SERIES_COLORS = {
    # Orbit geometry / state
    "altitude":      SEMANTIC["orbit"],
    "periapsis":     COLORS["accent"]["ochre"],
    "apoapsis":      COLORS["accent"]["red"],
    "rp":            COLORS["accent"]["ochre"],
    "ra":            COLORS["accent"]["red"],

    # Tracks / trajectories
    "ground_track":  SEMANTIC["orbit"],
    "trajectory_3d": SEMANTIC["orbit"],

    # Invariants / diagnostics
    "energy_drift":  COLORS["accent"]["gray"],
    "angmom_drift":  COLORS["accent"]["navy"],
    "e_omega_traj":  COLORS["accent"]["orange"],

    # Event-style series (if you plot events as points/lines in a legend)
    "start_event":   SEMANTIC["event_start"],
    "end_event":     SEMANTIC["event_end"],
    "impact_event":  SEMANTIC["event_impact"],
}
SERIES_COLORS: Mapping[str, str] = MappingProxyType(_SERIES_COLORS)


# Optional: canonicalization helper for “synonyms elsewhere”
# (Use this only if you actually have mixed keys coming into plotting.)
_SERIES_SYNONYMS = {
    "alt": "altitude",
    "alt_km": "altitude",
    "h": "altitude",

    "rp_km": "rp",
    "ra_km": "ra",

    "perigee": "periapsis",
    "apogee": "apoapsis",
}
SERIES_SYNONYMS: Mapping[str, str] = MappingProxyType(_SERIES_SYNONYMS)


def normalize_series_key(key: str) -> str:
    """Normalize incoming series keys to the canonical ones used in SERIES_COLORS."""
    k = (key or "").strip().lower()
    return SERIES_SYNONYMS.get(k, k)


def series_color(key: str, default: str | None = None) -> str:
    """Safe color lookup with normalization + fallback."""
    k = normalize_series_key(key)
    return SERIES_COLORS.get(k, default or THEME["text"])


# ----------------------------
# Common figure size presets (inches)
# Choose stable presets to prevent layout drift between runs/figures.
# ----------------------------
_FIGURE_SIZES = {
    "a4_landscape": (11.7, 8.3),
    "a4_portrait":  (8.3, 11.7),
    "standard":     (11.2, 7.4),
    "wide":         (12.0, 6.7),
    "wide_compact": (11.7, 6.6),
}
FIGURE_SIZES: Mapping[str, Tuple[float, float]] = MappingProxyType(_FIGURE_SIZES)


# ----------------------------
# Key -> (pretty label, unit)
# Centralized labels/units ensure consistent axis titles.
# ----------------------------
_LABEL_UNIT = {
    "alt_km":   ("Altitude", "km"),
    "a_km":     ("Semi-major axis", "km"),
    "r_km":     ("Radius", "km"),
    "v_kmps":   ("Speed", "km/s"),
    "lat_deg":  ("Latitude", "deg"),
    "lon_deg":  ("Longitude", "deg"),
    "i_deg":    ("Inclination", "deg"),
    "raan_deg": ("RAAN", "deg"),
    "argp_deg": ("Arg. of perigee", "deg"),
    "nu_deg":   ("True anomaly", "deg"),
    "t_days":   ("Time", "days"),
    "t_s":      ("Time", "s"),
}
LABEL_UNIT: Mapping[str, Tuple[str, str]] = MappingProxyType(_LABEL_UNIT)



# =============================================================================
# 2.                       GLOBAL THEME / RCPARAMS
# =============================================================================

def setup_global_style() -> None:
    """
    Configure Matplotlib global rcParams according to the project theme.

    Notes:
    - Uses THEME for layout-level styling (backgrounds, text, grid, legend).
    - Uses COLOR_CYCLE for the default axes color cycle.
    - Safe to call multiple times (idempotent update).
    """
    plt.rcParams.update({
        # ----------------------------
        # Figure / savefig backgrounds
        # ----------------------------
        "figure.facecolor":   THEME["bg_page"],
        "savefig.facecolor":  THEME["bg_page"],
        "savefig.edgecolor":  THEME["bg_page"],

        # Avoid label clipping on export (safe default)
        "savefig.bbox":       "tight",
        "savefig.pad_inches": 0.08,

        # ----------------------------
        # Axes styling
        # ----------------------------
        "axes.facecolor":     THEME["bg_axes"],
        "axes.edgecolor":     THEME["grid"],
        "axes.labelcolor":    THEME["text"],
        "axes.titlecolor":    THEME["text"],
        "axes.titleweight":   "bold",
        "axes.titlesize":     14,
        "axes.labelsize":     12,
        "axes.titlepad":      10,
        "axes.grid":          True,
        "axes.axisbelow":     True,
        "axes.spines.right":  False,
        "axes.spines.top":    False,

        # Default color cycle (categorical, print-friendly)
        "axes.prop_cycle":    plt.cycler(color=COLOR_CYCLE),

        # ----------------------------
        # Ticks / grid
        # ----------------------------
        "xtick.color":        THEME["text_dim"],
        "ytick.color":        THEME["text_dim"],
        "xtick.labelsize":    10,
        "ytick.labelsize":    10,
        "grid.color":         THEME["grid"],
        "grid.linestyle":     "-",
        "grid.linewidth":     0.8,
        # NOTE: grid.alpha is managed by PlotStyle/apply_rcparams (SSOT)

        # ----------------------------
        # Fonts / math text
        # ----------------------------
        "text.color":         THEME["text"],
        "font.family":        "sans-serif",
        "font.sans-serif":    ["DejaVu Sans", "Arial", "Helvetica"],
        # Make math rendering deterministic and consistent with sans-serif
        "mathtext.fontset":   "dejavusans",
        "mathtext.default":   "it",

        # ----------------------------
        # Lines
        # ----------------------------
        "lines.linewidth":        2.1,
        "lines.solid_capstyle":   "round",
        "lines.solid_joinstyle":  "round",

        # ----------------------------
        # Legend
        # ----------------------------
        "legend.frameon":     True,
        "legend.framealpha":  0.92,
        "legend.facecolor":   THEME["bg_axes"],
        "legend.edgecolor":   THEME["grid"],
        "legend.fancybox":    True,
        "legend.fontsize":    10,

        # ----------------------------
        # Layout
        # ----------------------------
        "figure.constrained_layout.use": False,
    })


@dataclass(frozen=True)
class PlotStyle:
    """High-level knobs for publication-quality defaults."""
    dpi: int = 180
    grid_alpha: float = 0.22
    font_family: str = "DejaVu Sans"
    mono_family: str = "DejaVu Sans Mono"
    base_fontsize: int = 10


DEFAULT_STYLE = PlotStyle()


def apply_rcparams(style: PlotStyle = DEFAULT_STYLE) -> None:
    """
    Apply global rcParams using the project's theme + user-overrides.

    Intended usage:
        apply_rcparams()  # once at program start
    """
    setup_global_style()

    # Minimal, safe overrides (keep theme defaults as the base)
    plt.rcParams.update({
        "figure.dpi": int(style.dpi),
        "savefig.dpi": int(style.dpi),

        # SSOT for grid transparency
        "grid.alpha": float(style.grid_alpha),

        "font.family": "sans-serif",
        "font.sans-serif": [style.font_family, "Arial", "Helvetica"],
        "font.monospace": [style.mono_family],

        "font.size": int(style.base_fontsize),
        "axes.titlesize": int(style.base_fontsize + 4),
        "axes.labelsize": int(style.base_fontsize + 2),
        "xtick.labelsize": int(style.base_fontsize),
        "ytick.labelsize": int(style.base_fontsize),
        "legend.fontsize": int(style.base_fontsize),
    })



# =============================================================================
# 3.                          AXES-LEVEL STYLING
# =============================================================================

def apply_axes_style(ax: "matplotlib.axes.Axes", title: str = "") -> None:
    """
    Apply a clean, publication-oriented style to a single Axes.

    Baseline comes from global rcParams (apply_rcparams). This function:
    - enforces axes facecolor
    - applies subtle major+minor grids using rcParams alpha as SSOT
    - styles title/labels/ticks/spines consistently
    """
    # Axes face color
    ax.set_facecolor(THEME["bg_axes"])

    # Minor ticks (safe for projections/custom axes)
    try:
        ax.minorticks_on()
    except Exception:
        pass

    # Grid: use rcParams as SSOT (avoid hardcoded alpha drift)
    major_alpha = float(plt.rcParams.get("grid.alpha", 0.22))
    minor_alpha = max(0.05, major_alpha * 0.45)

    ax.grid(True, which="major", alpha=major_alpha, linestyle="-", linewidth=0.8)
    ax.grid(True, which="minor", alpha=minor_alpha, linestyle=":", linewidth=0.6)

    # Title
    if title:
        ax.set_title(
            title,
            color=THEME["text"],
            fontweight="bold",
            fontsize=int(plt.rcParams.get("axes.titlesize", 14)),
            pad=12,
        )

    # Labels (keep existing text; just style them)
    ax.set_xlabel(
        ax.get_xlabel(),
        color=THEME["text"],
        fontsize=int(plt.rcParams.get("axes.labelsize", 12)),
    )
    ax.set_ylabel(
        ax.get_ylabel(),
        color=THEME["text"],
        fontsize=int(plt.rcParams.get("axes.labelsize", 12)),
    )

    # Ticks: consistent color and sizing
    ax.tick_params(
        axis="both",
        which="major",
        length=5,
        width=0.8,
        colors=THEME["text_dim"],
        labelsize=int(plt.rcParams.get("xtick.labelsize", 10)),
    )
    ax.tick_params(
        axis="both",
        which="minor",
        length=3,
        width=0.6,
        colors=THEME["text_dim"],
    )

    # Spines: subtle and consistent
    for name, spine in ax.spines.items():
        # Respect rcParams that already hide top/right spines
        if name in ("top", "right") and not spine.get_visible():
            continue
        spine.set_color(THEME["grid"])
        spine.set_linewidth(1.0)


def format_scientific_axis(
    ax: "matplotlib.axes.Axes",
    axis: str = "y",
    powerlimits: Tuple[int, int] = (-3, 3),
) -> None:
    """
    Apply scientific-notation formatting using mathtext.

    axis: "x", "y", or "both"
    powerlimits: switch to scientific outside these exponent limits.
    """
    fmt = ScalarFormatter(useMathText=True)
    try:
        fmt.set_powerlimits(powerlimits)
        # Avoid confusing additive offsets on engineering plots
        fmt.set_useOffset(False)
    except Exception:
        pass

    axis = (axis or "y").lower()
    if axis in ("y", "both"):
        ax.yaxis.set_major_formatter(fmt)
    if axis in ("x", "both"):
        ax.xaxis.set_major_formatter(fmt)


def format_log_axis_sci(ax: "matplotlib.axes.Axes", axis: str = "y", numticks: int = 8) -> None:
    """
    Improve log-axis tick locator + scientific formatting.
    """
    axis = (axis or "y").lower()
    locator = LogLocator(base=10.0, numticks=numticks)
    formatter = LogFormatterSciNotation()

    if axis == "y":
        ax.yaxis.set_major_locator(locator)
        ax.yaxis.set_major_formatter(formatter)
    elif axis == "x":
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
    elif axis == "both":
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_locator(locator)
        ax.yaxis.set_major_formatter(formatter)


def apply_standard_colorbar(cbar) -> None:
    """
    Standardize colorbar appearance across the project.
    """
    try:
        cbar.outline.set_linewidth(0.6)
    except Exception:
        pass

    try:
        cbar.ax.tick_params(colors=THEME["text_dim"])
    except Exception:
        pass



# =============================================================================
# 4.                     LEGEND & ANNOTATION HELPERS
# =============================================================================

def apply_legend_style(
    ax: "matplotlib.axes.Axes",
    loc: str = "best",
    *,
    title: Optional[str] = None,
    ncol: int = 1,
    framealpha: float = 0.92,
    linewidth: float = 0.9,
    text_color: Optional[str] = None,
    **kwargs,
):
    """
    Create a legend and enforce theme-consistent styling.

    Improvements:
    - Handles "no labeled artists" gracefully (returns None)
    - Supports title/ncol
    - Uses theme defaults but allows safe overrides
    """
    # If there are no labeled handles, Matplotlib will create an empty legend.
    try:
        handles, labels = ax.get_legend_handles_labels()
        if not any(lbl and not lbl.startswith("_") for lbl in labels):
            return None
    except Exception:
        pass

    leg = ax.legend(loc=loc, ncol=int(ncol), title=title, **kwargs)
    if leg is None:
        return None

    frame = leg.get_frame()
    frame.set_alpha(float(framealpha))
    frame.set_facecolor(THEME["bg_axes"])
    frame.set_edgecolor(THEME["grid"])
    try:
        frame.set_linewidth(float(linewidth))
    except Exception:
        pass

    # Title + text colors (robust across backends)
    try:
        if leg.get_title() is not None:
            leg.get_title().set_color(text_color or THEME["text"])
    except Exception:
        pass

    try:
        for t in leg.get_texts():
            t.set_color(text_color or THEME["text"])
    except Exception:
        pass

    return leg


def apply_standard_legend(ax: "matplotlib.axes.Axes", loc: str = "best", **kwargs):
    """Backwards-simple wrapper to keep legend usage consistent across the codebase."""
    return apply_legend_style(ax, loc=loc, **kwargs)



# =============================================================================
# 5.                        ASSETS & BACKGROUNDS
# =============================================================================

# Cache loaded images by absolute path to avoid repeated disk IO in batch plots
_LUNAR_MAP_CACHE: dict[str, Any] = {}


def _project_root_from_here(here: Path) -> Path:
    """
    Compatibility wrapper around the loader-layer project-root resolver.

    The actual repository discovery policy now lives in `loaders.io_helpers` so
    plotting/analysis code does not need to own filesystem heuristics.
    """
    return _project_root_from_path_loader(here)


def _iter_lunar_map_candidates(explicit_path: Optional[str] = None) -> list[Path]:
    """
    Compatibility wrapper around loader-side lunar texture discovery.

    Analysis code still calls this local helper, but the path search policy now
    lives in `loaders.io_helpers` so the asset-discovery behavior is shared by
    styling, reports, and any future export tools.
    """
    return _iter_lunar_map_candidates_from_loader(
        explicit_path,
        start_dir=Path(__file__).resolve().parent,
    )


def _find_lunar_map_path(explicit_path: Optional[str] = None) -> Optional[str]:
    """
    Resolve the first available lunar texture path using loader-layer policy.
    """
    return _find_lunar_map_path_from_loader(
        explicit_path,
        start_dir=Path(__file__).resolve().parent,
    )


def load_lunar_map(path: Optional[str] = None, *, cache: bool = True):
    """
    Load the lunar surface texture as a NumPy array (via matplotlib.image).

    - If `path` is not provided, uses `_find_lunar_map_path()` which checks
      LUNAR_SIMULATION/data/assets plus common local filenames and env vars.
    - Uses a simple in-memory cache keyed by resolved absolute path.

    Returns
    -------
    ndarray | None
        Image array (H×W or H×W×C), or None if not found / unreadable.
    """
    p = _find_lunar_map_path(path)
    if not p:
        return None

    if cache and p in _LUNAR_MAP_CACHE:
        return _LUNAR_MAP_CACHE[p]

    try:
        img = mpimg.imread(p)
    except Exception:
        return None

    # Basic sanity checks: ndim 2/3 only
    try:
        ndim = int(getattr(img, "ndim", 0))
        if ndim not in (2, 3):
            return None
    except Exception:
        pass

    if cache:
        _LUNAR_MAP_CACHE[p] = img
    return img


def add_lunar_background(
    ax: "matplotlib.axes.Axes",
    map_img=None,
    map_path: Optional[str] = None,
    alpha: float = 0.55,
    extent: Tuple[float, float, float, float] = (-180.0, 180.0, -90.0, 90.0),
    zorder: int = 0,
    *,
    origin: str = "upper",
    interpolation: str = "bilinear",
    keep_limits: bool = True,
):
    """
    Draw a lunar map behind the given Axes using `imshow`.

    Improvements:
    - Loads from canonical assets dir (LUNAR_SIMULATION/data/assets)
    - Keeps existing x/y limits by default (background shouldn't change view)
    - Adds interpolation control for nicer output
    - Robust handling for grayscale vs RGB/RGBA

    Returns
    -------
    AxesImage | None
    """
    if map_img is None:
        map_img = load_lunar_map(map_path)
    if map_img is None:
        return None

    # Preserve current limits so imshow doesn't reset view unexpectedly
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    imshow_kwargs: dict[str, Any] = {
        "extent": extent,
        "origin": origin,
        "alpha": float(alpha),
        "zorder": int(zorder),
        "aspect": "auto",
        "interpolation": interpolation,
    }

    # If single-channel, render as grayscale; if RGB/RGBA, let imshow handle colors.
    try:
        if getattr(map_img, "ndim", 0) == 2:
            imshow_kwargs["cmap"] = "gray"
    except Exception:
        pass

    im = ax.imshow(map_img, **imshow_kwargs)

    if keep_limits:
        try:
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
        except Exception:
            pass

    return im



# =============================================================================
# 6.                         LABEL & UNIT HELPERS
# =============================================================================

def get_unit_label(key: str, default: str = "") -> str:
    """Return the preferred unit string for a known key."""
    if not key:
        return default
    unit = LABEL_UNIT.get(key, ("", default))[1]
    return unit or default


def get_pretty_label(key: str, default: str = "") -> str:
    """Return the preferred human-readable label for a known key."""
    if not key:
        return default
    label = LABEL_UNIT.get(key, (default, ""))[0]
    return label or default


def make_axis_label(key: str, default: str = "") -> str:
    """
    Return a formatted axis label:
      - "Label [unit]" if both are known
      - "Label" if only label is known
      - fallback to `default` or the key itself
    """
    if not key:
        return default

    label = get_pretty_label(key, "")
    unit = get_unit_label(key, "")
    if label and unit:
        return f"{label} [{unit}]"
    if label:
        return label
    return default or key



# =============================================================================
# 7.                           COLOR SELECTION HELPERS
# =============================================================================

def _normalize_series_key(name: str) -> str:
    """
    Normalize series keys to a canonical internal form.

    Legacy/synonym mappings are intentionally NOT used here to keep the codebase strict.
    If you want aliases, do it upstream where the series is created.
    """
    if not name:
        return ""
    k = str(name).strip().lower()
    # normalize separators
    k = k.replace(" ", "_").replace("-", "_")
    while "__" in k:
        k = k.replace("__", "_")
    return k


def get_accel_color(name: str) -> str:
    """
    Return a reserved semantic color for acceleration/component names.

    Substring-based by design (robust to naming differences), but avoids
    broad false-positives by using slightly tighter checks.
    """
    if not name:
        return THEME["text"]

    key = str(name).strip().lower()

    # Spherical harmonics / gravity anomalies
    if ("spherical" in key) or ("harmonic" in key) or ("_sh" in key) or key.startswith("sh"):
        return SEMANTIC["spherical_harmonics"]

    # Solar radiation pressure
    if ("srp" in key) or ("solar_radiation" in key) or ("radiation_pressure" in key):
        return SEMANTIC["solar_radiation_pressure"]

    # Third-body gravity (Earth/Sun/etc.)
    # NOTE: avoid matching "sun" inside unrelated words by checking word-ish patterns
    if ("third" in key) or ("3rd" in key) or ("third_body" in key) or ("earth" in key) or (" sun" in key) or key.startswith("sun"):
        return SEMANTIC["third_body_gravity"]

    # Albedo
    if ("albedo" in key) or ("_alb" in key) or key.startswith("alb"):
        return SEMANTIC["albedo"]

    # Relativity
    if ("relativ" in key) or ("relativity" in key) or ("_gr" in key) or key.startswith("gr_"):
        return SEMANTIC["relativity"]

    # Baseline / central gravity / orbit reference
    if ("two_body" in key) or ("2body" in key) or ("central_gravity" in key) or (key == "grav") or key.startswith("grav_"):
        return SEMANTIC["orbit"]

    # Fallback
    return THEME["text"]


def get_series_color(name: Optional[str], *, idx: int = 0, default: Optional[str] = None) -> str:
    """
    Return a stable color for a named series.

    Behavior
    --------
    1) If the series name maps to a known canonical key -> use SERIES_COLORS
    2) Otherwise choose a deterministic color from COLOR_CYCLE using a stable hash

    Parameters
    ----------
    name:
        Series key or display name (expected canonical in a strict codebase).
    idx:
        Extra offset when you intentionally want distinct colors for related names.
    default:
        Fallback color if name is None/empty.
    """
    if not name:
        return default or COLOR_CYCLE[int(idx) % len(COLOR_CYCLE)]

    key = _normalize_series_key(name)
    if key in SERIES_COLORS:
        return SERIES_COLORS[key]

    # Deterministic hash -> index into COLOR_CYCLE
    h = hashlib.md5(key.encode("utf-8")).hexdigest()
    k = int(h[:8], 16)
    return COLOR_CYCLE[(k + int(idx)) % len(COLOR_CYCLE)]



# =============================================================================
# 8.                               SMOKE TEST
# =============================================================================

if __name__ == "__main__":
    # Lightweight smoke test: run this module directly to verify:
    # - imports & theme wiring
    # - rcParams application
    # - axes/legend helpers
    # - deterministic color selection
    # - optional background loading (non-fatal if missing)

    import numpy as np
    import matplotlib.pyplot as plt

    # 1) Apply global theme/rcParams
    apply_rcparams()

    # 2) Basic plot (deliberately small + fast)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    apply_axes_style(ax, "Smoke Test: Theme + Helpers")

    x = np.linspace(0.0, 10.0, 400)
    ax.plot(x, np.sin(x), label="sin(x)", color=get_series_color("altitude"))
    ax.plot(x, np.cos(x), label="cos(x)", color=get_series_color("energy_drift"))
    ax.set_xlabel(make_axis_label("t_s"))
    ax.set_ylabel("Value")

    # 3) Formatting helpers
    format_scientific_axis(ax, axis="y", powerlimits=(-2, 2))
    apply_standard_legend(ax, loc="upper right")

    # 4) Optional background (safe if missing)
    # We keep this non-fatal so this file can be run on clean machines/CI.
    bg = add_lunar_background(ax, alpha=0.12)
    if bg is None:
        print("[smoke] add_lunar_background: no texture found (OK)")

    # 5) Basic invariants / sanity prints
    print("[smoke] THEME keys:", sorted(THEME.keys()))
    print("[smoke] COLOR_CYCLE length:", len(COLOR_CYCLE))
    print("[smoke] sample accel colors:",
          get_accel_color("SRP"),
          get_accel_color("SH"),
          get_accel_color("third_body Earth"))

    plt.tight_layout()
    plt.show()


# =============================================================================
# 9.                               PUBLIC API
# =============================================================================
# Export only the stable symbols intended for external users of this module.
# Keep this list small and curated: it acts as the "surface area" of the plotting API.

__all__ = [
    # -------------------------------------------------------------------------
    # Palettes & theme dictionaries
    # -------------------------------------------------------------------------
    # COLORS:
    #   Raw hex colors grouped into neutrals + categorical accents.
    #   Use this if you need a specific named color (e.g., COLORS["accent"]["teal"]).
    "COLORS",

    # COLOR_CYCLE:
    #   Default categorical cycle for line plots (Matplotlib prop_cycle).
    "COLOR_CYCLE",

    # THEME:
    #   Layout-level colors (figure/axes backgrounds, grid color, text colors).
    "THEME",

    # SEMANTIC:
    #   Domain-meaning colors (orbit / SH / SRP / third-body / albedo / relativity / events).
    #   Keep these stable across the project so figures become instantly interpretable.
    "SEMANTIC",

    # SERIES_COLORS:
    #   Canonical series-key -> color mapping (used by get_series_color()).
    "SERIES_COLORS",

    # FIGURE_SIZES:
    #   Named size presets (inches) for consistent export layouts.
    "FIGURE_SIZES",

    # LABEL_UNIT:
    #   Key -> (pretty label, unit) mapping used by make_axis_label().
    "LABEL_UNIT",

    # -------------------------------------------------------------------------
    # Global styling (rcParams)
    # -------------------------------------------------------------------------
    # PlotStyle / DEFAULT_STYLE:
    #   Small, frozen config object for global plotting preferences (dpi, fonts, grid alpha).
    "PlotStyle",
    "DEFAULT_STYLE",

    # setup_global_style():
    #   Applies the baseline theme to Matplotlib rcParams (safe to call multiple times).
    "setup_global_style",

    # apply_rcparams():
    #   The recommended entry point: applies theme + PlotStyle overrides.
    "apply_rcparams",

    # -------------------------------------------------------------------------
    # Axes-level styling & formatters
    # -------------------------------------------------------------------------
    # apply_axes_style():
    #   Per-Axes finishing touches (minor grid, ticks, spines, title styling).
    "apply_axes_style",

    # format_scientific_axis():
    #   Scientific notation (mathtext) formatter for x/y/both axes.
    "format_scientific_axis",

    # format_log_axis_sci():
    #   Cleaner tick locator/formatter for log-scaled axes.
    "format_log_axis_sci",

    # apply_standard_colorbar():
    #   Minimal consistent styling for colorbars (ticks, outline).
    "apply_standard_colorbar",

    # -------------------------------------------------------------------------
    # Legend helpers
    # -------------------------------------------------------------------------
    # apply_legend_style():
    #   Creates a legend with theme-consistent frame/text styling.
    "apply_legend_style",

    # apply_standard_legend():
    #   Convenience wrapper for common legend usage.
    "apply_standard_legend",

    # -------------------------------------------------------------------------
    # Assets & backgrounds
    # -------------------------------------------------------------------------
    # load_lunar_map():
    #   Loads a lunar texture (supports canonical path: LUNAR_SIMULATION/data/assets).
    "load_lunar_map",

    # add_lunar_background():
    #   Places the lunar texture behind an Axes using imshow (safe no-op if missing).
    "add_lunar_background",

    # -------------------------------------------------------------------------
    # Labels & units
    # -------------------------------------------------------------------------
    # get_unit_label() / get_pretty_label() / make_axis_label():
    #   Helpers for consistent axis labeling.
    "get_unit_label",
    "get_pretty_label",
    "make_axis_label",

    # -------------------------------------------------------------------------
    # Color selection helpers
    # -------------------------------------------------------------------------
    # get_accel_color():
    #   Semantic color for acceleration/component names (substring-based).
    "get_accel_color",

    # get_series_color():
    #   Stable color for named series: canonical mapping first, then deterministic cycle hash.
    "get_series_color",
]
