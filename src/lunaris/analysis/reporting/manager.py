# # ST_LRPS/analysis/reporting/manager.py

"""
ST_LRPS Report Manager
======================

This module produces publication-ready visual reports from simulation outputs.

It is the orchestration layer that bridges:
- **Post-processing** (normalized history dictionaries)
- **Plotting** (figure factories)
- **Export** (PNG quicklooks and multi-page PDF reports)

Responsibilities
----------------
- Consume a normalized ``history`` mapping produced by ``analysis.postprocess``.
- Optionally merge runtime metadata with auto-discovered config (when available).
- Build a fixed sequence of report pages (executive summary → diagnostics).
- Save:
  - High-resolution PNG quicklooks (optional)
  - A multi-page PDF report (optional)

Non-Responsibilities
--------------------
- No numerical propagation, dynamics, or physics modeling.
- No ownership of plotting primitives or style definitions.
- No attempt to "fix" upstream numerical correctness.

Contracts and Inputs
--------------------
**history**
    A mapping that follows the postprocess contract. At minimum, report generation
    expects the time series fields required by the plotting API (e.g. ``t_s``,
    ``y`` or derived arrays). Optional fields (events, invariants, ground track)
    may be absent; pages that depend on them should degrade gracefully.

**meta**
    Optional metadata mapping. Used for:
    - run configuration (integrator, tolerances, SH degree, spacecraft params)
    - optional config discovery (e.g. output directory hints)

**ctx**
    Optional context/engine object for advanced plots (e.g. force budget or
    ephemeris-dependent products). If ``ctx`` is not provided, ctx-dependent
    pages are skipped.

Design Notes
------------
The file is intentionally organized in a top-down, "page-oriented" order:

1. Core helpers (small, reusable utilities)
2. Page builders (``figure_*_page``)
3. Output writers (PNG/PDF exporters)
4. High-level orchestration (``plot_all``)

This layout makes it easy to:
- read the report narrative from top to bottom, and
- jump directly to the page you want to modify.

Usage
-----
Typical batch use::

    from lunaris.analysis.postprocess import process_simulation_results
    from lunaris.analysis.reporting.manager import plot_all

    history = process_simulation_results(result, ctx=engine, cfg=config)
    outputs = plot_all(history, out_dir="outputs/simulations/run_01", ctx=engine)

Interactive use (e.g., notebooks) is also supported: call the page builders
directly and display the returned Matplotlib figures.

"""


# =============================================================================
# 0.                                 IMPORTS
# =============================================================================

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Callable

import datetime as _dt
import os
import re
from pathlib import Path

import numpy as np

from lunaris.loaders.io_helpers import find_lunar_map_path

# --- Matplotlib backend selection (headless-safe) ---
# IMPORTANT: backend must be selected BEFORE importing pyplot.
import matplotlib

_INTERACTIVE = os.environ.get("STLRPS_INTERACTIVE", "0").strip().lower() in {"1", "true", "yes", "y"}
if not _INTERACTIVE:
    # Agg is safe for servers/CI/headless PDF generation.
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def _noop_apply_rcparams(*_: Any, **__: Any) -> None:
    """Fallback when styling module is unavailable."""
    return


# --- Optional styling helpers (soft dependency) ---
try:
    from .styling import DEFAULT_STYLE, apply_rcparams  # type: ignore
except ImportError:  # styling is optional
    DEFAULT_STYLE = None  # type: ignore[assignment]
    apply_rcparams = _noop_apply_rcparams  # type: ignore[assignment]


# --- Postprocess extractors (hard dependency) ---
from lunaris.analysis.postprocess import (  # type: ignore
    extract_time_seconds,
    extract_time_days,
    extract_altitude_km,
    extract_elements,
    extract_invariants,
    extract_events,
)

from lunaris.analysis.formatting import (
    format_days,
    format_duration,
    format_count,
    format_km,
    format_sci_or_na,
)

# --- Plotting API (hard dependency) ---
from .plotting import (  # type: ignore
    # --- Public Figures ---
    figure_elements_timeseries,
    figure_invariants,
    figure_altitude_with_events,
    figure_relative_drift,
    figure_ground_track,
    figure_orbit_3d,
    figure_eomega,
    figure_perturbation_magnitude,
    figure_events_table,

    # --- Layout & Data Helpers (Shared API) ---
    draw_table,
    draw_kv_block,
    draw_kv_table,
    metrics_rows,

    # --- Config Discovery ---
    effects_from_meta_history,
    merge_meta_with_auto_config,
)


# =============================================================================
# 1.                                CORE HELPERS
# =============================================================================

def _as_1d_float_array(x: Any) -> np.ndarray:
    """Best-effort conversion to a 1D float NumPy array.

    - None -> empty array
    - scalar -> shape (1,)
    - array-like -> flattened
    - on failure -> empty array
    """
    if x is None:
        return np.array([], dtype=float)

    # Fast path: ndarray
    if isinstance(x, np.ndarray):
        try:
            arr = x.astype(float, copy=False)
        except (TypeError, ValueError):
            arr = np.asarray(x, dtype=float)
        return np.ravel(arr)

    # General path: array-like
    try:
        arr = np.asarray(x, dtype=float)
        return np.ravel(arr)
    except (TypeError, ValueError):
        pass

    # Iterable fallback (generators etc.)
    try:
        arr = np.fromiter((float(v) for v in x), dtype=float)  # type: ignore[arg-type]
        return np.ravel(arr)
    except Exception:
        return np.array([], dtype=float)


def _ensure_dir(path: Optional[str]) -> str:
    """Ensure directory exists; return normalized directory path as str."""
    p = Path(path or ".")
    p.mkdir(parents=True, exist_ok=True)
    return str(p)


def _find_default_lunar_surface_map() -> Optional[str]:
    """
    Compatibility wrapper around the loader-layer lunar-map locator.

    Report generation should not own repository asset-discovery policy. The
    shared loader helper keeps report exports, plot backgrounds, and future
    asset consumers aligned on the same canonical search rules.
    """
    return find_lunar_map_path(start_dir=Path(__file__).resolve().parent)


def _resolve_out_dir_hint(
    hint: Optional[str],
    meta: Optional[Mapping[str, Any]],
) -> Optional[str]:
    """Return a directory path usable for config discovery.

    Plotting-side auto-config expects a directory. Callers sometimes pass
    a PDF file path; normalize to parent directory.

    Rules:
    - If hint is None, try meta['_report_search_dir'].
    - If hint ends with '.pdf' (case-insensitive), treat it as a file path.
    - If hint exists on disk and is a file, treat it as a file path.
    """
    if not hint and meta:
        candidate = meta.get("_report_search_dir")
        hint = str(candidate) if candidate else None

    if not hint:
        return None

    try:
        p = Path(hint)
        suffix = p.suffix.lower()

        # "looks like a PDF output path" even if it doesn't exist yet
        if suffix == ".pdf":
            return str(p.parent)

        # If it exists and is a file, normalize to its parent
        if p.exists() and p.is_file():
            return str(p.parent)

        # Otherwise assume it's a directory path (existing or intended)
        return str(p)
    except Exception:
        return hint



# =============================================================================
# 2.                               REPORT PAGES
# =============================================================================

def figure_summary_page(
    history: Mapping[str, Any],
    meta: Optional[Mapping[str, Any]] = None,
    *,
    ctx: Any = None,
    pdf_or_out_dir_hint: Optional[str] = None,
) -> plt.Figure:
    """Build an executive-summary page with decision-oriented metrics and diagnostics."""
    apply_rcparams(DEFAULT_STYLE)

    out_dir_hint = _resolve_out_dir_hint(pdf_or_out_dir_hint, meta)
    meta2 = merge_meta_with_auto_config(meta or {}, history, out_dir_hint)

    t_s = extract_time_seconds(history)
    elems = extract_elements(history)
    inv = extract_invariants(history)
    alt_km = _as_1d_float_array(extract_altitude_km(history, meta=meta2))
    events = extract_events(history) or {}
    dur_s = float(np.nanmax(t_s) - np.nanmin(t_s)) if getattr(t_s, "size", 0) else float("nan")

    effects, source_label = effects_from_meta_history(meta2, history, ctx=ctx)
    orb_rows, inv_rows = metrics_rows(history, elems)

    altitude_min = float(np.nanmin(alt_km)) if alt_km.size else float("nan")
    altitude_max = float(np.nanmax(alt_km)) if alt_km.size else float("nan")
    altitude_final = float(alt_km[-1]) if alt_km.size else float("nan")
    ecc_final = float(_as_1d_float_array(elems.get("e", []))[-1]) if _as_1d_float_array(elems.get("e", [])).size else float("nan")
    inc_final = float(_as_1d_float_array(elems.get("i_deg", []))[-1]) if _as_1d_float_array(elems.get("i_deg", [])).size else float("nan")
    n_steps = int(getattr(t_s, "size", 0))
    n_peri = int(np.size(events.get("peri_idx", []))) if events.get("peri_idx", None) is not None else 0
    n_apo = int(np.size(events.get("apo_idx", []))) if events.get("apo_idx", None) is not None else 0
    has_impact = events.get("impact_idx", None) is not None
    active_effects = sum(1 for enabled in effects.values() if enabled)

    integrator_name = (
        meta2.get("integrator_method")
        or meta2.get("propagator_method")
        or meta2.get("method")
        or meta2.get("propagator")
        or "Unknown"
    )
    output_dt_s = meta2.get("output_dt_s", meta2.get("dt_out_s", None))

    rel_energy = _as_1d_float_array(inv.get("rel_energy_drift", []))
    rel_h = _as_1d_float_array(inv.get("rel_h_drift", []))
    if rel_energy.size == 0:
        energy = _as_1d_float_array(inv.get("energy_Jkg", []))
        if energy.size >= 2 and np.isfinite(energy[0]) and abs(float(energy[0])) > 0.0:
            rel_energy = (energy - float(energy[0])) / max(abs(float(energy[0])), 1e-30)
    if rel_h.size == 0:
        h_norm = _as_1d_float_array(inv.get("h_norm_m2s", []))
        if h_norm.size >= 2 and np.isfinite(h_norm[0]) and abs(float(h_norm[0])) > 0.0:
            rel_h = (h_norm - float(h_norm[0])) / max(abs(float(h_norm[0])), 1e-30)

    max_rel_energy = float(np.nanmax(np.abs(rel_energy))) if rel_energy.size else float("nan")
    max_rel_h = float(np.nanmax(np.abs(rel_h))) if rel_h.size else float("nan")

    primary_rows = [
        ("Trajectory span", f"{format_days(dur_s)} ({format_duration(dur_s)})"),
        ("Output epochs", format_count(n_steps)),
        ("Altitude floor", format_km(altitude_min)),
        ("Altitude ceiling", format_km(altitude_max)),
        ("Final altitude", format_km(altitude_final)),
        ("Final eccentricity", f"{ecc_final:.6f}" if np.isfinite(ecc_final) else "N/A"),
        ("Final inclination", f"{inc_final:.4f} deg" if np.isfinite(inc_final) else "N/A"),
        ("Impact detected", "Yes" if has_impact else "No"),
    ]
    health_rows = [
        ("Integrator", str(integrator_name)),
        ("Output cadence", "Auto" if output_dt_s in (None, "") else f"{output_dt_s} s"),
        ("Active models", f"{active_effects} / {len(effects)} enabled"),
        ("Periapsis passes", format_count(n_peri)),
        ("Apoapsis passes", format_count(n_apo)),
        ("Peak rel. energy drift", format_sci_or_na(max_rel_energy)),
        ("Peak rel. ang. mom. drift", format_sci_or_na(max_rel_h)),
        ("Config source", str(source_label)),
    ]

    fig = plt.figure(figsize=(8.27, 11.69))
    c_title = "#1E293B"
    c_muted = "#64748B"
    c_line = "#CBD5E1"

    fig.text(0.5, 0.955, "MISSION SIMULATION REPORT", ha="center", va="top", fontsize=22, fontweight="bold", color=c_title)
    fig.text(
        0.5,
        0.925,
        "Executive summary of trajectory health, orbital behavior, and enabled physics.",
        ha="center",
        va="top",
        fontsize=10,
        color=c_muted,
    )
    fig.text(
        0.5,
        0.905,
        f"Generated: {_dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        ha="center",
        va="top",
        fontsize=9.5,
        color=c_muted,
    )
    fig.add_artist(plt.Line2D([0.08, 0.92], [0.888, 0.888], transform=fig.transFigure, color=c_line, linewidth=1.0))

    card_specs = [
        ("Trajectory Span", format_duration(dur_s), "#1E3A8A"),
        ("Integrator", str(integrator_name).upper(), "#334155"),
        ("Output Epochs", format_count(n_steps), "#0F766E"),
        ("Impact State", "YES" if has_impact else "CLEAR", "#B91C1C" if has_impact else "#166534"),
    ]
    for idx, (label, value, color) in enumerate(card_specs):
        left = 0.08 + idx * 0.21
        ax_card = fig.add_axes([left, 0.79, 0.18, 0.075])
        ax_card.axis("off")
        ax_card.add_patch(
            plt.Rectangle(
                (0.0, 0.0),
                1.0,
                1.0,
                transform=ax_card.transAxes,
                facecolor="#F8FAFC",
                edgecolor="#D7E1F0",
                linewidth=1.1,
            )
        )
        ax_card.text(0.05, 0.72, label, fontsize=8.2, color="#64748B", fontweight="bold", transform=ax_card.transAxes)
        ax_card.text(0.05, 0.22, value, fontsize=14, color=color, fontweight="bold", transform=ax_card.transAxes)

    ax_primary = fig.add_axes([0.08, 0.56, 0.40, 0.18])
    draw_kv_table(ax_primary, "Primary Mission Metrics", primary_rows)

    ax_health = fig.add_axes([0.52, 0.56, 0.40, 0.18])
    draw_kv_table(ax_health, "Run Health & Diagnostics", health_rows)

    fig.text(0.08, 0.515, "Active Physical Models", fontsize=12, fontweight="bold", color="#334155")
    ax_phy = fig.add_axes([0.08, 0.38, 0.84, 0.12])
    ax_phy.axis("off")
    items = list(effects.items())
    cols = 4
    rows = max(1, (len(items) + cols - 1) // cols)
    badge_w = 0.22
    badge_h = 0.24

    def _wrap_effect_name(name: str, max_len: int = 16) -> str:
        words = str(name).split()
        if not words:
            return str(name)
        lines: list[str] = []
        current = words[0]
        for word in words[1:]:
            if len(current) + 1 + len(word) <= max_len:
                current += " " + word
            else:
                lines.append(current)
                current = word
        lines.append(current)
        return "\n".join(lines[:2]) if len(lines) > 1 else lines[0]

    for i, (name, is_active) in enumerate(items):
        r = i // cols
        c = i % cols
        x = c / cols
        y = 0.76 - r * 0.30
        bg = "#DCFCE7" if is_active else "#F1F5F9"
        fg = "#166534" if is_active else "#64748B"
        edge = "#86EFAC" if is_active else "#CBD5E1"
        ax_phy.add_patch(
            plt.Rectangle((x, y), badge_w, badge_h, facecolor=bg, edgecolor=edge, transform=ax_phy.transAxes, lw=1.0)
        )
        ax_phy.text(
            x + badge_w / 2,
            y + badge_h / 2,
            f"{_wrap_effect_name(name)}\n{'ON' if is_active else 'OFF'}",
            ha="center",
            va="center",
            fontsize=7.1,
            fontweight="bold",
            color=fg,
            transform=ax_phy.transAxes,
        )
    fig.text(0.92, 0.365, f"Configuration source: {source_label}", ha="right", fontsize=8, color="#94A3B8")

    ax_orb = fig.add_axes([0.08, 0.19, 0.84, 0.15])
    draw_table(ax_orb, "Orbital Elements Statistics", orb_rows)

    ax_inv = fig.add_axes([0.08, 0.03, 0.84, 0.15])
    draw_table(ax_inv, "Conservation Diagnostics", inv_rows)

    return fig


def figure_run_config_page(
    history: Mapping[str, Any],
    meta: Optional[Mapping[str, Any]] = None,
    *,
    ctx: Any = None,
    pdf_or_out_dir_hint: Optional[str] = None,
) -> plt.Figure:
    """
    Build the configuration page with cleaner engineering summaries.

    This later definition intentionally replaces the legacy layout above while
    keeping the public function name stable for the rest of the reporting
    pipeline.
    """

    apply_rcparams(DEFAULT_STYLE)

    out_dir_hint = _resolve_out_dir_hint(pdf_or_out_dir_hint, meta)
    meta2 = merge_meta_with_auto_config(meta or {}, history, out_dir_hint)

    def _as_dict(obj: Any) -> Dict[str, Any]:
        if obj is None:
            return {}
        if isinstance(obj, dict):
            return obj
        return {k: getattr(obj, k) for k in dir(obj) if not k.startswith("_")}

    integrator = (
        meta2.get("integrator_method")
        or meta2.get("propagator_method")
        or meta2.get("method")
        or meta2.get("propagator")
        or "Variable Step"
    )
    rtol = meta2.get("rtol", "1e-12")
    atol = meta2.get("atol", "1e-12")
    sh_degree = meta2.get("degree", 0)
    mu = float(meta2.get("mu_m3s2", 4.9048695e12))
    out_dt_s = meta2.get("output_dt_s")
    if out_dt_s is None:
        out_dt_s = meta2.get("dt_out_s")

    sc = _as_dict(meta2.get("spacecraft", {}))
    mass = sc.get("mass_kg", "N/A")
    area = sc.get("area_m2", "N/A")
    cd = sc.get("cd", 1.5)
    cr = sc.get("cr", 1.5)

    t_prop = float(meta2.get("propagation_time_s", 0.0))
    t_wall = float(meta2.get("wall_time_s", 0.0))
    ev = extract_events(history) or {}
    peri_idx = ev.get("peri_idx", [])
    apo_idx = ev.get("apo_idx", [])
    impact_idx = ev.get("impact_idx", None)
    n_peri = int(np.size(peri_idx)) if peri_idx is not None else 0
    n_apo = int(np.size(apo_idx)) if apo_idx is not None else 0
    has_impact = impact_idx is not None
    t_s = extract_time_seconds(history)
    n_steps = int(getattr(t_s, "size", 0))
    duration_s = float(np.nanmax(t_s) - np.nanmin(t_s)) if getattr(t_s, "size", 0) else float("nan")

    fig = plt.figure(figsize=(8.27, 11.69))
    fig.suptitle("CONFIGURATION & PERFORMANCE", fontsize=16, fontweight="bold", y=0.955, color="#334155")
    fig.text(
        0.5,
        0.928,
        "Simulation settings, spacecraft definition, event assessment, and runtime diagnostics.",
        ha="center",
        fontsize=9.5,
        color="#64748B",
    )

    settings_rows = [
        ("Integrator", str(integrator)),
        ("Tolerance (rtol / atol)", f"{rtol} / {atol}"),
        ("Gravity model", f"Spherical harmonics up to degree {sh_degree}"),
        ("Central-body GM", format_sci_or_na(mu, decimals=4) + " m^3/s^2"),
        ("Output step", "Auto" if out_dt_s is None else f"{out_dt_s} s"),
        ("Reported span", f"{format_days(duration_s)} ({format_duration(duration_s)})"),
    ]
    spacecraft_rows = [
        ("Dry mass", f"{mass} kg"),
        ("Cross-section area", f"{area} m^2"),
        ("Drag coefficient Cd", str(cd)),
        ("Reflectivity Cr", str(cr)),
    ]
    event_rows = [
        ("Periapsis detections", format_count(n_peri)),
        ("Apoapsis detections", format_count(n_apo)),
        ("Impact detected", "Yes" if has_impact else "No"),
        ("Impact threshold", f"{ev.get('impact_alt_km')} km" if ev.get("impact_alt_km", None) is not None else "N/A"),
    ]
    perf_rows = [
        ("Propagation time", f"{t_prop:.4f} s"),
        ("Total wall time", f"{t_wall:.4f} s"),
        ("Output epochs", format_count(n_steps)),
        ("Wall / propagation ratio", f"{(t_wall / t_prop):.2f}x" if t_prop > 0.0 else "N/A"),
    ]

    ax_cfg = fig.add_axes([0.08, 0.68, 0.40, 0.20])
    draw_kv_table(ax_cfg, "Simulation Settings", settings_rows)

    ax_sc = fig.add_axes([0.52, 0.68, 0.40, 0.20])
    draw_kv_table(ax_sc, "Spacecraft Parameters", spacecraft_rows)

    ax_events = fig.add_axes([0.08, 0.42, 0.40, 0.18])
    draw_kv_table(ax_events, "Mission Event Assessment", event_rows)

    ax_perf = fig.add_axes([0.52, 0.42, 0.40, 0.18])
    draw_kv_table(ax_perf, "Computational Performance", perf_rows)

    ax_notes = fig.add_axes([0.08, 0.18, 0.84, 0.16])
    note_items = [
        ("Interpretation", "Use low drift plus stable event statistics as a quick quality screen before detailed plot review."),
        ("Output source", Path(pdf_or_out_dir_hint).name if pdf_or_out_dir_hint else "In-memory report export"),
    ]
    draw_kv_block(ax_notes, "Operator Notes", note_items, ncols=1)

    return fig



# ============================================================
# 3.              OUTPUT WRITERS
# ============================================================

import logging

logger = logging.getLogger(__name__)


def _render_and_save(
    fig_factory: Callable[[], plt.Figure],
    save_func: Callable[[plt.Figure], None],
    desc: str,
) -> bool:
    """Render a figure and save it safely.

    - Applies rcParams for each figure (styling is optional).
    - Ensures the created figure is closed to avoid memory leaks.
    - Logs exceptions but does not raise (report generation should be resilient).
    """
    fig: Optional[plt.Figure] = None
    try:
        apply_rcparams(DEFAULT_STYLE)
        fig = fig_factory()
        save_func(fig)
        return True
    except Exception:
        logger.exception("Failed to generate '%s'", desc)
        return False
    finally:
        if fig is not None:
            plt.close(fig)


def save_quicklook_pngs(
    out_dir: str,
    history: Mapping[str, Any],
    *,
    prefix: str = "ST_LRPS",
    meta: Optional[Mapping[str, Any]] = None,
    dpi: int = 300,
    ctx: Any = None,
) -> Dict[str, str]:
    """Generate and save individual PNGs for quick preview."""
    out_path = _ensure_dir(out_dir)
    saved_files: Dict[str, str] = {}

    # Pre-fetch (avoid repeated extraction)
    t_days = _as_1d_float_array(extract_time_days(history))
    elems = extract_elements(history)
    inv = extract_invariants(history)
    alt_km = _as_1d_float_array(extract_altitude_km(history, meta=meta))
    events = extract_events(history) or {}

    # Plot tasks: (name, factory)
    plot_tasks: list[tuple[str, Callable[[], plt.Figure]]] = [
        ("dashboard",   lambda: figure_summary_page(history, meta, ctx=ctx, pdf_or_out_dir_hint=out_path)),
        ("config",      lambda: figure_run_config_page(history, meta, ctx=ctx, pdf_or_out_dir_hint=out_path)),
        ("orbit_3d",    lambda: figure_orbit_3d(history, meta=meta)),
        ("groundtrack", lambda: figure_ground_track(history, meta=meta, ctx=ctx)),
        ("elements",    lambda: figure_elements_timeseries(t_days, elems)),
        ("invariants",  lambda: figure_invariants(t_days, inv)),
        ("drift",       lambda: figure_relative_drift(t_days, inv)),
        ("phase_space", lambda: figure_eomega(t_days, elems)),
        ("altitude",    lambda: figure_altitude_with_events(t_days, alt_km, events=events)),
        ("events",      lambda: figure_events_table(events, t_days)),
    ]

    if ctx is not None:
        plot_tasks.append(("accel_budget", lambda: figure_perturbation_magnitude(history, ctx)))

    logger.info("Saving quicklook PNGs to: %s", out_path)

    for name, factory in plot_tasks:
        filename = f"{prefix}_{name}.png"
        full_path = str(Path(out_path) / filename)

        def _save_png(fig: plt.Figure, *, _path: str = full_path) -> None:
            fig.savefig(_path, dpi=dpi, bbox_inches="tight")

        if _render_and_save(factory, _save_png, name):
            saved_files[name] = full_path

    return saved_files


def make_report_pdf(
    pdf_path: str,
    history: Mapping[str, Any],
    *,
    meta: Optional[Mapping[str, Any]] = None,
    ctx: Any = None,
) -> str:
    """Compile a multi-page PDF report from simulation history."""
    pdf_path = str(pdf_path)
    out_dir = str(Path(pdf_path).resolve().parent)
    _ensure_dir(out_dir)

    # Pre-fetch
    t_days = _as_1d_float_array(extract_time_days(history))
    elems = extract_elements(history)
    inv = extract_invariants(history)
    alt_km = _as_1d_float_array(extract_altitude_km(history, meta=meta))
    events = extract_events(history) or {}

    pages: list[tuple[str, Callable[[], plt.Figure]]] = [
        ("Dashboard",      lambda: figure_summary_page(history, meta, ctx=ctx, pdf_or_out_dir_hint=pdf_path)),
        ("Configuration",  lambda: figure_run_config_page(history, meta, ctx=ctx, pdf_or_out_dir_hint=pdf_path)),
        ("3D Trajectory",  lambda: figure_orbit_3d(history, meta=meta)),
        ("Ground Track",   lambda: figure_ground_track(history, meta=meta, ctx=ctx)),
        ("Orbital Elements", lambda: figure_elements_timeseries(t_days, elems)),
        ("Invariants",     lambda: figure_invariants(t_days, inv)),
        ("Phase Space",    lambda: figure_eomega(t_days, elems)),
        ("Altitude Profile", lambda: figure_altitude_with_events(t_days, alt_km, events=events)),
        ("Event Logs",     lambda: figure_events_table(events, t_days)),
        ("Conservation Drift", lambda: figure_relative_drift(t_days, inv)),
    ]

    if ctx is not None:
        pages.append(("Force Budget", lambda: figure_perturbation_magnitude(history, ctx)))

    logger.info("Generating PDF report: %s", pdf_path)

    with PdfPages(pdf_path) as pdf:
        # Metadata
        d = pdf.infodict()
        d["Title"] = "ST_LRPS Mission Analysis Report"
        d["Author"] = "ST_LRPS"
        d["Subject"] = "Orbital Mechanics Simulation Results"
        d["Keywords"] = "Astrodynamics, Moon, Orbit, Simulation, Python"
        d["CreationDate"] = _dt.datetime.now()

        # Render pages
        for desc, factory in pages:
            def _save_pdf(fig: plt.Figure) -> None:
                pdf.savefig(fig)

            _render_and_save(factory, _save_pdf, desc)

    return pdf_path



# ============================================================
# 4.       INTERNAL HELPERS (Directory Management)
# ============================================================

def _create_run_directory(base_dir: str, *, prefix: str = "run") -> str:
    """Create a unique timestamped run directory.

    Format: base_dir/prefix_YYYYMMDD_HHMMSS[_N]
    """
    base = Path(base_dir or ".").expanduser().resolve()
    try:
        base.mkdir(parents=True, exist_ok=True)
    except Exception:
        logger.exception("Could not create base dir: %s", base)
        return str(base)

    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    root = base / f"{prefix}_{stamp}"

    if not root.exists():
        root.mkdir(parents=True, exist_ok=True)
        return str(root)

    for i in range(2, 10_000):
        candidate = base / f"{prefix}_{stamp}_{i}"
        if not candidate.exists():
            candidate.mkdir(parents=True, exist_ok=True)
            return str(candidate)

    # Extremely unlikely fallback
    return str(root)



# ============================================================
# 5.       HIGH-LEVEL ORCHESTRATION (Public Entrypoint)
# ============================================================

def plot_all(
    history: Mapping[str, Any],
    out_dir: str,
    *,
    title_prefix: str = "ST_LRPS",
    save_pdf: Optional[bool] = None,
    save_png: Optional[bool] = None,
    dpi: Optional[int] = None,
    meta: Optional[Mapping[str, Any]] = None,
    ctx: Any = None,
    use_run_subdir: bool = True,
    visual_cfg: Any = None,
) -> Dict[str, Any]:
    """Orchestrate the full reporting pipeline (PNGs + PDF)."""

    # --- Resolve defaults from visual_cfg (if present) ---
    if visual_cfg is not None:
        if save_pdf is None:
            save_pdf = bool(getattr(visual_cfg, "save_pdf", True))
        if save_png is None:
            # support both names: save_pngs or save_png
            save_png = bool(getattr(visual_cfg, "save_pngs", getattr(visual_cfg, "save_png", True)))
        if dpi is None:
            try:
                dpi = int(getattr(visual_cfg, "default_dpi", 300))
            except Exception:
                dpi = 300

    # --- Conservative fallbacks ---
    save_pdf = True if save_pdf is None else bool(save_pdf)
    save_png = True if save_png is None else bool(save_png)
    dpi = 300 if dpi is None else int(dpi)

    # --- Output directory ---
    if use_run_subdir:
        run_dir = _create_run_directory(out_dir, prefix="run")
    else:
        run_dir = str(Path(out_dir or ".").expanduser().resolve())
        _ensure_dir(run_dir)

    logger.info("Starting report pipeline")
    logger.info("Output directory: %s", run_dir)

    # --- Metadata (inject directory hint for config discovery) ---
    meta2: Dict[str, Any] = dict(meta or {})
    meta2["_report_search_dir"] = run_dir
    if not meta2.get("lunar_map_path"):
        lunar_map = _find_default_lunar_surface_map()
        if lunar_map:
            meta2["lunar_map_path"] = lunar_map

    results: Dict[str, Any] = {
        "out_dir": run_dir,
        "pdf": None,
        "pngs": {},
        "status": "failed",
    }

    try:
        # --- Quicklook PNGs ---
        if save_png:
            pngs = save_quicklook_pngs(
                run_dir,
                history,
                prefix=title_prefix,
                meta=meta2,
                dpi=dpi,
                ctx=ctx,
            )
            results["pngs"] = pngs
            logger.info("Generated %d quicklook PNG(s)", len(pngs))

        # --- PDF report ---
        if save_pdf:
            safe_title = re.sub(r"[^A-Za-z0-9_\-]+", "_", title_prefix).strip("_") or "ST_LRPS"
            pdf_path = str(Path(run_dir) / f"{safe_title}_Report.pdf")

            final_pdf_path = make_report_pdf(
                pdf_path,
                history,
                meta=meta2,
                ctx=ctx,
            )
            results["pdf"] = final_pdf_path
            logger.info("PDF report saved: %s", Path(final_pdf_path).name)

        results["status"] = "success"
        logger.info("Report pipeline completed successfully")
        return results

    except Exception as e:
        logger.exception("Report pipeline failed")
        results["error"] = str(e)
        return results



# =============================================================================
# 6.                           SELF TEST
# =============================================================================

def _smoke_test() -> None:
    """Minimal smoke test to validate that the report pipeline runs end-to-end.

    This is not a unit test framework; it's a quick sanity check for developers.
    It intentionally avoids requiring ctx/ephemeris.
    """
    # Minimal circular-ish orbit state: y = [x,y,z,vx,vy,vz] with two points
    t_s = np.array([0.0, 10.0], dtype=float)
    y = np.array(
        [
            [1.737e6 + 100e3, 1.737e6 + 100e3],  # x (m)
            [0.0, 1.0],                          # y (m)
            [0.0, 0.0],                          # z (m)
            [0.0, 0.0],                          # vx (m/s)
            [1600.0, 1600.0],                    # vy (m/s)
            [0.0, 0.0],                          # vz (m/s)
        ],
        dtype=float,
    )

    history = {
        "t_s": t_s,
        "y": y,
        "mu_m3s2": 4.9048695e12,
        "R_body_m": 1.7374e6,
    }

    # Try generating a PDF into a temp directory
    tmp = Path(".") / "_report_smoke"
    tmp.mkdir(parents=True, exist_ok=True)

    meta = {"_report_search_dir": str(tmp)}
    out = plot_all(history, str(tmp), meta=meta, save_png=False, save_pdf=False, use_run_subdir=False)
    if out.get("status") != "success":
        raise RuntimeError(f"Smoke test failed: {out}")


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    _smoke_test()
    print("[REPORT] smoke test OK")


# =============================================================================
# 7.                           PUBLIC API
# =============================================================================

__all__ = [
    # High-level entrypoint
    "plot_all",
    # Output writers
    "make_report_pdf",
    "save_quicklook_pngs",
    # Page builders (useful for embedding)
    "figure_summary_page",
    "figure_run_config_page",
]
