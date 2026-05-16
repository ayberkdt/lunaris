# LUNAR_SIMULATION/analysis/__init__.py
# -*- coding: utf-8 -*-
"""
LunarSim Analysis & Reporting Package
=====================================

This package orchestrates the transition from raw simulation data to
publication-quality engineering reports.

Core Workflow:
--------------
1. **Post-Processing**: `process_simulation_results`
   - Normalizes integrator outputs (dense vs stepwise).
   - Computes derived data (Orbital Elements, Invariants, Events).
   - Detects physical capabilities (Active perturbations).

2. **Visual Styling**: `analysis.styling`
   - Enforces a consistent "Scientific Paper" aesthetic.
   - Manages color palettes (Semantic & Categorical).

3. **Plotting**: `analysis.plotting`
   - Pure functions that generate Matplotlib Figures from history dicts.
   - No side effects (does not save files).

4. **Reporting**: `analysis.report_manager`
   - `plot_all`: The high-level orchestrator.
   - Generates PNGs and compiles a multi-page PDF report.

Usage:
------
    >>> from analysis import process_simulation_results, plot_all
    >>>
    >>> # 1. Convert raw propagator result to dictionary
    >>> history = process_simulation_results(result, ctx=engine, cfg=config)
    >>>
    >>> # 2. Generate full PDF report
    >>> plot_all(history, out_dir="results/run_01")

"""

from __future__ import annotations

# 1. Post-Processing (The Logic Layer)
from .postprocess import (
    process_simulation_results,  # Main entry point for main.py
    compute_history,             # Core math logic
    summarize_history,           # Quick stats (min/max altitude)
)

# 2. Styling (The Look & Feel)
from .styling import (
    apply_rcparams,      # Setup matplotlib global style
    PlotStyle,           # Configuration dataclass
    COLORS,              # Raw palettes
    THEME,               # UI Theme (backgrounds, text)
    SEMANTIC,            # Physics-based colors (e.g. SRP is always red)
    get_series_color,    # Helper for series consistency
    get_accel_color,     # Helper for force consistency
)

# 3. Plotting Primitives (The Figure Factory)
# Exposed for users who want to render specific plots manually (e.g. in Notebooks)
from .plotting import (
    # Time Series
    figure_altitude_with_events,
    figure_elements_timeseries,
    figure_invariants,
    figure_relative_drift,
    figure_perturbation_magnitude,
    
    # Spatial / Maps
    figure_ground_track,
    figure_orbit_3d,
    figure_eomega,
    
    # Tables / Layouts
    draw_kv_table,
    draw_kv_block,
    draw_table,
    metrics_rows,
)

# 4. Reporting (The Orchestrator)
# - report_manager is an optional dependency during development.
# - If unavailable, we still allow importing `analysis`, but the reporting entrypoint
#   will raise a clear error when called.

def _missing_report_manager(*_args, **_kwargs):
    raise RuntimeError(
        "Reporting is not available because 'analysis.report_manager' could not be imported. "
        "Make sure report_manager.py exists and its imports are satisfied."
    )

try:
    from .report_manager import plot_all as plot_all  # type: ignore
except ImportError:
    plot_all = _missing_report_manager  # type: ignore


# 5. Monte Carlo Analysis & Plotting
try:
    from .mc_analysis import (
        compute_mc_statistics,
        compute_ensemble_statistics,
        compute_error_ellipsoids,
        compute_impact_statistics,
        compute_oe_dispersion,
        MCStatistics,
        EnsembleStatistics,
        ErrorEllipsoids,
        ImpactStatistics,
        OEDispersion,
    )
    from .mc_plotting import (
        plot_mc_report,
        plot_altitude_envelope,
        plot_covariance_tubes_3d,
        plot_position_covariance_history,
        plot_impact_map,
        plot_impact_time_histogram,
        plot_oe_dispersion,
    )
    _MC_AVAILABLE = True
except ImportError:
    _MC_AVAILABLE = False


# =============================================================================
# PUBLIC API DEFINITION
# =============================================================================

__all__ = [
    # -- Primary Workflow --
    "process_simulation_results",
    "plot_all",
    
    # -- Data & Math --
    "compute_history",
    "summarize_history",
    
    # -- Styling --
    "apply_rcparams",
    "PlotStyle",
    "THEME",
    "COLORS",
    "SEMANTIC",
    "get_series_color",
    "get_accel_color",
    
    # -- Figures (Time Series) --
    "figure_altitude_with_events",
    "figure_elements_timeseries",
    "figure_invariants",
    "figure_relative_drift",
    "figure_perturbation_magnitude",
    
    # -- Figures (Spatial) --
    "figure_ground_track",
    "figure_orbit_3d",
    "figure_eomega",
    
    # -- Figures (Tables) --
    "draw_kv_table",
    "draw_kv_block",
    "draw_table",
    "metrics_rows",

    # -- Monte Carlo (conditionally available) --
    "compute_mc_statistics",
    "compute_ensemble_statistics",
    "compute_error_ellipsoids",
    "compute_impact_statistics",
    "compute_oe_dispersion",
    "MCStatistics",
    "EnsembleStatistics",
    "ErrorEllipsoids",
    "ImpactStatistics",
    "OEDispersion",
    "plot_mc_report",
    "plot_altitude_envelope",
    "plot_covariance_tubes_3d",
    "plot_position_covariance_history",
    "plot_impact_map",
    "plot_impact_time_histogram",
    "plot_oe_dispersion",
]