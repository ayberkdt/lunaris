# ST_LRPS/analysis/__init__.py
# -*- coding: utf-8 -*-
"""
ST_LRPS Analysis & Reporting Package
=====================================

This package orchestrates the transition from raw simulation data to
publication-quality engineering reports.

Usage:
------
    >>> from analysis import process_simulation_results, plot_all
    >>> history = process_simulation_results(result, ctx=engine, cfg=config)
    >>> plot_all(history, out_dir="results/run_01")
"""

from __future__ import annotations

# 1. Post-Processing
from .postprocess import (
    process_simulation_results,
    compute_history,
    summarize_history,
)

# 2. Styling
from .reporting.styling import (
    apply_rcparams,
    PlotStyle,
    COLORS,
    THEME,
    SEMANTIC,
    get_series_color,
    get_accel_color,
)

# 3. Plotting
from .reporting.plotting import (
    figure_altitude_with_events,
    figure_elements_timeseries,
    figure_invariants,
    figure_relative_drift,
    figure_perturbation_magnitude,
    figure_ground_track,
    figure_orbit_3d,
    figure_eomega,
    draw_kv_table,
    draw_kv_block,
    draw_table,
    metrics_rows,
)

# 4. Reporting
from .reporting.manager import plot_all

# 5. Monte Carlo
try:
    from .monte_carlo.statistics import (
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
    from .monte_carlo.plotting import (
        plot_mc_report,
        plot_altitude_envelope,
        plot_covariance_tubes_3d,
        plot_position_covariance_history,
        plot_impact_map,
        plot_impact_time_histogram,
        plot_oe_dispersion as plot_mc_oe_dispersion,
    )
    _MC_AVAILABLE = True
except ImportError:
    _MC_AVAILABLE = False


__all__ = [
    "process_simulation_results",
    "compute_history",
    "summarize_history",
    "plot_all",
]

# Expose styling and plotting to __all__
__all__.extend([
    "apply_rcparams", "PlotStyle", "THEME", "COLORS", "SEMANTIC",
    "get_series_color", "get_accel_color",
    "figure_altitude_with_events", "figure_elements_timeseries",
    "figure_invariants", "figure_relative_drift", "figure_perturbation_magnitude",
    "figure_ground_track", "figure_orbit_3d", "figure_eomega",
    "draw_kv_table", "draw_kv_block", "draw_table", "metrics_rows",
])

if _MC_AVAILABLE:
    __all__.extend([
        "compute_mc_statistics", "compute_ensemble_statistics",
        "compute_error_ellipsoids", "compute_impact_statistics", "compute_oe_dispersion",
        "MCStatistics", "EnsembleStatistics", "ErrorEllipsoids",
        "ImpactStatistics", "OEDispersion",
        "plot_mc_report", "plot_altitude_envelope", "plot_covariance_tubes_3d",
        "plot_position_covariance_history", "plot_impact_map",
        "plot_impact_time_histogram", "plot_mc_oe_dispersion",
    ])