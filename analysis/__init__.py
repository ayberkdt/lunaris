# ST_LRPS/analysis/__init__.py
# -*- coding: utf-8 -*-
"""
ST_LRPS Analysis Package
=========================

This package orchestrates the transition from raw simulation data to
normalized history outputs. Sub-packages handle reporting and statistics.

Usage:
------
    >>> from analysis import process_simulation_results
    >>> from analysis.reporting.manager import plot_all
    >>> history = process_simulation_results(result, ctx=engine, cfg=config)
    >>> plot_all(history, out_dir="results/run_01")
"""

from __future__ import annotations

# Expose only core post-processing entry points at the root.
from .postprocess import (
    process_simulation_results,
    compute_history,
    summarize_history,
)

__all__ = [
    "process_simulation_results",
    "compute_history",
    "summarize_history",
]