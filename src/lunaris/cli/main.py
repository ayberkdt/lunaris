"""
Lunaris - main entry point (STRICT, config.py + common.type_defs aligned)

Contract
--------
- config.py (SimConfig) is the single source of truth (SSOT).
- CLI helpers live in focused modules under ``lunaris.cli``.
- main.py remains the public compatibility surface for ``lunaris`` and tests.
- NO backward-compat aliases / legacy flags / schema-drift adapters.
"""

from __future__ import annotations

from lunaris.cli.common_args import (
    apply_args_to_config,
    init_surface_provider,
    need_ephemeris,
    parse_adaptive_table,
    parse_tide_bodies,
    resolve_orbit_elements,
    str2bool,
)
from lunaris.cli.options import parse_args, validate_args
from lunaris.cli.run import _y0_to_array, init_ephemeris, main, main_entry
from lunaris.cli.summary import _extract_rv6, median_dt, print_summary
from lunaris.core.config import SimConfig, load_default_config

# ST-LRPS model-dir validation lives in lunaris.cli.options and delegates to
# validate_st_lrps_model_dir; this facade keeps the historical source surface.

__all__ = [
    "SimConfig",
    "load_default_config",
    "str2bool",
    "parse_tide_bodies",
    "parse_adaptive_table",
    "resolve_orbit_elements",
    "init_surface_provider",
    "need_ephemeris",
    "apply_args_to_config",
    "parse_args",
    "validate_args",
    "init_ephemeris",
    "_extract_rv6",
    "print_summary",
    "median_dt",
    "_y0_to_array",
    "main",
    "main_entry",
]

if __name__ == "__main__":
    raise SystemExit(main_entry())
