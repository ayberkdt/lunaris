# lunaris/common/montecarlo_defs.py
"""Compatibility facade for historical Monte Carlo type names.

The canonical batch/ensemble propagation dataclasses live in
``lunaris.common.batch_defs``.  This module remains as the stable historical
import path for downstream scripts, old archives, and tests that still import
``MonteCarloConfig`` / ``MCRunResult`` from here.
"""

from __future__ import annotations

from lunaris.common.batch_defs import (
    BATCH_SAMPLING_METHODS,
    BatchPropagationConfig,
    BatchPropagationResult,
    MCRunResult,
    MonteCarloConfig,
    SpacecraftUncertainty,
    StateUncertainty,
    build_mc_output_grid,
    validate_st_lrps_model_dir,
)

__all__ = [
    "build_mc_output_grid",
    "BATCH_SAMPLING_METHODS",
    "StateUncertainty",
    "SpacecraftUncertainty",
    "BatchPropagationConfig",
    "BatchPropagationResult",
    "MonteCarloConfig",
    "MCRunResult",
    "validate_st_lrps_model_dir",
]
