"""
Surrogate Gravity Runtime Adapter
=================================

Compatibility facade for ST-LRPS runtime helpers.

The production implementation is split by responsibility across sibling
modules in ``lunaris.surrogate.runtime``.  This module keeps the historical
``lunaris.surrogate.runtime.adapter`` import path stable for callers and tests.
"""

from __future__ import annotations

import logging

from lunaris.surrogate.runtime.artifact import (
    DEFAULT_ST_LRPS_RUNS_DIR,
    discover_st_lrps_model_dirs,
    find_checkpoint_for_st_lrps_run,
    find_latest_st_lrps_model_dir,
)
from lunaris.surrogate.runtime.device import nn, torch
from lunaris.surrogate.runtime.gravity_provider import (
    SurrogateGravityMetadata,
    SurrogateGravityModel,
)

logger = logging.getLogger(__name__)

if torch is not None and nn is not None:  # Preserve optional legacy globals when torch is installed.
    from lunaris.surrogate.runtime.networks import (  # noqa: F401
        MLP,
        FourierInputEmbedding,
        PhysicsNet,
        Sine,
        SirenMLP,
        _build_model_from_config,
    )

__all__ = [
    "DEFAULT_ST_LRPS_RUNS_DIR",
    "SurrogateGravityMetadata",
    "SurrogateGravityModel",
    "discover_st_lrps_model_dirs",
    "find_checkpoint_for_st_lrps_run",
    "find_latest_st_lrps_model_dir",
]
