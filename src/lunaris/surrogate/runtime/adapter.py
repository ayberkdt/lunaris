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
    _find_checkpoint_for_run,
    _is_valid_surrogate_run,
    _looks_like_lunar_run,
    discover_st_lrps_model_dirs,
    find_checkpoint_for_st_lrps_run,
    find_latest_st_lrps_model_dir,
)
from lunaris.surrogate.runtime.device import _TORCH_IMPORT_ERROR, _require_torch, nn, torch
from lunaris.surrogate.runtime.gravity_provider import SurrogateGravityMetadata, SurrogateGravityModel
from lunaris.surrogate.runtime.metadata import (
    _config_path_value,
    _extract_degree_metadata,
    _resolve_baseline_gravity_path,
)
from lunaris.surrogate.runtime.networks import (
    _build_model_from_config,
    _extract_state_dict,
    _load_checkpoint,
)
from lunaris.surrogate.runtime.scalers import (
    _load_scaler_bundle,
    _normalize_scale_mapping,
    _ScalerBundle,
    _ScaleVector,
)

logger = logging.getLogger(__name__)

if torch is not None and nn is not None:  # Preserve optional legacy globals when torch is installed.
    from lunaris.surrogate.runtime.networks import (  # noqa: F401
        FourierInputEmbedding,
        MLP,
        PhysicsNet,
        Sine,
        SirenMLP,
    )

__all__ = [
    "DEFAULT_ST_LRPS_RUNS_DIR",
    "SurrogateGravityMetadata",
    "SurrogateGravityModel",
    "discover_st_lrps_model_dirs",
    "find_checkpoint_for_st_lrps_run",
    "find_latest_st_lrps_model_dir",
]
