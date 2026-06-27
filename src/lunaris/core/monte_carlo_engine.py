# lunaris.core.monte_carlo_engine
"""Compatibility shim for the historical Monte Carlo engine import path.

The canonical implementation now lives under ``lunaris.batch``. Console
entry points and downstream imports continue to resolve through this module.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from lunaris.batch import storage as _storage
from lunaris.batch.engine import (
    BatchPropagationEngine,
    MonteCarloEngine,
    batch_entry,
    mc_entry,
)
from lunaris.batch.memory_policy import (
    _available_host_memory_bytes,
)
from lunaris.batch.sampling import (
    generate_standard_normal_design,
    sample_initial_states,
    sample_spacecraft_props,
)
from lunaris.batch.storage import (
    HDF5TrajectoryView,
    load_mc_result,
)
from lunaris.batch.types import BatchPropagationConfig, BatchPropagationResult


def _resolve_result_storage(mc_cfg: Any, n_steps: int) -> tuple[str, int, int]:
    """Compatibility proxy preserving monkeypatches on this legacy module."""
    return _storage._resolve_result_storage(
        mc_cfg,
        n_steps,
        available_host_memory_bytes=_available_host_memory_bytes,
    )


def _allocate_result_buffer(
    storage_mode: str,
    writer_buffer: Any,
    shape: tuple[int, int, int],
) -> np.ndarray | None:
    """Compatibility proxy preserving monkeypatches on this legacy module."""
    if storage_mode == "disk":
        return None
    if storage_mode != "memory":
        raise ValueError(f"Unsupported result storage mode: {storage_mode!r}")
    if isinstance(writer_buffer, np.ndarray):
        if tuple(writer_buffer.shape) != tuple(shape):
            raise ValueError(
                f"Writer result buffer must have shape {shape}, "
                f"got {writer_buffer.shape}"
            )
        return writer_buffer
    return np.empty(shape, dtype=np.float64)


_COMPAT_PRIVATE_EXPORT_NAMES = (
    "_BACKEND_DISPLAY_NAMES",
    "REQUIRED_ARCHIVE_V2_FIELDS",
    "_HOST_MEMORY_SAFETY_FACTOR",
    "_available_host_memory_bytes",
    "_active_physics_capabilities",
    "_decode_archive_metadata",
    "_decode_metadata_value",
    "_metadata_value_to_jsonable",
    "_sha256_file",
    "_build_ephemeris_manager",
    "_impact_positions_fixed",
    "_need_body_vectors",
    "_need_ephemeris",
    "_state_to_array",
    "_surface_topography_requested",
    "_sobol_size_note",
    "_HDF5Writer",
    "_infer_valid_mask_from_dataset",
    "_make_writer",
    "_NPZWriter",
    "_validate_archive_v2_manifest",
)


__all__ = [
    "BatchPropagationConfig",
    "BatchPropagationEngine",
    "BatchPropagationResult",
    "MonteCarloEngine",
    "generate_standard_normal_design",
    "sample_initial_states",
    "sample_spacecraft_props",
    "HDF5TrajectoryView",
    "load_mc_result",
    "mc_entry",
    "batch_entry",
]
__all__.extend(_COMPAT_PRIVATE_EXPORT_NAMES)
