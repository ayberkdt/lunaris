"""Batch ensemble propagation package.

``lunaris.batch`` is the package boundary for ensemble orchestration,
sampling, storage, memory policy, backend policy, and provenance helpers.
"""

from __future__ import annotations

from lunaris.batch.engine import BatchPropagationEngine, batch_entry
from lunaris.batch.sampling import (
    generate_standard_normal_design,
    sample_initial_states,
    sample_spacecraft_props,
)
from lunaris.batch.storage import HDF5TrajectoryView, load_batch_result
from lunaris.batch.types import BatchPropagationConfig, BatchPropagationResult

__all__ = [
    "BatchPropagationConfig",
    "BatchPropagationEngine",
    "BatchPropagationResult",
    "generate_standard_normal_design",
    "sample_initial_states",
    "sample_spacecraft_props",
    "HDF5TrajectoryView",
    "load_batch_result",
    "batch_entry",
]
