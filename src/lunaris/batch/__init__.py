"""Batch ensemble propagation package.

Historical Monte Carlo names are preserved for public API compatibility;
``batch`` is the internal package boundary for ensemble orchestration,
sampling, storage, memory policy, and provenance helpers.
"""

from __future__ import annotations

from lunaris.batch.engine import MonteCarloEngine, batch_entry, mc_entry
from lunaris.batch.sampling import (
    generate_standard_normal_design,
    sample_initial_states,
    sample_spacecraft_props,
)
from lunaris.batch.storage import HDF5TrajectoryView, load_mc_result

__all__ = [
    "MonteCarloEngine",
    "generate_standard_normal_design",
    "sample_initial_states",
    "sample_spacecraft_props",
    "HDF5TrajectoryView",
    "load_mc_result",
    "mc_entry",
    "batch_entry",
]
