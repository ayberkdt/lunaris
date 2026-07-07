"""Batch trajectory archive schema contract."""

from __future__ import annotations

BATCH_ARCHIVE_SCHEMA_VERSION = 2

REQUIRED_ARCHIVE_V2_FIELDS: tuple[str, ...] = (
    "archive_schema_version",
    "n_samples",
    "seed",
    "duration_s",
    "output_dt_s",
    "backend",
    "requested_batch_backend",
    "actual_batch_backend",
    "batch_backend",
    "detect_impact",
    "compute_impact_statistics",
)

REQUIRED_ARCHIVE_V2_ARRAYS: tuple[str, ...] = (
    "t",
    "Y",
    "sc_samples",
    "impact_flags",
    "t_impact",
    "valid_mask",
    "impact_position_inertial_m",
    "impact_position_fixed_m",
)

__all__ = [
    "BATCH_ARCHIVE_SCHEMA_VERSION",
    "REQUIRED_ARCHIVE_V2_ARRAYS",
    "REQUIRED_ARCHIVE_V2_FIELDS",
]
