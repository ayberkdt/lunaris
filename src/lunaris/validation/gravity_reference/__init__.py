"""External-reference validation for lunar gravity.

This package separates fixed-field validation from trajectory validation. Field
validation compares Lunaris spherical-harmonic evaluations at identical
Moon-fixed coordinates against immutable reference values or an independent
oracle. Trajectory validation is contract-first and cannot produce PASS unless a
complete external reference trajectory is present.
"""

from __future__ import annotations

from .manifest import (
    FIELD_SCHEMA_VERSION,
    TRAJECTORY_SCHEMA_VERSION,
    HashMismatchError,
    ManifestError,
    load_field_manifest,
    load_trajectory_manifest,
)

__all__ = (
    "FIELD_SCHEMA_VERSION",
    "TRAJECTORY_SCHEMA_VERSION",
    "HashMismatchError",
    "ManifestError",
    "load_field_manifest",
    "load_trajectory_manifest",
)

