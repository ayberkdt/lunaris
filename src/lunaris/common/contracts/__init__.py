"""Shared lightweight runtime contracts.

These constants define schema versions and required archive keys for the
general propagation and batch layers. ST-LRPS keeps its richer artifact and
training contracts under ``lunaris.surrogate.st_lrps``.
"""

from lunaris.common.contracts.batch_archive import (
    BATCH_ARCHIVE_SCHEMA_VERSION,
    REQUIRED_ARCHIVE_V2_ARRAYS,
    REQUIRED_ARCHIVE_V2_FIELDS,
)
from lunaris.common.contracts.checkpoint import CHECKPOINT_SCHEMA_VERSION
from lunaris.common.contracts.diagnostics import PROPAGATION_DIAGNOSTICS_SCHEMA_VERSION

__all__ = [
    "BATCH_ARCHIVE_SCHEMA_VERSION",
    "CHECKPOINT_SCHEMA_VERSION",
    "PROPAGATION_DIAGNOSTICS_SCHEMA_VERSION",
    "REQUIRED_ARCHIVE_V2_ARRAYS",
    "REQUIRED_ARCHIVE_V2_FIELDS",
]
