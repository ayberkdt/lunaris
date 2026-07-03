"""CLI-facing batch command surfaces.

The ``lunaris-batch`` console script resolves through this module.
"""

from __future__ import annotations

from lunaris.batch.engine import batch_entry
from lunaris.cli.batch_runner import main

__all__ = ["batch_entry", "main"]
