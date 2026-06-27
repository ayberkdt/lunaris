"""CLI-facing batch command surfaces.

Console scripts resolve through this module; historical imports through
``lunaris.core.monte_carlo_engine`` remain available as compatibility shims.
"""

from __future__ import annotations

from lunaris.batch.engine import batch_entry, mc_entry
from lunaris.cli.batch_runner import main

__all__ = ["batch_entry", "mc_entry", "main"]
