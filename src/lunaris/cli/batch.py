"""CLI-facing batch command surfaces.

The historical console scripts still point at ``lunaris.core.monte_carlo_engine``.
This module gives the CLI package a named batch surface without changing those
entry points.
"""

from __future__ import annotations

from lunaris.batch.engine import batch_entry, mc_entry

__all__ = ["batch_entry", "mc_entry"]
