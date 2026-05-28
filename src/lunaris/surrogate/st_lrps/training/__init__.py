# st_lrps/training/__init__.py
"""ST-LRPS training subpackage: config, CLI, engine, losses, and metrics.

Import-safe: importing this package must not pull torch or the training engine.
Import the specific submodule you need (e.g. ``st_lrps.training.engine``).
The canonical training entry point is ``python -m lunaris.surrogate.st_lrps.training.cli``.
"""

from __future__ import annotations

__all__: list[str] = []
