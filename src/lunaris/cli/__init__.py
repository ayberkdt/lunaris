# cli/__init__.py
"""Lunaris command-line helper package.

Houses small, import-safe helpers shared by the CLI entry points
(``main.py`` and ``batch_runner.py``). Importing this package must not pull in
heavy runtime dependencies (numba / spiceypy / torch / loaders / core).
"""

from __future__ import annotations

__all__ = ["batch", "common_args", "options", "run", "summary"]
