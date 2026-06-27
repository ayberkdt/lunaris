"""Reserved surface for a future Numba RHS extraction.

The jitted RHS closures intentionally remain inside ``engine.py`` for now so
this refactor does not alter hot-loop object boundaries.
"""

from __future__ import annotations

__all__: list[str] = []
