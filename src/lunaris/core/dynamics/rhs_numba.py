"""Numba RHS implementation surface.

The first structural split keeps the jitted RHS closures inside
``engine.py`` so no Python objects are introduced into the hot loop.
"""

from __future__ import annotations

__all__: list[str] = []
