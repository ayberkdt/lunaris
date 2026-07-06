"""Compat shim: the canonical ST-LRPS runtime lives in ``canonical_runtime`` (R11).

Every public name (``SurrogateForceModel``, ``load_surrogate_force_model``,
``SUPPORTED_RUNTIME_FRAME``, ...) resolves to the same objects as
:mod:`lunaris.surrogate.st_lrps.runtime.canonical_runtime`; existing imports
and monkeypatches through this module keep working.

This is a DYNAMIC fold on purpose: static ``from canonical_runtime import X``
re-exports get stripped by ``ruff --fix`` as unused imports (this broke CI
three times on earlier modular refactors), so the shim forwards attribute
access at runtime instead.
"""

from __future__ import annotations

from typing import Any

from lunaris.surrogate.st_lrps.runtime import canonical_runtime as _canonical


def __getattr__(name: str) -> Any:
    return getattr(_canonical, name)


def __dir__() -> list[str]:
    return sorted(set(dir(_canonical)))
