"""Retired compatibility module for the old physics surrogate path.

The production ST-LRPS runtime adapter lives at
``lunaris.surrogate.runtime_adapter``.  This module intentionally does not
import or re-export that adapter so the ``physics`` layer has no dependency on
the optional surrogate subsystem.
"""

from __future__ import annotations

from typing import Any, Final

_CANONICAL_MODULE: Final[str] = "lunaris.surrogate.runtime_adapter"

__all__: tuple[str, ...] = ()


def __getattr__(name: str) -> Any:
    raise AttributeError(
        f"{__name__}.{name} was retired. "
        f"Import {name} from {_CANONICAL_MODULE} instead."
    )


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
