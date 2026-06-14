"""Compatibility access to the canonical surrogate runtime adapter.

New code must import :mod:`lunaris.surrogate.runtime_adapter`.  This module is
kept temporarily so existing callers continue to work while the public import
path is migrated.  The lazy lookup deliberately avoids making ``physics`` own
or import the surrogate subsystem at module-import time.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_ADAPTER_MODULE = "lunaris.surrogate.runtime_adapter"


def __getattr__(name: str) -> Any:
    module = import_module(_ADAPTER_MODULE)
    try:
        return getattr(module, name)
    except AttributeError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc


def __dir__() -> list[str]:
    module = import_module(_ADAPTER_MODULE)
    return sorted(set(globals()) | set(dir(module)))
