"""Lunaris Mission Monitor — provenance-aware live/replay observation workspace.

Qt-free foundations (contract parsing, bounded store, registry, downsampling)
live in the submodules and are importable without PySide6. The Qt surface
(``MonitorPage`` / ``MonitorController``) is exported lazily so tests and
headless tools can use the data layer without pulling in a GUI stack.
"""

from __future__ import annotations

from typing import Any

__all__ = ["MonitorController", "MonitorPage"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from lunaris.ui.monitor import workspace

        return getattr(workspace, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
