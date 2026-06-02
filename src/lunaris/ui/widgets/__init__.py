# -*- coding: utf-8 -*-
"""
Reusable composite widgets for the Lunaris Mission Studio desktop UI.

Currently exposes the buffered :class:`ExecutionLogPanel` used as the bottom
"Execution Console" of the main window.
"""

from __future__ import annotations

from lunaris.ui.widgets.log_panel import (
    MAX_LOG_LINES,
    ExecutionLogPanel,
    LogEntry,
)

__all__ = ["ExecutionLogPanel", "LogEntry", "MAX_LOG_LINES"]
