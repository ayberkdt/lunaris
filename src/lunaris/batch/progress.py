"""Progress callback helpers for batch propagation."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

ProgressCallback = Callable[[dict[str, Any]], None]

__all__ = ["ProgressCallback"]
