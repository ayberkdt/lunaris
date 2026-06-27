"""Package facade for the split dynamics implementation."""

from __future__ import annotations

from . import engine as _impl

for _name in dir(_impl):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_impl, _name)

__all__ = list(getattr(_impl, "__all__", ()))
