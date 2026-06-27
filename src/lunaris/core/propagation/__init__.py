"""Package facade for ``lunaris.core.propagation.propagator``.

The implementation is kept in one module for the first structural split
so behavior and monkeypatch semantics stay stable.
"""

from __future__ import annotations

from . import propagator as _impl

for _name in dir(_impl):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_impl, _name)

__all__ = list(getattr(_impl, "__all__", ()))
