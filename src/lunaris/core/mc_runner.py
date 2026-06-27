#!/usr/bin/env python
"""Compatibility wrapper for the historical batch runner script path.

The canonical implementation lives in ``lunaris.cli.batch_runner``.  This file
is kept so old imports and the desktop UI's script-path launcher continue to
work while the CLI-facing code lives with the other console entry points.
"""

from __future__ import annotations

from lunaris.cli import batch_runner as _impl

__all__: list[str] = []

for _name in dir(_impl):
    if not (_name.startswith("__") and _name.endswith("__")):
        globals()[_name] = getattr(_impl, _name)
        __all__.append(_name)

__all__.sort()


if __name__ == "__main__":
    raise SystemExit(_impl.main())
