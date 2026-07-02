#!/usr/bin/env python
"""Compatibility wrapper for the historical batch runner script path.

The canonical implementation lives in ``lunaris.cli.batch_runner``.  This file
is kept so old imports and the desktop UI's script-path launcher continue to
work while the CLI-facing code lives with the other console entry points.

The fold is lazy (PEP 562 module ``__getattr__``): importing
``lunaris.core.mc_runner`` must not pull the CLI layer into ``core`` at import
time. Names resolve on first access and are cached in the module globals, so
``from ... import X``, ``hasattr``, and monkeypatch paths behave exactly as
they did with the old eager fold. The remaining call-time edge to the CLI is
declared in the import-linter contract's ``ignore_imports``.
"""

from __future__ import annotations

from typing import Any


def _impl():
    from lunaris.cli import batch_runner

    return batch_runner


def __getattr__(name: str) -> Any:
    if name == "__all__":
        value = sorted(
            n for n in dir(_impl()) if not (n.startswith("__") and n.endswith("__"))
        )
        globals()["__all__"] = value
        return value
    if name.startswith("__") and name.endswith("__"):
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(_impl(), name)
    globals()[name] = value  # cache: snapshot semantics + stable monkeypatch target
    return value


def __dir__() -> list[str]:
    impl_names = (
        n for n in dir(_impl()) if not (n.startswith("__") and n.endswith("__"))
    )
    return sorted(set(globals()) | set(impl_names))


if __name__ == "__main__":
    raise SystemExit(_impl().main())
