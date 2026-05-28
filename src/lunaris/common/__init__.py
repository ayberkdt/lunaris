# LUNAR_SIMULATION/common/__init__.py
"""
Common package
==============

Shared, foundational building blocks for the project.

Public API
----------
- Flat exports from `common.constants` and `common.type_defs` are re-exported at
  package level.
- `constants` contains dependency-light project constants.
- `type_defs` contains public dataclasses, configuration containers, and result containers.
  Example:
      >>> from lunaris.common import MU_MOON, SpacecraftProps, GravityConfig

Lazy modules
------------
Heavier utilities are imported only when accessed as attributes, e.g.
`common.time_utils` or `common.math_utils`.
"""

from __future__ import annotations

import importlib
import sys
from typing import Any, Final

from . import constants, type_defs

# Lazy modules (attribute name -> short purpose)
_LAZY_MODULES: Final[dict[str, str]] = {
    "math_utils":       "Numerical helpers (vectors/quaternions/frames).",
    "time_utils":       "Time & calendar conversions (JD/MJD/J2000, ISO parsing).",
    "montecarlo_defs":  "Monte Carlo simulation configuration dataclasses.",
}


def _iter_public_names(mod) -> tuple[str, ...]:
    """Return module's public export list (from __all__), or empty."""
    names = getattr(mod, "__all__", ())
    if not names:
        return ()
    # Keep it strict: __all__ must be strings and must exist.
    out: list[str] = []
    for n in names:
        if not isinstance(n, str):
            raise TypeError(f"{mod.__name__}.__all__ must contain only strings; got {type(n)!r}")
        if not hasattr(mod, n):
            raise ImportError(f"{mod.__name__}.__all__ exports {n!r} but the attribute is missing")
        out.append(n)
    return tuple(out)


def _export_flat(*mods) -> tuple[str, ...]:
    """
    Re-export __all__ from given modules into this package namespace.
    Checks duplicates across modules.
    """
    exported: list[str] = []
    seen: dict[str, str] = {}
    g = globals()

    for mod in mods:
        for name in _iter_public_names(mod):
            prev = seen.get(name)
            if prev is not None:
                raise ImportError(f"Duplicate export {name!r} in {prev} and {mod.__name__}")
            seen[name] = mod.__name__
            exported.append(name)
            g[name] = getattr(mod, name)

    return tuple(exported)


# Flat, data-only exports
__all__: tuple[str, ...] = _export_flat(constants, type_defs)


def __getattr__(name: str) -> Any:
    """
    Lazy-load heavy utility modules on first access.

    - Unknown attribute -> AttributeError (with available lazy modules list).
    - Missing module file -> AttributeError (with guidance).
    - Missing internal dependency -> re-raise ModuleNotFoundError (do not mask).
    """
    if name not in _LAZY_MODULES:
        valid = ", ".join(sorted(_LAZY_MODULES))
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}. Available lazy modules: {valid}"
        )

    full_name = f"{__name__}.{name}"

    try:
        mod = importlib.import_module(f".{name}", __name__)
    except ModuleNotFoundError as e:
        # If it's not "the module file is missing", let it bubble (dependency missing).
        if getattr(e, "name", None) != full_name:
            raise

        hint = (
            f"Could not import lazy module {full_name!r}.\n"
            f"Purpose: {_LAZY_MODULES[name]}\n"
            "Likely causes:\n"
            f" - Missing file: common/{name}.py\n"
            " - Repo root not on PYTHONPATH\n"
            "Fix:\n"
            " - Run from project root, or install editable: pip install -e .\n"
            f"Python: {sys.executable}"
        )
        raise AttributeError(hint) from e

    globals()[name] = mod  # cache
    return mod


def __dir__() -> list[str]:
    """Expose public exports + lazy modules for IDE completion."""
    return sorted(set(globals()) | set(__all__) | set(_LAZY_MODULES))
