"""SciPy integrator method-name helpers."""

from __future__ import annotations

from typing import Any

from lunaris.common.integrator_methods import SCIPY_METHOD_NAMES
from lunaris.core.propagation.time_grid import _norm_method

# Compatibility export for callers that historically imported this private name.
_SCIPY_METHOD_NAMES = SCIPY_METHOD_NAMES

def _resolve_scipy_method(method: Any) -> str:
    """Return SciPy's exact spelling or reject an unsupported method."""
    token = _norm_method(method)
    try:
        return _SCIPY_METHOD_NAMES[token]
    except KeyError as exc:
        supported = ", ".join(sorted(_SCIPY_METHOD_NAMES))
        raise ValueError(
            f"Unsupported adaptive integration method: {method!r}. "
            f"Supported adaptive methods: {supported}."
        ) from exc


__all__ = ["_SCIPY_METHOD_NAMES", "_resolve_scipy_method"]
