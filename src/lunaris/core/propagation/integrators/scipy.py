"""SciPy integrator method-name helpers."""

from __future__ import annotations

from typing import Any

from lunaris.core.propagation.time_grid import _norm_method


# exact spelling SciPy expects so an adaptive selection never raises.
_SCIPY_METHOD_NAMES: dict[str, str] = {
    "DOP853": "DOP853",
    "RK45": "RK45",
    "RK23": "RK23",
    "RADAU": "Radau",
    "BDF": "BDF",
    "LSODA": "LSODA",
}

def _resolve_scipy_method(method: Any) -> str:
    """Return the exact SciPy ``solve_ivp`` method name (defaulting to DOP853)."""
    return _SCIPY_METHOD_NAMES.get(_norm_method(method), "DOP853")


__all__ = ["_SCIPY_METHOD_NAMES", "_resolve_scipy_method"]
