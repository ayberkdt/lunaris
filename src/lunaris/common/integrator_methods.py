"""Dependency-light registry for supported propagation integrators.

The registry lives in :mod:`lunaris.common` so configuration validation can
reject a misspelled solver before the numerical engine is entered.  Core
integrator modules consume the same accepted token set; do not add a second
method list at a UI or CLI boundary.
"""

from __future__ import annotations

from typing import Any, Final

SCIPY_METHOD_NAMES: Final[dict[str, str]] = {
    "DOP853": "DOP853",
    "RK45": "RK45",
    "RK23": "RK23",
    "RADAU": "Radau",
    "BDF": "BDF",
    "LSODA": "LSODA",
}

# Each accepted fixed-step spelling resolves to the canonical stepper key.
FIXED_STEP_METHOD_ALIASES: Final[dict[str, str]] = {
    "VV": "VV",
    "VERLET": "VV",
    "STORMER_VERLET": "VV",
    "STÖRMER_VERLET": "VV",
    "LEAPFROG": "VV",
    "Y4": "Y4",
    "YOSHIDA4": "Y4",
    "Y6": "Y6",
    "YOSHIDA6": "Y6",
    "Y8": "Y8",
    "YOSHIDA8": "Y8",
    "PEFRL": "PEFRL",
    "RKN4": "RKN4",
    "RKN": "RKN4",
    "RK4": "RK4",
    "RK8": "RK8",
}


def normalize_integrator_method(method: Any) -> str:
    """Return the stable upper-case token used by the integrator registry."""
    return str(method).strip().upper().replace("-", "_").replace(" ", "_")


def is_supported_integrator_method(method: Any) -> bool:
    """Return whether ``method`` is an adaptive or fixed-step Lunaris method."""
    token = normalize_integrator_method(method)
    return token in SCIPY_METHOD_NAMES or token in FIXED_STEP_METHOD_ALIASES


def supported_integrator_methods() -> tuple[str, ...]:
    """Return accepted method tokens in deterministic display order."""
    return tuple(sorted(set(SCIPY_METHOD_NAMES) | set(FIXED_STEP_METHOD_ALIASES)))


__all__ = [
    "FIXED_STEP_METHOD_ALIASES",
    "SCIPY_METHOD_NAMES",
    "is_supported_integrator_method",
    "normalize_integrator_method",
    "supported_integrator_methods",
]
