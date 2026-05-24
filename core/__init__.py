# ST_LRPS/core/__init__.py
# -*- coding: utf-8 -*-
"""
ST_LRPS core simulation engine.

This package contains the runtime logic of the simulation:
- state helpers (Cartesian/COE utilities)
- dynamics engine (RHS builder)
- propagator (integration driver)
- events (event factories / helpers)

Public API is exposed lazily via PEP 562 (__getattr__) to keep import time low.

Typical usage:
    from core import DynamicsEngine, propagate
    from core import create_state_from_keplerian, events
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

__version__ = "2.2.0"

__all__ = [
    "DynamicsEngine",
    "propagate",
    "build_events",
    "create_state_from_keplerian",
    "calculate_ae_from_altitudes",
    "calculate_ae_from_radii",
    "calculate_altitudes_from_ae",
    "events",
]

if TYPE_CHECKING:
    from .dynamics import DynamicsEngine
    from .propagator import build_events, propagate
    from .state import (
        create_state_from_keplerian,
        calculate_ae_from_altitudes,
        calculate_ae_from_radii,
        calculate_altitudes_from_ae,
    )
    from . import events


# name -> (relative_module, attribute)
_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "DynamicsEngine": (".dynamics", "DynamicsEngine"),
    "propagate": (".propagator", "propagate"),
    "build_events": (".propagator", "build_events"),
    "create_state_from_keplerian": (".state", "create_state_from_keplerian"),
    "calculate_ae_from_altitudes": (".state", "calculate_ae_from_altitudes"),
    "calculate_ae_from_radii": (".state", "calculate_ae_from_radii"),
    "calculate_altitudes_from_ae": (".state", "calculate_altitudes_from_ae"),
}


def __getattr__(name: str) -> Any:
    if name == "events":
        return import_module(".events", __name__)

    spec = _LAZY_ATTRS.get(name)
    if spec is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    mod_name, attr_name = spec
    mod = import_module(mod_name, __name__)
    return getattr(mod, attr_name)


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(__all__))
