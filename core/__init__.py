# LUNAR_SIMULATION/core/__init__.py
# -*- coding: utf-8 -*-
"""
LunarSim core simulation engine.

This package contains the runtime logic of the simulation:
- state helpers (Cartesian/COE utilities)
- dynamics engine (RHS builder)
- propagator (integration driver)
- events (event factories / helpers)

Public API is exposed lazily via PEP 562 (__getattr__) to keep import time low.

Typical usage:
    from core import DynamicsEngine, propagate
    from core import create_state_from_coe, ae_from_rp_ra, events
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

__version__ = "2.2.0"

__all__ = [
    "DynamicsEngine",
    "propagate",
    "build_events",
    "create_state_from_coe",
    "ae_from_rp_ra",
    "events",
]

if TYPE_CHECKING:
    from .dynamics import DynamicsEngine
    from .propagator import build_events, propagate
    from .state import ae_from_rp_ra, create_state_from_coe
    from . import events


# name -> (relative_module, attribute)
_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "DynamicsEngine": (".dynamics", "DynamicsEngine"),
    "propagate": (".propagator", "propagate"),
    "build_events": (".propagator", "build_events"),
    "create_state_from_coe": (".state", "create_state_from_coe"),
    "ae_from_rp_ra": (".state", "ae_from_rp_ra"),
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
