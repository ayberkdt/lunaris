"""Stable public Python facade for Lunaris workflows.

This module intentionally resolves implementation objects lazily so importing
``lunaris.api`` stays light in core-only and documentation environments.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

__all__ = [
    "BatchPropagationConfig",
    "BatchPropagationEngine",
    "BatchPropagationResult",
    "DynamicsEngine",
    "load_default_config",
    "propagate",
    "replace_sim_config",
]

if TYPE_CHECKING:
    from lunaris.batch import BatchPropagationEngine
    from lunaris.common.batch_defs import (
        BatchPropagationConfig,
        BatchPropagationResult,
    )
    from lunaris.core.config import load_default_config, replace_sim_config
    from lunaris.core.dynamics import DynamicsEngine
    from lunaris.core.propagation.propagator import propagate


_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "load_default_config": ("lunaris.core.config", "load_default_config"),
    "replace_sim_config": ("lunaris.core.config", "replace_sim_config"),
    "DynamicsEngine": ("lunaris.core.dynamics", "DynamicsEngine"),
    "propagate": ("lunaris.core.propagation.propagator", "propagate"),
    "BatchPropagationConfig": ("lunaris.common.batch_defs", "BatchPropagationConfig"),
    "BatchPropagationResult": ("lunaris.common.batch_defs", "BatchPropagationResult"),
    "BatchPropagationEngine": ("lunaris.batch", "BatchPropagationEngine"),
}


def __getattr__(name: str) -> Any:
    spec = _LAZY_ATTRS.get(name)
    if spec is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = spec
    module = import_module(module_name)
    return getattr(module, attr_name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
