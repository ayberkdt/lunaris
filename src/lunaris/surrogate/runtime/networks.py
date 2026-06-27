"""Responsibility-split re-export surface for ``lunaris.surrogate.runtime.adapter``."""

from __future__ import annotations

from lunaris.surrogate.runtime.adapter import (
    _build_model_from_config,
    _extract_state_dict,
    _load_checkpoint,
)

__all__ = [
    '_build_model_from_config',
    '_extract_state_dict',
    '_load_checkpoint',
]
