"""Scaler normalization for ST-LRPS surrogate runtime artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class _ScaleVector:
    """
    Scaling parameters for one quantity.

    ``scale`` may be either:
    - shape ``(1,)``  -> isotropic / scalar scaling
    - shape ``(N,)``  -> legacy per-axis scaling
    """

    mean: np.ndarray
    scale: np.ndarray

    @property
    def is_isometric(self) -> bool:
        return int(self.scale.size) == 1


@dataclass(frozen=True, slots=True)
class _ScalerBundle:
    """Normalized view of the artifact scaler pack."""

    x: _ScaleVector
    u: _ScaleVector
    a: _ScaleVector | None


def _normalize_scale_mapping(mapping: dict[str, Any], expected_dim: int, name: str) -> _ScaleVector:
    """
    Normalize legacy/new scaler JSON into a common in-memory representation.

    Legacy runs store ``std`` arrays while newer residual runs store a single
    ``scale`` value. The runtime keeps both formats alive so older experiments
    remain usable from the desktop app.
    """

    if "mean" not in mapping:
        raise ValueError(f"Scaler entry '{name}' is missing 'mean'.")

    mean = np.asarray(mapping["mean"], dtype=np.float64).reshape(-1)
    if mean.size != expected_dim:
        raise ValueError(
            f"Scaler entry '{name}.mean' must have {expected_dim} values, got {mean.size}."
        )

    raw_scale = mapping.get("scale", mapping.get("std"))
    if raw_scale is None:
        raise ValueError(f"Scaler entry '{name}' is missing 'scale'/'std'.")

    scale = np.asarray(raw_scale, dtype=np.float64).reshape(-1)
    if scale.size not in (1, expected_dim):
        raise ValueError(
            f"Scaler entry '{name}.scale' must be scalar or length {expected_dim}, got {scale.size}."
        )
    if np.any(~np.isfinite(scale)) or np.any(scale <= 0.0):
        raise ValueError(f"Scaler entry '{name}.scale' must contain positive finite values.")

    return _ScaleVector(mean=mean, scale=scale)


def _load_scaler_bundle(model_dir: Path, checkpoint_obj: dict[str, Any]) -> _ScalerBundle:
    """
    Load scaler metadata from checkpoint first, then ``scaler.json`` as fallback.

    Checkpoints tend to be the most self-consistent source because they are
    written at training time together with the model weights.
    """

    scaler_obj = checkpoint_obj.get("scaler")
    if not isinstance(scaler_obj, dict):
        scaler_path = model_dir / "scaler.json"
        if not scaler_path.is_file():
            raise FileNotFoundError(f"Scaler artifact not found: {scaler_path}")
        scaler_obj = json.loads(scaler_path.read_text(encoding="utf-8"))

    x = _normalize_scale_mapping(dict(scaler_obj.get("x", {})), expected_dim=3, name="x")
    u = _normalize_scale_mapping(dict(scaler_obj.get("u", {})), expected_dim=1, name="u")
    a_raw = scaler_obj.get("a")
    a = (
        _normalize_scale_mapping(dict(a_raw), expected_dim=3, name="a")
        if isinstance(a_raw, dict)
        else None
    )
    return _ScalerBundle(x=x, u=u, a=a)


# =============================================================================
# 3.                         NETWORK ARCHITECTURE
# =============================================================================

__all__ = [
    "_ScaleVector",
    "_ScalerBundle",
    "_normalize_scale_mapping",
    "_load_scaler_bundle",
]
