"""State normalization helpers for propagation results."""

from __future__ import annotations

from typing import Any

import numpy as np

from lunaris.core.state import OrbitState

STATE_MIN_SIZE = 6  # [x,y,z,vx,vy,vz]


def _as_state_array(y0: Any) -> np.ndarray:
    """
    Normalize initial state to a contiguous float64 1D array.
    Accepts OrbitState or array-like. Requires at least 6 elements.
    """
    if isinstance(y0, OrbitState):
        y = np.asarray(y0.y, dtype=np.float64).reshape(-1)
    else:
        y = np.asarray(y0, dtype=np.float64).reshape(-1)

    if y.size < STATE_MIN_SIZE:
        raise ValueError("Initial state must have at least 6 elements: [x,y,z,vx,vy,vz].")

    return np.array(y, dtype=np.float64, copy=True)


__all__ = ["STATE_MIN_SIZE", "_as_state_array"]
