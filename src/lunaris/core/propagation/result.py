"""State normalization helpers for propagation results."""

from __future__ import annotations

from typing import Any

import numpy as np

from lunaris.common.state_vector import normalize_cartesian_state

STATE_MIN_SIZE = 6  # [x,y,z,vx,vy,vz]; kept for backward compatibility


def _as_state_array(y0: Any) -> np.ndarray:
    """
    Normalize initial state to a contiguous float64 1D array.

    Accepts OrbitState-like objects (via ``.y``/``to_array()``) or array-likes.
    The dynamics RHS supports exactly 6 elements [x,y,z,vx,vy,vz] or 7
    (same + mass); anything else is rejected here instead of failing later
    inside DynamicsEngine.
    """
    return normalize_cartesian_state(y0, allow_mass=True, name="Initial state")


__all__ = ["STATE_MIN_SIZE", "_as_state_array"]
