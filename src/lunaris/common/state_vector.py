"""Shared Cartesian state-vector normalization contract.

Single source of truth for turning "state-like" inputs (raw array-likes,
``InitialState``-style objects exposing ``to_array()``, ``OrbitState``-style
objects exposing a packed ``.y`` vector, or plain records with
``x, y, z, vx, vy, vz`` attributes) into the exact state layout the dynamics
RHS supports:

- 6 elements: ``[x, y, z, vx, vy, vz]``
- 7 elements: same + spacecraft mass at index 6

Anything else fails loudly here instead of surfacing later inside the
propagator or being silently truncated by a batch path.

This module is dependency-light by design (numpy only): it must stay importable
from core, batch, and CLI layers without pulling physics modules. ``OrbitState``
is therefore accepted via its ``.y`` attribute (duck typing), never imported.
"""

from __future__ import annotations

from typing import Any

import numpy as np

STATE_SIZE_POS_VEL = 6  # [x,y,z,vx,vy,vz]
STATE_SIZE_WITH_MASS = 7  # [x,y,z,vx,vy,vz,mass]

_COMPONENT_ATTRS = ("x", "y", "z", "vx", "vy", "vz")


def _extract_state_array(state_like: Any, name: str) -> np.ndarray:
    """Pull a flat float64 vector out of any supported state container."""
    if state_like is None:
        raise ValueError(f"{name} is None.")

    # common.type_defs.InitialState (and friends): explicit packed export.
    if hasattr(state_like, "to_array"):
        return np.asarray(state_like.to_array(), dtype=np.float64).reshape(-1)

    # core.state.OrbitState (or similar): packed vector via `.y`. A plain
    # component record also has a `.y` attribute (the position component),
    # but that one is a scalar — fall through to the attribute path then.
    if hasattr(state_like, "y"):
        arr = np.asarray(state_like.y, dtype=np.float64).reshape(-1)
        if arr.size > 1:
            return arr

    # Plain object with x,y,z,vx,vy,vz component attributes.
    if all(hasattr(state_like, k) for k in _COMPONENT_ATTRS):
        return np.asarray(
            [float(getattr(state_like, k)) for k in _COMPONENT_ATTRS],
            dtype=np.float64,
        )

    # Raw array-like.
    return np.asarray(state_like, dtype=np.float64).reshape(-1)


def normalize_cartesian_state(
    state_like: Any,
    *,
    allow_mass: bool = True,
    name: str = "state",
) -> np.ndarray:
    """Normalize a state-like input to a contiguous float64 1D copy.

    With ``allow_mass=True`` the result must have exactly 6 or 7 elements
    (the 7th being spacecraft mass); with ``allow_mass=False`` exactly 6.
    Oversized states are rejected, never truncated.
    """
    arr = _extract_state_array(state_like, name)

    if allow_mass:
        if arr.size not in (STATE_SIZE_POS_VEL, STATE_SIZE_WITH_MASS):
            raise ValueError(
                f"{name} must have exactly {STATE_SIZE_POS_VEL} elements "
                f"[x,y,z,vx,vy,vz] or {STATE_SIZE_WITH_MASS} elements "
                f"[x,y,z,vx,vy,vz,mass], got {arr.size}."
            )
    elif arr.size != STATE_SIZE_POS_VEL:
        raise ValueError(
            f"{name} must have exactly {STATE_SIZE_POS_VEL} elements "
            f"[x,y,z,vx,vy,vz], got {arr.size}."
        )

    return np.array(arr, dtype=np.float64, copy=True)


def normalize_position_velocity_state(
    state_like: Any,
    *,
    drop_mass: bool = False,
    name: str = "state",
) -> np.ndarray:
    """Normalize to the exact 6-element ``[x,y,z,vx,vy,vz]`` layout.

    ``drop_mass=True`` is the *only* sanctioned way to accept a 7-element
    state here: the mass entry is discarded because the caller works in a
    strictly 6D Cartesian space (e.g. batch sampling). Anything larger than
    7 (or exactly 7 with ``drop_mass=False``) fails loudly — this helper
    never truncates silently.
    """
    arr = normalize_cartesian_state(state_like, allow_mass=drop_mass, name=name)
    if arr.size == STATE_SIZE_WITH_MASS:
        return np.ascontiguousarray(arr[:STATE_SIZE_POS_VEL])
    return arr


__all__ = [
    "STATE_SIZE_POS_VEL",
    "STATE_SIZE_WITH_MASS",
    "normalize_cartesian_state",
    "normalize_position_velocity_state",
]
