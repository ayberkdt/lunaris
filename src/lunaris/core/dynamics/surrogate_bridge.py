"""Surrogate gravity provider detection for the dynamics engine."""

from __future__ import annotations

from typing import Any

def _is_surrogate_gravity_provider(obj: Any) -> bool:
    """
    Return ``True`` when the gravity object exposes the surrogate-runtime API.

    The classical SH path is fully Numba-compiled. ST-LRPS gravity must
    be evaluated through Python/PyTorch, so the dynamics engine needs to detect
    that provider class and route the RHS build accordingly.
    """

    return bool(
        obj is not None
        and getattr(obj, "model_kind", None) == "st_lrps"
        and hasattr(obj, "acceleration_fixed")
    )

__all__ = ["_is_surrogate_gravity_provider"]
