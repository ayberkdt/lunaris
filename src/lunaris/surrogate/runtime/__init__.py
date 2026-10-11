"""Production ST-LRPS surrogate runtime adapter package.

Public names resolve lazily (PEP 562): ``import lunaris.surrogate.runtime``
does not import torch. The run-discovery helpers come from the light
``.artifact`` module; ``SurrogateGravityModel`` / ``SurrogateGravityMetadata``
import ``.adapter`` (and through it torch) on first access.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .adapter import (
        DEFAULT_ST_LRPS_RUNS_DIR,
        SurrogateGravityMetadata,
        SurrogateGravityModel,
        discover_st_lrps_model_dirs,
        find_checkpoint_for_st_lrps_run,
        find_latest_st_lrps_model_dir,
    )

__all__ = [
    "DEFAULT_ST_LRPS_RUNS_DIR",
    "SurrogateGravityMetadata",
    "SurrogateGravityModel",
    "discover_st_lrps_model_dirs",
    "find_checkpoint_for_st_lrps_run",
    "find_latest_st_lrps_model_dir",
]

_LIGHT_NAMES = frozenset(
    {
        "DEFAULT_ST_LRPS_RUNS_DIR",
        "discover_st_lrps_model_dirs",
        "find_checkpoint_for_st_lrps_run",
        "find_latest_st_lrps_model_dir",
    }
)


def __getattr__(name: str) -> Any:
    if name in _LIGHT_NAMES:
        from lunaris.surrogate.runtime import artifact

        return getattr(artifact, name)
    if name in ("SurrogateGravityMetadata", "SurrogateGravityModel"):
        from lunaris.surrogate.runtime import adapter

        return getattr(adapter, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
