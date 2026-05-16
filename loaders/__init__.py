"""
Loader-layer public package surface.

The `loaders` package owns disk-facing discovery/parsing helpers:
- gravity model readers
- surface/topography/albedo product readers
- generic path/asset discovery utilities
- SPICE kernel path preparation helpers

Higher layers (`models`, `analysis`, `ui_parts`) should prefer importing these
helpers instead of re-implementing repository/file discovery rules locally.
"""

from .io_helpers import (
    DataRootHints,
    autodetect_repository_data_roots,
    find_lunar_map_path,
    iter_lunar_map_candidates,
    prefer_dedicated_albedo_root,
    project_root_from_path,
)
from .spice_builder import maybe_autoinclude_lunar_fk, resolve_kernel_paths

__all__ = (
    "DataRootHints",
    "autodetect_repository_data_roots",
    "find_lunar_map_path",
    "iter_lunar_map_candidates",
    "prefer_dedicated_albedo_root",
    "project_root_from_path",
    "maybe_autoinclude_lunar_fk",
    "resolve_kernel_paths",
)
