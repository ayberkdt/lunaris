"""Package facade for the split dynamics implementation."""

from __future__ import annotations

from .engine import DynamicsEngine
from .requirements import (
    extract_ephem_state_tables_strict,
    extract_ephem_tables_strict,
    extract_gravity_strict,
    extract_surface_provider_strict,
)

# Only the public dynamics surface is re-exported from the package facade.
# Internal helpers (``_AlbedoPack``, ``_sample_albedo_dn_scaled``,
# ``_select_adaptive_sh_degree``, ``_is_surrogate_gravity_provider``) live in and
# are imported from their canonical submodules (``perturbation_packs``,
# ``adaptive_degree``, ``surrogate_bridge``); they are intentionally NOT promoted
# to package-level API.
__all__ = [
    "DynamicsEngine",
    "extract_ephem_state_tables_strict",
    "extract_ephem_tables_strict",
    "extract_gravity_strict",
    "extract_surface_provider_strict",
]
