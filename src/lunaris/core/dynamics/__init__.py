"""Package facade for the split dynamics implementation."""

from __future__ import annotations

from .adaptive_degree import _sample_albedo_dn_scaled, _select_adaptive_sh_degree
from .engine import DynamicsEngine
from .perturbation_packs import _AlbedoPack
from .requirements import (
    extract_ephem_tables_strict,
    extract_gravity_strict,
    extract_surface_provider_strict,
)
from .surrogate_bridge import _is_surrogate_gravity_provider

__all__ = [
    "DynamicsEngine",
    "extract_ephem_tables_strict",
    "extract_gravity_strict",
    "extract_surface_provider_strict",
    "_is_surrogate_gravity_provider",
    "_AlbedoPack",
    "_sample_albedo_dn_scaled",
    "_select_adaptive_sh_degree",
]
