"""Batch planning must see a terrain grid injected through InMemorySurfaceProvider.

``_topography_requested`` looked only at ``provider.grids().topo``, which an
``InMemorySurfaceProvider`` leaves empty, while the engine's
``_resolve_topo_payload`` still hands that provider's ``topo_payload()`` DN grid
to the terrain impact kernels. A point-mass + terrain batch run was therefore
planned without the Moon-fixed rotation table the terrain sampler needs.
"""

from __future__ import annotations

from lunaris.batch.requirements import _topography_requested
from lunaris.loaders.io_surface import InMemorySurfaceProvider
from tests.test_terrain_freeze import _write_synthetic_ldem


def test_inmemory_topo_grid_counts_as_topography(tmp_path) -> None:
    topo = _write_synthetic_ldem(tmp_path)
    provider = InMemorySurfaceProvider(topo=topo)
    assert provider.grids().topo is None  # not file-backed
    assert provider.topo_payload().get("dn") is not None
    assert _topography_requested(provider, None) is True


def test_albedo_only_or_empty_provider_is_not_topography() -> None:
    assert _topography_requested(InMemorySurfaceProvider(), None) is False
    assert _topography_requested(None, None) is False

