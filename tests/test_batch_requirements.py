from __future__ import annotations

from types import SimpleNamespace

from lunaris.batch.requirements import (
    _surface_force_provider_requested,
    _surface_provider_present,
    _surface_topography_requested,
    _topography_requested,
)
from lunaris.common.type_defs import PerturbationFlags
from lunaris.physics.surface_effects import AlbedoConfig


def _cfg(*, flags: PerturbationFlags, albedo: AlbedoConfig | None = None):
    return SimpleNamespace(
        flags=flags,
        albedo=albedo,
        thermal=None,
        solid_tides=None,
    )


def test_topography_requested_distinguishes_albedo_only_provider():
    albedo_only_provider = SimpleNamespace(
        grids=lambda: SimpleNamespace(albedo=object(), topo=None)
    )
    terrain_provider = SimpleNamespace(
        grids=lambda: SimpleNamespace(albedo=object(), topo=object())
    )

    assert _surface_provider_present(albedo_only_provider) is True
    assert _topography_requested(albedo_only_provider, None) is False
    assert _surface_topography_requested(albedo_only_provider, None) is False
    assert _topography_requested(terrain_provider, None) is True
    assert _topography_requested(None, object()) is True


def test_surface_force_provider_requested_tracks_grid_surface_forces():
    constant_albedo_cfg = _cfg(
        flags=PerturbationFlags(enable_albedo=True),
        albedo=AlbedoConfig(albedo_mode="constant_albedo"),
    )
    grid_albedo_cfg = _cfg(
        flags=PerturbationFlags(enable_albedo=True),
        albedo=AlbedoConfig(albedo_mode="albedo_grid"),
    )

    assert _surface_force_provider_requested(constant_albedo_cfg) is False
    assert _surface_force_provider_requested(grid_albedo_cfg) is True
