"""Terrain sampling: the requested interpolation is the one used, and fast.

- ``make_hybrid_impact_event`` chose nearest-neighbour with ``"near" in kind``,
  which is true for the default ``"bilinear"``; the clearance then jumped at
  every pixel edge.
- ``sample_2d_bilinear`` converted the whole grid to float64 on every call; a
  memory-mapped float32 LDEM raster (64 ppd: 2.1 GB as float64) took ~1 s per
  sample. Float32 grids now pass through, bit-identical to converting first.
"""

from __future__ import annotations

import numpy as np
import pytest

from lunaris.common.math_utils import _validate_grid_data, sample_2d_bilinear, sample_2d_nearest
from lunaris.core.events import make_hybrid_impact_event

R_REF = 1_738_000.0


class _TwoSamplerTopo:
    """Bilinear and nearest return different radii, so the event shows which ran."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def sample_bilinear(self, lat_deg: float, lon_deg: float, kind: str = "radius_m") -> float:
        self.calls.append("bilinear")
        return R_REF + 1000.0

    def sample_nearest(self, lat_deg: float, lon_deg: float, kind: str = "radius_m") -> float:
        self.calls.append("nearest")
        return R_REF + 2000.0


@pytest.mark.parametrize(
    ("kind", "expected", "radius"),
    [("bilinear", "bilinear", R_REF + 1000.0), ("nearest", "nearest", R_REF + 2000.0)],
)
def test_hybrid_event_uses_the_requested_sampler(kind, expected, radius):
    topo = _TwoSamplerTopo()
    event = make_hybrid_impact_event(
        R_REF, 0.0, topo=topo, r_i_to_bf=lambda t, r: np.asarray(r, dtype=float),
        switch_alt_m=11_000.0, kind=kind,
    )
    r = R_REF + 5000.0  # inside the terrain switch shell
    value = event(0.0, np.array([r, 0.0, 0.0, 0.0, 1600.0, 0.0]))
    assert topo.calls and set(topo.calls) == {expected}
    assert value == pytest.approx(r - radius)


def test_float32_grid_is_sampled_without_a_copy_and_bit_identical():
    rng = np.random.default_rng(7)
    grid32 = rng.uniform(-9.0, 10.0, size=(181, 360)).astype(np.float32)
    assert np.shares_memory(_validate_grid_data(grid32, 181, 360), grid32)
    grid64 = grid32.astype(np.float64)
    for _ in range(200):
        row, col = rng.uniform(-2.0, 183.0), rng.uniform(-400.0, 800.0)
        assert sample_2d_bilinear(grid32, row, col, 181, 360) == sample_2d_bilinear(grid64, row, col, 181, 360)
        assert sample_2d_nearest(grid32, row, col, 181, 360) == sample_2d_nearest(grid64, row, col, 181, 360)


def test_integer_grid_is_still_converted():
    grid = np.arange(12, dtype=np.int16).reshape(3, 4)
    out = _validate_grid_data(grid, 3, 4)
    assert out.dtype == np.float64
    assert sample_2d_bilinear(grid, 1.5, 0.5, 3, 4) == pytest.approx(6.5)
