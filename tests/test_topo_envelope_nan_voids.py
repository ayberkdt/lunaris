"""Terrain radius envelope of a float LDEM with NaN voids.

A plain ``min``/``max`` over a float raster with one NaN void returns NaN and
the envelope collapsed to the mean sphere, so the impact kernels never
terrain-checked anything above the mean radius.
"""

from __future__ import annotations

import numpy as np

from lunaris.loaders.io_surface import _topo_radius_envelope

R_MEAN = 1_737_400.0


def test_nan_void_is_excluded_from_the_envelope(tmp_path):
    dn = np.array([[-2.5, 0.0], [np.nan, 10.75]], dtype=np.float32)
    # radius_m = 1000 * DN + R_MEAN (km DN, metre output).
    r_min, r_max = _topo_radius_envelope(dn, 1000.0, R_MEAN, float("nan"), default_m=R_MEAN)
    assert r_min == R_MEAN - 2500.0
    assert r_max == R_MEAN + 10750.0

    path = tmp_path / "ldem.f32"
    dn.tofile(path)
    mm = np.memmap(path, dtype=np.float32, mode="r", shape=dn.shape)
    assert _topo_radius_envelope(mm, 1000.0, R_MEAN, float("nan"), default_m=R_MEAN) == (r_min, r_max)


def test_nan_void_and_missing_constant_are_both_excluded():
    dn = np.array([-9999.0, np.nan, 1.0, 3.0], dtype=np.float64)
    assert _topo_radius_envelope(dn, 1000.0, R_MEAN, -9999.0, default_m=R_MEAN) == (
        R_MEAN + 1000.0,
        R_MEAN + 3000.0,
    )


def test_all_nan_and_integer_rasters():
    all_nan = np.full((2, 2), np.nan)
    assert _topo_radius_envelope(all_nan, 1000.0, R_MEAN, float("nan"), default_m=R_MEAN) == (R_MEAN, R_MEAN)
    ints = np.array([[-3, 4], [0, 2]], dtype=np.int16)
    assert _topo_radius_envelope(ints, 500.0, R_MEAN, float("nan"), default_m=R_MEAN) == (
        R_MEAN - 1500.0,
        R_MEAN + 2000.0,
    )
