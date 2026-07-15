"""Edge-semantics tests for the shared altitude-binning helper in evaluation/cli.

The helper `_iter_altitude_bins` replaced five copy-pasted edge+mask scaffolds;
these tests pin down the contract those call sites rely on: outward edge
snapping to bin multiples, half-open [lo, hi) bins, NaN exclusion, and the
yields-nothing behaviour for degenerate inputs.
"""

from __future__ import annotations

import numpy as np

from lunaris.surrogate.st_lrps.evaluation.cli import (
    _iter_altitude_bins,
    spatial_mape_by_altitude,
    spatial_rmse_by_altitude,
)


def test_bins_are_half_open_and_snapped_outward():
    alt = np.array([12.0, 19.999, 20.0, 27.5], dtype=np.float64)
    bins = list(_iter_altitude_bins(alt, 10.0))
    # Edges snap to [10, 30]; values 12/19.999 fall in [10,20), 20/27.5 in [20,30).
    assert [(a0, a1, n) for a0, a1, _, n in bins] == [(10.0, 20.0, 2), (20.0, 30.0, 2)]
    # Boundary value 20.0 belongs to the upper bin (half-open lower bins).
    lower_mask = bins[0][2]
    assert not lower_mask[2]


def test_empty_bins_are_skipped():
    alt = np.array([5.0, 95.0], dtype=np.float64)
    bins = list(_iter_altitude_bins(alt, 10.0))
    assert [(a0, a1) for a0, a1, _, _ in bins] == [(0.0, 10.0), (90.0, 100.0)]


def test_nan_altitudes_are_excluded_from_masks():
    alt = np.array([15.0, np.nan, 16.0], dtype=np.float64)
    bins = list(_iter_altitude_bins(alt, 10.0))
    assert len(bins) == 1
    _, _, mask, n = bins[0]
    assert n == 2
    assert not mask[1]


def test_degenerate_inputs_yield_nothing():
    assert list(_iter_altitude_bins(np.array([], dtype=np.float64), 10.0)) == []
    all_nan = np.array([np.nan, np.nan], dtype=np.float64)
    assert list(_iter_altitude_bins(all_nan, 10.0)) == []


def test_rmse_and_mape_empty_contract():
    empty = np.array([], dtype=np.float64)
    out = spatial_rmse_by_altitude(empty, empty, 25.0)
    assert out == {"bin_km": 25.0, "bins": []}
    out = spatial_mape_by_altitude(empty, empty, empty, 25.0)
    assert out == {"bin_km": 25.0, "bins": []}


def test_rmse_values_match_direct_computation():
    rng = np.random.default_rng(7)
    alt = rng.uniform(0.0, 100.0, 500)
    err = rng.normal(0.0, 1.0, 500)
    out = spatial_rmse_by_altitude(alt, err, 50.0)
    assert [b["n"] for b in out["bins"]] and sum(b["n"] for b in out["bins"]) == 500
    for b in out["bins"]:
        mask = (alt >= b["alt_km_lo"]) & (alt < b["alt_km_hi"])
        assert b["rmse"] == float(np.sqrt(np.mean(err[mask] ** 2)))
