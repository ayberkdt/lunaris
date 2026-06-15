from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

from lunaris.analysis.monte_carlo.statistics import (
    compute_ensemble_statistics,
    compute_impact_statistics,
    compute_oe_dispersion,
)
from lunaris.common.montecarlo_defs import MCRunResult
from lunaris.core.monte_carlo_engine import HDF5TrajectoryView


def _result() -> MCRunResult:
    Y = np.zeros((2, 3, 6), dtype=np.float64)
    Y[:, 0, 0] = 1.8e6
    Y[:, 1, 0] = 1.9e6
    Y[:, 2, :] = 1.0e99
    fixed = np.full((3, 3), np.nan)
    fixed[1] = [0.0, 1.0, 0.0]
    return MCRunResult(
        t=np.array([0.0, 1.0]),
        Y=Y,
        sc_samples=np.ones((3, 4)),
        impact_mask=np.array([0.0, 1.0, 1.0]),
        t_impact=np.array([np.nan, 1.0, 1.0]),
        valid_mask=np.array([1.0, 1.0, 0.0]),
        impact_position_fixed_m=fixed,
    )


def test_statistics_exclude_invalid_samples_before_impact_filtering() -> None:
    result = _result()
    ens = compute_ensemble_statistics(result)
    assert ens.mean[0, 0] == pytest.approx(1.85e6)
    impacts = compute_impact_statistics(result)
    assert impacts.n_total == 2
    assert impacts.n_impacts == 1
    assert impacts.p_impact == pytest.approx(0.5)
    np.testing.assert_allclose(impacts.lat_deg, [0.0])
    np.testing.assert_allclose(impacts.lon_deg, [90.0])


def test_statistics_fail_with_fewer_than_two_valid_samples() -> None:
    result = _result()
    result.valid_mask[:] = [1.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="insufficient valid samples"):
        compute_ensemble_statistics(result)


def test_impact_statistics_fail_with_no_valid_samples() -> None:
    result = _result()
    result.valid_mask[:] = 0.0
    with pytest.raises(ValueError, match="no valid samples"):
        compute_impact_statistics(result)


class _TrackingLazyTrajectory:
    _lunaris_lazy_trajectory = True

    def __init__(self, values: np.ndarray) -> None:
        self._values = values
        self.shape = values.shape
        self.ndim = values.ndim
        self.dtype = values.dtype
        self.iterator_calls = 0
        self.block_count = 0

    def __getitem__(self, key):
        raise AssertionError("lazy statistics must use the epoch-block iterator")

    def __array__(self, *args, **kwargs):
        raise AssertionError("lazy statistics must not materialize the full trajectory")

    def iter_epoch_sample_blocks(self, sample_indices):
        self.iterator_calls += 1
        indices = np.asarray(sample_indices, dtype=np.int64)
        assert np.all(np.diff(indices) > 0)
        for epoch in range(self.shape[0]):
            self.block_count += 1
            yield self._values[epoch, indices, :]


def test_c2_lazy_statistics_use_one_epoch_block_iterator() -> None:
    t = np.array([0.0, 10.0, 20.0])
    r0 = 1.838e6
    v0 = (4.9e12 / r0) ** 0.5
    base = np.array([r0, 0.0, 0.0, 0.0, v0, 0.0])
    Y = np.broadcast_to(base, (t.size, 4, 6)).astype(np.float64).copy()
    Y[:, :, 0] += np.arange(4, dtype=np.float64)[None, :] * 100.0
    lazy = _TrackingLazyTrajectory(Y)
    result = MCRunResult(
        t=t,
        Y=lazy,
        sc_samples=np.ones((4, 4)),
        impact_mask=np.zeros(4),
        t_impact=np.full(4, np.nan),
    )
    eager_result = MCRunResult(
        t=t,
        Y=Y,
        sc_samples=np.ones((4, 4)),
        impact_mask=np.zeros(4),
        t_impact=np.full(4, np.nan),
    )

    expected_ensemble = compute_ensemble_statistics(eager_result)
    actual_ensemble = compute_ensemble_statistics(result)
    expected_oe = compute_oe_dispersion(eager_result)
    actual_oe = compute_oe_dispersion(result)

    assert lazy.iterator_calls == 2
    assert lazy.block_count == 2 * t.size
    np.testing.assert_allclose(actual_ensemble.mean, expected_ensemble.mean)
    np.testing.assert_allclose(actual_ensemble.cov, expected_ensemble.cov)
    np.testing.assert_allclose(
        actual_oe.a_mean_km,
        expected_oe.a_mean_km,
        equal_nan=True,
    )
    np.testing.assert_allclose(actual_oe.e_mean, expected_oe.e_mean, equal_nan=True)
    np.testing.assert_allclose(
        actual_oe.inc_mean_deg,
        expected_oe.inc_mean_deg,
        equal_nan=True,
    )


def test_c2_hdf5_epoch_iterator_opens_archive_once() -> None:
    values = np.arange(3 * 4 * 6, dtype=np.float64).reshape(3, 4, 6)
    open_count = 0

    class _FakeFile:
        def __init__(self, *args, **kwargs) -> None:
            nonlocal open_count
            open_count += 1

        def __enter__(self):
            return self

        def __exit__(self, *args) -> None:
            return None

        def __getitem__(self, key):
            assert key == "Y"
            return values

    fake_h5py = types.ModuleType("h5py")
    fake_h5py.File = _FakeFile
    previous = sys.modules.get("h5py")
    sys.modules["h5py"] = fake_h5py
    try:
        view = object.__new__(HDF5TrajectoryView)
        view.path = Path("unused.h5")
        view.dataset = "Y"
        view.shape = values.shape
        view.ndim = values.ndim
        view.dtype = values.dtype
        indices = np.array([0, 2], dtype=np.int64)
        blocks = list(view.iter_epoch_sample_blocks(indices))
    finally:
        if previous is None:
            sys.modules.pop("h5py", None)
        else:
            sys.modules["h5py"] = previous

    assert open_count == 1
    assert len(blocks) == values.shape[0]
    for epoch, block in enumerate(blocks):
        np.testing.assert_array_equal(block, values[epoch, indices, :])
