from __future__ import annotations

import numpy as np

from lunaris.core.mc_propagator import _initial_impact_bookkeeping


def test_numba_host_preflight_marks_t0_impacts_at_and_below_boundary() -> None:
    radius = 10.0
    Y0 = np.zeros((3, 6), dtype=np.float64)
    Y0[:, 0] = [9.0, 10.0, 11.0]
    flags, times, positions = _initial_impact_bookkeeping(
        Y0,
        radius,
        detect_impact=True,
    )
    np.testing.assert_array_equal(flags, [1, 1, 0])
    np.testing.assert_allclose(times[:2], [0.0, 0.0])
    assert np.isnan(times[2])
    # t=0 impacts record their initial position; survivors stay NaN.
    np.testing.assert_allclose(positions[0], [9.0, 0.0, 0.0])
    np.testing.assert_allclose(positions[1], [10.0, 0.0, 0.0])
    assert np.isnan(positions[2]).all()


def test_numba_host_preflight_honors_disabled_detection() -> None:
    Y0 = np.zeros((2, 6), dtype=np.float64)
    flags, times, positions = _initial_impact_bookkeeping(
        Y0,
        10.0,
        detect_impact=False,
    )
    np.testing.assert_array_equal(flags, [0, 0])
    assert np.isnan(times).all()
    assert np.isnan(positions).all()
