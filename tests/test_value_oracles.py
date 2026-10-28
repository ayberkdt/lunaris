"""Value tests against independent oracles (closed forms, hand computations).

A test here passes only if the code returns the right number, not just a number:
each expected value is computed independently of the code under test, and each
tolerance names the error source that sets it.
"""
from __future__ import annotations

import numpy as np
import pytest


def test_altitude_bins_count_every_sample_including_the_shell_edge() -> None:
    """The last bin was half-open, so a sample exactly at the 1000 km edge was dropped."""
    from lunaris.surrogate.st_lrps.evaluation.cli import spatial_rmse_by_altitude

    alt = np.array([100.0, 120.0, 950.0, 999.0, 1000.0])
    err = np.array([1.0, 3.0, 2.0, 2.0, 5.0])
    bins = spatial_rmse_by_altitude(alt, err, 50.0)["bins"]
    assert sum(b["n"] for b in bins) == alt.size
    by_lo = {b["alt_km_lo"]: b for b in bins}
    assert by_lo[100.0]["rmse"] == pytest.approx(np.sqrt((1 + 9) / 2), rel=1e-15)
    assert by_lo[950.0]["rmse"] == pytest.approx(np.sqrt((4 + 4 + 25) / 3), rel=1e-15)


def _pair_at_angle(theta: np.ndarray, norm: float = 1.62) -> tuple[np.ndarray, np.ndarray]:
    a = np.tile([norm, 0.0, 0.0], (theta.size, 1))
    b = norm * np.c_[np.cos(theta), np.sin(theta), np.zeros(theta.size)]
    return a, b


def test_angle_keeps_precision_for_small_angles() -> None:
    """arccos(cos) lost the angle below ~1e-7 rad and returned 0 at 1e-8 rad.
    Oracle: the construction angle; atan2 is exact to float64 roundoff (~1e-16/theta)."""
    from lunaris.surrogate.st_lrps.evaluation.cli import vector_angle_rad

    theta = np.array([1e-5, 1e-7, 1e-8, 1e-9, 0.3, 3.0])
    a, b = _pair_at_angle(theta)
    np.testing.assert_allclose(vector_angle_rad(a, b), theta, rtol=1e-6)


def test_a_zero_prediction_has_no_direction() -> None:
    """A zero vector used to give cos = 1 and 0 deg: a dead network looked perfect."""
    from lunaris.surrogate.st_lrps.evaluation.cli import vector_cosine_and_angle_deg

    true = np.array([[1e-6, 2e-6, -1e-6], [3e-7, 0.0, 0.0]])
    cos, ang = vector_cosine_and_angle_deg(np.zeros_like(true), true)
    np.testing.assert_array_equal(ang, [90.0, 90.0])
    np.testing.assert_array_equal(cos, [0.0, 0.0])
