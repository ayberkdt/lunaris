from __future__ import annotations

import numpy as np
import pytest

from lunaris.analysis.monte_carlo.statistics import (
    compute_ensemble_statistics,
    compute_impact_statistics,
)
from lunaris.common.montecarlo_defs import MCRunResult


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
