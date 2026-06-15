"""MCRunResult sample-validity contract (reviewer §7).

A Monte Carlo sample that fails to propagate must NOT enter ensemble statistics
as a spurious all-zero state. The CPU backend NaN-fills failed trajectories and
the engine surfaces a ``valid_mask`` so consumers can exclude them. These tests
lock that contract on the result container itself.
"""

from __future__ import annotations

import numpy as np
import pytest

from lunaris.common.montecarlo_defs import MCRunResult


def _result(valid_mask=None, n: int = 3, t: int = 4) -> MCRunResult:
    Y = np.ones((t, n, 6), dtype=np.float64)
    if valid_mask is not None:
        failed = np.asarray(valid_mask, dtype=np.float64) < 0.5
        Y[:, failed, :] = np.nan  # mirror the CPU backend's NaN-fill of failures
    return MCRunResult(
        t=np.linspace(0.0, 1.0, t),
        Y=Y,
        sc_samples=np.ones((n, 4)),
        impact_mask=np.zeros(n),
        t_impact=np.full(n, np.nan),
        valid_mask=valid_mask,
    )


def test_valid_mask_defaults_to_all_valid() -> None:
    r = _result(valid_mask=None, n=5)
    assert r.valid_mask is None
    assert r.n_valid == 5


def test_valid_mask_counts_only_valid_samples() -> None:
    r = _result(valid_mask=np.array([1.0, 0.0, 1.0]))
    assert r.n_valid == 2
    # The failed sample's trajectory is NaN, not a spurious zero state.
    assert np.isnan(r.Y[:, 1, :]).all()
    assert np.isfinite(r.Y[:, 0, :]).all()
    assert np.isfinite(r.Y[:, 2, :]).all()


def test_valid_mask_wrong_shape_raises() -> None:
    with pytest.raises(ValueError):
        MCRunResult(
            t=np.linspace(0.0, 1.0, 4),
            Y=np.ones((4, 3, 6)),
            sc_samples=np.ones((3, 4)),
            impact_mask=np.zeros(3),
            t_impact=np.full(3, np.nan),
            valid_mask=np.array([1.0, 0.0]),  # wrong length (2 != 3)
        )
