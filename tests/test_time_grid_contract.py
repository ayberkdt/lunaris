from __future__ import annotations

import numpy as np
import pytest

from lunaris.common.batch_defs import build_batch_output_grid
from lunaris.common.time_grid_contract import build_output_time_grid
from lunaris.core.propagation.time_grid import make_time_grid


def test_batch_alias_matches_neutral_output_time_grid() -> None:
    t_batch, n_batch, snap_batch = build_batch_output_grid(1000.0, 600.0)
    t_neutral, n_neutral, snap_neutral = build_output_time_grid(1000.0, 600.0)

    np.testing.assert_array_equal(t_batch, t_neutral)
    assert n_batch == n_neutral
    assert snap_batch == snap_neutral


def test_single_run_time_grid_uses_neutral_contract() -> None:
    t_contract, _, _ = build_output_time_grid(149.0, 100.0)

    np.testing.assert_array_equal(make_time_grid(0.0, 149.0, 100.0), t_contract)
    np.testing.assert_array_equal(make_time_grid(10.0, 159.0, 100.0), t_contract + 10.0)


def test_output_time_grid_golden_spacing_and_final_epoch() -> None:
    t, n_snaps, snap = build_output_time_grid(1000.0, 600.0)

    np.testing.assert_array_equal(t, np.array([0.0, 500.0, 1000.0], dtype=np.float64))
    assert n_snaps == 2
    assert snap == pytest.approx(500.0)
    assert t[0] == 0.0
    assert t[-1] == 1000.0
    assert np.all(np.diff(t) > 0.0)


@pytest.mark.parametrize(
    ("duration", "output_dt"),
    [(0.0, 60.0), (-1.0, 60.0), (60.0, 0.0), (60.0, -1.0)],
)
def test_output_time_grid_rejects_nonpositive_inputs(duration: float, output_dt: float) -> None:
    with pytest.raises(ValueError):
        build_output_time_grid(duration, output_dt)
