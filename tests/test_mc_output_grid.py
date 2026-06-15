"""Shared Monte Carlo output time-grid contract (reviewer §5).

``build_mc_output_grid`` is the single source of truth for the snapshot grid
used by every MC backend (CPU DOP853, Numba CUDA, torch_cuda_sh / torch_cpu_sh,
ST-LRPS torch), so trajectories are index-comparable across backends. These
tests lock the contract independently of any propagator / torch / CUDA.
"""

from __future__ import annotations

import numpy as np
import pytest

from lunaris.common.montecarlo_defs import build_mc_output_grid


def test_grid_contract_divisible() -> None:
    t, n_snaps, snap = build_mc_output_grid(1200.0, 300.0)
    assert t[0] == 0.0
    assert t[-1] == 1200.0
    assert n_snaps == 4
    assert len(t) == n_snaps + 1
    assert np.all(np.diff(t) > 0.0)
    assert snap == pytest.approx(300.0)


def test_grid_exact_final_when_not_divisible() -> None:
    # 1000 / 600 -> round() = 2 snaps. The last sample must land exactly on the
    # requested duration (1000 s), never overshoot to 1200 s as the old
    # ``n_snaps * out_dt`` endpoint did.
    t, n_snaps, snap = build_mc_output_grid(1000.0, 600.0)
    assert n_snaps == 2
    assert t[0] == 0.0
    assert t[-1] == 1000.0
    assert np.all(np.diff(t) > 0.0)
    assert snap == pytest.approx(500.0)


def test_grid_minimum_one_snapshot() -> None:
    # output_dt larger than duration still yields a valid 2-point grid.
    t, n_snaps, _ = build_mc_output_grid(100.0, 10_000.0)
    assert n_snaps == 1
    assert list(t) == [0.0, 100.0]


def test_grid_realized_spacing_never_exceeds_request() -> None:
    t, n_snaps, snap = build_mc_output_grid(149.0, 100.0)
    assert n_snaps == 2
    assert snap == pytest.approx(74.5)
    assert np.max(np.diff(t)) <= 100.0


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_grid_rejects_nonpositive_duration(bad: float) -> None:
    with pytest.raises(ValueError):
        build_mc_output_grid(bad, 60.0)


@pytest.mark.parametrize("bad", [0.0, -5.0])
def test_grid_rejects_nonpositive_output_dt(bad: float) -> None:
    with pytest.raises(ValueError):
        build_mc_output_grid(1000.0, bad)
