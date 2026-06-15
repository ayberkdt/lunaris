from __future__ import annotations

import numpy as np
import pytest

from lunaris.common.montecarlo_defs import MonteCarloConfig
from lunaris.core import monte_carlo_engine
from lunaris.core.monte_carlo_engine import (
    _allocate_result_buffer,
    _resolve_result_storage,
)


def _mc_config(
    *,
    output_format: str,
    storage_mode: str = "auto",
    memory_gb: float,
) -> MonteCarloConfig:
    suffix = "h5" if output_format == "hdf5" else "npz"
    return MonteCarloConfig(
        n_samples=8,
        use_gpu=False,
        output_format=output_format,
        output_path=f"outputs/tests/storage_policy.{suffix}",
        result_storage_mode=storage_mode,
        max_result_memory_gb=memory_gb,
    )


def test_auto_hdf5_uses_disk_only_above_memory_limit() -> None:
    small = _mc_config(output_format="hdf5", memory_gb=1.0)
    mode, result_bytes, limit_bytes = _resolve_result_storage(small, n_steps=10)
    assert mode == "memory"
    assert result_bytes < limit_bytes

    large = _mc_config(output_format="hdf5", memory_gb=1.0e-9)
    mode, result_bytes, limit_bytes = _resolve_result_storage(large, n_steps=10)
    assert mode == "disk"
    assert result_bytes > limit_bytes


def test_explicit_memory_preserves_eager_behavior_above_limit() -> None:
    cfg = _mc_config(
        output_format="hdf5",
        storage_mode="memory",
        memory_gb=1.0e-9,
    )
    mode, result_bytes, limit_bytes = _resolve_result_storage(cfg, n_steps=10)
    assert mode == "memory"
    assert result_bytes > limit_bytes


def test_large_auto_npz_is_rejected_with_hdf5_guidance() -> None:
    cfg = _mc_config(output_format="npz", memory_gb=1.0e-9)
    with pytest.raises(MemoryError, match="Use HDF5 output"):
        _resolve_result_storage(cfg, n_steps=10)


def test_low_available_ram_forces_disk_under_config_cap(monkeypatch) -> None:
    # Config cap is generous (1 GiB) but only ~1 KiB of RAM is free: the host
    # memory safety factor must push an auto/hdf5 run to disk even though the
    # result fits the configured cap.
    monkeypatch.setattr(
        monte_carlo_engine, "_available_host_memory_bytes", lambda: 1024
    )
    cfg = _mc_config(output_format="hdf5", memory_gb=1.0)
    mode, result_bytes, budget = _resolve_result_storage(cfg, n_steps=10)
    assert mode == "disk"
    assert budget <= 1024
    assert result_bytes > budget


def test_explicit_memory_warns_when_exceeding_safety_budget(monkeypatch) -> None:
    monkeypatch.setattr(
        monte_carlo_engine, "_available_host_memory_bytes", lambda: 1024
    )
    cfg = _mc_config(output_format="hdf5", storage_mode="memory", memory_gb=1.0)
    with pytest.warns(RuntimeWarning, match="host memory safety budget"):
        mode, _, _ = _resolve_result_storage(cfg, n_steps=10)
    assert mode == "memory"


def test_missing_psutil_disables_safety_factor(monkeypatch) -> None:
    # Without a RAM reading the budget falls back to the configured cap exactly.
    monkeypatch.setattr(
        monte_carlo_engine, "_available_host_memory_bytes", lambda: None
    )
    cfg = _mc_config(output_format="hdf5", memory_gb=1.0)
    mode, _result_bytes, budget = _resolve_result_storage(cfg, n_steps=10)
    assert mode == "memory"
    assert budget == int(1.0 * (1024.0 ** 3))


def test_disk_mode_never_allocates_full_trajectory(monkeypatch) -> None:
    def fail_allocation(*args, **kwargs):
        raise AssertionError("disk mode attempted a full trajectory allocation")

    monkeypatch.setattr(monte_carlo_engine.np, "empty", fail_allocation)
    assert _allocate_result_buffer("disk", None, (10, 8, 6)) is None


def test_memory_mode_reuses_writer_buffer() -> None:
    writer_buffer = np.empty((10, 8, 6), dtype=np.float64)
    result = _allocate_result_buffer("memory", writer_buffer, (10, 8, 6))
    assert result is writer_buffer
