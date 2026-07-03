from __future__ import annotations

import pytest

from lunaris.batch.storage import _resolve_result_storage
from lunaris.common.batch_defs import BatchPropagationConfig


def _batch_config(
    *,
    output_format: str,
    storage_mode: str = "auto",
    memory_gb: float,
) -> BatchPropagationConfig:
    suffix = "h5" if output_format == "hdf5" else "npz"
    return BatchPropagationConfig(
        n_samples=8,
        use_gpu=False,
        output_format=output_format,
        output_path=f"outputs/tests/batch_storage_policy.{suffix}",
        result_storage_mode=storage_mode,
        max_result_memory_gb=memory_gb,
    )


def test_auto_hdf5_uses_disk_when_host_budget_is_low() -> None:
    cfg = _batch_config(output_format="hdf5", memory_gb=1.0)

    mode, result_bytes, budget = _resolve_result_storage(
        cfg,
        n_steps=10,
        available_host_memory_bytes=lambda: 1024,
    )

    assert mode == "disk"
    assert budget <= 1024
    assert result_bytes > budget


def test_auto_npz_above_budget_points_to_hdf5() -> None:
    cfg = _batch_config(output_format="npz", memory_gb=1.0)

    with pytest.raises(MemoryError, match="Use HDF5 output"):
        _resolve_result_storage(
            cfg,
            n_steps=10,
            available_host_memory_bytes=lambda: 1024,
        )


def test_missing_host_memory_probe_uses_configured_budget() -> None:
    cfg = _batch_config(output_format="hdf5", memory_gb=1.0)

    mode, _result_bytes, budget = _resolve_result_storage(
        cfg,
        n_steps=10,
        available_host_memory_bytes=lambda: None,
    )

    assert mode == "memory"
    assert budget == int(1.0 * (1024.0**3))
