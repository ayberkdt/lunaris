"""P0 regression: ST-LRPS training determinism / reproducibility controls.

Covers the pre-training reproducibility hardening: ``set_seed`` must apply (and
report) full determinism — cuDNN deterministic, TF32 pinned off,
``use_deterministic_algorithms``, and the cuBLAS / PYTHONHASHSEED env vars — and
the DataLoader ``worker_init_fn`` must seed numpy/Python per worker so dataset
draws are reproducible. RNG-state checkpoint/resume is exercised elsewhere.
"""

from __future__ import annotations

import os
import random

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from lunaris.surrogate.st_lrps.training.engine import (  # noqa: E402
    _seed_dataloader_worker,
    set_seed,
)


@pytest.fixture
def _restore_determinism():
    """Snapshot and restore process-global determinism state around a test."""
    prev_cudnn_det = torch.backends.cudnn.deterministic
    prev_cudnn_bench = torch.backends.cudnn.benchmark
    prev_env = {k: os.environ.get(k) for k in ("PYTHONHASHSEED", "CUBLAS_WORKSPACE_CONFIG")}
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = prev_cudnn_det
        torch.backends.cudnn.benchmark = prev_cudnn_bench
        for key, value in prev_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_set_seed_deterministic_applies_and_reports_flags(_restore_determinism) -> None:
    flags = set_seed(123, deterministic=True, benchmark=False)
    assert flags["seed"] == 123
    assert flags["cudnn_deterministic"] is True
    assert flags["cudnn_benchmark"] is False
    assert flags["use_deterministic_algorithms"] == "warn_only"
    assert flags["allow_tf32"] is False
    assert flags["pythonhashseed"] == "123"
    # Env vars the deterministic toggle / spawn workers require.
    assert os.environ["PYTHONHASHSEED"] == "123"
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    # Actually applied to the live process.
    assert torch.backends.cudnn.deterministic is True
    assert torch.are_deterministic_algorithms_enabled() is True


def test_set_seed_nondeterministic_disables_controls(_restore_determinism) -> None:
    flags = set_seed(5, deterministic=False, benchmark=True)
    assert flags["cudnn_deterministic"] is False
    assert flags["cudnn_benchmark"] is True
    assert flags["use_deterministic_algorithms"] is False
    assert flags["allow_tf32"] is True
    assert torch.are_deterministic_algorithms_enabled() is False


def test_set_seed_makes_all_rngs_reproducible(_restore_determinism) -> None:
    set_seed(2024)
    t1 = torch.rand(4)
    n1 = np.random.rand(4)
    r1 = [random.random() for _ in range(4)]

    set_seed(2024)
    assert torch.equal(torch.rand(4), t1)
    np.testing.assert_array_equal(np.random.rand(4), n1)
    assert [random.random() for _ in range(4)] == r1


def test_worker_init_fn_seeds_numpy_and_python_deterministically() -> None:
    torch.manual_seed(777)
    _seed_dataloader_worker(2)
    a_np = np.random.rand(3).tolist()
    a_py = [random.random() for _ in range(3)]

    torch.manual_seed(777)
    _seed_dataloader_worker(2)
    assert np.random.rand(3).tolist() == a_np
    assert [random.random() for _ in range(3)] == a_py


def test_worker_init_fn_distinct_workers_get_distinct_streams() -> None:
    torch.manual_seed(777)
    _seed_dataloader_worker(2)
    worker2 = np.random.rand(3).tolist()

    torch.manual_seed(777)
    _seed_dataloader_worker(3)
    worker3 = np.random.rand(3).tolist()

    assert worker2 != worker3
