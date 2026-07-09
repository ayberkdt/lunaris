"""Task 1 — scaler leakage guard.

The residual ΔU/Δa target scalers must be fit on TRAIN ROWS ONLY. These tests
build a dataset whose validation rows carry extreme target values and prove the
train-only fit is unaffected, while the legacy whole-file fit is not — so the
test genuinely catches leakage rather than passing vacuously.
"""

from __future__ import annotations

import pytest

try:
    import torch
    import torch.nn
    _ = torch.cuda
    _ = getattr(torch, "compile", None)
except (ImportError, AttributeError, ModuleNotFoundError):
    pytest.skip("PyTorch not installed or missing required attributes like cuda/compile", allow_module_level=True)



import numpy as np
import pytest

torch = pytest.importorskip("torch")
_ = pytest.importorskip("torch.nn")

torch = pytest.importorskip('torch')

from dataset_pipeline_test_utils import write_toy_contract_h5
from lunaris.surrogate.st_lrps.data.dataset_parameters import MU_MOON_SI, R_MOON_SI
from lunaris.surrogate.st_lrps.data.datasets import DatasetMeta
from lunaris.surrogate.st_lrps.shared.scaling import fit_scaler_streaming


def _make_rows(n: int, *, u_value: float, a_value: float, seed: int) -> np.ndarray:
    """Residual rows [x,y,z,U,ax,ay,az] with constant-magnitude targets."""
    rng = np.random.default_rng(seed)
    dirs = rng.normal(size=(n, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    alt_km = rng.uniform(100.0, 500.0, size=n)
    xyz = dirs * (float(R_MOON_SI) + alt_km[:, None] * 1000.0)
    u = np.full((n, 1), float(u_value), dtype=np.float64)
    a = dirs * float(a_value)
    return np.concatenate([xyz, u, a], axis=1).astype(np.float32)


def _split_dataset(tmp_path, n_train: int, n_val: int):
    """Write a single file [train rows ; extreme val rows] and return paths/indices."""
    train_rows = _make_rows(n_train, u_value=1.0, a_value=1.0e-3, seed=1)
    # Validation rows: 6 orders of magnitude larger targets — classic leakage trap.
    val_rows = _make_rows(n_val, u_value=1.0e6, a_value=1.0e3, seed=2)
    all_rows = np.concatenate([train_rows, val_rows], axis=0)
    path = write_toy_contract_h5(tmp_path / "single.h5", rows=all_rows)
    train_idx = np.arange(0, n_train, dtype=np.int64)
    val_idx = np.arange(n_train, n_train + n_val, dtype=np.int64)
    return path, train_idx, val_idx, train_rows


def _fit(path, *, indices=None, split_provenance=None):
    meta = DatasetMeta.from_h5(path)
    return fit_scaler_streaming(
        h5_path=path,
        dset_name="data",
        meta=meta,
        use_si=True,
        mu_si=float(MU_MOON_SI),
        a_sign=1.0,
        degree_min=2,
        degree_max=4,
        target_mode="residual",
        indices=indices,
        split_provenance=split_provenance,
    )


def test_train_only_scaler_ignores_validation_outliers(tmp_path):
    path, train_idx, _val_idx, _train_rows = _split_dataset(tmp_path, n_train=400, n_val=400)

    train_only = _fit(path, indices=train_idx)
    full = _fit(path, indices=None)  # legacy whole-file fit (includes val rows)

    # Train-only target scales reflect the modest train targets, not the 1e6 outliers.
    assert train_only.u.scale < 10.0
    assert train_only.a.scale < 1.0
    # The legacy full-file fit is contaminated by the validation outliers, so it
    # differs by orders of magnitude. If this assertion fails, the train-only fit
    # is silently seeing validation rows.
    assert full.u.scale > 100.0 * train_only.u.scale
    assert full.a.scale > 100.0 * train_only.a.scale


def test_train_only_matches_dedicated_train_file(tmp_path):
    path, train_idx, _val_idx, train_rows = _split_dataset(tmp_path, n_train=400, n_val=400)

    train_only = _fit(path, indices=train_idx)

    # Independent train/val mode: scaler fit on the train file alone (indices=None).
    train_file = write_toy_contract_h5(tmp_path / "train_only.h5", rows=train_rows)
    independent = _fit(train_file, indices=None)

    assert independent.u.scale == pytest.approx(train_only.u.scale, rel=1e-6)
    assert independent.a.scale == pytest.approx(train_only.a.scale, rel=1e-6)


def test_scaler_provenance_records_train_only(tmp_path):
    path, train_idx, _val_idx, _train_rows = _split_dataset(tmp_path, n_train=200, n_val=200)

    provenance = {
        "fit_scope": "train_only",
        "split_policy": "seeded_random",
        "split_seed": 123,
        "train_count": int(train_idx.size),
        "val_count": 200,
        "train_index_hash": "deadbeef",
    }
    scaler = _fit(path, indices=train_idx, split_provenance=provenance)

    assert scaler.provenance["fit_scope"] == "train_only"
    assert scaler.provenance["split_policy"] == "seeded_random"
    assert scaler.provenance["split_seed"] == 123
    assert scaler.provenance["train_count"] == int(train_idx.size)
    assert scaler.provenance["train_index_hash"] == "deadbeef"


def test_empty_indices_is_rejected(tmp_path):
    path, _train_idx, _val_idx, _train_rows = _split_dataset(tmp_path, n_train=64, n_val=64)
    with pytest.raises(ValueError):
        _fit(path, indices=np.asarray([], dtype=np.int64))


def test_full_file_fit_marks_full_dataset_scope(tmp_path):
    path, _train_idx, _val_idx, _train_rows = _split_dataset(tmp_path, n_train=64, n_val=64)
    scaler = _fit(path, indices=None)
    assert scaler.provenance["fit_scope"] == "full_dataset"
