from __future__ import annotations

import json

import h5py
import pytest

from lunaris.surrogate.st_lrps.data.dataset_contract import DatasetContractError
from lunaris.surrogate.st_lrps.data.dataset_parameters import MU_MOON_SI, R_MOON_SI
from lunaris.surrogate.st_lrps.training.config import TrainConfig
from lunaris.surrogate.st_lrps.training.engine import train

from dataset_pipeline_test_utils import make_toy_residual_rows, write_toy_contract_h5

pytest.importorskip("torch")


def _tiny_train_cfg(data_path, out_dir, **overrides):
    kwargs = {
        "data": str(data_path),
        "out": str(out_dir),
        "epochs": 1,
        "batch_size": 8,
        "num_workers": 0,
        "pin_memory": False,
        "quick_check": True,
        "max_train_batches": 1,
        "max_val_batches": 1,
        "activation": "silu",
        "hidden": 8,
        "depth": 2,
        "model_preset": "custom",
        "use_residual_blocks": False,
        "n_bands": 1,
        "val_ratio": 0.25,
        "split_seed": 11,
        "split_policy": "seeded_random",
        "a_sign": 1.0,
        "gradnorm_mode": "fixed",
        "direction_loss_weight": 0.0,
        "use_altitude_balanced_loss": False,
        "use_radial_cross_loss": False,
        "preload_policy": "never",
        "auto_preload_mb": 0.0,
        "fit_rows": 32,
        "fit_chunk_rows": 32,
        "cache_rows": 16,
        "sampler_block_size": 16,
        "warmup_epochs": 0,
        "best_ckpt_start_epoch": 0,
        "log_every": 0,
    }
    kwargs.update(overrides)
    return TrainConfig(**kwargs)


def test_training_rejects_hdf5_without_dataset_contract(tmp_path):
    data_path = tmp_path / "legacy.h5"
    rows = make_toy_residual_rows(n=16)
    with h5py.File(data_path, "w") as handle:
        handle.create_dataset("data", data=rows)
        handle.attrs["unit_system"] = "si"
        handle.attrs["central_body"] = "moon"
        handle.attrs["mu_si"] = float(MU_MOON_SI)
        handle.attrs["r_ref_m"] = float(R_MOON_SI)
        handle.attrs["requested_degree"] = 4
        handle.attrs["degree_min"] = 2
        handle.attrs["degree_max"] = 4
        handle.attrs["target_mode"] = "residual"
        handle.attrs["baseline_kind"] = "spherical_harmonics"
        handle.attrs["alt_min_km"] = 100.0
        handle.attrs["alt_max_km"] = 500.0
        handle.attrs["a_sign_convention"] = "+1"
        handle.attrs["derivative_convention_version"] = "dP_dphi_corrected_v1"
        handle.attrs["columns"] = "[x,y,z,dU,dax,day,daz]"

    cfg = _tiny_train_cfg(data_path, tmp_path / "run_missing_contract")

    with pytest.raises(DatasetContractError, match="dataset_contract_json"):
        train(cfg)


def test_training_writes_dataset_validation_and_split_manifest(tmp_path):
    data_path = write_toy_contract_h5(tmp_path / "toy_train.h5", n=48)
    run_dir = tmp_path / "run"

    train(_tiny_train_cfg(data_path, run_dir, split_policy="altitude_stratified"))

    validation_path = run_dir / "provenance" / "dataset_validation_report.json"
    split_path = run_dir / "provenance" / "split_manifest.json"
    dataset_meta_path = run_dir / "provenance" / "dataset_meta.json"
    manifest_path = run_dir / "run_manifest.json"

    assert validation_path.exists()
    assert split_path.exists()
    assert dataset_meta_path.exists()
    assert manifest_path.exists()
    assert json.loads(validation_path.read_text(encoding="utf-8"))["passed"] is True
    split = json.loads(split_path.read_text(encoding="utf-8"))
    assert split["split_policy"] == "altitude_stratified"
    assert split["train_count"] + split["val_count"] == 48
    dataset_meta = json.loads(dataset_meta_path.read_text(encoding="utf-8"))
    assert dataset_meta["dataset_contract"]["dataset_id"] == "toy_residual_cloud"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["dataset_validation_passed"] is True
    assert manifest["split_manifest_path"] == str(split_path)
