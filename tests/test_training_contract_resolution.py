from __future__ import annotations

import sys

import pytest

torch = pytest.importorskip("torch")

from lunaris.surrogate.st_lrps.artifacts.manager import validate_checkpoint_contract
from lunaris.surrogate.st_lrps.training.config import parse_args
from st_lrps_contract_test_utils import make_contract_run


pytestmark = pytest.mark.requires_torch


def test_resolved_training_config_contains_full_artifact_contract(tmp_path):
    run = make_contract_run(tmp_path, degree_min=20, degree_max=60)
    cfg = run["cfg"]
    contract = cfg["artifact_contract"]

    assert cfg["dataset_contract"]["dataset_sha256"] == "0" * 64
    assert len(cfg["training_config_hash"]) == 64
    assert contract["schema_version"] == 1
    assert contract["target_mode"] == "residual"
    assert contract["baseline_kind"] == "spherical_harmonics"
    assert contract["base_degree"] == 20
    assert contract["target_degree"] == 60
    assert contract["runtime_model_kind"] == "potential_autograd"
    assert contract["architecture_signature"] == run["architecture_signature"]
    assert set(contract["scaler_contract"]) >= {"x", "u", "a"}
    assert contract["dataset_contract"]["degree_min"] == 20


def test_checkpoint_payload_repeats_contract_at_top_level(tmp_path):
    run = make_contract_run(tmp_path)
    payload = run["payload"]

    assert payload["artifact_contract"] == payload["config"]["artifact_contract"]
    assert payload["dataset_contract"] == payload["config"]["dataset_contract"]
    assert payload["training_config_hash"] == payload["config"]["training_config_hash"]
    report = validate_checkpoint_contract(payload, cfg=run["cfg"], strict=True)
    assert report["contract_source"] == "checkpoint"
    assert report["legacy_contract"] is False


def test_training_cli_exposes_legacy_dataset_escape_hatches(tmp_path, monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "st_lrps_train.py",
            "--data",
            str(tmp_path / "old_cloud.h5"),
            "--out",
            str(tmp_path / "run"),
            "--allow-legacy-target-mode-inference",
            "--allow-missing-dataset-contract",
        ],
    )

    cfg = parse_args()

    assert cfg.allow_legacy_target_mode_inference is True
    assert cfg.allow_missing_dataset_contract is True
