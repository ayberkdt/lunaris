from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from lunaris.surrogate.st_lrps.artifacts.manager import validate_checkpoint_contract
from lunaris.surrogate.st_lrps.data.dataset_parameters import R_MOON_SI
from lunaris.surrogate.st_lrps.runtime.force_model import (
    DirectForceRuntime,
    load_surrogate_force_model,
)
from lunaris.surrogate.st_lrps.shared.contracts import ArtifactContractError
from st_lrps_contract_test_utils import make_contract_run

pytestmark = pytest.mark.requires_torch


def test_runtime_loader_exposes_valid_artifact_contract(tmp_path):
    run = make_contract_run(tmp_path, degree_min=20, degree_max=60)

    fm = load_surrogate_force_model(run["run_dir"], device="cpu")

    assert fm.artifact_contract.target_mode == "residual"
    assert fm.artifact_contract.base_degree == 20
    assert fm.artifact_contract.target_degree == 60
    assert fm.target_contract.baseline_kind == "spherical_harmonics"
    status = fm.domain_status(np.array([R_MOON_SI + 200_000.0, 0.0, 0.0]))
    assert status["in_training_altitude_range"] is True


def test_runtime_rejects_missing_artifact_contract(tmp_path):
    run = make_contract_run(tmp_path, include_contract=False)

    with pytest.raises(ArtifactContractError, match="missing artifact_contract"):
        load_surrogate_force_model(run["run_dir"], device="cpu")


def test_checkpoint_contract_cross_check_rejects_mismatched_baseline_degree(tmp_path):
    run = make_contract_run(tmp_path, degree_min=20, degree_max=60, contract_overrides={"base_degree": 30})

    with pytest.raises(ArtifactContractError, match="disagrees|degree"):
        validate_checkpoint_contract(run["payload"], cfg=run["cfg"], strict=True)
    with pytest.raises(ArtifactContractError, match="disagrees|degree"):
        load_surrogate_force_model(run["run_dir"], device="cpu")


def test_runtime_loads_force_direct_artifact(tmp_path):
    run = make_contract_run(
        tmp_path,
        cfg_overrides={"runtime_model_kind": "force_direct", "prediction_kind": "residual_force", "output_dim": 3},
    )

    fm = load_surrogate_force_model(run["run_dir"], device="cpu")

    assert isinstance(fm, DirectForceRuntime)


def test_runtime_rejects_force_direct_contract_mismatch(tmp_path):
    run = make_contract_run(
        tmp_path,
        contract_overrides={"runtime_model_kind": "force_direct", "prediction_kind": "residual_force", "output_dim": 3},
    )

    with pytest.raises(ArtifactContractError, match="runtime_model_kind|output_dim"):
        load_surrogate_force_model(run["run_dir"], device="cpu")


def test_strict_domain_uses_artifact_altitude_envelope(tmp_path):
    run = make_contract_run(tmp_path, alt_min_km=100.0, alt_max_km=300.0)
    fm = load_surrogate_force_model(run["run_dir"], device="cpu", strict_domain=True)

    status = fm.domain_status(np.array([R_MOON_SI + 600_000.0, 0.0, 0.0]))

    assert status["in_training_altitude_range"] is False
    with pytest.raises(RuntimeError, match="strict_domain"):
        fm.predict_residual_accel(np.array([R_MOON_SI + 600_000.0, 0.0, 0.0]))
