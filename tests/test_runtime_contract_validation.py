from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from lunaris.surrogate.st_lrps.artifacts.manager import validate_checkpoint_contract
from lunaris.surrogate.st_lrps.data.dataset_parameters import R_MOON_SI
from lunaris.surrogate.st_lrps.runtime.force_model import (
    load_surrogate_force_model,
)
from lunaris.surrogate.st_lrps.shared.contracts import ArtifactContractError
from st_lrps_contract_test_utils import make_contract_run, strip_paper_safe_fields

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


def test_runtime_rejects_archived_force_direct_contract(tmp_path):
    # A legacy artifact whose contract still declares the archived force_direct
    # kind must be rejected fail-closed rather than loaded.
    run = make_contract_run(
        tmp_path,
        contract_overrides={"runtime_model_kind": "force_direct", "prediction_kind": "residual_force", "output_dim": 3},
    )

    with pytest.raises(ArtifactContractError, match="archive|force_direct"):
        load_surrogate_force_model(run["run_dir"], device="cpu")


# ---------------------------------------------------------------------------
# Paper-safe metadata gate at load time (R26)
# ---------------------------------------------------------------------------


def test_paper_safe_load_accepts_complete_artifact(tmp_path):
    run = make_contract_run(tmp_path, degree_min=20, degree_max=60)

    fm = load_surrogate_force_model(run["run_dir"], device="cpu", paper_safe=True)

    assert fm.paper_safe_metadata is not None
    assert fm.paper_safe_metadata["complete"] is True
    assert fm.paper_safe_metadata["missing"] == []
    fields = fm.paper_safe_metadata["fields"]
    assert fields["runtime_kind"] == "potential_autograd"
    assert fields["gravity_model_name"] == "synthetic_gggrx_1200a_fixture"
    assert fields["parameter_count"] > 0
    assert fields["loss_config"]


def test_paper_safe_load_rejects_incomplete_artifact(tmp_path):
    run = make_contract_run(tmp_path, degree_min=20, degree_max=60)
    strip_paper_safe_fields(run, fields=("loss_config", "parameter_count", "domain_guard_policy"))

    with pytest.raises(ArtifactContractError, match="parameter_count") as excinfo:
        load_surrogate_force_model(run["run_dir"], device="cpu", paper_safe=True)
    message = str(excinfo.value)
    assert "loss_config" in message
    assert "domain_guard_policy" in message
    assert "paper_safe" in message


def test_research_mode_records_legacy_metadata_override(tmp_path):
    run = make_contract_run(tmp_path, degree_min=20, degree_max=60)
    strip_paper_safe_fields(run, fields=("loss_config", "parameter_count"))

    with pytest.warns(RuntimeWarning, match="paper-safe"):
        fm = load_surrogate_force_model(run["run_dir"], device="cpu", paper_safe=False)

    assert fm.paper_safe_metadata is not None
    assert fm.paper_safe_metadata["complete"] is False
    assert fm.paper_safe_metadata["legacy_override"] is True
    assert {"loss_config", "parameter_count"}.issubset(set(fm.paper_safe_metadata["missing"]))


def test_strict_domain_uses_artifact_altitude_envelope(tmp_path):
    run = make_contract_run(tmp_path, alt_min_km=100.0, alt_max_km=300.0)
    fm = load_surrogate_force_model(run["run_dir"], device="cpu", strict_domain=True)

    status = fm.domain_status(np.array([R_MOON_SI + 600_000.0, 0.0, 0.0]))

    assert status["in_training_altitude_range"] is False
    with pytest.raises(RuntimeError, match="strict_domain"):
        fm.predict_residual_accel(np.array([R_MOON_SI + 600_000.0, 0.0, 0.0]))
