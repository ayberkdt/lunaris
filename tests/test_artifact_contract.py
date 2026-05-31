from __future__ import annotations

import pytest

from lunaris.surrogate.st_lrps.data.dataset_parameters import MU_MOON_SI, R_MOON_SI
from lunaris.surrogate.st_lrps.shared.contracts import ArtifactContract, ArtifactContractError


def _scaler() -> dict:
    return {
        "schema_version": 1,
        "kind": "isometric",
        "x": {"scale": R_MOON_SI + 200_000.0},
        "u": {"scale": 1.0},
        "a": {"scale": 1.0},
    }


def _dataset(*, degree_min: int = 20, degree_max: int = 200) -> dict:
    return {
        "schema_version": 1,
        "dataset_kind": "st_lrps_spatial_cloud",
        "target_mode": "residual",
        "degree_min": degree_min,
        "degree_max": degree_max,
        "mu_si": MU_MOON_SI,
        "r_ref_m": R_MOON_SI,
        "altitude_min_km": 100.0,
        "altitude_max_km": 1000.0,
        "units": {"position": "m", "potential": "m^2/s^2", "acceleration": "m/s^2"},
    }


def _contract(**overrides) -> ArtifactContract:
    payload = {
        "schema_version": 1,
        "target_mode": "residual",
        "baseline_kind": "spherical_harmonics",
        "base_degree": 20,
        "target_degree": 200,
        "runtime_model_kind": "potential_autograd",
        "prediction_kind": "residual_potential",
        "mu_si": MU_MOON_SI,
        "r_ref_m": R_MOON_SI,
        "a_sign": 1.0,
        "altitude_min_km": 100.0,
        "altitude_max_km": 1000.0,
        "input_encoding": {"embedding_type": "raw", "input_feature_dim": 3},
        "scaler_contract": _scaler(),
        "dataset_contract": _dataset(),
        "architecture_signature": "abc123",
    }
    payload.update(overrides)
    return ArtifactContract.from_dict(payload)


def test_valid_residual_contract_passes():
    assert _contract().base_degree == 20


def test_valid_full_field_contract_passes():
    c = _contract(
        target_mode="full",
        baseline_kind="none",
        base_degree=-1,
        target_degree=200,
        prediction_kind="potential",
        dataset_contract={**_dataset(degree_min=-1, degree_max=200), "target_mode": "full"},
    )
    assert c.target_mode == "full"


def test_missing_target_mode_fails():
    with pytest.raises(ArtifactContractError, match="target_mode"):
        _contract(target_mode="")


def test_invalid_runtime_model_kind_fails():
    with pytest.raises(ArtifactContractError, match="runtime_model_kind"):
        _contract(runtime_model_kind="force_direct")


def test_residual_contract_missing_baseline_degree_fails():
    with pytest.raises(ArtifactContractError, match="base_degree"):
        _contract(base_degree=-1)


def test_degree_min_greater_or_equal_degree_max_fails():
    with pytest.raises(ArtifactContractError, match="target_degree"):
        _contract(base_degree=20, target_degree=20)


def test_incompatible_baseline_degree_detected():
    artifact = _contract()
    requested = _contract(base_degree=30)
    report = artifact.compatibility_report(requested)
    assert not report["compatible"]
    assert any("degree 20" in e or "degree 30" in e for e in report["errors"])


def test_incompatible_target_degree_detected():
    report = _contract().compatibility_report(_contract(target_degree=100))
    assert any("target_degree mismatch" in e for e in report["errors"])


def test_incompatible_mu_and_r_ref_detected():
    report = _contract().compatibility_report(_contract(mu_si=MU_MOON_SI + 100.0, r_ref_m=R_MOON_SI + 10.0))
    assert any("mu_si mismatch" in e for e in report["errors"])
    assert any("r_ref_m mismatch" in e for e in report["errors"])


def test_altitude_range_mismatch_warning_or_error():
    artifact = _contract(altitude_min_km=100.0, altitude_max_km=500.0)
    requested = _contract(altitude_min_km=50.0, altitude_max_km=600.0)
    assert artifact.compatibility_report(requested)["warnings"]
    assert artifact.compatibility_report(requested, strict_domain=True)["errors"]


def test_roundtrip_to_dict_from_dict():
    c = _contract()
    assert ArtifactContract.from_dict(c.to_dict()).to_dict() == c.to_dict()
