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


def test_force_direct_contract_is_rejected_as_archived():
    # force_direct is archived in experimental/force-direct-archive and can no
    # longer be validated/loaded on main.
    with pytest.raises(ArtifactContractError, match="archive|force_direct"):
        _contract(runtime_model_kind="force_direct", prediction_kind="residual_force", output_dim=3)


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


# ---------------------------------------------------------------------------
# validate(): every rejection branch (mutate one field of a valid contract).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"schema_version": 99}, "schema_version"),
        ({"baseline_kind": "bogus"}, "baseline_kind"),
        ({"runtime_model_kind": "bogus"}, "runtime_model_kind"),
        ({"prediction_kind": "bogus"}, "prediction_kind"),
        ({"a_sign": 2.0}, "a_sign"),
        ({"mu_si": 1.0, "r_ref_m": 1.0}, "lunar"),
        ({"altitude_min_km": 1000.0, "altitude_max_km": 100.0}, "altitude_max_km"),
        ({"input_encoding": {}}, "input_encoding"),
        ({"scaler_contract": {}}, "scaler_contract is required"),
        ({"scaler_contract": {"schema_version": 1, "kind": "x"}}, "x/u/a"),
        ({"dataset_contract": {}}, "dataset_contract is required"),
        ({"dataset_contract": {**_dataset(), "degree_min": None}}, "dataset_contract.degree_min"),
    ],
)
def test_validate_rejects_each_invalid_field(overrides, match):
    with pytest.raises(ArtifactContractError, match=match):
        _contract(**overrides)


def test_validate_full_field_requires_nonnegative_target_degree():
    with pytest.raises(ArtifactContractError, match="target_degree"):
        _contract(
            target_mode="full",
            baseline_kind="none",
            base_degree=-1,
            target_degree=-5,
            prediction_kind="potential",
            dataset_contract={**_dataset(degree_min=-1, degree_max=200), "target_mode": "full"},
        )


# ---------------------------------------------------------------------------
# from_benchmark_config
# ---------------------------------------------------------------------------

def _benchmark_cfg(*, enabled: bool = True) -> dict:
    return {
        "scenario": {"altitude_min_km": 100.0, "altitude_max_km": 1000.0},
        "truth": {"degree": 200},
        "surrogate": {"enabled": enabled, "baseline_degree": 20 if enabled else -1,
                      "runtime_model_kind": "potential_autograd"},
        "propagation": {"dtype": "float64"},
        "resolved_mu_si": MU_MOON_SI,
        "resolved_r_ref_m": R_MOON_SI,
        "resolved_a_sign": 1.0,
    }


def test_from_benchmark_config_residual_and_full():
    residual = ArtifactContract.from_benchmark_config(_benchmark_cfg(enabled=True))
    assert residual.target_mode == "residual"
    assert residual.base_degree == 20 and residual.target_degree == 200
    full = ArtifactContract.from_benchmark_config(_benchmark_cfg(enabled=False))
    assert full.target_mode == "full"
    assert full.baseline_kind == "point_mass"


# ---------------------------------------------------------------------------
# require_compatible + remaining compatibility_report branches
# ---------------------------------------------------------------------------

def test_require_compatible_passes_and_raises():
    artifact = _contract()
    # Identical request -> compatible, returns the report without raising.
    report = artifact.require_compatible(_contract())
    assert report["compatible"] is True
    # Incompatible request -> raises in strict mode.
    with pytest.raises(ArtifactContractError):
        artifact.require_compatible(_contract(target_degree=100))
    # strict=False -> returns the report even with errors.
    soft = artifact.require_compatible(_contract(target_degree=100), strict=False)
    assert soft["compatible"] is False and soft["errors"]


def test_compatibility_report_a_sign_and_missing_envelope_branches():
    # a_sign mismatch is a hard error.
    rep = _contract().compatibility_report(_contract(a_sign=-1.0))
    assert any("a_sign mismatch" in e for e in rep["errors"])
    # Missing artifact altitude envelope -> domain cannot be audited (warning).
    art = _contract(altitude_min_km=None, altitude_max_km=None)
    rep2 = art.compatibility_report(_contract())
    assert any("altitude envelope is missing" in w for w in rep2["warnings"])
    # Missing dataset provenance hashes -> warnings.
    rep3 = _contract().compatibility_report(_contract())
    assert any("dataset_sha256" in w for w in rep3["warnings"])
