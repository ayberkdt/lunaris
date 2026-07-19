from __future__ import annotations

import pytest

torch = pytest.importorskip('torch')

from lunaris.surrogate.st_lrps.data.dataset_parameters import MU_MOON_SI, R_MOON_SI
from lunaris.surrogate.st_lrps.shared.contracts import (
    PAPER_SAFE_REQUIRED_METADATA,
    ArtifactContract,
    ArtifactContractError,
    paper_safe_metadata_report,
    require_paper_safe_metadata,
)


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
    report = _contract().compatibility_report(
        _contract(mu_si=MU_MOON_SI + 100.0, r_ref_m=R_MOON_SI + 1.5)
    )
    assert any("mu_si mismatch" in e for e in report["errors"])
    assert any("r_ref_m mismatch" in e for e in report["errors"])


def test_altitude_range_mismatch_warning_or_error():
    artifact = _contract(altitude_min_km=100.0, altitude_max_km=500.0)
    requested = _contract(altitude_min_km=50.0, altitude_max_km=600.0)
    assert artifact.compatibility_report(requested)["warnings"]
    assert artifact.compatibility_report(requested, strict_domain=True)["errors"]


# ---------------------------------------------------------------------------
# Paper-safe required metadata (R26)
# ---------------------------------------------------------------------------


def _paper_safe_sources() -> tuple[ArtifactContract, dict, dict]:
    contract = _contract(
        scaler_contract={**_scaler(), "provenance": {"fit_rows": 16, "alt_min_km": 100.0, "alt_max_km": 1000.0}},
        dataset_contract={
            **_dataset(),
            "dataset_sha256": "0" * 64,
            "source_gravity_model": "gggrx_1200a",
            "source_gravity_file_sha256": "1" * 64,
        },
    )
    config = {
        "central_body": "moon",
        "dtype": "float32",
        "parameter_count": 12345,
        "loss_config": {"w_u": 1.0, "w_a": 1.0, "laplacian_weight": 2e-9},
        "domain_guard_policy": "warn",
    }
    provenance = {
        "model_kind": "potential_autograd",
        "git_commit": "deadbeef" * 5,
        "created_at_utc": "2026-07-05T00:00:00Z",
        "training_config_hash": "2" * 64,
        "split_manifest_sha256": "3" * 64,
    }
    return contract, config, provenance


def test_paper_safe_metadata_report_complete():
    contract, config, provenance = _paper_safe_sources()
    report = paper_safe_metadata_report(contract=contract, config=config, run_provenance=provenance)
    assert report["complete"] is True
    assert report["missing"] == []
    # Every required logical field resolved to a value.
    for name in PAPER_SAFE_REQUIRED_METADATA:
        assert report["fields"][name] is not None, name


def test_paper_safe_metadata_report_lists_missing_fields():
    contract, config, provenance = _paper_safe_sources()
    config.pop("loss_config")
    config.pop("parameter_count")
    provenance.pop("git_commit")
    report = paper_safe_metadata_report(contract=contract, config=config, run_provenance=provenance)
    assert report["complete"] is False
    assert {"loss_config", "parameter_count", "git_commit"}.issubset(set(report["missing"]))


def test_require_paper_safe_metadata_raises_naming_missing_fields():
    contract, config, provenance = _paper_safe_sources()
    provenance.pop("git_commit")
    config.pop("domain_guard_policy")
    with pytest.raises(ArtifactContractError, match="git_commit") as excinfo:
        require_paper_safe_metadata(contract=contract, config=config, run_provenance=provenance)
    assert "domain_guard_policy" in str(excinfo.value)
    assert "paper_safe" in str(excinfo.value)


def test_require_paper_safe_metadata_passes_and_returns_report():
    contract, config, provenance = _paper_safe_sources()
    report = require_paper_safe_metadata(contract=contract, config=config, run_provenance=provenance)
    assert report["complete"] is True


def test_paper_safe_metadata_missing_scaler_provenance_detected():
    contract, config, provenance = _paper_safe_sources()
    # _scaler() has no provenance block -> scaler_source unresolvable.
    bare = _contract(
        dataset_contract={
            **_dataset(),
            "dataset_sha256": "0" * 64,
            "source_gravity_model": "gggrx_1200a",
            "source_gravity_file_sha256": "1" * 64,
        }
    )
    report = paper_safe_metadata_report(contract=bare, config=config, run_provenance=provenance)
    assert "scaler_source" in report["missing"]


def test_roundtrip_to_dict_from_dict():
    c = _contract()
    assert ArtifactContract.from_dict(c.to_dict()).to_dict() == c.to_dict()


# ---------------------------------------------------------------------------
# Dataset hash normalization (AUD-002): the canonical DatasetContract writes
# content_sha256; readers must not treat the rename as a missing hash.
# ---------------------------------------------------------------------------


def test_resolve_dataset_hash_prefers_canonical_content_sha256():
    from lunaris.surrogate.st_lrps.shared.contracts import resolve_dataset_hash

    canonical = {"content_sha256": "c" * 64, "dataset_sha256": "d" * 64}
    assert resolve_dataset_hash(canonical) == "c" * 64
    assert resolve_dataset_hash({"dataset_sha256": "d" * 64}) == "d" * 64
    assert resolve_dataset_hash({"sha256": "e" * 64}) == "e" * 64
    # Source order: the first mapping that carries any hash key wins.
    assert resolve_dataset_hash({}, {"dataset_hash": "f" * 64}) == "f" * 64
    assert resolve_dataset_hash(None, {"content_sha256": ""}) is None


def test_paper_safe_training_data_hash_resolves_from_content_sha256():
    contract, config, provenance = _paper_safe_sources()
    canonical_only = _contract(
        scaler_contract={**_scaler(), "provenance": {"fit_rows": 16, "alt_min_km": 100.0, "alt_max_km": 1000.0}},
        dataset_contract={
            **_dataset(),
            # Canonical DatasetContract key only — no legacy dataset_sha256.
            "content_sha256": "a" * 64,
            "source_gravity_model": "gggrx_1200a",
            "source_gravity_file_sha256": "1" * 64,
        },
    )
    report = paper_safe_metadata_report(contract=canonical_only, config=config, run_provenance=provenance)
    assert report["fields"]["training_data_hash"] == "a" * 64
    assert "training_data_hash" not in report["missing"]


def test_compatibility_report_accepts_content_sha256_as_dataset_hash():
    canonical_only = _contract(
        dataset_contract={
            **_dataset(),
            "content_sha256": "a" * 64,
            "source_gravity_file_sha256": "1" * 64,
        },
    )
    report = canonical_only.compatibility_report(canonical_only.to_dict())
    assert not any("dataset_contract.content_sha256" in w for w in report["warnings"]), (
        "content_sha256-only contract must not warn about a missing dataset hash"
    )


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
