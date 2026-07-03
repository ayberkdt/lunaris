"""D2 (reviewer §10): unit tests for the result trust classifier."""

from __future__ import annotations

from lunaris.analysis.ensemble.result_audit import (
    INVALID,
    QUARANTINED,
    RERUN,
    TRUSTED,
    classify_batch_archive,
    classify_st_lrps_run,
)
from lunaris.common.hashing import canonical_json_sha256

# Minimal complete v2 manifest backbone (mirrors the strict-loader contract).
_V2_BACKBONE = {
    "archive_schema_version": 2,
    "n_samples": 16,
    "seed": 1,
    "duration_s": 3600.0,
    "output_dt_s": 60.0,
    "backend": "cpu",
    "requested_batch_backend": "cpu",
    "actual_batch_backend": "cpu",
    "batch_backend": "cpu",
    "detect_impact": True,
    "compute_impact_statistics": True,
}


_FULL_V2 = {
    **_V2_BACKBONE,
    "impact_frame_available": True,
    "backend_diagnostics": {"impact_position_method": "line_sphere_quadratic"},
    "kernel_source_sha256": "a" * 64,
}


def test_missing_schema_archive_is_invalid() -> None:
    status, reasons = classify_batch_archive({"backend": "cpu"}, has_impacts=False)
    assert status == INVALID
    assert any("archive_schema_version" in r for r in reasons)


def test_pre_contract_schema_v1_is_invalid() -> None:
    status, reasons = classify_batch_archive(
        {"archive_schema_version": 1}, has_impacts=False
    )
    assert status == INVALID
    assert any("v1" in r for r in reasons)


def test_corrupt_schema_version_is_invalid() -> None:
    status, reasons = classify_batch_archive(
        {"archive_schema_version": "not-an-int"}, has_impacts=False
    )
    assert status == INVALID
    assert any("corrupt" in r for r in reasons)


def test_v2_missing_required_manifest_field_is_invalid() -> None:
    meta = dict(_FULL_V2)
    del meta["seed"]
    status, reasons = classify_batch_archive(meta, has_impacts=True)
    assert status == INVALID
    assert any("seed" in r for r in reasons)


def test_full_v2_with_impacts_is_trusted() -> None:
    status, reasons = classify_batch_archive(_FULL_V2, has_impacts=True)
    assert status == TRUSTED
    assert reasons == []


def test_v2_impacts_without_frame_is_rerun() -> None:
    meta = dict(_FULL_V2)
    meta["impact_frame_available"] = False
    status, reasons = classify_batch_archive(meta, has_impacts=True)
    assert status == RERUN
    assert any("§3" in r for r in reasons)


def test_v2_impacts_with_step_endpoint_is_rerun() -> None:
    meta = dict(_FULL_V2)
    meta["backend_diagnostics"] = {"impact_position_method": "rk4_step_frozen"}
    status, reasons = classify_batch_archive(meta, has_impacts=True)
    assert status == RERUN
    assert any("§6" in r for r in reasons)


def test_v2_impacts_with_old_radius_interpolation_is_rerun() -> None:
    meta = dict(_FULL_V2)
    meta["backend_diagnostics"] = {
        "impact_position_method": "rk4_crossing_interpolated"
    }
    status, reasons = classify_batch_archive(meta, has_impacts=True)
    assert status == RERUN
    assert any("line-sphere" in r for r in reasons)


def test_v2_no_impacts_missing_hashes_is_quarantined() -> None:
    meta = {
        **_V2_BACKBONE,
        "impact_frame_available": False,  # irrelevant: no impacts
        "backend_diagnostics": {},
    }
    status, reasons = classify_batch_archive(meta, has_impacts=False)
    assert status == QUARANTINED
    assert any("§7" in r for r in reasons)


def test_v2_empty_or_malformed_hash_is_not_provenance() -> None:
    for value in (None, "", "not-a-sha256", "g" * 64):
        meta = dict(_FULL_V2)
        meta["kernel_source_sha256"] = value
        status, reasons = classify_batch_archive(meta, has_impacts=False)
        assert status == QUARANTINED
        assert any("§7" in r for r in reasons)


def test_st_lrps_run_without_contract_is_rerun() -> None:
    status, reasons = classify_st_lrps_run({"hidden": 256}, has_checkpoint=True)
    assert status == RERUN
    assert any("artifact_contract" in r for r in reasons)


# Contract used across the ST-LRPS classifier tests; the provenance digest must
# match a fresh recompute (the auditor verifies it), so derive it for real.
_CONTRACT = {"x": 1}

# Complete run_provenance block (mirrors build_run_provenance's TRUSTED contract).
_FULL_PROVENANCE = {
    "provenance_version": "st_lrps_run_provenance_v1",
    "created_at_utc": "2026-06-16T00:00:00Z",
    "git_commit": None,  # legitimately None outside a checkout
    "torch_version": "2.4.0",
    "cuda_version": None,  # legitimately None on CPU-only hosts
    "model_kind": "potential_autograd",
    "determinism": {"use_deterministic_algorithms": True},
    "artifact_contract_sha256": canonical_json_sha256(_CONTRACT),
    "split_manifest_sha256": "b" * 64,
}


def _st_lrps_config(provenance=None, *, contract=_CONTRACT):
    cfg = {"artifact_contract": dict(contract)}
    if provenance is not None:
        cfg["run_provenance"] = dict(provenance)
    return cfg


def test_st_lrps_run_with_contract_but_no_provenance_is_quarantined() -> None:
    status, reasons = classify_st_lrps_run(_st_lrps_config(), has_checkpoint=True)
    assert status == QUARANTINED
    assert any("run_provenance" in r for r in reasons)


def test_st_lrps_run_with_full_provenance_is_trusted() -> None:
    status, reasons = classify_st_lrps_run(
        _st_lrps_config(_FULL_PROVENANCE), has_checkpoint=True
    )
    assert status == TRUSTED
    assert reasons == []


def test_st_lrps_run_missing_core_provenance_field_is_invalid() -> None:
    prov = dict(_FULL_PROVENANCE)
    del prov["torch_version"]
    status, reasons = classify_st_lrps_run(_st_lrps_config(prov), has_checkpoint=True)
    assert status == INVALID
    assert any("torch_version" in r for r in reasons)


def test_st_lrps_run_unknown_model_kind_is_invalid() -> None:
    prov = dict(_FULL_PROVENANCE, model_kind="magic")
    status, reasons = classify_st_lrps_run(_st_lrps_config(prov), has_checkpoint=True)
    assert status == INVALID
    assert any("model_kind" in r for r in reasons)


def test_st_lrps_run_corrupt_contract_hash_is_invalid() -> None:
    prov = dict(_FULL_PROVENANCE, artifact_contract_sha256="not-a-hash")
    status, reasons = classify_st_lrps_run(_st_lrps_config(prov), has_checkpoint=True)
    assert status == INVALID
    assert any("artifact_contract_sha256" in r for r in reasons)


def test_st_lrps_run_contract_hash_mismatch_is_invalid() -> None:
    # Valid-format digest, but it does not match the stored contract: config.json
    # was edited after training, or the contract/hash drifted.
    prov = dict(_FULL_PROVENANCE, artifact_contract_sha256=canonical_json_sha256({"x": 2}))
    status, reasons = classify_st_lrps_run(_st_lrps_config(prov), has_checkpoint=True)
    assert status == INVALID
    assert any("does not match" in r for r in reasons)


def test_st_lrps_run_without_determinism_or_split_is_quarantined() -> None:
    # Real run from a lighter harness: core valid, reproducibility evidence absent.
    for drop in ("determinism", "split_manifest_sha256"):
        prov = dict(_FULL_PROVENANCE)
        del prov[drop]
        status, reasons = classify_st_lrps_run(_st_lrps_config(prov), has_checkpoint=True)
        assert status == QUARANTINED, drop
        assert any(drop.split("_")[0] in r for r in reasons)


def test_st_lrps_run_with_empty_or_malformed_contract_is_rerun() -> None:
    for contract in ({}, None, "legacy"):
        status, reasons = classify_st_lrps_run(
            {"artifact_contract": contract},
            has_checkpoint=True,
        )
        assert status == RERUN
        assert any("artifact_contract" in r for r in reasons)


def test_st_lrps_run_without_checkpoint_is_rerun() -> None:
    status, reasons = classify_st_lrps_run({"artifact_contract": {}}, has_checkpoint=False)
    assert status == RERUN
    assert any("checkpoint" in r for r in reasons)
