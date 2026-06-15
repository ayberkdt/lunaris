"""D2 (reviewer §10): unit tests for the legacy-result trust classifier."""

from __future__ import annotations

from lunaris.analysis.monte_carlo.result_audit import (
    QUARANTINED,
    RERUN,
    TRUSTED,
    classify_mc_archive,
    classify_st_lrps_run,
)

_FULL_V2 = {
    "archive_schema_version": 2,
    "impact_frame_available": True,
    "backend_diagnostics": {"impact_position_method": "line_sphere_quadratic"},
    "kernel_source_sha256": "a" * 64,
}


def test_pre_v2_archive_is_rerun_required() -> None:
    status, reasons = classify_mc_archive({"backend": "cpu"}, has_impacts=False)
    assert status == RERUN
    assert any("pre-v2" in r for r in reasons)


def test_full_v2_with_impacts_is_trusted() -> None:
    status, reasons = classify_mc_archive(_FULL_V2, has_impacts=True)
    assert status == TRUSTED
    assert reasons == []


def test_v2_impacts_without_frame_is_rerun() -> None:
    meta = dict(_FULL_V2)
    meta["impact_frame_available"] = False
    status, reasons = classify_mc_archive(meta, has_impacts=True)
    assert status == RERUN
    assert any("§3" in r for r in reasons)


def test_v2_impacts_with_step_endpoint_is_rerun() -> None:
    meta = dict(_FULL_V2)
    meta["backend_diagnostics"] = {"impact_position_method": "rk4_step_frozen"}
    status, reasons = classify_mc_archive(meta, has_impacts=True)
    assert status == RERUN
    assert any("§6" in r for r in reasons)


def test_v2_impacts_with_old_radius_interpolation_is_rerun() -> None:
    meta = dict(_FULL_V2)
    meta["backend_diagnostics"] = {
        "impact_position_method": "rk4_crossing_interpolated"
    }
    status, reasons = classify_mc_archive(meta, has_impacts=True)
    assert status == RERUN
    assert any("line-sphere" in r for r in reasons)


def test_v2_no_impacts_missing_hashes_is_quarantined() -> None:
    meta = {
        "archive_schema_version": 2,
        "impact_frame_available": False,  # irrelevant: no impacts
        "backend_diagnostics": {},
    }
    status, reasons = classify_mc_archive(meta, has_impacts=False)
    assert status == QUARANTINED
    assert any("§7" in r for r in reasons)


def test_v2_empty_or_malformed_hash_is_not_provenance() -> None:
    for value in (None, "", "not-a-sha256", "g" * 64):
        meta = dict(_FULL_V2)
        meta["kernel_source_sha256"] = value
        status, reasons = classify_mc_archive(meta, has_impacts=False)
        assert status == QUARANTINED
        assert any("§7" in r for r in reasons)


def test_st_lrps_run_without_contract_is_rerun() -> None:
    status, reasons = classify_st_lrps_run({"hidden": 256}, has_checkpoint=True)
    assert status == RERUN
    assert any("artifact_contract" in r for r in reasons)


def test_st_lrps_run_with_contract_is_trusted() -> None:
    status, reasons = classify_st_lrps_run({"artifact_contract": {"x": 1}}, has_checkpoint=True)
    assert status == TRUSTED
    assert reasons == []


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
