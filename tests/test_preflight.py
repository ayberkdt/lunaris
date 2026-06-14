"""
Tests for the shared, flow-agnostic pre-flight service.

These assert the §4/§5 contract: invalid backend/degree/force-model combinations
are caught *before* a run, with no silent degree clipping and no silent
force-model disabling. CPU-only; no CUDA or large data required.
"""

from __future__ import annotations

from lunaris.common.type_defs import PerturbationFlags
from lunaris.core.preflight import (
    ERROR,
    PreflightReport,
    backend_preflight,
    check_backend_capability,
    check_output_dir_writable,
)


def test_unknown_backend_is_error() -> None:
    issues = check_backend_capability(requested_backend="made_up")
    assert len(issues) == 1
    assert issues[0].severity == ERROR
    assert issues[0].code == "backend_unknown"


def test_gpu_sh_high_degree_is_reported_not_clipped() -> None:
    issues = check_backend_capability(requested_backend="gpu_sh", requested_sh_degree=100)
    codes = {i.code for i in issues}
    assert "sh_degree_exceeds_backend" in codes
    issue = next(i for i in issues if i.code == "sh_degree_exceeds_backend")
    assert issue.severity == ERROR
    # The requested degree must appear verbatim — proof it is surfaced, not reduced.
    assert "100" in issue.message
    assert "requested_degree=100" in issue.detail


def test_gpu_sh_within_degree_limit_is_clean() -> None:
    issues = check_backend_capability(requested_backend="gpu_sh", requested_sh_degree=20)
    assert issues == []


def test_gpu_sh_unsupported_force_model_is_error() -> None:
    flags = PerturbationFlags(enable_sh=True, enable_albedo=True)
    issues = check_backend_capability(requested_backend="gpu_sh", requested_sh_degree=20, flags=flags)
    codes = {i.code for i in issues}
    assert "force_model_unsupported" in codes
    issue = next(i for i in issues if i.code == "force_model_unsupported")
    assert "albedo" in issue.message and "albedo" in issue.detail


def test_gpu_sh_supported_perturbations_are_clean() -> None:
    # Classic-SH GPU supports third-body, Earth J2, SRP, and 1PN relativity.
    flags = PerturbationFlags(
        enable_sh=True,
        enable_3rd_body_sun=True,
        enable_earth_j2=True,
        enable_srp=True,
        enable_relativity_1pn=True,
    )
    issues = check_backend_capability(requested_backend="gpu_sh", requested_sh_degree=20, flags=flags)
    assert issues == []


def test_gpu_st_lrps_extra_physics_is_error() -> None:
    flags = PerturbationFlags(enable_sh=True, enable_3rd_body_sun=True)
    issues = check_backend_capability(requested_backend="gpu_st_lrps_potential", flags=flags)
    assert any(i.code == "force_model_unsupported" for i in issues)


def test_cpu_backend_accepts_everything() -> None:
    flags = PerturbationFlags(
        enable_sh=True,
        enable_3rd_body_sun=True,
        enable_3rd_body_earth=True,
        enable_earth_j2=True,
        enable_srp=True,
        enable_albedo=True,
        enable_thermal=True,
        enable_tides_k2=True,
        enable_tides_k3=True,
        enable_relativity_1pn=True,
    )
    issues = check_backend_capability(requested_backend="cpu_sh", requested_sh_degree=200, flags=flags)
    assert issues == []


def test_output_dir_writable_roundtrip(tmp_path) -> None:
    issues = check_output_dir_writable(tmp_path / "new_run_dir")
    assert issues == []
    assert (tmp_path / "new_run_dir").is_dir()


def test_output_dir_empty_is_error() -> None:
    issues = check_output_dir_writable("")
    assert len(issues) == 1 and issues[0].code == "output_dir_missing"


def test_report_aggregates_and_separates_user_and_technical() -> None:
    flags = PerturbationFlags(enable_sh=True, enable_albedo=True)
    report = backend_preflight(
        requested_backend="gpu_sh",
        requested_sh_degree=100,
        flags=flags,
        output_dir="",
    )
    assert isinstance(report, PreflightReport)
    assert not report.ok
    # degree + force-model + output-dir → three errors
    assert len(report.errors) == 3
    # user messages and technical log are distinct views over the same issues
    assert len(report.user_messages()) == 3
    assert all(" | " in line for line in report.technical_log())


def test_report_ok_when_no_errors(tmp_path) -> None:
    report = backend_preflight(
        requested_backend="cpu_sh",
        requested_sh_degree=200,
        flags=PerturbationFlags(enable_sh=True, enable_thermal=True),
        output_dir=tmp_path / "run",
    )
    assert report.ok
    assert report.errors == []
