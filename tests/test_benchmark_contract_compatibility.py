from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from lunaris.surrogate.st_lrps.evaluation.benchmark_pipeline import run_configured_benchmark
from lunaris.surrogate.st_lrps.shared.contracts import ArtifactContractError
from st_lrps_contract_test_utils import make_contract_run


pytestmark = pytest.mark.requires_torch


def _write_benchmark_config(
    path: Path,
    *,
    baseline_degree: int = 20,
    truth_degree: int = 60,
    alt_min_km: float = 100.0,
    alt_max_km: float = 1000.0,
) -> Path:
    payload = {
        "schema_version": 1,
        "name": "contract_compat",
        "description": "Contract compatibility smoke benchmark",
        "scenario": {
            "seed": 7,
            "count": 2,
            "type": "bounded_keplerian",
            "altitude_min_km": float(alt_min_km),
            "altitude_max_km": float(alt_max_km),
            "eccentricity_mode": "circular_to_elliptic",
        },
        "propagation": {
            "duration_days": 0.01,
            "output_dt_s": 60.0,
            "integrator": "RK4",
            "dt_s": 30.0,
            "dtype": "float64",
        },
        "truth": {
            "model": "spherical_harmonics",
            "degree": int(truth_degree),
            "integrator": "DOP853",
            "rtol": 1.0e-10,
            "atol": 1.0e-12,
            "gravity_file": None,
        },
        "baselines": [{"name": f"SH{baseline_degree}", "model": "spherical_harmonics", "degree": int(baseline_degree)}],
        "surrogate": {
            "enabled": True,
            "name": "ST-LRPS",
            "model_dir": None,
            "baseline_degree": int(baseline_degree),
            "runtime_model_kind": "potential_autograd",
        },
        "outputs": {"out_dir": None, "write_figures": True, "write_csv": True, "write_json": True},
        "validation": {"allow_truth_baseline": True},
        "run_options": {"synthetic": True},
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def test_benchmark_manifest_records_compatible_artifact_contract(tmp_path):
    run = make_contract_run(tmp_path, degree_min=20, degree_max=60)
    config = _write_benchmark_config(tmp_path / "benchmark.json")
    out = tmp_path / "benchmark_out"

    rc = run_configured_benchmark(config, out_dir=out, model_dir=run["run_dir"])

    assert rc == 0
    manifest = json.loads((out / "benchmark_manifest.json").read_text(encoding="utf-8"))
    report = manifest["contract_compatibility"]
    assert report["checked"] is True
    assert report["compatible"] is True
    assert report["errors"] == []
    validation = json.loads((out / "validation_report.json").read_text(encoding="utf-8"))
    assert validation["passed"] is True
    assert "artifact_contract_compatibility" in validation["checked_metrics"]


def test_benchmark_rejects_baseline_degree_mismatch_by_default(tmp_path):
    run = make_contract_run(tmp_path, degree_min=20, degree_max=60)
    config = _write_benchmark_config(tmp_path / "benchmark.json", baseline_degree=30)

    with pytest.raises(ArtifactContractError, match="degree"):
        run_configured_benchmark(config, out_dir=tmp_path / "out", model_dir=run["run_dir"])


def test_benchmark_can_downgrade_explicitly_allowed_contract_mismatch(tmp_path):
    run = make_contract_run(tmp_path, degree_min=20, degree_max=60)
    config = _write_benchmark_config(tmp_path / "benchmark.json", baseline_degree=30)
    out = tmp_path / "allowed_out"

    rc = run_configured_benchmark(
        config,
        out_dir=out,
        model_dir=run["run_dir"],
        allow_contract_mismatch=True,
    )

    assert rc == 0
    validation = json.loads((out / "validation_report.json").read_text(encoding="utf-8"))
    assert validation["passed"] is True
    assert any("contract mismatch allowed explicitly" in warning for warning in validation["warnings"])


def test_benchmark_domain_extrapolation_requires_flag(tmp_path):
    run = make_contract_run(tmp_path, degree_min=20, degree_max=60, alt_min_km=100.0, alt_max_km=300.0)
    config = _write_benchmark_config(tmp_path / "benchmark.json", alt_min_km=100.0, alt_max_km=600.0)

    with pytest.raises(ArtifactContractError, match="altitude envelope"):
        run_configured_benchmark(config, out_dir=tmp_path / "domain_fail", model_dir=run["run_dir"])

    rc = run_configured_benchmark(
        config,
        out_dir=tmp_path / "domain_allowed",
        model_dir=run["run_dir"],
        allow_domain_extrapolation=True,
    )
    assert rc == 0
