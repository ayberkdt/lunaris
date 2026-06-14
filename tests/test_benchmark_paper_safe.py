"""Task 4 — paper-safe benchmark mode.

Paper-safe mode must hard-fail on synthetic/quick/legacy/mismatch/extrapolation
settings *before* producing scientific-looking output, and cannot be bypassed by
the permissive ``allow_*`` flags. It must also write the provenance files.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lunaris.surrogate.st_lrps.evaluation.benchmark_config import (
    SYNTHETIC_BANNER,
    BenchmarkConfigError,
    apply_paper_safe,
)

# ---------------------------------------------------------------------------
# Pure enforcement function (no torch needed)
# ---------------------------------------------------------------------------

def _minimal_config(**over):
    cfg = {
        "surrogate": {"enabled": True, "model_dir": "runs/some_model"},
        "run_options": {},
        "validation": {},
    }
    cfg.update(over)
    return cfg


def test_apply_paper_safe_enforces_strict_flags():
    cfg = _minimal_config()
    enforced = apply_paper_safe(cfg)
    assert enforced["synthetic"] is False
    assert enforced["allow_contract_mismatch"] is False
    assert enforced["allow_domain_extrapolation"] is False
    assert enforced["allow_legacy_artifact"] is False
    assert enforced["allow_validation_fail"] is False
    assert enforced["strict_domain"] is True
    assert cfg["paper_safe"] is True
    assert cfg["validation"]["strict_domain"] is True
    assert cfg["run_options"]["synthetic"] is False


def test_apply_paper_safe_rejects_synthetic():
    cfg = _minimal_config(run_options={"synthetic": True})
    with pytest.raises(BenchmarkConfigError, match="synthetic"):
        apply_paper_safe(cfg)


def test_apply_paper_safe_rejects_quick():
    cfg = _minimal_config(run_options={"quick": True})
    with pytest.raises(BenchmarkConfigError, match="quick"):
        apply_paper_safe(cfg)


def test_apply_paper_safe_rejects_truth_baseline_without_justification():
    cfg = _minimal_config(validation={"allow_truth_baseline": True})
    with pytest.raises(BenchmarkConfigError, match="truth"):
        apply_paper_safe(cfg)


def test_apply_paper_safe_allows_justified_truth_baseline():
    cfg = _minimal_config(
        validation={"allow_truth_baseline": True, "truth_baseline_justification": "reference sanity check"}
    )
    enforced = apply_paper_safe(cfg)
    assert enforced["allow_truth_baseline"] is True
    assert enforced["truth_baseline_justification"] == "reference sanity check"


def test_apply_paper_safe_requires_surrogate_model_dir():
    cfg = _minimal_config(surrogate={"enabled": True, "model_dir": None})
    with pytest.raises(BenchmarkConfigError, match="model_dir"):
        apply_paper_safe(cfg)


# ---------------------------------------------------------------------------
# End-to-end guard via the pipeline (requires torch for the fixture artifact)
# ---------------------------------------------------------------------------

torch = pytest.importorskip("torch")

from lunaris.surrogate.st_lrps.evaluation.benchmark_pipeline import run_configured_benchmark
from st_lrps_contract_test_utils import make_contract_run


def _write_config(
    path: Path,
    *,
    baseline_degree: int = 20,
    truth_degree: int = 60,
    alt_min_km: float = 100.0,
    alt_max_km: float = 1000.0,
    synthetic: bool = False,
) -> Path:
    payload = {
        "schema_version": 1,
        "name": "paper_safe_test",
        "description": "Paper-safe enforcement test",
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
        "validation": {},
        "run_options": {"synthetic": bool(synthetic)},
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def test_paper_safe_rejects_synthetic_config(tmp_path):
    run = make_contract_run(tmp_path, degree_min=20, degree_max=60)
    config = _write_config(tmp_path / "bench.json", synthetic=True)
    with pytest.raises(BenchmarkConfigError, match="synthetic"):
        run_configured_benchmark(config, out_dir=tmp_path / "out", model_dir=run["run_dir"], paper_safe=True)


def test_paper_safe_overrides_allow_contract_mismatch(tmp_path):
    run = make_contract_run(tmp_path, degree_min=20, degree_max=60)
    config = _write_config(tmp_path / "bench.json", baseline_degree=30)
    # Even with allow_contract_mismatch=True, paper-safe forces strict and rejects.
    with pytest.raises(Exception, match="degree"):
        run_configured_benchmark(
            config,
            out_dir=tmp_path / "out",
            model_dir=run["run_dir"],
            paper_safe=True,
            allow_contract_mismatch=True,
        )


def test_paper_safe_overrides_allow_domain_extrapolation(tmp_path):
    run = make_contract_run(tmp_path, degree_min=20, degree_max=60, alt_min_km=100.0, alt_max_km=300.0)
    config = _write_config(tmp_path / "bench.json", alt_min_km=100.0, alt_max_km=600.0)
    with pytest.raises(Exception, match="altitude"):
        run_configured_benchmark(
            config,
            out_dir=tmp_path / "out",
            model_dir=run["run_dir"],
            paper_safe=True,
            allow_domain_extrapolation=True,
        )


def test_paper_safe_overrides_allow_legacy_artifact(tmp_path):
    run = make_contract_run(tmp_path, degree_min=20, degree_max=60, include_contract=False)
    config = _write_config(tmp_path / "bench.json")
    with pytest.raises(Exception):  # noqa: B017  # legacy artifact rejected with various types
        run_configured_benchmark(
            config,
            out_dir=tmp_path / "out",
            model_dir=run["run_dir"],
            paper_safe=True,
            allow_legacy_artifact=True,
        )


def test_paper_safe_via_config_field(tmp_path):
    run = make_contract_run(tmp_path, degree_min=20, degree_max=60)
    config_path = tmp_path / "bench.json"
    _write_config(config_path, synthetic=True)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["paper_safe"] = True  # request via config, not the flag
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(BenchmarkConfigError, match="synthetic"):
        run_configured_benchmark(config_path, out_dir=tmp_path / "out", model_dir=run["run_dir"])


def test_provenance_and_synthetic_banner_written(tmp_path):
    # A normal (non-paper-safe) synthetic run still writes run_command.txt, records
    # the paper_safe block as disabled, and stamps the synthetic banner.
    run = make_contract_run(tmp_path, degree_min=20, degree_max=60)
    config = _write_config(tmp_path / "bench.json", synthetic=True)
    out = tmp_path / "out"
    rc = run_configured_benchmark(config, out_dir=out, model_dir=run["run_dir"])
    assert rc == 0
    assert (out / "run_command.txt").exists()
    assert (out / "resolved_config.json").exists()
    assert (out / "validation_report.json").exists()
    manifest = json.loads((out / "benchmark_manifest.json").read_text(encoding="utf-8"))
    assert manifest["paper_safe"]["enabled"] is False
    metrics = json.loads((out / "metrics_summary.json").read_text(encoding="utf-8"))
    assert metrics["warning"] == SYNTHETIC_BANNER
    assert SYNTHETIC_BANNER in (out / "report.md").read_text(encoding="utf-8")
