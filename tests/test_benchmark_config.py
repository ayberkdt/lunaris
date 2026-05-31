from __future__ import annotations

import json
from pathlib import Path

import pytest

from lunaris.surrogate.st_lrps.evaluation.benchmark_config import (
    BenchmarkConfigError,
    load_benchmark_config,
)


def _config() -> dict:
    return {
        "schema_version": 1,
        "name": "fixture_benchmark",
        "description": "tiny fixture",
        "scenario": {
            "seed": 42,
            "count": 5,
            "type": "bounded_keplerian",
            "altitude_min_km": 100.0,
            "altitude_max_km": 200.0,
        },
        "propagation": {
            "duration_days": 0.1,
            "output_dt_s": 60.0,
            "integrator": "RK4",
            "dt_s": 30.0,
            "dtype": "float64",
        },
        "truth": {
            "model": "spherical_harmonics",
            "degree": 20,
            "integrator": "DOP853",
            "rtol": 1.0e-10,
            "atol": 1.0e-12,
        },
        "baselines": [
            {"name": "SH20", "model": "spherical_harmonics", "degree": 20, "allow_truth_duplicate": True}
        ],
        "surrogate": {
            "enabled": True,
            "name": "ST-LRPS",
            "model_dir": None,
            "baseline_degree": 20,
        },
        "outputs": {
            "out_dir": "benchmark_out",
            "write_figures": True,
            "write_csv": True,
            "write_json": True,
        },
    }


def _write(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "benchmark.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_valid_config_loads(tmp_path):
    path = _write(tmp_path, _config())
    loaded = load_benchmark_config(path)
    assert loaded["name"] == "fixture_benchmark"
    assert loaded["scenario"]["seed"] == 42


def test_missing_name_fails(tmp_path):
    payload = _config()
    payload.pop("name")
    with pytest.raises(BenchmarkConfigError, match="name"):
        load_benchmark_config(_write(tmp_path, payload))


def test_missing_scenario_seed_fails(tmp_path):
    payload = _config()
    payload["scenario"].pop("seed")
    with pytest.raises(BenchmarkConfigError, match="scenario.seed"):
        load_benchmark_config(_write(tmp_path, payload))


def test_invalid_scenario_count_fails(tmp_path):
    payload = _config()
    payload["scenario"]["count"] = 0
    with pytest.raises(BenchmarkConfigError, match="scenario.count"):
        load_benchmark_config(_write(tmp_path, payload))


def test_invalid_duration_fails(tmp_path):
    payload = _config()
    payload["propagation"]["duration_days"] = -1.0
    with pytest.raises(BenchmarkConfigError, match="duration_days"):
        load_benchmark_config(_write(tmp_path, payload))


def test_invalid_dtype_fails(tmp_path):
    payload = _config()
    payload["propagation"]["dtype"] = "float16"
    with pytest.raises(BenchmarkConfigError, match="dtype"):
        load_benchmark_config(_write(tmp_path, payload))


def test_unsupported_truth_model_fails(tmp_path):
    payload = _config()
    payload["truth"]["model"] = "point_mass"
    with pytest.raises(BenchmarkConfigError, match="truth.model"):
        load_benchmark_config(_write(tmp_path, payload))


def test_output_directory_is_resolved_relative_to_config(tmp_path):
    path = _write(tmp_path, _config())
    loaded = load_benchmark_config(path)
    assert loaded["outputs"]["out_dir"] == str((tmp_path / "benchmark_out").resolve())


def test_cli_override_changes_only_intended_fields(tmp_path):
    path = _write(tmp_path, _config())
    loaded = load_benchmark_config(
        path,
        {
            "out_dir": tmp_path / "override_out",
            "model_dir": tmp_path / "model",
            "scenario_count": 2,
            "seed": 7,
            "dtype": "float32",
        },
    )
    assert loaded["name"] == "fixture_benchmark"
    assert loaded["scenario"]["count"] == 2
    assert loaded["scenario"]["seed"] == 7
    assert loaded["propagation"]["dtype"] == "float32"
    assert loaded["surrogate"]["model_dir"] == str((tmp_path / "model").resolve())
    assert loaded["outputs"]["out_dir"] == str((tmp_path / "override_out").resolve())
