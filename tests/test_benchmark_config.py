from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip('torch')

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


def test_cpu_adaptive_defaults_false_and_maps_to_argv(tmp_path):
    loaded = load_benchmark_config(_write(tmp_path, _config()))
    assert loaded["surrogate"]["cpu_adaptive"] is False

    from lunaris.surrogate.st_lrps.evaluation.benchmark_pipeline import config_to_legacy_argv

    argv = config_to_legacy_argv(loaded, tmp_path / "out")
    assert "--cpu-adaptive-surrogate" not in argv

    payload = _config()
    payload["surrogate"]["cpu_adaptive"] = True
    payload["surrogate"]["cpu_adaptive_rtol"] = 1.0e-7
    loaded = load_benchmark_config(_write(tmp_path, payload))
    argv = config_to_legacy_argv(loaded, tmp_path / "out")
    assert "--cpu-adaptive-surrogate" in argv

    # The legacy harness must actually accept the flags.
    from lunaris.surrogate.st_lrps.evaluation import compare_gravity_models as cgm

    args = cgm.parse_args(argv)
    assert args.cpu_adaptive_surrogate is True
    assert args.cpu_adaptive_rtol == pytest.approx(1.0e-7)
    assert args.cpu_adaptive_atol == pytest.approx(1.0e-6)  # default preserved


def test_require_st_lrps_defaults_false_and_maps_to_argv(tmp_path):
    loaded = load_benchmark_config(_write(tmp_path, _config()))
    assert loaded["surrogate"]["require_st_lrps"] is False

    from lunaris.surrogate.st_lrps.evaluation.benchmark_pipeline import config_to_legacy_argv

    argv = config_to_legacy_argv(loaded, tmp_path / "out")
    assert "--require-st-lrps" not in argv

    payload = _config()
    payload["surrogate"]["require_st_lrps"] = True
    loaded = load_benchmark_config(_write(tmp_path, payload))
    argv = config_to_legacy_argv(loaded, tmp_path / "out")
    assert "--require-st-lrps" in argv

    from lunaris.surrogate.st_lrps.evaluation import compare_gravity_models as cgm

    args = cgm.parse_args(argv)
    assert args.require_st_lrps is True


def test_require_st_lrps_requires_enabled_surrogate_and_bool(tmp_path):
    payload = _config()
    payload["surrogate"]["require_st_lrps"] = "yes"
    with pytest.raises(BenchmarkConfigError, match="require_st_lrps"):
        load_benchmark_config(_write(tmp_path, payload))

    payload = _config()
    payload["surrogate"]["enabled"] = False
    payload["surrogate"]["require_st_lrps"] = True
    with pytest.raises(BenchmarkConfigError, match="require_st_lrps"):
        load_benchmark_config(_write(tmp_path, payload))


def test_fit_region_sweep_config_is_fail_closed_on_st_lrps():
    """The shipped fit-region sweep exists to characterize ST-LRPS; it must
    refuse to run (rather than silently degrade to an SH-only ladder) when no
    valid surrogate model directory can be resolved."""
    config_path = (
        Path(__file__).resolve().parents[1]
        / "configs" / "benchmarks" / "st_lrps_fit_region_sweep.json"
    )
    loaded = load_benchmark_config(config_path)
    assert loaded["surrogate"]["enabled"] is True
    assert loaded["surrogate"]["require_st_lrps"] is True

    from lunaris.surrogate.st_lrps.evaluation.benchmark_pipeline import config_to_legacy_argv

    argv = config_to_legacy_argv(loaded, Path("out"))
    assert "--require-st-lrps" in argv


def test_cpu_adaptive_requires_enabled_surrogate_and_bool(tmp_path):
    payload = _config()
    payload["surrogate"]["cpu_adaptive"] = "yes"
    with pytest.raises(BenchmarkConfigError, match="cpu_adaptive"):
        load_benchmark_config(_write(tmp_path, payload))

    payload = _config()
    payload["surrogate"]["enabled"] = False
    payload["surrogate"]["cpu_adaptive"] = True
    with pytest.raises(BenchmarkConfigError, match="cpu_adaptive"):
        load_benchmark_config(_write(tmp_path, payload))


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
