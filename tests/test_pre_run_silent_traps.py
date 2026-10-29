"""Regression tests for traps that silently produced wrong or overwritten results.

Each test pins one failure found in the 2026-10-28 pre-run audit:

* benchmark trajectory caches are keyed by scenario_id only, so another scenario
  design in the same cache directory reused the old trajectories;
* ``--scenario-limit`` was ignored when an existing manifest was extended;
* a failed benchmark series exited 0 without ``--require-st-lrps``;
* the benchmark picked the newest model directory even while it was training;
* evaluating a read-only (vaulted) model aborted on writing its run manifest;
* ``lunaris-batch`` ran ``--degree 200`` at the default ``--sh-degree`` 10 on the
  GPU and ignored ``--out-dir``, replacing the shared default archive;
* ``lunaris-frozen-search`` accepted different third-body sets for screening and
  validation;
* the family report raised on metrics that a JSON round trip had made None.
"""
from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("torch")

from lunaris.surrogate.st_lrps.evaluation import compare_gravity_models as cgm  # noqa: E402
from lunaris.surrogate.st_lrps.evaluation._gravity_benchmark import modes  # noqa: E402
from lunaris.surrogate.st_lrps.evaluation._gravity_benchmark import results_io as rio  # noqa: E402

_BASE = [
    "--gpu-batch-compare", "--truth", "sh25", "--scenario-seed", "42",
    "--sampling-method", "sobol_scrambled", "--scenario-mode", "bounded_keplerian",
    "--duration-days", "5", "--dt-out", "60", "--rk4-dt-s", "15", "--torch-dtype", "float64",
    "--cache-truth", "--cache-trajectories", "--gpu-models", "sh200",
]


def _bench_args(*extra: str):
    return cgm._apply_common_aliases(cgm.parse_args([*_BASE, *extra]))


def _fill_cache(args, scenarios, cache_dir: Path) -> None:
    rio._write_cache_manifest(args, cache_dir, scenarios, ["sh200"])
    for s in scenarios:
        rio._save_cached_trajectory(
            cache_dir, s, "sh200", "comparison_model",
            np.array([0.0, 60.0]), np.vstack([s.initial_state] * 2), args,
        )


# --------------------------------------------------------------------------- benchmark


@pytest.mark.parametrize("extra", [
    ("--scenario-seed", "43"),
    ("--altitude-min-km", "20", "--altitude-max-km", "100"),
    ("--scenario-mode", "near_circular_altitude"),
])
def test_cache_written_for_another_scenario_design_is_refused(tmp_path, extra) -> None:
    first = _bench_args("--random-scenarios", "8")
    scenarios = rio.prepare_scenarios(first, tmp_path)
    cache_dir = tmp_path / "benchmark_cache"
    _fill_cache(first, scenarios, cache_dir)
    assert rio._model_cache_completion(cache_dir, "sh200", scenarios)[0] == 8

    manifest = tmp_path / rio.SCENARIO_MANIFEST_JSON
    before = manifest.read_bytes()
    other = _bench_args("--random-scenarios", "8", *extra)
    # Refused before the scenario manifest is rewritten for the new design...
    with pytest.raises(rio.StaleScenarioCacheError):
        rio.prepare_scenarios(other, tmp_path)
    assert manifest.read_bytes() == before
    # ...and at load time when the cache is shared from another output directory.
    other_scenarios = rio.prepare_scenarios(other, tmp_path / "elsewhere")
    assert not np.array_equal(scenarios[0].initial_state, other_scenarios[0].initial_state)
    with pytest.raises(rio.StaleScenarioCacheError):
        rio._model_cache_completion(cache_dir, "sh200", other_scenarios)


def test_cache_for_the_same_design_is_still_reused(tmp_path) -> None:
    args = _bench_args("--random-scenarios", "8")
    scenarios = rio.prepare_scenarios(args, tmp_path)
    cache_dir = tmp_path / "benchmark_cache"
    _fill_cache(args, scenarios, cache_dir)
    again = rio.prepare_scenarios(_bench_args("--random-scenarios", "8"), tmp_path)
    assert rio._model_cache_completion(cache_dir, "sh200", again) == (8, [])


def test_scenario_limit_caps_an_extended_manifest(tmp_path) -> None:
    rio.prepare_scenarios(_bench_args("--random-scenarios", "32", "--scenario-limit", "8"), tmp_path)
    extended = rio.prepare_scenarios(
        _bench_args("--random-scenarios", "32", "--scenario-limit", "16", "--reuse-cache"), tmp_path
    )
    assert len(extended) == 16
    full = rio.prepare_scenarios(_bench_args("--random-scenarios", "32"), tmp_path / "fresh")
    assert all(np.array_equal(a.initial_state, b.initial_state) for a, b in zip(extended, full[:16], strict=True))


def test_any_failed_series_makes_the_run_fail() -> None:
    ok = SimpleNamespace(status="ok", display_name="SH200", failure_reason="")
    bad = SimpleNamespace(status="failed", display_name="ST-LRPS", failure_reason="non-finite state")
    modes._raise_if_any_model_failed([ok])
    with pytest.raises(RuntimeError, match="1 of 2 benchmark series failed"):
        modes._raise_if_any_model_failed([ok, bad])


def _runs_with_status(tmp_path: Path, *statuses: str | None) -> list[Path]:
    runs = []
    for k, status in enumerate(statuses):
        run = tmp_path / f"run{k}"
        run.mkdir()
        if status is not None:
            (run / "run_manifest.json").write_text(json.dumps({"status": status}), encoding="utf-8")
        runs.append(run)
    return runs


@pytest.mark.parametrize("newest", ["running", "interrupted", None])
def test_benchmark_skips_an_unfinished_newest_model(tmp_path, monkeypatch, newest) -> None:
    """The newest run by mtime may still be training; never benchmark it implicitly."""
    runs = _runs_with_status(tmp_path, newest, "completed")
    monkeypatch.setattr(modes, "discover_st_lrps_model_dirs", lambda: runs)
    assert modes._auto_find_st_lrps_dir() == str(runs[1])


def test_benchmark_finds_no_model_when_none_finished(tmp_path, monkeypatch) -> None:
    runs = _runs_with_status(tmp_path, "running", "failed")
    monkeypatch.setattr(modes, "discover_st_lrps_model_dirs", lambda: runs)
    assert modes._auto_find_st_lrps_dir() is None


# --------------------------------------------------------------------------- evaluation


@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root ignores file modes")
def test_evaluating_a_read_only_run_leaves_it_untouched(tmp_path) -> None:
    from lunaris.surrogate.st_lrps.artifacts.manager import append_run_evaluation, make_run_layout

    layout = make_run_layout(tmp_path)
    manifest = Path(layout.run_manifest_json)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({"status": "completed", "evaluations": []}), encoding="utf-8")
    before = manifest.read_bytes()
    os.chmod(manifest, stat.S_IREAD)
    try:
        append_run_evaluation(layout, {"split": "test"})
        assert manifest.read_bytes() == before
        assert sorted(p.name for p in manifest.parent.iterdir()) == [manifest.name]
    finally:
        os.chmod(manifest, stat.S_IREAD | stat.S_IWRITE)


# --------------------------------------------------------------------------- lunaris-batch


@pytest.mark.parametrize(("argv", "degree"), [
    ([], 10),
    (["--degree", "200"], 200),
    (["--degree", "200", "--sh-degree", "50"], 50),
])
def test_batch_sh_degree_follows_degree_unless_given(argv, degree) -> None:
    from lunaris.cli import batch_runner

    assert batch_runner._parse_args(argv).sh_degree == degree


@pytest.mark.parametrize(("argv", "path"), [
    ([], "outputs/ensemble/batch_output.h5"),
    (["--out-dir", "runs/a"], str(Path("runs/a") / "batch_output.h5")),
    (["--out-dir", "runs/a", "--batch-output-path", "x.h5"], "x.h5"),
])
def test_batch_archive_path_honours_out_dir(argv, path) -> None:
    from lunaris.cli import batch_runner

    assert batch_runner._parse_args(argv).batch_output_path == path


# --------------------------------------------------------------------------- frozen search


def test_frozen_search_refuses_mismatched_third_body_sets(tmp_path, capsys) -> None:
    from lunaris.cli import frozen_search

    code = frozen_search.main([
        "--out", str(tmp_path / "run"), "--screening-backend", "st-lrps",
        "--st-lrps-model-dir", str(tmp_path / "model"),
        "--screening-third-body", "sun,earth", "--validation-third-body", "none",
    ])
    assert code == 2
    assert "third-body terms differ" in capsys.readouterr().err
    assert not (tmp_path / "run").exists()


def test_family_report_tolerates_metrics_nulled_by_json() -> None:
    """stage3_validation.json stores NaN as null; float(None) used to raise."""
    from lunaris.analysis.frozen.family_report import group_candidates_into_families

    def member(idx: int, e_range: float | None) -> dict:
        return {
            "sample_index": idx,
            "elements": {"a_km": 1880.0, "e": 0.05, "i_deg": 85.0, "argp_deg": 90.0},
            "classification": {"status": "long_lived_not_frozen", "score": 1.0},
            "metrics": {"e_range": e_range, "h_peri_range_m": None, "omega_span_rad": 0.2},
        }

    families = group_candidates_into_families(
        json.loads(json.dumps([member(0, None), member(1, 0.004)])),
        screening_backend="test", gravity_model={}, third_body={}, provenance={},
    )
    metrics = families[0]["stability_metrics"]
    assert metrics["e_range_max"] == 0.004
    assert np.isnan(metrics["h_peri_range_m_max"])


def _model_dir_with_envelope(tmp_path: Path, lo: float = 100.0, hi: float = 1000.0) -> Path:
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps({"altitude_min_km": lo, "altitude_max_km": hi}), encoding="utf-8"
    )
    return model


@pytest.mark.parametrize("window", [("20", "1000"), ("100", "20000")])
def test_frozen_search_extrapolation_needs_a_domain_window_inside_the_envelope(
    tmp_path, capsys, window
) -> None:
    """With --st-lrps-allow-extrapolation, samples outside the trained envelope but
    inside the domain window (CLI defaults 20-20000 km) were scored as valid."""
    from lunaris.cli import frozen_search

    code = frozen_search.main([
        "--out", str(tmp_path / "run"), "--screening-backend", "st-lrps",
        "--st-lrps-model-dir", str(_model_dir_with_envelope(tmp_path)),
        "--st-lrps-allow-extrapolation",
        "--domain-alt-min-km", window[0], "--domain-alt-max-km", window[1],
    ])
    assert code == 2
    assert "trained envelope [100, 1000] km" in capsys.readouterr().err
    assert not (tmp_path / "run").exists()


def test_frozen_search_screens_in_float64_by_default() -> None:
    from lunaris.cli import frozen_search

    assert frozen_search.build_parser().parse_args(["--out", "x"]).screening_torch_dtype == "float64"


# --------------------------------------------------------------------------- second audit


def _tiny_run(tmp_path: Path, file_cfg: dict, ckpt_cfg: dict) -> Path:
    import torch

    from lunaris.common.constants import MU_MOON
    from lunaris.surrogate.runtime.networks import _build_model_from_config

    run = tmp_path / "run"
    (run / "checkpoints").mkdir(parents=True)
    base = {"hidden": 8, "depth": 1, "activation": "tanh", "dropout": 0.0,
            "resolved_mu_si": float(MU_MOON), "resolved_a_sign": 1.0,
            "scaler_kind": "isometric", "degree_min": 0, "degree_max": 50}
    scaler = {"x": {"mean": [0.0, 0.0, 0.0], "scale": 2_000_000.0},
              "u": {"mean": [0.0], "scale": 1.0}, "a": {"mean": [0.0, 0.0, 0.0], "scale": 1.0}}
    model = _build_model_from_config(base)
    (run / "config.json").write_text(json.dumps({**base, **file_cfg}), encoding="utf-8")
    (run / "scaler.json").write_text(json.dumps(scaler), encoding="utf-8")
    torch.save({"model": model.state_dict(), "config": {**base, **ckpt_cfg}, "scaler": scaler},
               run / "checkpoints" / "ckpt_best.pt")
    return run


def test_config_and_checkpoint_disagreeing_on_a_sign_is_refused(tmp_path, monkeypatch) -> None:
    """The canonical runtime reads a_sign from config.json, this provider from the
    checkpoint; a disagreement flipped the residual on one of the two paths."""
    from lunaris.surrogate.runtime import SurrogateGravityModel

    monkeypatch.setenv("LUNARIS_ALLOW_LEGACY_ARTIFACT", "1")
    run = _tiny_run(tmp_path, {"resolved_a_sign": -1.0}, {"resolved_a_sign": 1.0})
    with pytest.raises(ValueError, match="physics key"):
        SurrogateGravityModel.from_model_dir(run, device_preference="cpu")


def test_baseline_gravity_file_must_match_the_dataset_hash(tmp_path) -> None:
    from lunaris.surrogate.runtime.gravity_provider import _check_baseline_file_matches_dataset

    f = tmp_path / "field.tab"
    f.write_bytes(b"coefficients")
    import hashlib

    good = hashlib.sha256(b"coefficients").hexdigest()
    _check_baseline_file_matches_dataset(f, {"dataset_contract": {"source_gravity_file_sha256": good}}, tmp_path)
    _check_baseline_file_matches_dataset(f, {}, tmp_path)  # unrecorded: not checked
    with pytest.raises(ValueError, match="not the gravity file"):
        _check_baseline_file_matches_dataset(
            f, {"dataset_contract": {"source_gravity_file_sha256": "0" * 64}}, tmp_path
        )


def test_benchmark_reports_scenarios_below_the_trained_envelope(tmp_path, capsys) -> None:
    """The runtime warned once and tagged nothing; a 76 km orbit was status ok."""
    model = _model_dir_with_envelope(tmp_path)
    metrics = tmp_path / "metrics"
    metrics.mkdir()
    (metrics / "gpu_batch_per_scenario_metrics.csv").write_text(
        "scenario_id,model,min_alt_model_km,status\n"
        "0,GPU_ST_LRPS_RK4,150.0,ok\n1,GPU_ST_LRPS_RK4,76.2,ok\n2,CPU_SH25_RK4,50.0,ok\n",
        encoding="utf-8",
    )
    (metrics / "gpu_batch_summary.json").write_text("{}", encoding="utf-8")
    out = modes._report_surrogate_envelope_excursions(SimpleNamespace(st_lrps_model_dir=str(model)), metrics)
    assert [e["scenario_id"] for e in out["GPU_ST_LRPS_RK4"]["scenarios"]] == [1]
    summary = json.loads((metrics / "gpu_batch_summary.json").read_text(encoding="utf-8"))
    assert summary["surrogate_envelope_excursions"]["GPU_ST_LRPS_RK4"]["envelope_km"] == [100.0, 1000.0]
    assert "lowest 76.2 km" in capsys.readouterr().out
