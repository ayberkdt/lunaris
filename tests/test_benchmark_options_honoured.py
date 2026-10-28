"""Regression tests for benchmark options that were accepted but silently not honoured.

Each test pins one failure found in the 2026-10-28 pre-run audit of the gravity
benchmark harness:

1. a duration that is not a multiple of ``--dt-out`` made every GPU-batch row fail
   time coverage while the summary said completed and the exit code was 0
   (``--quick`` produced exactly such a duration);
2. ``--rk4-dt-s`` was silently rounded to a divisor of ``--dt-out`` while the
   metadata recorded the requested value;
3. the truth cache ignored the truth integrator and was always named sh200_dop853;
4. the trajectory-cache fingerprint ignored rtol/atol/max-step/integrator/gravity file;
5. ``--config`` silently dropped every legacy flag given next to it;
6. the config pipeline collapsed every propagation integrator to ``medium`` and
   never used ``truth.gravity_file``;
7. ``--ecc-min/--ecc-max`` were ignored under ``bounded_keplerian``.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

from lunaris.common.provenance import sha256_file, sha256_text  # noqa: E402
from lunaris.surrogate.st_lrps.evaluation import benchmark_pipeline as bp  # noqa: E402
from lunaris.surrogate.st_lrps.evaluation import compare_gravity_models as cgm  # noqa: E402
from lunaris.surrogate.st_lrps.evaluation._gravity_benchmark import compute, modes  # noqa: E402
from lunaris.surrogate.st_lrps.evaluation._gravity_benchmark import results_io as rio  # noqa: E402
from lunaris.surrogate.st_lrps.evaluation._gravity_benchmark.types import (
    TruthTrajectorySet,  # noqa: E402
)

DAY_S = 86400.0

_BASE = [
    "--gpu-batch-compare", "--truth", "sh25", "--scenario-seed", "42",
    "--sampling-method", "sobol_scrambled", "--scenario-mode", "bounded_keplerian",
    "--duration-days", "5", "--dt-out", "60", "--rk4-dt-s", "15", "--torch-dtype", "float64",
    "--cache-truth", "--cache-trajectories", "--gpu-models", "sh200",
]

# The fingerprint field list before the 2026-10-28 audit, kept here as an
# independent oracle for "a manifest written by the previous version".
_LEGACY_FINGERPRINT_FIELDS = (
    "cache_schema_version", "truth", "truth_integrator", "duration_days", "dt_out",
    "rk4_dt_s", "gpu_rk4_dt_s_list", "st_lrps_rk4_dt", "gpu_integrator", "torch_dtype",
    "batch_frame_mode", "frame_interpolation", "st_lrps_weight_sha256",
)
_NEW_CACHE_KEYS = ("rtol", "atol", "max_step_s", "integrator", "gravity_file", "gravity_file_sha256")


def _args(*extra: str, base: list[str] | None = None):
    return cgm._apply_common_aliases(cgm.parse_args([*(_BASE if base is None else base), *extra]))


@pytest.fixture(autouse=True)
def _fixed_gravity_identity(monkeypatch):
    """Avoid hashing the 200 MB gravity file in every test."""
    monkeypatch.setattr(
        rio, "_harness_gravity_identity",
        lambda: {"gravity_file": "fake_gravity.tab", "gravity_file_sha256": "a" * 64},
        raising=False,
    )


# --------------------------------------------------------------- items 1 and 2


def test_duration_not_a_multiple_of_dt_out_is_refused_before_compute(tmp_path, monkeypatch) -> None:
    class _ComputeReached(Exception):
        pass

    def _boom(*_a, **_k):
        raise _ComputeReached

    monkeypatch.setattr(modes, "build_truth_trajectory_set", _boom)
    # 0.01 d = 864 s = 14.4 x 60 s.
    args = _args("--duration-days", "0.01", "--output-dir", str(tmp_path), "--gpu-fallback", "cpu")
    with pytest.raises(ValueError, match="multiple of --dt-out"):
        modes.run_gpu_batch_compare_mode(args, None, None)


def test_propagate_refuses_duration_that_is_not_a_multiple_of_dt_out() -> None:
    with pytest.raises(ValueError, match="multiple of"):
        compute.propagate_gpu_batch_model(
            "sh20", None, np.zeros((1, 6)), duration_s=100.0, rk4_dt_s=10.0, output_dt_s=60.0,
            ephem=None, device="cpu", dtype=None, dtype_name="float64", frame_mode="identity_diagnostic",
        )


def test_quick_mode_duration_is_a_whole_number_of_dt_out_steps() -> None:
    args = _args("--quick")
    n = args.duration_days * DAY_S / args.dt_out
    assert n == pytest.approx(round(n), abs=1e-9 * max(1.0, n))
    assert 0 < args.duration_days <= 0.01


@pytest.mark.parametrize("duration_days", ["1", "5", "0.1"])
def test_whole_multiple_durations_are_accepted(duration_days) -> None:
    compute._validate_gpu_batch_timing(_args("--duration-days", duration_days))


def test_rk4_step_that_does_not_divide_dt_out_is_refused() -> None:
    with pytest.raises(ValueError, match="rk4"):
        compute._validate_gpu_batch_timing(_args("--rk4-dt-s", "45"))


def test_rk4_step_list_with_a_non_divisor_is_refused() -> None:
    # 45 s used to be rounded to 60 s, giving two identical series labelled 45 and 60.
    with pytest.raises(ValueError, match="45"):
        compute._validate_gpu_batch_timing(_args("--gpu-rk4-dt-s-list", "45,60"))


def test_default_st_lrps_rk4_step_is_validated_too() -> None:
    base = [a for a in _BASE if a not in ("--rk4-dt-s", "15")]
    with pytest.raises(ValueError, match="rk4"):
        # --st-lrps-rk4-dt defaults to 30 s, which does not divide 45 s.
        compute._validate_gpu_batch_timing(_args("--dt-out", "45", base=base))


def _point_mass_run(monkeypatch, **kw):
    import torch

    from lunaris.common.constants import MU_MOON

    def _fake_accel(model_name, gravity_model, *, device, dtype):
        mu = torch.tensor(float(MU_MOON), device=device, dtype=dtype)

        def accel(pos_fixed):
            r = torch.linalg.norm(pos_fixed, dim=1, keepdim=True).clamp_min(1.0)
            return -mu * pos_fixed / (r ** 3)

        return accel, "fake_pointmass"

    monkeypatch.setattr(compute, "_make_gpu_accelerator", _fake_accel)
    y0 = np.array([[1.90e6, 0.0, 0.0, 0.0, 1.60e3, 0.0]], dtype=np.float64)
    params = {"duration_s": 300.0, "rk4_dt_s": 10.0, "output_dt_s": 60.0, **kw}
    return compute.propagate_gpu_batch_model(
        "sh20", None, y0, ephem=None, device=torch.device("cpu"), dtype=torch.float64,
        dtype_name="float64", frame_mode="identity_diagnostic", **params,
    )


@pytest.mark.parametrize("rk4_dt_s", [7.0, 45.0, 120.0])
def test_propagate_refuses_rk4_step_that_does_not_divide_dt_out(monkeypatch, rk4_dt_s) -> None:
    with pytest.raises(ValueError, match="rk4_dt_s"):
        _point_mass_run(monkeypatch, rk4_dt_s=rk4_dt_s)


def test_recorded_rk4_step_equals_the_step_used(monkeypatch) -> None:
    result = _point_mass_run(monkeypatch, rk4_dt_s=10.0)
    assert result.status == "ok"
    assert result.rk4_dt_s == 10.0
    assert result.n_steps == 30  # 300 s / 10 s
    assert math.isclose(result.t[-1], 300.0)


# --------------------------------------------------------------------- item 3


def _truth_set(scenarios, name: str) -> TruthTrajectorySet:
    t = np.array([0.0, 60.0])
    y = np.zeros((2, 6))
    return TruthTrajectorySet(
        name,
        {s.scenario_id: t for s in scenarios},
        {s.scenario_id: y for s in scenarios},
        {s.scenario_id: 1.0 for s in scenarios},
    )


def _truth_args(*extra: str):
    return _args("--random-scenarios", "3", *extra)


def test_truth_cache_file_is_named_after_the_actual_truth(tmp_path) -> None:
    args = _truth_args()
    scenarios = rio.prepare_scenarios(args, tmp_path)
    rio._save_truth_cache(tmp_path / "truth", args, scenarios, _truth_set(scenarios, "sh25_dop853"))
    files = sorted(p.name for p in (tmp_path / "truth").glob("*.npz"))
    assert files == ["sh25_dop853_trajectories.npz"]
    loaded = rio._load_truth_cache(tmp_path / "truth", args, scenarios)
    assert loaded is not None and loaded.model_name == "sh25_dop853"


@pytest.mark.parametrize("extra", [
    ("--truth-integrator", "RK45"),
    ("--rtol", "1e-9"),
    ("--atol", "1e-13"),
    ("--truth", "sh50"),
])
def test_truth_cache_of_another_integrator_or_tolerance_is_not_reused(tmp_path, extra) -> None:
    args = _truth_args()
    scenarios = rio.prepare_scenarios(args, tmp_path)
    rio._save_truth_cache(tmp_path / "truth", args, scenarios, _truth_set(scenarios, "sh25_dop853"))
    other = _truth_args(*extra)
    assert rio._load_truth_cache(tmp_path / "truth", other, scenarios) is None
    assert not rio._truth_cache_available(tmp_path / "truth", other, scenarios)


def _write_legacy_truth_cache(directory: Path, args, scenarios) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    meta = rio._truth_cache_metadata(args, scenarios)
    meta.pop("truth_integrator", None)  # recorded only since the audit
    (directory / "truth_metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    n = len(scenarios)
    np.savez_compressed(
        directory / "sh200_dop853_trajectories.npz",
        t=np.array([0.0, 60.0]), y=np.zeros((n, 2, 6)), runtime=np.ones(n),
    )


def test_legacy_truth_cache_name_is_read_only_when_its_metadata_matches(tmp_path) -> None:
    args = _truth_args("--truth", "sh200")
    scenarios = rio.prepare_scenarios(args, tmp_path)
    _write_legacy_truth_cache(tmp_path / "truth", args, scenarios)
    assert rio._load_truth_cache(tmp_path / "truth", args, scenarios) is not None
    # The legacy file does not record the integrator, so it is only trusted for the default.
    assert rio._load_truth_cache(
        tmp_path / "truth", _truth_args("--truth", "sh200", "--truth-integrator", "RK45"), scenarios
    ) is None
    # ...and not for another scenario design.
    assert rio._load_truth_cache(
        tmp_path / "truth", _truth_args("--truth", "sh200", "--rtol", "1e-9"), scenarios
    ) is None


# --------------------------------------------------------------------- item 4


def _manifest_for(args, tmp_path: Path):
    scenarios = rio.prepare_scenarios(args, tmp_path)
    cache_dir = tmp_path / "benchmark_cache"
    rio._write_cache_manifest(args, cache_dir, scenarios, ["sh200"])
    return cache_dir


@pytest.mark.parametrize("extra", [
    ("--rtol", "1e-9"),
    ("--atol", "1e-13"),
    ("--max-step", "10"),
    ("--integrator", "RK45"),
])
def test_cache_written_with_other_integrator_settings_is_refused(tmp_path, extra) -> None:
    cache_dir = _manifest_for(_args("--random-scenarios", "3"), tmp_path)
    other = _args("--random-scenarios", "3", *extra)
    with pytest.raises(ValueError, match="Refusing to reuse|fingerprint"):
        rio._validate_cache_compatibility(other, cache_dir)
    warnings = rio._validate_cache_compatibility(
        _args("--random-scenarios", "3", *extra, "--allow-stale-cache"), cache_dir
    )
    assert warnings and all(w.startswith("[allow-stale-cache]") for w in warnings)


def test_cache_written_for_another_gravity_file_is_refused(tmp_path, monkeypatch) -> None:
    args = _args("--random-scenarios", "3")
    cache_dir = _manifest_for(args, tmp_path)
    assert rio._validate_cache_compatibility(args, cache_dir) == []
    monkeypatch.setattr(
        rio, "_harness_gravity_identity",
        lambda: {"gravity_file": "other.tab", "gravity_file_sha256": "b" * 64},
    )
    with pytest.raises(ValueError, match="gravity_file"):
        rio._validate_cache_compatibility(args, cache_dir)


def test_gravity_file_identity_is_content_hash(tmp_path) -> None:
    f = tmp_path / "gravity.tab"
    f.write_text("0 1 2 3", encoding="utf-8")
    ident = rio._gravity_file_identity(f)
    assert ident == {"gravity_file": "gravity.tab", "gravity_file_sha256": str(sha256_file(f))}


def _make_legacy_manifest(args, cache_dir: Path) -> None:
    path = cache_dir / "cache_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    for key in _NEW_CACHE_KEYS:
        payload["metadata"].pop(key, None)
    canon = {k: payload["metadata"].get(k) for k in _LEGACY_FINGERPRINT_FIELDS}
    payload["config_fingerprint"] = sha256_text(json.dumps(canon, sort_keys=True, default=str))
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_cache_manifest_without_new_keys_is_accepted_with_one_warning(tmp_path) -> None:
    args = _args("--random-scenarios", "3")
    cache_dir = _manifest_for(args, tmp_path)
    _make_legacy_manifest(args, cache_dir)
    warnings = rio._validate_cache_compatibility(args, cache_dir)
    assert len(warnings) == 1
    for key in _NEW_CACHE_KEYS:
        assert key in warnings[0]
    assert "WARNING" in warnings[0] or "cannot be verified" in warnings[0]


def test_cache_manifest_without_new_keys_still_refuses_old_fields(tmp_path) -> None:
    args = _args("--random-scenarios", "3")
    cache_dir = _manifest_for(args, tmp_path)
    _make_legacy_manifest(args, cache_dir)
    with pytest.raises(ValueError, match="dt_out"):
        rio._validate_cache_compatibility(_args("--random-scenarios", "3", "--dt-out", "30"), cache_dir)


# --------------------------------------------------------------------- item 5


@pytest.fixture
def _config_calls(monkeypatch):
    calls: list[dict] = []

    def _fake(config_path, **kwargs):
        calls.append({"config": config_path, **kwargs})
        return 0

    monkeypatch.setattr(bp, "run_configured_benchmark", _fake)
    return calls


@pytest.mark.parametrize("flags,named", [
    (["--rk4-dt-s", "10"], ["--rk4-dt-s"]),
    (["--torch-dtype", "float64"], ["--torch-dtype"]),
    (["--duration-days", "2", "--gpu-models", "sh20"], ["--duration-days", "--gpu-models"]),
    (["--dur", "2"], ["--duration-days"]),  # argparse prefix abbreviation
    (["--dt-out", "60"], ["--dt-out"]),  # explicitly given, even when equal to the default
    (["--ecc-max", "0.2", "--integrator", "RK45"], ["--ecc-max", "--integrator"]),
    (["--truth", "sh100", "--rtol", "1e-9"], ["--truth", "--rtol"]),
])
def test_config_mode_refuses_legacy_flags(_config_calls, capsys, flags, named) -> None:
    with pytest.raises(SystemExit) as exc:
        cgm.main(["--config", "bench.json", *flags])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    for flag in named:
        assert flag in err
    assert _config_calls == []


def test_config_mode_still_accepts_its_own_overrides(_config_calls, tmp_path) -> None:
    rc = cgm.main(["--config", "bench.json", "--out", str(tmp_path), "--quick", "--seed", "7",
                   "--scenario-count", "3", "--dtype", "float64", "--paper-safe"])
    assert rc == 0
    assert len(_config_calls) == 1
    assert _config_calls[0]["seed"] == 7 and _config_calls[0]["quick"] is True


# --------------------------------------------------------------------- item 6


@pytest.mark.parametrize("name,expected", [
    ("RK4", "medium"), ("rk4", "medium"), ("medium", "medium"),
    ("RK2", "light"), ("light", "light"), ("robust", "robust"),
])
def test_config_integrator_maps_to_the_gpu_tier_it_names(name, expected) -> None:
    assert bp._legacy_gpu_integrator(name) == expected


@pytest.mark.parametrize("name", ["DOP853", "RK45", "Euler", ""])
def test_config_integrator_without_a_gpu_tier_is_refused(name) -> None:
    with pytest.raises(ValueError, match="propagation.integrator"):
        bp._legacy_gpu_integrator(name)


def _gravity_config(gravity_file):
    return {"truth": {"gravity_file": gravity_file}, "run_options": {}}


def test_config_gravity_file_other_than_the_harness_one_is_refused(tmp_path, monkeypatch) -> None:
    harness = tmp_path / "harness.tab"
    harness.write_text("harness field", encoding="utf-8")
    other = tmp_path / "other.tab"
    other.write_text("a different field", encoding="utf-8")
    same_content = tmp_path / "copy.tab"
    same_content.write_text("harness field", encoding="utf-8")
    monkeypatch.setattr(bp, "_harness_gravity_file", lambda: harness, raising=False)

    bp._require_harness_gravity_file(_gravity_config(None))
    bp._require_harness_gravity_file(_gravity_config(str(harness)))
    bp._require_harness_gravity_file(_gravity_config(str(same_content)))
    with pytest.raises(ValueError, match="truth.gravity_file"):
        bp._require_harness_gravity_file(_gravity_config(str(other)))
    with pytest.raises(ValueError, match="does not exist"):
        bp._require_harness_gravity_file(_gravity_config(str(tmp_path / "missing.tab")))


# --------------------------------------------------------------------- item 7


def test_ecc_bounds_are_refused_for_bounded_keplerian(tmp_path) -> None:
    args = _args("--random-scenarios", "4", "--ecc-max", "0.2")
    with pytest.raises(ValueError, match="derived from"):
        rio.prepare_scenarios(args, tmp_path)
    assert not (tmp_path / rio.SCENARIO_MANIFEST_JSON).exists()


def test_ecc_bounds_still_drive_near_circular_scenarios(tmp_path) -> None:
    args = _args("--random-scenarios", "4", "--scenario-mode", "near_circular_altitude",
                 "--ecc-min", "0.0", "--ecc-max", "0.2")
    scenarios = rio.prepare_scenarios(args, tmp_path)
    assert max(s.e for s in scenarios) <= 0.2 and max(s.e for s in scenarios) > 0.0


def test_bounded_keplerian_manifest_records_the_realized_eccentricity_range(tmp_path) -> None:
    args = _args("--random-scenarios", "4")
    scenarios = rio.prepare_scenarios(args, tmp_path)
    manifest = json.loads((tmp_path / rio.SCENARIO_MANIFEST_JSON).read_text(encoding="utf-8"))
    realized = manifest["metadata"]["realized_eccentricity_range"]
    assert realized["min"] == pytest.approx(min(s.e for s in scenarios))
    assert realized["max"] == pytest.approx(max(s.e for s in scenarios))
    assert realized["max"] > 0.0


def test_config_pipeline_does_not_pass_ignored_ecc_bounds_for_bounded_keplerian() -> None:
    config = {
        "scenario": {"count": 4, "seed": 1, "type": "bounded_keplerian", "altitude_min_km": 100.0,
                     "altitude_max_km": 500.0, "eccentricity_mode": "circular_to_elliptic"},
        "propagation": {"duration_days": 1.0, "output_dt_s": 60.0, "integrator": "RK4",
                        "dt_s": 30.0, "dtype": "float64"},
        "truth": {"degree": 200, "integrator": "DOP853"},
        "baselines": [{"name": "SH20", "model": "spherical_harmonics", "degree": 20}],
        "surrogate": {"enabled": False},
    }
    argv = bp.config_to_legacy_argv(config, "out")
    assert "--ecc-max" not in argv and "--ecc-min" not in argv
    assert cgm.parse_args(argv).scenario_mode == "bounded_keplerian"

    config["scenario"]["type"] = "near_circular_altitude"
    argv = bp.config_to_legacy_argv(config, "out")
    assert argv[argv.index("--ecc-max") + 1] == "0.2"
