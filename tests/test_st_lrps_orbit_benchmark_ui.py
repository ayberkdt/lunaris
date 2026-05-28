"""Tests for the Orbit-Level Benchmark Studio page and the relocated harness.

Covers:
- the gravity benchmark harness is importable at its new package path
  (``lunaris.surrogate.st_lrps.evaluation.compare_gravity_models``) and no longer at the old
  ``validation.gravity`` path;
- the Studio benchmark tab builds correct command-line arguments for both run
  modes (per-model DOP853/RK8 and GPU batch RK4);
- the benchmark page is registered in the Studio navigation.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Harness relocation
# ---------------------------------------------------------------------------
def test_harness_lives_in_st_lrps_evaluation():
    mod = importlib.import_module("lunaris.surrogate.st_lrps.evaluation.compare_gravity_models")
    assert hasattr(mod, "main")
    assert hasattr(mod, "parse_args")


def test_harness_file_moved_on_disk():
    assert (
        ROOT / "src" / "lunaris" / "surrogate" / "st_lrps" / "evaluation" / "compare_gravity_models.py"
    ).is_file()
    assert not (ROOT / "validation" / "gravity" / "compare_gravity_models.py").exists()


# ---------------------------------------------------------------------------
# UI command builders (skipped without PyQt6)
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _clear_settings():
    from PyQt6.QtCore import QSettings
    QSettings("ST_LRPS_Project", "ST_LRPS_Dashboard").clear()


@pytest.fixture(scope="module")
def qapp():
    pytest.importorskip("PyQt6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    from lunaris.surrogate.st_lrps.ui.studio_parts.common_widgets import _settings

    settings = _settings()
    settings.beginGroup("orbit_benchmark")
    settings.remove("")
    settings.endGroup()
    settings.sync()
    return app


def test_benchmark_tab_dop853_mode_args(qapp):
    from lunaris.surrogate.st_lrps.ui.studio import BENCHMARK_CLI_MODULE, OrbitBenchmarkTab

    tab = OrbitBenchmarkTab()
    tab.run_mode.setCurrentIndex(tab.run_mode.findData("dop853"))
    args = tab._build_args(show_errors=False)
    assert args is not None
    assert "-m" in args and BENCHMARK_CLI_MODULE in args
    assert "--integrator" in args and args[args.index("--integrator") + 1] == "DOP853"
    assert "--models" in args
    assert "--truth" in args
    assert "--gpu-batch-compare" not in args
    assert "--rtol" in args and "--atol" in args
    assert "--sampling-method" not in args
    assert "--inclination-sampling" not in args
    assert "--cache-trajectories" in args
    assert "--reuse-cache" in args
    tab.deleteLater()


def test_benchmark_tab_gpu_rk4_mode_args(qapp):
    from lunaris.surrogate.st_lrps.ui.studio import OrbitBenchmarkTab

    tab = OrbitBenchmarkTab()
    tab.run_mode.setCurrentIndex(tab.run_mode.findData("gpu_rk4"))
    tab.rk4_dt.setValue(10.0)
    tab.truth_workers.setValue(3)
    args = tab._build_args(show_errors=False)
    assert args is not None
    assert "--gpu-batch-compare" in args
    assert "--gpu-models" in args
    assert "--rk4-dt-s" in args and args[args.index("--rk4-dt-s") + 1] == "10.0"
    assert "--workers" in args and args[args.index("--workers") + 1] == "3"
    assert "--torch-dtype" in args
    assert "--integrator" not in args
    assert "--models" not in args
    assert "--cache-trajectories" in args
    assert "--reuse-cache" in args
    tab.deleteLater()


def test_benchmark_tab_uses_laptop_friendly_defaults(qapp):
    from lunaris.surrogate.st_lrps.ui.studio import OrbitBenchmarkTab

    tab = OrbitBenchmarkTab()
    assert tab.alt_min.value() == 100.0
    assert tab.alt_max.value() == 1000.0
    assert tab.cpu_workers.value() == 4
    assert tab.truth_workers.value() == 4
    assert tab.rk4_dt.value() == 30.0
    assert tab.torch_dtype.currentData() == "float32"
    assert "not recommended on laptops" in tab.torch_dtype.itemText(1)

    args_cpu = tab._build_args(show_errors=False)
    assert args_cpu[args_cpu.index("--workers") + 1] == "4"
    assert args_cpu[args_cpu.index("--altitude-min-km") + 1] == "100.0"
    assert args_cpu[args_cpu.index("--altitude-max-km") + 1] == "1000.0"

    tab.run_mode.setCurrentIndex(tab.run_mode.findData("gpu_rk4"))
    args_gpu = tab._build_args(show_errors=False)
    assert args_gpu[args_gpu.index("--rk4-dt-s") + 1] == "30.0"
    assert args_gpu[args_gpu.index("--workers") + 1] == "4"
    assert args_gpu[args_gpu.index("--torch-dtype") + 1] == "float32"
    tab.deleteLater()


def test_benchmark_tab_requires_at_least_one_model(qapp):
    from lunaris.surrogate.st_lrps.ui.studio import OrbitBenchmarkTab

    tab = OrbitBenchmarkTab()
    for cb in tab._model_checks.values():
        cb.setChecked(False)
    assert tab._build_args(show_errors=False) is None
    tab.deleteLater()


def test_benchmark_tab_st_lrps_dir_emitted_when_set(qapp, tmp_path):
    from lunaris.surrogate.st_lrps.ui.studio import OrbitBenchmarkTab

    tab = OrbitBenchmarkTab()
    tab.run_mode.setCurrentIndex(tab.run_mode.findData("dop853"))
    tab._model_checks["st_lrps"].setChecked(True)
    run_dir = tmp_path / "stlrps_run"
    run_dir.mkdir()
    tab.st_lrps_dir.setText(str(run_dir))
    args = tab._build_args(show_errors=False)
    assert "--st-lrps-model-dir" in args
    assert args[args.index("--st-lrps-model-dir") + 1] == str(run_dir)
    tab.deleteLater()


def test_benchmark_page_registered_in_nav(qapp):
    from lunaris.surrogate.st_lrps.ui.studio_parts.main_window import MainWindow

    w = MainWindow()
    assert w._stack.count() == 7
    assert "Orbit-Level Benchmark" in w._page_titles
    assert "Gravity Plots" in w._page_titles
    w.deleteLater()


# ---------------------------------------------------------------------------
# Harness: GPU fixed-step integrators (backend-agnostic, no CUDA/torch needed)
# ---------------------------------------------------------------------------
def test_gpu_fixed_step_integrators_converge_by_order():
    """light=RK2 (order 2), medium=RK4 (order 4), robust (order >=5).

    Verify on a harmonic oscillator: error shrinks at the expected rate when the
    step is halved, and robust < medium < light at a coarse step.
    """
    import numpy as np

    from lunaris.surrogate.st_lrps.evaluation.compare_gravity_models import (
        GPU_INTEGRATORS,
        gpu_fixed_step_advance,
    )

    assert GPU_INTEGRATORS == ("light", "medium", "robust")

    def rhs(t, s):
        x = s[..., 0]
        v = s[..., 1]
        return np.stack([v, -x], axis=-1)

    def final_err(method, dt, T=10.0):
        s = np.array([1.0, 0.0])
        t = 0.0
        for _ in range(int(round(T / dt))):
            s = gpu_fixed_step_advance(rhs, t, s, dt, method)
            t += dt
        exact = np.array([np.cos(T), -np.sin(T)])
        return float(np.linalg.norm(s - exact))

    # Convergence-rate ratios when halving the step.
    ratios = {}
    for m in GPU_INTEGRATORS:
        e_coarse = final_err(m, 0.1)
        e_fine = final_err(m, 0.05)
        ratios[m] = e_coarse / max(e_fine, 1e-18)
    assert ratios["light"] > 3.0          # ~4  (order 2)
    assert ratios["medium"] > 12.0        # ~16 (order 4)
    assert ratios["robust"] > ratios["medium"]  # higher order than RK4

    # Accuracy ordering at a coarse step.
    coarse = {m: final_err(m, 0.1) for m in GPU_INTEGRATORS}
    assert coarse["robust"] < coarse["medium"] < coarse["light"]


def test_unknown_gpu_method_falls_back_to_rk4():
    import numpy as np

    from lunaris.surrogate.st_lrps.evaluation.compare_gravity_models import gpu_fixed_step_advance

    def rhs(t, s):
        return -s

    s0 = np.array([1.0, 2.0])
    a = gpu_fixed_step_advance(rhs, 0.0, s0, 0.01, "nonsense")
    b = gpu_fixed_step_advance(rhs, 0.0, s0, 0.01, "medium")
    assert np.allclose(a, b)


def test_gpu_batch_propagator_avoids_per_step_host_syncs():
    import inspect

    from lunaris.surrogate.st_lrps.evaluation.compare_gravity_models import (
        TorchFrameProvider,
        propagate_gpu_batch_model,
    )

    source = inspect.getsource(propagate_gpu_batch_model)
    assert "bad_state.logical_or_" in source
    assert "if not torch.isfinite(state).all()" not in source
    assert source.count(".detach().cpu().numpy()") == 1
    assert "y_gpu" in source

    frame_source = inspect.getsource(TorchFrameProvider.quat_i2f)
    assert "if dot <" not in frame_source
    assert "if dot >" not in frame_source


# ---------------------------------------------------------------------------
# Harness: new CLI flags parse
# ---------------------------------------------------------------------------
def test_harness_parses_new_flags(monkeypatch):
    import sys as _sys

    from lunaris.surrogate.st_lrps.evaluation import compare_gravity_models as cgm

    argv = [
        "compare_gravity_models",
        "--gpu-integrator", "robust",
        "--truth-integrator", "RK45",
        "--workers", "4",
        "--sampling-method", "sobol_scrambled",
        "--inclination-sampling", "uniform_cos",
        "--cache-trajectories",
        "--reuse-cache",
        "--append-scenarios", "12",
        "--rebuild-metrics",
        "--strict-complete",
        "--allow-lhs-append",
        "--gpu-rk4-dt-s-list", "10,30",
    ]
    monkeypatch.setattr(_sys, "argv", argv)
    args = cgm.parse_args()
    assert args.gpu_integrator == "robust"
    assert args.truth_integrator == "RK45"
    assert args.workers == 4
    assert args.sampling_method == "sobol_scrambled"
    assert args.inclination_sampling == "uniform_cos"
    assert args.cache_trajectories is True
    assert args.reuse_cache is True
    assert args.append_scenarios == 12
    assert args.rebuild_metrics is True
    assert args.strict_complete is True
    assert args.allow_lhs_append is True
    assert args.gpu_rk4_dt_s_list == "10,30"


def test_harness_laptop_friendly_cli_defaults(monkeypatch):
    import sys as _sys

    from lunaris.surrogate.st_lrps.evaluation import compare_gravity_models as cgm

    monkeypatch.setattr(_sys, "argv", ["compare_gravity_models"])
    args = cgm.parse_args()
    assert args.altitude_min_km == 100.0
    assert args.altitude_max_km == 1000.0
    assert args.workers == 4
    assert args.torch_dtype == "float32"
    assert args.rk4_dt_s is None
    assert args.st_lrps_rk4_dt == 30.0


def test_gpu_rk4_dt_variants_build_distinct_cache_and_display_names():
    from lunaris.surrogate.st_lrps.evaluation import compare_gravity_models as cgm

    args = argparse.Namespace(gpu_rk4_dt_s_list="10,30", rk4_dt_s=None, st_lrps_rk4_dt=30.0)
    tasks = cgm._build_gpu_batch_tasks(["sh20"], args)
    assert [t.cache_name for t in tasks] == ["sh20_rk4_dt10", "sh20_rk4_dt30"]
    assert [t.display_name for t in tasks] == ["GPU_SH20_RK4_DT10", "GPU_SH20_RK4_DT30"]
    assert cgm.display_label("GPU_SH20_RK4_DT10") == "SH20 dt10"


def test_cfg_with_integrator_overrides_method():
    from lunaris.core.config import load_default_config
    from lunaris.surrogate.st_lrps.evaluation.compare_gravity_models import _cfg_with_integrator

    cfg = load_default_config()
    out = _cfg_with_integrator(cfg, "RK45")
    assert out.propagator.method == "RK45"
    # Original config untouched (frozen-dataclass copy semantics).
    assert cfg.propagator.method != "RK45" or out is not cfg


# ---------------------------------------------------------------------------
# UI: new controls / flags
# ---------------------------------------------------------------------------
def test_ui_dop853_mode_emits_truth_integrator_and_workers(qapp):
    from lunaris.surrogate.st_lrps.ui.studio import OrbitBenchmarkTab

    tab = OrbitBenchmarkTab()
    tab.run_mode.setCurrentIndex(tab.run_mode.findData("dop853"))
    tab.cpu_workers.setValue(4)
    tab.truth_integrator.setCurrentIndex(tab.truth_integrator.findData("RK45"))
    args = tab._build_args(show_errors=False)
    assert "--truth-integrator" in args
    assert args[args.index("--truth-integrator") + 1] == "RK45"
    assert "--workers" in args and args[args.index("--workers") + 1] == "4"
    assert "--gpu-integrator" not in args
    tab.deleteLater()


def test_ui_gpu_mode_emits_gpu_integrator(qapp):
    from lunaris.surrogate.st_lrps.ui.studio import OrbitBenchmarkTab

    tab = OrbitBenchmarkTab()
    tab.run_mode.setCurrentIndex(tab.run_mode.findData("gpu_rk4"))
    tab.gpu_integrator.setCurrentIndex(tab.gpu_integrator.findData("robust"))
    tab.rk4_dt_list.setText("10,30")
    args = tab._build_args(show_errors=False)
    assert "--gpu-integrator" in args
    assert args[args.index("--gpu-integrator") + 1] == "robust"
    assert "--gpu-rk4-dt-s-list" in args
    assert args[args.index("--gpu-rk4-dt-s-list") + 1] == "10,30"
    assert "--truth-integrator" in args
    assert "--workers" in args
    tab.deleteLater()


def test_ui_gpu_mode_shows_selectable_gpu_settings(qapp):
    from lunaris.surrogate.st_lrps.ui.studio import OrbitBenchmarkTab

    tab = OrbitBenchmarkTab()
    assert tab.run_mode.currentData() == "dop853"
    assert not tab._grp_cpu_settings.isHidden()
    assert tab._grp_gpu_settings.isHidden()

    tab.run_mode.setCurrentIndex(tab.run_mode.findData("gpu_rk4"))
    assert tab._grp_cpu_settings.isHidden()
    assert not tab._grp_gpu_settings.isHidden()
    assert tab.gpu_integrator.isEnabled()
    assert tab.truth_workers.isEnabled()
    tab.gpu_integrator.setCurrentIndex(tab.gpu_integrator.findData("robust"))
    tab.truth_workers.setValue(5)
    args = tab._build_args(show_errors=False)
    assert args[args.index("--gpu-integrator") + 1] == "robust"
    assert args[args.index("--workers") + 1] == "5"
    tab.deleteLater()


def test_ui_sampling_flags_emitted_when_selected(qapp):
    from lunaris.surrogate.st_lrps.ui.studio import OrbitBenchmarkTab

    tab = OrbitBenchmarkTab()
    tab.sampling_method.setCurrentIndex(tab.sampling_method.findData("sobol_scrambled"))
    tab.inclination_sampling.setCurrentIndex(tab.inclination_sampling.findData("uniform_cos"))
    args = tab._build_args(show_errors=False)
    assert "--sampling-method" in args
    assert args[args.index("--sampling-method") + 1] == "sobol_scrambled"
    assert "--inclination-sampling" in args
    assert args[args.index("--inclination-sampling") + 1] == "uniform_cos"
    tab.deleteLater()


def test_ui_add_custom_model(qapp):
    from lunaris.surrogate.st_lrps.ui.studio import OrbitBenchmarkTab

    tab = OrbitBenchmarkTab()
    ok, err = tab._try_add_model("sh100")
    assert ok and err == ""
    ok2, err2 = tab._try_add_model("sh30")
    assert ok2 and err2 == ""
    assert "sh100" in tab._model_checks and tab._model_checks["sh100"].isChecked()
    assert "sh30" in tab._model_checks and tab._model_checks["sh30"].isChecked()
    tab.run_mode.setCurrentIndex(tab.run_mode.findData("gpu_rk4"))
    args = tab._build_args(show_errors=False)
    gpu_models = args[args.index("--gpu-models") + 1].split(",")
    assert "sh100" in gpu_models
    assert "sh30" in gpu_models
    tab.deleteLater()


def test_ui_rejects_invalid_custom_model(qapp):
    from lunaris.surrogate.st_lrps.ui.studio import OrbitBenchmarkTab

    tab = OrbitBenchmarkTab()
    ok, err = tab._try_add_model("garbage")
    assert not ok and err
    assert "garbage" not in tab._model_checks
    ok2, _ = tab._try_add_model("sh99999")  # degree out of range
    assert not ok2 and "sh99999" not in tab._model_checks
    tab.deleteLater()


def test_ui_accumulate_toggle_emits_resume(qapp):
    from lunaris.surrogate.st_lrps.ui.studio import OrbitBenchmarkTab

    tab = OrbitBenchmarkTab()
    assert "--resume" not in (tab._build_args(show_errors=False) or [])
    tab.accumulate.setChecked(True)
    assert "--resume" in tab._build_args(show_errors=False)
    tab.deleteLater()


def test_ui_cache_resume_flags(qapp, tmp_path):
    from lunaris.surrogate.st_lrps.ui.studio import OrbitBenchmarkTab

    tab = OrbitBenchmarkTab()
    cache_dir = tmp_path / "cache"
    tab.accumulate.setChecked(True)
    tab.append_scenarios.setValue(25)
    tab.rebuild_metrics.setChecked(True)
    tab.strict_complete.setChecked(True)
    tab.cache_dir.setText(str(cache_dir))
    args = tab._build_args(show_errors=False)
    assert "--cache-trajectories" in args
    assert "--reuse-cache" in args
    assert "--resume" in args
    assert args[args.index("--append-scenarios") + 1] == "25"
    assert "--rebuild-metrics" in args
    assert "--strict-complete" in args
    assert args[args.index("--cache-dir") + 1] == str(cache_dir)

    tab.cache_trajectories.setChecked(False)
    tab.reuse_cache.setChecked(False)
    args2 = tab._build_args(show_errors=False)
    assert "--cache-trajectories" not in args2
    assert "--reuse-cache" not in args2
    tab.deleteLater()


def test_ui_qsettings_persists_sampling(qapp):
    from lunaris.surrogate.st_lrps.ui.studio import OrbitBenchmarkTab
    from lunaris.surrogate.st_lrps.ui.studio_parts.common_widgets import _settings

    settings = _settings()
    settings.beginGroup("orbit_benchmark")
    settings.remove("")
    settings.endGroup()
    settings.sync()

    tab = OrbitBenchmarkTab()
    tab.sampling_method.setCurrentIndex(tab.sampling_method.findData("lhs"))
    tab.inclination_sampling.setCurrentIndex(tab.inclination_sampling.findData("uniform_cos"))
    tab.truth_workers.setValue(6)
    tab.cache_trajectories.setChecked(False)
    tab.reuse_cache.setChecked(False)
    tab.accumulate.setChecked(True)
    tab.append_scenarios.setValue(9)
    tab.rebuild_metrics.setChecked(True)
    tab.strict_complete.setChecked(True)
    tab.cache_dir.setText(str(ROOT / "tmp_cache"))
    tab.rk4_dt_list.setText("10,30")
    tab._save_settings()
    tab.deleteLater()

    restored = OrbitBenchmarkTab()
    assert restored.sampling_method.currentData() == "lhs"
    assert restored.inclination_sampling.currentData() == "uniform_cos"
    assert restored.truth_workers.value() == 6
    assert restored.cache_trajectories.isChecked() is False
    assert restored.reuse_cache.isChecked() is False
    assert restored.accumulate.isChecked() is True
    assert restored.append_scenarios.value() == 9
    assert restored.rebuild_metrics.isChecked() is True
    assert restored.strict_complete.isChecked() is True
    assert restored.cache_dir.text() == str(ROOT / "tmp_cache")
    assert restored.rk4_dt_list.text() == "10,30"
    restored.deleteLater()

    settings.beginGroup("orbit_benchmark")
    settings.remove("")
    settings.endGroup()
    settings.sync()


# ---------------------------------------------------------------------------
# Backend scenario sampling and manifests
# ---------------------------------------------------------------------------
def _sampling_args(**overrides):
    data = dict(
        random_scenarios=8,
        scenario_seed=42,
        scenario_mode="bounded_keplerian",
        sampling_method="random",
        inclination_sampling="uniform_deg",
        altitude_min_km=100.0,
        altitude_max_km=1000.0,
        ecc_min=0.0,
        ecc_max=0.0,
        inc_min_deg=5.0,
        inc_max_deg=145.0,
        raan_min_deg=0.0,
        raan_max_deg=360.0,
        argp_min_deg=0.0,
        argp_max_deg=360.0,
        ta_min_deg=0.0,
        ta_max_deg=360.0,
        scenario_limit=None,
        resume=False,
        truth="sh200",
        truth_integrator="DOP853",
        duration_days=0.01,
        dt_out=60.0,
        rk4_dt_s=10.0,
        st_lrps_rk4_dt=30.0,
        gpu_integrator="medium",
        torch_dtype="float64",
        batch_frame_mode="precomputed_slerp",
        st_lrps_model_dir=None,
        models="sh20",
        gpu_models="sh20",
        cache_trajectories=True,
        reuse_cache=True,
        cache_dir=None,
        append_scenarios=0,
        rebuild_metrics=False,
        strict_complete=False,
        allow_lhs_append=False,
        plot_theme="report_light",
        plot_error_logscale=False,
        plot_3d=False,
        plot_best_scenario_id=None,
        plot_worst_scenario_id=None,
        plot_representative_scenario_id=None,
        plot_scenario_id=None,
    )
    data.update(overrides)
    return argparse.Namespace(**data)


def _assert_valid_scenarios(scenarios, args):
    assert len(scenarios) == int(args.random_scenarios)
    for s in scenarios:
        assert args.altitude_min_km <= s.hp_km <= args.altitude_max_km
        assert args.altitude_min_km <= s.ha_km <= args.altitude_max_km
        assert s.hp_km <= s.ha_km
        assert 0.0 <= s.e < 1.0
        assert args.inc_min_deg <= s.inc_deg <= args.inc_max_deg


def test_backend_random_sampling_remains_valid_within_bounds():
    from lunaris.surrogate.st_lrps.evaluation import compare_gravity_models as cgm

    args = _sampling_args(sampling_method="random")
    scenarios = cgm.generate_validation_scenarios(args)
    _assert_valid_scenarios(scenarios, args)
    assert all(s.raw_unit_sample is None for s in scenarios)


@pytest.mark.parametrize("method", ["lhs", "sobol"])
def test_backend_space_filling_sampling_generates_valid_scenarios(method):
    pytest.importorskip("scipy.stats.qmc")
    from lunaris.surrogate.st_lrps.evaluation import compare_gravity_models as cgm

    args = _sampling_args(sampling_method=method, random_scenarios=7)
    scenarios = cgm.generate_validation_scenarios(args)
    _assert_valid_scenarios(scenarios, args)
    assert all(s.raw_unit_sample and len(s.raw_unit_sample) == 6 for s in scenarios)


def test_backend_sobol_scrambled_seed_determinism():
    pytest.importorskip("scipy.stats.qmc")
    from lunaris.surrogate.st_lrps.evaluation import compare_gravity_models as cgm

    a = cgm.generate_unit_samples(5, 6, "sobol_scrambled", 42)
    b = cgm.generate_unit_samples(5, 6, "sobol_scrambled", 42)
    c = cgm.generate_unit_samples(5, 6, "sobol_scrambled", 43)
    assert a.shape == (5, 6)
    assert (a == b).all()
    assert not (a == c).all()


def test_backend_lhs_near_circular_and_uniform_cos_stay_in_bounds():
    pytest.importorskip("scipy.stats.qmc")
    from lunaris.surrogate.st_lrps.evaluation import compare_gravity_models as cgm

    args = _sampling_args(
        sampling_method="lhs",
        scenario_mode="near_circular_altitude",
        inclination_sampling="uniform_cos",
        ecc_min=0.0,
        ecc_max=0.02,
        inc_min_deg=20.0,
        inc_max_deg=110.0,
    )
    scenarios = cgm.generate_validation_scenarios(args)
    assert len(scenarios) == args.random_scenarios
    for s in scenarios:
        assert 0.0 <= s.e <= 0.02
        assert 20.0 <= s.inc_deg <= 110.0
        assert s.hp_km <= s.ha_km


def test_backend_manifest_written_json_safe_and_resume_conflict(tmp_path):
    pytest.importorskip("scipy.stats.qmc")
    from lunaris.surrogate.st_lrps.evaluation import compare_gravity_models as cgm

    args = _sampling_args(sampling_method="sobol_scrambled", random_scenarios=5)
    scenarios = cgm.prepare_scenarios(args, tmp_path)
    assert len(scenarios) == 5
    csv_path = tmp_path / "scenario_manifest.csv"
    json_path = tmp_path / "scenario_manifest.json"
    assert csv_path.exists()
    assert json_path.exists()

    manifest = json.loads(json_path.read_text(encoding="utf-8"))
    json.dumps(manifest)
    assert manifest["metadata"]["sampling_method"] == "sobol_scrambled"
    assert manifest["metadata"]["scenario_count"] == 5
    assert manifest["metadata"]["sampling_note"]
    assert isinstance(manifest["scenarios"][0]["raw_unit_sample"], list)

    resume_args = _sampling_args(sampling_method="sobol_scrambled", random_scenarios=5, resume=True)
    loaded = cgm.prepare_scenarios(resume_args, tmp_path)
    assert [s.hp_km for s in loaded] == [s.hp_km for s in scenarios]

    bad_args = _sampling_args(sampling_method="lhs", random_scenarios=5, resume=True)
    with pytest.raises(ValueError, match="sampling_method=sobol_scrambled"):
        cgm.prepare_scenarios(bad_args, tmp_path)


def test_backend_sobol_manifest_extends_with_stable_prefix(tmp_path):
    pytest.importorskip("scipy.stats.qmc")
    from lunaris.surrogate.st_lrps.evaluation import compare_gravity_models as cgm

    args4 = _sampling_args(sampling_method="sobol_scrambled", random_scenarios=4)
    first = cgm.prepare_scenarios(args4, tmp_path)
    args8 = _sampling_args(
        sampling_method="sobol_scrambled",
        random_scenarios=8,
        resume=True,
    )
    extended = cgm.prepare_scenarios(args8, tmp_path)
    assert len(extended) == 8
    assert [s.scenario_id for s in extended] == list(range(8))
    assert [round(s.hp_km, 12) for s in extended[:4]] == [round(s.hp_km, 12) for s in first]
    manifest = json.loads((tmp_path / "scenario_manifest.json").read_text(encoding="utf-8"))
    assert manifest["metadata"]["scenario_count"] == 8


def test_backend_rebuild_metrics_uses_existing_manifest_count(tmp_path):
    pytest.importorskip("scipy.stats.qmc")
    from lunaris.surrogate.st_lrps.evaluation import compare_gravity_models as cgm

    args4 = _sampling_args(sampling_method="sobol_scrambled", random_scenarios=4)
    cgm.prepare_scenarios(args4, tmp_path)
    rebuild = _sampling_args(
        sampling_method="sobol_scrambled",
        random_scenarios=8,
        rebuild_metrics=True,
    )
    scenarios = cgm.prepare_scenarios(rebuild, tmp_path)
    assert len(scenarios) == 4
    manifest = json.loads((tmp_path / "scenario_manifest.json").read_text(encoding="utf-8"))
    assert manifest["metadata"]["scenario_count"] == 4


def test_backend_lhs_extension_requires_explicit_blockwise_opt_in(tmp_path):
    pytest.importorskip("scipy.stats.qmc")
    from lunaris.surrogate.st_lrps.evaluation import compare_gravity_models as cgm

    args4 = _sampling_args(sampling_method="lhs", random_scenarios=4)
    cgm.prepare_scenarios(args4, tmp_path)

    args8 = _sampling_args(sampling_method="lhs", random_scenarios=8, resume=True)
    with pytest.raises(ValueError, match="LHS is not naturally nested"):
        cgm.prepare_scenarios(args8, tmp_path)

    allowed = _sampling_args(
        sampling_method="lhs",
        random_scenarios=8,
        resume=True,
        allow_lhs_append=True,
    )
    extended = cgm.prepare_scenarios(allowed, tmp_path)
    assert len(extended) == 8
    manifest = json.loads((tmp_path / "scenario_manifest.json").read_text(encoding="utf-8"))
    assert manifest["metadata"]["lhs_append_mode"] == "blockwise"


def test_backend_atomic_cache_save_load_and_corrupt_recompute_detection(tmp_path):
    import numpy as np
    from lunaris.surrogate.st_lrps.evaluation import compare_gravity_models as cgm

    args = _sampling_args(random_scenarios=1)
    scenario = cgm.generate_validation_scenarios(args)[0]
    cache_dir = tmp_path / "benchmark_cache"
    t = np.array([0.0, 60.0])
    y = np.array([
        scenario.initial_state,
        scenario.initial_state + np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3]),
    ])

    path = cgm._save_cached_trajectory(
        cache_dir, scenario, "sh200_dop853", "truth", t, y, args,
        runtime_s=0.5, integrator="DOP853", dtype="float64",
        device="cpu", backend="cpu_truth", truth_model="sh200",
    )
    assert path.exists()
    assert not path.with_name(path.name + ".tmp").exists()
    loaded = cgm._load_cached_trajectory(path)
    assert loaded is not None
    assert loaded.runtime_s == 0.5
    assert np.allclose(loaded.t, t)
    assert np.allclose(loaded.y, y)

    corrupt = cache_dir / "truth" / "sh200_dop853" / "scenario_000099.npz"
    corrupt.write_bytes(b"not a zip")
    assert cgm._load_cached_trajectory(corrupt) is None


def test_backend_truth_and_model_cache_completion_counts(tmp_path):
    import numpy as np
    from lunaris.surrogate.st_lrps.evaluation import compare_gravity_models as cgm

    args = _sampling_args(random_scenarios=2)
    scenarios = cgm.generate_validation_scenarios(args)
    cache_dir = tmp_path / "benchmark_cache"
    t = np.array([0.0, 60.0])
    y = np.vstack([scenarios[0].initial_state, scenarios[0].initial_state])

    cgm._save_cached_trajectory(
        cache_dir, scenarios[0], cgm._truth_cache_name(args), "truth", t, y, args,
        runtime_s=0.1, integrator="DOP853", dtype="float64",
        device="cpu", backend="cpu_truth", truth_model="sh200",
    )
    cgm._save_cached_trajectory(
        cache_dir, scenarios[0], "sh20", "comparison_model", t, y, args,
        runtime_s=0.2, integrator="DOP853", dtype="float64",
        device="cpu", backend="cpu_adaptive", truth_model="sh200",
    )

    truth_complete, truth_missing = cgm._truth_cache_completion(cache_dir, args, scenarios)
    model_complete, model_missing = cgm._model_cache_completion(cache_dir, "sh20", scenarios)
    assert truth_complete == 1 and [s.scenario_id for s in truth_missing] == [1]
    assert model_complete == 1 and [s.scenario_id for s in model_missing] == [1]


def test_backend_cache_duration_mismatch_rejects_reuse(tmp_path):
    from lunaris.surrogate.st_lrps.evaluation import compare_gravity_models as cgm

    args = _sampling_args(random_scenarios=1)
    scenarios = cgm.generate_validation_scenarios(args)
    cache_dir = tmp_path / "benchmark_cache"
    cgm._write_cache_manifest(args, cache_dir, scenarios, ["sh20"])
    bad = _sampling_args(random_scenarios=1, duration_days=0.02)
    with pytest.raises(ValueError, match="duration_days"):
        cgm._validate_cache_compatibility(bad, cache_dir)


def test_backend_rebuild_gpu_metrics_from_cached_trajectories(tmp_path):
    import numpy as np
    from lunaris.surrogate.st_lrps.evaluation import compare_gravity_models as cgm

    args = _sampling_args(random_scenarios=1, strict_complete=True)
    scenario = cgm.generate_validation_scenarios(args)[0]
    cache_dir = tmp_path / "benchmark_cache"
    metrics_dir = tmp_path / "metrics"
    plots_dir = tmp_path / "plots"
    reports_dir = tmp_path / "reports"
    t = np.array([0.0, 60.0])
    y = np.vstack([scenario.initial_state, scenario.initial_state])
    cgm._save_cached_trajectory(
        cache_dir, scenario, cgm._truth_cache_name(args), "truth", t, y, args,
        runtime_s=0.1, integrator="DOP853", dtype="float64",
        device="cpu", backend="cpu_truth", truth_model="sh200",
    )
    cgm._save_cached_trajectory(
        cache_dir, scenario, "sh20", "comparison_model", t, y, args,
        runtime_s=0.2, integrator="medium", rk4_dt_s=10.0,
        dtype="float64", device="cpu", backend="gpu_batch", truth_model="sh200",
    )

    cgm.rebuild_gpu_batch_metrics_from_cache(
        args, [scenario], cache_dir, ["sh20"], metrics_dir, plots_dir, reports_dir
    )
    per = metrics_dir / "gpu_batch_per_scenario_metrics.csv"
    agg = metrics_dir / "gpu_batch_aggregate_metrics.csv"
    assert per.exists()
    assert agg.exists()
    assert "GPU_SH20_RK4" in per.read_text(encoding="utf-8")
    assert (cache_dir / "metrics" / "per_model_scenario_metrics.csv").exists()
    assert (cache_dir / "metrics" / "aggregate_metrics.csv").exists()
    assert (plots_dir / "ensemble_mean_position_error_vs_time.png").exists()
    assert (plots_dir / "selected_representative_position_error_all_models.png").exists()


def test_backend_strict_complete_rejects_missing_model_cache(tmp_path):
    import numpy as np
    from lunaris.surrogate.st_lrps.evaluation import compare_gravity_models as cgm

    args = _sampling_args(random_scenarios=1, strict_complete=True)
    scenario = cgm.generate_validation_scenarios(args)[0]
    cache_dir = tmp_path / "benchmark_cache"
    t = np.array([0.0, 60.0])
    y = np.vstack([scenario.initial_state, scenario.initial_state])
    cgm._save_cached_trajectory(
        cache_dir, scenario, cgm._truth_cache_name(args), "truth", t, y, args,
        runtime_s=0.1, integrator="DOP853", dtype="float64",
        device="cpu", backend="cpu_truth", truth_model="sh200",
    )

    with pytest.raises(RuntimeError, match="missing 1 cached scenario"):
        cgm.rebuild_gpu_batch_metrics_from_cache(
            args, [scenario], cache_dir, ["sh20"],
            tmp_path / "metrics", tmp_path / "plots", tmp_path / "reports",
        )


# ---------------------------------------------------------------------------
# Accumulation helpers (CSV reload / numeric coercion)
# ---------------------------------------------------------------------------
def test_metric_row_coercion_and_read(tmp_path):
    from lunaris.surrogate.st_lrps.evaluation.compare_gravity_models import _coerce_numeric_row, _read_csv_rows

    coerced = _coerce_numeric_row({
        "scenario_id": "3", "model": "GPU_SH20_RK4", "status": "ok",
        "rms_pos_err_km": "0.125", "final_pos_err_km": "", "max_pos_err_km": "None",
    })
    assert coerced["scenario_id"] == 3.0
    assert coerced["model"] == "GPU_SH20_RK4"   # string column preserved
    assert coerced["status"] == "ok"
    assert coerced["rms_pos_err_km"] == 0.125
    import math
    assert math.isnan(coerced["final_pos_err_km"])
    assert math.isnan(coerced["max_pos_err_km"])

    csv_path = tmp_path / "m.csv"
    csv_path.write_text("scenario_id,model,rms_pos_err_km\n0,GPU_SH20_RK4,1.5\n1,GPU_SH20_RK4,2.5\n",
                        encoding="utf-8")
    rows = _read_csv_rows(csv_path)
    assert len(rows) == 2 and rows[0]["model"] == "GPU_SH20_RK4"
    assert _read_csv_rows(tmp_path / "missing.csv") == []


# ---------------------------------------------------------------------------
# Relocated publication-plot module
#
# NOTE: the publication module imports pandas and the report toolkit drives
# matplotlib/PdfPages. Importing those heavyweight native libraries into the
# shared pytest process perturbs MKL/OpenMP threading state and can destabilise
# unrelated torch double-backward tests later in the run. We therefore exercise
# these two paths in isolated subprocesses.
# ---------------------------------------------------------------------------
def test_publication_plots_importable_and_parses():
    # --help imports the module (and pandas) then exits 0; no args -> SystemExit.
    help_p = subprocess.run(
        [sys.executable, "-m", "lunaris.surrogate.st_lrps.evaluation.publication_plots", "--help"],
        cwd=str(ROOT), capture_output=True, text=True, timeout=180,
    )
    combined = (help_p.stdout + help_p.stderr).lower()
    if "requires pandas" in combined:
        pytest.skip("pandas not installed")
    assert help_p.returncode == 0
    assert "publication" in combined or "compare_gravity_models" in combined

    noargs_p = subprocess.run(
        [sys.executable, "-m", "lunaris.surrogate.st_lrps.evaluation.publication_plots"],
        cwd=str(ROOT), capture_output=True, text=True, timeout=180,
    )
    assert noargs_p.returncode != 0  # clean SystemExit ("Provide --run ..."), not a crash


# ---------------------------------------------------------------------------
# Professional PDF report toolkit (rendered in a subprocess; see note above)
# ---------------------------------------------------------------------------
def test_report_pager_renders_pdf(tmp_path):
    pdf_path = tmp_path / "report.pdf"
    code = textwrap.dedent(
        """
        import sys
        from pathlib import Path
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
        from lunaris.surrogate.st_lrps.evaluation.compare_gravity_models import _ReportPager

        out = Path(sys.argv[1])
        fig, ax = plt.subplots(figsize=(3, 2)); ax.plot([0, 1], [0, 1])
        png = out.parent / "demo.png"; fig.savefig(png); plt.close(fig)
        pdf_path = out
        with PdfPages(pdf_path) as pdf:
            pager = _ReportPager(pdf, title="Test Report", subtitle="unit test")
            pager.cover(meta=[("Scenarios", "200"), ("Seed", "42")], note="reference note")
            pager.table_page("Ranking", ["Model", "RMS [km]"],
                             [["GPU_ST_LRPS_RK4", "0.0002"], ["GPU_SH20_RK4", "0.0985"]],
                             highlight_row=0, intro="lower is better")
            assert pager.figure_page("Figure", png, "caption") is True
            assert pager.figure_page("Figure", out.parent / "missing.png", "x") is False
            pager.text_page("Notes", ["- one", "- two"])
            assert pager.page_no >= 4
        assert pdf_path.exists() and pdf_path.stat().st_size > 2000
        print("REPORT_OK")
        """
    )
    p = subprocess.run([sys.executable, "-c", code, str(pdf_path)],
                       cwd=str(ROOT), capture_output=True, text=True, timeout=180)
    assert p.returncode == 0, (p.stdout + p.stderr)
    assert "REPORT_OK" in p.stdout
    assert pdf_path.exists()


# ---------------------------------------------------------------------------
# Run-monitor dashboard: status parser, pipeline state machine, telemetry
# ---------------------------------------------------------------------------
_TELEMETRY_LINE = '{"t_s":21600.0,"alt_km":120.0,"v_km_s":1.6,"ecc":0.001}'


def test_dashboard_parses_progress_phase_line(qapp):
    from lunaris.surrogate.st_lrps.ui.studio import OrbitBenchmarkTab

    tab = OrbitBenchmarkTab()
    tab._rebuild_pipeline(["sh20"])
    tab.runner.append(
        "[progress] phase=gpu_model model=sh20 current_step=433 total_steps=14400 "
        "percent=3.0 elapsed_s=96 eta_s=3082 steps_per_s=4.53 device=cuda:0 "
        "dtype=float64 n_scenarios=4"
    )
    assert tab._st_phase.text() == "GPU model"
    assert tab._st_model.text() == "SH20"
    assert tab._st_phase_pct.text() == "3.0%"
    assert tab._st_steps.text() == "4.5"
    assert tab._st_scn.text() == "4"
    assert tab.phase_bar.maximum() == 100 and tab.phase_bar.value() == 3
    assert "433" in tab._phase_detail.text()
    tab.deleteLater()


def test_dashboard_parses_progress_total_line(qapp):
    from lunaris.surrogate.st_lrps.ui.studio import OrbitBenchmarkTab

    tab = OrbitBenchmarkTab()
    tab._rebuild_pipeline(["sh20"])
    tab.runner.append(
        "[progress_total] percent=40.4 phase=gpu_model model=sh20 elapsed_s=400 eta_s=591"
    )
    assert tab._st_overall_pct.text() == "40.4%"
    assert tab.overall_bar.maximum() == 100 and tab.overall_bar.value() == 40
    assert tab._st_eta.text() != "-"
    tab.deleteLater()


def test_dashboard_cache_line_marks_truth_cached(qapp):
    from lunaris.surrogate.st_lrps.ui.studio import OrbitBenchmarkTab

    tab = OrbitBenchmarkTab()
    tab._rebuild_pipeline(["sh20"])
    tab.runner.append("[cache] Truth cache sh200_dop853: 4/4 complete.")
    assert tab._model_status["truth"] == "cached"
    tab.deleteLater()


def test_dashboard_cache_line_marks_model_cached(qapp):
    from lunaris.surrogate.st_lrps.ui.studio import OrbitBenchmarkTab

    tab = OrbitBenchmarkTab()
    tab._rebuild_pipeline(["sh20", "sh80"])
    tab.runner.append("[cache] Model GPU_SH80_RK4: 4/4 complete.")
    assert tab._model_status["gpu_sh80_rk4"] == "cached"
    tab.deleteLater()


def test_dashboard_partial_cache_keeps_model_queued(qapp):
    from lunaris.surrogate.st_lrps.ui.studio import OrbitBenchmarkTab

    tab = OrbitBenchmarkTab()
    tab._rebuild_pipeline(["sh20"])
    tab.runner.append("[cache] Model GPU_SH20_RK4: 0/4 complete. Recomputing 4 missing.")
    assert tab._model_status["gpu_sh20_rk4"] == "queued"
    tab.deleteLater()


def test_dashboard_gpu_start_marks_running_live(qapp):
    from lunaris.surrogate.st_lrps.ui.studio import OrbitBenchmarkTab

    tab = OrbitBenchmarkTab()
    tab._rebuild_pipeline(["sh20", "sh80"])
    # Live: marked running on the very first start line (not only at the end).
    tab.runner.append("[gpu-batch] Model 01/2 | GPU_SH20_RK4 starting for 4 scenario(s) (rk4_dt=10s) ...")
    assert tab._model_status["gpu_sh20_rk4"] == "running"
    tab.deleteLater()


def test_dashboard_gpu_done_marks_completed(qapp):
    from lunaris.surrogate.st_lrps.ui.studio import OrbitBenchmarkTab

    tab = OrbitBenchmarkTab()
    tab._rebuild_pipeline(["sh20"])
    tab.runner.append("[gpu-batch] Model 01/1 | GPU_SH20_RK4 starting for 4 scenario(s) ...")
    tab.runner.append(
        "[gpu-batch] Model 01/1 done | GPU_SH20_RK4: 9.9s backend=torch_sh status=ok | ETA 00:00:10"
    )
    assert tab._model_status["gpu_sh20_rk4"] == "completed"
    tab.deleteLater()


def test_dashboard_gpu_error_marks_failed(qapp):
    from lunaris.surrogate.st_lrps.ui.studio import OrbitBenchmarkTab

    tab = OrbitBenchmarkTab()
    tab._rebuild_pipeline(["sh20", "st_lrps"])
    tab.runner.append("[gpu-batch] Model 02/2 | GPU_ST_LRPS_RK4 starting for 4 scenario(s) ...")
    # Error lines carry the base name; the running variant must be marked failed.
    tab.runner.append("[gpu-batch] ERROR st_lrps: CUDA out of memory")
    assert tab._model_status["gpu_st_lrps_rk4"] == "failed"
    tab.deleteLater()


def test_dashboard_next_model_completes_previous(qapp):
    from lunaris.surrogate.st_lrps.ui.studio import OrbitBenchmarkTab

    tab = OrbitBenchmarkTab()
    tab._rebuild_pipeline(["sh20", "sh80"])
    tab.runner.append("[gpu-batch] Model 01/2 | GPU_SH20_RK4 starting for 4 scenario(s) ...")
    tab.runner.append("[gpu-batch] Model 02/2 | GPU_SH80_RK4 starting for 4 scenario(s) ...")
    assert tab._model_status["gpu_sh20_rk4"] == "completed"
    assert tab._model_status["gpu_sh80_rk4"] == "running"
    tab.deleteLater()


def test_dashboard_pipeline_adds_step_variants_in_order(qapp):
    """Δt (step-size) variants are added as separate chips, live and in order."""
    from lunaris.surrogate.st_lrps.ui.studio import OrbitBenchmarkTab

    tab = OrbitBenchmarkTab()
    tab._rebuild_pipeline(["sh20"])
    tab.runner.append("[gpu-batch] Model 01/3 | GPU_SH20_RK4_DT10 starting for 4 scenario(s) (rk4_dt=10s) ...")
    tab.runner.append("[gpu-batch] Model 01/3 done | GPU_SH20_RK4_DT10: 12s status=ok | ETA 00:00:20")
    tab.runner.append("[gpu-batch] Model 02/3 | GPU_SH20_RK4_DT5 starting for 4 scenario(s) (rk4_dt=5s) ...")
    assert tab._pipeline_order == ["truth", "gpu_sh20_rk4_dt10", "gpu_sh20_rk4_dt5", "report"]
    assert tab._model_status["gpu_sh20_rk4_dt10"] == "completed"
    assert tab._model_status["gpu_sh20_rk4_dt5"] == "running"
    tab.deleteLater()


def test_dashboard_hides_machine_progress_lines(qapp):
    from lunaris.surrogate.st_lrps.ui.studio import OrbitBenchmarkTab

    tab = OrbitBenchmarkTab()
    tab._rebuild_pipeline(["sh20"])
    tab.runner.append("[progress] phase=gpu_model model=GPU_SH20_RK4 current_step=1 total_steps=10 percent=10.0")
    tab.runner.append("[progress_total] percent=42.0 phase=gpu_model model=GPU_SH20_RK4 elapsed_s=10 eta_s=10")
    log = tab.runner.log.toPlainText()
    assert "[progress]" not in log and "[progress_total]" not in log
    # ...but the parser still consumed them: the dashboard updated.
    assert tab._st_overall_pct.text() == "42.0%"
    tab.deleteLater()


def test_dashboard_telemetry_hidden_by_default(qapp):
    from lunaris.surrogate.st_lrps.ui.studio import OrbitBenchmarkTab

    tab = OrbitBenchmarkTab()
    assert tab.show_telemetry.isChecked() is False
    tab.runner.append(_TELEMETRY_LINE)
    assert "t_s" not in tab.runner.log.toPlainText()
    assert "Telemetry lines hidden" in tab._telemetry_note.text()
    tab.deleteLater()


def test_dashboard_telemetry_shown_when_enabled(qapp):
    from lunaris.surrogate.st_lrps.ui.studio import OrbitBenchmarkTab

    tab = OrbitBenchmarkTab()
    tab.show_telemetry.setChecked(True)
    tab.runner.append(_TELEMETRY_LINE)
    assert "21600" in tab.runner.log.toPlainText()
    tab.deleteLater()


def test_dashboard_normal_lines_always_shown(qapp):
    from lunaris.surrogate.st_lrps.ui.studio import OrbitBenchmarkTab

    tab = OrbitBenchmarkTab()
    tab.runner.append("[gpu-batch] Model 01/2 | GPU_SH20_RK4 starting for 4 scenario(s) ...")
    assert "starting for 4 scenario" in tab.runner.log.toPlainText()
    tab.deleteLater()


def test_dashboard_unknown_lines_do_not_crash(qapp):
    from lunaris.surrogate.st_lrps.ui.studio import OrbitBenchmarkTab

    tab = OrbitBenchmarkTab()
    tab._rebuild_pipeline(["sh20"])
    for line in (
        "totally unstructured output",
        "[progress]",
        "[progress_total]",
        "{not really json",
        "",
        "[gpu-batch] Model weird line",
    ):
        tab.runner.append(line)  # must not raise
    tab.runner.append("[progress_total] percent=12.5 phase=truth elapsed_s=10 eta_s=70")
    assert tab._st_overall_pct.text() == "12.5%"
    tab.deleteLater()


def test_dashboard_pipeline_order_follows_selection(qapp):
    from lunaris.surrogate.st_lrps.ui.studio import OrbitBenchmarkTab

    tab = OrbitBenchmarkTab()
    tab._rebuild_pipeline(["sh160", "sh20", "st_lrps"])
    assert tab._pipeline_order == ["truth", "sh160", "sh20", "st_lrps", "report"]
    assert tab._model_status["sh160"] == "queued"
    assert tab._model_status["truth"] == "pending"
    assert tab._model_status["report"] == "pending"
    tab.deleteLater()


def test_dashboard_finish_finalizes_pipeline(qapp):
    from lunaris.surrogate.st_lrps.ui.studio import OrbitBenchmarkTab
    from lunaris.surrogate.st_lrps.ui.studio_parts.qt_common import QProcess

    tab = OrbitBenchmarkTab()
    tab._rebuild_pipeline(["sh20", "st_lrps"])
    tab.runner.append("[gpu-batch] Model 01/2 | GPU_SH20_RK4 starting for 4 scenario(s) ...")
    tab._on_finished(0, QProcess.ExitStatus.NormalExit)
    assert tab._model_status["gpu_sh20_rk4"] == "completed"
    assert tab._model_status["report"] == "completed"
    assert tab._status_badge.text() == "Completed"
    assert tab.overall_bar.value() == 100
    tab.deleteLater()


# ---------------------------------------------------------------------------
# Single-page layout: in-place collapsible Results/Plots (no secondary page)
# ---------------------------------------------------------------------------

def test_no_secondary_plot_page(qapp):
    """The benchmark tab is one page — no QSplitter dividing config from plots.

    (The gallery's own QTabWidget contains a QStackedWidget internally for
    switching plots; that is not a navigation sub-page, so we only assert the
    absence of a splitter / bottom pane here.)
    """
    from lunaris.surrogate.st_lrps.ui.studio import OrbitBenchmarkTab
    from lunaris.surrogate.st_lrps.ui.studio_parts.qt_common import QSplitter

    tab = OrbitBenchmarkTab()
    assert tab.findChildren(QSplitter) == []
    # The plots live inside the in-place Results section, not a separate pane.
    assert tab._results_section.isAncestorOf(tab._gallery)
    tab.deleteLater()


def test_results_section_present_on_same_page(qapp):
    from lunaris.surrogate.st_lrps.ui.studio import OrbitBenchmarkTab
    from lunaris.surrogate.st_lrps.ui.studio_parts.common_widgets import CollapsibleSection, ImageGallery

    tab = OrbitBenchmarkTab()
    assert isinstance(tab._results_section, CollapsibleSection)
    assert isinstance(tab._logs_section, CollapsibleSection)
    # The single, persistent gallery lives inside the Results/Plots section.
    assert isinstance(tab._gallery, ImageGallery)
    assert tab._results_section.isAncestorOf(tab._gallery)
    # Both sections share the same parent chain (one page, not a sub-page).
    assert tab.isAncestorOf(tab._results_section)
    assert tab.isAncestorOf(tab._logs_section)
    tab.deleteLater()


def test_results_section_collapse_expand(qapp):
    from lunaris.surrogate.st_lrps.ui.studio import OrbitBenchmarkTab

    tab = OrbitBenchmarkTab()
    # Starts collapsed (minimal vertical space until a run produces plots).
    assert tab._results_section.is_expanded() is False
    tab._results_section.set_expanded(True)
    assert tab._results_section.is_expanded() is True
    assert tab._gallery.isVisibleTo(tab._results_section) is True
    tab._results_section.set_expanded(False)
    assert tab._results_section.is_expanded() is False
    assert tab._gallery.isVisibleTo(tab._results_section) is False
    tab.deleteLater()


def test_gallery_persistent_across_toggle(qapp):
    """The gallery is created once and only shown/hidden, never recreated."""
    from lunaris.surrogate.st_lrps.ui.studio import OrbitBenchmarkTab

    tab = OrbitBenchmarkTab()
    gallery_id = id(tab._gallery)
    for _ in range(3):
        tab._results_section.set_expanded(True)
        tab._results_section.set_expanded(False)
    assert id(tab._gallery) == gallery_id
    # Roomy when expanded so plots are not squeezed.
    assert tab._gallery.minimumHeight() >= 500
    tab.deleteLater()


def test_refresh_results_loads_plots_in_place(qapp, tmp_path):
    """Refresh re-scans the output dir and expands the in-place section."""
    from lunaris.surrogate.st_lrps.ui.studio import OrbitBenchmarkTab

    # Write a tiny valid PNG into the output dir.
    import base64
    png = base64.b64decode(
        b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==")
    (tmp_path / "plot_one.png").write_bytes(png)

    tab = OrbitBenchmarkTab()
    tab.out_dir.setText(str(tmp_path))
    tab._effective_out_dir = str(tmp_path)
    assert tab._results_section.is_expanded() is False
    tab._refresh_results()
    assert tab._results_section.is_expanded() is True
    tab.deleteLater()


# ---------------------------------------------------------------------------
# Separate cache-only "Gravity Plots" page (no run / no training)
# ---------------------------------------------------------------------------

def test_plots_page_registered_and_separate(qapp):
    from lunaris.surrogate.st_lrps.ui.studio_parts.main_window import MainWindow

    w = MainWindow()
    assert "Gravity Plots" in w._page_titles
    # It is its own top-level page, distinct from the benchmark run page.
    assert w._orbit_plots_page is not w._orbit_benchmark_page
    w.deleteLater()


def _write_fake_benchmark_cache(out_dir, models, *, truth="sh200",
                                integrator="DOP853", rk4_dt=10.0, dt_list=None):
    """Create a minimal benchmark_cache (model folders + manifest) under out_dir.

    Mirrors the on-disk layout the harness writes so the Gravity Plots page can
    discover models from the folder and auto-detect truth / step size.
    """
    cache = Path(out_dir) / "benchmark_cache"
    (cache / "truth" / f"{truth}_{integrator.lower()}").mkdir(parents=True, exist_ok=True)
    for m in models:
        d = cache / "models" / m
        d.mkdir(parents=True, exist_ok=True)
        (d / "scenario_000000.npz").write_bytes(b"")
    manifest = {
        "cache_schema_version": 1,
        "metadata": {
            "truth": truth,
            "truth_integrator": integrator,
            "rk4_dt_s": rk4_dt,
            "gpu_rk4_dt_s_list": list(dt_list or []),
        },
        "selected_models": list(models),
    }
    (cache / "cache_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return cache


def test_plots_page_builds_cache_only_command(qapp, tmp_path):
    from lunaris.surrogate.st_lrps.ui.studio import OrbitBenchmarkPlotsTab

    _write_fake_benchmark_cache(tmp_path, ["sh20", "sh80", "st_lrps"])
    tab = OrbitBenchmarkPlotsTab()
    tab.out_dir.setText(str(tmp_path))
    tab._scan_and_populate()
    # Folder-first: the models are discovered from the chosen folder's cache.
    assert set(tab._model_checks) == {"sh20", "sh80", "st_lrps"}
    for n, cb in tab._model_checks.items():
        cb.setChecked(n in ("sh20", "sh80", "st_lrps"))
    args = tab._build_args(show_errors=False)
    assert args is not None
    # Cache-only rebuild — never propagates or trains.
    assert "--rebuild-metrics" in args
    assert "--reuse-cache" in args
    assert "--gpu-batch-compare" in args
    assert args[args.index("--gpu-models") + 1] == "sh20,sh80,st_lrps"
    assert args[args.index("--output-dir") + 1] == str(tmp_path)
    # Must NOT carry flags that would start a fresh run/sweep.
    assert "--resume" not in args
    assert "--cache-trajectories" not in args
    tab.deleteLater()


def test_plots_page_emits_step_variant_list(qapp, tmp_path):
    from lunaris.surrogate.st_lrps.ui.studio import OrbitBenchmarkPlotsTab

    _write_fake_benchmark_cache(tmp_path, ["sh20"])
    tab = OrbitBenchmarkPlotsTab()
    tab.out_dir.setText(str(tmp_path))
    tab._scan_and_populate()
    for n, cb in tab._model_checks.items():
        cb.setChecked(n == "sh20")
    tab.rk4_dt_list.setText("10,5")
    args = tab._build_args(show_errors=False)
    assert args[args.index("--gpu-rk4-dt-s-list") + 1] == "10,5"
    tab.deleteLater()


def test_plots_page_lists_models_from_folder_and_autodetects(qapp, tmp_path):
    """Folder-first flow: pick a folder, its cached models are listed, and the
    truth model / step size are auto-detected from the cache manifest."""
    from lunaris.surrogate.st_lrps.ui.studio import OrbitBenchmarkPlotsTab

    _write_fake_benchmark_cache(
        tmp_path, ["sh80", "sh20", "st_lrps"],
        truth="sh160", integrator="DOP853", rk4_dt=5.0,
    )
    tab = OrbitBenchmarkPlotsTab()
    # Nothing is listed before a folder is chosen.
    assert tab._model_checks == {}
    tab.out_dir.setText(str(tmp_path))
    tab._scan_and_populate()
    # Discovered and sorted (SH by degree, ST-LRPS last); all checked by default.
    assert list(tab._model_checks) == ["sh20", "sh80", "st_lrps"]
    assert all(cb.isChecked() for cb in tab._model_checks.values())
    # Truth model + fixed step auto-filled from the manifest.
    assert tab.truth.currentData() == "sh160"
    assert tab.rk4_dt.value() == 5.0
    tab.deleteLater()


def test_plots_page_requires_output_dir_and_models(qapp):
    from lunaris.surrogate.st_lrps.ui.studio import OrbitBenchmarkPlotsTab

    tab = OrbitBenchmarkPlotsTab()
    tab.out_dir.setText("")
    assert tab._build_args(show_errors=False) is None       # no output dir
    for cb in tab._model_checks.values():
        cb.setChecked(False)
    tab.out_dir.setText("/some/dir")
    assert tab._build_args(show_errors=False) is None       # no models
    tab.deleteLater()


# ---------------------------------------------------------------------------
# GPU frame mode (dynamic / precomputed_slerp) exposure
# ---------------------------------------------------------------------------

def test_cli_accepts_precomputed_slerp_frame_mode(monkeypatch):
    import sys as _sys
    from lunaris.surrogate.st_lrps.evaluation import compare_gravity_models as cgm

    monkeypatch.setattr(_sys, "argv", [
        "compare_gravity_models", "--batch-frame-mode", "precomputed_slerp",
    ])
    args = cgm.parse_args()
    assert args.batch_frame_mode == "precomputed_slerp"


def test_ui_gpu_mode_emits_frame_mode(qapp):
    from lunaris.surrogate.st_lrps.ui.studio import OrbitBenchmarkTab

    tab = OrbitBenchmarkTab()
    tab.run_mode.setCurrentIndex(tab.run_mode.findData("gpu_rk4"))
    # Default is now the optimized precomputed path.
    args = tab._build_args(show_errors=False)
    assert args[args.index("--batch-frame-mode") + 1] == "precomputed_slerp"
    # User can switch to the conservative dynamic path.
    tab.gpu_frame_mode.setCurrentIndex(tab.gpu_frame_mode.findData("match_dynamics_engine"))
    args = tab._build_args(show_errors=False)
    assert args[args.index("--batch-frame-mode") + 1] == "match_dynamics_engine"
    tab.deleteLater()


def test_ui_cpu_mode_omits_frame_mode(qapp):
    from lunaris.surrogate.st_lrps.ui.studio import OrbitBenchmarkTab

    tab = OrbitBenchmarkTab()
    tab.run_mode.setCurrentIndex(tab.run_mode.findData("dop853"))
    args = tab._build_args(show_errors=False)
    assert "--batch-frame-mode" not in args
    tab.deleteLater()
