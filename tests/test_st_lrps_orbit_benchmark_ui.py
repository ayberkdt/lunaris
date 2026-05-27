"""Tests for the Orbit-Level Benchmark Studio page and the relocated harness.

Covers:
- the gravity benchmark harness is importable at its new package path
  (``st_lrps.evaluation.compare_gravity_models``) and no longer at the old
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
    mod = importlib.import_module("st_lrps.evaluation.compare_gravity_models")
    assert hasattr(mod, "main")
    assert hasattr(mod, "parse_args")


def test_harness_file_moved_on_disk():
    assert (ROOT / "st_lrps" / "evaluation" / "compare_gravity_models.py").is_file()
    assert not (ROOT / "validation" / "gravity" / "compare_gravity_models.py").exists()


# ---------------------------------------------------------------------------
# UI command builders (skipped without PyQt6)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def qapp():
    pytest.importorskip("PyQt6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    from st_lrps.ui.studio_parts.common_widgets import _settings

    settings = _settings()
    settings.beginGroup("orbit_benchmark")
    settings.remove("")
    settings.endGroup()
    settings.sync()
    return app


def test_benchmark_tab_dop853_mode_args(qapp):
    from st_lrps.ui.studio import BENCHMARK_CLI_MODULE, OrbitBenchmarkTab

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
    tab.deleteLater()


def test_benchmark_tab_gpu_rk4_mode_args(qapp):
    from st_lrps.ui.studio import OrbitBenchmarkTab

    tab = OrbitBenchmarkTab()
    tab.run_mode.setCurrentIndex(tab.run_mode.findData("gpu_rk4"))
    tab.rk4_dt.setValue(10.0)
    args = tab._build_args(show_errors=False)
    assert args is not None
    assert "--gpu-batch-compare" in args
    assert "--gpu-models" in args
    assert "--rk4-dt-s" in args and args[args.index("--rk4-dt-s") + 1] == "10.0"
    assert "--torch-dtype" in args
    assert "--integrator" not in args
    assert "--models" not in args
    tab.deleteLater()


def test_benchmark_tab_requires_at_least_one_model(qapp):
    from st_lrps.ui.studio import OrbitBenchmarkTab

    tab = OrbitBenchmarkTab()
    for cb in tab._model_checks.values():
        cb.setChecked(False)
    assert tab._build_args(show_errors=False) is None
    tab.deleteLater()


def test_benchmark_tab_st_lrps_dir_emitted_when_set(qapp, tmp_path):
    from st_lrps.ui.studio import OrbitBenchmarkTab

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
    from st_lrps.ui.studio_parts.main_window import MainWindow

    w = MainWindow()
    assert w._stack.count() == 6
    assert "Orbit-Level Benchmark" in w._page_titles
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

    from st_lrps.evaluation.compare_gravity_models import (
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

    from st_lrps.evaluation.compare_gravity_models import gpu_fixed_step_advance

    def rhs(t, s):
        return -s

    s0 = np.array([1.0, 2.0])
    a = gpu_fixed_step_advance(rhs, 0.0, s0, 0.01, "nonsense")
    b = gpu_fixed_step_advance(rhs, 0.0, s0, 0.01, "medium")
    assert np.allclose(a, b)


# ---------------------------------------------------------------------------
# Harness: new CLI flags parse
# ---------------------------------------------------------------------------
def test_harness_parses_new_flags(monkeypatch):
    import sys as _sys

    from st_lrps.evaluation import compare_gravity_models as cgm

    argv = [
        "compare_gravity_models",
        "--gpu-integrator", "robust",
        "--truth-integrator", "RK45",
        "--workers", "4",
        "--sampling-method", "sobol_scrambled",
        "--inclination-sampling", "uniform_cos",
    ]
    monkeypatch.setattr(_sys, "argv", argv)
    args = cgm.parse_args()
    assert args.gpu_integrator == "robust"
    assert args.truth_integrator == "RK45"
    assert args.workers == 4
    assert args.sampling_method == "sobol_scrambled"
    assert args.inclination_sampling == "uniform_cos"


def test_cfg_with_integrator_overrides_method():
    from config import load_default_config
    from st_lrps.evaluation.compare_gravity_models import _cfg_with_integrator

    cfg = load_default_config()
    out = _cfg_with_integrator(cfg, "RK45")
    assert out.propagator.method == "RK45"
    # Original config untouched (frozen-dataclass copy semantics).
    assert cfg.propagator.method != "RK45" or out is not cfg


# ---------------------------------------------------------------------------
# UI: new controls / flags
# ---------------------------------------------------------------------------
def test_ui_dop853_mode_emits_truth_integrator_and_workers(qapp):
    from st_lrps.ui.studio import OrbitBenchmarkTab

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
    from st_lrps.ui.studio import OrbitBenchmarkTab

    tab = OrbitBenchmarkTab()
    tab.run_mode.setCurrentIndex(tab.run_mode.findData("gpu_rk4"))
    tab.gpu_integrator.setCurrentIndex(tab.gpu_integrator.findData("robust"))
    args = tab._build_args(show_errors=False)
    assert "--gpu-integrator" in args
    assert args[args.index("--gpu-integrator") + 1] == "robust"
    assert "--truth-integrator" in args
    assert "--workers" not in args
    tab.deleteLater()


def test_ui_sampling_flags_emitted_when_selected(qapp):
    from st_lrps.ui.studio import OrbitBenchmarkTab

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
    from st_lrps.ui.studio import OrbitBenchmarkTab

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
    from st_lrps.ui.studio import OrbitBenchmarkTab

    tab = OrbitBenchmarkTab()
    ok, err = tab._try_add_model("garbage")
    assert not ok and err
    assert "garbage" not in tab._model_checks
    ok2, _ = tab._try_add_model("sh99999")  # degree out of range
    assert not ok2 and "sh99999" not in tab._model_checks
    tab.deleteLater()


def test_ui_accumulate_toggle_emits_resume(qapp):
    from st_lrps.ui.studio import OrbitBenchmarkTab

    tab = OrbitBenchmarkTab()
    assert "--resume" not in (tab._build_args(show_errors=False) or [])
    tab.accumulate.setChecked(True)
    assert "--resume" in tab._build_args(show_errors=False)
    tab.deleteLater()


def test_ui_qsettings_persists_sampling(qapp):
    from st_lrps.ui.studio import OrbitBenchmarkTab
    from st_lrps.ui.studio_parts.common_widgets import _settings

    settings = _settings()
    settings.beginGroup("orbit_benchmark")
    settings.remove("")
    settings.endGroup()
    settings.sync()

    tab = OrbitBenchmarkTab()
    tab.sampling_method.setCurrentIndex(tab.sampling_method.findData("lhs"))
    tab.inclination_sampling.setCurrentIndex(tab.inclination_sampling.findData("uniform_cos"))
    tab._save_settings()
    tab.deleteLater()

    restored = OrbitBenchmarkTab()
    assert restored.sampling_method.currentData() == "lhs"
    assert restored.inclination_sampling.currentData() == "uniform_cos"
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
    from st_lrps.evaluation import compare_gravity_models as cgm

    args = _sampling_args(sampling_method="random")
    scenarios = cgm.generate_validation_scenarios(args)
    _assert_valid_scenarios(scenarios, args)
    assert all(s.raw_unit_sample is None for s in scenarios)


@pytest.mark.parametrize("method", ["lhs", "sobol"])
def test_backend_space_filling_sampling_generates_valid_scenarios(method):
    pytest.importorskip("scipy.stats.qmc")
    from st_lrps.evaluation import compare_gravity_models as cgm

    args = _sampling_args(sampling_method=method, random_scenarios=7)
    scenarios = cgm.generate_validation_scenarios(args)
    _assert_valid_scenarios(scenarios, args)
    assert all(s.raw_unit_sample and len(s.raw_unit_sample) == 6 for s in scenarios)


def test_backend_sobol_scrambled_seed_determinism():
    pytest.importorskip("scipy.stats.qmc")
    from st_lrps.evaluation import compare_gravity_models as cgm

    a = cgm.generate_unit_samples(5, 6, "sobol_scrambled", 42)
    b = cgm.generate_unit_samples(5, 6, "sobol_scrambled", 42)
    c = cgm.generate_unit_samples(5, 6, "sobol_scrambled", 43)
    assert a.shape == (5, 6)
    assert (a == b).all()
    assert not (a == c).all()


def test_backend_lhs_near_circular_and_uniform_cos_stay_in_bounds():
    pytest.importorskip("scipy.stats.qmc")
    from st_lrps.evaluation import compare_gravity_models as cgm

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
    from st_lrps.evaluation import compare_gravity_models as cgm

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


# ---------------------------------------------------------------------------
# Accumulation helpers (CSV reload / numeric coercion)
# ---------------------------------------------------------------------------
def test_metric_row_coercion_and_read(tmp_path):
    from st_lrps.evaluation.compare_gravity_models import _coerce_numeric_row, _read_csv_rows

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
        [sys.executable, "-m", "st_lrps.evaluation.publication_plots", "--help"],
        cwd=str(ROOT), capture_output=True, text=True, timeout=180,
    )
    combined = (help_p.stdout + help_p.stderr).lower()
    if "requires pandas" in combined:
        pytest.skip("pandas not installed")
    assert help_p.returncode == 0
    assert "publication" in combined or "compare_gravity_models" in combined

    noargs_p = subprocess.run(
        [sys.executable, "-m", "st_lrps.evaluation.publication_plots"],
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
        from st_lrps.evaluation.compare_gravity_models import _ReportPager

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
