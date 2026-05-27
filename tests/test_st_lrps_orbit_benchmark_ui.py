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

import importlib
import os
import sys
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

    return QApplication.instance() or QApplication([])


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
    ]
    monkeypatch.setattr(_sys, "argv", argv)
    args = cgm.parse_args()
    assert args.gpu_integrator == "robust"
    assert args.truth_integrator == "RK45"
    assert args.workers == 4


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


def test_ui_add_custom_model(qapp):
    from st_lrps.ui.studio import OrbitBenchmarkTab

    tab = OrbitBenchmarkTab()
    ok, err = tab._try_add_model("sh45")
    assert ok and err == ""
    assert "sh45" in tab._model_checks and tab._model_checks["sh45"].isChecked()
    tab.run_mode.setCurrentIndex(tab.run_mode.findData("gpu_rk4"))
    args = tab._build_args(show_errors=False)
    assert "sh45" in args[args.index("--gpu-models") + 1].split(",")
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
