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
