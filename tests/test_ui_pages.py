# -*- coding: utf-8 -*-
"""
Widget-level regression tests for the modular UI pages.

These tests sit between the pure helper tests and the heavier full-window smoke
tests.  The goal is to verify that the newer page modules own their state
correctly and keep user-visible interactions stable without needing to boot the
entire desktop workflow for every assertion.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtCore, QtWidgets

from ui_parts.data_files_page import DataFilesState, DataPage
from ui_parts.mission_propagation_page import MissionPropagationPage, UISolverConfig
from ui_parts.monte_carlo_page import MonteCarloPage, UIMonteCarloConfig
from ui_parts.result_exports_page import OutputPageState, ResultsExportPage
from ui_parts.solver_policy import DEFAULT_ADAPTIVE_RTOL, DEFAULT_MAX_STEP_S
from ui_parts.ui_commons import THEME


def _app() -> QtWidgets.QApplication:
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


def _create_card(title: str) -> QtWidgets.QGroupBox:
    return QtWidgets.QGroupBox(title)


def test_data_page_reuses_ldem_root_for_albedo_when_requested(tmp_path: Path) -> None:
    app = _app()
    topo_dir = tmp_path / "topografy_models"
    albedo_dir = tmp_path / "albedo_models"
    topo_dir.mkdir()
    albedo_dir.mkdir()

    page = DataPage(
        project_root=tmp_path,
        normalize_path=lambda text: str(Path(text).expanduser().resolve()),
        log_message=lambda _msg: None,
        create_card=_create_card,
        initial_state=DataFilesState(
            ldem_root=str(topo_dir),
            albedo_root=str(albedo_dir),
            kernel_dir=str(tmp_path / "ephemeris_models"),
            ldem_ppd=16,
            use_ldem_for_albedo=True,
        ),
    )
    page.show()
    app.processEvents()

    state = page.get_state()
    assert state.ldem_root == str(topo_dir)
    assert state.albedo_root == str(topo_dir)
    assert state.use_ldem_for_albedo is True
    assert page.albedo_container.isHidden() is True

    page.chk_use_ldem_for_albedo.setChecked(False)
    page.ent_albedo_root.setText(str(albedo_dir))
    app.processEvents()

    state = page.get_state()
    assert state.albedo_root == str(albedo_dir)
    assert state.use_ldem_for_albedo is False
    assert page.albedo_container.isHidden() is False

    page.close()


def test_results_export_page_restores_state_and_flags_preview_errors(tmp_path: Path) -> None:
    app = _app()
    page = ResultsExportPage(
        project_root=tmp_path,
        create_card=_create_card,
        initial_state=OutputPageState(
            output_dir=str(tmp_path / "mission_results"),
            generate_3d_plots=True,
            downsample_3d=8,
        ),
    )
    page.show()
    app.processEvents()

    assert page.get_state() == OutputPageState(
        output_dir=str(tmp_path / "mission_results"),
        generate_3d_plots=True,
        downsample_3d=8,
    )
    assert page.spin_downsample_3d.isEnabled() is True

    page.set_output_dir(str(tmp_path / "mission_results" / "run_01"))
    page.set_command_preview("invalid command preview", is_error=True)
    page.toggle_anim3d.setChecked(False)
    app.processEvents()

    state = page.get_state()
    assert state.output_dir.endswith("run_01")
    assert state.generate_3d_plots is False
    assert page.spin_downsample_3d.isEnabled() is False
    assert page.txt_preview.toPlainText() == "invalid command preview"
    assert THEME["error"] in page.txt_preview.styleSheet()

    page.close()


def test_mission_propagation_page_normalizes_legacy_integrator_values() -> None:
    app = _app()
    solver_cfg = UISolverConfig(
        rtol=DEFAULT_ADAPTIVE_RTOL,
        atol=1e-12,
        max_step=DEFAULT_MAX_STEP_S,
    )
    mission_epoch = QtCore.QDateTime.fromString(
        "2026-05-10T16:19:47Z",
        QtCore.Qt.DateFormat.ISODate,
    )
    page = MissionPropagationPage(solver_cfg=solver_cfg, mission_epoch=mission_epoch)
    page.show()
    app.processEvents()

    assert page.dt_epoch.displayFormat() == "yyyy-MM-dd HH:mm:ss 'UTC'"
    assert page.to_dict()["timeline"]["epoch"] == "2026-05-10T16:19:47Z"

    page.apply_dict(
        {
            "timeline": {
                "epoch": "2027-03-03T02:32:37+03:00",
                "duration": "12",
                "unit": "Hours",
            },
            "integrator": {
                "method": "DOP853 (Adaptive)",
                "rtol": "0",
                "dt_out": "30",
                "max_step": "0",
            },
        }
    )
    app.processEvents()

    assert page.ent_rtol.text() == f"{DEFAULT_ADAPTIVE_RTOL:g}"
    assert page.ent_max_step.text() == f"{DEFAULT_MAX_STEP_S:g}"
    assert page.to_dict()["timeline"]["unit"] == "Hours"
    assert page.to_dict()["timeline"]["epoch"] == "2027-03-02T23:32:37Z"

    page.apply_dict({"integrator": {"method": "YOSHIDA4 (Symplectic)"}})
    app.processEvents()

    assert page.cb_integrator.currentText() == "YOSHIDA4 (Symplectic)"
    assert page.tolerance_group.isHidden() is True

    page.close()


def test_monte_carlo_page_restores_state_and_renders_structured_progress() -> None:
    app = _app()
    page = MonteCarloPage(mc_cfg=UIMonteCarloConfig(use_gpu=False))
    page.show()
    app.processEvents()

    page.load_data(
        {
            "n_samples": 500,
            "seed": 7,
            "sigma_r_m": 250.0,
            "sigma_v_m_s": 0.25,
            "use_gpu": False,
            "gravity_mode_override": "st_lrps",
            "output_format": "npz",
            "output_path": "mc_results/legacy_output.h5",
            "impact_alt_km": 2.0,
        }
    )
    app.processEvents()

    state = page.get_data()
    assert state["output_format"] == "npz"
    assert state["output_path"].endswith(".npz")
    assert state["n_samples"] == 500
    assert state["use_gpu"] is False
    assert state["gravity_mode_override"] == "st_lrps"

    page.update_progress_payload(
        {
            "stage": "propagating",
            "percent": 47.5,
            "fraction": 0.475,
            "done_samples": 237.5,
            "total_samples": 500,
            "batch_index": 2,
            "batch_count": 4,
            "eta_s": 136.0,
            "backend": "CPU",
        }
    )
    app.processEvents()

    assert page.progress_mc.format() == "47.5%"
    assert page.lbl_progress_summary.text() == "Propagating scenarios (CPU)"
    assert page.lbl_progress_meta.text() == "~237 / 500 scenarios | Batch 2/4 | ETA 02:16"

    page.close()
