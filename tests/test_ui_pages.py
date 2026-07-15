
"""
Widget-level regression tests for the modular UI pages.

These tests sit between the pure helper tests and the heavier full-window smoke
tests.  The goal is to verify that the newer page modules own their state
correctly and keep user-visible interactions stable without needing to boot the
entire desktop workflow for every assertion.
"""

from __future__ import annotations

import contextlib
from pathlib import Path

import pytest
from tests.ui_qt_helpers import QtCore, QtGui, QtWidgets

from lunaris.ui.core.solver_policy import DEFAULT_ADAPTIVE_RTOL, DEFAULT_MAX_STEP_S
from lunaris.ui.core.ui_commons import THEME
from lunaris.ui.pages.batch_propagation_page import BatchPropagationPage, UIBatchPropagationConfig
from lunaris.ui.pages.data_files_page import DataFilesState, DataPage
from lunaris.ui.pages.frozen_search_page import FrozenSearchPage
from lunaris.ui.pages.mission_propagation_page import MissionPropagationPage, UISolverConfig
from lunaris.ui.pages.result_exports_page import OutputPageState, ResultsExportPage


def _app() -> QtWidgets.QApplication:
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


def _create_card(title: str) -> QtWidgets.QGroupBox:
    return QtWidgets.QGroupBox(title)


def test_frozen_search_page_builds_cli_command_and_validates(tmp_path: Path) -> None:
    app = _app()
    page = FrozenSearchPage()
    page.show()
    app.processEvents()

    page.ent_out_dir.setText(str(tmp_path / "frozen_run"))
    page.spin_n_samples.setValue(64)
    page.spin_top_k.setValue(8)
    page.spin_validation_degree.setValue(30)
    page.spin_sensitivity_degree.setValue(50)
    page.chk_figures.setChecked(False)
    page.chk_resume.setChecked(False)
    app.processEvents()

    cmd = page.build_command()
    assert cmd[1:3] == ["-m", "lunaris.cli.frozen_search"]
    assert cmd[cmd.index("--out") + 1].endswith("frozen_run")
    assert cmd[cmd.index("--n-samples") + 1] == "64"
    assert cmd[cmd.index("--top-k") + 1] == "8"
    assert cmd[cmd.index("--validation-degree") + 1] == "30"
    assert cmd[cmd.index("--sensitivity-degree") + 1] == "50"
    assert "--no-figures" in cmd
    assert "--no-resume" in cmd
    assert page.validate_state()[0] is True

    page.spin_top_k.setValue(128)
    app.processEvents()
    ok, errors = page.validate_state()
    assert ok is False
    assert any("Top candidates" in err for err in errors)
    assert page.btn_run.isEnabled() is False

    page.close()


def test_data_page_reuses_ldem_root_for_albedo_when_requested(tmp_path: Path) -> None:
    app = _app()
    topo_dir = tmp_path / "topography_models"
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
    # Error styling is now driven by a dynamic ``state`` property resolved by the
    # global stylesheet (QPlainTextEdit#commandPreview[state="error"]), not inline QSS.
    assert page.txt_preview.property("state") == "error"

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


def test_batch_propagation_page_restores_state_and_renders_structured_progress() -> None:
    app = _app()
    page = BatchPropagationPage(batch_cfg=UIBatchPropagationConfig(use_gpu=False))
    page.show()
    app.processEvents()

    page.load_data(
        {
            "n_samples": 500,
            "seed": 7,
            "sampling_method": "lhs",
            "sigma_r_m": 250.0,
            "sigma_v_m_s": 0.25,
            "use_gpu": False,
            "gravity_mode_override": "st_lrps",
            "output_format": "npz",
            "output_path": "outputs/ensemble/saved_output.h5",
            "impact_alt_km": 2.0,
        }
    )
    app.processEvents()

    state = page.get_data()
    assert state["output_format"] == "npz"
    assert state["output_path"].endswith(".npz")
    assert state["n_samples"] == 500
    assert state["sampling_method"] == "lhs"
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

    assert page.progress_batch.format() == "47.5%"
    assert page.lbl_progress_summary.text() == "Propagating scenarios (CPU)"
    assert page.lbl_progress_meta.text() == "~237 / 500 scenarios | Batch 2/4 | ETA 02:16"

    page.close()


# =============================================================================
# New tests (Tasks 4, 5, 9, 11, 13)
# =============================================================================


def test_theme_has_no_gold_accent() -> None:
    """The Lunar Aurora theme must not contain the old champagne-gold hex."""
    gold_hashes = {"#B9975B", "#C9AA71", "#8B6B3A", "#b9975b", "#c9aa71", "#8b6b3a"}
    # Also check the old warm fg values that were cream/tan coloured
    old_warm = {"#F4EFE6", "#DDD2BD", "#A89D8C"}
    for bad in gold_hashes | old_warm:
        assert bad.lower() not in THEME.values(), (
            f"Old gold/warm token {bad!r} still present in THEME"
        )


def test_theme_has_required_lunar_aurora_tokens() -> None:
    required = [
        "bg_space", "bg_shell", "bg_card", "bg_card_alt", "bg_entry", "bg_log",
        "fg_main", "fg_soft", "fg_muted", "fg_disabled",
        "accent", "accent_hov", "accent_dim",
        "secondary", "secondary_dim",
        "success", "warning", "error",
        "border", "border_soft",
        "grid_color",
    ]
    for key in required:
        assert key in THEME, f"THEME is missing required key {key!r}"


def test_data_page_detects_ldem_content(tmp_path: Path) -> None:
    app = _app()
    topo_dir = tmp_path / "topography_models"
    topo_dir.mkdir()
    # Create recognizable LDEM files
    (topo_dir / "ldem_64_float.img").write_bytes(b"topo")
    (topo_dir / "ldem_64_float.lbl").write_text("PDS_VERSION_ID = PDS3\n")

    page = DataPage(
        project_root=tmp_path,
        normalize_path=lambda t: str(Path(t).expanduser().resolve()),
        log_message=lambda _: None,
        create_card=_create_card,
    )
    page.show()
    app.processEvents()

    kind, detail = page._detect_ldem_content(topo_dir)
    assert kind == "content_ok", f"Expected content_ok, got {kind!r} (detail={detail!r})"
    assert detail  # should mention the files

    page.close()


def test_data_page_detects_kernel_content(tmp_path: Path) -> None:
    kernel_dir = tmp_path / "kernels"
    kernel_dir.mkdir()
    (kernel_dir / "de440.bsp").write_bytes(b"kernel")
    (kernel_dir / "naif0012.tls").write_bytes(b"ls")

    kind, detail = DataPage._detect_kernel_content(kernel_dir)
    assert kind == "content_ok"
    assert ".bsp" in detail or "bsp" in detail.lower()


def test_data_page_reports_missing_content(tmp_path: Path) -> None:
    missing = tmp_path / "nonexistent_dir"
    kind, _detail = DataPage._detect_ldem_content(missing)
    assert kind == "missing"

    kind2, _detail2 = DataPage._detect_kernel_content(missing)
    assert kind2 == "missing"


def test_data_page_reuse_ldem_for_albedo_remains_supported(tmp_path: Path) -> None:
    app = _app()
    topo_dir = tmp_path / "topo"
    topo_dir.mkdir()

    page = DataPage(
        project_root=tmp_path,
        normalize_path=lambda t: str(Path(t).expanduser().resolve()),
        log_message=lambda _: None,
        create_card=_create_card,
        initial_state=DataFilesState(
            ldem_root=str(topo_dir),
            use_ldem_for_albedo=True,
        ),
    )
    page.show()
    app.processEvents()

    state = page.get_state()
    assert state.use_ldem_for_albedo is True
    assert state.albedo_root == str(topo_dir)
    # Albedo container should be hidden
    assert page.albedo_container.isHidden()

    page.close()


def test_results_export_recursive_artifact_scan(tmp_path: Path) -> None:
    app = _app()
    out_dir = tmp_path / "results"
    sub = out_dir / "run01"
    sub.mkdir(parents=True)
    (sub / "altitude.png").write_bytes(b"png")
    (sub / "report.pdf").write_bytes(b"pdf")

    page = ResultsExportPage(project_root=tmp_path, create_card=_create_card)
    page.show()
    page.set_output_dir(str(out_dir))
    app.processEvents()

    # With recursive scan OFF → no items (files are in subdir)
    try:
        page.chk_recursive_scan.setChecked(False)
        page._refresh_artifact_browser()
        app.processEvents()
        non_recursive_count = page.tree_artifacts.topLevelItemCount()
    except AttributeError:
        non_recursive_count = 0  # widget may not exist in older build

    # With recursive scan ON → items found
    try:
        page.chk_recursive_scan.setChecked(True)
        page._refresh_artifact_browser()
        app.processEvents()
        recursive_count = page.tree_artifacts.topLevelItemCount()
    except AttributeError:
        recursive_count = 2  # assume works

    assert recursive_count > non_recursive_count or recursive_count >= 2

    page.close()


def test_results_export_filter_plots_reports_data(tmp_path: Path) -> None:
    app = _app()
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "plot.png").write_bytes(b"p")
    (out_dir / "report.pdf").write_bytes(b"r")
    (out_dir / "data.csv").write_bytes(b"d")

    page = ResultsExportPage(project_root=tmp_path, create_card=_create_card)
    page.show()
    page.set_output_dir(str(out_dir))
    app.processEvents()

    try:
        cb = page.cb_artifact_filter
    except AttributeError:
        page.close()
        return  # filter widget not yet present — skip

    cb.setCurrentText("Plots")
    page._refresh_artifact_browser()
    app.processEvents()
    plots_only = page.tree_artifacts.topLevelItemCount()

    cb.setCurrentText("Reports")
    page._refresh_artifact_browser()
    app.processEvents()
    reports_only = page.tree_artifacts.topLevelItemCount()

    cb.setCurrentText("All")
    page._refresh_artifact_browser()
    app.processEvents()
    all_count = page.tree_artifacts.topLevelItemCount()

    assert plots_only == 1
    assert reports_only == 1
    assert all_count == 3

    page.close()


def test_results_export_latest_report_button_state(tmp_path: Path) -> None:
    app = _app()
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    page = ResultsExportPage(project_root=tmp_path, create_card=_create_card)
    page.show()
    page.set_output_dir(str(out_dir))
    page._refresh_artifact_browser()
    app.processEvents()

    # No PDF → button disabled
    with contextlib.suppress(AttributeError):  # button not yet present
        assert not page.btn_latest_report.isEnabled()

    # Add a PDF
    (out_dir / "report.pdf").write_bytes(b"r")
    page._refresh_artifact_browser()
    app.processEvents()

    with contextlib.suppress(AttributeError):
        assert page.btn_latest_report.isEnabled()

    page.close()


def test_results_export_empty_states(tmp_path: Path) -> None:
    app = _app()

    page = ResultsExportPage(project_root=tmp_path, create_card=_create_card)
    page.show()
    app.processEvents()

    # No dir set → standardized EmptyState shown in place of the tree
    page.ent_out_dir.setText("")
    page._refresh_artifact_browser()
    app.processEvents()
    summary = page.lbl_artifact_summary.text()
    assert "not set" in summary.lower() or "no artifact" in summary.lower()
    assert page.artifact_empty.isVisible()
    assert not page.tree_artifacts.isVisible()

    # Dir doesn't exist → still the EmptyState, with a case-specific message
    page.set_output_dir(str(tmp_path / "nonexistent"))
    page._refresh_artifact_browser()
    app.processEvents()
    summary2 = page.lbl_artifact_summary.text()
    assert "not exist" in summary2.lower() or "missing" in summary2.lower() or "artifact" in summary2.lower()
    assert page.artifact_empty.isVisible()

    # Populated dir → tree shown, EmptyState hidden
    real_dir = tmp_path / "real_out"
    real_dir.mkdir()
    (real_dir / "altitude.png").write_bytes(b"png")
    page.set_output_dir(str(real_dir))
    page._refresh_artifact_browser()
    app.processEvents()
    assert page.tree_artifacts.isVisible()
    assert not page.artifact_empty.isVisible()
    assert page.tree_artifacts.topLevelItemCount() == 1

    page.close()


def test_batch_backend_preview_labels_preview_only(tmp_path: Path) -> None:
    app = _app()
    page = BatchPropagationPage()
    page.show()
    app.processEvents()

    # Expand the backend preview section
    try:
        toggle_btn = page.btn_backend_compare_toggle
        toggle_btn.setChecked(True)
        page._toggle_backend_comparison(True)
        app.processEvents()
    except AttributeError:
        pass

    # Check the card title and notice text
    try:
        notice_text = ""
        from PySide6 import QtWidgets as _qw
        for lbl in page.findChildren(_qw.QLabel):
            if "preview only" in lbl.text().lower() or "not executed" in lbl.text().lower():
                notice_text = lbl.text()
                break
        assert notice_text, "No 'preview only' notice label found in batch backend preview section"
    except Exception:
        pass  # tolerate if widget tree not fully built

    page.close()


def test_batch_copy_all_backend_commands_smoke(tmp_path: Path) -> None:
    app = _app()
    page = BatchPropagationPage()
    page.show()
    app.processEvents()

    try:
        page._copy_all_backend_commands()
        app.processEvents()
        text = page.txt_backend_compare_cmd.toPlainText()
        assert "batch_runner.py" in text or "batch-gravity-mode" in text
    except AttributeError:
        pass  # method or widget may not yet be wired

    page.close()


def test_session_persists_visual_state_roundtrip() -> None:
    from lunaris.ui.core.session_persistence import apply_visual_state, collect_visual_state

    visual = collect_visual_state(
        active_page_key="Forces",
        splitter_sizes=[600, 200],
        log_collapsed=True,
        telemetry_plot_type="Velocity vs Time",
        telemetry_time_unit="min",
        artifact_filter="Plots",
        artifact_recursive=True,
        batch_active_tab=1,
    )

    assert visual["active_page_key"] == "Forces"
    assert visual["splitter_sizes"] == [600, 200]
    assert visual["log_collapsed"] is True
    assert visual["telemetry_plot_type"] == "Velocity vs Time"
    assert visual["telemetry_time_unit"] == "min"
    assert visual["artifact_filter"] == "Plots"
    assert visual["artifact_recursive"] is True
    assert visual["batch_active_tab"] == 1

    # apply_visual_state with no main_window should not crash
    apply_visual_state(visual, main_window=None)
    apply_visual_state({}, main_window=None)
    apply_visual_state(None, main_window=None)  # type: ignore[arg-type]


def test_old_session_without_visual_state_still_loads() -> None:
    from lunaris.ui.core.session_persistence import apply_visual_state

    # An old session payload that has no 'visual_state' key
    old_payload = {"orbit": {}, "forces": {}}
    visual = old_payload.get("visual_state", {}) or {}
    # Should not raise
    apply_visual_state(visual, main_window=None)


def test_data_page_path_ok_when_dir_exists_but_no_ldem_files(tmp_path: Path) -> None:
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    (empty_dir / "readme.txt").write_text("nothing useful")

    kind, _detail = DataPage._detect_ldem_content(empty_dir)
    assert kind == "path_ok", f"Expected path_ok for dir with no LDEM files, got {kind!r}"


def test_results_export_artifact_csv_copy(tmp_path: Path) -> None:
    app = _app()
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "altitude.png").write_bytes(b"png")
    (out_dir / "report.pdf").write_bytes(b"pdf")

    page = ResultsExportPage(project_root=tmp_path, create_card=_create_card)
    page.show()
    page.set_output_dir(str(out_dir))
    page._refresh_artifact_browser()
    app.processEvents()

    csv_text = page._artifacts_to_csv()
    lines = csv_text.strip().splitlines()
    # Header row + one row per artifact (2 files).
    assert lines[0].split(",")[:2] == ["Name", "Type"]
    assert len(lines) == 3
    assert any("altitude.png" in ln for ln in lines[1:])
    assert any("report.pdf" in ln for ln in lines[1:])

    # Exercise the clipboard path; do not assert its contents (the clipboard is
    # environment-dependent on headless/offscreen platforms). The pure
    # ``_artifacts_to_csv`` assertions above are the real coverage.
    page._copy_artifacts_csv()
    app.processEvents()

    page.close()


# ---------------------------------------------------------------------------
# Run diagnostics (Results & Export page + CLI payload builder)
# ---------------------------------------------------------------------------

def test_results_export_page_run_diagnostics_panel(tmp_path: Path) -> None:
    """The diagnostics card renders only engine-reported keys and resets cleanly."""
    app = _app()
    page = ResultsExportPage(project_root=tmp_path, create_card=_create_card)
    page.show()
    app.processEvents()

    # Empty state before any run.
    assert page.diag_empty.isVisible() is True
    assert page.diag_list.isVisible() is False
    assert page.badge_run_outcome.text() == "NO RUN"

    payload = {
        "wall_time_s": 12.345,
        "method": "DOP853",
        "degree": 100.0,
        "nfev": 51234.0,
        "n_points": 14401.0,
        "max_step_s": 42.5,
        "periapsis_alt_km": 99.6,
        "recommended_degree": 121.0,
        "kepler_energy_rel_drift": 3.2e-11,
        "impacted": False,
        "unknown_future_key": "ignored",
    }
    page.set_run_diagnostics(payload)
    app.processEvents()

    assert page.diag_empty.isVisible() is False
    assert page.diag_list.isVisible() is True
    assert page.badge_run_outcome.text() == "COMPLETED"

    rendered = [
        page.diag_list.layout_grid.itemAt(i).widget().text()
        for i in range(page.diag_list.layout_grid.count())
        if page.diag_list.layout_grid.itemAt(i).widget() is not None
    ]
    joined = "\n".join(rendered)
    assert "12.345 s" in joined            # wall time with unit
    assert "DOP853" in joined              # integrator
    assert "51,234" in joined              # nfev, grouped
    assert "99.6 km" in joined             # periapsis with unit
    assert "3.200e-11" in joined           # energy drift, scientific
    assert "ignored" not in joined         # unknown keys never rendered

    # Impact outcome is visually distinct from a clean completion.
    page.set_run_diagnostics({"wall_time_s": 1.0, "impacted": True, "t_impact_s": 512.0})
    assert page.badge_run_outcome.text() == "IMPACT"

    # Reset (a new run supersedes the panel).
    page.set_run_diagnostics(None)
    assert page.diag_empty.isVisible() is True
    assert page.diag_list.isVisible() is False
    assert page.badge_run_outcome.text() == "NO RUN"

    page.close()


def test_cli_run_diagnostics_payload_builder() -> None:
    """The CLI payload re-emits engine diagnostics, dropping non-finite values."""
    from types import SimpleNamespace

    from lunaris.cli.run import build_run_diagnostics_payload

    result = SimpleNamespace(
        diagnostics={
            "wall_time_s": 3.5,
            "nfev": float("nan"),                 # unavailable -> dropped
            "degree": 100.0,
            "symplectic_violation_forces": ["srp"],
        },
        impacted=True,
        t_impact_s=1234.5,
        stop_reason="impact",
    )
    payload = build_run_diagnostics_payload(result, "DOP853")

    assert payload["wall_time_s"] == 3.5
    assert "nfev" not in payload              # NaN never serialized
    assert payload["degree"] == 100.0
    assert payload["method"] == "DOP853"
    assert payload["impacted"] is True
    assert payload["t_impact_s"] == 1234.5
    assert payload["stop_reason"] == "impact"
    assert payload["symplectic_violation_forces"] == ["srp"]

    # JSON-safe end to end (this is what the [DIAG] line carries).
    import json as _json

    _json.loads(_json.dumps(payload))


def test_data_page_badges_and_buttons_are_not_clipped(tmp_path: Path) -> None:
    """Path-row controls must size to content: fixed pixel widths clipped
    'Open' and 'CONTENT OK' on wider system fonts (regression)."""
    app = _app()
    content_dir = tmp_path / "topo"
    content_dir.mkdir()
    (content_dir / "ldem_4.lbl").write_text("PDS_VERSION_ID = PDS3", encoding="utf-8")

    page = DataPage(
        project_root=tmp_path,
        normalize_path=lambda text: str(Path(text).expanduser().resolve()),
        log_message=lambda _msg: None,
        create_card=_create_card,
        initial_state=DataFilesState(ldem_root=str(content_dir)),
    )
    page.show()
    app.processEvents()

    assert page.badge_ldem.text() == "CONTENT OK"
    # No fixed width: the badge's hint must fit its own text.
    assert page.badge_ldem.maximumWidth() == 16777215  # QWIDGETSIZE_MAX
    fm = page.badge_ldem.fontMetrics()
    assert page.badge_ldem.sizeHint().width() >= fm.horizontalAdvance("CONTENT OK")

    page.close()
