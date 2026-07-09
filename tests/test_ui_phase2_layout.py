"""Phase 2 visual hierarchy and responsive-shell contracts."""

from __future__ import annotations
import pytest
pytest.importorskip('PySide6.QtWidgets')


import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6 import QtWidgets

from lunaris.ui.theme.tokens import DESIGN_TOKENS


def _app() -> QtWidgets.QApplication:
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_mission_shell_compacts_and_exposes_only_relevant_run_actions(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("LUNARIS_APP_DATA_DIR", str(tmp_path / "appdata"))
    from lunaris.ui.app import MainWindow

    app = _app()
    window = MainWindow()
    window.resize(1024, 768)
    window.show()
    app.processEvents()
    try:
        assert window.nav_list.width() == DESIGN_TOKENS.layout.nav_compact_width
        assert window.btn_stop.isHidden()

        window._set_preflight_state("validating")
        app.processEvents()
        assert not window.btn_run.isEnabled()
        assert window.progress_bar.isVisible()
        assert window.btn_stop.isHidden()

        window._set_run_state("running")
        app.processEvents()
        assert window.btn_stop.isVisible()
        assert window.btn_stop.isEnabled()

        window._set_run_state("idle")
        app.processEvents()
        assert window.btn_stop.isHidden()
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()


def test_st_lrps_shell_uses_shared_palette_and_compact_navigation() -> None:
    from lunaris.surrogate.st_lrps.ui.studio_parts import qt_common
    from lunaris.surrogate.st_lrps.ui.studio_parts.main_window import MainWindow
    from lunaris.ui.core.ui_commons import THEME

    assert qt_common.THEME is THEME

    app = _app()
    qt_common.apply_premium_dark_theme(app)
    window = MainWindow()
    window.resize(1024, 768)
    window.show()
    app.processEvents()
    try:
        assert window._sidebar.width() == DESIGN_TOKENS.layout.nav_compact_width
        # The global ST-LRPS experiment header was retired in favour of
        # page-local headers, so the shell no longer mounts one.
        assert window._experiment_header is None
        # The Data workspace section nav is now a compact horizontal strip.
        assert window._data_page._section_nav.maximumHeight() == 50

        window._navigate(2)
        app.processEvents()
        assert window._stack.currentIndex() == 2
        control_bar = window._train_monitor_page.findChild(
            QtWidgets.QFrame, "trainRunBar"
        )
        assert control_bar is not None
        assert control_bar.sizeHint().width() <= window._train_monitor_page.width()
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()


def test_settings_dialogs_share_title_description_and_primary_action() -> None:
    from lunaris.ui.pages.force_models_page import (
        AdaptiveDegreeDialog,
        AlbedoSettingsDialog,
        GravitySettingsDialog,
        UIAdaptiveConfig,
        UIAlbedoConfig,
        UIGravityConfig,
    )
    from lunaris.ui.pages.mission_propagation_page import (
        SolverSettingsDialog,
        SpacecraftBusDialog,
        UISolverConfig,
        UISpacecraftConfig,
    )

    app = _app()
    parent = QtWidgets.QWidget()
    dialogs = [
        GravitySettingsDialog(parent, UIGravityConfig()),
        AdaptiveDegreeDialog(parent, UIAdaptiveConfig()),
        AlbedoSettingsDialog(parent, UIAlbedoConfig()),
        SolverSettingsDialog(parent, UISolverConfig()),
        SpacecraftBusDialog(parent, UISpacecraftConfig()),
    ]
    try:
        for dialog in dialogs:
            assert dialog.objectName() == "settingsDialog"
            assert dialog.findChild(QtWidgets.QLabel, "dialogTitle") is not None
            assert dialog.findChild(QtWidgets.QLabel, "dialogDescription") is not None
            primary = [
                button
                for button in dialog.findChildren(QtWidgets.QPushButton)
                if button.property("kind") == "primary"
            ]
            assert len(primary) == 1
            assert dialog.minimumWidth() <= 1366
            assert dialog.minimumHeight() <= 768
    finally:
        for dialog in dialogs:
            dialog.deleteLater()
        parent.deleteLater()
        app.processEvents()
