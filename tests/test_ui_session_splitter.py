"""
Session-restore safety and splitter/collapse stability (objectives 5 & 6).

Pure tests pin the splitter-size sanitizer and the legacy session migration.
Integration tests construct the real ``MainWindow`` (isolated to a temp
app-data dir) to verify collapse/expand/resize/restore behavior and that
corrupt or partial restore data can never produce unusable geometry or block
startup.
"""

from __future__ import annotations

import pytest

pytest.importorskip('PySide6.QtWidgets')


import json
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6 import QtWidgets

from lunaris.ui.core.session_persistence import (
    SESSION_SCHEMA_VERSION,
    apply_visual_state,
    migrate_session_payload,
    sanitize_splitter_sizes,
)
from lunaris.ui.widgets.log_panel import COLLAPSED_HEIGHT, EXPANDED_MIN_HEIGHT


def _app() -> QtWidgets.QApplication:
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


# --------------------------------------------------------------------------- #
# sanitize_splitter_sizes (pure)
# --------------------------------------------------------------------------- #
def test_sanitize_accepts_valid_sizes() -> None:
    assert sanitize_splitter_sizes([500, 250]) == [500, 250]


def test_sanitize_rejects_negative_sizes() -> None:
    assert sanitize_splitter_sizes([-100, 50]) is None


def test_sanitize_rejects_all_zero_sizes() -> None:
    assert sanitize_splitter_sizes([0, 0]) is None


def test_sanitize_rejects_wrong_length() -> None:
    assert sanitize_splitter_sizes([100]) is None
    assert sanitize_splitter_sizes([]) is None


def test_sanitize_rejects_non_numeric_and_non_sequence() -> None:
    assert sanitize_splitter_sizes(["a", "b"]) is None
    assert sanitize_splitter_sizes(None) is None
    assert sanitize_splitter_sizes({"a": 1}) is None


def test_sanitize_scales_oversized_layout_to_fit_window() -> None:
    out = sanitize_splitter_sizes([800, 800], total=600)
    assert out is not None
    assert sum(out) <= 600
    assert all(v >= 1 for v in out)


# --------------------------------------------------------------------------- #
# legacy migration (pure)
# --------------------------------------------------------------------------- #
def test_migrate_legacy_session_folds_gravity_and_bumps_schema() -> None:
    legacy = {"meta": {"schema_version": 1}, "gravity_config": {"degree": 100}}
    migrated = migrate_session_payload(legacy)
    assert migrated["meta"]["schema_version"] == SESSION_SCHEMA_VERSION
    assert migrated["forces"]["gravity"]["config"] == {"degree": 100}


def test_migrate_handles_non_dict_payload() -> None:
    out = migrate_session_payload(["not", "a", "dict"])  # type: ignore[arg-type]
    assert out["meta"]["schema_version"] == SESSION_SCHEMA_VERSION


def test_apply_visual_state_tolerates_missing_and_partial() -> None:
    # No main window, empty / None / partial dicts must never raise.
    apply_visual_state(None, main_window=None)  # type: ignore[arg-type]
    apply_visual_state({}, main_window=None)
    apply_visual_state({"telemetry_plot_type": "Velocity vs Time"}, main_window=None)


# --------------------------------------------------------------------------- #
# MainWindow integration (isolated app-data dir)
# --------------------------------------------------------------------------- #
@pytest.fixture()
def main_window(tmp_path, monkeypatch):
    monkeypatch.setenv("LUNARIS_APP_DATA_DIR", str(tmp_path / "appdata"))
    app = _app()
    from lunaris.ui.app import MainWindow

    win = MainWindow()
    win.resize(1200, 900)
    win.show()
    app.processEvents()
    yield win, app
    win.close()
    win.deleteLater()
    app.processEvents()


def test_collapse_expand_resize_restore_sequence(main_window) -> None:
    win, app = main_window

    # Establish an expanded baseline with a known geometry.
    if win.log_panel.is_collapsed:
        win._toggle_log_collapsed()
        app.processEvents()
    assert win.log_panel.is_collapsed is False
    win.main_splitter.setSizes([520, 260])
    app.processEvents()

    # Collapse: console shrinks to the header height (never zero).
    win._toggle_log_collapsed()
    app.processEvents()
    assert win.log_panel.is_collapsed is True
    collapsed = win.main_splitter.sizes()
    assert 0 < collapsed[1] <= COLLAPSED_HEIGHT + 4
    remembered = list(win._log_expanded_sizes)
    assert remembered[1] >= EXPANDED_MIN_HEIGHT

    # Resizing while collapsed must not destroy the remembered expanded height.
    win.resize(1000, 760)
    app.processEvents()
    assert list(win._log_expanded_sizes) == remembered

    # Expanding restores a usable (non-trivial) console height.
    win._toggle_log_collapsed()
    app.processEvents()
    assert win.log_panel.is_collapsed is False
    restored = win.main_splitter.sizes()
    assert restored[1] >= EXPANDED_MIN_HEIGHT


def test_corrupt_splitter_sizes_never_produce_unusable_geometry(main_window) -> None:
    win, app = main_window
    for bad in ([-100, -50], [0, 0], [999999, 999999]):
        apply_visual_state({"splitter_sizes": bad}, main_window=win)
        app.processEvents()
        sizes = win.main_splitter.sizes()
        assert all(s >= 0 for s in sizes)
        assert sum(sizes) > 0


def test_invalid_active_page_key_is_ignored(main_window) -> None:
    win, app = main_window
    before = win.nav_list.currentRow()
    apply_visual_state({"active_page_key": "NoSuchPage"}, main_window=win)
    app.processEvents()
    assert win.nav_list.currentRow() == before


def test_visual_state_restores_collapsed_then_expanded(main_window) -> None:
    win, app = main_window
    apply_visual_state({"log_collapsed": True}, main_window=win)
    app.processEvents()
    assert win.log_panel.is_collapsed is True
    apply_visual_state({"log_collapsed": False}, main_window=win)
    app.processEvents()
    assert win.log_panel.is_collapsed is False


# --------------------------------------------------------------------------- #
# Startup resilience against missing / corrupt session files
# --------------------------------------------------------------------------- #
def test_missing_session_file_allows_startup(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("LUNARIS_APP_DATA_DIR", str(tmp_path / "fresh"))
    app = _app()
    from lunaris.ui.app import MainWindow

    win = MainWindow()  # must not raise
    try:
        assert win.log_panel is not None
    finally:
        win.close()
        win.deleteLater()
        app.processEvents()


def test_fresh_start_keeps_terminal_collapsed(main_window) -> None:
    win, app = main_window
    app.processEvents()
    assert win.log_panel.is_collapsed is True
    assert win.main_splitter.sizes()[1] <= COLLAPSED_HEIGHT + 4


def test_expanded_visual_state_restores_splitter_size(main_window) -> None:
    win, app = main_window
    apply_visual_state(
        {"log_collapsed": False, "splitter_sizes": [620, 260]},
        main_window=win,
    )
    app.processEvents()
    sizes = win.main_splitter.sizes()
    assert win.log_panel.is_collapsed is False
    assert sizes[1] >= EXPANDED_MIN_HEIGHT


def test_corrupt_session_file_allows_startup_without_overwrite(tmp_path, monkeypatch) -> None:
    appdir = tmp_path / "appdata"
    appdir.mkdir(parents=True)
    corrupt = appdir / "studio_session.json"
    corrupt.write_text("{ this is not valid json ", encoding="utf-8")
    monkeypatch.setenv("LUNARIS_APP_DATA_DIR", str(appdir))

    app = _app()
    from lunaris.ui.app import MainWindow

    win = MainWindow()  # startup must survive a corrupt session
    try:
        # The corrupt file must not be silently overwritten during init.
        assert corrupt.read_text(encoding="utf-8") == "{ this is not valid json "
    finally:
        win.close()
        win.deleteLater()
        app.processEvents()


def test_partial_session_file_restores_with_defaults(tmp_path, monkeypatch) -> None:
    appdir = tmp_path / "appdata"
    appdir.mkdir(parents=True)
    # A minimal, schema-light session: unknown/missing fields must be tolerated.
    (appdir / "studio_session.json").write_text(
        json.dumps({"orbit": {"mode": "circular", "alt_km": "75"}, "unknown_field": 1}),
        encoding="utf-8",
    )
    monkeypatch.setenv("LUNARIS_APP_DATA_DIR", str(appdir))

    app = _app()
    from lunaris.ui.app import MainWindow

    win = MainWindow()
    try:
        assert win.page_orbit.btn_mode_circular.isChecked()
    finally:
        win.close()
        win.deleteLater()
        app.processEvents()
