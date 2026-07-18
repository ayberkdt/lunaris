"""Phase 2 visual hierarchy and responsive-shell contracts."""

from __future__ import annotations

import pytest
from tests.ui_qt_helpers import QtCore, QtGui, QtWidgets, wait_until

torch = pytest.importorskip('torch')


from lunaris.ui.theme.tokens import DESIGN_TOKENS


def _app() -> QtWidgets.QApplication:
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.mark.parametrize("size", [(1024, 768), (1280, 860)])
def test_batch_setup_uses_one_primary_scroll_owner(size) -> None:
    from lunaris.ui.pages.batch_propagation_page import BatchPropagationPage

    app = _app()
    page = BatchPropagationPage()
    page.resize(*size)
    page.show()
    app.processEvents()
    try:
        visible_scrolls = [
            scroll
            for scroll in page.findChildren(QtWidgets.QScrollArea)
            if scroll.isVisibleTo(page)
        ]
        assert visible_scrolls == [page._run_scroll]
        assert page._run_scroll.horizontalScrollBar().maximum() == 0
        assert page._run_action_strip.isVisibleTo(page)
        assert page.btn_run_batch.isVisibleTo(page)
    finally:
        page.close()
        page.deleteLater()
        app.processEvents()


@pytest.mark.parametrize("size", [(1024, 768), (1280, 860)])
def test_frozen_search_uses_one_primary_scroll_owner(size) -> None:
    from lunaris.ui.pages.frozen_search_page import FrozenSearchPage

    app = _app()
    page = FrozenSearchPage()
    # The page sits inside Mission Studio's navigation/header chrome; exercise
    # the actual content budget produced by the requested desktop viewport.
    page.resize(size[0] - 280, size[1] - 140)
    page.show()
    app.processEvents()
    try:
        visible_scrolls = [
            scroll
            for scroll in page.findChildren(QtWidgets.QScrollArea)
            if scroll.isVisibleTo(page)
        ]
        assert visible_scrolls == [page._workspace_scroll]
        assert page._workspace_scroll.horizontalScrollBar().maximum() == 0
        assert page._action_strip.isVisibleTo(page)
        assert page.btn_run.isVisibleTo(page)
        assert page.btn_cancel.isVisibleTo(page)
        assert page.txt_command.isHidden()
        assert page._workspace_columns.stacked is (size == (1024, 768))
    finally:
        page.close()
        page.deleteLater()
        app.processEvents()


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


def test_mission_preflight_blocks_inverted_apsides(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("LUNARIS_APP_DATA_DIR", str(tmp_path / "appdata"))
    from lunaris.ui.app import MainWindow

    app = _app()
    window = MainWindow()
    window.resize(1024, 768)
    window.show()
    app.processEvents()
    try:
        window.page_orbit.ent_hp.setText("200")
        window.page_orbit.ent_ha.setText("100")
        window._start_preflight_validation()
        app.processEvents()

        assert window.preflight_worker is None
        assert window.nav_list.currentRow() == window._page_map["Orbit"]
        assert window.page_orbit.ent_hp.property("fieldError") is True
        assert window.page_orbit.ent_ha.property("fieldError") is True
        assert window.page_orbit.ent_hp.selectedText() == "200"
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()


def test_header_never_clips_its_load_bearing_items_during_a_run(
    tmp_path, monkeypatch
) -> None:
    """A run must not squeeze header labels into half-rendered glyphs.

    Starting a run adds ~450px of chrome (progress bar, run-state chip, Stop)
    to a header whose items are otherwise fixed. Qt answers that by shrinking
    every widget below its size hint, which clipped the page badge from both
    ends ("PROPAGATION" -> "ROPAGATIO") and chopped "Run Analysis". The shell
    must instead shed the optional context chips and elide the two labels that
    are allowed to elide.
    """
    monkeypatch.setenv("LUNARIS_APP_DATA_DIR", str(tmp_path / "appdata"))
    from lunaris.ui.app import MainWindow

    app = _app()
    window = MainWindow()
    window.resize(1280, 860)
    window.show()
    app.processEvents()
    try:
        window._set_run_state("running")
        window.progress_bar.show()
        window.lbl_progress.show()
        window.lbl_progress.setText("Propagating trajectory...")
        app.processEvents()

        # Everything still on screen must have at least the width its own size
        # hint asks for; anything narrower is rendering a clipped label.
        for widget in (window.badge_page, window.btn_run, window.btn_stop):
            assert not widget.isHidden()
            assert widget.width() >= widget.sizeHint().width(), (
                f"{widget.objectName() or widget} is clipped: "
                f"{widget.width()}px < {widget.sizeHint().width()}px hint"
            )

        # The header must have made room by shedding, not by clipping.
        assert window._header_required_width() <= window._header_frame.width()

        def hidden_chips() -> set[str]:
            return {
                chip.accessibleName() or chip.text()
                for chip in window._header_optional_chips()
                if chip.isHidden()
            }

        shed_at_1280 = hidden_chips()

        # Squeezing further sheds more, never fewer. No required<=available
        # assertion at this width: below ~1100px the fixed items genuinely do
        # not fit even with every optional chip gone, and how far below depends
        # on the UI font (the offscreen test platform measures wider than the
        # real one). The contract here is the shed *ordering*, not that an
        # arbitrarily narrow header can fit everything.
        window.resize(1000, 860)
        assert wait_until(app, lambda: hidden_chips() > shed_at_1280)
        shed_at_1000 = hidden_chips()
        assert shed_at_1000 >= shed_at_1280

        # ...and widening gives them back: shedding is a response to pressure,
        # not a one-way door. Asserted as a strict subset rather than "nothing
        # hidden at width W", because the exact width at which the last chip
        # fits depends on the length of the real output path rendered in it.
        # Pumped via wait_until: the re-show is driven by the header's own
        # Resize/LayoutRequest events, delivered over several loop cycles.
        window.resize(1920, 900)
        assert wait_until(app, lambda: hidden_chips() < shed_at_1000), (
            "widening the header did not restore any shed chip: still hidden="
            f"{hidden_chips()}, required={window._header_required_width()}, "
            f"available={window._header_frame.width()}"
        )
        # The highest-priority optional chip is always the first one back.
        assert not window.lbl_gravity_status.isHidden()
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()


def test_elided_label_shows_ellipsis_and_keeps_full_text() -> None:
    """The elide primitive must degrade to "…" and never lose the real value."""
    from lunaris.ui.components import ElidedLabel

    _app()
    label = ElidedLabel("Lunaris Mission Studio")
    # Reports the full text as its hint, but is willing to shrink to nothing so
    # a crowded layout squeezes it rather than its fixed-size neighbours.
    assert label.sizeHint().width() > 0
    assert label.minimumSizeHint().width() == 0
    # The unelided string stays reachable no matter how narrow the paint is.
    assert label.full_text() == "Lunaris Mission Studio"
    assert label.toolTip() == "" or "Lunaris" in label.toolTip()
    label.setText("Another Title")
    assert label.full_text() == "Another Title"
    assert label.toolTip() == "Another Title"


def test_settings_dialogs_are_tall_enough_for_their_styled_content(
    tmp_path, monkeypatch
) -> None:
    """Every settings dialog must fit its content at its default size.

    A dialog shorter than its layout's size hint does not scroll — Qt
    compresses the children past their minimum until they overlap. That is what
    made the gravity dialog paint its hint text on top of its combobox, sliced
    the albedo facet-resolution row in half, and crushed the adaptive-degree
    table to a ~20px sliver with its header text cut through the middle.

    Parented to a real MainWindow on purpose: the app stylesheet is set on the
    window and inherited down the object tree, and it is what inflates control
    heights. Measured against a bare QWidget parent these dialogs look like
    they fit by ~200px, which is exactly how this defect survived review.

    The real UI font is loaded for the same reason: the offscreen platform's
    default font measures wider than Segoe UI, so without this the test sizes
    the dialogs against a font no user ever sees.
    """
    monkeypatch.setenv("LUNARIS_APP_DATA_DIR", str(tmp_path / "appdata"))
    from lunaris.ui.app import MainWindow
    from lunaris.ui.core.ui_commons import load_fonts
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
    app.setFont(load_fonts())
    window = MainWindow()
    window.resize(1280, 860)
    window.show()
    app.processEvents()
    dialogs = {
        "Gravity": GravitySettingsDialog(window, UIGravityConfig()),
        "AdaptiveDegree": AdaptiveDegreeDialog(window, UIAdaptiveConfig()),
        "Albedo": AlbedoSettingsDialog(window, UIAlbedoConfig()),
        "SolverSettings": SolverSettingsDialog(window, UISolverConfig()),
        "SpacecraftBus": SpacecraftBusDialog(window, UISpacecraftConfig()),
    }
    try:
        for name, dialog in dialogs.items():
            dialog.show()
            app.processEvents()
            needed = dialog.layout().sizeHint()
            assert needed.height() <= dialog.height(), (
                f"{name} is {needed.height() - dialog.height()}px too short for "
                f"its styled content ({needed.height()}px needed, "
                f"{dialog.height()}px tall) — Qt will overlap its children"
            )
            assert needed.width() <= dialog.width(), (
                f"{name} is {needed.width() - dialog.width()}px too narrow "
                f"({needed.width()}px needed, {dialog.width()}px wide)"
            )
    finally:
        for dialog in dialogs.values():
            dialog.close()
            dialog.deleteLater()
        window.close()
        window.deleteLater()
        app.processEvents()


def test_solver_dialog_value_column_is_aligned(tmp_path, monkeypatch) -> None:
    """Form values share one column; each row must not size its own label."""
    monkeypatch.setenv("LUNARIS_APP_DATA_DIR", str(tmp_path / "appdata"))
    from lunaris.ui.app import MainWindow
    from lunaris.ui.pages.mission_propagation_page import (
        SolverSettingsDialog,
        UISolverConfig,
    )

    app = _app()
    window = MainWindow()
    window.show()
    app.processEvents()
    dialog = SolverSettingsDialog(window, UISolverConfig())
    dialog.show()
    app.processEvents()
    try:
        xs = {
            field.mapTo(dialog, field.rect().topLeft()).x()
            for field in (dialog.ent_rtol, dialog.ent_atol, dialog.ent_maxstep)
        }
        assert len(xs) == 1, f"solver value column zigzags across x={sorted(xs)}"
    finally:
        dialog.close()
        dialog.deleteLater()
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
