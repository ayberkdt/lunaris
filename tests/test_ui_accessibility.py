"""Accessibility regression tests (Step B/E closeout).

Locks in the accessible-name coverage on icon-only controls and the primary
mode-selecting combos so future edits cannot silently regress screen-reader
support.
"""

from __future__ import annotations
import pytest
_ = pytest.importorskip("PySide6.QtWidgets")

import os
from PySide6.QtWidgets import QApplication
from tests.ui_qt_helpers import QtCore, QtGui, QtWidgets


def _app():
    return QApplication.instance() or QApplication([])


def test_force_models_icon_buttons_have_accessible_names() -> None:
    _app()
    from lunaris.ui.pages.force_models_page import ForceModelsPage

    page = ForceModelsPage()
    # Icon-only gear buttons must be named for screen readers.
    assert page.btn_gravity_settings.accessibleName()
    assert page.btn_albedo_settings.accessibleName()
    # Every force toggle is icon-only paint → must carry a name.
    for switch in (
        page.sw_gravity, page.sw_sun, page.sw_earth, page.sw_earth_j2,
        page.sw_srp, page.sw_albedo, page.sw_thermal,
        page.sw_tides_k2, page.sw_tides_k3, page.sw_relativity_1pn,
    ):
        assert switch.accessibleName()


def test_batch_page_selectors_have_accessible_names() -> None:
    _app()
    from lunaris.ui.pages.batch_propagation_page import (
        BatchPropagationPage,
        UIBatchPropagationConfig,
    )

    page = BatchPropagationPage(batch_cfg=UIBatchPropagationConfig(use_gpu=False))
    try:
        for combo in (
            page.cb_sampling_method,
            page.cb_batch_gravity_mode,
            page.cb_batch_backend,
            page.cb_format,
        ):
            assert combo.accessibleName()
        assert page.toggle_gpu.accessibleName()
    finally:
        page.shutdown()
