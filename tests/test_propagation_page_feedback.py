# tests/test_propagation_page_feedback.py
"""UI wiring checks for the solver cost/accuracy/validation feedback + atol."""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PySide6 = pytest.importorskip("PySide6")
from PySide6 import QtWidgets  # noqa: E402

from lunaris.ui.pages.mission_propagation_page import MissionPropagationPage  # noqa: E402


def _app() -> QtWidgets.QApplication:
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_fixed_step_shows_cost_estimate():
    app = _app()
    page = MissionPropagationPage()
    page.cb_integrator.setCurrentText("RK4 (Fixed-step)")
    page.ent_duration.setText("1")
    page.cb_duration_unit.setCurrentText("Days")
    page.ent_max_step.setText("60")
    app.processEvents()

    assert page.tolerance_group.isHidden() is True
    text = page.step_feedback.label.text()
    assert "steps" in text and "force evaluations" in text
    page.close()


def test_fixed_step_under_resolution_warns():
    app = _app()
    page = MissionPropagationPage()
    page.cb_integrator.setCurrentText("RK4 (Fixed-step)")
    page.ent_duration.setText("1")
    page.cb_duration_unit.setCurrentText("Days")
    page.ent_max_step.setText("5000")  # ~17 steps over a day
    app.processEvents()

    assert page.step_feedback.property("kind") == "warning"
    assert "under-resolved" in page.step_feedback.label.text()
    page.close()


def test_invalid_step_shows_error():
    app = _app()
    page = MissionPropagationPage()
    page.cb_integrator.setCurrentText("RK4 (Fixed-step)")
    page.ent_max_step.setText("-5")
    app.processEvents()

    assert page.step_feedback.property("kind") == "error"
    page.close()


def test_adaptive_accuracy_band_and_loose_warning():
    app = _app()
    page = MissionPropagationPage()
    page.cb_integrator.setCurrentText("DOP853 (Adaptive)")
    page.ent_rtol.setText("1e-10")
    app.processEvents()
    assert "Accuracy band" in page.tol_feedback.label.text()

    page.ent_rtol.setText("1e-1")
    app.processEvents()
    assert page.tol_feedback.property("kind") == "warning"
    page.close()


def test_atol_is_exposed_and_persists():
    app = _app()
    page = MissionPropagationPage()
    page.cb_integrator.setCurrentText("DOP853 (Adaptive)")
    page.ent_atol.setText("1e-13")
    app.processEvents()

    assert page.to_dict()["integrator"]["atol"] == "1e-13"

    page.apply_dict({"integrator": {"method": "DOP853 (Adaptive)", "rtol": "1e-9", "atol": "1e-11"}})
    app.processEvents()
    assert page.ent_atol.text() == "1e-11"
    page.close()
