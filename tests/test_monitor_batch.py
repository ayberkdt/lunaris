"""Batch observability tests (Mission Monitor phase 6).

The Batch Progress widget consumes the existing [BATCH_PROGRESS] /
[BATCH_METRICS] protocol via controller-scoped payloads (never the run
store, so a batch next to a live run cannot contaminate run provenance).
"""

from __future__ import annotations

import pytest
from tests.ui_qt_helpers import QtWidgets

from lunaris.ui.monitor.widgets.batch_progress import (
    BATCH_PROGRESS_SPEC,
    BatchProgressWidget,
)
from lunaris.ui.monitor.workspace import MonitorController


def _app() -> QtWidgets.QApplication:
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture()
def controller():
    app = _app()
    ctrl = MonitorController()
    yield ctrl
    app.processEvents()


def _shown(widget: QtWidgets.QWidget) -> QtWidgets.QWidget:
    widget.resize(500, 400)
    widget.show()
    _app().processEvents()
    return widget


PROGRESS = {
    "stage": "propagating", "percent": 42.5, "fraction": 0.425,
    "done_samples": 425.0, "total_samples": 1000, "elapsed_s": 63.2,
    "eta_s": 85.4, "backend": "torch_cuda_sh", "detail": "chunk 3/8",
}

METRICS = {
    "n_samples": 1000, "n_impacts": 37, "p_impact": 0.037,
    "p_impact_ci95": [0.0261, 0.0499], "wall_time_s": 148.6,
    "backend": "torch_cuda_sh",
    "requested_batch_backend": "torch_cuda_sh",
    "actual_batch_backend": "numba_cpu_sh",
    "fallback_reason": "CUDA device unavailable",
    "runtime_model_kind": "force_direct",
    "requested_sh_degree": 120, "actual_sh_degree": 64,
    "device_name": "CPU (fallback)",
    "output_path": "outputs/batch/run.h5",
}


class TestBatchProgressWidget:
    def test_empty_state_before_any_batch_run(self, controller):
        widget = _shown(BatchProgressWidget(BATCH_PROGRESS_SPEC, controller))
        assert widget._stack.currentWidget() is widget._empty
        assert "No batch run observed" in widget._empty.title_label.text()

    def test_progress_payload_populates_live_rows(self, controller):
        widget = _shown(BatchProgressWidget(BATCH_PROGRESS_SPEC, controller))
        controller.set_batch_progress(PROGRESS)
        controller.flush_now()
        widget._do_refresh()
        assert widget._stack.currentWidget() is widget._content
        assert widget.progress_bar.value() == 425
        assert "propagating — chunk 3/8" == widget._row_labels["stage"].text()
        assert widget._row_labels["samples"].text() == "425 / 1,000"
        assert "torch_cuda_sh" in widget.badge_label.text()
        assert "eta" in widget._row_labels

    def test_metrics_payload_adds_final_statistics_and_fallback(self, controller):
        widget = _shown(BatchProgressWidget(BATCH_PROGRESS_SPEC, controller))
        controller.set_batch_progress(PROGRESS)
        controller.set_batch_metrics(METRICS)
        controller.flush_now()
        widget._do_refresh()
        assert widget.progress_bar.value() == 1000
        assert widget._row_labels["impacts"].text() == "37 / 1,000"
        assert "3.70 %" in widget._row_labels["p_impact"].text()
        assert "2.61" in widget._row_labels["p_impact"].text()  # CI shown
        assert widget._row_labels["backend"].text() == "torch_cuda_sh → numba_cpu_sh"
        assert widget._row_labels["model_kind"].text() == "force_direct"
        assert widget._row_labels["degree"].text() == "120 → 64"
        assert widget.fallback_notice.isVisibleTo(widget)
        assert "CUDA device unavailable" in widget.fallback_notice.text()
        # The final badge names the backend that actually ran.
        assert "numba_cpu_sh" in widget.badge_label.text()

    def test_new_batch_run_resets_observability(self, controller):
        widget = _shown(BatchProgressWidget(BATCH_PROGRESS_SPEC, controller))
        controller.set_batch_metrics(METRICS)
        controller.begin_batch_run()
        controller.flush_now()
        widget._do_refresh()
        assert controller.batch_metrics is None
        assert widget._stack.currentWidget() is widget._empty

    def test_batch_payloads_never_touch_the_run_store(self, controller):
        controller.begin_live_run()
        controller.set_batch_progress(PROGRESS)
        controller.set_batch_metrics(METRICS)
        assert controller.store.provenance is None
        assert controller.store.n_samples == 0
        assert controller.store.run_diagnostics is None


class TestAppIntegration:
    def test_batch_stdout_lines_reach_the_monitor(self, tmp_path, monkeypatch):
        import json

        monkeypatch.setenv("LUNARIS_APP_DATA_DIR", str(tmp_path / "appdata"))
        app = _app()
        from lunaris.ui.app import MainWindow

        w = MainWindow()
        try:
            class FakeProc:
                def __init__(self, payload: bytes) -> None:
                    self._payload = payload

                def readAllStandardOutput(self):  # noqa: N802 (Qt API shape)
                    data, self._payload = self._payload, b""
                    return data

            lines = (
                f"[BATCH_PROGRESS] {json.dumps(PROGRESS)}\n"
                f"[BATCH_METRICS] {json.dumps(METRICS)}\n"
            ).encode()
            w.batch_process = FakeProc(lines)
            w._on_batch_stdout()
            assert w.monitor_controller.batch_progress is not None
            assert w.monitor_controller.batch_progress["stage"] == "propagating"
            assert w.monitor_controller.batch_metrics is not None
            assert w.monitor_controller.batch_metrics["n_impacts"] == 37
        finally:
            w.batch_process = None
            w.close()
            w.deleteLater()
            app.processEvents()
