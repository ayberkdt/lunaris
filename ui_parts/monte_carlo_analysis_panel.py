# -*- coding: utf-8 -*-
"""
Monte Carlo Analysis Workspace
==============================

This module provides a dedicated Qt widget for loading, analyzing, and
visualizing completed Monte Carlo result archives (`.npz` / `.h5`).

Why this lives in its own module
--------------------------------
The Monte Carlo configuration page already owns a substantial amount of
run-time UI.  Keeping the post-run analysis workspace separate prevents the
page from becoming a monolith and makes the intent of each area explicit:

1. `monte_carlo_page.py`
   Owns ensemble setup, backend selection, live progress, and last-run status.
2. `monte_carlo_analysis_panel.py`
   Owns archive loading, statistical post-processing, plot preview, and
   report export for completed MC runs.

Design goals
------------
- Non-blocking analysis bootstrap via a Qt worker thread.
- Friendly operator workflow: choose file -> analyze -> inspect plots/export.
- Reuse the project's existing pure analysis kernels instead of duplicating
  Monte Carlo statistics logic inside the UI layer.
"""

from __future__ import annotations

import io
import math
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

try:
    from .ui_commons import THEME, get_icon
except ImportError:
    if __name__ == "__main__" and (__package__ is None or __package__ == ""):
        import sys

        print("Run as: python -m ui_parts.monte_carlo_analysis_panel", file=sys.stderr)
        raise SystemExit(2)
    raise


def _card(title: str) -> QtWidgets.QGroupBox:
    """
    Return a themed group-box card consistent with the rest of the desktop UI.

    The analysis workspace reuses the same visual vocabulary as the run page so
    it feels like a natural extension of the Monte Carlo module rather than an
    unrelated tool bolted on later.
    """

    gb = QtWidgets.QGroupBox(title)
    gb.setStyleSheet(
        f"""
        QGroupBox {{
            border: 1px solid {THEME['border']};
            border-radius: 10px;
            margin-top: 14px;
            padding-top: 6px;
            background: {THEME['bg_card']};
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            subcontrol-position: top left;
            left: 12px;
            padding: 0 6px;
            color: {THEME['fg_main']};
            font-weight: 700;
            font-size: 10pt;
        }}
        """
    )
    return gb


def _label(text: str, *, muted: bool = False) -> QtWidgets.QLabel:
    """Create a simple themed label used across the workspace."""

    lbl = QtWidgets.QLabel(text)
    if muted:
        lbl.setStyleSheet(f"color: {THEME['fg_muted']}; font-size: 9pt;")
    return lbl


def _entry_style() -> str:
    """
    Shared entry/selector style used by the analysis workspace controls.

    Using one helper keeps the forms visually consistent and makes spacing
    adjustments easier when the overall density of the page is tuned later.
    """

    return f"""
        background: {THEME['bg_entry']};
        border: 1px solid {THEME['border']};
        border-radius: 8px;
        padding: 7px 10px;
        color: {THEME['fg_main']};
        min-height: 24px;
    """


def _format_span(seconds: Optional[float]) -> str:
    """
    Render short engineering-style durations for metric tables.

    The UI favors compact, scannable text over verbose natural language because
    these values often appear side-by-side with other dense run statistics.
    """

    if seconds is None or not math.isfinite(float(seconds)):
        return "—"

    total = max(0, int(round(float(seconds))))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _format_percent(probability: Optional[float], *, decimals: int = 2) -> str:
    """Render probabilities as percentages for operator-facing summary cards."""

    if probability is None or not math.isfinite(float(probability)):
        return "N/A"
    return f"{float(probability) * 100.0:.{decimals}f}%"


def _format_days(seconds: Optional[float], *, decimals: int = 3) -> str:
    """Render elapsed seconds as a day-based engineering quantity."""

    if seconds is None or not math.isfinite(float(seconds)):
        return "N/A"
    return f"{float(seconds) / 86400.0:.{decimals}f} d"


def _format_km(value: Optional[float], *, decimals: int = 3) -> str:
    """Render kilometre-scale values with a consistent suffix."""

    if value is None or not math.isfinite(float(value)):
        return "N/A"
    return f"{float(value):.{decimals}f} km"


class MCAnalysisWorker(QtCore.QThread):
    """
    Load a Monte Carlo archive and compute reusable statistics off the UI thread.

    The heavy statistical kernels are pure Python/NumPy and can take noticeable
    time for larger ensembles.  Running them in a background `QThread` keeps the
    workspace responsive while still reusing the project's canonical analysis
    pipeline.
    """

    analysis_complete = QtCore.Signal(object, object, str)
    analysis_progress = QtCore.Signal(str)
    analysis_error = QtCore.Signal(str)

    def __init__(
        self,
        result_path: str,
        *,
        compute_oe: bool,
        use_survived_only: bool,
    ) -> None:
        super().__init__()
        self.result_path = str(result_path)
        self.compute_oe = bool(compute_oe)
        self.use_survived_only = bool(use_survived_only)
        self._stop_requested = False

    def stop(self) -> None:
        """Request cancellation between worker phases."""

        self._stop_requested = True

    def _is_cancelled(self) -> bool:
        """Return True when the host has asked the worker to stop."""

        return bool(self._stop_requested)

    def run(self) -> None:
        """Load the archive and compute the canonical MC statistics bundle."""

        try:
            from analysis.mc_analysis import compute_mc_statistics
            from core.monte_carlo_engine import load_mc_result

            if self._is_cancelled():
                return
            self.analysis_progress.emit("Loading Monte Carlo archive...")
            result = load_mc_result(self.result_path)

            if self._is_cancelled():
                return
            self.analysis_progress.emit("Computing ensemble statistics...")
            stats = compute_mc_statistics(
                result,
                compute_oe=self.compute_oe,
                use_survived_only=self.use_survived_only,
            )

            if self._is_cancelled():
                return
            self.analysis_complete.emit(result, stats, self.result_path)
        except Exception as exc:
            self.analysis_error.emit(str(exc))


class MonteCarloAnalysisPanel(QtWidgets.QWidget):
    """
    Dedicated analysis workspace for completed Monte Carlo result archives.

    The panel separates post-run reasoning from run configuration.  Users can
    revisit old archives, compare outputs, inspect impact risk, and export a
    polished PDF report without re-running the simulation itself.
    """

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._worker: Optional[MCAnalysisWorker] = None
        self._result: Optional[Any] = None
        self._stats: Optional[Any] = None
        self._current_result_path: str = ""
        self._last_report_path: Optional[str] = None
        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self.scroll_root = QtWidgets.QScrollArea()
        self.scroll_root.setWidgetResizable(True)
        self.scroll_root.setFrameShape(QtWidgets.QFrame.NoFrame)
        root.addWidget(self.scroll_root, 1)

        self.content_widget = QtWidgets.QWidget()
        self.content_layout = QtWidgets.QVBoxLayout(self.content_widget)
        self.content_layout.setContentsMargins(12, 10, 12, 18)
        self.content_layout.setSpacing(18)

        controls_card = self._build_controls_card()
        self.content_layout.addWidget(controls_card)

        self.workspace_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.workspace_splitter.setChildrenCollapsible(False)
        self.workspace_splitter.setHandleWidth(8)

        self.summary_card = self._build_summary_card_professional()
        self.summary_card.setMinimumWidth(420)
        self.plot_card = self._build_plot_card()
        self.plot_card.setMinimumWidth(620)

        self.workspace_splitter.addWidget(self.summary_card)
        self.workspace_splitter.addWidget(self.plot_card)
        self.workspace_splitter.setStretchFactor(0, 0)
        self.workspace_splitter.setStretchFactor(1, 1)

        self.content_layout.addWidget(self.workspace_splitter, 1)
        self.scroll_root.setWidget(self.content_widget)

        QtCore.QTimer.singleShot(0, self._apply_default_splitter_sizes)

    def _apply_default_splitter_sizes(self) -> None:
        """
        Give the summary/preview split a readable desktop-first starting ratio.

        The summary column needs enough width for metric cards, while the plot
        preview benefits from owning most of the horizontal space.  A one-time
        size pass after the widget is shown avoids Qt squeezing everything into
        equal widths on first render.
        """

        if not hasattr(self, "workspace_splitter"):
            return

        total = self.workspace_splitter.width()
        if total <= 0:
            total = max(self.width(), 1100)

        summary_width = max(420, min(500, int(total * 0.30)))
        preview_width = max(680, total - summary_width)
        self.workspace_splitter.setSizes([summary_width, preview_width])

    def _build_controls_card(self) -> QtWidgets.QGroupBox:
        gb = _card("Analysis Workspace")
        layout = QtWidgets.QVBoxLayout(gb)
        layout.setContentsMargins(18, 22, 18, 18)
        layout.setSpacing(14)

        intro = _label(
            "Load a completed Monte Carlo result archive to compute ensemble "
            "statistics, inspect uncertainty growth, review impact risk, and "
            "export a multi-plot PDF report.",
            muted=True,
        )
        intro.setWordWrap(True)
        intro.setStyleSheet(f"color: {THEME['fg_muted']}; font-size: 10pt; line-height: 1.4;")
        layout.addWidget(intro)

        path_row = QtWidgets.QHBoxLayout()
        path_row.setSpacing(12)
        path_row.addWidget(_label("Result Archive:"))
        self.ent_result_path = QtWidgets.QLineEdit()
        self.ent_result_path.setPlaceholderText("Select a Monte Carlo .npz or .h5 archive")
        self.ent_result_path.setMinimumHeight(40)
        self.ent_result_path.setStyleSheet(
            f"""
            QLineEdit {{
                {_entry_style()}
            }}
            QLineEdit:focus {{ border-color: {THEME['accent']}; }}
            """
        )
        btn_browse = QtWidgets.QPushButton("Browse…")
        btn_browse.setFixedHeight(40)
        btn_browse.setMinimumWidth(120)
        btn_browse.clicked.connect(self._browse_result_file)
        path_row.addWidget(self.ent_result_path, 1)
        path_row.addWidget(btn_browse)
        layout.addLayout(path_row)

        option_row = QtWidgets.QHBoxLayout()
        option_row.setSpacing(18)
        self.chk_compute_oe = QtWidgets.QCheckBox("Compute orbital-element dispersion")
        self.chk_compute_oe.setChecked(False)
        self.chk_compute_oe.setToolTip(
            "Adds mean ±1σ curves for a/e/i. This can be slower for large ensembles."
        )
        self.chk_survived_only = QtWidgets.QCheckBox("Use survived-only ensemble statistics")
        self.chk_survived_only.setChecked(False)
        self.chk_survived_only.setToolTip(
            "Exclude impacted samples from the mean/covariance envelope calculations."
        )
        option_row.addWidget(self.chk_compute_oe)
        option_row.addWidget(self.chk_survived_only)
        option_row.addStretch(1)
        layout.addLayout(option_row)

        status_row = QtWidgets.QHBoxLayout()
        status_row.setSpacing(12)
        self.badge_status = QtWidgets.QLabel("READY")
        self.badge_status.setAlignment(QtCore.Qt.AlignCenter)
        self.badge_status.setFixedHeight(28)
        self.badge_status.setContentsMargins(10, 4, 10, 4)
        status_row.addWidget(self.badge_status)

        self.lbl_status = _label("Choose an archive to begin analysis.", muted=True)
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setStyleSheet(f"color: {THEME['fg_muted']}; font-size: 10pt;")
        status_row.addWidget(self.lbl_status, 1)
        layout.addLayout(status_row)
        self._set_status("READY", "Choose an archive to begin analysis.", accent=THEME["accent"])

        btn_row = QtWidgets.QHBoxLayout()
        btn_row.setSpacing(12)
        self.btn_analyze = QtWidgets.QPushButton("  Analyze Results")
        self.btn_analyze.setObjectName("primaryBtn")
        self.btn_analyze.setIcon(get_icon("fa6s.chart-line", THEME["fg_main"]))
        self.btn_analyze.setCursor(QtCore.Qt.PointingHandCursor)
        self.btn_analyze.setFixedHeight(42)
        self.btn_analyze.clicked.connect(self._start_analysis)

        self.btn_export_report = QtWidgets.QPushButton("  Export PDF Report")
        self.btn_export_report.setIcon(get_icon("fa6s.file-pdf", THEME["fg_muted"]))
        self.btn_export_report.setCursor(QtCore.Qt.PointingHandCursor)
        self.btn_export_report.setFixedHeight(42)
        self.btn_export_report.setEnabled(False)
        self.btn_export_report.clicked.connect(self._export_pdf_report)

        self.btn_open_folder = QtWidgets.QPushButton("  Open Folder")
        self.btn_open_folder.setIcon(get_icon("fa6s.folder-open", THEME["fg_muted"]))
        self.btn_open_folder.setCursor(QtCore.Qt.PointingHandCursor)
        self.btn_open_folder.setFixedHeight(42)
        self.btn_open_folder.clicked.connect(self._open_result_folder)

        btn_row.addWidget(self.btn_analyze, 2)
        btn_row.addWidget(self.btn_export_report, 1)
        btn_row.addWidget(self.btn_open_folder, 1)
        layout.addLayout(btn_row)

        return gb

    def _build_summary_card(self) -> QtWidgets.QGroupBox:
        gb = _card("Analysis Summary")
        grid = QtWidgets.QGridLayout(gb)
        grid.setContentsMargins(18, 22, 18, 18)
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(14)

        self._summary_labels: Dict[str, QtWidgets.QLabel] = {}

        def _add(row: int, col: int, key: str, title: str) -> None:
            tile = QtWidgets.QFrame()
            tile.setStyleSheet(
                f"""
                QFrame {{
                    background: {THEME['bg_entry']};
                    border: 1px solid {THEME['border']};
                    border-radius: 10px;
                }}
                """
            )
            tile.setMinimumHeight(72)
            tile_layout = QtWidgets.QVBoxLayout(tile)
            tile_layout.setContentsMargins(12, 10, 12, 10)
            tile_layout.setSpacing(4)

            key_lbl = _label(title, muted=True)
            key_lbl.setStyleSheet(f"color: {THEME['fg_muted']}; font-size: 9pt;")
            val_lbl = QtWidgets.QLabel("—")
            val_lbl.setWordWrap(True)
            val_lbl.setStyleSheet(
                f"color: {THEME['fg_main']}; font-weight: 700; font-size: 10.5pt;"
            )
            tile_layout.addWidget(key_lbl)
            tile_layout.addWidget(val_lbl)

            grid.addWidget(tile, row, col)
            self._summary_labels[key] = val_lbl

        _add(0, 0, "archive", "Archive")
        _add(0, 1, "format", "Format")
        _add(1, 0, "n_samples", "N Samples")
        _add(1, 1, "n_impacts", "N Impacts")
        _add(2, 0, "p_impact", "Impact Probability")
        _add(2, 1, "p_ci95", "95% Confidence Interval")
        _add(3, 0, "mean_impact_time", "Mean Impact Time")
        _add(3, 1, "duration", "Trajectory Duration")
        _add(4, 0, "final_alt_mean", "Final Mean Altitude")
        _add(4, 1, "final_alt_sigma", "Final Altitude 1σ")
        _add(5, 0, "max_tube", "Peak 3σ Tube Radius")
        _add(5, 1, "archive_size", "Archive Size")

        return gb

    def _build_summary_card_professional(self) -> QtWidgets.QGroupBox:
        """
        Build a denser but more operator-friendly summary card set.

        The original table worked functionally, but the updated card titles are
        more explicit and align better with the language used in engineering
        Monte Carlo reviews and PDF summaries.
        """

        gb = _card("Analysis Summary")
        grid = QtWidgets.QGridLayout(gb)
        grid.setContentsMargins(18, 22, 18, 18)
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(14)

        self._summary_labels = {}

        def _add(row: int, col: int, key: str, title: str) -> None:
            tile = QtWidgets.QFrame()
            tile.setStyleSheet(
                f"""
                QFrame {{
                    background: {THEME['bg_entry']};
                    border: 1px solid {THEME['border']};
                    border-radius: 10px;
                }}
                """
            )
            tile.setMinimumHeight(76)
            tile_layout = QtWidgets.QVBoxLayout(tile)
            tile_layout.setContentsMargins(12, 10, 12, 10)
            tile_layout.setSpacing(4)

            key_lbl = _label(title, muted=True)
            key_lbl.setStyleSheet(f"color: {THEME['fg_muted']}; font-size: 9pt;")
            val_lbl = QtWidgets.QLabel("N/A")
            val_lbl.setWordWrap(True)
            val_lbl.setStyleSheet(
                f"color: {THEME['fg_main']}; font-weight: 700; font-size: 10.5pt;"
            )
            tile_layout.addWidget(key_lbl)
            tile_layout.addWidget(val_lbl)

            grid.addWidget(tile, row, col)
            self._summary_labels[key] = val_lbl

        _add(0, 0, "archive", "Archive")
        _add(0, 1, "format", "Format")
        _add(1, 0, "n_samples", "Scenarios")
        _add(1, 1, "n_impacts", "Impacted Samples")
        _add(2, 0, "p_impact", "Impact Probability")
        _add(2, 1, "p_ci95", "95% Wilson CI")
        _add(3, 0, "mean_impact_time", "Mean Impact Epoch")
        _add(3, 1, "duration", "Trajectory Span")
        _add(4, 0, "final_alt_mean", "Final Mean Altitude")
        _add(4, 1, "final_alt_sigma", "Final Altitude 1-sigma")
        _add(5, 0, "max_tube", "Peak 3-sigma Tube")
        _add(5, 1, "archive_size", "Archive Footprint")

        return gb

    def _build_plot_card(self) -> QtWidgets.QGroupBox:
        gb = _card("Plot Preview")
        layout = QtWidgets.QVBoxLayout(gb)
        layout.setContentsMargins(18, 22, 18, 18)
        layout.setSpacing(14)

        top_row = QtWidgets.QHBoxLayout()
        top_row.setSpacing(12)
        top_row.addWidget(_label("Plot:"))
        self.cb_plot = QtWidgets.QComboBox()
        self.cb_plot.addItems(
            [
                "Altitude Envelope",
                "Position Covariance",
                "Impact Time Histogram",
                "Impact Map",
                "3σ Covariance Tubes (3D)",
                "Orbital-Element Dispersion",
            ]
        )
        self.cb_plot.setMinimumHeight(40)
        self.cb_plot.setStyleSheet(
            f"""
            QComboBox {{
                {_entry_style()}
            }}
            QComboBox::drop-down {{
                border: none;
                width: 22px;
            }}
            """
        )
        self.cb_plot.currentTextChanged.connect(lambda _text: self._render_selected_plot())
        top_row.addWidget(self.cb_plot, 1)

        self.btn_refresh_plot = QtWidgets.QPushButton("Refresh Plot")
        self.btn_refresh_plot.setFixedHeight(40)
        self.btn_refresh_plot.setMinimumWidth(140)
        self.btn_refresh_plot.clicked.connect(self._render_selected_plot)
        self.btn_refresh_plot.setEnabled(False)
        top_row.addWidget(self.btn_refresh_plot)
        layout.addLayout(top_row)

        self.lbl_plot_caption = _label(
            "Analyze an archive to preview uncertainty and impact plots.",
            muted=True,
        )
        self.lbl_plot_caption.setWordWrap(True)
        self.lbl_plot_caption.setStyleSheet(f"color: {THEME['fg_muted']}; font-size: 10pt;")
        layout.addWidget(self.lbl_plot_caption)

        self.plot_scroll = QtWidgets.QScrollArea()
        self.plot_scroll.setWidgetResizable(True)
        self.plot_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.plot_scroll.setMinimumHeight(500)

        self.lbl_plot = QtWidgets.QLabel("No analysis loaded yet.")
        self.lbl_plot.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_plot.setStyleSheet(
            f"""
            QLabel {{
                border: 1px dashed {THEME['border']};
                border-radius: 8px;
                background: {THEME['bg_entry']};
                color: {THEME['fg_muted']};
                padding: 24px;
            }}
            """
        )
        self.lbl_plot.setWordWrap(True)
        self.lbl_plot.setMinimumSize(760, 500)

        self.plot_scroll.setWidget(self.lbl_plot)
        layout.addWidget(self.plot_scroll, 1)

        return gb

    # ------------------------------------------------------------------
    # Public helpers used by the MC page / host window
    # ------------------------------------------------------------------

    def set_result_path(self, result_path: str, *, auto_analyze: bool = False) -> None:
        """
        Update the visible archive path and optionally start analysis immediately.

        This is used after a successful Monte Carlo run so the analysis
        workspace can seamlessly pivot to the freshly written archive.
        """

        normalized = str(result_path or "").strip()
        if not normalized:
            return
        self.ent_result_path.setText(normalized)
        self._current_result_path = normalized
        if auto_analyze:
            self._start_analysis()

    def shutdown(self) -> None:
        """
        Stop any in-flight background analysis worker during application close.

        The statistical kernels are pure Python, so cancellation can only be
        honored between major phases.  A short wait still prevents most thread
        leaks during shutdown.
        """

        if self._worker is not None and self._worker.isRunning():
            self._worker.stop()
            self._worker.wait(1000)

    # ------------------------------------------------------------------
    # Controls
    # ------------------------------------------------------------------

    def _browse_result_file(self) -> None:
        """Open a file dialog for selecting a saved Monte Carlo archive."""

        current = self.ent_result_path.text().strip() or str(Path.cwd())
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open Monte Carlo Result Archive",
            current,
            "Monte Carlo Archives (*.npz *.h5 *.hdf5);;All Files (*.*)",
        )
        if path:
            self.ent_result_path.setText(path)

    def _open_result_folder(self) -> None:
        """Open the directory containing the current result archive."""

        path_text = self.ent_result_path.text().strip()
        if not path_text:
            QtWidgets.QMessageBox.information(
                self,
                "No Archive Selected",
                "Select or analyze a Monte Carlo archive first.",
            )
            return

        path = Path(path_text).expanduser().resolve()
        folder = path.parent if path.suffix else path
        if folder.exists():
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(folder)))
        else:
            QtWidgets.QMessageBox.warning(
                self,
                "Folder Not Found",
                f"The result folder does not exist:\n{folder}",
            )

    def _start_analysis(self) -> None:
        """
        Kick off archive loading and statistics generation in the worker thread.

        This is the central user action of the workspace.  It validates the
        selected path, resets the current preview state, and then lets the
        worker produce a reusable analysis bundle.
        """

        result_path = self.ent_result_path.text().strip()
        if not result_path:
            QtWidgets.QMessageBox.warning(
                self,
                "Result Archive Required",
                "Choose a Monte Carlo .npz or .h5 archive to analyze.",
            )
            return

        resolved = Path(result_path).expanduser().resolve()
        if not resolved.exists():
            QtWidgets.QMessageBox.warning(
                self,
                "Archive Not Found",
                f"The selected Monte Carlo archive does not exist:\n{resolved}",
            )
            return

        if self._worker is not None and self._worker.isRunning():
            QtWidgets.QMessageBox.information(
                self,
                "Analysis In Progress",
                "Wait for the current analysis job to finish before starting another one.",
            )
            return

        self._set_busy(True)
        self._set_status("ANALYZING", "Initializing Monte Carlo analysis...", accent=THEME["warning"])
        self.btn_export_report.setEnabled(False)
        self.btn_refresh_plot.setEnabled(False)
        self._set_plot_message("Computing analysis bundle...", "Statistical post-processing is running in the background.")

        self._worker = MCAnalysisWorker(
            str(resolved),
            compute_oe=self.chk_compute_oe.isChecked(),
            use_survived_only=self.chk_survived_only.isChecked(),
        )
        self._worker.analysis_progress.connect(self._on_analysis_progress)
        self._worker.analysis_complete.connect(self._on_analysis_complete)
        self._worker.analysis_error.connect(self._on_analysis_error)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.start()

    def _on_analysis_progress(self, text: str) -> None:
        """Reflect worker progress messages in the lightweight status area."""

        self._set_status("ANALYZING", text, accent=THEME["warning"])

    def _on_analysis_complete(self, result: Any, stats: Any, result_path: str) -> None:
        """
        Store the loaded analysis bundle and refresh the workspace visuals.

        Once the worker emits this signal, all downstream UI work becomes cheap:
        summary labels read from cached statistics and plots are generated on
        demand from the already-loaded data structures.
        """

        self._result = result
        self._stats = stats
        self._current_result_path = str(result_path)
        self.ent_result_path.setText(self._current_result_path)

        self._refresh_summary_metrics()
        self._set_status("READY", "Analysis completed successfully.", accent=THEME["success"])
        self.btn_export_report.setEnabled(True)
        self.btn_refresh_plot.setEnabled(True)
        self._render_selected_plot()

    def _on_analysis_error(self, text: str) -> None:
        """Surface worker failures without crashing the rest of the page."""

        self._set_status("FAILED", f"Analysis failed: {text}", accent=THEME["error"])
        self._set_plot_message("Analysis failed.", "Review the selected file and try again.")

    def _on_worker_finished(self) -> None:
        """Restore interactive controls after the worker exits for any reason."""

        self._set_busy(False)

    def _export_pdf_report(self) -> None:
        """
        Export the canonical multi-figure Monte Carlo PDF report.

        The plotting layer already knows how to assemble the engineering report,
        so the workspace simply collects the destination path and delegates to
        `analysis.mc_plotting.plot_mc_report`.
        """

        if self._result is None or self._stats is None or not self._current_result_path:
            QtWidgets.QMessageBox.information(
                self,
                "No Analysis Available",
                "Analyze a Monte Carlo archive before exporting a PDF report.",
            )
            return

        src = Path(self._current_result_path).expanduser().resolve()
        default_pdf = src.with_name(src.stem + "_analysis_report.pdf")
        out_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export Monte Carlo PDF Report",
            str(default_pdf),
            "PDF Files (*.pdf)",
        )
        if not out_path:
            return

        try:
            from analysis.mc_plotting import plot_mc_report
            from matplotlib import pyplot as plt

            plot_mc_report(self._result, self._stats, output_path=out_path, show=False)
            plt.close("all")
            self._last_report_path = out_path
            self._set_status(
                "READY",
                f"PDF report exported: {Path(out_path).name}",
                accent=THEME["success"],
            )
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self,
                "PDF Export Failed",
                f"Could not export the Monte Carlo report:\n\n{exc}",
            )

    # ------------------------------------------------------------------
    # Summary + plot rendering
    # ------------------------------------------------------------------

    def _update_summary_labels(self) -> None:
        """Populate the summary table from the cached analysis bundle."""

        if self._result is None or self._stats is None or not self._current_result_path:
            return

        from common.constants import DAY_S

        path = Path(self._current_result_path).expanduser().resolve()
        impacts = self._stats.impacts
        ensemble = self._stats.ensemble
        ellipsoids = self._stats.ellipsoids

        archive_size_mb = "—"
        try:
            archive_size_mb = f"{path.stat().st_size / (1024 * 1024):.2f} MB"
        except OSError:
            pass

        def _set(key: str, value: str) -> None:
            lbl = self._summary_labels.get(key)
            if lbl is not None:
                lbl.setText(value)

        _set("archive", path.name)
        _set("format", path.suffix.lower().lstrip(".") or "—")
        _set("n_samples", str(self._result.n_samples))
        _set("n_impacts", f"{impacts.n_impacts} / {impacts.n_total}")
        _set("p_impact", f"{impacts.p_impact:.4f}")
        _set("p_ci95", f"[{impacts.p_impact_ci95[0]:.4f}, {impacts.p_impact_ci95[1]:.4f}]")

        mean_impact = "No impacts"
        if math.isfinite(float(impacts.t_impact_mean)):
            mean_impact = f"{impacts.t_impact_mean / DAY_S:.3f} d"
        _set("mean_impact_time", mean_impact)

        duration = float(self._result.t[-1]) if len(self._result.t) > 0 else 0.0
        _set("duration", f"{duration / DAY_S:.3f} d ({_format_span(duration)})")
        _set("final_alt_mean", f"{float(ensemble.alt_mean[-1]):.3f} km")
        _set("final_alt_sigma", f"{float(ensemble.alt_std[-1]):.3f} km")

        tube_km = np.asarray(ellipsoids.tube_radii(), dtype=np.float64) / 1_000.0
        peak_tube = float(np.nanmax(tube_km)) if tube_km.size else math.nan
        _set("max_tube", f"{peak_tube:.3f} km" if math.isfinite(peak_tube) else "—")
        _set("archive_size", archive_size_mb)

    def _refresh_summary_metrics(self) -> None:
        """Populate the summary tiles using presentation-grade metric formatting."""

        if self._result is None or self._stats is None or not self._current_result_path:
            return

        path = Path(self._current_result_path).expanduser().resolve()
        impacts = self._stats.impacts
        ensemble = self._stats.ensemble
        ellipsoids = self._stats.ellipsoids

        archive_size_mb = "N/A"
        try:
            archive_size_mb = f"{path.stat().st_size / (1024 * 1024):.2f} MB"
        except OSError:
            pass

        def _set(key: str, value: str) -> None:
            lbl = self._summary_labels.get(key)
            if lbl is not None:
                lbl.setText(value)

        total_samples = int(self._result.n_samples)
        impact_rate = impacts.n_impacts / max(1, impacts.n_total)
        duration_s = float(self._result.t[-1] - self._result.t[0]) if len(self._result.t) > 1 else 0.0

        _set("archive", path.name)
        _set("format", path.suffix.lower().lstrip(".") or "N/A")
        _set("n_samples", f"{total_samples:,} scenarios")
        _set("n_impacts", f"{impacts.n_impacts:,} ({_format_percent(impact_rate)})")
        _set("p_impact", _format_percent(impacts.p_impact))
        _set(
            "p_ci95",
            f"{_format_percent(impacts.p_impact_ci95[0])} to {_format_percent(impacts.p_impact_ci95[1])}",
        )

        mean_impact = "No impacts"
        if math.isfinite(float(impacts.t_impact_mean)):
            mean_impact = _format_days(impacts.t_impact_mean)
            if math.isfinite(float(impacts.t_impact_std)):
                mean_impact = f"{mean_impact} +/- {_format_days(impacts.t_impact_std)}"
        _set("mean_impact_time", mean_impact)

        _set("duration", f"{_format_days(duration_s)} ({_format_span(duration_s)})")
        _set("final_alt_mean", _format_km(float(ensemble.alt_mean[-1]) if ensemble.alt_mean.size else math.nan))
        _set("final_alt_sigma", _format_km(float(ensemble.alt_std[-1]) if ensemble.alt_std.size else math.nan))

        tube_km = np.asarray(ellipsoids.tube_radii(), dtype=np.float64) / 1_000.0
        peak_tube = float(np.nanmax(tube_km)) if tube_km.size else math.nan
        _set("max_tube", _format_km(peak_tube))
        _set("archive_size", archive_size_mb)

    def _render_selected_plot(self) -> None:
        """
        Generate the selected preview plot and show it as a raster image.

        Rendering to an in-memory PNG keeps the Qt integration simple and avoids
        backend conflicts between Matplotlib's non-interactive `Agg` usage and
        the desktop application's Qt event loop.
        """

        if self._result is None or self._stats is None:
            self._set_plot_message(
                "No analysis loaded yet.",
                "Analyze an archive to preview uncertainty and impact plots.",
            )
            return

        try:
            from analysis.mc_plotting import (
                plot_altitude_envelope,
                plot_covariance_tubes_3d,
                plot_impact_map,
                plot_impact_time_histogram,
                plot_oe_dispersion,
                plot_position_covariance_history,
            )
            from matplotlib import pyplot as plt

            title = self.cb_plot.currentText()
            figure = None
            caption = ""

            if title == "Altitude Envelope":
                figure = plot_altitude_envelope(self._result, self._stats.ensemble)
                caption = "Mean altitude with uncertainty bands and a subset of individual trajectories."
            elif title == "Position Covariance":
                figure = plot_position_covariance_history(self._stats.ensemble)
                caption = "Cartesian position dispersion history and cross-axis correlation trends."
            elif title == "Impact Time Histogram":
                figure = plot_impact_time_histogram(self._stats.impacts, self._result)
                caption = "Distribution of impact epochs across the ensemble."
            elif title == "Impact Map":
                figure = plot_impact_map(self._stats.impacts)
                caption = "Lunar-surface impact footprint of impacting samples."
            elif title == "3σ Covariance Tubes (3D)":
                figure = plot_covariance_tubes_3d(self._result, self._stats.ellipsoids)
                caption = "Mean orbit with sampled trajectories and selected 3σ position ellipsoids."
            elif title == "Orbital-Element Dispersion":
                if self._stats.oe_disp is None:
                    self._set_plot_message(
                        "Orbital-element dispersion is unavailable.",
                        "Enable 'Compute orbital-element dispersion' and analyze again to generate this preview.",
                    )
                    return
                figure = plot_oe_dispersion(self._stats.oe_disp)
                caption = "Mean ±1σ evolution of semi-major axis, eccentricity, and inclination."
            else:
                self._set_plot_message("Unsupported plot selection.", title)
                return

            if figure is None:
                self._set_plot_message("No figure was produced.", title)
                return

            pixmap = self._figure_to_pixmap(figure)
            plt.close(figure)

            self.lbl_plot.setStyleSheet("border: none; background: transparent;")
            self.lbl_plot.setText("")
            self.lbl_plot.setPixmap(pixmap)
            self.lbl_plot.resize(pixmap.size())
            self.lbl_plot_caption.setText(caption)
        except Exception as exc:
            self._set_plot_message(
                "Could not render the selected plot.",
                f"Plot generation failed: {exc}",
            )

    def _figure_to_pixmap(self, figure: Any) -> QtGui.QPixmap:
        """Rasterize a Matplotlib figure to a high-resolution `QPixmap`."""

        buffer = io.BytesIO()
        figure.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
        png = buffer.getvalue()
        pixmap = QtGui.QPixmap()
        if not pixmap.loadFromData(png, "PNG"):
            raise RuntimeError("Could not decode generated plot image.")
        return pixmap

    def _set_plot_message(self, title: str, detail: str) -> None:
        """Show a textual placeholder in the plot preview area."""

        self.lbl_plot.setPixmap(QtGui.QPixmap())
        self.lbl_plot.setStyleSheet(
            f"""
            QLabel {{
                border: 1px dashed {THEME['border']};
                border-radius: 8px;
                background: {THEME['bg_entry']};
                color: {THEME['fg_muted']};
                padding: 18px;
            }}
            """
        )
        self.lbl_plot.setText(f"{title}\n\n{detail}")
        self.lbl_plot_caption.setText(detail)

    # ------------------------------------------------------------------
    # Small UI-state helpers
    # ------------------------------------------------------------------

    def _set_busy(self, busy: bool) -> None:
        """Disable/enable controls while an analysis job is running."""

        self.btn_analyze.setEnabled(not busy)
        self.btn_export_report.setEnabled((not busy) and (self._stats is not None))
        self.btn_refresh_plot.setEnabled((not busy) and (self._stats is not None))
        self.chk_compute_oe.setEnabled(not busy)
        self.chk_survived_only.setEnabled(not busy)

    def _set_status(self, badge: str, text: str, *, accent: str) -> None:
        """Update the workspace badge + helper text as a compact status block."""

        self.badge_status.setText(badge)
        self.badge_status.setStyleSheet(
            f"""
            border-radius: 10px;
            border: 1px solid {accent};
            background: transparent;
            color: {accent};
            font-weight: 700;
            padding: 0 8px;
            """
        )
        self.lbl_status.setText(text)


__all__ = [
    "MCAnalysisWorker",
    "MonteCarloAnalysisPanel",
]
