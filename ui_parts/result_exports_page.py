# LUNAR_SIMULATION/ui_parts/result_exports_page.py
# -*- coding: utf-8 -*-
"""
Results & Export Page (UI Part)
===============================

This module encapsulates the output/export workflow for the Lunar Simulation
desktop UI. Earlier revisions kept this page embedded directly inside the main
window, which made the window class responsible for both high-level process
orchestration and low-level widget layout. The goal of this module is to give
the output page the same level of ownership already present in the other
`ui_parts/*_page.py` files.

Responsibilities
----------------
1. Collect where mission results should be written.
2. Expose a small set of backend-facing output options:
   - whether 3D plot generation is requested
   - optional downsample factor for 3D rendering
3. Display the generated command preview in a page-owned code block.
4. Emit Qt signals for host-owned actions such as opening a file dialog or
   copying the preview to the clipboard.

Why the CSV toggle was removed
------------------------------
The previous embedded UI showed a "CSV Export" toggle, but the backend CLI does
not currently consume a matching flag. Keeping a control that the backend
ignores is misleading, so this page replaces that toggle with an explicit note
that tabular/report artifacts are backend-managed.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional

from PySide6 import QtCore, QtGui, QtWidgets

try:
    from .ui_commons import THEME, ToggleSwitch, get_icon
except ImportError:
    if __name__ == "__main__" and (__package__ is None or __package__ == ""):
        import sys

        print("\n" + "!" * 60, file=sys.stderr)
        print("  [ERROR] This module must be run as part of the package.", file=sys.stderr)
        print("  From the project root, run:", file=sys.stderr)
        print("\n      python -m ui_parts.result_exports_page\n", file=sys.stderr)
        print("!" * 60 + "\n", file=sys.stderr)
        raise SystemExit(2)
    raise


@dataclass
class OutputPageState:
    """
    Serializable state owned by the output page.

    Attributes
    ----------
    output_dir:
        Filesystem directory where mission outputs should be written.
    generate_3d_plots:
        Mirrors the backend `--make-3d-plots` flag.
    downsample_3d:
        Optional backend `--downsample-3d` factor. A value of 1 means
        "no extra downsampling".
    """

    output_dir: str = ""
    generate_3d_plots: bool = False
    downsample_3d: int = 1


class ResultsExportPage(QtWidgets.QWidget):
    """
    Page 4: results directory, export-related options, and command preview.

    The widget owns all controls on the page and communicates outward through a
    small signal surface. The host window stays responsible for actions that
    need broader application context, such as opening dialogs or copying to the
    system clipboard.
    """

    browse_output_dir_requested = QtCore.Signal()
    open_output_dir_requested = QtCore.Signal()
    refresh_preview_requested = QtCore.Signal()
    copy_preview_requested = QtCore.Signal()

    def __init__(
        self,
        *,
        project_root: Path,
        create_card: Callable[[str], QtWidgets.QGroupBox],
        initial_state: Optional[OutputPageState] = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._project_root = Path(project_root)
        self._create_card = create_card
        self._state = initial_state or OutputPageState(
            output_dir=str(self._project_root / "mission_results"),
            generate_3d_plots=False,
            downsample_3d=1,
        )

        self._build_ui()
        self.apply_state(self._state)

    def get_state(self) -> OutputPageState:
        """
        Read the current page widgets and return a serializable snapshot.

        The returned dataclass is intentionally Qt-free so it can be handed to
        command builders, persistence helpers, or tests without pulling any
        widget dependencies along with it.
        """

        return OutputPageState(
            output_dir=self.ent_out_dir.text().strip(),
            generate_3d_plots=bool(self.toggle_anim3d.isChecked()),
            downsample_3d=int(self.spin_downsample_3d.value()),
        )

    def apply_state(self, state: OutputPageState) -> None:
        """
        Restore a previously captured page state.

        This method is used by session restore flows and also acts as the
        canonical place for default initialization so the host window does not
        need to know which widgets exist on the page.
        """

        self._state = state
        self.ent_out_dir.setText(state.output_dir or str(self._project_root / "mission_results"))
        self.toggle_anim3d.setChecked(bool(state.generate_3d_plots))
        self.spin_downsample_3d.setValue(max(1, int(state.downsample_3d or 1)))
        self._sync_3d_controls()

    def set_output_dir(self, output_dir: str) -> None:
        """
        Update only the directory field after a host-driven file dialog action.

        Keeping this as a dedicated helper avoids the host reaching into widget
        internals whenever it needs to push a selected path back into the page.
        """

        self.ent_out_dir.setText(output_dir)

    def set_command_preview(self, text: str, *, is_error: bool = False) -> None:
        """
        Render the backend command preview using page-owned styling rules.

        Parameters
        ----------
        text:
            The shell-safe command preview or an explanatory error message.
        is_error:
            When True the preview is styled as a failure state so the user can
            distinguish "command unavailable" from a valid preview.
        """

        self.txt_preview.setPlainText(text)
        color = THEME["error"] if is_error else "#A0A0A0"
        self.txt_preview.setStyleSheet(
            f"""
            background-color: {THEME['bg_log']};
            color: {color};
            font-family: Consolas, monospace;
            border-radius: 6px;
            """
        )

    def _build_ui(self) -> None:
        """
        Build the full page layout.

        The page follows the same card-based visual language as the other UI
        parts so the main window can embed it directly without page-specific
        styling logic.
        """

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(20)

        layout.addWidget(self._build_output_config_card())
        layout.addWidget(self._build_artifacts_card())
        layout.addWidget(self._build_artifact_browser_card())
        layout.addWidget(self._build_preview_card())
        layout.addStretch(1)

    def _build_output_config_card(self) -> QtWidgets.QGroupBox:
        """
        Create the directory/option card shown at the top of the page.

        The host is expected to connect the emitted signals to application-wide
        actions such as showing a directory picker or opening the selected path
        in the OS file explorer.
        """

        group_box = self._create_card("Data Export")
        layout = QtWidgets.QVBoxLayout(group_box)
        layout.setContentsMargins(20, 24, 20, 20)
        layout.setSpacing(15)

        layout.addWidget(QtWidgets.QLabel("Results Directory:"))

        dir_row = QtWidgets.QHBoxLayout()
        self.ent_out_dir = QtWidgets.QLineEdit()
        self.ent_out_dir.setPlaceholderText("Select an output directory...")
        dir_row.addWidget(self.ent_out_dir, 1)

        btn_browse = QtWidgets.QPushButton("Browse")
        btn_browse.setIcon(get_icon("fa6s.folder-open", THEME["fg_main"]))
        btn_browse.clicked.connect(self.browse_output_dir_requested.emit)
        dir_row.addWidget(btn_browse)

        btn_open = QtWidgets.QPushButton("Open")
        btn_open.setIcon(get_icon("fa6s.arrow-up-right-from-square", THEME["fg_main"]))
        btn_open.clicked.connect(self.open_output_dir_requested.emit)
        dir_row.addWidget(btn_open)

        layout.addLayout(dir_row)

        options_row = QtWidgets.QHBoxLayout()

        anim_row = QtWidgets.QHBoxLayout()
        self.toggle_anim3d = ToggleSwitch()
        self.toggle_anim3d.toggled.connect(self._sync_3d_controls)
        anim_row.addWidget(self.toggle_anim3d)

        anim_label = QtWidgets.QLabel("3D Animation / Plot Outputs")
        anim_label.setToolTip("Maps to the backend --make-3d-plots flag.")
        anim_row.addWidget(anim_label)
        options_row.addLayout(anim_row)

        options_row.addSpacing(24)

        downsample_row = QtWidgets.QHBoxLayout()
        downsample_row.addWidget(QtWidgets.QLabel("3D Downsample:"))
        self.spin_downsample_3d = QtWidgets.QSpinBox()
        self.spin_downsample_3d.setRange(1, 1000)
        self.spin_downsample_3d.setValue(1)
        self.spin_downsample_3d.setToolTip("1 means full density. Higher values lighten 3D post-processing.")
        downsample_row.addWidget(self.spin_downsample_3d)
        options_row.addLayout(downsample_row)

        options_row.addStretch(1)
        layout.addLayout(options_row)

        note = QtWidgets.QLabel(
            "Tabular/report outputs are currently managed by the backend engine. "
            "This page only exposes options that are actually consumed by the CLI."
        )
        note.setWordWrap(True)
        note.setStyleSheet(
            f"color: {THEME['fg_muted']}; font-size: 9pt; font-style: italic; margin-top: 4px;"
        )
        layout.addWidget(note)

        return group_box

    def _build_artifacts_card(self) -> QtWidgets.QGroupBox:
        """
        Create the Generated Artifacts information card.

        Shows what files will be created after a successful run and provides
        quick access to the output directory plus a file count refresh.
        """

        group_box = self._create_card("Generated Artifacts")
        layout = QtWidgets.QVBoxLayout(group_box)
        layout.setContentsMargins(20, 24, 20, 20)
        layout.setSpacing(12)

        info = QtWidgets.QLabel(
            "The following outputs are generated automatically after a successful propagation run:"
        )
        info.setWordWrap(True)
        info.setStyleSheet(f"color: {THEME['fg_muted']};")
        layout.addWidget(info)

        always_items = [
            "Altitude History Plot (PNG)",
            "Ground Track Plot (PNG)",
            "Orbital Elements Timeseries (PNG)",
            "PDF Mission Report (PDF)",
        ]
        for item_text in always_items:
            row = QtWidgets.QHBoxLayout()
            dot = QtWidgets.QLabel("•")
            dot.setStyleSheet(f"color: {THEME['success']}; font-size: 12pt;")
            dot.setFixedWidth(18)
            lbl = QtWidgets.QLabel(item_text)
            lbl.setStyleSheet(f"color: {THEME['fg_soft']};")
            row.addWidget(dot)
            row.addWidget(lbl)
            row.addStretch()
            layout.addLayout(row)

        # 3D plot row (controlled by toggle above)
        row_3d = QtWidgets.QHBoxLayout()
        dot_3d = QtWidgets.QLabel("•")
        dot_3d.setStyleSheet(f"color: {THEME['accent']}; font-size: 12pt;")
        dot_3d.setFixedWidth(18)
        lbl_3d = QtWidgets.QLabel("3D Orbit Plot (PNG) — enabled by 3D Animation toggle above")
        lbl_3d.setStyleSheet(f"color: {THEME['fg_muted']}; font-style: italic;")
        row_3d.addWidget(dot_3d)
        row_3d.addWidget(lbl_3d)
        row_3d.addStretch()
        layout.addLayout(row_3d)

        # Artifact file count + action buttons
        btn_row = QtWidgets.QHBoxLayout()

        btn_open_out = QtWidgets.QPushButton("Open Output Folder")
        btn_open_out.setIcon(get_icon("fa6s.folder-open", THEME["fg_main"]))
        btn_open_out.clicked.connect(self.open_output_dir_requested.emit)
        btn_row.addWidget(btn_open_out)

        btn_refresh_artifacts = QtWidgets.QPushButton("Refresh Artifacts")
        btn_refresh_artifacts.setIcon(get_icon("fa6s.rotate", THEME["fg_main"]))
        btn_refresh_artifacts.clicked.connect(self._scan_artifacts)
        btn_row.addWidget(btn_refresh_artifacts)

        btn_row.addStretch()

        self.lbl_artifact_count = QtWidgets.QLabel("No output directory scanned yet.")
        self.lbl_artifact_count.setStyleSheet(f"color: {THEME['fg_muted']}; font-size: 9pt;")
        btn_row.addWidget(self.lbl_artifact_count)

        layout.addLayout(btn_row)

        # Connect output dir changes to auto-refresh count
        return group_box

    def _scan_artifacts(self) -> None:
        """Scan the output directory for PNG/PDF artifacts and update the count label."""
        out_dir = self.ent_out_dir.text().strip() if hasattr(self, "ent_out_dir") else ""
        if not out_dir:
            self.lbl_artifact_count.setText("Output directory not set.")
            return
        p = Path(out_dir)
        if not p.exists():
            self.lbl_artifact_count.setText("Output directory does not exist yet.")
            return
        pngs = list(p.glob("*.png"))
        pdfs = list(p.glob("*.pdf"))
        total = len(pngs) + len(pdfs)
        self.lbl_artifact_count.setText(
            f"{total} artifact(s) found ({len(pngs)} PNG, {len(pdfs)} PDF)"
        )

    def _build_preview_card(self) -> QtWidgets.QGroupBox:
        """
        Create the command preview card used to inspect the generated CLI call.

        The text box is read-only by design. Editing the preview string directly
        would create a mismatch between what the page displays and what the host
        actually launches.
        """

        group_box = self._create_card("Execution Command")
        layout = QtWidgets.QVBoxLayout(group_box)
        layout.setContentsMargins(20, 24, 20, 20)
        layout.setSpacing(12)

        info = QtWidgets.QLabel("Command that will be sent to the propagation engine:")
        info.setStyleSheet(f"color: {THEME['fg_muted']};")
        layout.addWidget(info)

        self.txt_preview = QtWidgets.QPlainTextEdit()
        self.txt_preview.setReadOnly(True)
        self.txt_preview.setFixedHeight(120)
        layout.addWidget(self.txt_preview)

        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addStretch(1)

        btn_refresh = QtWidgets.QPushButton("Refresh")
        btn_refresh.setIcon(get_icon("fa6s.rotate", THEME["fg_main"]))
        btn_refresh.clicked.connect(self.refresh_preview_requested.emit)
        btn_row.addWidget(btn_refresh)

        btn_copy = QtWidgets.QPushButton("Copy")
        btn_copy.setIcon(get_icon("fa6s.copy", THEME["fg_main"]))
        btn_copy.clicked.connect(self.copy_preview_requested.emit)
        btn_row.addWidget(btn_copy)

        layout.addLayout(btn_row)
        return group_box

    def _sync_3d_controls(self, _checked: bool = False) -> None:
        """
        Keep dependent controls visually honest.

        The downsample factor only matters when 3D plot generation is enabled,
        so the spin box is disabled when that backend feature is off.
        """

        enabled = bool(self.toggle_anim3d.isChecked())
        self.spin_downsample_3d.setEnabled(enabled)

    # ------------------------------------------------------------------
    # Artifact Browser
    # ------------------------------------------------------------------

    _TYPE_FOR_SUFFIX: dict[str, str] = {
        ".png": "Plot",
        ".jpg": "Plot",
        ".jpeg": "Plot",
        ".pdf": "Report",
        ".csv": "CSV",
        ".json": "JSON",
        ".h5": "HDF5",
        ".hdf5": "HDF5",
        ".npz": "NPZ",
        ".npy": "NPY",
        ".txt": "Text",
        ".log": "Text",
    }

    # -------------------------------------------------------------------------
    # Artifact browser state
    # -------------------------------------------------------------------------
    _FILTER_TYPES: dict[str, set[str]] = {
        "All":     set(),   # empty = no filter
        "Plots":   {".png", ".jpg", ".jpeg", ".svg"},
        "Reports": {".pdf"},
        "Data":    {".csv", ".json", ".h5", ".hdf5", ".npz", ".npy"},
        "Logs":    {".txt", ".log"},
    }

    def _build_artifact_browser_card(self) -> QtWidgets.QGroupBox:
        """
        Render the enhanced per-file artifact browser.

        Features:
        - Optional recursive directory scan
        - File type filter (All / Plots / Reports / Data / Logs)
        - Sort by modified time descending by default
        - Open Latest Report / Plot shortcuts
        - Copy Selected Path action
        - Informative empty states
        """

        group_box = self._create_card("Artifact Browser")
        layout = QtWidgets.QVBoxLayout(group_box)
        layout.setContentsMargins(20, 24, 20, 20)
        layout.setSpacing(10)

        # --- Row 1: path + action buttons ---
        header_row = QtWidgets.QHBoxLayout()
        self.lbl_browser_out_dir = QtWidgets.QLabel("Output Directory: —")
        self.lbl_browser_out_dir.setStyleSheet(f"color: {THEME['fg_muted']};")
        header_row.addWidget(self.lbl_browser_out_dir, 1)

        btn_browser_refresh = QtWidgets.QPushButton("Refresh")
        btn_browser_refresh.setIcon(get_icon("fa6s.rotate", THEME["fg_main"]))
        btn_browser_refresh.clicked.connect(self._refresh_artifact_browser)
        header_row.addWidget(btn_browser_refresh)

        btn_browser_open_folder = QtWidgets.QPushButton("Open Folder")
        btn_browser_open_folder.setIcon(get_icon("fa6s.folder-open", THEME["fg_main"]))
        btn_browser_open_folder.clicked.connect(self.open_output_dir_requested.emit)
        header_row.addWidget(btn_browser_open_folder)

        layout.addLayout(header_row)

        # --- Row 2: filter + recursive controls ---
        filter_row = QtWidgets.QHBoxLayout()
        filter_row.addWidget(QtWidgets.QLabel("Filter:"))
        self.cb_artifact_filter = QtWidgets.QComboBox()
        self.cb_artifact_filter.addItems(list(self._FILTER_TYPES.keys()))
        self.cb_artifact_filter.setFixedWidth(100)
        self.cb_artifact_filter.currentTextChanged.connect(self._refresh_artifact_browser)
        filter_row.addWidget(self.cb_artifact_filter)

        self.chk_recursive_scan = QtWidgets.QCheckBox("Recursive scan")
        self.chk_recursive_scan.setToolTip(
            "Scan subdirectories for artifacts (useful when outputs are placed in run subfolders)"
        )
        self.chk_recursive_scan.toggled.connect(self._refresh_artifact_browser)
        filter_row.addWidget(self.chk_recursive_scan)

        filter_row.addStretch(1)

        btn_latest_report = QtWidgets.QPushButton("Open Latest Report")
        btn_latest_report.setIcon(get_icon("fa6s.file-pdf", THEME["fg_main"]))
        btn_latest_report.clicked.connect(self._open_latest_report)
        self.btn_latest_report = btn_latest_report
        filter_row.addWidget(btn_latest_report)

        btn_latest_plot = QtWidgets.QPushButton("Open Latest Plot")
        btn_latest_plot.setIcon(get_icon("fa6s.image", THEME["fg_main"]))
        btn_latest_plot.clicked.connect(self._open_latest_plot)
        self.btn_latest_plot = btn_latest_plot
        filter_row.addWidget(btn_latest_plot)

        layout.addLayout(filter_row)

        # --- Tree ---
        self.tree_artifacts = QtWidgets.QTreeWidget()
        self.tree_artifacts.setColumnCount(5)
        self.tree_artifacts.setHeaderLabels(["Name", "Type", "Size", "Modified", "Path"])
        self.tree_artifacts.setRootIsDecorated(False)
        self.tree_artifacts.setAlternatingRowColors(True)
        self.tree_artifacts.setUniformRowHeights(True)
        self.tree_artifacts.setSortingEnabled(True)
        # Sort by Modified (col 3) descending by default
        self.tree_artifacts.sortByColumn(3, QtCore.Qt.DescendingOrder)
        self.tree_artifacts.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.tree_artifacts.customContextMenuRequested.connect(self._on_artifacts_context_menu)
        self.tree_artifacts.itemDoubleClicked.connect(self._on_artifacts_open_selected)
        self.tree_artifacts.setMinimumHeight(200)
        # Hide the raw path column (just used as data)
        self.tree_artifacts.setColumnHidden(4, True)
        layout.addWidget(self.tree_artifacts)

        # --- Action row ---
        action_row = QtWidgets.QHBoxLayout()

        btn_open = QtWidgets.QPushButton("Open File")
        btn_open.setIcon(get_icon("fa6s.up-right-from-square", THEME["fg_main"]))
        btn_open.clicked.connect(self._on_artifacts_open_selected)
        action_row.addWidget(btn_open)

        btn_copy_path = QtWidgets.QPushButton("Copy Path")
        btn_copy_path.setIcon(get_icon("fa6s.copy", THEME["fg_main"]))
        btn_copy_path.clicked.connect(self._on_artifacts_copy_path)
        action_row.addWidget(btn_copy_path)

        action_row.addStretch(1)

        self.lbl_artifact_summary = QtWidgets.QLabel("No artifacts yet.")
        self.lbl_artifact_summary.setStyleSheet(
            f"color: {THEME['fg_muted']}; font-size: 9pt;"
        )
        action_row.addWidget(self.lbl_artifact_summary)

        layout.addLayout(action_row)

        # Wire auto-refresh when output dir changes
        try:
            self.ent_out_dir.editingFinished.connect(self._refresh_artifact_browser)
            self.ent_out_dir.textChanged.connect(self._on_out_dir_text_changed_for_browser)
        except Exception:
            pass

        QtCore.QTimer.singleShot(0, self._refresh_artifact_browser)
        return group_box

    def _on_out_dir_text_changed_for_browser(self, _text: str) -> None:
        # Avoid hammering disk on every keystroke; rely on editingFinished
        # combined with the Refresh button. Still update the displayed path.
        try:
            txt = self.ent_out_dir.text().strip()
            display = txt or "—"
            if len(display) > 70:
                display = "..." + display[-67:]
            self.lbl_browser_out_dir.setText(f"Output Directory: {display}")
        except Exception:
            pass

    def refresh_artifacts(self, output_dir: str) -> None:
        """
        Public API used by the host window when the output directory changes.

        Falls back gracefully if the directory does not exist yet.
        """
        try:
            if output_dir:
                self.ent_out_dir.setText(output_dir)
        except Exception:
            pass
        self._refresh_artifact_browser()

    def _refresh_artifact_browser(self, *_args) -> None:
        """Scan output dir (optionally recursive) and populate the tree."""
        try:
            self.tree_artifacts.setSortingEnabled(False)
            self.tree_artifacts.clear()
        except Exception:
            return

        out_dir_text = ""
        try:
            out_dir_text = self.ent_out_dir.text().strip()
        except Exception:
            pass

        display = out_dir_text or "—"
        if len(display) > 70:
            display = "..." + display[-67:]
        self.lbl_browser_out_dir.setText(f"Output Directory: {display}")

        if not out_dir_text:
            self.lbl_artifact_summary.setText("Output directory not set.")
            self._update_latest_buttons([], [])
            return

        out_dir = Path(out_dir_text)
        if not out_dir.exists() or not out_dir.is_dir():
            self.lbl_artifact_summary.setText("Output directory does not exist yet.")
            self._update_latest_buttons([], [])
            return

        # Determine scan depth
        recursive = False
        try:
            recursive = bool(self.chk_recursive_scan.isChecked())
        except Exception:
            pass

        # Determine active type filter
        active_filter: set[str] = set()
        try:
            filter_key = self.cb_artifact_filter.currentText()
            active_filter = self._FILTER_TYPES.get(filter_key, set())
        except Exception:
            pass

        try:
            if recursive:
                all_entries: List[Path] = [p for p in out_dir.rglob("*") if p.is_file()]
            else:
                all_entries = [p for p in out_dir.iterdir() if p.is_file()]
        except Exception as exc:
            self.lbl_artifact_summary.setText(f"Could not list directory: {exc}")
            self._update_latest_buttons([], [])
            return

        if not all_entries:
            self.lbl_artifact_summary.setText("No artifacts found.")
            self._update_latest_buttons([], [])
            return

        # Apply filter
        if active_filter:
            entries = [e for e in all_entries if e.suffix.lower() in active_filter]
        else:
            entries = all_entries

        if not entries:
            total = len(all_entries)
            self.lbl_artifact_summary.setText(
                f"Filter hides all artifacts ({total} total; change filter to see them)."
            )
            self._update_latest_buttons(all_entries, all_entries)
            return

        # Sort by mtime descending
        def _mtime(p: Path) -> float:
            try:
                return p.stat().st_mtime
            except Exception:
                return 0.0

        entries.sort(key=_mtime, reverse=True)

        plots: List[Path] = []
        reports: List[Path] = []
        data_count = 0

        for entry in entries:
            suffix = entry.suffix.lower()
            type_label = self._TYPE_FOR_SUFFIX.get(suffix, "File")
            try:
                stat = entry.stat()
                size_bytes = stat.st_size
                mtime_raw = stat.st_mtime
                mtime = datetime.fromtimestamp(mtime_raw).strftime("%Y-%m-%d %H:%M")
            except Exception:
                size_bytes = 0
                mtime = "?"
            size_str = self._format_size(size_bytes)

            # Show relative path when recursive scan is active
            if recursive:
                try:
                    display_name = str(entry.relative_to(out_dir))
                except Exception:
                    display_name = entry.name
            else:
                display_name = entry.name

            item = QtWidgets.QTreeWidgetItem(
                [display_name, type_label, size_str, mtime, str(entry)]
            )
            item.setData(0, QtCore.Qt.UserRole, str(entry))
            try:
                if type_label == "Plot":
                    item.setIcon(0, get_icon("fa6s.image", THEME["fg_main"]))
                    plots.append(entry)
                elif type_label == "Report":
                    item.setIcon(0, get_icon("fa6s.file-pdf", THEME["fg_main"]))
                    reports.append(entry)
                elif type_label in ("HDF5", "NPZ", "NPY", "CSV", "JSON"):
                    item.setIcon(0, get_icon("fa6s.database", THEME["fg_main"]))
                    data_count += 1
                else:
                    item.setIcon(0, get_icon("fa6s.file", THEME["fg_main"]))
            except Exception:
                pass
            self.tree_artifacts.addTopLevelItem(item)

        try:
            self.tree_artifacts.setSortingEnabled(True)
            self.tree_artifacts.sortByColumn(3, QtCore.Qt.DescendingOrder)
            for col in range(4):
                self.tree_artifacts.resizeColumnToContents(col)
        except Exception:
            pass

        shown = len(entries)
        total = len(all_entries)
        scan_note = " (recursive)" if recursive else ""
        filter_note = f" [{self.cb_artifact_filter.currentText()} filter]" if active_filter else ""
        self.lbl_artifact_summary.setText(
            f"{shown} / {total} artifact(s){scan_note}{filter_note}  —  "
            f"{len(plots)} plots, {len(reports)} reports, {data_count} data files"
        )
        self._update_latest_buttons(plots, reports)

    def _update_latest_buttons(self, plots: List[Path], reports: List[Path]) -> None:
        """Enable/disable the Open Latest buttons based on what was found."""
        try:
            self.btn_latest_plot.setEnabled(bool(plots))
            self.btn_latest_report.setEnabled(bool(reports))
        except Exception:
            pass
        try:
            self._latest_plot = plots[0] if plots else None
            self._latest_report = reports[0] if reports else None
        except Exception:
            self._latest_plot = None
            self._latest_report = None

    def _open_latest_report(self) -> None:
        p = getattr(self, "_latest_report", None)
        if p and Path(p).exists():
            self._open_path_externally(Path(p))

    def _open_latest_plot(self) -> None:
        p = getattr(self, "_latest_plot", None)
        if p and Path(p).exists():
            self._open_path_externally(Path(p))

    def _open_path_externally(self, p: Path) -> None:
        """Open *p* in the OS default viewer."""
        try:
            url = QtCore.QUrl.fromLocalFile(str(p))
            if QtGui.QDesktopServices.openUrl(url):
                return
        except Exception:
            pass
        try:
            if sys.platform == "win32":
                os.startfile(str(p))  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(p)])
            else:
                subprocess.Popen(["xdg-open", str(p)])
        except Exception:
            pass

    @staticmethod
    def _format_size(num_bytes: int) -> str:
        """Render a byte count as a short, human readable string."""
        try:
            size = float(num_bytes)
        except Exception:
            return "?"
        for unit in ("B", "KB", "MB", "GB", "TB"):
            if size < 1024.0 or unit == "TB":
                if unit == "B":
                    return f"{int(size)} {unit}"
                return f"{size:.1f} {unit}"
            size /= 1024.0
        return "?"

    def _on_artifacts_context_menu(self, pos: QtCore.QPoint) -> None:
        item = self.tree_artifacts.itemAt(pos)
        if item is None:
            return
        menu = QtWidgets.QMenu(self)
        act_open = menu.addAction("Open File")
        act_copy = menu.addAction("Copy Path")
        chosen = menu.exec(self.tree_artifacts.viewport().mapToGlobal(pos))
        if chosen is act_open:
            self._on_artifacts_open_selected()
        elif chosen is act_copy:
            self._on_artifacts_copy_path()

    def _selected_artifact_path(self) -> Optional[str]:
        item = self.tree_artifacts.currentItem()
        if item is None:
            return None
        data = item.data(0, QtCore.Qt.UserRole)
        return str(data) if data else None

    def _on_artifacts_open_selected(self, *_args) -> None:
        path = self._selected_artifact_path()
        if not path:
            return
        p = Path(path)
        if p.exists():
            self._open_path_externally(p)

    def _on_artifacts_copy_path(self, *_args) -> None:
        path = self._selected_artifact_path()
        if not path:
            return
        try:
            QtWidgets.QApplication.clipboard().setText(path)
        except Exception:
            pass


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")

    window = QtWidgets.QMainWindow()
    window.resize(1000, 700)

    def create_card(title: str) -> QtWidgets.QGroupBox:
        return QtWidgets.QGroupBox(title)

    page = ResultsExportPage(project_root=Path.cwd(), create_card=create_card)
    window.setCentralWidget(page)
    window.show()

    sys.exit(app.exec())
