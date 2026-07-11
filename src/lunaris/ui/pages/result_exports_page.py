# lunaris.ui.pages.result_exports_page
"""
Results & Export Page (UI Part)
===============================

This module encapsulates the output/export workflow for the Lunaris
desktop UI. Earlier revisions kept this page embedded directly inside the main
window, which made the window class responsible for both high-level process
orchestration and low-level widget layout. The goal of this module is to give
the output page the same level of ownership already present in the other
`lunaris.ui.widgets.*_page` modules.

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
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

try:
    from lunaris.ui.components.primitives import EmptyState, KeyValueList, Section
    from lunaris.ui.core.ui_commons import (
        THEME,
        NoWheelComboBox,
        NoWheelSpinBox,
        StatusBadge,
        ToggleSwitch,
        get_icon,
    )
    from lunaris.ui.theme.tokens import DESIGN_TOKENS
except ImportError:
    if __name__ == "__main__" and (__package__ is None or __package__ == ""):
        import sys

        print("\n" + "!" * 60, file=sys.stderr)
        print("  [ERROR] This module must be run as part of the package.", file=sys.stderr)
        print("  From the project root, run:", file=sys.stderr)
        print("\n      python -m lunaris.ui.pages.result_exports_page\n", file=sys.stderr)
        print("!" * 60 + "\n", file=sys.stderr)
        raise SystemExit(2) from None
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
        initial_state: OutputPageState | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._project_root = Path(project_root)
        self._create_card = create_card
        self._state = initial_state or OutputPageState(
            output_dir=str(self._project_root / "outputs" / "missions"),
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
        self.ent_out_dir.setText(state.output_dir or str(self._project_root / "outputs" / "missions"))
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
        # Surface styling comes from QPlainTextEdit#commandPreview; only the
        # error/ok state is toggled here via a dynamic property.
        self.txt_preview.setProperty("state", "error" if is_error else "ok")
        style = self.txt_preview.style()
        style.unpolish(self.txt_preview)
        style.polish(self.txt_preview)

    def _build_ui(self) -> None:
        """
        Build the full page layout.

        The page follows the same card-based visual language as the other UI
        parts so the main window can embed it directly without page-specific
        styling logic.
        """

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(DESIGN_TOKENS.layout.page_gap)

        layout.addWidget(self._build_output_config_card())
        layout.addWidget(self._build_diagnostics_card())
        layout.addWidget(self._build_artifacts_card())
        layout.addWidget(self._build_artifact_browser_card())
        layout.addWidget(self._build_preview_card())
        layout.addStretch(1)

    @staticmethod
    def _field_label(text: str) -> QtWidgets.QLabel:
        label = QtWidgets.QLabel(text)
        label.setObjectName("fieldLabel")
        return label

    def _build_output_config_card(self) -> QtWidgets.QFrame:
        """
        Create the directory/option card shown at the top of the page.

        The host is expected to connect the emitted signals to application-wide
        actions such as showing a directory picker or opening the selected path
        in the OS file explorer.
        """

        section = Section(
            "Output Destination",
            "Where mission plots, reports, and run metadata are written.",
        )
        layout = section.content_layout

        layout.addWidget(self._field_label("Results directory"))

        dir_row = QtWidgets.QHBoxLayout()
        self.ent_out_dir = QtWidgets.QLineEdit()
        self.ent_out_dir.setAccessibleName("Output directory")
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
        self.toggle_anim3d.setAccessibleName("Generate 3D animation and plot outputs")
        self.toggle_anim3d.toggled.connect(self._sync_3d_controls)
        anim_row.addWidget(self.toggle_anim3d)

        anim_label = QtWidgets.QLabel("3D Animation / Plot Outputs")
        anim_label.setToolTip("Maps to the backend --make-3d-plots flag.")
        anim_row.addWidget(anim_label)
        options_row.addLayout(anim_row)

        options_row.addSpacing(24)

        downsample_row = QtWidgets.QHBoxLayout()
        downsample_row.addWidget(QtWidgets.QLabel("3D downsample"))
        self.spin_downsample_3d = NoWheelSpinBox()
        self.spin_downsample_3d.setAccessibleName("3D downsample factor")
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
        note.setObjectName("fieldHint")
        layout.addWidget(note)

        return section

    def _build_diagnostics_card(self) -> QtWidgets.QFrame:
        """
        Create the post-run diagnostics card.

        Values come exclusively from the propagation engine's own diagnostics
        payload (emitted as a structured ``[DIAG]`` line and mirrored to
        ``run_diagnostics.json``). Nothing is estimated or invented here: until
        a run completes in this session the card shows an explicit empty state.
        """

        section = Section(
            "Run Diagnostics",
            "Numerical health of the most recent propagation in this session, "
            "as reported by the engine.",
        )
        layout = section.content_layout

        header_row = QtWidgets.QHBoxLayout()
        header_row.setSpacing(DESIGN_TOKENS.spacing.sm)
        self.badge_run_outcome = StatusBadge("NO RUN", kind="info")
        self.badge_run_outcome.setAccessibleName("Last run outcome")
        header_row.addWidget(self.badge_run_outcome, 0, QtCore.Qt.AlignVCenter)
        self.lbl_diag_context = QtWidgets.QLabel("")
        self.lbl_diag_context.setObjectName("statusLabel")
        header_row.addWidget(self.lbl_diag_context, 1)
        layout.addLayout(header_row)

        self.diag_empty = EmptyState(
            "No diagnostics yet",
            "Run a propagation to see wall time, solver effort, and "
            "conservation-drift metrics reported by the engine.",
        )
        layout.addWidget(self.diag_empty)

        self.diag_list = KeyValueList()
        self.diag_list.setVisible(False)
        layout.addWidget(self.diag_list)

        return section

    # Display order, labels, units, and formatting for the engine diagnostics
    # payload. Only keys present in the payload are rendered; nothing is
    # synthesized. Keys: see PropagationResult.diagnostics + the CLI [DIAG] line.
    _DIAG_FIELDS: tuple[tuple[str, str, str, str], ...] = (
        ("wall_time_s",              "Propagation wall time", "s",   "{:.3f}"),
        ("method",                   "Integrator",            "",    "{}"),
        ("degree",                   "SH degree",             "",    "{:.0f}"),
        ("nfev",                     "RHS evaluations",       "",    "{:,.0f}"),
        ("n_points",                 "Output samples",        "",    "{:,.0f}"),
        ("output_dt_s",              "Output interval",       "s",   "{:g}"),
        ("max_step_s",               "Max solver step",       "s",   "{:.3f}"),
        ("periapsis_alt_km",         "Osculating periapsis",  "km",  "{:.1f}"),
        ("recommended_degree",       "Recommended SH degree", "",    "{:.0f}"),
        ("kepler_energy_rel_drift",  "Energy drift (rel.)",   "",    "{:.3e}"),
        ("angmom_rel_drift",         "Ang. momentum drift (rel.)", "", "{:.3e}"),
        ("t_impact_s",               "Impact time",           "s",   "{:.1f}"),
        ("stop_reason",              "Stop reason",           "",    "{}"),
    )

    def set_run_diagnostics(self, payload: dict | None) -> None:
        """
        Render an engine diagnostics payload (or reset to the empty state).

        ``payload`` is the parsed ``[DIAG]`` JSON emitted by the run CLI. Only
        recognized, present keys are shown; unknown keys are ignored so a newer
        backend cannot break the panel.
        """

        # Rebuild the key/value grid from scratch each time.
        while self.diag_list.layout_grid.count():
            item = self.diag_list.layout_grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        if not payload:
            self.badge_run_outcome.set_status("info", "NO RUN")
            self.lbl_diag_context.setText("")
            self.diag_list.setVisible(False)
            self.diag_empty.setVisible(True)
            return

        rendered = 0
        for key, label, unit, fmt in self._DIAG_FIELDS:
            if key not in payload or payload[key] is None:
                continue
            try:
                text = fmt.format(payload[key])
            except (ValueError, TypeError):
                text = str(payload[key])
            if unit:
                text = f"{text} {unit}"
            self.diag_list.add_item(label, text)
            rendered += 1

        impacted = bool(payload.get("impacted", False))
        if impacted:
            self.badge_run_outcome.set_status("warning", "IMPACT")
        else:
            self.badge_run_outcome.set_status("success", "COMPLETED")

        method = str(payload.get("method", "") or "")
        wall = payload.get("wall_time_s")
        context = "Engine-reported values from the last completed run"
        if method:
            context += f" ({method})"
        if isinstance(wall, int | float):
            context += f", {wall:.2f} s wall time"
        self.lbl_diag_context.setText(context + ".")

        self.diag_empty.setVisible(rendered == 0)
        self.diag_list.setVisible(rendered > 0)

    def _build_artifacts_card(self) -> QtWidgets.QFrame:
        """
        Create the Generated Artifacts information card.

        Shows what files will be created after a successful run and provides
        quick access to the output directory plus a file count refresh.
        """

        section = Section("Generated Artifacts")
        layout = section.content_layout

        info = QtWidgets.QLabel(
            "The following outputs are generated after a successful propagation run:"
        )
        info.setWordWrap(True)
        info.setObjectName("fieldHint")
        layout.addWidget(info)

        always_items = [
            "Altitude History Plot (PNG)",
            "Ground Track Plot (PNG)",
            "Orbital Elements Timeseries (PNG)",
            "PDF Mission Report (PDF)",
        ]
        for item_text in always_items:
            row = QtWidgets.QHBoxLayout()
            # Semantic status marker: token-based local color (no component fit).
            dot = QtWidgets.QLabel("•")
            dot.setStyleSheet(f"color: {THEME['success']}; font-size: 12pt;")
            dot.setFixedWidth(18)
            lbl = QtWidgets.QLabel(item_text)
            row.addWidget(dot)
            row.addWidget(lbl)
            row.addStretch()
            layout.addLayout(row)

        # 3D plot row (controlled by toggle above)
        row_3d = QtWidgets.QHBoxLayout()
        # Semantic status marker: token-based local color (no component fit).
        dot_3d = QtWidgets.QLabel("•")
        dot_3d.setStyleSheet(f"color: {THEME['accent']}; font-size: 12pt;")
        dot_3d.setFixedWidth(18)
        lbl_3d = QtWidgets.QLabel("3D Orbit Plot (PNG) — enabled by 3D Animation toggle above")
        lbl_3d.setObjectName("fieldHint")
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
        self.lbl_artifact_count.setObjectName("statusLabel")
        btn_row.addWidget(self.lbl_artifact_count)

        layout.addLayout(btn_row)

        # Connect output dir changes to auto-refresh count
        return section

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

    def _build_preview_card(self) -> QtWidgets.QFrame:
        """
        Create the command preview card used to inspect the generated CLI call.

        The text box is read-only by design. Editing the preview string directly
        would create a mismatch between what the page displays and what the host
        actually launches.
        """

        section = Section(
            "Execution Command",
            "Exact CLI invocation that will be sent to the propagation engine.",
        )
        layout = section.content_layout

        self.txt_preview = QtWidgets.QPlainTextEdit()
        self.txt_preview.setObjectName("commandPreview")
        self.txt_preview.setAccessibleName("Execution command preview")
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
        return section

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

    def _build_artifact_browser_card(self) -> QtWidgets.QFrame:
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

        section = Section(
            "Artifact Browser",
            "Generated files in the results directory, newest first.",
        )
        layout = section.content_layout

        # --- Row 1: path + action buttons ---
        header_row = QtWidgets.QHBoxLayout()
        self.lbl_browser_out_dir = QtWidgets.QLabel("Output Directory: —")
        self.lbl_browser_out_dir.setObjectName("statusLabel")
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
        filter_row.addWidget(QtWidgets.QLabel("Filter"))
        self.cb_artifact_filter = NoWheelComboBox()
        self.cb_artifact_filter.setAccessibleName("Artifact type filter")
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
        self.tree_artifacts.setAccessibleName("Artifact browser file list")
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

        # Standardized empty/first-run state shown in place of the tree when
        # there is nothing to list (no dir, missing dir, or no matching files).
        self.artifact_empty = EmptyState(
            "No artifacts yet",
            "Run a mission to generate plots and reports here.",
        )
        self.artifact_empty.setMinimumHeight(200)
        self.artifact_empty.setVisible(False)
        layout.addWidget(self.artifact_empty)

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

        btn_copy_csv = QtWidgets.QPushButton("Copy CSV")
        btn_copy_csv.setIcon(get_icon("fa6s.table-list", THEME["fg_main"]))
        btn_copy_csv.setToolTip("Copy all listed artifacts to the clipboard as CSV")
        btn_copy_csv.clicked.connect(self._copy_artifacts_csv)
        action_row.addWidget(btn_copy_csv)

        # Ctrl+C over the tree copies the selected rows (TSV), else the whole list.
        copy_shortcut = QtGui.QShortcut(QtGui.QKeySequence.Copy, self.tree_artifacts)
        copy_shortcut.setContext(QtCore.Qt.WidgetShortcut)
        copy_shortcut.activated.connect(self._copy_artifacts_selection)

        action_row.addStretch(1)

        self.lbl_artifact_summary = QtWidgets.QLabel("No artifacts yet.")
        self.lbl_artifact_summary.setObjectName("statusLabel")
        action_row.addWidget(self.lbl_artifact_summary)

        layout.addLayout(action_row)

        # Wire auto-refresh when output dir changes
        try:
            self.ent_out_dir.editingFinished.connect(self._refresh_artifact_browser)
            self.ent_out_dir.textChanged.connect(self._on_out_dir_text_changed_for_browser)
        except Exception:
            pass

        QtCore.QTimer.singleShot(0, self._refresh_artifact_browser)
        return section

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

    def _set_artifact_empty(self, title: str, description: str) -> None:
        """Show the standardized empty-state in place of the (empty) tree."""
        self.artifact_empty.set_message(title, description)
        self.tree_artifacts.setVisible(False)
        self.artifact_empty.setVisible(True)

    def _show_artifact_tree(self) -> None:
        """Show the populated tree and hide the empty-state."""
        self.artifact_empty.setVisible(False)
        self.tree_artifacts.setVisible(True)

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
            self._set_artifact_empty(
                "No output directory",
                "Choose a results directory above to browse generated artifacts.",
            )
            self._update_latest_buttons([], [])
            return

        out_dir = Path(out_dir_text)
        if not out_dir.exists() or not out_dir.is_dir():
            self.lbl_artifact_summary.setText("Output directory does not exist yet.")
            self._set_artifact_empty(
                "Directory not created yet",
                "The output directory does not exist until a mission run writes to it.",
            )
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
                all_entries: list[Path] = [p for p in out_dir.rglob("*") if p.is_file()]
            else:
                all_entries = [p for p in out_dir.iterdir() if p.is_file()]
        except Exception as exc:
            self.lbl_artifact_summary.setText(f"Could not list directory: {exc}")
            self._set_artifact_empty("Could not read directory", str(exc))
            self._update_latest_buttons([], [])
            return

        if not all_entries:
            self.lbl_artifact_summary.setText("No artifacts found.")
            self._set_artifact_empty(
                "No artifacts yet",
                "Run a mission to generate plots and reports in this directory.",
            )
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
            self._set_artifact_empty(
                "No artifacts match this filter",
                f"{total} artifact(s) are hidden — switch the filter to 'All' to see them.",
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
        self._show_artifact_tree()

        plots: list[Path] = []
        reports: list[Path] = []
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

    def _update_latest_buttons(self, plots: list[Path], reports: list[Path]) -> None:
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

    def _selected_artifact_path(self) -> str | None:
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

    def _artifacts_to_csv(self, *, selected_only: bool = False) -> str:
        """Render the artifact tree (or just the selected rows) as CSV text."""
        import csv
        import io

        cols = range(self.tree_artifacts.columnCount())
        header = self.tree_artifacts.headerItem()
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow([header.text(c) if header else "" for c in cols])
        if selected_only:
            rows = self.tree_artifacts.selectedItems()
        else:
            rows = [
                self.tree_artifacts.topLevelItem(i)
                for i in range(self.tree_artifacts.topLevelItemCount())
            ]
        for item in rows:
            if item is not None:
                writer.writerow([item.text(c) for c in cols])
        return buffer.getvalue()

    def _copy_artifacts_csv(self, *_args) -> None:
        """Copy every listed artifact to the clipboard as CSV."""
        try:
            QtWidgets.QApplication.clipboard().setText(self._artifacts_to_csv())
        except Exception:
            pass

    def _copy_artifacts_selection(self) -> None:
        """Ctrl+C: copy selected rows as TSV, or the whole list when nothing is selected."""
        selected = self.tree_artifacts.selectedItems()
        if not selected:
            self._copy_artifacts_csv()
            return
        cols = range(self.tree_artifacts.columnCount())
        lines = ["\t".join(item.text(c) for c in cols) for item in selected]
        try:
            QtWidgets.QApplication.clipboard().setText("\n".join(lines))
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
