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

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from PySide6 import QtCore, QtWidgets

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
