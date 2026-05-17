# LUNAR_SIMULATION/ui_parts/data_files_page.py
"""
Data & Files Page (UI)

This module defines the **DataPage** (and its lightweight state container) used by the
Lunar Simulation UI to configure external data sources required by the simulation.

Typical responsibilities
- Select and validate filesystem paths for:
  - LDEM/topography datasets (e.g., DEM tiles)
  - Albedo/reflectance datasets (optional)
  - SPICE kernels / kernel directories (optional, depending on runtime mode)
- Configure a small set of page-level options such as:
  - LDEM resolution / sampling (e.g., "ppd" – points per degree)
  - whether to reuse the LDEM directory for albedo data
- Provide a clean interface for the main UI controller:
  - `get_state()` returns a serializable snapshot (dataclass)
  - `apply_state(...)` (if present) restores a previous snapshot
  - any logging is delegated via a `log_message` callback passed in by the host

Design notes
- This page intentionally does **not** start or manage simulation processes.
  It only collects user inputs and exposes them in a structured form.
- UI styling and reusable controls (icons, theme colors, custom line edits / chips)
  are sourced from `ui_commons.py`.
- The module is written to be testable in isolation (via a small `__main__` block),
  with the host providing:
  - `project_root` (Path)
  - `normalize_path` helper
  - `log_message` function
  - `create_card` factory for consistent card styling

Project
Lunar Simulation Core (LunarSim) – UI components.
"""

# =============================================================================
# 0.                                    IMPORTS 
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from PySide6 import QtCore, QtWidgets


try:
    from ui_parts.ui_commons import THEME, get_icon, StatusBadge
except ImportError:
        # Only handle the "ran as a script" case; don't mask real import errors.
    if __name__ == "__main__" and (__package__ is None or __package__ == ""):
        import sys
        print("\n" + "!" * 60, file=sys.stderr)
        print("  [ERROR] This module must be run as part of the package.", file=sys.stderr)
        print("  When executed directly, relative imports like '.constants' fail.", file=sys.stderr)
        print("-" * 60, file=sys.stderr)
        print("  From the project root, run:", file=sys.stderr)
        print("\n      python -m ui_parts.data_files_page\n", file=sys.stderr)
        print("!" * 60 + "\n", file=sys.stderr)
        raise SystemExit(2)
    raise




# =============================================================================
# 1.                              DataPage
# =============================================================================

@dataclass
class DataFilesState:
    ldem_root: str = ""
    albedo_root: str = ""
    kernel_dir: str = ""
    ldem_ppd: int = 4
    use_ldem_for_albedo: bool = False


class DataPage(QtWidgets.QWidget):
    """
    Page 6: Data & Files configuration.
    Owns its widgets and state; MainWindow should not expect ent_ldem_root, etc. to exist on itself.
    """

    def __init__(
        self,
        *,
        project_root: Path,
        normalize_path: Callable[[str], str],
        log_message: Callable[[str], None],
        create_card: Callable[[str], QtWidgets.QGroupBox],
        initial_state: Optional[DataFilesState] = None,
        parent=None,
    ):
        super().__init__(parent)
        self._project_root = Path(project_root)
        self._normalize_path = normalize_path
        self._log_message = log_message
        self._create_card = create_card

        starting_state = initial_state or DataFilesState()
        self._state = starting_state
        self._build_ui()
        # `_build_ui()` wires a few signals that opportunistically snapshot page
        # state while controls are still empty. Re-apply the caller-provided
        # snapshot explicitly so construction cannot erase a restored session.
        self.apply_state(starting_state)

    # -------------------------------------------------------------------------
    # Public API (MainWindow uses these)
    # -------------------------------------------------------------------------
    def get_state(self) -> DataFilesState:
        """Read current UI -> state.

        Note: During early construction, some widgets may not exist yet (because
        groups are built sequentially). This method must therefore be robust to
        partial UI initialization.
        """
        ldem_root = self.ent_ldem_root.text().strip() if hasattr(self, "ent_ldem_root") else (self._state.ldem_root if self._state else "")
        albedo_root = self.ent_albedo_root.text().strip() if hasattr(self, "ent_albedo_root") else (self._state.albedo_root if self._state else "")
        kernel_dir = self.ent_kernel_dir.text().strip() if hasattr(self, "ent_kernel_dir") else (self._state.kernel_dir if self._state else "")
        ldem_ppd = int(self.spin_ldem_ppd.value()) if hasattr(self, "spin_ldem_ppd") else (int(self._state.ldem_ppd) if self._state else 4)
        use_ldem_for_albedo = bool(self.chk_use_ldem_for_albedo.isChecked()) if hasattr(self, "chk_use_ldem_for_albedo") else (bool(self._state.use_ldem_for_albedo) if self._state else False)

        st = DataFilesState(
            ldem_root=ldem_root,
            albedo_root=albedo_root,
            kernel_dir=kernel_dir,
            ldem_ppd=ldem_ppd,
            use_ldem_for_albedo=use_ldem_for_albedo,
        )

        # enforce coupling
        if st.use_ldem_for_albedo and st.ldem_root:
            st.albedo_root = st.ldem_root

        return st


    def apply_state(self, st: DataFilesState) -> None:
        """Apply state -> UI."""
        self._state = st

        self.ent_ldem_root.setText(st.ldem_root or "")
        self.spin_ldem_ppd.setValue(int(st.ldem_ppd) if st.ldem_ppd else 4)

        self.chk_use_ldem_for_albedo.setChecked(bool(st.use_ldem_for_albedo))

        # albedo root shown only if checkbox off
        self.ent_albedo_root.setText(st.albedo_root or "")
        self.ent_kernel_dir.setText(st.kernel_dir or "")

        self._sync_albedo_path()
        self.refresh_badges()

    # -------------------------------------------------------------------------
    # UI
    # -------------------------------------------------------------------------
    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(20)

        layout.addWidget(self._group_surface_topography())
        layout.addWidget(self._group_spice_kernels())

        layout.addStretch(1)

    def _group_surface_topography(self) -> QtWidgets.QGroupBox:
        gb = self._create_card("Surface & Topography (LDEM)")
        layout = QtWidgets.QVBoxLayout(gb)
        layout.setContentsMargins(20, 24, 20, 20)
        layout.setSpacing(15)

        layout.addWidget(QtWidgets.QLabel("LDEM Root Directory:"))

        ldem_row = QtWidgets.QHBoxLayout()
        self.ent_ldem_root = QtWidgets.QLineEdit("")
        self.ent_ldem_root.setReadOnly(True)
        self.ent_ldem_root.setPlaceholderText("Select LDEM root directory...")
        self.ent_ldem_root.setStyleSheet(f"""
            background: {THEME['bg_entry']};
            border: 1px solid {THEME['border']};
            border-radius: 6px;
            padding: 6px;
            color: {THEME['fg_main']};
        """)
        ldem_row.addWidget(self.ent_ldem_root, 1)

        btn_ldem_browse = QtWidgets.QPushButton("Browse...")
        btn_ldem_browse.setIcon(get_icon("fa6s.folder-open", THEME["fg_main"]))
        btn_ldem_browse.clicked.connect(self._browse_ldem_root)
        ldem_row.addWidget(btn_ldem_browse)

        btn_ldem_open = QtWidgets.QPushButton("Open")
        btn_ldem_open.setIcon(get_icon("fa6s.arrow-up-right-from-square", THEME["fg_main"]))
        btn_ldem_open.setFixedWidth(72)
        btn_ldem_open.clicked.connect(lambda: self._open_path(self.ent_ldem_root.text()))
        ldem_row.addWidget(btn_ldem_open)

        self.badge_ldem = StatusBadge("NOT SET", kind="warning")
        self.badge_ldem.setFixedWidth(80)
        ldem_row.addWidget(self.badge_ldem)

        layout.addLayout(ldem_row)

        # Connect path change to badge update
        self.ent_ldem_root.textChanged.connect(lambda _: self._update_badge(self.ent_ldem_root.text(), self.badge_ldem))

        # Resolution control
        res_row = QtWidgets.QHBoxLayout()
        res_row.addWidget(QtWidgets.QLabel("LDEM Resolution:"))

        self.spin_ldem_ppd = QtWidgets.QSpinBox()
        self.spin_ldem_ppd.setRange(1, 128)
        self.spin_ldem_ppd.setValue(4)
        self.spin_ldem_ppd.setSuffix(" PPD")
        self.spin_ldem_ppd.setFixedWidth(100)
        self.spin_ldem_ppd.setStyleSheet(f"""
            background: {THEME['bg_entry']};
            border: 1px solid {THEME['border']};
            border-radius: 6px;
            padding: 4px;
        """)
        self.spin_ldem_ppd.valueChanged.connect(lambda _: self._state_changed())
        res_row.addWidget(self.spin_ldem_ppd)

        res_row.addWidget(QtWidgets.QLabel("(Pixels Per Degree)"))
        res_row.addStretch()
        layout.addLayout(res_row)

        # Albedo path checkbox
        self.chk_use_ldem_for_albedo = QtWidgets.QCheckBox("Reuse LDEM directory for Albedo data")
        self.chk_use_ldem_for_albedo.setChecked(False)
        self.chk_use_ldem_for_albedo.setStyleSheet(f"color: {THEME['fg_main']};")
        self.chk_use_ldem_for_albedo.toggled.connect(self._sync_albedo_path)
        layout.addWidget(self.chk_use_ldem_for_albedo)

        # Albedo container (shown only when not using LDEM)
        self.albedo_container = QtWidgets.QWidget()
        albedo_layout = QtWidgets.QVBoxLayout(self.albedo_container)
        albedo_layout.setContentsMargins(0, 10, 0, 0)

        albedo_layout.addWidget(QtWidgets.QLabel("Albedo Root Directory:"))

        albedo_path_row = QtWidgets.QHBoxLayout()
        self.ent_albedo_root = QtWidgets.QLineEdit("")
        self.ent_albedo_root.setReadOnly(True)
        self.ent_albedo_root.setPlaceholderText("Select Albedo root directory...")
        self.ent_albedo_root.setStyleSheet(f"""
            background: {THEME['bg_entry']};
            border: 1px solid {THEME['border']};
            border-radius: 6px;
            padding: 6px;
            color: {THEME['fg_main']};
        """)
        albedo_path_row.addWidget(self.ent_albedo_root, 1)

        btn_albedo_browse = QtWidgets.QPushButton("Browse...")
        btn_albedo_browse.setIcon(get_icon("fa6s.folder-open", THEME["fg_main"]))
        btn_albedo_browse.clicked.connect(self._browse_albedo_root)
        albedo_path_row.addWidget(btn_albedo_browse)

        btn_albedo_open = QtWidgets.QPushButton("Open")
        btn_albedo_open.setIcon(get_icon("fa6s.arrow-up-right-from-square", THEME["fg_main"]))
        btn_albedo_open.setFixedWidth(72)
        btn_albedo_open.clicked.connect(lambda: self._open_path(self.ent_albedo_root.text()))
        albedo_path_row.addWidget(btn_albedo_open)

        self.badge_albedo = StatusBadge("NOT SET", kind="warning")
        self.badge_albedo.setFixedWidth(80)
        albedo_path_row.addWidget(self.badge_albedo)

        albedo_layout.addLayout(albedo_path_row)
        self.ent_albedo_root.textChanged.connect(lambda _: self._update_badge(self.ent_albedo_root.text(), self.badge_albedo))
        layout.addWidget(self.albedo_container)

        note = QtWidgets.QLabel(
            "ℹ️ LDEM (Lunar Digital Elevation Model) provides high-resolution topography "
            "for collision detection and surface visualization."
        )
        note.setStyleSheet(f"color: {THEME['fg_muted']}; font-size: 9pt; font-style: italic; margin-top: 10px;")
        note.setWordWrap(True)
        layout.addWidget(note)

        self._sync_albedo_path()
        return gb

    def _group_spice_kernels(self) -> QtWidgets.QGroupBox:
        gb = self._create_card("SPICE Kernels")
        layout = QtWidgets.QVBoxLayout(gb)
        layout.setContentsMargins(20, 24, 20, 20)
        layout.setSpacing(15)

        layout.addWidget(QtWidgets.QLabel("SPICE Kernel Directory:"))

        kernel_row = QtWidgets.QHBoxLayout()
        self.ent_kernel_dir = QtWidgets.QLineEdit("")
        self.ent_kernel_dir.setReadOnly(True)
        self.ent_kernel_dir.setPlaceholderText("Select SPICE kernel directory...")
        self.ent_kernel_dir.setStyleSheet(f"""
            background: {THEME['bg_entry']};
            border: 1px solid {THEME['border']};
            border-radius: 6px;
            padding: 6px;
            color: {THEME['fg_main']};
        """)
        kernel_row.addWidget(self.ent_kernel_dir, 1)

        btn_kernel_browse = QtWidgets.QPushButton("Browse...")
        btn_kernel_browse.setIcon(get_icon("fa6s.folder-open", THEME["fg_main"]))
        btn_kernel_browse.clicked.connect(self._browse_kernel_dir)
        kernel_row.addWidget(btn_kernel_browse)

        btn_kernel_open = QtWidgets.QPushButton("Open")
        btn_kernel_open.setIcon(get_icon("fa6s.arrow-up-right-from-square", THEME["fg_main"]))
        btn_kernel_open.setFixedWidth(72)
        btn_kernel_open.clicked.connect(lambda: self._open_path(self.ent_kernel_dir.text()))
        kernel_row.addWidget(btn_kernel_open)

        self.badge_kernel = StatusBadge("NOT SET", kind="warning")
        self.badge_kernel.setFixedWidth(80)
        kernel_row.addWidget(self.badge_kernel)

        layout.addLayout(kernel_row)
        self.ent_kernel_dir.textChanged.connect(lambda _: self._update_badge(self.ent_kernel_dir.text(), self.badge_kernel))

        note = QtWidgets.QLabel(
            "ℹ️ SPICE kernels provide planetary ephemerides, time conversions, and frame "
            "definitions for precise orbital calculations."
        )
        note.setStyleSheet(f"color: {THEME['fg_muted']}; font-size: 9pt; font-style: italic; margin-top: 10px;")
        note.setWordWrap(True)
        layout.addWidget(note)

        return gb

    # -------------------------------------------------------------------------
    # Callbacks
    # -------------------------------------------------------------------------
    def _browse_ldem_root(self, _checked: bool = False) -> None:
        current = self.ent_ldem_root.text().strip() or str(self._project_root)
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select LDEM Root Directory", current)
        if not path:
            return

        norm = self._normalize_path(path)
        self.ent_ldem_root.setText(norm)

        if self.chk_use_ldem_for_albedo.isChecked():
            self.ent_albedo_root.setText(norm)

        self._state_changed()
        self._log_message(f"[UI] LDEM root set to: {Path(path).name}")

    def _browse_albedo_root(self, _checked: bool = False) -> None:
        current = self.ent_albedo_root.text().strip() or str(self._project_root)
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Albedo Root Directory", current)
        if not path:
            return

        norm = self._normalize_path(path)
        self.ent_albedo_root.setText(norm)

        self._state_changed()
        self._log_message(f"[UI] Albedo root set to: {Path(path).name}")

    def _browse_kernel_dir(self, _checked: bool = False) -> None:
        current = self.ent_kernel_dir.text().strip() or str(self._project_root)
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select SPICE Kernel Directory", current)
        if not path:
            return

        norm = self._normalize_path(path)
        self.ent_kernel_dir.setText(norm)

        self._state_changed()
        self._log_message(f"[UI] SPICE kernel directory set to: {Path(path).name}")

    def _sync_albedo_path(self, _checked: bool = False) -> None:
        use_ldem = self.chk_use_ldem_for_albedo.isChecked()
        self.albedo_container.setVisible(not use_ldem)

        if use_ldem:
            ldem = self.ent_ldem_root.text().strip()
            if ldem:
                self.ent_albedo_root.setText(ldem)

        self._state_changed()

    def _update_badge(self, path_text: str, badge: "StatusBadge") -> None:
        """Update a path validity badge based on whether the path exists."""
        path_text = path_text.strip()
        if not path_text:
            badge.set_status("warning", "NOT SET")
        elif Path(path_text).exists():
            badge.set_status("success", "VALID")
        else:
            badge.set_status("error", "MISSING")

    def _open_path(self, path_text: str) -> None:
        """Open a directory path in the OS file explorer."""
        path_text = path_text.strip()
        if path_text and Path(path_text).exists():
            QtCore.QDesktopServices.openUrl(
                QtCore.QUrl.fromLocalFile(path_text)
            )

    def refresh_badges(self) -> None:
        """Re-check all path badges."""
        if hasattr(self, "badge_ldem"):
            self._update_badge(self.ent_ldem_root.text(), self.badge_ldem)
        if hasattr(self, "badge_albedo"):
            self._update_badge(self.ent_albedo_root.text(), self.badge_albedo)
        if hasattr(self, "badge_kernel"):
            self._update_badge(self.ent_kernel_dir.text(), self.badge_kernel)

    def _state_changed(self) -> None:
        # keep internal snapshot up to date
        self._state = self.get_state()


# =============================================================================
# 2.                     TESTING DATA & FILES PAGE
# =============================================================================

if __name__ == "__main__":
    import sys
    import dataclasses
    from pathlib import Path

    # Start the application
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")

    # Create the test window
    window = QtWidgets.QMainWindow()
    window.setWindowTitle("Data & Files Page Test")
    window.resize(1000, 700)

    # Set the background color (to simulate a dark theme)
    window.setStyleSheet(
        f"background-color: {THEME['bg_space']}; color: {THEME['fg_main']};"
    )

    # Helpers required by DataPage
    from ui_parts.ui_commons import find_project_root, normalize_path  # keep consistent with your imports

    def log_message(msg: str) -> None:
        print(msg)

    def create_card(title: str) -> QtWidgets.QGroupBox:
        gb = QtWidgets.QGroupBox(title)
        gb.setStyleSheet(f"""
            QGroupBox {{
                background-color: {THEME['bg_card']};
                border: 1px solid {THEME['border']};
                border-radius: 12px;
                margin-top: 12px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 14px;
                padding: 0 6px;
                color: {THEME['fg_main']};
                font-weight: 600;
            }}
        """)
        return gb

    # Load the page
    page = DataPage(
        project_root=find_project_root(),
        normalize_path=normalize_path,
        log_message=log_message,
        create_card=create_card,
        # initial_state=DataFilesState(ldem_root="", albedo_root="", kernel_dir="", ldem_ppd=4, use_ldem_for_albedo=True),
    )
    window.setCentralWidget(page)

    window.show()

    print("Test started...")
    print("Initial State:", dataclasses.asdict(page.get_state()))

    sys.exit(app.exec())
