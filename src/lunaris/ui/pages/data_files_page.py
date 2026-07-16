# ST_LRPS/ui_parts/data_files_page.py
"""
Data & Files Page (UI)

This module defines the **DataPage** (and its lightweight state container) used by the
Lunaris Mission Studio UI to configure external data sources required by the simulation.

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
ST_LRPS Core – UI components.
"""

# =============================================================================
# 0.                                    IMPORTS
# =============================================================================

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from PySide6 import QtCore, QtWidgets

try:
    from lunaris.ui.components import Section
    from lunaris.ui.core.ui_commons import THEME, NoWheelSpinBox, StatusBadge, get_icon
    from lunaris.ui.theme.tokens import DESIGN_TOKENS
except ImportError:
        # Only handle the "ran as a script" case; don't mask real import errors.
    if __name__ == "__main__" and (__package__ is None or __package__ == ""):
        import sys
        print("\n" + "!" * 60, file=sys.stderr)
        print("  [ERROR] This module must be run as part of the package.", file=sys.stderr)
        print("  When executed directly, relative imports like '.constants' fail.", file=sys.stderr)
        print("-" * 60, file=sys.stderr)
        print("  From the project root, run:", file=sys.stderr)
        print("\n      python -m lunaris.ui.pages.data_files_page\n", file=sys.stderr)
        print("!" * 60 + "\n", file=sys.stderr)
        raise SystemExit(2) from None
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
        initial_state: DataFilesState | None = None,
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
        layout.setSpacing(DESIGN_TOKENS.layout.page_gap)

        layout.addWidget(self._group_surface_topography())
        layout.addWidget(self._group_spice_kernels())

        layout.addStretch(1)

    @staticmethod
    def _field_label(text: str) -> QtWidgets.QLabel:
        label = QtWidgets.QLabel(text)
        label.setObjectName("fieldLabel")
        return label

    def _path_row(
        self,
        entry: QtWidgets.QLineEdit,
        badge: StatusBadge,
        *,
        on_browse,
        on_open,
    ) -> QtWidgets.QHBoxLayout:
        """One data-source row: read-only path, Browse/Open actions, status badge.

        Buttons and badge size to their content — fixed pixel widths clipped
        "Open" and "CONTENT OK" on wider system fonts.
        """
        row = QtWidgets.QHBoxLayout()
        row.setSpacing(DESIGN_TOKENS.spacing.sm)
        row.addWidget(entry, 1)

        btn_browse = QtWidgets.QPushButton("Browse…")
        btn_browse.setIcon(get_icon("fa6s.folder-open", THEME["fg_main"]))
        btn_browse.clicked.connect(on_browse)
        row.addWidget(btn_browse)

        btn_open = QtWidgets.QPushButton("Open")
        btn_open.setIcon(get_icon("fa6s.arrow-up-right-from-square", THEME["fg_main"]))
        btn_open.setToolTip("Open this directory in the file explorer.")
        btn_open.clicked.connect(on_open)
        row.addWidget(btn_open)

        row.addWidget(badge, 0, QtCore.Qt.AlignVCenter)
        return row

    def _group_surface_topography(self) -> QtWidgets.QFrame:
        section = Section(
            "Surface & Topography (LDEM)",
            "High-resolution lunar elevation data used for terrain-aware impact "
            "detection and surface visualization.",
        )
        layout = section.content_layout

        layout.addWidget(self._field_label("LDEM root directory"))

        self.ent_ldem_root = QtWidgets.QLineEdit("")
        self.ent_ldem_root.setReadOnly(True)
        self.ent_ldem_root.setPlaceholderText("Select LDEM root directory…")
        self.ent_ldem_root.setAccessibleName("LDEM root directory")
        self.badge_ldem = StatusBadge("NOT SET", kind="warning")
        layout.addLayout(
            self._path_row(
                self.ent_ldem_root,
                self.badge_ldem,
                on_browse=self._browse_ldem_root,
                on_open=lambda: self._open_path(self.ent_ldem_root.text()),
            )
        )

        # Detail label under the LDEM path row
        self.lbl_ldem_detail = QtWidgets.QLabel("")
        self.lbl_ldem_detail.setObjectName("statusLabel")
        self.lbl_ldem_detail.setVisible(False)
        layout.addWidget(self.lbl_ldem_detail)

        # Connect path change to content-aware badge update
        self.ent_ldem_root.textChanged.connect(lambda _: self._update_ldem_badge())

        # Resolution control. The label stacks above the control like every
        # other field in this card: an inline left label here meant one card
        # mixed two form idioms, so the eye had to re-find the label position
        # halfway down.
        layout.addWidget(self._field_label("LDEM resolution"))

        res_row = QtWidgets.QHBoxLayout()
        res_row.setSpacing(DESIGN_TOKENS.spacing.sm)

        self.spin_ldem_ppd = NoWheelSpinBox()
        self.spin_ldem_ppd.setRange(1, 128)
        self.spin_ldem_ppd.setValue(4)
        self.spin_ldem_ppd.setSuffix(" ppd")
        self.spin_ldem_ppd.setMinimumWidth(110)
        self.spin_ldem_ppd.setAccessibleName("LDEM resolution in pixels per degree")
        self.spin_ldem_ppd.valueChanged.connect(lambda _: self._state_changed())
        res_row.addWidget(self.spin_ldem_ppd)

        # The spin box already renders the " ppd" suffix, so a "pixels per
        # degree" label beside it stated the unit twice. The expansion is worth
        # keeping for anyone who does not know the abbreviation, but it belongs
        # in the tooltip and the accessible name, not in a second visible chip.
        self.spin_ldem_ppd.setToolTip(
            "LDEM resolution in pixels per degree (ppd): samples of elevation "
            "data per degree of latitude/longitude."
        )
        res_row.addStretch()
        layout.addLayout(res_row)

        # Albedo path checkbox
        self.chk_use_ldem_for_albedo = QtWidgets.QCheckBox("Reuse LDEM directory for albedo data")
        self.chk_use_ldem_for_albedo.setChecked(False)
        self.chk_use_ldem_for_albedo.toggled.connect(self._sync_albedo_path)
        self.chk_use_ldem_for_albedo.toggled.connect(lambda _: self._update_albedo_badge())
        layout.addWidget(self.chk_use_ldem_for_albedo)

        # Albedo container (shown only when not using LDEM)
        self.albedo_container = QtWidgets.QWidget()
        albedo_layout = QtWidgets.QVBoxLayout(self.albedo_container)
        albedo_layout.setContentsMargins(0, DESIGN_TOKENS.spacing.sm, 0, 0)
        albedo_layout.setSpacing(DESIGN_TOKENS.spacing.sm)

        albedo_layout.addWidget(self._field_label("Albedo root directory"))

        self.ent_albedo_root = QtWidgets.QLineEdit("")
        self.ent_albedo_root.setReadOnly(True)
        self.ent_albedo_root.setPlaceholderText("Select albedo root directory…")
        self.ent_albedo_root.setAccessibleName("Albedo root directory")
        self.badge_albedo = StatusBadge("NOT SET", kind="warning")
        albedo_layout.addLayout(
            self._path_row(
                self.ent_albedo_root,
                self.badge_albedo,
                on_browse=self._browse_albedo_root,
                on_open=lambda: self._open_path(self.ent_albedo_root.text()),
            )
        )

        self.lbl_albedo_detail = QtWidgets.QLabel("")
        self.lbl_albedo_detail.setObjectName("statusLabel")
        self.lbl_albedo_detail.setVisible(False)
        albedo_layout.addWidget(self.lbl_albedo_detail)

        self.ent_albedo_root.textChanged.connect(lambda _: self._update_albedo_badge())
        layout.addWidget(self.albedo_container)

        self._sync_albedo_path()
        return section

    def _group_spice_kernels(self) -> QtWidgets.QFrame:
        section = Section(
            "SPICE Kernels",
            "Planetary ephemerides, time systems, and reference-frame definitions "
            "(LSK / SPK / PCK / frame kernels) used by the propagator.",
        )
        layout = section.content_layout

        layout.addWidget(self._field_label("SPICE kernel directory"))

        self.ent_kernel_dir = QtWidgets.QLineEdit("")
        self.ent_kernel_dir.setReadOnly(True)
        self.ent_kernel_dir.setPlaceholderText("Select SPICE kernel directory…")
        self.ent_kernel_dir.setAccessibleName("SPICE kernel directory")
        self.badge_kernel = StatusBadge("NOT SET", kind="warning")
        layout.addLayout(
            self._path_row(
                self.ent_kernel_dir,
                self.badge_kernel,
                on_browse=self._browse_kernel_dir,
                on_open=lambda: self._open_path(self.ent_kernel_dir.text()),
            )
        )

        self.lbl_kernel_detail = QtWidgets.QLabel("")
        self.lbl_kernel_detail.setObjectName("statusLabel")
        self.lbl_kernel_detail.setVisible(False)
        layout.addWidget(self.lbl_kernel_detail)

        self.ent_kernel_dir.textChanged.connect(lambda _: self._update_kernel_badge())

        return section

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

    # -------------------------------------------------------------------------
    # Content-aware validation helpers (Task 4)
    # -------------------------------------------------------------------------

    @staticmethod
    def _detect_ldem_content(root: Path) -> tuple[str, str]:
        """
        Return (kind, detail) for an LDEM root directory.

        Possible kinds: 'not_set', 'missing', 'content_ok', 'path_ok'.
        """
        if not root or not str(root).strip():
            return ("not_set", "")
        if not root.exists():
            return ("missing", "Directory not found")
        # Check for typical LDEM files
        patterns = ["*.lbl", "*.lbl.txt", "*.img", "ldem_*", "*.tif", "*.tiff"]
        found: list[str] = []
        for pat in patterns:
            hits = list(root.glob(pat))
            if hits:
                found.append(f"{len(hits)} {pat.strip('*').lstrip('.')} file(s)")
        if found:
            return ("content_ok", ", ".join(found[:2]))
        # Directory exists but no recognizable LDEM files
        all_files = list(root.iterdir())
        if all_files:
            return ("path_ok", f"{len(all_files)} file(s) found (no LDEM pattern matched)")
        return ("path_ok", "Directory is empty")

    @staticmethod
    def _detect_albedo_content(root: Path) -> tuple[str, str]:
        """Return (kind, detail) for an Albedo root directory."""
        if not root or not str(root).strip():
            return ("not_set", "")
        if not root.exists():
            return ("missing", "Directory not found")
        patterns = ["ldam_*", "*.img", "*.lbl", "albedo*", "*.tif"]
        found: list[str] = []
        for pat in patterns:
            hits = list(root.glob(pat))
            if hits:
                found.append(f"{len(hits)} {pat.strip('*').lstrip('.')} file(s)")
        if found:
            return ("content_ok", ", ".join(found[:2]))
        all_files = list(root.iterdir())
        if all_files:
            return ("path_ok", f"{len(all_files)} file(s) found (no albedo pattern matched)")
        return ("path_ok", "Directory is empty")

    @staticmethod
    def _detect_kernel_content(root: Path) -> tuple[str, str]:
        """Return (kind, detail) for a SPICE kernel directory."""
        if not root or not str(root).strip():
            return ("not_set", "")
        if not root.exists():
            return ("missing", "Directory not found")
        kernel_exts = {".bsp", ".tls", ".tpc", ".tf", ".bc", ".bpc"}
        found_by_ext: dict[str, int] = {}
        for p in root.rglob("*"):
            if p.suffix.lower() in kernel_exts:
                found_by_ext[p.suffix.lower()] = found_by_ext.get(p.suffix.lower(), 0) + 1
        if found_by_ext:
            total = sum(found_by_ext.values())
            detail = f"{total} kernel file(s): " + ", ".join(
                f"{cnt} {ext}" for ext, cnt in sorted(found_by_ext.items())
            )
            return ("content_ok", detail)
        all_files = list(root.iterdir())
        if all_files:
            return ("path_ok", f"{len(all_files)} file(s) found (no .bsp/.tls/.tpc kernels)")
        return ("path_ok", "Directory is empty")

    def _update_badge(self, path_text: str, badge: StatusBadge) -> None:
        """
        Update a path validity badge.
        'NOT SET' → path is blank.
        'MISSING' → path does not exist.
        'PATH OK' → directory exists but no recognized content detected.
        'CONTENT OK' → directory exists and recognized content was found.
        """
        path_text = path_text.strip()
        if not path_text:
            badge.set_status("warning", "NOT SET")
            return
        p = Path(path_text)
        if not p.exists():
            badge.set_status("error", "MISSING")
            return
        # Determine which badge this is by its object name or identity
        # (We call the right detection function from refresh_badges instead)
        badge.set_status("success", "PATH OK")

    def _update_ldem_badge(self) -> None:
        """Content-aware update for the LDEM badge + detail label."""
        path_text = self.ent_ldem_root.text().strip()
        p = Path(path_text) if path_text else Path("")
        kind, detail = self._detect_ldem_content(p)
        self._set_badge_from_kind(self.badge_ldem, kind)
        if hasattr(self, "lbl_ldem_detail"):
            self.lbl_ldem_detail.setText(detail or "")
            self.lbl_ldem_detail.setVisible(bool(detail))

    def _update_albedo_badge(self) -> None:
        """Content-aware update for the Albedo badge + detail label."""
        if self.chk_use_ldem_for_albedo.isChecked():
            ldem_text = self.ent_ldem_root.text().strip()
            p = Path(ldem_text) if ldem_text else Path("")
            kind, detail = self._detect_albedo_content(p)
            detail = f"(using LDEM dir)  {detail}" if detail else "(using LDEM dir)"
        else:
            path_text = self.ent_albedo_root.text().strip()
            p = Path(path_text) if path_text else Path("")
            kind, detail = self._detect_albedo_content(p)
        self._set_badge_from_kind(self.badge_albedo, kind)
        if hasattr(self, "lbl_albedo_detail"):
            self.lbl_albedo_detail.setText(detail or "")
            self.lbl_albedo_detail.setVisible(bool(detail))

    def _update_kernel_badge(self) -> None:
        """Content-aware update for the SPICE kernel badge + detail label."""
        path_text = self.ent_kernel_dir.text().strip()
        p = Path(path_text) if path_text else Path("")
        kind, detail = self._detect_kernel_content(p)
        self._set_badge_from_kind(self.badge_kernel, kind)
        if hasattr(self, "lbl_kernel_detail"):
            self.lbl_kernel_detail.setText(detail or "")
            self.lbl_kernel_detail.setVisible(bool(detail))

    @staticmethod
    def _set_badge_from_kind(badge: StatusBadge, kind: str) -> None:
        labels = {
            "not_set":   ("warning", "NOT SET"),
            "missing":   ("error",   "MISSING"),
            "path_ok":   ("info",    "PATH OK"),
            "content_ok": ("success", "CONTENT OK"),
        }
        status_kind, text = labels.get(kind, ("warning", "NOT SET"))
        badge.set_status(status_kind, text)

    def _open_path(self, path_text: str) -> None:
        """Open a directory path in the OS file explorer."""
        path_text = path_text.strip()
        if path_text and Path(path_text).exists():
            QtCore.QDesktopServices.openUrl(
                QtCore.QUrl.fromLocalFile(path_text)
            )

    def refresh_badges(self) -> None:
        """Re-check all path badges with content-aware validation."""
        if hasattr(self, "badge_ldem"):
            self._update_ldem_badge()
        if hasattr(self, "badge_albedo"):
            self._update_albedo_badge()
        if hasattr(self, "badge_kernel"):
            self._update_kernel_badge()

    def _state_changed(self) -> None:
        # keep internal snapshot up to date
        self._state = self.get_state()


# =============================================================================
# 2.                     TESTING DATA & FILES PAGE
# =============================================================================

if __name__ == "__main__":
    import dataclasses
    import sys
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
    from lunaris.ui.core.ui_commons import (  # keep consistent with your imports
        find_project_root,
        normalize_path,
    )

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
