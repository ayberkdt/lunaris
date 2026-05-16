# LUNAR_SIMULATION/ui_parts/mission_propagation_page.py
"""
Mission Propagation Page (Page 3)
- Mission Timeline (epoch + duration)
- Integrator settings (method + rtol + dt_out + max_step)
- Emits signals for opening advanced dialogs (solver / spacecraft) in MainWindow
"""

# =============================================================================
# 0.                                    IMPORTS 
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any

from PySide6 import QtCore, QtWidgets

from common.time_utils import (
    normalize_iso_datetime_to_utc_string,
    parse_iso_datetime_to_utc_datetime,
)

try:
    from .ui_commons import THEME, NumericDragLineEdit, QuickChip, get_icon
    from .solver_policy import (
        DEFAULT_ADAPTIVE_ATOL,
        DEFAULT_ADAPTIVE_RTOL,
        DEFAULT_MAX_STEP_S,
        DEFAULT_SOLVER_METHOD,
        choose_max_step,
        choose_solver_tolerances,
        coerce_positive_float,
        normalize_solver_config_object,
    )
except ImportError:
        # Only handle the "ran as a script" case; don't mask real import errors.
    if __name__ == "__main__" and (__package__ is None or __package__ == ""):
        import sys
        print("\n" + "!" * 60, file=sys.stderr)
        print("  [ERROR] This module must be run as part of the package.", file=sys.stderr)
        print("  When executed directly, relative imports like '.constants' fail.", file=sys.stderr)
        print("-" * 60, file=sys.stderr)
        print("  From the project root, run:", file=sys.stderr)
        print("\n      python -m ui_parts.mission_propagation_page\n", file=sys.stderr)
        print("!" * 60 + "\n", file=sys.stderr)
        raise SystemExit(2)
    raise


# =============================================================================
# 1.                             DATACLASSES
# =============================================================================

@dataclass
class UISolverConfig:
    """
    Mutable UI copy of the adaptive-solver settings.

    The defaults intentionally mirror the backend SSOT so a fresh session starts
    from a stable tolerance pair instead of an over-tight legacy value set.
    """

    rtol: float = DEFAULT_ADAPTIVE_RTOL
    atol: float = DEFAULT_ADAPTIVE_ATOL
    max_step: float = DEFAULT_MAX_STEP_S  # seconds


@dataclass
class UISpacecraftConfig:
    """Spacecraft physical properties."""
    mass_kg: float = 1000.0
    area_m2: float = 5.0
    cd: float = 2.2
    cr: float = 1.5


# =============================================================================
# 2.                           OPTIONAL DIALOGS 
# =============================================================================

class SolverSettingsDialog(QtWidgets.QDialog):
    """Advanced solver configuration dialog."""
    def __init__(self, parent: QtWidgets.QWidget, cfg: UISolverConfig):
        super().__init__(parent)
        self.setWindowTitle("Solver Configuration")
        self.setModal(True)
        self.resize(500, 400)
        self._cfg = cfg

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        self.setStyleSheet(f"background-color: {THEME['bg_space']}; color: {THEME['fg_main']};")

        header = QtWidgets.QLabel("Numerical Solver Settings")
        header.setStyleSheet(f"font-size: 16px; font-weight: bold; color: {THEME['accent']};")
        layout.addWidget(header)

        desc = QtWidgets.QLabel(
            "Configure integration tolerances for adaptive solvers. Blank or "
            "invalid values are normalized to a safe default pair before launch."
        )
        desc.setStyleSheet(f"color: {THEME['fg_muted']};")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        tol_frame = QtWidgets.QFrame()
        tol_frame.setStyleSheet(
            f"background-color: {THEME['bg_card']}; border-radius: 8px; border: 1px solid {THEME['border']};"
        )
        tol_layout = QtWidgets.QVBoxLayout(tol_frame)
        tol_layout.setContentsMargins(15, 15, 15, 15)
        tol_layout.setSpacing(12)

        rtol_row = QtWidgets.QHBoxLayout()
        rtol_row.addWidget(QtWidgets.QLabel("Relative Tolerance (rtol):"))
        self.ent_rtol = NumericDragLineEdit(f"{self._cfg.rtol:g}", step=1e-13, min_value=1e-20, max_value=1e-3, decimals=0)
        self.ent_rtol.setFixedWidth(140)
        rtol_row.addWidget(self.ent_rtol)
        rtol_row.addStretch()
        tol_layout.addLayout(rtol_row)

        atol_row = QtWidgets.QHBoxLayout()
        atol_row.addWidget(QtWidgets.QLabel("Absolute Tolerance (atol):"))
        self.ent_atol = NumericDragLineEdit(f"{self._cfg.atol:g}", step=1e-15, min_value=1e-30, max_value=1e-5, decimals=0)
        self.ent_atol.setFixedWidth(140)
        atol_row.addWidget(self.ent_atol)
        atol_row.addStretch()
        tol_layout.addLayout(atol_row)

        maxstep_row = QtWidgets.QHBoxLayout()
        maxstep_row.addWidget(QtWidgets.QLabel("Maximum Step Size:"))
        self.ent_maxstep = NumericDragLineEdit(f"{self._cfg.max_step:.1f}", step=10.0, min_value=0.1, max_value=86400.0, decimals=1)
        self.ent_maxstep.setFixedWidth(140)
        maxstep_row.addWidget(self.ent_maxstep)
        maxstep_row.addWidget(QtWidgets.QLabel("s"))
        maxstep_row.addStretch()
        tol_layout.addLayout(maxstep_row)

        layout.addWidget(tol_frame)
        layout.addStretch(1)

        btn_layout = QtWidgets.QHBoxLayout()
        btn_layout.addStretch()

        self.btn_cancel = QtWidgets.QPushButton("Cancel")
        self.btn_save = QtWidgets.QPushButton("Apply")

        for btn in (self.btn_cancel, self.btn_save):
            btn.setCursor(QtCore.Qt.PointingHandCursor)
            btn.setFixedHeight(32)
            btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {THEME['bg_entry']};
                    border: 1px solid {THEME['border']};
                    border-radius: 6px;
                    color: {THEME['fg_main']};
                    padding: 0 16px;
                }}
                QPushButton:hover {{
                    background-color: {THEME['border']};
                }}
            """)

        self.btn_save.setStyleSheet(self.btn_save.styleSheet() + f"""
            QPushButton {{
                background-color: {THEME['accent']};
                border: 1px solid {THEME['accent']};
            }}
            QPushButton:hover {{
                background-color: {THEME['accent_hov']};
            }}
        """)

        btn_layout.addWidget(self.btn_cancel)
        btn_layout.addWidget(self.btn_save)
        layout.addLayout(btn_layout)

        self.btn_cancel.clicked.connect(self.reject)
        self.btn_save.clicked.connect(self._on_save)

    def _on_save(self, _checked: bool = False) -> None:
        try:
            rtol_value, atol_value = choose_solver_tolerances(
                "DOP853 (Adaptive)",
                rtol=self.ent_rtol.text(),
                atol=self.ent_atol.text(),
            )
            self._cfg.rtol = rtol_value
            self._cfg.atol = atol_value
            self._cfg.max_step = choose_max_step(self.ent_maxstep.text()) or DEFAULT_MAX_STEP_S
        except ValueError:
            QtWidgets.QMessageBox.warning(self, "Invalid Input", "Please enter valid numeric values.")
            return
        self.accept()


class SpacecraftBusDialog(QtWidgets.QDialog):
    """Spacecraft physical properties configuration dialog."""
    def __init__(self, parent: QtWidgets.QWidget, cfg: UISpacecraftConfig):
        super().__init__(parent)
        self.setWindowTitle("Spacecraft Properties")
        self.setModal(True)
        self.resize(500, 400)
        self._cfg = cfg

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        self.setStyleSheet(f"background-color: {THEME['bg_space']}; color: {THEME['fg_main']};")

        header = QtWidgets.QLabel("Spacecraft Physical Properties")
        header.setStyleSheet(f"font-size: 16px; font-weight: bold; color: {THEME['accent']};")
        layout.addWidget(header)

        desc = QtWidgets.QLabel("Configure spacecraft mass, dimensions, and force coefficients.")
        desc.setStyleSheet(f"color: {THEME['fg_muted']};")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        prop_frame = QtWidgets.QFrame()
        prop_frame.setStyleSheet(
            f"background-color: {THEME['bg_card']}; border-radius: 8px; border: 1px solid {THEME['border']};"
        )
        prop_layout = QtWidgets.QGridLayout(prop_frame)
        prop_layout.setContentsMargins(15, 15, 15, 15)
        prop_layout.setVerticalSpacing(12)
        prop_layout.setHorizontalSpacing(20)

        prop_layout.addWidget(QtWidgets.QLabel("Wet Mass:"), 0, 0)
        self.ent_mass = NumericDragLineEdit(f"{self._cfg.mass_kg:.1f}", step=10.0, min_value=0.1, max_value=100000.0, decimals=1)
        self.ent_mass.setFixedWidth(140)
        prop_layout.addWidget(self.ent_mass, 0, 1)
        prop_layout.addWidget(QtWidgets.QLabel("kg"), 0, 2)

        prop_layout.addWidget(QtWidgets.QLabel("Cross-section Area:"), 1, 0)
        self.ent_area = NumericDragLineEdit(f"{self._cfg.area_m2:.2f}", step=0.1, min_value=0.01, max_value=1000.0, decimals=2)
        self.ent_area.setFixedWidth(140)
        prop_layout.addWidget(self.ent_area, 1, 1)
        prop_layout.addWidget(QtWidgets.QLabel("m^2"), 1, 2)

        prop_layout.addWidget(QtWidgets.QLabel("Drag Coefficient (C_D):"), 2, 0)
        self.ent_cd = NumericDragLineEdit(f"{self._cfg.cd:.2f}", step=0.1, min_value=0.1, max_value=5.0, decimals=2)
        self.ent_cd.setFixedWidth(140)
        prop_layout.addWidget(self.ent_cd, 2, 1)

        prop_layout.addWidget(QtWidgets.QLabel("Reflectivity Coefficient (C_R):"), 3, 0)
        self.ent_cr = NumericDragLineEdit(f"{self._cfg.cr:.2f}", step=0.1, min_value=0.0, max_value=3.0, decimals=2)
        self.ent_cr.setFixedWidth(140)
        prop_layout.addWidget(self.ent_cr, 3, 1)

        layout.addWidget(prop_frame)
        layout.addStretch(1)

        btn_layout = QtWidgets.QHBoxLayout()
        btn_layout.addStretch()

        self.btn_cancel = QtWidgets.QPushButton("Cancel")
        self.btn_save = QtWidgets.QPushButton("Apply")

        for btn in (self.btn_cancel, self.btn_save):
            btn.setCursor(QtCore.Qt.PointingHandCursor)
            btn.setFixedHeight(32)
            btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {THEME['bg_entry']};
                    border: 1px solid {THEME['border']};
                    border-radius: 6px;
                    color: {THEME['fg_main']};
                    padding: 0 16px;
                }}
                QPushButton:hover {{
                    background-color: {THEME['border']};
                }}
            """)

        self.btn_save.setStyleSheet(self.btn_save.styleSheet() + f"""
            QPushButton {{
                background-color: {THEME['accent']};
                border: 1px solid {THEME['accent']};
            }}
            QPushButton:hover {{
                background-color: {THEME['accent_hov']};
            }}
        """)

        btn_layout.addWidget(self.btn_cancel)
        btn_layout.addWidget(self.btn_save)
        layout.addLayout(btn_layout)

        self.btn_cancel.clicked.connect(self.reject)
        self.btn_save.clicked.connect(self._on_save)

    def _on_save(self, _checked: bool = False) -> None:
        try:
            self._cfg.mass_kg = float(self.ent_mass.text())
            self._cfg.area_m2 = float(self.ent_area.text())
            self._cfg.cd = float(self.ent_cd.text())
            self._cfg.cr = float(self.ent_cr.text())
        except ValueError:
            QtWidgets.QMessageBox.warning(self, "Invalid Input", "Please enter valid numeric values.")
            return
        self.accept()


# =============================================================================
# 3.                            PAGE WIDGET
# =============================================================================

class MissionPropagationPage(QtWidgets.QWidget):
    """
    Page 3: Mission Timeline + Integrator Settings.

    Exposes widgets:
      dt_epoch, ent_duration, cb_duration_unit,
      cb_integrator, ent_rtol, ent_dt_out, ent_max_step
    """

    solver_settings_requested = QtCore.Signal()
    spacecraft_settings_requested = QtCore.Signal()

    def __init__(
        self,
        mission_epoch: Optional[QtCore.QDateTime] = None,
        solver_cfg: Optional[UISolverConfig] = None,
        spacecraft_cfg: Optional[UISpacecraftConfig] = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)

        self.mission_epoch = self._coerce_qdatetime_to_utc(
            mission_epoch or QtCore.QDateTime.currentDateTimeUtc()
        )
        self.solver_cfg = solver_cfg if solver_cfg is not None else UISolverConfig()
        self.spacecraft_cfg = spacecraft_cfg if spacecraft_cfg is not None else UISpacecraftConfig()
        normalize_solver_config_object(self.solver_cfg)

        self._build_ui()

    @staticmethod
    def _coerce_qdatetime_to_utc(qdt: QtCore.QDateTime) -> QtCore.QDateTime:
        """
        Return a UTC-normalized `QDateTime` suitable for the epoch editor.

        The propagation UI is the human-facing source of truth for mission
        epochs, so we keep the widget explicitly in UTC rather than allowing a
        local-time display to masquerade as a backend UTC timestamp.
        """

        if not isinstance(qdt, QtCore.QDateTime) or not qdt.isValid():
            return QtCore.QDateTime.currentDateTimeUtc()
        return qdt.toUTC()

    @classmethod
    def _epoch_text_to_qdatetime(cls, epoch_text: str) -> QtCore.QDateTime:
        """
        Parse an ISO-like epoch string and return an explicit UTC `QDateTime`.

        Saved sessions may contain legacy naive strings (`YYYY-MM-DD HH:MM:SS`)
        or newer canonical strings ending in `Z`.  We normalize both forms
        through the shared civil-time helper before updating the widget.
        """

        canonical = normalize_iso_datetime_to_utc_string(epoch_text, precision=0)
        qdt = QtCore.QDateTime.fromString(
            canonical,
            QtCore.Qt.DateFormat.ISODate,
        )
        if not qdt.isValid():
            dt_utc = parse_iso_datetime_to_utc_datetime(epoch_text)
            qdt = QtCore.QDateTime.fromString(
                dt_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
                QtCore.Qt.DateFormat.ISODate,
            )
        return cls._coerce_qdatetime_to_utc(qdt)

    @staticmethod
    def _qdatetime_to_epoch_text(qdt: QtCore.QDateTime) -> str:
        """
        Serialize the epoch widget value to the canonical UTC wire format.

        The CLI/backend contract now uses explicit UTC (`...Z`) so the same run
        configuration cannot mean different absolute epochs on different
        operator machines.
        """

        qdt_utc = qdt.toUTC() if isinstance(qdt, QtCore.QDateTime) and qdt.isValid() else QtCore.QDateTime.currentDateTimeUtc()
        return normalize_iso_datetime_to_utc_string(
            qdt_utc.toString("yyyy-MM-ddTHH:mm:ss'Z'"),
            precision=0,
        )

    def _create_card(self, title: str) -> QtWidgets.QGroupBox:
        gb = QtWidgets.QGroupBox(title)
        gb.setStyleSheet(f"""
            QGroupBox {{
                background: {THEME['bg_card']};
                border: 1px solid {THEME['border']};
                border-radius: 14px;
                margin-top: 12px;
                padding-top: 8px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 10px;
                padding: 0 8px;
                color: {THEME['fg_soft']};
                font-weight: 700;
            }}
        """)
        return gb

    def _style_inner_group(self, group: QtWidgets.QGroupBox) -> None:
        """Apply a softer nested-card treatment for secondary sections."""
        group.setStyleSheet(f"""
            QGroupBox {{
                background: {THEME['bg_card_alt']};
                border: 1px solid {THEME['border_soft']};
                border-radius: 10px;
                margin-top: 12px;
                padding-top: 8px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 10px;
                padding: 0 8px;
                color: {THEME['fg_soft']};
                font-weight: 700;
            }}
        """)

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(20)

        layout.addWidget(self._group_mission_timeline())
        layout.addWidget(self._group_integrator_settings())

        btn_layout = QtWidgets.QHBoxLayout()
        btn_layout.addStretch()

        self.btn_solver_settings = QtWidgets.QPushButton("Advanced Solver Settings...")
        self.btn_solver_settings.setIcon(get_icon("fa6s.gear", THEME["fg_main"]))
        self.btn_solver_settings.clicked.connect(self.solver_settings_requested.emit)
        btn_layout.addWidget(self.btn_solver_settings)
        layout.addLayout(btn_layout)

        btn_layout2 = QtWidgets.QHBoxLayout()
        btn_layout2.addStretch()

        self.btn_spacecraft_settings = QtWidgets.QPushButton("Spacecraft Properties...")
        self.btn_spacecraft_settings.setIcon(get_icon("fa6s.rocket", THEME["fg_main"]))
        self.btn_spacecraft_settings.clicked.connect(self.spacecraft_settings_requested.emit)
        btn_layout2.addWidget(self.btn_spacecraft_settings)
        layout.addLayout(btn_layout2)

        layout.addStretch(1)

    # -------------------------------------------------------------------------
    # Timeline
    # -------------------------------------------------------------------------
    def _group_mission_timeline(self) -> QtWidgets.QGroupBox:
        gb = self._create_card("Mission Timeline")
        layout = QtWidgets.QVBoxLayout(gb)
        layout.setContentsMargins(20, 24, 20, 20)
        layout.setSpacing(20)

        intro = QtWidgets.QLabel(
            "Choose when the mission begins and how long the analysis should follow the spacecraft."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet(f"color: {THEME['fg_muted']};")
        layout.addWidget(intro)

        epoch_group = QtWidgets.QGroupBox("Start Epoch")
        self._style_inner_group(epoch_group)
        epoch_layout = QtWidgets.QVBoxLayout(epoch_group)

        self.dt_epoch = QtWidgets.QDateTimeEdit()
        self.dt_epoch.setTimeZone(QtCore.QTimeZone(b"UTC"))
        self.dt_epoch.setDateTime(self.mission_epoch)
        self.dt_epoch.setDisplayFormat("yyyy-MM-dd HH:mm:ss 'UTC'")
        self.dt_epoch.setCalendarPopup(True)
        self.dt_epoch.setToolTip(
            "Mission start time is kept in UTC so saved runs, results, and reports stay aligned."
        )
        self.dt_epoch.setStyleSheet(f"""
            QDateTimeEdit {{
                background: {THEME['bg_entry']};
                border: 1px solid {THEME['border']};
                border-radius: 9px;
                padding: 7px 10px;
                color: {THEME['fg_main']};
            }}
        """)
        epoch_layout.addWidget(self.dt_epoch)

        utc_note = QtWidgets.QLabel(
            "Start time is shown in UTC so mission plans, saved sessions, and generated reports all refer to the same moment."
        )
        utc_note.setWordWrap(True)
        utc_note.setStyleSheet(f"color: {THEME['fg_muted']}; font-size: 9pt;")
        epoch_layout.addWidget(utc_note)

        layout.addWidget(epoch_group)

        duration_group = QtWidgets.QGroupBox("Propagation Duration")
        self._style_inner_group(duration_group)
        duration_layout = QtWidgets.QVBoxLayout(duration_group)

        duration_note = QtWidgets.QLabel(
            "Set the full analysis window. Quick presets help you move between short checks and long mission studies."
        )
        duration_note.setWordWrap(True)
        duration_note.setStyleSheet(f"color: {THEME['fg_muted']}; font-size: 9pt;")
        duration_layout.addWidget(duration_note)

        value_row = QtWidgets.QHBoxLayout()
        self.ent_duration = NumericDragLineEdit("10.0", step=0.1, min_value=0.001, decimals=2)
        value_row.addWidget(self.ent_duration, 1)

        self.cb_duration_unit = QtWidgets.QComboBox()
        self.cb_duration_unit.addItems(["Days", "Hours"])
        value_row.addWidget(self.cb_duration_unit)

        duration_layout.addLayout(value_row)

        presets_row = QtWidgets.QHBoxLayout()
        presets_row.setSpacing(8)
        for label, unit, value in [
            ("12h", "Hours", "12"),
            ("1d", "Days", "1"),
            ("10d", "Days", "10"),
            ("100d", "Days", "100"),
        ]:
            btn = QuickChip(label)
            btn.clicked.connect(lambda _=False, u=unit, v=value: self._set_duration_preset(u, v))
            presets_row.addWidget(btn)

        presets_row.addStretch()
        duration_layout.addLayout(presets_row)

        layout.addWidget(duration_group)
        return gb

    def _set_duration_preset(self, unit: str, value: str) -> None:
        self.cb_duration_unit.setCurrentText(unit)
        self.ent_duration.setText(str(value))

    # -------------------------------------------------------------------------
    # Integrator
    # -------------------------------------------------------------------------
    def _group_integrator_settings(self) -> QtWidgets.QGroupBox:
        gb = self._create_card("Numerical Integrator")
        layout = QtWidgets.QVBoxLayout(gb)
        layout.setContentsMargins(20, 24, 20, 20)
        layout.setSpacing(15)

        intro = QtWidgets.QLabel(
            "Select how the trajectory is propagated and how often mission results are written to the output set."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet(f"color: {THEME['fg_muted']};")
        layout.addWidget(intro)

        method_row = QtWidgets.QHBoxLayout()
        method_row.addWidget(QtWidgets.QLabel("Propagation Method:"))

        self.cb_integrator = QtWidgets.QComboBox()
        self.cb_integrator.addItems([
            "DOP853 (Adaptive)",
            "YOSHIDA4 (Symplectic)",
            "VV (Symplectic)",
        ])
        self.cb_integrator.currentTextChanged.connect(self._sync_integrator_widgets)

        method_row.addWidget(self.cb_integrator)
        method_row.addStretch()
        layout.addLayout(method_row)

        self.tolerance_group = QtWidgets.QGroupBox("Accuracy Target")
        self._style_inner_group(self.tolerance_group)
        tolerance_layout = QtWidgets.QVBoxLayout(self.tolerance_group)

        tolerance_note = QtWidgets.QLabel(
            "Recommended for adaptive solvers. Smaller values increase accuracy, but can make long runs slower."
        )
        tolerance_note.setWordWrap(True)
        tolerance_note.setStyleSheet(f"color: {THEME['fg_muted']}; font-size: 9pt;")
        tolerance_layout.addWidget(tolerance_note)

        tol_row = QtWidgets.QHBoxLayout()
        tol_row.addWidget(QtWidgets.QLabel("Relative Tolerance:"))
        self.ent_rtol = NumericDragLineEdit(
            f"{self.solver_cfg.rtol:g}",
            step=1e-13,
            min_value=1e-20,
            max_value=1e-3,
            decimals=0,
        )
        self.ent_rtol.setFixedWidth(140)
        tol_row.addWidget(self.ent_rtol)
        tol_row.addStretch()

        tolerance_layout.addLayout(tol_row)
        layout.addWidget(self.tolerance_group)

        step_group = QtWidgets.QGroupBox("Saved Output and Step Control")
        self._style_inner_group(step_group)
        step_layout = QtWidgets.QGridLayout(step_group)

        step_note = QtWidgets.QLabel(
            "Output interval controls how often trajectory samples are saved. Max step limits how far the solver can advance between evaluations."
        )
        step_note.setWordWrap(True)
        step_note.setStyleSheet(f"color: {THEME['fg_muted']}; font-size: 9pt;")
        step_layout.addWidget(step_note, 0, 0, 1, 3)

        self.ent_dt_out = NumericDragLineEdit("60.0", step=10.0, min_value=0.1)
        step_layout.addWidget(QtWidgets.QLabel("Output Interval:"), 1, 0)
        step_layout.addWidget(self.ent_dt_out, 1, 1)
        step_layout.addWidget(QtWidgets.QLabel("s"), 1, 2)

        self.ent_max_step = NumericDragLineEdit("", step=10.0, min_value=0.1)
        self.ent_max_step.setPlaceholderText("Auto (Nyquist)")
        step_layout.addWidget(QtWidgets.QLabel("Max Step:"), 2, 0)
        step_layout.addWidget(self.ent_max_step, 2, 1)
        step_layout.addWidget(QtWidgets.QLabel("s"), 2, 2)

        layout.addWidget(step_group)

        self._sync_integrator_widgets()
        return gb

    def _sync_integrator_widgets(self) -> None:
        txt = self.cb_integrator.currentText() or ""
        self.tolerance_group.setVisible("Adaptive" in txt)
        if "Adaptive" in txt and not self.ent_rtol.text().strip():
            self.ent_rtol.setText(f"{self.solver_cfg.rtol:g}")

    # -------------------------------------------------------------------------
    # State helpers (preset/save/load)
    # -------------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timeline": {
                "epoch": self._qdatetime_to_epoch_text(self.dt_epoch.dateTime()),
                "duration": self.ent_duration.text(),
                "unit": self.cb_duration_unit.currentText(),
            },
            "integrator": {
                "method": self.cb_integrator.currentText(),
                "rtol": self.ent_rtol.text(),
                "dt_out": self.ent_dt_out.text(),
                "max_step": self.ent_max_step.text(),
            },
        }

    def _apply_integrator_snapshot(self, integrator: Dict[str, Any]) -> None:
        """
        Apply integrator fields using the shared solver policy instead of raw text.

        Saved sessions may contain stale or invalid strings such as `rtol="0"` or
        `max_step="0.00"`. Normalizing here keeps the page display aligned with
        the solver config object and prevents the UI from visually falling back to
        broken values after restore.
        """

        method_label = str(
            integrator.get("method", self.cb_integrator.currentText() or DEFAULT_SOLVER_METHOD)
            or DEFAULT_SOLVER_METHOD
        )
        self.cb_integrator.setCurrentText(method_label)

        rtol_value, _ = choose_solver_tolerances(
            method_label,
            rtol=integrator.get("rtol", getattr(self.solver_cfg, "rtol", None)),
            atol=getattr(self.solver_cfg, "atol", None),
        )
        self.ent_rtol.setText(f"{float(rtol_value):g}")

        dt_out_text = str(integrator.get("dt_out", self.ent_dt_out.text() or "60.0") or "60.0")
        self.ent_dt_out.setText(dt_out_text)

        raw_max_step = integrator.get("max_step", self.ent_max_step.text())
        raw_max_step_text = "" if raw_max_step is None else str(raw_max_step).strip()
        if not raw_max_step_text:
            self.ent_max_step.setText("")
        elif coerce_positive_float(raw_max_step_text) is None:
            max_step_value = choose_max_step(
                raw_max_step_text,
                default=getattr(self.solver_cfg, "max_step", DEFAULT_MAX_STEP_S),
            )
            self.ent_max_step.setText("" if max_step_value is None else f"{float(max_step_value):g}")
        else:
            self.ent_max_step.setText(raw_max_step_text)

        self._sync_integrator_widgets()

    def sync_solver_widgets_from_config(self) -> None:
        """
        Refresh the visible solver fields from the shared mutable config object.

        The main window and the advanced solver dialog both edit `self.solver_cfg`.
        This helper keeps the lightweight page inputs visually consistent after
        those out-of-band updates.
        """

        self._apply_integrator_snapshot(
            {
                "method": self.cb_integrator.currentText() or DEFAULT_SOLVER_METHOD,
                "rtol": getattr(self.solver_cfg, "rtol", None),
                "dt_out": self.ent_dt_out.text() or "60.0",
                "max_step": getattr(self.solver_cfg, "max_step", None),
            }
        )

    def apply_dict(self, data: Dict[str, Any]) -> None:
        tl = data.get("timeline", {})
        epoch_str = tl.get("epoch", self._qdatetime_to_epoch_text(self.dt_epoch.dateTime()))
        try:
            epoch_qdt = self._epoch_text_to_qdatetime(str(epoch_str))
        except Exception:
            epoch_qdt = self._coerce_qdatetime_to_utc(self.dt_epoch.dateTime())
        self.dt_epoch.setDateTime(epoch_qdt)

        self.ent_duration.setText(str(tl.get("duration", self.ent_duration.text() or "10.0")))
        self.cb_duration_unit.setCurrentText(str(tl.get("unit", self.cb_duration_unit.currentText() or "Days")))

        self._apply_integrator_snapshot(data.get("integrator", {}))



# =============================================================================
# 4.                      TESTING PROPAGATION PAGE
# =============================================================================

if __name__ == "__main__":
    import sys

    # Start the application
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")

    # Create the test window
    window = QtWidgets.QMainWindow()
    window.setWindowTitle("Mission Propagation Page Test")
    window.resize(1000, 700)

    # Set the background color (to simulate a dark theme)
    window.setStyleSheet(
        f"background-color: {THEME['bg_space']}; color: {THEME['fg_main']};"
    )

    # Create shared configs (so dialogs edit the same objects)
    solver_cfg = UISolverConfig()
    spacecraft_cfg = UISpacecraftConfig()

    # Load the page
    page = MissionPropagationPage(
        mission_epoch=QtCore.QDateTime.currentDateTimeUtc(),
        solver_cfg=solver_cfg,
        spacecraft_cfg=spacecraft_cfg,
    )
    window.setCentralWidget(page)
    window.show()

    # Wire the "Advanced..." buttons to open dialogs (optional, but useful for testing)
    def open_solver_dialog():
        dlg = SolverSettingsDialog(window, solver_cfg)
        if dlg.exec() == QtWidgets.QDialog.Accepted:
            # Reflect updated config back into the page UI (optional)
            page.ent_rtol.setText(str(solver_cfg.rtol))
            print("[Solver cfg updated]", vars(solver_cfg))

    def open_spacecraft_dialog():
        dlg = SpacecraftBusDialog(window, spacecraft_cfg)
        if dlg.exec() == QtWidgets.QDialog.Accepted:
            print("[Spacecraft cfg updated]", vars(spacecraft_cfg))

    page.solver_settings_requested.connect(open_solver_dialog)
    page.spacecraft_settings_requested.connect(open_spacecraft_dialog)

    print("Test started...")
    print("Initial State:", page.to_dict())
    print("Initial Solver cfg:", vars(solver_cfg))
    print("Initial Spacecraft cfg:", vars(spacecraft_cfg))

    sys.exit(app.exec())
