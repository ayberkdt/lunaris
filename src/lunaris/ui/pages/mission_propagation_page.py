# ST_LRPS/ui_parts/mission_propagation_page.py
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
from typing import Any

from PySide6 import QtCore, QtWidgets

from lunaris.common.constants import DAY_S
from lunaris.common.time_utils import (
    normalize_iso_datetime_to_utc_string,
    parse_iso_datetime_to_utc_datetime,
)

try:
    from lunaris.ui.components import (
        ActionBar,
        FormGrid,
        InlineNotice,
        MetricRow,
        Section,
        SegmentedControl,
        Subsection,
    )
    from lunaris.ui.core.integrator_catalog import (
        IntegratorSpec,
        grouped_labels,
        spec_for_label,
    )
    from lunaris.ui.core.integrator_estimates import (
        accuracy_label,
        estimate_fixed_step_cost,
        validate_solver_inputs,
    )
    from lunaris.ui.core.solver_policy import (
        DEFAULT_ADAPTIVE_ATOL,
        DEFAULT_ADAPTIVE_RTOL,
        DEFAULT_MAX_STEP_S,
        DEFAULT_SOLVER_METHOD,
        choose_max_step,
        choose_solver_tolerances,
        coerce_positive_float,
        normalize_solver_config_object,
    )
    from lunaris.ui.core.ui_commons import (
        THEME,
        NoWheelComboBox,
        NumericDragLineEdit,
        QuickChip,
        StatusBadge,
        get_icon,
    )
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
        print("\n      python -m lunaris.ui.pages.mission_propagation_page\n", file=sys.stderr)
        print("!" * 60 + "\n", file=sys.stderr)
        raise SystemExit(2) from None
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
        self.setObjectName("settingsDialog")
        self.setModal(True)
        self.resize(500, 400)
        self.setMinimumSize(460, 360)
        self._cfg = cfg

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        header = QtWidgets.QLabel("Numerical Solver Settings")
        header.setObjectName("dialogTitle")
        layout.addWidget(header)

        desc = QtWidgets.QLabel(
            "Configure integration tolerances for adaptive solvers. Blank or "
            "invalid values are normalized to a safe default pair before launch."
        )
        desc.setObjectName("dialogDescription")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        tol_frame = QtWidgets.QFrame()
        tol_frame.setObjectName("section")
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
        self.ent_maxstep = NumericDragLineEdit(f"{self._cfg.max_step:.1f}", step=10.0, min_value=0.1, max_value=DAY_S, decimals=1)
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
            btn.setMinimumHeight(34)
        self.btn_cancel.setProperty("kind", "ghost")
        self.btn_save.setProperty("kind", "primary")

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
        self.setObjectName("settingsDialog")
        self.setModal(True)
        self.resize(500, 400)
        self.setMinimumSize(460, 360)
        self._cfg = cfg

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        header = QtWidgets.QLabel("Spacecraft Physical Properties")
        header.setObjectName("dialogTitle")
        layout.addWidget(header)

        desc = QtWidgets.QLabel("Configure spacecraft mass, dimensions, and force coefficients.")
        desc.setObjectName("dialogDescription")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        prop_frame = QtWidgets.QFrame()
        prop_frame.setObjectName("section")
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
            btn.setMinimumHeight(34)
        self.btn_cancel.setProperty("kind", "ghost")
        self.btn_save.setProperty("kind", "primary")

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
        mission_epoch: QtCore.QDateTime | None = None,
        solver_cfg: UISolverConfig | None = None,
        spacecraft_cfg: UISpacecraftConfig | None = None,
        parent: QtWidgets.QWidget | None = None,
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

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(DESIGN_TOKENS.layout.page_gap)

        layout.addWidget(self._group_mission_summary())
        layout.addWidget(self._group_mission_timeline())
        layout.addWidget(self._group_integrator_settings())

        # Spacecraft Bus moved to the Force Models page (it scales the
        # non-gravitational forces, so it belongs beside them); the Propagation
        # page keeps only the numerical solver settings.
        actions = ActionBar()
        self.btn_solver_settings = QtWidgets.QPushButton("Solver Settings")
        self.btn_solver_settings.setIcon(get_icon("fa6s.gear", THEME["fg_main"]))
        self.btn_solver_settings.setProperty("kind", "ghost")
        self.btn_solver_settings.setToolTip("Open advanced numerical solver tolerances.")
        self.btn_solver_settings.setAccessibleName("Advanced solver settings")
        self.btn_solver_settings.clicked.connect(self.solver_settings_requested.emit)

        actions.add_action(self.btn_solver_settings)
        layout.addWidget(actions)
        layout.addStretch(1)

        self._wire_summary_updates()
        self._update_summary()

    @staticmethod
    def _field_label(text: str) -> QtWidgets.QLabel:
        label = QtWidgets.QLabel(text)
        label.setObjectName("fieldLabel")
        label.setMinimumWidth(DESIGN_TOKENS.controls.form_label_width)
        return label

    @staticmethod
    def _unit_label(text: str) -> QtWidgets.QLabel:
        label = QtWidgets.QLabel(text)
        label.setObjectName("fieldUnit")
        return label

    @staticmethod
    def _row_container() -> tuple[QtWidgets.QWidget, QtWidgets.QHBoxLayout]:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(DESIGN_TOKENS.spacing.sm)
        return widget, layout

    def _group_mission_summary(self) -> QtWidgets.QWidget:
        section = Section(
            "Propagation Setup",
            "Current timeline, cadence, and numerical method.",
            elevated=True,
        )
        summary = MetricRow()
        self.lbl_summary_epoch = summary.add_metric("Epoch", "UTC")
        self.lbl_summary_duration = summary.add_metric("Window", "10 Days")
        self.lbl_summary_method = summary.add_metric("Method", "DOP853")
        self.lbl_summary_cadence = summary.add_metric("Cadence", "60 s")
        section.add_widget(summary)
        return section

    def _wire_summary_updates(self) -> None:
        self.dt_epoch.dateTimeChanged.connect(self._update_summary)
        self.ent_duration.textChanged.connect(self._update_summary)
        self.cb_duration_unit.currentTextChanged.connect(self._update_summary)
        self.cb_integrator.currentTextChanged.connect(self._update_summary)
        self.cb_output_mode.currentIndexChanged.connect(self._update_summary)
        self.ent_dt_out.textChanged.connect(self._update_summary)
        self.ent_samples_per_period.textChanged.connect(self._update_summary)
        self.ent_max_step.textChanged.connect(self._update_summary)

        # Live solver feedback (cost / accuracy / validation) reacts to the
        # inputs that change those estimates.
        for editor in (self.ent_max_step, self.ent_rtol, self.ent_atol, self.ent_duration):
            editor.textChanged.connect(self._update_solver_feedback)
        self.cb_duration_unit.currentTextChanged.connect(self._update_solver_feedback)

    def _update_summary(self) -> None:
        if not hasattr(self, "lbl_summary_epoch"):
            return

        qdt = self.dt_epoch.dateTime().toUTC()
        self.lbl_summary_epoch.setText(qdt.toString("yyyy-MM-dd HH:mm 'UTC'"))

        duration = self.ent_duration.text().strip() or "-"
        unit = self.cb_duration_unit.currentText().strip() or "Days"
        self.lbl_summary_duration.setText(f"{duration} {unit}")

        method = (self.cb_integrator.currentText() or "-").split(" ", 1)[0]
        self.lbl_summary_method.setText(method)

        mode = self.cb_output_mode.currentData() or "dt"
        if mode == "dt":
            cadence = f"{self.ent_dt_out.text().strip() or '-'} s"
        else:
            cadence = f"{self.ent_samples_per_period.text().strip() or '-'} pts/orbit"
        max_step = self.ent_max_step.text().strip()
        if max_step:
            cadence = f"{cadence} | max {max_step} s"
        self.lbl_summary_cadence.setText(cadence)

    # -------------------------------------------------------------------------
    # Timeline
    # -------------------------------------------------------------------------
    def _group_mission_timeline(self) -> QtWidgets.QWidget:
        section = Section(
            "Mission Timeline",
            "UTC epoch and analysis window.",
        )

        section.add_widget(
            InlineNotice(
                "Epoch is stored as explicit UTC and exported as ISO-8601 Z.",
                kind="info",
            )
        )

        form = FormGrid()
        self.dt_epoch = QtWidgets.QDateTimeEdit()
        self.dt_epoch.setTimeZone(QtCore.QTimeZone(b"UTC"))
        self.dt_epoch.setDateTime(self.mission_epoch)
        self.dt_epoch.setDisplayFormat("yyyy-MM-dd HH:mm:ss 'UTC'")
        self.dt_epoch.setCalendarPopup(True)
        self.dt_epoch.setToolTip(
            "Mission start time is kept in UTC so saved runs, results, and reports stay aligned."
        )
        self.dt_epoch.setAccessibleName("Mission start epoch in UTC")
        form.add_row("Start epoch", self.dt_epoch, "UTC")

        duration_field, value_row = self._row_container()
        self.ent_duration = NumericDragLineEdit("10.0", step=0.1, min_value=0.001, decimals=2)
        self.ent_duration.setAccessibleName("Propagation duration")
        value_row.addWidget(self.ent_duration, 1)

        self.cb_duration_unit = NoWheelComboBox()
        self.cb_duration_unit.addItems(["Days", "Hours"])
        self.cb_duration_unit.setAccessibleName("Propagation duration unit")
        value_row.addWidget(self.cb_duration_unit)
        form.add_row("Duration", duration_field)
        section.add_widget(form)

        presets, presets_row = self._row_container()
        presets_row.addWidget(self._field_label("Presets"))
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
        section.add_widget(presets)
        return section

    def _set_duration_preset(self, unit: str, value: str) -> None:
        self.cb_duration_unit.setCurrentText(unit)
        self.ent_duration.setText(str(value))

    # -------------------------------------------------------------------------
    # Integrator
    # -------------------------------------------------------------------------
    def _group_integrator_settings(self) -> QtWidgets.QWidget:
        section = Section(
            "Numerical Integrator",
            "Integration method, its characteristics, and the settings it uses.",
        )

        # --- Method selector (grouped by family) -----------------------------
        method_grid = FormGrid()
        self.cb_integrator = NoWheelComboBox()
        first_group = True
        for _family_label, labels in grouped_labels():
            if not first_group:
                self.cb_integrator.insertSeparator(self.cb_integrator.count())
            first_group = False
            for label in labels:
                self.cb_integrator.addItem(label)
        self.cb_integrator.setAccessibleName("Propagation method")
        self.cb_integrator.currentTextChanged.connect(self._sync_integrator_widgets)
        method_grid.add_row("Method", self.cb_integrator)
        section.add_widget(method_grid)

        # --- Characteristics card --------------------------------------------
        section.add_widget(self._build_integrator_card())

        # --- Adaptive-only accuracy controls ---------------------------------
        self.tolerance_group = Subsection("Adaptive Accuracy", "Relative tolerance for adaptive propagation.")
        tolerance_grid = FormGrid()
        self.ent_rtol = NumericDragLineEdit(
            f"{self.solver_cfg.rtol:g}",
            step=1e-13,
            min_value=1e-20,
            max_value=1e-3,
            decimals=0,
        )
        self.ent_rtol.setMaximumWidth(220)
        self.ent_rtol.setAccessibleName("Relative tolerance")
        tolerance_grid.add_row("Relative tolerance", self.ent_rtol)

        self.ent_atol = NumericDragLineEdit(
            f"{self.solver_cfg.atol:g}",
            step=1e-15,
            min_value=1e-30,
            max_value=1e-5,
            decimals=0,
        )
        self.ent_atol.setMaximumWidth(220)
        self.ent_atol.setAccessibleName("Absolute tolerance")
        tolerance_grid.add_row("Absolute tolerance", self.ent_atol)
        self.tolerance_group.add_widget(tolerance_grid)

        self.tol_feedback = InlineNotice("", kind="info")
        self.tol_feedback.setAccessibleName("Accuracy estimate")
        self.tolerance_group.add_widget(self.tol_feedback)
        section.add_widget(self.tolerance_group)

        # --- Step-size control (meaning depends on method family) ------------
        self.step_group = Subsection("Step Size")
        self.step_desc = QtWidgets.QLabel("")
        self.step_desc.setObjectName("sectionDescription")
        self.step_desc.setWordWrap(True)
        self.step_group.add_widget(self.step_desc)

        step_grid = QtWidgets.QGridLayout()
        step_grid.setContentsMargins(0, 0, 0, 0)
        step_grid.setHorizontalSpacing(DESIGN_TOKENS.spacing.md)
        step_grid.setVerticalSpacing(DESIGN_TOKENS.spacing.sm)
        step_grid.setColumnStretch(1, 1)

        self.ent_max_step = NumericDragLineEdit("", step=10.0, min_value=0.1)
        self.ent_max_step.setPlaceholderText("Auto (Nyquist)")
        self.ent_max_step.setText("")
        self.ent_max_step.setAccessibleName("Solver step size in seconds")
        self.lbl_max_step = self._field_label("Max step")
        self.lbl_max_step.setBuddy(self.ent_max_step)
        step_grid.addWidget(self.lbl_max_step, 0, 0, QtCore.Qt.AlignVCenter)
        step_grid.addWidget(self.ent_max_step, 0, 1)
        step_grid.addWidget(self._unit_label("s"), 0, 2, QtCore.Qt.AlignVCenter)

        step_widget = QtWidgets.QWidget()
        step_widget.setLayout(step_grid)
        self.step_group.add_widget(step_widget)

        self.step_feedback = InlineNotice("", kind="info")
        self.step_feedback.setAccessibleName("Step cost estimate")
        self.step_group.add_widget(self.step_feedback)
        section.add_widget(self.step_group)

        # --- Common output cadence (shared by every method) ------------------
        cadence_group = Subsection("Output Cadence", "Saved-sample density (applies to every integrator).")
        cadence_grid = QtWidgets.QGridLayout()
        cadence_grid.setContentsMargins(0, 0, 0, 0)
        cadence_grid.setHorizontalSpacing(DESIGN_TOKENS.spacing.md)
        cadence_grid.setVerticalSpacing(DESIGN_TOKENS.spacing.sm)
        cadence_grid.setColumnStretch(1, 1)

        mode_label = self._field_label("Output mode")
        self.cb_output_mode = NoWheelComboBox()
        self.cb_output_mode.addItem("Fixed Interval (dt)", "dt")
        self.cb_output_mode.addItem("Samples per Period", "spp")
        self.cb_output_mode.setVisible(False)

        self.output_mode_segments = SegmentedControl(["Fixed interval", "Samples/period"])
        self.output_mode_segments.setAccessibleName("Output sampling mode")
        self.output_mode_segments.current_changed.connect(self._set_output_mode_index)
        cadence_grid.addWidget(mode_label, 0, 0, QtCore.Qt.AlignVCenter)
        cadence_grid.addWidget(self.output_mode_segments, 0, 1, 1, 2)

        self.lbl_dt_out = self._field_label("Output interval")
        self.ent_dt_out = NumericDragLineEdit("60.0", step=10.0, min_value=0.1)
        self.ent_dt_out.setAccessibleName("Output interval in seconds")
        self.lbl_dt_out.setBuddy(self.ent_dt_out)
        self.lbl_dt_out_unit = self._unit_label("s")
        cadence_grid.addWidget(self.lbl_dt_out, 1, 0, QtCore.Qt.AlignVCenter)
        cadence_grid.addWidget(self.ent_dt_out, 1, 1)
        cadence_grid.addWidget(self.lbl_dt_out_unit, 1, 2, QtCore.Qt.AlignVCenter)

        self.lbl_spp = self._field_label("Samples per period")
        self.ent_samples_per_period = NumericDragLineEdit("360", step=10.0, min_value=2.0, max_value=2000.0, decimals=0)
        self.ent_samples_per_period.setAccessibleName("Samples saved per orbital period")
        self.lbl_spp.setBuddy(self.ent_samples_per_period)
        self.lbl_spp_unit = self._unit_label("pts")
        cadence_grid.addWidget(self.lbl_spp, 2, 0, QtCore.Qt.AlignVCenter)
        cadence_grid.addWidget(self.ent_samples_per_period, 2, 1)
        cadence_grid.addWidget(self.lbl_spp_unit, 2, 2, QtCore.Qt.AlignVCenter)

        cadence_widget = QtWidgets.QWidget()
        cadence_widget.setLayout(cadence_grid)
        cadence_group.add_widget(cadence_widget)

        self.cb_output_mode.currentIndexChanged.connect(self._sync_output_mode_widgets)
        self._sync_output_mode_widgets()
        section.add_widget(cadence_group)

        self._sync_integrator_widgets()
        return section

    def _build_integrator_card(self) -> QtWidgets.QWidget:
        """Card that surfaces the selected integrator's characteristics + use case."""
        card = Subsection("Method Characteristics")

        header, header_row = self._row_container()
        self.card_badge = StatusBadge("ADAPTIVE", kind="info")
        self.card_badge.setAccessibleName("Integrator family")
        self.card_title = QtWidgets.QLabel("")
        self.card_title.setObjectName("sectionTitle")
        self.card_title.setWordWrap(True)
        header_row.addWidget(self.card_badge, 0, QtCore.Qt.AlignVCenter)
        header_row.addWidget(self.card_title, 1, QtCore.Qt.AlignVCenter)
        card.add_widget(header)

        # Compact dashboard-style facts; values are kept short so the four cells
        # stay aligned like the summary row at the top of the page.
        self.card_metrics = MetricRow()
        self.card_val_order = self.card_metrics.add_metric("Order", "-")
        self.card_val_type = self.card_metrics.add_metric("Type", "-")
        self.card_val_step = self.card_metrics.add_metric("Step", "-")
        self.card_val_error = self.card_metrics.add_metric("Local error", "-")
        card.add_widget(self.card_metrics)

        self.card_reco = InlineNotice("", kind="info")
        self.card_reco.setAccessibleName("Recommended use")
        card.add_widget(self.card_reco)
        return card

    def _update_integrator_card(self, spec: IntegratorSpec | None) -> None:
        if not hasattr(self, "card_badge"):
            return
        if spec is None:
            self.card_badge.set_status("info", "INTEGRATOR")
            self.card_title.setText("")
            for value in (self.card_val_order, self.card_val_type, self.card_val_step, self.card_val_error):
                value.setText("-")
            self._set_notice_kind(self.card_reco, "info")
            self.card_reco.label.setText("Select an integration method.")
            return

        self.card_badge.set_status(spec.badge_kind, spec.family_label)
        self.card_title.setText(spec.title)
        self.card_val_order.setText(spec.order)
        self.card_val_type.setText(spec.metric_type)
        self.card_val_step.setText(spec.step_mode)
        self.card_val_error.setText(spec.metric_error)
        self._set_notice_kind(self.card_reco, spec.notice_kind)
        self.card_reco.label.setText(f"Recommended for: {spec.recommended}")

    @staticmethod
    def _set_notice_kind(notice: InlineNotice, kind: str) -> None:
        """Re-style an InlineNotice in place when its severity changes."""
        notice.setProperty("kind", kind)
        notice.style().unpolish(notice)
        notice.style().polish(notice)

    def _set_output_mode_index(self, index: int) -> None:
        if index != self.cb_output_mode.currentIndex() and 0 <= index < self.cb_output_mode.count():
            self.cb_output_mode.setCurrentIndex(index)
        else:
            self._sync_output_mode_widgets()

    def _sync_output_mode_widgets(self) -> None:
        """Show/hide output interval vs samples-per-period inputs based on selection."""
        if hasattr(self, "output_mode_segments"):
            index = self.cb_output_mode.currentIndex()
            if self.output_mode_segments.current_index() != index:
                self.output_mode_segments.set_current_index(index)

        mode = self.cb_output_mode.currentData() or "dt"
        is_dt = (mode == "dt")
        self.lbl_dt_out.setVisible(is_dt)
        self.ent_dt_out.setVisible(is_dt)
        self.lbl_dt_out_unit.setVisible(is_dt)
        self.lbl_spp.setVisible(not is_dt)
        self.ent_samples_per_period.setVisible(not is_dt)
        self.lbl_spp_unit.setVisible(not is_dt)
        self._update_summary()

    def _sync_integrator_widgets(self) -> None:
        txt = self.cb_integrator.currentText() or ""
        spec = spec_for_label(txt)
        is_adaptive = spec.is_adaptive if spec is not None else ("Adaptive" in txt)

        # Adaptive-only tolerance controls.
        self.tolerance_group.setVisible(is_adaptive)
        if is_adaptive and not self.ent_rtol.text().strip():
            self.ent_rtol.setText(f"{self.solver_cfg.rtol:g}")

        # The step field means different things per family; relabel it so the
        # user knows whether they are setting a guard cap or the actual step.
        if hasattr(self, "step_group"):
            if is_adaptive:
                self.lbl_max_step.setText("Max step")
                self.ent_max_step.setPlaceholderText("Auto (Nyquist)")
                self.step_desc.setText(
                    "Optional cap on the adaptive solver step. Leave blank and the "
                    "engine picks a Nyquist-safe limit."
                )
            else:
                self.lbl_max_step.setText("Fixed step")
                self.ent_max_step.setPlaceholderText("Auto (Nyquist)")
                self.step_desc.setText(
                    "Integration step size for this fixed-step method. Leave blank to use "
                    "a Nyquist-safe step derived from the gravity field and orbit."
                )

        self._update_integrator_card(spec)
        self._update_solver_feedback()
        self._update_summary()

    def _duration_seconds(self) -> float | None:
        """Best-effort propagation duration in seconds from the timeline inputs."""
        try:
            value = float(self.ent_duration.text().strip())
        except (TypeError, ValueError):
            return None
        if value <= 0.0:
            return None
        unit = (self.cb_duration_unit.currentText() or "").strip().lower()
        factor = DAY_S if unit.startswith("day") else 3600.0
        return value * factor

    def _update_solver_feedback(self) -> None:
        """Refresh the live accuracy / cost / validation notices for the solver."""
        if not hasattr(self, "step_feedback"):
            return

        spec = spec_for_label(self.cb_integrator.currentText() or "")
        duration_s = self._duration_seconds()
        step_text = self.ent_max_step.text().strip()
        rtol_text = self.ent_rtol.text().strip()

        issues = validate_solver_inputs(
            spec,
            duration_s=duration_s,
            step_s=step_text or None,
            rtol=rtol_text or None,
        )
        errors = [msg for sev, msg in issues if sev == "error"]
        warnings = [msg for sev, msg in issues if sev == "warning"]
        cost = estimate_fixed_step_cost(spec, duration_s, step_text or None)

        is_adaptive = bool(spec and spec.is_adaptive)
        # Validation belongs with the field it concerns: tolerance issues live in
        # the tolerance group, step issues in the step group.
        primary = self.tol_feedback if is_adaptive else self.step_feedback
        if is_adaptive:
            default_kind, default_text = "info", f"Accuracy band: {accuracy_label(rtol_text)}."
        else:
            default_kind, default_text = "info", cost.summary

        if errors:
            self._set_notice_kind(primary, "error")
            primary.label.setText(errors[0])
        elif warnings:
            self._set_notice_kind(primary, "warning")
            text = warnings[0]
            if not is_adaptive and cost.mode == "fixed":
                text = f"{warnings[0]}  ({cost.summary})"
            primary.label.setText(text)
        else:
            self._set_notice_kind(primary, default_kind)
            primary.label.setText(default_text)

        # For adaptive methods the step field is just a cap; show what it implies.
        if is_adaptive:
            self._set_notice_kind(self.step_feedback, "info")
            self.step_feedback.label.setText(cost.summary)

    # -------------------------------------------------------------------------
    # State helpers (preset/save/load)
    # -------------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        output_mode = self.cb_output_mode.currentData() or "dt"
        return {
            "timeline": {
                "epoch": self._qdatetime_to_epoch_text(self.dt_epoch.dateTime()),
                "duration": self.ent_duration.text(),
                "unit": self.cb_duration_unit.currentText(),
                "samples_per_period": self.ent_samples_per_period.text(),
            },
            "integrator": {
                "method": self.cb_integrator.currentText(),
                "rtol": self.ent_rtol.text(),
                "atol": self.ent_atol.text(),
                "dt_out": self.ent_dt_out.text() if output_mode == "dt" else "",
                "max_step": self.ent_max_step.text(),
                "output_mode": output_mode,
                "samples_per_period": self.ent_samples_per_period.text(),
            },
        }

    def _apply_integrator_snapshot(self, integrator: dict[str, Any]) -> None:
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

        rtol_value, atol_value = choose_solver_tolerances(
            method_label,
            rtol=integrator.get("rtol", getattr(self.solver_cfg, "rtol", None)),
            atol=integrator.get("atol", getattr(self.solver_cfg, "atol", None)),
        )
        self.ent_rtol.setText(f"{float(rtol_value):g}")
        self.ent_atol.setText(f"{float(atol_value):g}")

        # Restore output sampling mode
        output_mode = str(integrator.get("output_mode", "dt") or "dt")
        idx = self.cb_output_mode.findData(output_mode)
        if idx >= 0:
            self.cb_output_mode.setCurrentIndex(idx)
        else:
            self.cb_output_mode.setCurrentIndex(0)

        samples_per_period_text = str(integrator.get("samples_per_period", self.ent_samples_per_period.text() or "360") or "360")
        self.ent_samples_per_period.setText(samples_per_period_text)

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
        self._sync_output_mode_widgets()

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
                "atol": getattr(self.solver_cfg, "atol", None),
                "dt_out": self.ent_dt_out.text() or "60.0",
                "max_step": getattr(self.solver_cfg, "max_step", None),
            }
        )

    def apply_dict(self, data: dict[str, Any]) -> None:
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
