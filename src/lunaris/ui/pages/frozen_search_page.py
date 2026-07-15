"""Frozen-orbit search UI page.

This page exposes the ``lunaris-frozen-search`` CLI from the desktop app without
running the staged search on the UI thread.  It owns a small QProcess runner,
command preview, validation state, and cancellation.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from lunaris.ui.components.primitives import FormGrid, InlineNotice, Section
from lunaris.ui.core.ui_commons import (
    THEME,
    NoWheelComboBox,
    NoWheelDoubleSpinBox,
    NoWheelSpinBox,
    find_project_root,
    get_icon,
    normalize_path,
)
from lunaris.ui.theme.tokens import DESIGN_TOKENS


@dataclass(slots=True)
class FrozenSearchPageState:
    out_dir: str
    n_samples: int = 10_000
    seed: int = 0
    sampling_method: str = "sobol_scrambled"
    a_lo_km: float = 1838.0
    a_hi_km: float = 2238.0
    e_lo: float = 0.0
    e_hi: float = 0.25
    i_lo_deg: float = 60.0
    i_hi_deg: float = 120.0
    screening_days: float = 7.0
    screening_degree: int = 8
    screening_dt_s: float = 60.0
    screening_output_dt_s: float = 3600.0
    screening_device: str = "auto"
    top_k: int = 10
    validation_days: float = 30.0
    validation_degree: int = 50
    validation_output_dt_s: float = 3600.0
    sensitivity_degree: int = 0
    domain_alt_min_km: float = 20.0
    domain_alt_max_km: float = 20_000.0
    perilune_safety_km: float = 20.0
    refine_top_n: int = 0
    refine_max_iterations: int = 60
    gravity_file: str = ""
    make_figures: bool = True
    resume: bool = True
    verbose: bool = True


class FrozenSearchPage(QtWidgets.QWidget):
    """Desktop launcher for the staged frozen-orbit search workflow."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.project_root = find_project_root()
        self._process: QtCore.QProcess | None = None
        self._build_ui()
        self._connect_validation()
        self._update_preview()

    def _build_ui(self) -> None:
        root = QtWidgets.QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(DESIGN_TOKENS.spacing.lg)

        left_scroll = QtWidgets.QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        left_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        left = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, DESIGN_TOKENS.spacing.sm, 0)
        left_layout.setSpacing(DESIGN_TOKENS.spacing.md)
        left_layout.addWidget(self._build_run_contract_section())
        left_layout.addWidget(self._build_sampling_section())
        left_layout.addWidget(self._build_screening_section())
        left_layout.addWidget(self._build_validation_section())
        left_layout.addStretch(1)
        left_scroll.setWidget(left)
        left_scroll.setMinimumWidth(430)
        root.addWidget(left_scroll, 6)

        right_scroll = QtWidgets.QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        right_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        right = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, DESIGN_TOKENS.spacing.sm, 0)
        right_layout.setSpacing(DESIGN_TOKENS.spacing.md)
        right_layout.addWidget(self._build_execution_section())
        right_layout.addWidget(self._build_command_section())
        right_layout.addStretch(1)
        right_scroll.setWidget(right)
        right_scroll.setMinimumWidth(360)
        root.addWidget(right_scroll, 4)

    def _build_run_contract_section(self) -> Section:
        section = Section(
            "Run Contract",
            "A resumable staged search: Sobol/LHS sampling, torch-SH screening, "
            "classical-SH validation, family JSON, and optional figures.",
            elevated=True,
        )
        section.content_layout.addWidget(
            InlineNotice(
                "Long searches should use a CUDA-enabled ML/HPC environment. "
                "The desktop page launches the same CLI contract and keeps the UI responsive.",
                "info",
            )
        )
        form = FormGrid()
        self.ent_out_dir = QtWidgets.QLineEdit(str(self.project_root / "outputs" / "frozen_search" / "ui_run"))
        # The FormGrid caption labels the row container, not the edit itself,
        # so the edit needs its own accessible name.
        self.ent_out_dir.setAccessibleName("Frozen search output directory")
        self.ent_out_dir.setClearButtonEnabled(True)
        browse = QtWidgets.QPushButton("Browse")
        browse.setIcon(get_icon("fa6s.folder-open", THEME["fg_muted"]))
        browse.clicked.connect(self._browse_out_dir)
        out_row = QtWidgets.QWidget()
        out_layout = QtWidgets.QHBoxLayout(out_row)
        out_layout.setContentsMargins(0, 0, 0, 0)
        out_layout.addWidget(self.ent_out_dir, 1)
        out_layout.addWidget(browse)
        form.add_row("Output directory", out_row, hint="Resumable stage files and report JSON are written here.")

        self.ent_gravity_file = QtWidgets.QLineEdit("")
        self.ent_gravity_file.setAccessibleName("Gravity coefficient file")
        self.ent_gravity_file.setClearButtonEnabled(True)
        gravity_browse = QtWidgets.QPushButton("Browse")
        gravity_browse.setIcon(get_icon("fa6s.file", THEME["fg_muted"]))
        gravity_browse.clicked.connect(self._browse_gravity_file)
        gravity_row = QtWidgets.QWidget()
        gravity_layout = QtWidgets.QHBoxLayout(gravity_row)
        gravity_layout.setContentsMargins(0, 0, 0, 0)
        gravity_layout.addWidget(self.ent_gravity_file, 1)
        gravity_layout.addWidget(gravity_browse)
        form.add_row("Gravity file", gravity_row, hint="Optional. Leave blank to use the default config gravity model.")

        # Checkbox rows carry their own sentence label; an additional left
        # caption would duplicate it ("Resume | Resume existing stage files").
        self.chk_resume = QtWidgets.QCheckBox("Resume existing stage files")
        self.chk_resume.setChecked(True)
        form.add_row("", self.chk_resume)
        self.chk_figures = QtWidgets.QCheckBox("Generate report figures")
        self.chk_figures.setChecked(True)
        form.add_row("", self.chk_figures)
        self.chk_verbose = QtWidgets.QCheckBox("Verbose run log")
        self.chk_verbose.setChecked(True)
        form.add_row("", self.chk_verbose)
        section.content_layout.addWidget(form)
        return section

    def _build_sampling_section(self) -> Section:
        section = Section("Sampling Space", "Candidate orbital-element bounds and sample count.", elevated=True)
        form = FormGrid()
        self.spin_n_samples = self._spin_int(1, 1_000_000, 10_000)
        form.add_row("Samples", self.spin_n_samples)
        self.spin_seed = self._spin_int(0, 2_147_483_647, 0)
        form.add_row("Seed", self.spin_seed)
        self.cb_sampling = NoWheelComboBox()
        for label, value in (
            ("Sobol scrambled", "sobol_scrambled"),
            ("Sobol deterministic", "sobol"),
            ("Latin hypercube", "lhs"),
        ):
            self.cb_sampling.addItem(label, value)
        form.add_row("Sampling method", self.cb_sampling)
        self.spin_a_lo = self._spin_float(1000.0, 10_000.0, 1838.0, suffix=" km")
        self.spin_a_hi = self._spin_float(1000.0, 10_000.0, 2238.0, suffix=" km")
        form.add_row("a lower", self.spin_a_lo)
        form.add_row("a upper", self.spin_a_hi)
        self.spin_e_lo = self._spin_float(0.0, 0.9, 0.0, decimals=4)
        self.spin_e_hi = self._spin_float(0.0, 0.9, 0.25, decimals=4)
        form.add_row("e lower", self.spin_e_lo)
        form.add_row("e upper", self.spin_e_hi)
        self.spin_i_lo = self._spin_float(0.0, 180.0, 60.0, suffix=" deg")
        self.spin_i_hi = self._spin_float(0.0, 180.0, 120.0, suffix=" deg")
        form.add_row("i lower", self.spin_i_lo)
        form.add_row("i upper", self.spin_i_hi)
        section.content_layout.addWidget(form)
        return section

    def _build_screening_section(self) -> Section:
        section = Section("Screening", "Fast broad search before classical validation.", elevated=True)
        form = FormGrid()
        self.spin_screen_days = self._spin_float(0.01, 365.0, 7.0, suffix=" d")
        form.add_row("Screening duration", self.spin_screen_days)
        self.spin_screen_degree = self._spin_int(1, 1800, 8)
        form.add_row("Screening degree", self.spin_screen_degree)
        self.spin_screen_dt = self._spin_float(1.0, 86_400.0, 60.0, suffix=" s")
        form.add_row("RK step", self.spin_screen_dt)
        self.spin_screen_out_dt = self._spin_float(1.0, 86_400.0, 3600.0, suffix=" s")
        form.add_row("Output cadence", self.spin_screen_out_dt)
        self.cb_screen_device = NoWheelComboBox()
        self.cb_screen_device.addItem("Auto", "auto")
        self.cb_screen_device.addItem("CPU", "cpu")
        self.cb_screen_device.addItem("CUDA", "cuda")
        form.add_row("Device", self.cb_screen_device)
        self.spin_top_k = self._spin_int(1, 100_000, 10)
        form.add_row("Top candidates", self.spin_top_k)
        section.content_layout.addWidget(form)
        return section

    def _build_validation_section(self) -> Section:
        section = Section("Validation & Guards", "Classical-SH validation, domain bounds, and optional refinement.", elevated=True)
        form = FormGrid()
        self.spin_validation_days = self._spin_float(0.01, 365.0, 30.0, suffix=" d")
        form.add_row("Validation duration", self.spin_validation_days)
        self.spin_validation_degree = self._spin_int(1, 1800, 50)
        form.add_row("Validation degree", self.spin_validation_degree)
        self.spin_validation_out_dt = self._spin_float(1.0, 86_400.0, 3600.0, suffix=" s")
        form.add_row("Validation cadence", self.spin_validation_out_dt)
        self.spin_sensitivity_degree = self._spin_int(0, 1800, 0)
        self.spin_sensitivity_degree.setSpecialValueText("Off")
        form.add_row("Sensitivity degree", self.spin_sensitivity_degree)
        self.spin_alt_min = self._spin_float(-10_000.0, 100_000.0, 20.0, suffix=" km")
        self.spin_alt_max = self._spin_float(-10_000.0, 1_000_000.0, 20_000.0, suffix=" km")
        form.add_row("Domain alt min", self.spin_alt_min)
        form.add_row("Domain alt max", self.spin_alt_max)
        self.spin_perilune_safety = self._spin_float(0.0, 10_000.0, 20.0, suffix=" km")
        form.add_row("Perilune safety", self.spin_perilune_safety)
        self.spin_refine_top_n = self._spin_int(0, 10_000, 0)
        self.spin_refine_top_n.setSpecialValueText("Off")
        form.add_row("Refine top N", self.spin_refine_top_n)
        self.spin_refine_iterations = self._spin_int(1, 10_000, 60)
        form.add_row("Refine iterations", self.spin_refine_iterations)
        section.content_layout.addWidget(form)
        return section

    def _build_execution_section(self) -> Section:
        section = Section("Run Frozen Search", elevated=True)
        self.notice_validation = InlineNotice("Configuration looks ready.", "ok")
        section.content_layout.addWidget(self.notice_validation)

        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        self.progress.setTextVisible(False)
        section.content_layout.addWidget(self.progress)

        self.txt_log = QtWidgets.QPlainTextEdit()
        self.txt_log.setObjectName("logConsole")
        self.txt_log.setAccessibleName("Frozen search process log")
        self.txt_log.setReadOnly(True)
        self.txt_log.setMinimumHeight(180)
        self.txt_log.setPlaceholderText("Frozen-search output appears here...")
        section.content_layout.addWidget(self.txt_log)

        btn_row = QtWidgets.QHBoxLayout()
        self.btn_run = QtWidgets.QPushButton("  Run Search")
        self.btn_run.setObjectName("primaryBtn")
        self.btn_run.setIcon(get_icon("fa6s.play", THEME["fg_main"]))
        self.btn_run.clicked.connect(self._run_search)
        self.btn_cancel = QtWidgets.QPushButton("  Cancel")
        self.btn_cancel.setIcon(get_icon("fa6s.stop", THEME["fg_muted"]))
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self.cancel_run)
        self.btn_open = QtWidgets.QPushButton("  Open Output")
        self.btn_open.setIcon(get_icon("fa6s.folder-open", THEME["fg_muted"]))
        self.btn_open.clicked.connect(self._open_output_dir)
        # Primary action on its own full-width row, secondary actions below:
        # three labeled buttons side by side exceed the rail's minimum width
        # and the layout clipped their text ("Run Searc…" / "Open Outp…").
        section.content_layout.addWidget(self.btn_run)
        btn_row.addWidget(self.btn_cancel, 1)
        btn_row.addWidget(self.btn_open, 1)
        section.content_layout.addLayout(btn_row)
        return section

    def _build_command_section(self) -> Section:
        section = Section("Command Preview", "The UI launches this command in a background QProcess.", elevated=True)
        self.txt_command = QtWidgets.QPlainTextEdit()
        self.txt_command.setObjectName("commandPreview")
        self.txt_command.setReadOnly(True)
        self.txt_command.setMinimumHeight(150)
        self.txt_command.setAccessibleName("Frozen search command preview")
        section.content_layout.addWidget(self.txt_command)
        return section

    def _connect_validation(self) -> None:
        widgets: list[QtCore.QObject] = [
            self.ent_out_dir,
            self.ent_gravity_file,
            self.chk_resume,
            self.chk_figures,
            self.chk_verbose,
            self.spin_n_samples,
            self.spin_seed,
            self.cb_sampling,
            self.spin_a_lo,
            self.spin_a_hi,
            self.spin_e_lo,
            self.spin_e_hi,
            self.spin_i_lo,
            self.spin_i_hi,
            self.spin_screen_days,
            self.spin_screen_degree,
            self.spin_screen_dt,
            self.spin_screen_out_dt,
            self.cb_screen_device,
            self.spin_top_k,
            self.spin_validation_days,
            self.spin_validation_degree,
            self.spin_validation_out_dt,
            self.spin_sensitivity_degree,
            self.spin_alt_min,
            self.spin_alt_max,
            self.spin_perilune_safety,
            self.spin_refine_top_n,
            self.spin_refine_iterations,
        ]
        for widget in widgets:
            if isinstance(widget, QtWidgets.QLineEdit):
                widget.textChanged.connect(self._update_preview)
            elif isinstance(widget, QtWidgets.QComboBox):
                widget.currentIndexChanged.connect(self._update_preview)
            elif isinstance(widget, QtWidgets.QAbstractSpinBox):
                widget.editingFinished.connect(self._update_preview)
                if isinstance(widget, QtWidgets.QSpinBox) or isinstance(widget, QtWidgets.QDoubleSpinBox):
                    widget.valueChanged.connect(self._update_preview)
            elif isinstance(widget, QtWidgets.QCheckBox):
                widget.toggled.connect(self._update_preview)

    def get_state(self) -> FrozenSearchPageState:
        return FrozenSearchPageState(
            out_dir=self.ent_out_dir.text().strip(),
            n_samples=int(self.spin_n_samples.value()),
            seed=int(self.spin_seed.value()),
            sampling_method=str(self.cb_sampling.currentData()),
            a_lo_km=float(self.spin_a_lo.value()),
            a_hi_km=float(self.spin_a_hi.value()),
            e_lo=float(self.spin_e_lo.value()),
            e_hi=float(self.spin_e_hi.value()),
            i_lo_deg=float(self.spin_i_lo.value()),
            i_hi_deg=float(self.spin_i_hi.value()),
            screening_days=float(self.spin_screen_days.value()),
            screening_degree=int(self.spin_screen_degree.value()),
            screening_dt_s=float(self.spin_screen_dt.value()),
            screening_output_dt_s=float(self.spin_screen_out_dt.value()),
            screening_device=str(self.cb_screen_device.currentData()),
            top_k=int(self.spin_top_k.value()),
            validation_days=float(self.spin_validation_days.value()),
            validation_degree=int(self.spin_validation_degree.value()),
            validation_output_dt_s=float(self.spin_validation_out_dt.value()),
            sensitivity_degree=int(self.spin_sensitivity_degree.value()),
            domain_alt_min_km=float(self.spin_alt_min.value()),
            domain_alt_max_km=float(self.spin_alt_max.value()),
            perilune_safety_km=float(self.spin_perilune_safety.value()),
            refine_top_n=int(self.spin_refine_top_n.value()),
            refine_max_iterations=int(self.spin_refine_iterations.value()),
            gravity_file=self.ent_gravity_file.text().strip(),
            make_figures=bool(self.chk_figures.isChecked()),
            resume=bool(self.chk_resume.isChecked()),
            verbose=bool(self.chk_verbose.isChecked()),
        )

    def validate_state(self, state: FrozenSearchPageState | None = None) -> tuple[bool, list[str]]:
        s = state or self.get_state()
        errors: list[str] = []
        if not s.out_dir:
            errors.append("Output directory is required.")
        if s.a_lo_km >= s.a_hi_km:
            errors.append("Semi-major axis lower bound must be below the upper bound.")
        if s.e_lo >= s.e_hi:
            errors.append("Eccentricity lower bound must be below the upper bound.")
        if s.i_lo_deg >= s.i_hi_deg:
            errors.append("Inclination lower bound must be below the upper bound.")
        if s.top_k > s.n_samples:
            errors.append("Top candidates cannot exceed the sample count.")
        if s.domain_alt_min_km >= s.domain_alt_max_km:
            errors.append("Domain altitude min must be below max.")
        if s.refine_top_n > s.top_k:
            errors.append("Refine top N cannot exceed top candidates.")
        if s.gravity_file and not Path(s.gravity_file).expanduser().exists():
            errors.append("Gravity file does not exist.")
        return not errors, errors

    def build_command(self) -> list[str]:
        s = self.get_state()
        cmd = [
            sys.executable,
            "-m",
            "lunaris.cli.frozen_search",
            "--out",
            normalize_path(s.out_dir),
            "--n-samples",
            str(s.n_samples),
            "--seed",
            str(s.seed),
            "--sampling-method",
            s.sampling_method,
            "--a-km",
            f"{s.a_lo_km:g}",
            f"{s.a_hi_km:g}",
            "--e",
            f"{s.e_lo:g}",
            f"{s.e_hi:g}",
            "--i-deg",
            f"{s.i_lo_deg:g}",
            f"{s.i_hi_deg:g}",
            "--screening-days",
            f"{s.screening_days:g}",
            "--screening-degree",
            str(s.screening_degree),
            "--screening-dt-s",
            f"{s.screening_dt_s:g}",
            "--screening-output-dt-s",
            f"{s.screening_output_dt_s:g}",
            "--screening-device",
            s.screening_device,
            "--top-k",
            str(s.top_k),
            "--validation-days",
            f"{s.validation_days:g}",
            "--validation-degree",
            str(s.validation_degree),
            "--validation-output-dt-s",
            f"{s.validation_output_dt_s:g}",
            "--domain-alt-min-km",
            f"{s.domain_alt_min_km:g}",
            "--domain-alt-max-km",
            f"{s.domain_alt_max_km:g}",
            "--perilune-safety-km",
            f"{s.perilune_safety_km:g}",
            "--refine-top-n",
            str(s.refine_top_n),
            "--refine-max-iterations",
            str(s.refine_max_iterations),
        ]
        if s.sensitivity_degree > 0:
            cmd.extend(["--sensitivity-degree", str(s.sensitivity_degree)])
        if s.gravity_file:
            cmd.extend(["--gravity-file", normalize_path(s.gravity_file)])
        if not s.make_figures:
            cmd.append("--no-figures")
        if not s.resume:
            cmd.append("--no-resume")
        if s.verbose:
            cmd.append("--verbose")
        return cmd

    @QtCore.Slot()
    def _update_preview(self) -> None:
        state = self.get_state()
        ok, errors = self.validate_state(state)
        self.btn_run.setEnabled(ok and not self._is_running())
        self.notice_validation.setProperty("kind", "ok" if ok else "error")
        self.notice_validation.label.setText("Configuration looks ready." if ok else "\n".join(errors))
        self.notice_validation.style().unpolish(self.notice_validation)
        self.notice_validation.style().polish(self.notice_validation)
        try:
            command = self.build_command()
            # Lead with the informative part (module + flags): the interpreter's
            # absolute path used to wrap over several lines and push the actual
            # command out of view. The exact interpreter is in the tooltip and
            # is still what the QProcess launches.
            preview = subprocess.list2cmdline(["python", *command[1:]])
            self.txt_command.setPlainText(preview)
            self.txt_command.setToolTip(f"Interpreter used at launch: {command[0]}")
        except Exception as exc:
            self.txt_command.setPlainText(f"# PREVIEW ERROR\n{exc}")

    @QtCore.Slot()
    def _run_search(self) -> None:
        state = self.get_state()
        ok, errors = self.validate_state(state)
        if not ok:
            self.txt_log.appendPlainText("[FROZEN] Cannot start: " + "; ".join(errors))
            self._update_preview()
            return
        out_dir = Path(normalize_path(state.out_dir))
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = self.build_command()
        proc = QtCore.QProcess(self)
        proc.setProgram(cmd[0])
        proc.setArguments(cmd[1:])
        proc.setWorkingDirectory(str(self.project_root))
        proc.setProcessChannelMode(QtCore.QProcess.MergedChannels)
        proc.readyReadStandardOutput.connect(self._drain_process_output)
        proc.finished.connect(self._on_process_finished)
        proc.errorOccurred.connect(self._on_process_error)
        self._process = proc
        self.txt_log.clear()
        self.txt_log.appendPlainText("[FROZEN] Starting frozen-orbit search...")
        self.txt_log.appendPlainText(subprocess.list2cmdline(cmd))
        self._set_running(True)
        proc.start()

    @QtCore.Slot()
    def cancel_run(self) -> None:
        proc = self._process
        if proc is None or proc.state() == QtCore.QProcess.NotRunning:
            return
        self.txt_log.appendPlainText("[FROZEN] Cancellation requested.")
        proc.terminate()
        QtCore.QTimer.singleShot(3000, self._kill_if_running)

    def shutdown(self) -> None:
        self.cancel_run()

    def _kill_if_running(self) -> None:
        proc = self._process
        if proc is not None and proc.state() != QtCore.QProcess.NotRunning:
            proc.kill()

    def _drain_process_output(self) -> None:
        proc = self._process
        if proc is None:
            return
        text = bytes(proc.readAllStandardOutput()).decode(errors="replace")
        if text:
            self.txt_log.appendPlainText(text.rstrip())

    def _on_process_finished(self, exit_code: int, _status: QtCore.QProcess.ExitStatus) -> None:
        self._drain_process_output()
        if exit_code == 0:
            self.txt_log.appendPlainText("[FROZEN] Search completed.")
        else:
            self.txt_log.appendPlainText(f"[FROZEN] Search failed or was cancelled (exit={exit_code}).")
        self._set_running(False)
        self._process = None

    def _on_process_error(self, error: QtCore.QProcess.ProcessError) -> None:
        error_name = getattr(error, "name", str(error))
        self.txt_log.appendPlainText(f"[FROZEN] Process error: {error_name}")
        self._set_running(False)
        if self._process is not None and self._process.state() == QtCore.QProcess.NotRunning:
            self._process = None

    def _set_running(self, running: bool) -> None:
        self.progress.setVisible(running)
        self.btn_cancel.setEnabled(running)
        self.btn_run.setEnabled((not running) and self.validate_state()[0])
        for cls in (
            QtWidgets.QLineEdit,
            QtWidgets.QComboBox,
            QtWidgets.QAbstractSpinBox,
            QtWidgets.QCheckBox,
        ):
            for widget in self.findChildren(cls):
                widget.setEnabled(not running)

    def _is_running(self) -> bool:
        return self._process is not None and self._process.state() != QtCore.QProcess.NotRunning

    def _browse_out_dir(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Frozen Search Output", self.ent_out_dir.text())
        if path:
            self.ent_out_dir.setText(normalize_path(path))

    def _browse_gravity_file(self) -> None:
        path, _filter = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Gravity Model File",
            str(self.project_root / "data"),
            "Gravity files (*.sha *.tab *.txt *.gfc);;All files (*)",
        )
        if path:
            self.ent_gravity_file.setText(normalize_path(path))

    def _open_output_dir(self) -> None:
        path = Path(normalize_path(self.ent_out_dir.text().strip() or str(self.project_root)))
        path.mkdir(parents=True, exist_ok=True)
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(path)))

    @staticmethod
    def _spin_int(minimum: int, maximum: int, value: int) -> QtWidgets.QSpinBox:
        spin = NoWheelSpinBox()
        spin.setRange(minimum, maximum)
        spin.setValue(value)
        spin.setAccelerated(True)
        return spin

    @staticmethod
    def _spin_float(
        minimum: float,
        maximum: float,
        value: float,
        *,
        decimals: int = 3,
        suffix: str = "",
    ) -> QtWidgets.QDoubleSpinBox:
        spin = NoWheelDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setDecimals(decimals)
        spin.setValue(value)
        spin.setAccelerated(True)
        if suffix:
            spin.setSuffix(suffix)
        return spin


__all__ = ["FrozenSearchPage", "FrozenSearchPageState"]
