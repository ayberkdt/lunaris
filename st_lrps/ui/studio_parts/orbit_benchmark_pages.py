# -*- coding: utf-8 -*-
"""
st_lrps.ui.studio_parts.orbit_benchmark_pages

Studio page for the orbit-level lunar gravity benchmark. It drives the relocated
harness ``st_lrps.evaluation.compare_gravity_models`` as a subprocess, exposing
the parameters most useful for orbit-level validation:

* run mode — per-model DOP853 (RK8) vs a high-degree truth, OR GPU batch
  fixed-step RK4 vs a DOP853 truth;
* which models to run (SH20..SH160, ST-LRPS) and which truth model;
* the RK4 fixed step (GPU mode) and DOP853 tolerances (RK8 mode);
* scenario count/seed/mode/sampling, altitude band, duration, output cadence.

The page only builds and launches a command; the harness owns all physics.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import List, Optional

from .qt_common import *
from .qt_common import NoScrollComboBox

from st_lrps.evaluation import progress as _progress

from .common_widgets import (
    CollapsibleSection,
    ImageGallery,
    ProcessPane,
    ValidatedPathEdit,
    _format_command,
    _mono_font,
    _norm_path,
    _row_lineedit_with_button,
    _scroll_wrap,
    _settings,
    _split_cli_args,
    _tune_form,
    _tune_inputs,
)

SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = SCRIPT_DIR.parents[2]

BENCHMARK_CLI_MODULE = "st_lrps.evaluation.compare_gravity_models"
BENCHMARK_CLI_PATH = _REPO_ROOT / "st_lrps" / "evaluation" / "compare_gravity_models.py"
BENCHMARK_OUTPUT_ROOT = _REPO_ROOT / "outputs" / "gravity_benchmark"

# Comparison models offered as checkboxes (truth is selected separately).
_COMPARISON_MODELS = ("sh20", "sh60", "sh80", "sh120", "sh160", "st_lrps")
_DEFAULT_CHECKED = {"sh20", "sh80", "sh160", "st_lrps"}
_TRUTH_CHOICES = ("sh120", "sh160", "sh200")
_TRUTH_INTEGRATORS = ("DOP853", "RK45")
_GPU_INTEGRATORS = ("light", "medium", "robust")

_MODEL_NAME_RE = re.compile(r"^sh\d{1,4}$")


def _valid_model_name(name: str) -> bool:
    """A model is either 'st_lrps' or a spherical-harmonic degree like 'sh80'."""
    name = str(name).strip().lower()
    if name == "st_lrps":
        return True
    if not _MODEL_NAME_RE.match(name):
        return False
    return 1 <= int(name[2:]) <= 1800


class OrbitBenchmarkTab(QWidget):
    """Configure and launch the orbit-level gravity benchmark harness."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        # -- Run mode ------------------------------------------------------
        grp_mode = QGroupBox("Run Mode")
        form_mode = QFormLayout()
        _tune_form(form_mode)
        self.run_mode = NoScrollComboBox()
        self.run_mode.addItem("CPU adaptive sweep vs truth", "dop853")
        self.run_mode.addItem("GPU batch RK4 vs CPU truth", "gpu_rk4")
        self.run_mode.setCurrentIndex(0)
        self.run_mode.setToolTip(
            "DOP853 (RK8): each model is propagated with the adaptive 8th-order "
            "integrator and compared to the high-degree truth.\n"
            "GPU batch RK4: all scenarios are propagated together with a fixed-step "
            "RK4 kernel on the GPU and compared to a DOP853 truth."
        )
        self.truth = NoScrollComboBox()
        for t in _TRUTH_CHOICES:
            self.truth.addItem(t.upper(), t)
        self.truth.setCurrentIndex(_TRUTH_CHOICES.index("sh200"))
        self.truth.setToolTip("High-degree spherical-harmonic ground-truth model.")
        self.accumulate = QCheckBox("Resume / extend benchmark")
        self.accumulate.setChecked(False)
        self.accumulate.setToolTip(
            "Reuse the SAME output dir and SAME scenario settings. Existing scenario "
            "manifests and trajectory cache are checked before completed work is reused."
        )
        # Ground-truth integrator (applies in both modes).
        self.truth_integrator = NoScrollComboBox()
        for ti in _TRUTH_INTEGRATORS:
            self.truth_integrator.addItem(ti + (" (RK8)" if ti == "DOP853" else ""), ti)
        self.truth_integrator.setCurrentIndex(0)
        self.truth_integrator.setToolTip(
            "Adaptive integrator used to build the ground-truth reference trajectories."
        )
        form_mode.addRow("Mode", self.run_mode)
        form_mode.addRow("Truth model", self.truth)
        form_mode.addRow("Truth integrator", self.truth_integrator)
        form_mode.addRow(self.accumulate)
        grp_mode.setLayout(form_mode)

        # -- Models --------------------------------------------------------
        grp_models = QGroupBox("Models to Run")
        models_lo = QVBoxLayout()
        self._model_checks: dict[str, QCheckBox] = {}
        self._custom_models: List[str] = []
        self._models_grid = QGridLayout()
        self._models_grid.setContentsMargins(0, 0, 0, 0)
        self._model_grid_count = 0
        for name in _COMPARISON_MODELS:
            self._add_model_checkbox(name, checked=(name in _DEFAULT_CHECKED))
        models_grid_w = QWidget()
        models_grid_w.setLayout(self._models_grid)
        models_lo.addWidget(models_grid_w)

        # Create/add a custom comparison model (another SH degree, e.g. sh45 / sh250).
        add_row = QHBoxLayout()
        add_row.setContentsMargins(0, 0, 0, 0)
        self.new_model_edit = QLineEdit()
        self.new_model_edit.setPlaceholderText("Add model, e.g. sh45")
        self.new_model_edit.setToolTip(
            "Create a new comparison model and add it to the list above. "
            "Use a spherical-harmonic degree like sh45 (sh1..sh1800)."
        )
        self.new_model_edit.returnPressed.connect(self._on_add_model)
        btn_add_model = QPushButton("Add model")
        btn_add_model.clicked.connect(self._on_add_model)
        add_row.addWidget(self.new_model_edit, 1)
        add_row.addWidget(btn_add_model)
        add_row_w = QWidget()
        add_row_w.setLayout(add_row)
        models_lo.addWidget(add_row_w)

        models_hint = QLabel(
            "Selected models are compared against the truth model above. Add custom "
            "spherical-harmonic degrees as shNN. ST-LRPS requires a trained run "
            "directory (auto-detected if left empty)."
        )
        models_hint.setWordWrap(True)
        models_hint.setStyleSheet("color: #94a3b8; font-size: 11px;")
        models_lo.addWidget(models_hint)
        grp_models.setLayout(models_lo)

        # -- Scenarios -----------------------------------------------------
        grp_scn = QGroupBox("Scenarios")
        form_scn = QFormLayout()
        _tune_form(form_scn)
        self.random_scenarios = QSpinBox()
        self.random_scenarios.setRange(1, 1_000_000)
        self.random_scenarios.setValue(100)
        self.scenario_seed = QSpinBox()
        self.scenario_seed.setRange(0, 2_147_483_647)
        self.scenario_seed.setValue(42)
        self.scenario_mode = NoScrollComboBox()
        self.scenario_mode.addItem("near_circular_altitude", "near_circular_altitude")
        self.scenario_mode.addItem("bounded_keplerian", "bounded_keplerian")
        self.sampling_method = NoScrollComboBox()
        self.sampling_method.addItem("Random / legacy", "random")
        self.sampling_method.addItem("Latin Hypercube", "lhs")
        self.sampling_method.addItem("Sobol deterministic", "sobol")
        self.sampling_method.addItem("Sobol scrambled", "sobol_scrambled")
        self.sampling_method.setCurrentIndex(0)
        self.sampling_method.setToolTip(
            "Opt-in deterministic scenario coverage. Random preserves the legacy generator."
        )
        self.inclination_sampling = NoScrollComboBox()
        self.inclination_sampling.addItem("uniform_deg", "uniform_deg")
        self.inclination_sampling.addItem("uniform_cos", "uniform_cos")
        self.inclination_sampling.setCurrentIndex(0)
        self.inclination_sampling.setToolTip(
            "uniform_deg preserves the legacy inclination distribution."
        )
        self.alt_min = QDoubleSpinBox()
        self.alt_min.setDecimals(1)
        self.alt_min.setRange(1.0, 100_000.0)
        self.alt_min.setValue(200.0)
        self.alt_max = QDoubleSpinBox()
        self.alt_max.setDecimals(1)
        self.alt_max.setRange(1.0, 100_000.0)
        self.alt_max.setValue(400.0)
        self.duration_days = QDoubleSpinBox()
        self.duration_days.setDecimals(4)
        self.duration_days.setRange(0.0001, 3650.0)
        self.duration_days.setValue(1.0)
        self.dt_out = QDoubleSpinBox()
        self.dt_out.setDecimals(2)
        self.dt_out.setRange(0.01, 86400.0)
        self.dt_out.setValue(60.0)
        form_scn.addRow("Scenario count", self.random_scenarios)
        form_scn.addRow("Seed", self.scenario_seed)
        form_scn.addRow("Orbit mode", self.scenario_mode)
        form_scn.addRow("Sampling", self.sampling_method)
        form_scn.addRow("Inclination draw", self.inclination_sampling)
        form_scn.addRow("Altitude min (km)", self.alt_min)
        form_scn.addRow("Altitude max (km)", self.alt_max)
        form_scn.addRow("Duration (days)", self.duration_days)
        form_scn.addRow("Output dt (s)", self.dt_out)
        grp_scn.setLayout(form_scn)

        # -- Persistent cache / resume ------------------------------------
        grp_cache = QGroupBox("Caching / Resume")
        form_cache = QFormLayout()
        _tune_form(form_cache)
        self.cache_trajectories = QCheckBox("Cache all trajectories")
        self.cache_trajectories.setChecked(True)
        self.cache_trajectories.setToolTip(
            "Save each completed truth/model trajectory under benchmark_cache."
        )
        self.reuse_cache = QCheckBox("Reuse existing cache")
        self.reuse_cache.setChecked(True)
        self.reuse_cache.setToolTip(
            "Skip compatible cached truth/model trajectories and compute only missing files."
        )
        self.append_scenarios = QSpinBox()
        self.append_scenarios.setRange(0, 1_000_000)
        self.append_scenarios.setValue(0)
        self.append_scenarios.setToolTip(
            "Append this many new scenarios after the existing manifest. 0 uses the "
            "scenario count as the target total."
        )
        self.rebuild_metrics = QCheckBox("Rebuild metrics from cache")
        self.rebuild_metrics.setChecked(False)
        self.rebuild_metrics.setToolTip(
            "Load cached trajectories and regenerate metrics/reports without propagating."
        )
        self.strict_complete = QCheckBox("Require complete model set")
        self.strict_complete.setChecked(False)
        self.strict_complete.setToolTip(
            "Fail if selected models are missing cached trajectories during metric rebuild."
        )
        self.cache_dir = ValidatedPathEdit(
            placeholder="Empty -> output_dir/benchmark_cache", check_file=False
        )
        btn_cache = QPushButton("Select...")
        btn_cache.clicked.connect(self._pick_cache_dir)
        cache_row = _row_lineedit_with_button(self.cache_dir, btn_cache)
        form_cache.addRow(self.cache_trajectories)
        form_cache.addRow(self.reuse_cache)
        form_cache.addRow(self.accumulate)
        form_cache.addRow("Append scenarios", self.append_scenarios)
        form_cache.addRow(self.rebuild_metrics)
        form_cache.addRow(self.strict_complete)
        form_cache.addRow("Cache dir", cache_row)
        grp_cache.setLayout(form_cache)

        # -- Mode-specific numerics ----------------------------------------
        grp_cpu = QGroupBox("CPU DOP853 Settings")
        form_cpu = QFormLayout()
        _tune_form(form_cpu)
        # Per-model adaptive integrator (CPU / DOP853 mode).
        self.integrator = NoScrollComboBox()
        self.integrator.addItem("DOP853 (RK8)", "DOP853")
        self.integrator.addItem("RK45", "RK45")
        self.integrator.setCurrentIndex(0)
        self.integrator.setToolTip("Adaptive integrator for the compared models (CPU mode).")
        # CPU parallelism (CPU / DOP853 mode).
        self.cpu_workers = QSpinBox()
        self.cpu_workers.setRange(1, 256)
        self.cpu_workers.setValue(1)
        self.cpu_workers.setToolTip(
            "CPU worker processes for the per-model adaptive sweep. 1 = sequential. "
            "Each worker builds its own ephemeris + gravity caches."
        )
        form_cpu.addRow("Compare integrator", self.integrator)
        form_cpu.addRow("CPU workers", self.cpu_workers)
        grp_cpu.setLayout(form_cpu)

        grp_gpu = QGroupBox("GPU RK4 Settings")
        form_gpu = QFormLayout()
        _tune_form(form_gpu)
        # GPU fixed-step method (GPU mode).
        self.gpu_integrator = NoScrollComboBox()
        self.gpu_integrator.addItem("light (RK2 midpoint)", "light")
        self.gpu_integrator.addItem("medium (classic RK4)", "medium")
        self.gpu_integrator.addItem("robust (RK4 + Richardson)", "robust")
        self.gpu_integrator.setCurrentIndex(1)
        self.gpu_integrator.setToolTip(
            "GPU fixed-step fidelity: light=RK2 (cheap), medium=RK4 (standard), "
            "robust=RK4 with Richardson extrapolation (most accurate)."
        )
        self.rk4_dt = QDoubleSpinBox()
        self.rk4_dt.setDecimals(3)
        self.rk4_dt.setRange(0.001, 600.0)
        self.rk4_dt.setValue(10.0)
        self.rk4_dt.setToolTip("Fixed step size (seconds) for the GPU integrator.")
        self.torch_dtype = NoScrollComboBox()
        self.torch_dtype.addItems(["float64", "float32"])
        self.gpu_fallback = NoScrollComboBox()
        self.gpu_fallback.addItem("error (require CUDA)", "error")
        self.gpu_fallback.addItem("cpu (fallback)", "cpu")
        self.truth_workers = QSpinBox()
        self.truth_workers.setRange(1, 256)
        self.truth_workers.setValue(1)
        self.truth_workers.setToolTip(
            "CPU worker processes for DOP853 truth generation before GPU RK4 comparison."
        )
        form_gpu.addRow("RK method", self.gpu_integrator)
        form_gpu.addRow("Fixed step (s)", self.rk4_dt)
        form_gpu.addRow("Truth workers", self.truth_workers)
        form_gpu.addRow("Torch dtype", self.torch_dtype)
        form_gpu.addRow("Fallback", self.gpu_fallback)
        grp_gpu.setLayout(form_gpu)

        mode_settings_w = QWidget()
        mode_settings_l = QVBoxLayout()
        mode_settings_l.setContentsMargins(0, 0, 0, 0)
        mode_settings_l.setSpacing(8)
        mode_settings_l.addWidget(grp_cpu)
        mode_settings_l.addWidget(grp_gpu)
        mode_settings_w.setLayout(mode_settings_l)

        # -- DOP853 tolerances (advanced) ----------------------------------
        form_tol = QFormLayout()
        _tune_form(form_tol)
        self.rtol = QLineEdit("1e-10")
        self.atol = QLineEdit("1e-12")
        self.max_step = QDoubleSpinBox()
        self.max_step.setDecimals(2)
        self.max_step.setRange(0.0, 100_000.0)
        self.max_step.setValue(30.0)
        self.max_step.setToolTip("Maximum DOP853 step (s); 0 disables the user cap.")
        form_tol.addRow("rtol", self.rtol)
        form_tol.addRow("atol", self.atol)
        form_tol.addRow("max step (s)", self.max_step)
        tol_inner = QWidget()
        tol_inner.setLayout(form_tol)
        self._tol_section = CollapsibleSection("DOP853 Tolerances (advanced)")
        tol_wrap = QVBoxLayout()
        tol_wrap.setContentsMargins(0, 0, 0, 0)
        tol_wrap.addWidget(tol_inner)
        self._tol_section.set_content_layout(tol_wrap)

        # -- Paths ---------------------------------------------------------
        grp_paths = QGroupBox("ST-LRPS & Output")
        form_paths = QFormLayout()
        _tune_form(form_paths)
        self.st_lrps_dir = ValidatedPathEdit(
            placeholder="Empty -> auto-detect newest ST-LRPS run", check_file=False
        )
        btn_stl = QPushButton("Select...")
        btn_stl.clicked.connect(self._pick_st_lrps_dir)
        stl_row = _row_lineedit_with_button(self.st_lrps_dir, btn_stl)
        self.out_dir = ValidatedPathEdit(
            placeholder=f"Empty -> {BENCHMARK_OUTPUT_ROOT}", check_file=False
        )
        btn_out = QPushButton("Select...")
        btn_out.clicked.connect(self._pick_out_dir)
        out_row = _row_lineedit_with_button(self.out_dir, btn_out)
        form_paths.addRow("ST-LRPS model dir", stl_row)
        form_paths.addRow("Output dir", out_row)
        grp_paths.setLayout(form_paths)

        # -- Extra args + command preview ----------------------------------
        self.extra_args = QLineEdit("")
        self.extra_args.setPlaceholderText("Extra CLI arguments (optional)")
        self.command_preview = QPlainTextEdit()
        self.command_preview.setReadOnly(True)
        self.command_preview.setFont(_mono_font())
        self.command_preview.setMinimumHeight(60)
        self.command_preview.setMaximumHeight(96)
        self.command_warning = QLabel("")
        self.command_warning.setWordWrap(True)
        self.command_warning.setStyleSheet("color: #fbbf24; font-size: 11px;")
        btn_preview = QPushButton("Preview Command")
        btn_preview.clicked.connect(self._refresh_command_preview)
        btn_copy = QPushButton("Copy Command")
        btn_copy.clicked.connect(self._copy_command_preview)
        preview_btns = QHBoxLayout()
        preview_btns.setContentsMargins(0, 0, 0, 0)
        preview_btns.addWidget(btn_preview)
        preview_btns.addWidget(btn_copy)
        preview_btns.addStretch(1)
        preview_btns_w = QWidget()
        preview_btns_w.setLayout(preview_btns)

        form_extra = QFormLayout()
        _tune_form(form_extra)
        form_extra.addRow("Extra CLI args", self.extra_args)
        form_extra.addRow("", preview_btns_w)
        form_extra.addRow("Generated Command", self.command_preview)
        form_extra.addRow("", self.command_warning)
        extra_w = QWidget()
        extra_w.setLayout(form_extra)

        # -- Layout assembly ----------------------------------------------
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(12)
        grid.addWidget(grp_mode, 0, 0)
        grid.addWidget(grp_models, 0, 1)
        grid.addWidget(grp_scn, 1, 0)
        grid.addWidget(mode_settings_w, 1, 1)
        grid.addWidget(grp_cache, 2, 0, 1, 2)
        grid.addWidget(self._tol_section, 3, 0, 1, 2)
        grid.addWidget(grp_paths, 4, 0, 1, 2)
        grid.addWidget(extra_w, 5, 0, 1, 2)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        for g in (grp_mode, grp_models, grp_scn, grp_cache, grp_cpu, grp_gpu, grp_paths):
            _tune_inputs(g)

        self.runner = ProcessPane()
        self.runner.btn_start.setText("Run Benchmark")
        self.runner.btn_start.clicked.connect(self._start)
        self.runner.set_finished_hook(self._on_finished)
        self.runner.set_progress_parser(self._parse_progress)
        self._status_strip = self._build_status_strip()
        self._gallery = ImageGallery()
        self._effective_out_dir = ""

        top = QWidget()
        top_l = QVBoxLayout()
        top_l.setContentsMargins(8, 8, 8, 8)
        top_l.addLayout(grid)
        top.setLayout(top_l)

        bottom = QWidget()
        bl = QVBoxLayout()
        bl.setContentsMargins(0, 0, 0, 0)
        bl.setSpacing(8)
        bl.addWidget(self._status_strip)
        bl.addWidget(self.runner, 1)
        bl.addWidget(self._gallery, 1)
        bottom.setLayout(bl)

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(_scroll_wrap(top))
        splitter.addWidget(bottom)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([440, 520])

        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        layout.addWidget(splitter, 1)
        self.setLayout(layout)

        # Wiring
        self.run_mode.currentIndexChanged.connect(self._on_mode_changed)
        for w in (
            self.truth, self.truth_integrator, self.scenario_mode, self.integrator,
            self.sampling_method, self.inclination_sampling,
            self.gpu_integrator, self.torch_dtype, self.gpu_fallback,
        ):
            w.currentIndexChanged.connect(self._refresh_command_preview)
        for w in (
            self.random_scenarios, self.scenario_seed, self.alt_min, self.alt_max,
            self.duration_days, self.dt_out, self.rk4_dt, self.max_step,
            self.cpu_workers, self.truth_workers, self.append_scenarios,
        ):
            w.valueChanged.connect(self._refresh_command_preview)
        self.accumulate.toggled.connect(self._refresh_command_preview)
        self.cache_trajectories.toggled.connect(self._refresh_command_preview)
        self.reuse_cache.toggled.connect(self._refresh_command_preview)
        self.rebuild_metrics.toggled.connect(self._refresh_command_preview)
        self.strict_complete.toggled.connect(self._refresh_command_preview)
        self.rtol.textChanged.connect(self._refresh_command_preview)
        self.atol.textChanged.connect(self._refresh_command_preview)
        self.st_lrps_dir.textChanged.connect(self._refresh_command_preview)
        self.out_dir.textChanged.connect(self._refresh_command_preview)
        self.cache_dir.textChanged.connect(self._refresh_command_preview)
        self.extra_args.textChanged.connect(self._refresh_command_preview)

        self._grp_cpu_settings = grp_cpu
        self._grp_gpu_settings = grp_gpu
        self._restore_settings()
        self._on_mode_changed()

    # ------------------------------------------------------------------
    # Mode dependence
    # ------------------------------------------------------------------
    def _on_mode_changed(self, *_a) -> None:
        mode = self.run_mode.currentData() or "dop853"
        is_gpu = mode == "gpu_rk4"
        # Show only the numerics panel that belongs to the selected run mode.
        self._grp_cpu_settings.setVisible(not is_gpu)
        self._grp_gpu_settings.setVisible(is_gpu)
        self._tol_section.setVisible(not is_gpu)
        # Truth integrator applies in both modes — always enabled.
        self.truth_integrator.setEnabled(True)
        self._refresh_command_preview()

    # ------------------------------------------------------------------
    # Progress status strip
    # ------------------------------------------------------------------
    def _build_status_strip(self) -> QWidget:
        """Compact one-line strip showing phase / model / progress / ETA."""
        strip = QFrame()
        strip.setObjectName("benchStatusStrip")
        strip.setStyleSheet(
            "#benchStatusStrip { background: rgba(16,24,48,0.55); "
            "border: 1px solid rgba(185,194,221,0.14); border-radius: 8px; }"
        )
        lo = QHBoxLayout()
        lo.setContentsMargins(12, 6, 12, 6)
        lo.setSpacing(18)

        def _metric(caption: str) -> QLabel:
            cell = QVBoxLayout()
            cell.setContentsMargins(0, 0, 0, 0)
            cell.setSpacing(0)
            cap = QLabel(caption)
            cap.setStyleSheet("color:#6f7ca8; font-size:10px; font-weight:600;")
            val = QLabel("-")
            val.setStyleSheet("color:#d8e1f7; font-size:13px; font-weight:600;")
            cell.addWidget(cap)
            cell.addWidget(val)
            holder = QWidget()
            holder.setLayout(cell)
            lo.addWidget(holder)
            return val

        self._st_phase = _metric("Phase")
        self._st_model = _metric("Model")
        self._st_phase_pct = _metric("Phase %")
        self._st_overall_pct = _metric("Overall %")
        self._st_elapsed = _metric("Elapsed")
        self._st_eta = _metric("ETA")
        self._st_steps = _metric("steps/s")
        lo.addStretch(1)
        strip.setLayout(lo)
        return strip

    def _reset_status_strip(self) -> None:
        for lbl in (
            self._st_phase, self._st_model, self._st_phase_pct,
            self._st_overall_pct, self._st_elapsed, self._st_eta, self._st_steps,
        ):
            lbl.setText("-")
        self._st_phase.setText("starting")

    def _parse_progress(self, line: str) -> None:
        """Parse a harness ``[progress]`` / ``[progress_total]`` line.

        Never raises into the ProcessPane: unparseable or non-progress lines are
        ignored so plain logs keep flowing.
        """
        try:
            info = _progress.parse_progress_line(line)
        except Exception:
            info = None
        if not info:
            return

        model = info.get("model")
        if model:
            self._st_model.setText(str(model))

        if info.get("kind") == "progress_total":
            pct = info.get("percent")
            if pct is not None:
                clamped = int(round(min(100.0, max(0.0, float(pct)))))
                self.runner.progress.setRange(0, 100)
                self.runner.progress.setValue(clamped)
                self._st_overall_pct.setText(f"{float(pct):.1f}%")
            elapsed = info.get("elapsed_s")
            if elapsed is not None:
                self._st_elapsed.setText(_progress.format_duration(float(elapsed)))
            eta = info.get("eta_s")
            self._st_eta.setText(
                _progress.format_eta(float(eta)) if eta is not None else "-"
            )
            return

        # Per-phase progress line.
        phase = info.get("phase")
        if phase:
            self._st_phase.setText(str(phase))
        pct = info.get("percent")
        if pct is not None:
            self._st_phase_pct.setText(f"{float(pct):.1f}%")
        steps_per_s = info.get("steps_per_s")
        if steps_per_s is not None:
            self._st_steps.setText(f"{float(steps_per_s):.1f}")

    # ------------------------------------------------------------------
    # Model selection (with custom additions)
    # ------------------------------------------------------------------
    def _add_model_checkbox(self, name: str, checked: bool = True) -> bool:
        """Add a model checkbox to the grid. Returns False if name is empty/duplicate."""
        name = str(name).strip().lower()
        if not name or name in self._model_checks:
            return False
        label = "ST-LRPS" if name == "st_lrps" else name.upper()
        cb = QCheckBox(label)
        cb.setChecked(checked)
        cb.toggled.connect(self._refresh_command_preview)
        self._model_checks[name] = cb
        r, c = divmod(self._model_grid_count, 3)
        self._models_grid.addWidget(cb, r, c)
        self._model_grid_count += 1
        return True

    def _try_add_model(self, raw: str) -> tuple[bool, str]:
        """Validate and add a model. Returns (ok, error_message). No UI dialogs.

        Kept dialog-free so it is unit-testable headlessly.
        """
        raw = str(raw).strip().lower()
        if not raw:
            return False, ""
        if not _valid_model_name(raw):
            return False, (
                "Model must be 'st_lrps' or a spherical-harmonic degree like 'sh80' "
                "(sh1..sh1800)."
            )
        if raw in self._model_checks:
            self._model_checks[raw].setChecked(True)
            return True, ""
        if self._add_model_checkbox(raw, checked=True):
            if raw not in self._custom_models:
                self._custom_models.append(raw)
            return True, ""
        return False, "Could not add model."

    def _on_add_model(self) -> None:
        ok, err = self._try_add_model(self.new_model_edit.text())
        if not ok:
            if err:
                QMessageBox.warning(self, "Invalid model", err)
            return
        self.new_model_edit.clear()
        self._refresh_command_preview()

    # ------------------------------------------------------------------
    # File pickers
    # ------------------------------------------------------------------
    def _pick_st_lrps_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(
            self, "ST-LRPS model dir", self.st_lrps_dir.text() or str(SCRIPT_DIR)
        )
        if d:
            self.st_lrps_dir.setText(_norm_path(d))

    def _pick_out_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(
            self, "Output dir", self.out_dir.text() or str(BENCHMARK_OUTPUT_ROOT)
        )
        if d:
            self.out_dir.setText(_norm_path(d))

    def _pick_cache_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(
            self, "Benchmark cache dir", self.cache_dir.text() or str(BENCHMARK_OUTPUT_ROOT)
        )
        if d:
            self.cache_dir.setText(_norm_path(d))

    # ------------------------------------------------------------------
    # Command construction
    # ------------------------------------------------------------------
    def _selected_models(self) -> List[str]:
        return [name for name, cb in self._model_checks.items() if cb.isChecked()]

    def _build_args(self, show_errors: bool = True) -> Optional[List[str]]:
        def fail(title: str, message: str) -> Optional[List[str]]:
            if show_errors:
                QMessageBox.critical(self, title, message)
            else:
                self.command_warning.setText(message)
            return None

        if not show_errors:
            self.command_warning.setText("")

        if not BENCHMARK_CLI_PATH.exists():
            return fail("Missing script", "st_lrps/evaluation/compare_gravity_models.py not found.")

        models = self._selected_models()
        if not models:
            return fail("No models", "Select at least one model to run.")

        mode = self.run_mode.currentData() or "dop853"
        truth = self.truth.currentData() or "sh200"

        args = ["-u", "-m", BENCHMARK_CLI_MODULE]
        # Common scenario settings.
        args += ["--random-scenarios", str(self.random_scenarios.value())]
        args += ["--scenario-seed", str(self.scenario_seed.value())]
        args += ["--scenario-mode", self.scenario_mode.currentData() or "near_circular_altitude"]
        sampling_method = self.sampling_method.currentData() or "random"
        inclination_sampling = self.inclination_sampling.currentData() or "uniform_deg"
        if sampling_method != "random":
            args += ["--sampling-method", sampling_method]
        if inclination_sampling != "uniform_deg":
            args += ["--inclination-sampling", inclination_sampling]
        args += ["--altitude-min-km", str(self.alt_min.value())]
        args += ["--altitude-max-km", str(self.alt_max.value())]
        args += ["--duration-days", str(self.duration_days.value())]
        args += ["--dt-out", str(self.dt_out.value())]
        args += ["--truth", truth]
        args += ["--truth-integrator", self.truth_integrator.currentData() or "DOP853"]

        if mode == "gpu_rk4":
            args += ["--gpu-batch-compare"]
            args += ["--gpu-models", ",".join(models)]
            args += ["--gpu-integrator", self.gpu_integrator.currentData() or "medium"]
            args += ["--rk4-dt-s", str(self.rk4_dt.value())]
            args += ["--workers", str(self.truth_workers.value())]
            args += ["--torch-dtype", self.torch_dtype.currentText()]
            args += ["--gpu-fallback", self.gpu_fallback.currentData() or "error"]
        else:
            args += ["--models", ",".join(models)]
            args += ["--integrator", self.integrator.currentData() or "DOP853"]
            args += ["--workers", str(self.cpu_workers.value())]
            rtol = self.rtol.text().strip()
            atol = self.atol.text().strip()
            for label, value in (("rtol", rtol), ("atol", atol)):
                if value:
                    try:
                        float(value)
                    except ValueError:
                        return fail("Invalid tolerance", f"{label} must be a number, got {value!r}.")
            if rtol:
                args += ["--rtol", rtol]
            if atol:
                args += ["--atol", atol]
            args += ["--max-step", str(self.max_step.value())]

        if "st_lrps" in models:
            stl = self.st_lrps_dir.text().strip()
            if stl:
                if not Path(stl).exists():
                    return fail("Missing ST-LRPS dir", f"ST-LRPS model dir not found:\n{stl}")
                args += ["--st-lrps-model-dir", stl]

        out_dir = self.out_dir.text().strip() or str(BENCHMARK_OUTPUT_ROOT)
        args += ["--output-dir", out_dir]
        if self.accumulate.isChecked():
            args += ["--resume"]
        if self.cache_trajectories.isChecked():
            args += ["--cache-trajectories"]
        if self.reuse_cache.isChecked():
            args += ["--reuse-cache"]
        if self.append_scenarios.value() > 0:
            args += ["--append-scenarios", str(self.append_scenarios.value())]
        if self.rebuild_metrics.isChecked():
            args += ["--rebuild-metrics"]
        if self.strict_complete.isChecked():
            args += ["--strict-complete"]
        cache_dir = self.cache_dir.text().strip()
        if cache_dir:
            args += ["--cache-dir", cache_dir]

        extra = self.extra_args.text().strip()
        if extra:
            extra_args, err = _split_cli_args(extra)
            if err:
                return fail("Invalid extra CLI arguments", err)
            args += extra_args or []
        return args

    def _refresh_command_preview(self, *_a) -> None:
        args = self._build_args(show_errors=False)
        if not args:
            self.command_preview.clear()
            return
        self.command_preview.setPlainText(_format_command(sys.executable, args))

    def _copy_command_preview(self) -> None:
        if not self.command_preview.toPlainText().strip():
            self._refresh_command_preview()
        QGuiApplication.clipboard().setText(self.command_preview.toPlainText())

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    def _start(self) -> None:
        args = self._build_args(show_errors=True)
        if not args:
            return
        out_dir = self.out_dir.text().strip() or str(BENCHMARK_OUTPUT_ROOT)
        try:
            Path(out_dir).mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            QMessageBox.critical(self, "Output dir", f"Could not create output dir:\n{exc}")
            return
        self._effective_out_dir = out_dir
        self.runner.set_output_dir(out_dir)
        self.runner.progress.setRange(0, 0)
        self._reset_status_strip()
        self._gallery.clear_gallery()
        self._save_settings()
        self.runner.start(sys.executable, args, workdir=str(_REPO_ROOT))

    def _on_finished(self, exit_code, exit_status) -> None:
        out_dir = self._effective_out_dir
        if not out_dir or not Path(out_dir).is_dir():
            return
        imgs: List[Path] = []
        base = Path(out_dir)
        imgs += list(base.glob("*.png")) + list(base.glob("*.jpg"))
        for sub in base.glob("*/"):
            if sub.is_dir():
                imgs += list(sub.glob("*.png")) + list(sub.glob("*.jpg"))
        imgs = sorted(set(imgs))
        if imgs:
            cnt = self._gallery.load_images(imgs)
            if cnt:
                self.runner.append(f"\n[UI] {cnt} plot(s) loaded: {out_dir}")
        self.runner.set_output_dir(out_dir)
        self.runner.btn_open_folder.setVisible(True)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def _save_settings(self) -> None:
        s = _settings()
        s.beginGroup("orbit_benchmark")
        s.setValue("run_mode", self.run_mode.currentData())
        s.setValue("truth", self.truth.currentData())
        s.setValue("truth_integrator", self.truth_integrator.currentData())
        s.setValue("accumulate", self.accumulate.isChecked())
        s.setValue("resume_benchmark", self.accumulate.isChecked())
        s.setValue("cache_trajectories", self.cache_trajectories.isChecked())
        s.setValue("reuse_cache", self.reuse_cache.isChecked())
        s.setValue("append_scenarios", self.append_scenarios.value())
        s.setValue("rebuild_metrics", self.rebuild_metrics.isChecked())
        s.setValue("strict_complete", self.strict_complete.isChecked())
        s.setValue("cache_dir", self.cache_dir.text())
        s.setValue("models", ",".join(self._selected_models()))
        s.setValue("custom_models", ",".join(self._custom_models))
        s.setValue("random_scenarios", self.random_scenarios.value())
        s.setValue("scenario_seed", self.scenario_seed.value())
        s.setValue("scenario_mode", self.scenario_mode.currentData())
        s.setValue("sampling_method", self.sampling_method.currentData())
        s.setValue("inclination_sampling", self.inclination_sampling.currentData())
        s.setValue("alt_min", self.alt_min.value())
        s.setValue("alt_max", self.alt_max.value())
        s.setValue("duration_days", self.duration_days.value())
        s.setValue("dt_out", self.dt_out.value())
        s.setValue("integrator", self.integrator.currentData())
        s.setValue("cpu_workers", self.cpu_workers.value())
        s.setValue("truth_workers", self.truth_workers.value())
        s.setValue("gpu_integrator", self.gpu_integrator.currentData())
        s.setValue("rk4_dt", self.rk4_dt.value())
        s.setValue("torch_dtype", self.torch_dtype.currentText())
        s.setValue("gpu_fallback", self.gpu_fallback.currentData())
        s.setValue("rtol", self.rtol.text())
        s.setValue("atol", self.atol.text())
        s.setValue("max_step", self.max_step.value())
        s.setValue("st_lrps_dir", self.st_lrps_dir.text())
        s.setValue("out_dir", self.out_dir.text())
        s.endGroup()
        s.sync()

    def _restore_settings(self) -> None:
        s = _settings()
        s.beginGroup("orbit_benchmark")

        def _combo(combo, key):
            if s.contains(key):
                idx = combo.findData(str(s.value(key)))
                if idx >= 0:
                    combo.setCurrentIndex(idx)

        _combo(self.run_mode, "run_mode")
        _combo(self.truth, "truth")
        _combo(self.truth_integrator, "truth_integrator")
        resume_key = "resume_benchmark" if s.contains("resume_benchmark") else "accumulate"
        if s.contains(resume_key):
            self.accumulate.setChecked(str(s.value(resume_key, "false")).lower() == "true")
        for key, cb in (
            ("cache_trajectories", self.cache_trajectories),
            ("reuse_cache", self.reuse_cache),
            ("rebuild_metrics", self.rebuild_metrics),
            ("strict_complete", self.strict_complete),
        ):
            if s.contains(key):
                cb.setChecked(str(s.value(key, "false")).lower() == "true")
        # Recreate custom models before applying the saved checked set.
        if s.contains("custom_models"):
            for name in str(s.value("custom_models", "")).split(","):
                name = name.strip().lower()
                if name and _valid_model_name(name) and name not in self._model_checks:
                    if self._add_model_checkbox(name, checked=False):
                        self._custom_models.append(name)
        if s.contains("models"):
            wanted = {m for m in str(s.value("models", "")).split(",") if m}
            if wanted:
                for name, cb in self._model_checks.items():
                    cb.setChecked(name in wanted)
        for key, spin in (
            ("random_scenarios", self.random_scenarios),
            ("scenario_seed", self.scenario_seed),
            ("cpu_workers", self.cpu_workers),
            ("truth_workers", self.truth_workers),
            ("append_scenarios", self.append_scenarios),
        ):
            if s.contains(key):
                try:
                    spin.setValue(int(s.value(key)))
                except (TypeError, ValueError):
                    pass
        for key, spin in (
            ("alt_min", self.alt_min), ("alt_max", self.alt_max),
            ("duration_days", self.duration_days), ("dt_out", self.dt_out),
            ("rk4_dt", self.rk4_dt), ("max_step", self.max_step),
        ):
            if s.contains(key):
                try:
                    spin.setValue(float(s.value(key)))
                except (TypeError, ValueError):
                    pass
        _combo(self.scenario_mode, "scenario_mode")
        _combo(self.sampling_method, "sampling_method")
        _combo(self.inclination_sampling, "inclination_sampling")
        _combo(self.integrator, "integrator")
        _combo(self.gpu_integrator, "gpu_integrator")
        _combo(self.gpu_fallback, "gpu_fallback")
        if s.contains("torch_dtype"):
            self.torch_dtype.setCurrentText(str(s.value("torch_dtype", "float64")))
        if s.contains("rtol"):
            self.rtol.setText(str(s.value("rtol", "1e-10")))
        if s.contains("atol"):
            self.atol.setText(str(s.value("atol", "1e-12")))
        if s.contains("st_lrps_dir"):
            self.st_lrps_dir.setText(str(s.value("st_lrps_dir", "")))
        if s.contains("out_dir"):
            self.out_dir.setText(str(s.value("out_dir", "")))
        if s.contains("cache_dir"):
            self.cache_dir.setText(str(s.value("cache_dir", "")))
        s.endGroup()


class OrbitBenchmarkPage(QWidget):
    """Analysis workspace page: orbit-level gravity model benchmark."""

    def __init__(self, benchmark_tab: QWidget, parent: Optional[QWidget] = None):
        super().__init__(parent)
        lo = QVBoxLayout()
        lo.setContentsMargins(22, 20, 22, 20)
        lo.setSpacing(14)
        title = QLabel("Orbit-Level Benchmark")
        title.setStyleSheet("font-size: 18px; font-weight: 700; color: #e8ecf8;")
        subtitle = QLabel(
            "Propagate full orbits and compare gravity models (SH / ST-LRPS) "
            "against a high-degree truth — DOP853 (RK8) or GPU fixed-step RK4."
        )
        subtitle.setStyleSheet("color: #94a3b8; font-size: 12px;")
        lo.addWidget(title)
        lo.addWidget(subtitle)
        lo.addWidget(benchmark_tab, 1)
        self.setLayout(lo)


__all__ = ["OrbitBenchmarkTab", "OrbitBenchmarkPage", "BENCHMARK_CLI_MODULE"]
