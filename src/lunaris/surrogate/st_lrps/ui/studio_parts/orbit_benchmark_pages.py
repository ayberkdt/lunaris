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

import json
import re
import sys
from pathlib import Path

from lunaris.common.paths import project_root_from_file
from typing import List, Optional

from .qt_common import *
from .qt_common import NoScrollComboBox

from lunaris.surrogate.st_lrps.evaluation import progress as _progress

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
_REPO_ROOT = project_root_from_file(__file__)
_STLRPS_ROOT = SCRIPT_DIR.parents[1]

BENCHMARK_CLI_MODULE = "lunaris.surrogate.st_lrps.evaluation.compare_gravity_models"
BENCHMARK_CLI_PATH = _STLRPS_ROOT / "evaluation" / "compare_gravity_models.py"
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


# Pipeline chip status -> (glyph, label, text-color, fill, border).
_STATUS_STYLE = {
    "pending":   ("○", "Pending",  "#6f7ca8", "rgba(111,124,168,0.10)", "rgba(111,124,168,0.28)"),
    "queued":    ("○", "Queued",   "#9aa7c7", "rgba(154,167,199,0.10)", "rgba(154,167,199,0.30)"),
    "running":   ("●", "Running",  "#f59e0b", "rgba(245,158,11,0.16)",  "rgba(245,158,11,0.60)"),
    "completed": ("✓", "Done",     "#34d399", "rgba(52,211,153,0.16)",  "rgba(52,211,153,0.55)"),
    "cached":    ("✓", "Cached",   "#34d399", "rgba(52,211,153,0.10)",  "rgba(52,211,153,0.42)"),
    "failed":    ("✕", "Failed",   "#f87171", "rgba(248,113,113,0.18)", "rgba(248,113,113,0.60)"),
    "skipped":   ("–", "Skipped",  "#6f7ca8", "rgba(111,124,168,0.06)", "rgba(111,124,168,0.20)"),
}

# Run-status badge -> (text, text-color, fill).
_BADGE_STYLE = {
    "idle":      ("Idle",      "#9aa7c7", "rgba(154,167,199,0.12)"),
    "running":   ("Running",   "#f59e0b", "rgba(245,158,11,0.16)"),
    "completed": ("Completed", "#34d399", "rgba(52,211,153,0.16)"),
    "failed":    ("Failed",    "#f87171", "rgba(248,113,113,0.18)"),
}

# Phase keys (from [progress] phase=...) -> human label.
_PHASE_LABELS = {
    "scenario": "Scenario setup",
    "truth": "Truth (DOP853)",
    "gpu_model": "GPU model",
    "sweep": "CPU sweep",
    "aggregate": "Aggregation",
    "report": "Report",
}


def _pipeline_key(name: str) -> str:
    """Canonical pipeline-chip key for a model identity.

    Accepts either a UI base name (``sh20``, ``st_lrps``) or a runtime display
    name (``GPU_SH20_RK4``, ``GPU_SH20_RK4_DT10``); always lower-cased so chips
    match the exact identity the harness emits on stdout.
    """
    return str(name).strip().lower()


def _pipeline_label(key: str) -> str:
    """Human label for a pipeline node key (base or runtime display name).

    Examples: ``truth`` -> "Truth", ``sh20`` / ``GPU_SH20_RK4`` -> "SH20",
    ``GPU_ST_LRPS_RK4`` -> "ST-LRPS", ``GPU_SH20_RK4_DT10`` -> "SH20 Δt10s".
    """
    k = str(key).strip().lower()
    if k == "truth":
        return "Truth"
    if k == "report":
        return "Report"
    if k.startswith("gpu_"):
        k = k[4:]
    base, _, rest = k.partition("_rk4")
    if base == "st_lrps":
        label = "ST-LRPS"
    else:
        label = base.upper()
    mdt = re.search(r"dt([0-9pmn.]+)", rest)
    if mdt:
        dt = mdt.group(1).replace("p", ".").replace("m", "-").rstrip(".")
        label = f"{label} Δt{dt}s"
    return label


# Back-compat alias.
_chip_label = _pipeline_label


def _is_telemetry_line(line: str) -> bool:
    """True for per-step JSON telemetry lines (hidden in the log by default)."""
    s = str(line).lstrip()
    if not s.startswith("{"):
        return False
    return any(k in s for k in ('"t_s"', '"alt_km"', '"v_km_s"', '"ecc"'))


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
        self.alt_min.setValue(100.0)
        self.alt_max = QDoubleSpinBox()
        self.alt_max.setDecimals(1)
        self.alt_max.setRange(1.0, 100_000.0)
        self.alt_max.setValue(1000.0)
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
        self.cpu_workers.setValue(4)
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
        self.rk4_dt.setValue(30.0)
        self.rk4_dt.setToolTip("Fixed step size (seconds) for the GPU integrator.")
        self.rk4_dt_list = QLineEdit("")
        self.rk4_dt_list.setPlaceholderText("Optional, e.g. 10,30")
        self.rk4_dt_list.setToolTip(
            "Optional comma-separated RK4 step sizes. When set, each selected model "
            "is compared once per step size, e.g. SH20 dt10 vs SH20 dt30."
        )
        self.torch_dtype = NoScrollComboBox()
        self.torch_dtype.addItem("float32", "float32")
        self.torch_dtype.addItem("float64 (not recommended on laptops)", "float64")
        self.torch_dtype.setToolTip(
            "float32 is the default throughput setting. float64 is available for "
            "precision-sensitive checks, but is usually much slower on laptop GPUs."
        )
        self.gpu_fallback = NoScrollComboBox()
        self.gpu_fallback.addItem("error (require CUDA)", "error")
        self.gpu_fallback.addItem("cpu (fallback)", "cpu")
        self.gpu_frame_mode = NoScrollComboBox()
        self.gpu_frame_mode.addItem("dynamic (per-step)", "match_dynamics_engine")
        self.gpu_frame_mode.addItem("precomputed SLERP (faster)", "precomputed_slerp")
        self.gpu_frame_mode.setCurrentIndex(1)  # Default to precomputed SLERP
        self.gpu_frame_mode.setToolTip(
            "Frame rotation strategy for the GPU integrator.\n"
            "dynamic: interpolate the body-fixed quaternion on every RHS call.\n"
            "precomputed SLERP: precompute all fixed-step stage quaternions once "
            "before the loop (same interpolation, less per-step overhead). "
            "Numerically equivalent to dynamic within strict tolerance."
        )
        self.truth_workers = QSpinBox()
        self.truth_workers.setRange(1, 256)
        self.truth_workers.setValue(4)
        self.truth_workers.setToolTip(
            "CPU worker processes for DOP853 truth generation before GPU RK4 comparison."
        )
        form_gpu.addRow("RK method", self.gpu_integrator)
        form_gpu.addRow("Fixed step (s)", self.rk4_dt)
        form_gpu.addRow("Compare dt list", self.rk4_dt_list)
        form_gpu.addRow("Truth workers", self.truth_workers)
        form_gpu.addRow("Torch dtype", self.torch_dtype)
        form_gpu.addRow("Fallback", self.gpu_fallback)
        form_gpu.addRow("Frame mode", self.gpu_frame_mode)
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
        self.runner.btn_start.clicked.connect(self._start)
        self.runner.set_finished_hook(self._on_finished)
        self.runner.set_progress_parser(self._parse_progress)
        self._gallery = ImageGallery()
        self._effective_out_dir = ""

        # -- Run-monitor dashboard state ----------------------------------
        self._model_status: dict[str, str] = {}
        self._pipeline_order: List[str] = []
        self._chips: dict[str, dict] = {}
        self._current_model: Optional[str] = None
        self._pipeline_dynamic = False
        self._hidden_telemetry = 0

        # Dashboard cards. Built here because some reuse ProcessPane widgets
        # (run/stop buttons, the log view, the auto-scroll toggle).
        self._control_header = self._build_control_header()
        self._metrics_card = self._build_metrics_card()
        self._pipeline_card = self._build_pipeline_card()
        self._progress_card = self._build_progress_card()
        self._logs_section = self._build_logs_section()
        self._results_section = self._build_results_section()
        self.runner.set_display_filter(self._display_filter_line)

        # -- Single-page vertical layout ----------------------------------
        # Everything (configuration, run controls, logs and results/plots)
        # lives on ONE page. The log and results areas are in-place
        # collapsible sections — there is no secondary/bottom plot page.
        content = QWidget()
        col = QVBoxLayout()
        col.setContentsMargins(8, 8, 8, 8)
        col.setSpacing(12)
        col.addLayout(grid)
        col.addWidget(self._control_header)
        col.addWidget(self._metrics_card)
        col.addWidget(self._pipeline_card)
        col.addWidget(self._progress_card)
        col.addWidget(self._logs_section)
        col.addWidget(self._results_section)
        col.addStretch(1)
        content.setLayout(col)

        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        layout.addWidget(_scroll_wrap(content), 1)
        self.setLayout(layout)

        # Wiring
        self.run_mode.currentIndexChanged.connect(self._on_mode_changed)
        for w in (
            self.truth, self.truth_integrator, self.scenario_mode, self.integrator,
            self.sampling_method, self.inclination_sampling,
            self.gpu_integrator, self.torch_dtype, self.gpu_fallback, self.gpu_frame_mode,
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
        self.rk4_dt_list.textChanged.connect(self._refresh_command_preview)
        self.st_lrps_dir.textChanged.connect(self._refresh_command_preview)
        self.out_dir.textChanged.connect(self._refresh_command_preview)
        self.cache_dir.textChanged.connect(self._refresh_command_preview)
        self.extra_args.textChanged.connect(self._refresh_command_preview)

        self._grp_cpu_settings = grp_cpu
        self._grp_gpu_settings = grp_gpu
        self._restore_settings()
        self._on_mode_changed()
        self._reset_dashboard()

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
    # Run-monitor dashboard — builders
    # ------------------------------------------------------------------
    @staticmethod
    def _card(object_name: str) -> QFrame:
        frame = QFrame()
        frame.setObjectName(object_name)
        frame.setStyleSheet(
            f"#{object_name} {{ background: rgba(16,24,48,0.55); "
            "border: 1px solid rgba(185,194,221,0.14); border-radius: 10px; }"
        )
        return frame

    def _build_control_header(self) -> QWidget:
        """Run / Stop / Open-folder controls plus a run-status badge."""
        frame = self._card("benchHeader")
        lo = QHBoxLayout()
        lo.setContentsMargins(14, 10, 14, 10)
        lo.setSpacing(10)
        self.runner.btn_start.setText("Run Benchmark")
        lo.addWidget(self.runner.btn_start)
        lo.addWidget(self.runner.btn_stop)
        lo.addWidget(self.runner.btn_open_folder)
        lo.addStretch(1)
        self._status_badge = QLabel()
        lo.addWidget(self._status_badge)
        frame.setLayout(lo)
        return frame

    def _build_metrics_card(self) -> QWidget:
        """Top status dashboard: the key run metrics at a glance."""
        frame = self._card("benchMetrics")
        lo = QHBoxLayout()
        lo.setContentsMargins(14, 10, 14, 10)
        lo.setSpacing(20)

        def _metric(caption: str) -> QLabel:
            cell = QVBoxLayout()
            cell.setContentsMargins(0, 0, 0, 0)
            cell.setSpacing(1)
            cap = QLabel(caption)
            cap.setStyleSheet("color:#6f7ca8; font-size:10px; font-weight:600;")
            val = QLabel("-")
            val.setStyleSheet("color:#e8ecf8; font-size:14px; font-weight:700;")
            cell.addWidget(cap)
            cell.addWidget(val)
            holder = QWidget()
            holder.setLayout(cell)
            lo.addWidget(holder)
            return val

        self._st_overall_pct = _metric("Overall")
        self._st_phase = _metric("Phase")
        self._st_model = _metric("Current model")
        self._st_phase_pct = _metric("Phase %")
        self._st_eta = _metric("ETA")
        self._st_elapsed = _metric("Elapsed")
        self._st_steps = _metric("steps/s")
        self._st_scn = _metric("Scenarios")
        lo.addStretch(1)
        frame.setLayout(lo)
        return frame

    def _build_pipeline_card(self) -> QWidget:
        """Horizontal model pipeline / queue tracker."""
        frame = self._card("benchPipeline")
        outer = QVBoxLayout()
        outer.setContentsMargins(14, 8, 14, 10)
        outer.setSpacing(6)
        cap = QLabel("Model Pipeline")
        cap.setStyleSheet("color:#6f7ca8; font-size:10px; font-weight:600;")
        outer.addWidget(cap)

        self._pipeline_host = QWidget()
        self._pipeline_layout = QHBoxLayout()
        self._pipeline_layout.setContentsMargins(0, 0, 0, 0)
        self._pipeline_layout.setSpacing(8)
        self._pipeline_host.setLayout(self._pipeline_layout)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setFixedHeight(62)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        scroll.setWidget(self._pipeline_host)
        outer.addWidget(scroll)
        frame.setLayout(outer)
        return frame

    def _build_progress_card(self) -> QWidget:
        """Compact overall + current-phase progress bars and telemetry toggle."""
        frame = self._card("benchProgress")
        v = QVBoxLayout()
        v.setContentsMargins(14, 12, 14, 12)
        v.setSpacing(7)

        slim = (
            "QProgressBar { background: rgba(7,11,20,0.85); "
            "border: 1px solid rgba(185,194,221,0.18); border-radius: 5px; height: 10px; }"
            "QProgressBar::chunk { border-radius: 5px; background: %s; }"
        )

        cap1 = QLabel("Overall Progress")
        cap1.setStyleSheet("color:#9aa7c7; font-size:11px; font-weight:600;")
        self._overall_value = QLabel("-")
        self._overall_value.setStyleSheet("color:#34d399; font-size:11px; font-weight:700;")
        row1 = QHBoxLayout()
        row1.setContentsMargins(0, 0, 0, 0)
        row1.addWidget(cap1)
        row1.addStretch(1)
        row1.addWidget(self._overall_value)
        v.addLayout(row1)
        self.overall_bar = QProgressBar()
        self.overall_bar.setTextVisible(False)
        self.overall_bar.setFixedHeight(10)
        self.overall_bar.setStyleSheet(slim % "#34d399")
        v.addWidget(self.overall_bar)

        self._phase_caption = QLabel("Current Phase: -")
        self._phase_caption.setStyleSheet("color:#9aa7c7; font-size:11px; font-weight:600;")
        self._phase_value = QLabel("-")
        self._phase_value.setStyleSheet("color:#35d0ff; font-size:11px; font-weight:700;")
        row2 = QHBoxLayout()
        row2.setContentsMargins(0, 0, 0, 0)
        row2.addWidget(self._phase_caption)
        row2.addStretch(1)
        row2.addWidget(self._phase_value)
        v.addLayout(row2)
        self.phase_bar = QProgressBar()
        self.phase_bar.setTextVisible(False)
        self.phase_bar.setFixedHeight(10)
        self.phase_bar.setStyleSheet(slim % "#35d0ff")
        v.addWidget(self.phase_bar)

        self._phase_detail = QLabel("")
        self._phase_detail.setStyleSheet("color:#6f7ca8; font-size:10px;")
        v.addWidget(self._phase_detail)

        self.show_telemetry = QCheckBox("Show raw telemetry lines")
        self.show_telemetry.setChecked(False)
        self.show_telemetry.setToolTip(
            "Per-step JSON telemetry lines are hidden from the log by default. "
            "Enable to show them (applies to new lines)."
        )
        self.show_telemetry.toggled.connect(self._on_telemetry_toggled)
        self._telemetry_note = QLabel("")
        self._telemetry_note.setStyleSheet("color:#6f7ca8; font-size:10px;")
        trow = QHBoxLayout()
        trow.setContentsMargins(0, 0, 0, 0)
        trow.addWidget(self.show_telemetry)
        trow.addStretch(1)
        trow.addWidget(self._telemetry_note)
        v.addLayout(trow)
        frame.setLayout(v)
        return frame

    def _build_logs_section(self) -> CollapsibleSection:
        """Collapsible logs / diagnostics panel reusing the ProcessPane log."""
        sec = CollapsibleSection("Logs / Diagnostics")
        inner = QVBoxLayout()
        inner.setContentsMargins(0, 6, 0, 0)
        inner.setSpacing(6)

        bar = QHBoxLayout()
        bar.setContentsMargins(0, 0, 0, 0)
        bar.setSpacing(8)
        self.runner._auto_scroll.setText("Auto-scroll")
        bar.addWidget(self.runner._auto_scroll)
        bar.addStretch(1)
        btn_copy = QPushButton("Copy logs")
        btn_copy.setProperty("kind", "ghost")
        btn_copy.clicked.connect(self._copy_logs)
        btn_clear = QPushButton("Clear view")
        btn_clear.setProperty("kind", "ghost")
        btn_clear.setToolTip("Clear the log view only (does not delete output files).")
        btn_clear.clicked.connect(self.runner.log.clear)
        bar.addWidget(btn_copy)
        bar.addWidget(btn_clear)
        inner.addLayout(bar)

        self.runner.log.setMaximumHeight(240)
        inner.addWidget(self.runner.log)
        sec.set_content_layout(inner)
        sec.set_expanded(True)
        return sec

    def _build_results_section(self) -> CollapsibleSection:
        """In-place collapsible Results / Plots section (no secondary page).

        Holds the single, persistent ImageGallery — created once and only
        shown/hidden via the collapsible, never recreated. Starts collapsed so
        it takes minimal vertical space until a run produces plots.
        """
        sec = CollapsibleSection("Results / Plots")
        inner = QVBoxLayout()
        inner.setContentsMargins(0, 6, 0, 0)
        inner.setSpacing(6)

        bar = QHBoxLayout()
        bar.setContentsMargins(0, 0, 0, 0)
        bar.setSpacing(8)
        btn_refresh = QPushButton("Refresh results")
        btn_refresh.setProperty("kind", "ghost")
        btn_refresh.setToolTip("Re-scan the output directory for plots.")
        btn_refresh.clicked.connect(self._refresh_results)
        bar.addWidget(btn_refresh)
        bar.addStretch(1)
        inner.addLayout(bar)

        # Roomy when expanded so plots are not squeezed.
        self._gallery.setMinimumHeight(540)
        inner.addWidget(self._gallery, 1)
        sec.set_content_layout(inner)
        sec.set_expanded(False)
        return sec

    def _discover_result_images(self, out_dir: str) -> List[Path]:
        """Collect plot images from the output dir and its immediate subdirs."""
        base = Path(out_dir)
        if not base.is_dir():
            return []
        imgs: List[Path] = list(base.glob("*.png")) + list(base.glob("*.jpg"))
        for sub in base.glob("*/"):
            if sub.is_dir():
                imgs += list(sub.glob("*.png")) + list(sub.glob("*.jpg"))
        return sorted(set(imgs))

    def _refresh_results(self) -> None:
        """Re-scan the output directory and (re)load the gallery in place."""
        out_dir = self._effective_out_dir or self.out_dir.text().strip() or str(BENCHMARK_OUTPUT_ROOT)
        if not out_dir or not Path(out_dir).is_dir():
            return
        imgs = self._discover_result_images(out_dir)
        if imgs:
            self._gallery.load_images(imgs)
            self._results_section.set_expanded(True)
            self.runner.set_output_dir(out_dir)
            self.runner.btn_open_folder.setVisible(True)

    # ------------------------------------------------------------------
    # Run-monitor dashboard — state helpers
    # ------------------------------------------------------------------
    def _set_badge(self, state: str) -> None:
        text, fg, bg = _BADGE_STYLE.get(state, _BADGE_STYLE["idle"])
        self._status_badge.setText(text)
        self._status_badge.setStyleSheet(
            f"color:{fg}; background:{bg}; border-radius:9px; "
            "padding:3px 12px; font-size:11px; font-weight:700;"
        )

    def _reset_dashboard(self) -> None:
        for lbl in (
            self._st_overall_pct, self._st_phase, self._st_model, self._st_phase_pct,
            self._st_eta, self._st_elapsed, self._st_steps, self._st_scn,
        ):
            lbl.setText("-")
        self._overall_value.setText("-")
        self._phase_value.setText("-")
        self._phase_caption.setText("Current Phase: -")
        self._phase_detail.setText("")
        for bar in (self.overall_bar, self.phase_bar):
            bar.setRange(0, 0)  # indeterminate until first progress line
        self._hidden_telemetry = 0
        self._telemetry_note.setText("")
        self._current_model = None
        self._set_badge("idle")
        self._rebuild_pipeline(self._selected_models())

    # ------------------------------------------------------------------
    # Model pipeline chips
    # ------------------------------------------------------------------
    def _rebuild_pipeline(self, models: List[str]) -> None:
        """(Re)build the chip strip: Truth, a preview of the selected models, Report.

        This is the pre-run preview keyed by UI base names. Once the run starts
        emitting real model identities, the model chips are replaced live (see
        :meth:`_ensure_model_chip`) so they update one-by-one and include any
        step-size (Δt) variants in order.
        """
        while self._pipeline_layout.count():
            item = self._pipeline_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()
        self._chips = {}
        self._model_status = {}
        self._pipeline_dynamic = False
        order = ["truth"] + [_pipeline_key(m) for m in models] + ["report"]
        self._pipeline_order = order
        for key in order:
            self._pipeline_layout.addWidget(self._make_chip(key))
        self._pipeline_layout.addStretch(1)
        self._set_chip_status("truth", "pending")
        for m in order[1:-1]:
            self._set_chip_status(m, "queued")
        self._set_chip_status("report", "pending")

    def _make_chip(self, key: str) -> QWidget:
        frame = QFrame()
        v = QVBoxLayout()
        v.setContentsMargins(11, 5, 11, 5)
        v.setSpacing(1)
        name = QLabel(_pipeline_label(key))
        name.setStyleSheet(
            "color:#d8e1f7; font-size:12px; font-weight:700; "
            "background:transparent; border:none;"
        )
        status = QLabel("○ Pending")
        v.addWidget(name)
        v.addWidget(status)
        frame.setLayout(v)
        self._chips[_pipeline_key(key)] = {"frame": frame, "name": name, "status": status}
        return frame

    def _clear_model_chips(self) -> None:
        """Remove every chip except Truth and Report (keeps their widgets)."""
        for key in [k for k in self._chips if k not in ("truth", "report")]:
            chip = self._chips.pop(key)
            self._model_status.pop(key, None)
            self._pipeline_layout.removeWidget(chip["frame"])
            chip["frame"].setParent(None)
            chip["frame"].deleteLater()
        self._pipeline_order = [k for k in self._pipeline_order if k in ("truth", "report")]

    def _ensure_model_chip(self, key: str) -> str:
        """Make sure a chip exists for a runtime model identity; return its key.

        On the first runtime identity the base-name preview chips are dropped and
        the strip switches to live, identity-keyed chips inserted before Report.
        """
        key = _pipeline_key(key)
        if not self._pipeline_dynamic:
            self._clear_model_chips()
            self._pipeline_dynamic = True
        if key in self._chips:
            return key
        frame = self._make_chip(key)
        report_chip = self._chips.get("report")
        if report_chip is not None:
            idx = self._pipeline_layout.indexOf(report_chip["frame"])
            self._pipeline_layout.insertWidget(max(0, idx), frame)
            ins = (self._pipeline_order.index("report")
                   if "report" in self._pipeline_order else len(self._pipeline_order))
            self._pipeline_order.insert(ins, key)
        else:
            self._pipeline_layout.addWidget(frame)
            self._pipeline_order.append(key)
        self._set_chip_status(key, "queued")
        return key

    def _set_chip_status(self, key: str, status: str) -> None:
        key = str(key).lower()
        chip = self._chips.get(key)
        if chip is None:
            return
        self._model_status[key] = status
        glyph, label, fg, bg, border = _STATUS_STYLE.get(status, _STATUS_STYLE["queued"])
        chip["frame"].setStyleSheet(
            f"QFrame {{ background:{bg}; border:1px solid {border}; border-radius:8px; }}"
            "QLabel { background: transparent; border: none; }"
        )
        chip["status"].setText(f"{glyph} {label}")
        chip["status"].setStyleSheet(
            f"color:{fg}; font-size:11px; font-weight:600; "
            "background:transparent; border:none;"
        )

    # ------------------------------------------------------------------
    # Log filtering / tools
    # ------------------------------------------------------------------
    def _display_filter_line(self, text: str) -> bool:
        s = str(text).lstrip()
        # Machine-readable progress lines drive the dashboard but only clutter
        # the human log — hide them (the parser still receives every line).
        if s.startswith("[progress]") or s.startswith("[progress_total]"):
            return False
        if _is_telemetry_line(text):
            cb = getattr(self, "show_telemetry", None)
            if cb is not None and cb.isChecked():
                return True
            self._hidden_telemetry += 1
            self._telemetry_note.setText(
                f"Telemetry lines hidden: {self._hidden_telemetry}"
            )
            return False
        return True

    def _on_telemetry_toggled(self, checked: bool) -> None:
        if checked:
            self._telemetry_note.setText("Showing raw telemetry")
        elif self._hidden_telemetry:
            self._telemetry_note.setText(
                f"Telemetry lines hidden: {self._hidden_telemetry}"
            )
        else:
            self._telemetry_note.setText("")

    def _copy_logs(self) -> None:
        QGuiApplication.clipboard().setText(self.runner.log.toPlainText())

    # ------------------------------------------------------------------
    # Progress / status line parsing (UI-side only; never crashes the UI)
    # ------------------------------------------------------------------
    def _parse_progress(self, line: str) -> None:
        try:
            self._update_from_line(line)
        except Exception:
            pass  # a single log line must never break the monitor

    def _update_from_line(self, line: str) -> None:
        text = str(line).strip()
        if not text:
            return
        info = None
        try:
            info = _progress.parse_progress_line(text)
        except Exception:
            info = None
        if info:
            self._apply_structured(info)
        # Human-readable cache / gpu-batch lines are complementary signals.
        self._apply_human(text)

    def _apply_structured(self, info: dict) -> None:
        model = info.get("model")
        if model:
            self._st_model.setText(_pipeline_label(str(model)))

        if info.get("kind") == "progress_total":
            pct = info.get("percent")
            if pct is not None:
                self._set_overall(float(pct))
            elapsed = info.get("elapsed_s")
            if elapsed is not None:
                self._st_elapsed.setText(_progress.format_duration(float(elapsed)))
            eta = info.get("eta_s")
            self._st_eta.setText(
                _progress.format_eta(float(eta)) if eta is not None else "-"
            )
            return

        phase = info.get("phase")
        if phase:
            label = _PHASE_LABELS.get(str(phase), str(phase))
            self._st_phase.setText(label)
            cap = f"Current Phase: {label}"
            if model:
                cap += f" — {_pipeline_label(str(model))}"
            self._phase_caption.setText(cap)
            self._on_phase(str(phase), _pipeline_key(model) if model else None)

        pct = info.get("percent")
        if pct is not None:
            self._set_phase(float(pct))
            self._st_phase_pct.setText(f"{float(pct):.1f}%")
        sps = info.get("steps_per_s")
        if sps is not None:
            self._st_steps.setText(f"{float(sps):.1f}")
        n_scn = info.get("n_scenarios")
        if n_scn is not None:
            self._st_scn.setText(str(int(n_scn)))
        cs, ts = info.get("current_step"), info.get("total_steps")
        if cs is not None and ts is not None:
            self._phase_detail.setText(f"{int(cs)} / {int(ts)} steps")

    def _apply_human(self, text: str) -> None:
        low = text.lower()

        m = re.match(r"\[cache\]\s+truth(?:\s+cache)?\s+\S+:\s*(\d+)\s*/\s*(\d+)\s+complete",
                     low)
        if m:
            done, total = int(m.group(1)), int(m.group(2))
            if total > 0 and done >= total:
                self._set_chip_status("truth", "cached")
            return

        m = re.match(r"\[cache\]\s+model\s+(\S+):\s*(\d+)\s*/\s*(\d+)\s+complete", low)
        if m:
            key = self._ensure_model_chip(m.group(1))
            done, total = int(m.group(2)), int(m.group(3))
            if total > 0 and done >= total and "recomput" not in low:
                self._set_chip_status(key, "cached")
            return

        if re.match(r"\[truth\]\s+building", low):
            self._mark_truth_running()
            return
        if re.match(r"\[truth\]\s+reused cache", low):
            self._set_chip_status("truth", "cached")
            return

        # Model start: "[gpu-batch] Model 01/4 | GPU_SH20_RK4 starting for ..."
        m = re.search(r"\[gpu-batch\]\s+model\s+\d+/\d+\s+\|\s+(\S+)\s+starting", low)
        if m:
            self._mark_model_running(m.group(1))
            return
        # Model done: "[gpu-batch] Model 01/4 done | GPU_SH20_RK4: ..."
        m = re.search(r"\[gpu-batch\]\s+model\s+\d+/\d+\s+done\s+\|\s+([^\s:]+)", low)
        if m:
            key = self._ensure_model_chip(m.group(1))
            self._set_chip_status(key, "completed")
            return
        # Errors carry the base model name, not the display identity.
        m = re.search(r"\[gpu-batch\]\s+error\s+(\S+)", low)
        if m:
            self._mark_failed_by_base(m.group(1).rstrip(":"))
            return
        m = re.match(r"\[gpu-batch\]\s+(\S+)\s+failed:", low)
        if m:
            self._mark_failed_by_base(m.group(1))
            return

        if re.match(r"\[harness\]\s+(computing aggregate|generating|writing)", low):
            self._mark_report_running()
            return

    # ------------------------------------------------------------------
    # Pipeline state transitions
    # ------------------------------------------------------------------
    def _on_phase(self, phase: str, model: Optional[str]) -> None:
        """Map a structured phase signal onto pipeline chip transitions."""
        if phase == "truth":
            self._mark_truth_running()
        elif phase == "gpu_model" and model:
            self._mark_model_running(model)
        elif phase in ("report", "aggregate"):
            self._mark_report_running()
        # "scenario" / "sweep" carry no per-model identity -> no chip change.

    def _mark_truth_running(self) -> None:
        if self._model_status.get("truth") not in ("completed", "cached"):
            self._set_chip_status("truth", "running")

    def _mark_model_running(self, model: str) -> None:
        # Insert the chip live (handles single + Δt-variant identities in order).
        model = self._ensure_model_chip(model)
        # Truth precedes the model sweep.
        if self._model_status.get("truth") in ("running", "pending", None):
            self._set_chip_status("truth", "completed")
        # Any other model still shown as running has finished.
        for key, status in list(self._model_status.items()):
            if key not in ("truth", "report") and key != model and status == "running":
                self._set_chip_status(key, "completed")
        if self._model_status.get(model) not in ("failed", "cached"):
            self._set_chip_status(model, "running")
        self._current_model = model

    def _mark_failed_by_base(self, base: str) -> None:
        """Mark a model failed from a base name (errors omit the Δt identity)."""
        base = _pipeline_key(base).rstrip(":")
        if self._current_model and base in self._current_model:
            self._set_chip_status(self._current_model, "failed")
            return
        for key in self._pipeline_order:
            if key not in ("truth", "report") and base in key:
                self._set_chip_status(key, "failed")
                return
        if self._current_model:
            self._set_chip_status(self._current_model, "failed")

    def _mark_report_running(self) -> None:
        if self._model_status.get("truth") in ("running", "pending", None):
            self._set_chip_status("truth", "completed")
        for key, status in list(self._model_status.items()):
            if key not in ("truth", "report") and status == "running":
                self._set_chip_status(key, "completed")
        if self._model_status.get("report") != "completed":
            self._set_chip_status("report", "running")

    def _finalize_pipeline(self, ok: bool) -> None:
        if ok:
            for key in self._pipeline_order:
                if key == "report":
                    self._set_chip_status("report", "completed")
                elif self._model_status.get(key) in ("running", "queued", "pending", None):
                    self._set_chip_status(key, "completed")
        else:
            for key in self._pipeline_order:
                if self._model_status.get(key) == "running":
                    self._set_chip_status(key, "failed")

    def _set_overall(self, pct: float) -> None:
        v = int(round(min(100.0, max(0.0, pct))))
        self.overall_bar.setRange(0, 100)
        self.overall_bar.setValue(v)
        self._overall_value.setText(f"{pct:.1f}%")
        self._st_overall_pct.setText(f"{pct:.1f}%")

    def _set_phase(self, pct: float) -> None:
        v = int(round(min(100.0, max(0.0, pct))))
        self.phase_bar.setRange(0, 100)
        self.phase_bar.setValue(v)
        self._phase_value.setText(f"{pct:.1f}%")

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
            return fail(
                "Missing script",
                "src/lunaris/surrogate/st_lrps/evaluation/compare_gravity_models.py not found.",
            )

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
            dt_list = self.rk4_dt_list.text().strip()
            if dt_list:
                args += ["--gpu-rk4-dt-s-list", dt_list]
            args += ["--workers", str(self.truth_workers.value())]
            args += ["--torch-dtype", self.torch_dtype.currentData() or "float32"]
            args += ["--gpu-fallback", self.gpu_fallback.currentData() or "error"]
            args += ["--batch-frame-mode", self.gpu_frame_mode.currentData() or "match_dynamics_engine"]
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
        self._reset_dashboard()
        self._set_badge("running")
        self._st_phase.setText("starting")
        self._gallery.clear_gallery()
        self._save_settings()
        self.runner.start(sys.executable, args, workdir=str(_REPO_ROOT))

    def _on_finished(self, exit_code, exit_status) -> None:
        ok = (exit_status == QProcess.ExitStatus.NormalExit) and (int(exit_code) == 0)
        self._finalize_pipeline(ok)
        self._set_badge("completed" if ok else "failed")
        if ok:
            self._set_overall(100.0)
        out_dir = self._effective_out_dir
        if not out_dir or not Path(out_dir).is_dir():
            return
        imgs = self._discover_result_images(out_dir)
        if imgs:
            cnt = self._gallery.load_images(imgs)
            if cnt:
                self.runner.append(f"\n[UI] {cnt} plot(s) loaded: {out_dir}")
                # Reveal the in-place Results / Plots section once plots exist.
                self._results_section.set_expanded(True)
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
        s.setValue("gpu_frame_mode", self.gpu_frame_mode.currentData())
        s.setValue("rk4_dt", self.rk4_dt.value())
        s.setValue("rk4_dt_list", self.rk4_dt_list.text())
        s.setValue("torch_dtype", self.torch_dtype.currentData() or "float32")
        s.setValue("gpu_fallback", self.gpu_fallback.currentData())
        s.setValue("rtol", self.rtol.text())
        s.setValue("atol", self.atol.text())
        s.setValue("max_step", self.max_step.value())
        s.setValue("st_lrps_dir", self.st_lrps_dir.text())
        s.setValue("out_dir", self.out_dir.text())
        s.setValue("logs_expanded", self._logs_section.is_expanded())
        s.setValue("results_expanded", self._results_section.is_expanded())
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
        _combo(self.gpu_frame_mode, "gpu_frame_mode")
        _combo(self.gpu_fallback, "gpu_fallback")
        _combo(self.torch_dtype, "torch_dtype")
        if s.contains("rtol"):
            self.rtol.setText(str(s.value("rtol", "1e-10")))
        if s.contains("atol"):
            self.atol.setText(str(s.value("atol", "1e-12")))
        if s.contains("rk4_dt_list"):
            self.rk4_dt_list.setText(str(s.value("rk4_dt_list", "")))
        if s.contains("st_lrps_dir"):
            self.st_lrps_dir.setText(str(s.value("st_lrps_dir", "")))
        if s.contains("out_dir"):
            self.out_dir.setText(str(s.value("out_dir", "")))
        if s.contains("cache_dir"):
            self.cache_dir.setText(str(s.value("cache_dir", "")))
        if s.contains("logs_expanded"):
            self._logs_section.set_expanded(
                str(s.value("logs_expanded", "true")).lower() == "true")
        if s.contains("results_expanded"):
            self._results_section.set_expanded(
                str(s.value("results_expanded", "false")).lower() == "true")
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


class OrbitBenchmarkPlotsTab(QWidget):
    """Cache-only *compare & plot* surface.

    Regenerates the comparison plots/report from **already-computed** benchmark
    trajectories (``--rebuild-metrics --reuse-cache``). This path returns before
    any propagation/torch import, so it can never accidentally launch a new
    benchmark run or training. The user just picks an existing output directory
    and which models to compare.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        # -- Models to compare (populated from the chosen folder) ----------
        grp_models = QGroupBox("2. Models to Compare")
        models_lo = QVBoxLayout()
        self._model_checks: dict[str, QCheckBox] = {}
        self._pending_models: set = set()
        self._models_grid = QGridLayout()
        self._models_grid.setContentsMargins(0, 0, 0, 0)
        self._model_grid_count = 0
        grid_w = QWidget()
        grid_w.setLayout(self._models_grid)
        models_lo.addWidget(grid_w)

        sel_row = QHBoxLayout()
        sel_row.setContentsMargins(0, 0, 0, 0)
        sel_row.setSpacing(8)
        self.btn_select_all = QPushButton("Select all")
        self.btn_select_all.setProperty("kind", "ghost")
        self.btn_select_all.clicked.connect(lambda: self._set_all_models(True))
        self.btn_select_none = QPushButton("Select none")
        self.btn_select_none.setProperty("kind", "ghost")
        self.btn_select_none.clicked.connect(lambda: self._set_all_models(False))
        sel_row.addWidget(self.btn_select_all)
        sel_row.addWidget(self.btn_select_none)
        sel_row.addStretch(1)
        sel_row_w = QWidget()
        sel_row_w.setLayout(sel_row)
        models_lo.addWidget(sel_row_w)

        self._models_status = QLabel(
            "Choose a results folder above — the models cached in it are listed here."
        )
        self._models_status.setWordWrap(True)
        self._models_status.setStyleSheet("color:#94a3b8; font-size:11px;")
        models_lo.addWidget(self._models_status)
        grp_models.setLayout(models_lo)

        # -- Source: pick a folder first, then its models are listed ------
        grp_src = QGroupBox("1. Results Folder")
        form = QFormLayout()
        _tune_form(form)
        self.out_dir = ValidatedPathEdit(
            placeholder=f"Existing results dir (e.g. {BENCHMARK_OUTPUT_ROOT})", check_file=False
        )
        btn_out = QPushButton("Select...")
        btn_out.clicked.connect(self._pick_out_dir)
        self.cache_dir = ValidatedPathEdit(
            placeholder="Empty -> output_dir/benchmark_cache", check_file=False
        )
        btn_cache = QPushButton("Select...")
        btn_cache.clicked.connect(self._pick_cache_dir)
        btn_rescan = QPushButton("Rescan folder")
        btn_rescan.setToolTip(
            "Re-read the folder and refresh the cached-model list and detected settings."
        )
        btn_rescan.clicked.connect(self._scan_and_populate)
        rescan_row = QHBoxLayout()
        rescan_row.setContentsMargins(0, 0, 0, 0)
        rescan_row.addWidget(btn_rescan)
        rescan_row.addStretch(1)
        rescan_row_w = QWidget()
        rescan_row_w.setLayout(rescan_row)
        self._scan_status = QLabel("")
        self._scan_status.setWordWrap(True)
        self._scan_status.setStyleSheet("color:#94a3b8; font-size:11px;")

        detected_note = QLabel("Detected from the cache — edit only if a value is wrong:")
        detected_note.setStyleSheet("color:#6f7ca8; font-size:11px;")
        self.truth = NoScrollComboBox()
        for t in _TRUTH_CHOICES:
            self.truth.addItem(t.upper(), t)
        self.truth.setCurrentIndex(_TRUTH_CHOICES.index("sh200"))
        self.truth_integrator = NoScrollComboBox()
        for ti in _TRUTH_INTEGRATORS:
            self.truth_integrator.addItem(ti, ti)
        self.rk4_dt = QDoubleSpinBox()
        self.rk4_dt.setDecimals(3)
        self.rk4_dt.setRange(0.001, 600.0)
        self.rk4_dt.setValue(10.0)
        self.rk4_dt.setToolTip("Must match the fixed step used when the cache was built.")
        self.rk4_dt_list = QLineEdit()
        self.rk4_dt_list.setPlaceholderText("Optional Δt list, e.g. 10,5 (multi-step cache)")
        form.addRow("Output dir", _row_lineedit_with_button(self.out_dir, btn_out))
        form.addRow("Cache dir", _row_lineedit_with_button(self.cache_dir, btn_cache))
        form.addRow("", rescan_row_w)
        form.addRow("", self._scan_status)
        form.addRow("", detected_note)
        form.addRow("Truth model", self.truth)
        form.addRow("Truth integrator", self.truth_integrator)
        form.addRow("RK4 dt (s)", self.rk4_dt)
        form.addRow("Δt list (s)", self.rk4_dt_list)
        grp_src.setLayout(form)

        # -- Command preview + actions ------------------------------------
        self.command_preview = QPlainTextEdit()
        self.command_preview.setReadOnly(True)
        self.command_preview.setFont(_mono_font())
        self.command_preview.setMaximumHeight(80)
        self.command_warning = QLabel("")
        self.command_warning.setWordWrap(True)
        self.command_warning.setStyleSheet("color:#fbbf24; font-size:11px;")
        safe_note = QLabel(
            "Plot-only: regenerates plots/report from cached trajectories. "
            "Does NOT propagate, run a benchmark, or train."
        )
        safe_note.setWordWrap(True)
        safe_note.setStyleSheet("color:#34d399; font-size:11px; font-weight:600;")

        # -- Runner + results ---------------------------------------------
        self.runner = ProcessPane()
        self.runner.btn_start.setText("Generate Plots")
        self.runner.btn_start.clicked.connect(self._start)
        self.runner.set_finished_hook(self._on_finished)
        self._gallery = ImageGallery()
        self._effective_out_dir = ""

        ctrl = QHBoxLayout()
        ctrl.setContentsMargins(0, 0, 0, 0)
        ctrl.setSpacing(8)
        ctrl.addWidget(self.runner.btn_start)
        ctrl.addWidget(self.runner.btn_stop)
        ctrl.addWidget(self.runner.btn_open_folder)
        btn_copy = QPushButton("Copy Command")
        btn_copy.setProperty("kind", "ghost")
        btn_copy.clicked.connect(self._copy_command)
        ctrl.addStretch(1)
        ctrl.addWidget(btn_copy)
        ctrl_w = QWidget()
        ctrl_w.setLayout(ctrl)

        self._logs_section = CollapsibleSection("Logs / Diagnostics")
        log_inner = QVBoxLayout()
        log_inner.setContentsMargins(0, 6, 0, 0)
        self.runner.log.setMaximumHeight(220)
        log_inner.addWidget(self.runner.log)
        self._logs_section.set_content_layout(log_inner)
        self._logs_section.set_expanded(False)

        self._results_section = CollapsibleSection("Results / Plots")
        res_inner = QVBoxLayout()
        res_inner.setContentsMargins(0, 6, 0, 0)
        self._gallery.setMinimumHeight(540)
        res_inner.addWidget(self._gallery, 1)
        self._results_section.set_content_layout(res_inner)
        self._results_section.set_expanded(True)

        content = QWidget()
        col = QVBoxLayout()
        col.setContentsMargins(8, 8, 8, 8)
        col.setSpacing(12)
        col.addWidget(grp_src)
        col.addWidget(grp_models)
        col.addWidget(safe_note)
        col.addWidget(ctrl_w)
        col.addWidget(self.command_preview)
        col.addWidget(self.command_warning)
        col.addWidget(self._logs_section)
        col.addWidget(self._results_section)
        col.addStretch(1)
        content.setLayout(col)
        for g in (grp_models, grp_src):
            _tune_inputs(g)

        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        layout.addWidget(_scroll_wrap(content), 1)
        self.setLayout(layout)

        # Wiring
        for w in (self.truth, self.truth_integrator):
            w.currentIndexChanged.connect(self._refresh_command_preview)
        self.rk4_dt.valueChanged.connect(self._refresh_command_preview)
        for le in (self.out_dir, self.cache_dir, self.rk4_dt_list):
            le.textChanged.connect(self._refresh_command_preview)
        self._restore_settings()
        self._refresh_command_preview()

    # -- model selection ---------------------------------------------------
    def _add_model_checkbox(self, name: str, checked: bool = True) -> bool:
        name = str(name).strip().lower()
        if not name or name in self._model_checks:
            return False
        cb = QCheckBox(_pipeline_label(name))
        cb.setChecked(checked)
        cb.toggled.connect(self._refresh_command_preview)
        self._model_checks[name] = cb
        r, c = divmod(self._model_grid_count, 3)
        self._models_grid.addWidget(cb, r, c)
        self._model_grid_count += 1
        return True

    def _clear_model_checks(self) -> None:
        while self._models_grid.count():
            item = self._models_grid.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()
        self._model_checks = {}
        self._model_grid_count = 0

    @staticmethod
    def _model_sort_key(name: str):
        n = str(name).strip().lower()
        m = re.match(r"^sh(\d+)", n)
        if m:
            return (0, int(m.group(1)), n)
        if n.startswith("st_lrps"):
            return (2, 0, n)
        return (1, 0, n)

    def _populate_models(self, names: List[str], checked: Optional[set] = None) -> None:
        self._clear_model_checks()
        want = set(checked) if checked is not None else set(names)
        for name in names:
            self._add_model_checkbox(name, checked=(name in want))
        self._refresh_command_preview()

    def _set_all_models(self, value: bool) -> None:
        for cb in self._model_checks.values():
            cb.setChecked(value)

    def _selected_models(self) -> List[str]:
        return [n for n, cb in self._model_checks.items() if cb.isChecked()]

    # -- folder scan / auto-detection -------------------------------------
    def _resolve_cache_dir(self) -> Optional[Path]:
        """Effective cache dir: explicit cache override, else <out_dir>/benchmark_cache.

        If the chosen output dir already *is* a cache (holds ``models/`` or a
        manifest), it is used directly so the user can also point straight at a
        ``benchmark_cache`` folder.
        """
        cache = self.cache_dir.text().strip()
        if cache:
            return Path(cache)
        out = self.out_dir.text().strip()
        if not out:
            return None
        out_p = Path(out)
        if (out_p / "models").is_dir() or (out_p / "cache_manifest.json").exists():
            return out_p
        return out_p / "benchmark_cache"

    @staticmethod
    def _read_manifest(cache_dir: Path) -> Optional[dict]:
        path = cache_dir / "cache_manifest.json"
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _discover_cached_models(self, cache_dir: Path) -> List[str]:
        """Base model names that have at least one cached scenario trajectory."""
        root = cache_dir / "models"
        found: set = set()
        if root.is_dir():
            for sub in root.iterdir():
                if sub.is_dir() and any(sub.glob("scenario_*.npz")):
                    base = sub.name.strip().lower()
                    if base:
                        found.add(base)
        return sorted(found, key=self._model_sort_key)

    @staticmethod
    def _select_combo_value(combo, value) -> None:
        """Select a combo entry by data (case-insensitive), inserting it if absent."""
        if value is None:
            return
        sval = str(value)
        for i in range(combo.count()):
            if str(combo.itemData(i)).lower() == sval.lower():
                combo.setCurrentIndex(i)
                return
        combo.addItem(sval.upper(), sval)
        combo.setCurrentIndex(combo.count() - 1)

    def _apply_manifest(self, manifest: Optional[dict]) -> None:
        """Auto-fill truth / integrator / step size from the cache manifest."""
        if not isinstance(manifest, dict):
            return
        meta = manifest.get("metadata", {}) or {}
        self._select_combo_value(self.truth, meta.get("truth"))
        self._select_combo_value(self.truth_integrator, meta.get("truth_integrator"))
        dt_list = meta.get("gpu_rk4_dt_s_list") or []
        rk4 = meta.get("rk4_dt_s")
        if dt_list:
            self.rk4_dt_list.setText(",".join("%g" % float(v) for v in dt_list))
            try:
                self.rk4_dt.setValue(float(dt_list[0]))
            except (TypeError, ValueError):
                pass
        elif rk4 is not None:
            self.rk4_dt_list.setText("")
            try:
                self.rk4_dt.setValue(float(rk4))
            except (TypeError, ValueError):
                pass

    def _scan_and_populate(self) -> None:
        """Scan the chosen folder, auto-fill detected settings, list its models."""
        cache_dir = self._resolve_cache_dir()
        prev = set(self._selected_models()) or set(self._pending_models)
        self._pending_models = set()
        if cache_dir is None:
            self._clear_model_checks()
            self._scan_status.setText("Pick an output (or cache) folder to scan for models.")
            self._models_status.setText(
                "Choose a results folder above — the models cached in it are listed here.")
            self._refresh_command_preview()
            return
        if not cache_dir.is_dir():
            self._clear_model_checks()
            self._scan_status.setText(f"No benchmark cache at: {cache_dir}")
            self._models_status.setText(
                "No cache here. Pick the results folder that contains 'benchmark_cache'.")
            self._refresh_command_preview()
            return
        manifest = self._read_manifest(cache_dir)
        self._apply_manifest(manifest)
        models = self._discover_cached_models(cache_dir)
        if not models:
            self._clear_model_checks()
            self._scan_status.setText(
                f"Cache found ({cache_dir}) but it holds no model trajectories yet.")
            self._models_status.setText("No cached model trajectories were found in this folder.")
            self._refresh_command_preview()
            return
        keep = (prev & set(models)) or set(models)
        self._populate_models(models, checked=keep)
        src = "cache_manifest.json" if manifest else "folder scan"
        self._scan_status.setText(f"Cache: {cache_dir}  •  detected via {src}")
        self._models_status.setText(
            f"{len(models)} cached model(s) found — tick the ones to plot.")

    # -- pickers -----------------------------------------------------------
    def _pick_out_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(
            self, "Existing results dir", self.out_dir.text() or str(BENCHMARK_OUTPUT_ROOT))
        if d:
            self.out_dir.setText(_norm_path(d))
            self._scan_and_populate()

    def _pick_cache_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(
            self, "Benchmark cache dir", self.cache_dir.text() or str(BENCHMARK_OUTPUT_ROOT))
        if d:
            self.cache_dir.setText(_norm_path(d))
            self._scan_and_populate()

    # -- command -----------------------------------------------------------
    def _build_args(self, show_errors: bool = True) -> Optional[List[str]]:
        def fail(title: str, message: str) -> Optional[List[str]]:
            if show_errors:
                QMessageBox.critical(self, title, message)
            else:
                self.command_warning.setText(message)
            return None

        if not show_errors:
            self.command_warning.setText("")
        models = self._selected_models()
        if not models:
            return fail("No models", "Select at least one model to compare.")
        out_dir = self.out_dir.text().strip()
        if not out_dir:
            return fail("Output dir", "Choose the existing results/output directory.")

        args = ["-u", "-m", BENCHMARK_CLI_MODULE,
                "--gpu-batch-compare", "--rebuild-metrics", "--reuse-cache",
                "--gpu-models", ",".join(models),
                "--truth", self.truth.currentData() or "sh200",
                "--truth-integrator", self.truth_integrator.currentData() or "DOP853",
                "--rk4-dt-s", str(self.rk4_dt.value())]
        args += ["--output-dir", out_dir]
        cache_dir = self.cache_dir.text().strip()
        if cache_dir:
            args += ["--cache-dir", cache_dir]
        return args

    def _refresh_command_preview(self, *_a) -> None:
        args = self._build_args(show_errors=False)
        if not args:
            self.command_preview.clear()
            return
        self.command_preview.setPlainText(_format_command(sys.executable, args))

    def _copy_command(self) -> None:
        if not self.command_preview.toPlainText().strip():
            self._refresh_command_preview()
        QGuiApplication.clipboard().setText(self.command_preview.toPlainText())

    # -- run / results -----------------------------------------------------
    def _start(self) -> None:
        args = self._build_args(show_errors=True)
        if not args:
            return
        out_dir = self.out_dir.text().strip()
        if not Path(out_dir).is_dir():
            QMessageBox.critical(self, "Output dir",
                                 f"Output directory does not exist:\n{out_dir}\n\n"
                                 "Run a benchmark first, then generate plots here.")
            return
        self._effective_out_dir = out_dir
        self.runner.set_output_dir(out_dir)
        self._gallery.clear_gallery()
        self._save_settings()
        self.runner.start(sys.executable, args, workdir=str(_REPO_ROOT))

    def _discover_result_images(self, out_dir: str) -> List[Path]:
        base = Path(out_dir)
        if not base.is_dir():
            return []
        imgs: List[Path] = list(base.glob("*.png")) + list(base.glob("*.jpg"))
        for sub in base.glob("*/"):
            if sub.is_dir():
                imgs += list(sub.glob("*.png")) + list(sub.glob("*.jpg"))
        return sorted(set(imgs))

    def _on_finished(self, exit_code, exit_status) -> None:
        out_dir = self._effective_out_dir
        if not out_dir or not Path(out_dir).is_dir():
            return
        imgs = self._discover_result_images(out_dir)
        if imgs:
            cnt = self._gallery.load_images(imgs)
            if cnt:
                self.runner.append(f"\n[UI] {cnt} plot(s) loaded: {out_dir}")
                self._results_section.set_expanded(True)
        self.runner.set_output_dir(out_dir)
        self.runner.btn_open_folder.setVisible(True)

    # -- persistence -------------------------------------------------------
    def _save_settings(self) -> None:
        s = _settings()
        s.beginGroup("orbit_benchmark_plots")
        s.setValue("models", ",".join(self._selected_models()))
        s.setValue("out_dir", self.out_dir.text())
        s.setValue("cache_dir", self.cache_dir.text())
        s.setValue("truth", self.truth.currentData())
        s.setValue("truth_integrator", self.truth_integrator.currentData())
        s.setValue("rk4_dt", self.rk4_dt.value())
        s.setValue("rk4_dt_list", self.rk4_dt_list.text())
        s.endGroup()
        s.sync()

    def _restore_settings(self) -> None:
        s = _settings()
        s.beginGroup("orbit_benchmark_plots")
        # Detected fields first; a folder scan below overrides these when the
        # cache carries a manifest, but they stand in if it does not.
        for combo, key in ((self.truth, "truth"), (self.truth_integrator, "truth_integrator")):
            if s.contains(key):
                idx = combo.findData(str(s.value(key)))
                if idx >= 0:
                    combo.setCurrentIndex(idx)
        if s.contains("rk4_dt"):
            try:
                self.rk4_dt.setValue(float(s.value("rk4_dt")))
            except (TypeError, ValueError):
                pass
        if s.contains("rk4_dt_list"):
            self.rk4_dt_list.setText(str(s.value("rk4_dt_list", "")))
        if s.contains("out_dir"):
            self.out_dir.setText(str(s.value("out_dir", "")))
        if s.contains("cache_dir"):
            self.cache_dir.setText(str(s.value("cache_dir", "")))
        if s.contains("models"):
            self._pending_models = {m for m in str(s.value("models", "")).split(",") if m}
        s.endGroup()
        # Folder-first: rescan the restored folder and list its cached models,
        # re-checking the previously selected ones that are still present.
        self._scan_and_populate()


class OrbitBenchmarkPlotsPage(QWidget):
    """Analysis workspace page: regenerate gravity-benchmark plots from cache."""

    def __init__(self, plots_tab: QWidget, parent: Optional[QWidget] = None):
        super().__init__(parent)
        lo = QVBoxLayout()
        lo.setContentsMargins(22, 20, 22, 20)
        lo.setSpacing(14)
        title = QLabel("Gravity Plots")
        title.setStyleSheet("font-size: 18px; font-weight: 700; color: #e8ecf8;")
        subtitle = QLabel(
            "Pick which models to compare and regenerate the comparison plots from "
            "previously-cached benchmark results — no new run or training is started."
        )
        subtitle.setStyleSheet("color: #94a3b8; font-size: 12px;")
        lo.addWidget(title)
        lo.addWidget(subtitle)
        lo.addWidget(plots_tab, 1)
        self.setLayout(lo)


__all__ = [
    "OrbitBenchmarkTab", "OrbitBenchmarkPage",
    "OrbitBenchmarkPlotsTab", "OrbitBenchmarkPlotsPage",
    "BENCHMARK_CLI_MODULE",
]
