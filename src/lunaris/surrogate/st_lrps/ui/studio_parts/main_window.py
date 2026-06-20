"""
ST-LRPS Studio.

PySide6 dashboard for the lunar scalar potential surrogate codebase.

The model predicts residual potential dU(x); residual acceleration da is
computed from the gradient of that scalar field. ST-LRPS is a Sobolev-trained
lunar residual potential surrogate, not a classical q,p state-space model.

What you can do from the UI
---------------------------
- Train a potential surrogate   (runs `python -m lunaris.surrogate.st_lrps.training.cli` as a subprocess)
- Resume an interrupted training run
- Evaluate a surrogate run      (runs `python -m lunaris.surrogate.st_lrps.evaluation.cli` as a subprocess)
- Profile ST-LRPS runtime inference (runs `python -m lunaris.surrogate.st_lrps.runtime.profiling`)
- Browse evaluation plots inline (post-processing dashboard)
- Inspect runtime profiling summaries and plots
- Watch live loss curves during training (pyqtgraph)
- Queue multiple training runs for overnight execution

UX Architecture — Core features
---------------------------------
1–6.  Groups, Grid, Tooltips, Collapsible, QSettings, Path validation.

UX Architecture — Productivity features
---------------------------------
7–12. Image gallery, Log highlight, Auto-scroll, Presets, Post-run, Dependent params.

UX Architecture — Live & introspection features
-----------------------------
13. Live Loss Plotting: real-time pyqtgraph chart of train/val loss parsed from logs.
14. Dataset Introspection: auto-read HDF5 metadata (row count, attrs) on path selection.
15. Training Queue: enqueue multiple configs and run them sequentially overnight.

Run
---
  python -m lunaris.surrogate.st_lrps.ui.studio
"""

# =============================================================================
# 0. IMPORTS
# =============================================================================

from __future__ import annotations

from pathlib import Path
from typing import Any

from lunaris.common.paths import project_root_from_file

from .data_pages import CloudAnalysisTab, CloudGenTab, DataPage
from .evaluation_pages import EvaluationPage, STLRPSEvalTab
from .orbit_benchmark_pages import (
    OrbitBenchmarkPage,
    OrbitBenchmarkPlotsPage,
    OrbitBenchmarkPlotsTab,
    OrbitBenchmarkTab,
)
from .qt_common import *
from .runtime_pages import RuntimePerformancePage, STLRPSProfilingTab
from .training_pages import STLRPSTrainTab

# pyqtgraph — optional, graceful fallback
try:
    import pyqtgraph as pg

    _HAS_PYQTGRAPH = pyqtgraph_matches_qt(pg)
except ImportError:
    _HAS_PYQTGRAPH = False

# h5py — optional, for dataset introspection
try:
    import h5py  # noqa: F401  # availability probe for _HAS_H5PY

    _HAS_H5PY = True
except ImportError:
    _HAS_H5PY = False

try:
    from lunaris.surrogate.st_lrps.artifacts.manager import (
        CHECKPOINT_SCHEMA_VERSION,
        CRITICAL_CONFIG_FIELDS,
        compute_payload_sha256,
        make_run_layout,
        read_run_manifest,
    )
    from lunaris.surrogate.st_lrps.artifacts.manager import (
        load_checkpoint as load_artifact_checkpoint,
    )
    from lunaris.surrogate.st_lrps.artifacts.manager import (
        resolve_run_dir as resolve_artifact_run_dir,
    )
except Exception:  # pragma: no cover - UI remains usable without artifact deps
    CHECKPOINT_SCHEMA_VERSION = "st_lrps_checkpoint_v2"  # type: ignore[assignment]
    CRITICAL_CONFIG_FIELDS = tuple()  # type: ignore[assignment]
    compute_payload_sha256 = None  # type: ignore[assignment]
    load_artifact_checkpoint = None  # type: ignore[assignment]
    make_run_layout = None  # type: ignore[assignment]
    read_run_manifest = None  # type: ignore[assignment]
    resolve_artifact_run_dir = None  # type: ignore[assignment]

SCRIPT_DIR = Path(__file__).resolve().parent
_PRESETS_DIR = SCRIPT_DIR / "presets"

# The training/evaluation entry points are now subpackage modules launched via
# ``python -m``. Module execution requires the repo root (which contains the
# importable ``st_lrps`` package) as the subprocess working directory.
_REPO_ROOT = project_root_from_file(__file__)
_STLRPS_ROOT = SCRIPT_DIR.parents[1]
TRAIN_CLI_MODULE = "lunaris.surrogate.st_lrps.training.cli"
EVAL_CLI_MODULE = "lunaris.surrogate.st_lrps.evaluation.cli"

# Short, header-friendly labels for the model-representation presets.
_PRESET_SHORT = {
    "baseline_raw": "baseline",
    "recommended_physical_radial_decay": "phys-radial",
    "ablation_radial_separation": "abl:radial-sep",
    "ablation_radial_decay_scaled": "abl:radial-decay",
    "ablation_real_sh_low_degree": "abl:real-sh",
    "custom": "custom",
}
PROFILE_CLI_MODULE = "lunaris.surrogate.st_lrps.runtime.profiling"
# Filesystem locations are used only for preflight existence checks; launching
# always goes through ``-m`` so package-relative imports resolve correctly.
TRAIN_CLI_PATH = _STLRPS_ROOT / "training" / "cli.py"
EVAL_CLI_PATH = _STLRPS_ROOT / "evaluation" / "cli.py"
PROFILE_CLI_PATH = _STLRPS_ROOT / "runtime" / "profiling.py"

OUTPUT_ROOT = _REPO_ROOT / "outputs"
TRAINING_OUTPUT_ROOT = OUTPUT_ROOT / "training"
DATASET_REPORTS_OUTPUT_ROOT = OUTPUT_ROOT / "dataset_reports"
RUNTIME_PERFORMANCE_OUTPUT_ROOT = OUTPUT_ROOT / "runtime"
EVALUATION_OUTPUT_ROOT = OUTPUT_ROOT / "evaluations"
DATASET_SUITE_OUTPUT_ROOT = OUTPUT_ROOT / "datasets" / "cloud_suites"

# UI defaults are intentionally read from the generator configuration module.
# This keeps the dashboard from drifting away from the command-line SSOT when
# dataset-suite sizes, altitude ranges, seeds, or sampling knobs are tuned.
try:
    from lunaris.surrogate.st_lrps.data.spatial_cloud_parameters import (
        DEFAULT_CLOUD_SUITE_CONFIG,
        DEFAULT_SPATIAL_CLOUD_CONFIG,
        SUITE_PRESETS,
    )
except Exception:  # pragma: no cover - UI remains usable without generator deps
    DEFAULT_CLOUD_SUITE_CONFIG = None  # type: ignore[assignment]
    DEFAULT_SPATIAL_CLOUD_CONFIG = None  # type: ignore[assignment]
    SUITE_PRESETS = {}  # type: ignore[assignment]


from lunaris.ui_foundation import DESIGN_TOKENS

from .common_widgets import *
from .common_widgets import (
    _apply_status_tips,
)


def _attr_lookup(attrs: dict[str, Any], *keys: str) -> Any:
    """Return the first present attribute among ``keys`` (case-insensitive)."""
    if not isinstance(attrs, dict):
        return None
    lower = {str(k).lower(): v for k, v in attrs.items()}
    for k in keys:
        if k in attrs:
            return attrs[k]
        if k.lower() in lower:
            return lower[k.lower()]
    return None


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ST-LRPS Studio")
        self.resize(1320, 860)
        self.setMinimumSize(1024, 680)

        # --- Underlying workflow widgets (logic preserved, pages re-homed) ---
        self._cloud_tab    = CloudGenTab()
        self._train_tab    = STLRPSTrainTab()
        self._profile_tab  = STLRPSProfilingTab()
        self._eval_tab     = STLRPSEvalTab()
        self._analysis_tab = CloudAnalysisTab()
        self._orbit_benchmark_tab = OrbitBenchmarkTab()
        self._orbit_plots_tab = OrbitBenchmarkPlotsTab()

        self._cloud_tab.set_train_tab(self._train_tab)
        self._cloud_tab.cloud_params_changed.connect(self._train_tab.sync_from_cloud)

        # --- Top-level workspace pages ---
        self._data_page = DataPage(self._cloud_tab, self._analysis_tab)
        self._train_setup_page = self._train_tab.setup_page
        self._train_monitor_page = self._train_tab.monitor_page
        self._eval_page = EvaluationPage(self._eval_tab)
        self._runtime_page = RuntimePerformancePage(self._profile_tab)
        self._orbit_benchmark_page = OrbitBenchmarkPage(self._orbit_benchmark_tab)
        self._orbit_plots_page = OrbitBenchmarkPlotsPage(self._orbit_plots_tab)
        self._data_page.inspect_panel.send_to_training.connect(self._on_dataset_to_training)
        self._train_tab.navigate_monitor_requested.connect(lambda: self._navigate(2))

        self._stack = QStackedWidget()
        self._stack.addWidget(self._data_page)              # index 0: Data
        self._stack.addWidget(self._train_setup_page)       # index 1: Training Setup
        self._stack.addWidget(self._train_monitor_page)     # index 2: Training Monitor
        self._stack.addWidget(self._eval_page)              # index 3: Evaluation
        self._stack.addWidget(self._runtime_page)           # index 4: Runtime Performance
        self._stack.addWidget(self._orbit_benchmark_page)   # index 5: Orbit-Level Benchmark
        self._stack.addWidget(self._orbit_plots_page)       # index 6: Gravity Plots
        self._page_titles = [
            "Data",
            "Training Setup",
            "Training Monitor",
            "Evaluation",
            "Runtime Performance",
            "Orbit-Level Benchmark",
            "Gravity Plots",
        ]

        dep_info = []
        if not _HAS_PYQTGRAPH:
            dep_info.append("pyqtgraph not installed (live plotting disabled)")
        if not _HAS_H5PY:
            dep_info.append("h5py not installed (dataset preview disabled)")

        # Page-specific headers now carry the workspace identity. The old global
        # ST-LRPS Studio header consumed vertical space without changing per page.
        self._experiment_header = None
        if hasattr(self._train_tab, "set_experiment_header"):
            self._train_tab.set_experiment_header(None)

        # --- Sidebar navigation ---
        self._nav_buttons: list[QPushButton] = []
        sidebar = self._build_sidebar()
        self._sidebar = sidebar

        # --- Main content area: sidebar + page stack ---
        content_area = QWidget()
        content_lo = QHBoxLayout()
        content_lo.setContentsMargins(0, 0, 0, 0)
        content_lo.setSpacing(10)
        content_lo.addWidget(sidebar)
        content_lo.addWidget(self._stack, 1)
        content_area.setLayout(content_lo)

        root = QWidget()
        root.setObjectName("centralRoot")
        root_lo = QVBoxLayout()
        root_lo.setContentsMargins(
            DESIGN_TOKENS.layout.shell_margin,
            DESIGN_TOKENS.layout.shell_margin,
            DESIGN_TOKENS.layout.shell_margin,
            DESIGN_TOKENS.layout.shell_margin,
        )
        root_lo.setSpacing(DESIGN_TOKENS.layout.shell_gap)
        root_lo.addWidget(content_area, 1)
        root.setLayout(root_lo)
        self.setCentralWidget(root)
        self._update_responsive_chrome(self.width())

        # --- Status bar: parameter descriptions are shown on hover ---
        sb = self.statusBar()
        sb.setSizeGripEnabled(False)
        sb.showMessage("Hover over a parameter - its description appears here.")

        # Wire every page's input-widget tooltips to the status bar
        for tab in (
            self._cloud_tab, self._analysis_tab,
            self._train_tab, self._profile_tab, self._eval_tab,
            self._orbit_benchmark_tab, self._orbit_plots_tab,
        ):
            _apply_status_tips(tab)

        # --- Header context badges: keep preset/dataset in sync while idle ---
        hdr = getattr(self, "_experiment_header", None)
        if hdr is not None and hasattr(hdr, "set_preset"):
            def _sync_preset():
                _p = self._train_tab.model_preset.currentData() or "custom"
                hdr.set_preset(_PRESET_SHORT.get(_p, _p))
            self._train_tab.model_preset.currentIndexChanged.connect(lambda *_: _sync_preset())
            _sync_preset()
            self._train_tab.data.textChanged.connect(
                lambda *_: hdr.set_dataset(Path(self._train_tab.data.text().strip()).name
                                           if self._train_tab.data.text().strip() else "—")
            )

    def _build_sidebar(self) -> QFrame:
        sidebar = QFrame()
        sidebar.setObjectName("navSidebar")
        sidebar.setFixedWidth(DESIGN_TOKENS.layout.nav_width)

        def _section_lbl(text: str) -> QLabel:
            lbl = QLabel(text)
            lbl.setObjectName("navSectionLabel")
            return lbl

        def _nav_btn(label: str, page_idx: int, hint: str = "") -> QPushButton:
            btn = QPushButton(label)
            btn.setObjectName("navButton")
            btn.setCheckable(True)
            btn.setAccessibleName(f"{label} workspace")
            if hint:
                btn.setToolTip(hint)
            btn.clicked.connect(lambda _c, i=page_idx: self._navigate(i))
            self._nav_buttons.append(btn)
            return btn

        def _group_box() -> QFrame:
            box = QFrame()
            box.setObjectName("navGroup")
            gl = QVBoxLayout(box)
            gl.setContentsMargins(4, 4, 4, 4)
            gl.setSpacing(2)
            return box, gl

        lo = QVBoxLayout()
        lo.setContentsMargins(10, 14, 10, 14)
        lo.setSpacing(6)

        # ── DATA ──
        lo.addWidget(_section_lbl("DATA"))
        lo.addWidget(_nav_btn("Data", 0, "Inspect, generate, and analyze ST-LRPS datasets."))

        # ── TRAINING (Setup + Monitor are one category, boxed together) ──
        lo.addWidget(_section_lbl("TRAINING"))
        train_box, train_l = _group_box()
        train_l.addWidget(_nav_btn("Training Setup", 1, "Configure data, model, loss, resume, and launch readiness."))
        train_l.addWidget(_nav_btn("Training Monitor", 2, "Track lifecycle, loss, checkpoints, logs, and queue state."))
        lo.addWidget(train_box)

        # ── ANALYSIS ──
        lo.addWidget(_section_lbl("ANALYSIS"))
        analysis_box, analysis_l = _group_box()
        analysis_l.addWidget(_nav_btn("Evaluation", 3, "Inspect artifacts and run in-band or OOD accuracy analysis."))
        analysis_l.addWidget(_nav_btn("Runtime Performance", 4, "Profile artifact loading, batching, chunks, and device throughput."))
        analysis_l.addWidget(_nav_btn("Orbit-Level Benchmark", 5, "Run orbit-level SH versus ST-LRPS validation scenarios."))
        analysis_l.addWidget(_nav_btn("Gravity Plots", 6, "Regenerate figures from cached benchmark results."))
        lo.addWidget(analysis_box)

        lo.addStretch(1)
        sidebar.setLayout(lo)

        self._navigate(0)
        return sidebar

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._update_responsive_chrome(event.size().width())

    def _update_responsive_chrome(self, window_width: int) -> None:
        sidebar = getattr(self, "_sidebar", None)
        if sidebar is None:
            return
        sidebar.setFixedWidth(
            DESIGN_TOKENS.layout.nav_compact_width
            if window_width < 1160
            else DESIGN_TOKENS.layout.nav_width
        )

    def _navigate(self, page_idx: int) -> None:
        self._stack.setCurrentIndex(page_idx)
        for i, btn in enumerate(self._nav_buttons):
            btn.setChecked(i == page_idx)
        # Reflect the active page in the header.
        hdr = getattr(self, "_experiment_header", None)
        if hdr is not None and hasattr(hdr, "set_page"):
            titles = getattr(self, "_page_titles", [])
            if 0 <= page_idx < len(titles):
                hdr.set_page(titles[page_idx])

        # Phase 10: Dynamically manage badges visibility on small screen scopes
        if hdr is not None:
            # Hide Preset and Dataset badges on pages where they aren't relevant to save header space
            has_preset = hasattr(hdr, "_preset")
            has_dataset = hasattr(hdr, "_dataset")
            if has_preset and has_dataset:
                is_train = page_idx in (1, 2)
                hdr._preset.setVisible(is_train)
                hdr._dataset.setVisible(is_train)

    def _on_dataset_to_training(self, path: str) -> None:
        """Data page → Training: load the chosen dataset and switch pages."""
        try:
            idx = self._train_tab.dataset_mode.findData("single")
            if idx >= 0:
                self._train_tab.dataset_mode.setCurrentIndex(idx)
            self._train_tab.data.setText(path)
        except Exception:
            pass
        self._navigate(1)

