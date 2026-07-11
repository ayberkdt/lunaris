# lunaris/ui/pages/batch_propagation_page.py
"""
Batch Propagation Analysis Page (Page 7)
========================================

Provides a dedicated PySide6 page for configuring, launching, and monitoring
batch orbital injection-dispersion runs. Random sampling is the Monte Carlo
option; LHS and Sobol designs support validation-oriented coverage.

Layout
------
The page is split into two workspace tabs:

1. **Setup & Run**
   Left column  (60%) — scrollable configuration cards:
     - Ensemble
     - Injection state dispersion
     - Spacecraft property dispersion
     - Backend / integration
     - Output / impact settings
   Right column (40%) — run controls + live metrics

2. **Result Analysis**
   A dedicated post-processing workspace for loading ensemble archives,
   computing statistics, previewing plots, and exporting a PDF report.

Integration with the rest of the application
---------------------------------------------
- ``get_data()``   → dict fed to ``build_batch_command()`` in command_builder.py
- ``load_data()``  → called by session_persistence to restore a saved profile
- ``update_results()`` → called by MainWindow after the batch subprocess finishes
  and the output file has been read back
- ``update_progress()`` → called by MainWindow for human-readable batch log lines
- ``update_progress_payload()`` → called by MainWindow for structured batch
  progress updates containing percent, stage, scenario counts, and ETA
"""

from __future__ import annotations

import math
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PySide6 import QtCore, QtGui, QtWidgets

try:
    from lunaris.ui.components.ensemble_analysis_panel import EnsembleAnalysisPanel
    from lunaris.ui.components.primitives import DataTable, InlineNotice, Section
    from lunaris.ui.core.ui_commons import (
        THEME,
        NoWheelComboBox,
        NumericDragLineEdit,
        ToggleSwitch,
        get_icon,
        prefers_reduced_motion,
    )
    from lunaris.ui.pages.force_models_page import ST_LRPS_RUNS_DIR, list_st_lrps_model_dirs
    from lunaris.ui.theme.tokens import DESIGN_TOKENS
except ImportError:
    if __name__ == "__main__" and (__package__ is None or __package__ == ""):
        import sys
        print("Run as:  python -m lunaris.ui.pages.batch_propagation_page", file=sys.stderr)
        raise SystemExit(2) from None
    raise


# =============================================================================
# 1.                       STATE DATACLASS
# =============================================================================

@dataclass
class UIBatchPropagationConfig:
    """
    Mutable mirror of ``common.batch_defs.BatchPropagationConfig`` for the UI.

    All values are kept as plain Python types so the page can safely serialize
    them to JSON (for session persistence) and pass them to the CLI argument
    builder without importing the heavy backend modules.
    """
    # Ensemble
    n_samples: int = 500
    seed: int = 42
    sampling_method: str = "random"

    # Injection state dispersion
    sigma_r_m: float = 500.0    # position 1-sigma [m]
    sigma_v_m_s: float = 0.5    # velocity 1-sigma [m/s]

    # Spacecraft property dispersion (0 = deterministic)
    sigma_mass_kg: float = 0.0
    sigma_area_m2: float = 0.0
    sigma_cd: float = 0.0
    sigma_cr: float = 0.0

    # Backend
    use_gpu: bool = True
    batch_backend: str = "auto"
    gpu_device_id: int = 0
    sh_degree: int = 10        # requested; true GPU classic-SH currently supports <=24
    gpu_threads_per_block: int = 128
    gravity_mode_override: str = "follow_mission"
    st_lrps_model_dir: str = ""

    # GPU torch-path tuning (torch_cuda_sh / GPU ST-LRPS)
    torch_dtype: str = "float64"       # "float32" or "float64"
    torch_sh_chunk_size: int = 0       # samples per GPU chunk; 0 = auto

    # Integration (GPU RK4 fixed-step)
    dt_s: float = 60.0             # RK4 step [s]
    max_vram_gb: float = 4.0

    # Output
    output_format: str = "hdf5"    # "hdf5" or "npz"
    output_path: str = "outputs/ensemble/batch_output.h5"
    result_storage_mode: str = "auto"
    max_result_memory_gb: float = 1.0

    # Impact detection
    detect_impact: bool = True
    compute_impact_statistics: bool = True
    impact_alt_km: float = 0.0

    # UQ covariance report (empty = skip report generation)
    uq_report_dir: str = ""


# =============================================================================
# 2.                       HELPER WIDGETS
# =============================================================================

def _detect_cuda_available() -> bool:
    """
    Best-effort CUDA availability probe for the batch page defaults.

    Returns True when *either* PyTorch CUDA (needed for ST-LRPS GPU path) or
    Numba CUDA (needed for classic-SH GPU path) is available.  Either probe
    failing is not fatal — we simply default the backend toggle to CPU.
    """

    # PyTorch CUDA — sufficient for ST-LRPS TorchBatchPropagator
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            return True
    except Exception:
        pass

    # Numba CUDA — required for the classic-SH GPUBatchPropagator
    try:
        from numba import cuda  # type: ignore
        if cuda.is_available():
            return True
    except Exception:
        pass

    return False


def _preferred_output_suffix(fmt: str) -> str:
    """Return the canonical filename suffix for the selected batch archive format."""

    return ".npz" if str(fmt).strip().lower() == "npz" else ".h5"


def _normalize_output_path_for_format(path_text: str, fmt: str) -> str:
    """
    Keep the visible output path aligned with the chosen archive format.
    """
    raw = str(path_text).strip()
    suffix = _preferred_output_suffix(fmt)
    if not raw:
        return f"outputs/ensemble/batch_output{suffix}"

    current = Path(raw)
    lower_name = current.name.lower()

    for known in (".h5", ".hdf5", ".npz"):
        if lower_name.endswith(known):
            if known == ".hdf5" and str(fmt).strip().lower() == "hdf5":
                return raw
            base = current.name[:-len(known)]
            return str(current.with_name(base + suffix))

    return raw

def _card(title: str) -> Section:
    """Card built from the shared ``Section`` primitive.

    Surface, border, radius, and title typography come from the global
    stylesheet (``QFrame#section`` / ``QLabel#sectionTitle``) instead of
    per-card inline QSS, keeping every page card token-driven and consistent.
    """
    return Section(title)


def _label(text: str, muted: bool = False) -> QtWidgets.QLabel:
    lbl = QtWidgets.QLabel(text)
    if muted:
        # ``QLabel#fieldHint`` is the global muted-caption style (fg_muted, 9pt).
        lbl.setObjectName("fieldHint")
    return lbl


def _repolish(widget: QtWidgets.QWidget) -> None:
    """Re-evaluate a widget's QSS after a dynamic property (e.g. ``kind``) change."""
    style = widget.style()
    style.unpolish(widget)
    style.polish(widget)
    widget.update()


def _metric_row(key: str, value: str = "—") -> QtWidgets.QHBoxLayout:
    row = QtWidgets.QHBoxLayout()
    row.addWidget(_label(key, muted=True))
    row.addStretch(1)
    val_lbl = QtWidgets.QLabel(value)
    val_lbl.setAlignment(QtCore.Qt.AlignRight)
    val_lbl.setObjectName("metricValue")
    # Monospace numerics so digits align column-wise across rows. Use
    # setFamilies()+Monospace style hint, NOT setFamily() with the comma-joined
    # token string (setFamily expects a single family and silently falls back to
    # the UI font, which also made family() order-dependent after polish).
    mono = val_lbl.font()
    mono.setStyleHint(QtGui.QFont.StyleHint.Monospace)
    mono.setFamilies(
        [f.strip().strip('"') for f in DESIGN_TOKENS.typography.family_mono.split(",")]
    )
    val_lbl.setFont(mono)
    row.addWidget(val_lbl)
    return row, val_lbl


def _format_clock_span(seconds: float | None) -> str:
    """
    Convert a duration in seconds to a compact human-readable clock string.

    The progress panel needs short, scan-friendly time stamps rather than the
    verbose natural-language durations usually used in message boxes.  Values
    are therefore rendered as ``MM:SS`` or ``H:MM:SS`` depending on span.
    """

    if seconds is None or not math.isfinite(float(seconds)) or float(seconds) < 0.0:
        return "\u2014"

    total = int(round(float(seconds)))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


# =============================================================================
# 3.                       MAIN PAGE WIDGET
# =============================================================================

class BatchPropagationPage(QtWidgets.QWidget):
    """
    Page 7: Batch propagation analysis - configuration + live metrics.

    Signals
    -------
    run_requested :
        Emitted when the user clicks "Run Batch".  The main window
        collects all page states, builds the CLI command, and spawns the
        backend process.
    """

    run_requested = QtCore.Signal()

    def __init__(
        self,
        batch_cfg: UIBatchPropagationConfig | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        if batch_cfg is None:
            batch_cfg = UIBatchPropagationConfig(use_gpu=_detect_cuda_available())
        self.batch_cfg = batch_cfg
        self._last_progress_payload: dict[str, Any] = {}
        self._build_ui()
        self._autoname_grid_fields()
        self._setup_validation_signals()
        self._update_validation()

    def _autoname_grid_fields(self) -> None:
        """Give every grid-placed input an accessible name from its row label.

        The configuration cards lay out ``label (col 0) | field (col 1)`` in raw
        QGridLayouts (not FormGrid), so the fields would otherwise reach screen
        readers unnamed. This walks each grid once and mirrors the column-0 label
        text into the field's accessible name, without overriding an explicit one.
        """
        need = (QtWidgets.QLineEdit, QtWidgets.QComboBox, QtWidgets.QAbstractSpinBox)
        for grid in self.findChildren(QtWidgets.QGridLayout):
            for i in range(grid.count()):
                item = grid.itemAt(i)
                field = item.widget() if item else None
                if not isinstance(field, need) or field.accessibleName().strip():
                    continue
                row, _col, _rs, _cs = grid.getItemPosition(grid.indexOf(field))
                label_item = grid.itemAtPosition(row, 0)
                label = label_item.widget() if label_item else None
                if isinstance(label, QtWidgets.QLabel) and label.text().strip():
                    field.setAccessibleName(label.text().rstrip(": ").strip())

    # -------------------------------------------------------------------------
    # UI construction
    # -------------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setDocumentMode(True)
        # Tab surfaces/states come from the global stylesheet (QTabWidget / QTabBar).
        root.addWidget(self.tabs, 1)

        run_tab = QtWidgets.QWidget()
        run_root = QtWidgets.QHBoxLayout(run_tab)
        run_root.setContentsMargins(0, 0, 0, 0)
        run_root.setSpacing(16)

        # ----- Left: scrollable configuration --------------------------------
        left_scroll = QtWidgets.QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        left_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)

        left_container = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_container)
        left_layout.setContentsMargins(0, 0, 8, 0)
        left_layout.setSpacing(12)

        left_layout.addWidget(self._card_ensemble())
        left_layout.addWidget(self._card_state_uncertainty())
        left_layout.addWidget(self._card_spacecraft_uncertainty())
        left_layout.addWidget(self._card_backend())
        left_layout.addWidget(self._card_integration())
        left_layout.addWidget(self._card_output())
        left_layout.addWidget(self._card_impact())
        left_layout.addStretch(1)

        left_scroll.setWidget(left_container)
        # Minimum column widths keep the two-column run tab from collapsing into
        # an unusable, overlapping layout at the window's minimum width (1000 px):
        # without these the right control card was crushed and its contents
        # visually collided. Both stay above their minimums at every supported
        # window size, so no horizontal page scrollbar is introduced.
        left_scroll.setMinimumWidth(320)
        run_root.addWidget(left_scroll, 6)

        # ----- Right: run controls + metrics (scrollable) --------------------
        # The right column must scroll independently like the left one. Without
        # its own scroll area the run-controls + 12-row results stack overflowed
        # short windows, squeezing widgets below their minimum size — which is
        # what made the text overlap and clipped the Run / Open Folder buttons.
        right_scroll = QtWidgets.QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        right_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)

        right_widget = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 8, 0)
        right_layout.setSpacing(12)

        right_layout.addWidget(self._card_run_controls())
        right_layout.addWidget(self._card_metrics())
        right_layout.addStretch(1)

        right_scroll.setWidget(right_widget)
        right_scroll.setMinimumWidth(340)
        run_root.addWidget(right_scroll, 4)

        # Backend comparison card lives below the existing run cards so users
        # discover it after configuring a baseline run.  The widget is
        # preview-only — building a command just writes it to the clipboard.
        left_layout.addWidget(self._card_backend_comparison())

        self.analysis_panel = EnsembleAnalysisPanel(parent=self)

        # "&&" renders a literal ampersand; a single "&" would become a
        # mnemonic and draw as an underscore in the tab title.
        self.tabs.addTab(run_tab, "Setup && Run")
        self.tabs.addTab(self.analysis_panel, "Result Analysis")

    # ------------------------------------------------------------------
    # Backend Command Preview (preview only — commands are copied, NOT executed)
    # ------------------------------------------------------------------

    def _card_backend_comparison(self) -> QtWidgets.QGroupBox:
        """
        Render a collapsible backend command preview card.

        Each row maps to a notional backend (classic-SH at three degrees plus
        ST-LRPS).  Generating the command formats the row's parameters into a
        ready-to-paste ``batch_runner.py`` command line.

        IMPORTANT: Nothing is executed here.  This section is PREVIEW ONLY.
        Commands must be copied and run manually in a terminal.
        """

        gb = _card("Backend Command Preview")
        outer = gb.content_layout
        outer.setSpacing(10)

        # Prominent preview-only notice
        notice = InlineNotice(
            "Preview only — commands are copied to clipboard, NOT executed.",
            "warning",
        )
        outer.addWidget(notice)

        # Header / collapse toggle
        header_row = QtWidgets.QHBoxLayout()
        self.btn_backend_compare_toggle = QtWidgets.QToolButton()
        self.btn_backend_compare_toggle.setCheckable(True)
        self.btn_backend_compare_toggle.setChecked(False)
        self.btn_backend_compare_toggle.setArrowType(QtCore.Qt.RightArrow)
        self.btn_backend_compare_toggle.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.btn_backend_compare_toggle.setText("Show command matrix")
        # QToolButton base style (borderless, 4px padding) comes from global QSS.
        self.btn_backend_compare_toggle.clicked.connect(self._toggle_backend_comparison)
        header_row.addWidget(self.btn_backend_compare_toggle)
        header_row.addStretch(1)
        outer.addLayout(header_row)

        self._backend_compare_body = QtWidgets.QWidget()
        body_layout = QtWidgets.QVBoxLayout(self._backend_compare_body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(8)

        intro = _label(
            "Each row generates a ready-to-paste batch_runner.py command line.\n"
            "Click 'Copy Command' on a row, or use 'Copy All' to copy all commands.\n"
            "Paste into a terminal to run. Comparisons are manual.",
            muted=True,
        )
        intro.setWordWrap(True)
        body_layout.addWidget(intro)

        self.tbl_backend_compare = DataTable()
        self.tbl_backend_compare.setAccessibleName("Backend comparison table")
        # Rows carry positional "Copy Command" buttons, so keep row order fixed.
        self.tbl_backend_compare.setSortingEnabled(False)
        self.tbl_backend_compare.setColumnCount(5)
        self.tbl_backend_compare.setHorizontalHeaderLabels(
            ["Backend", "Degree / Model", "batch Gravity Mode", "GPU", "Output Path"]
        )
        self.tbl_backend_compare.horizontalHeader().setStretchLastSection(True)
        self.tbl_backend_compare.verticalHeader().setVisible(False)
        self.tbl_backend_compare.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.tbl_backend_compare.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)

        rows = [
            ("SH-20", "20", "classic_sh", "On", "numba_cuda_sh", "outputs/ensemble/preview_sh20.h5"),
            ("SH-60", "60", "classic_sh", "On", "torch_cuda_sh", "outputs/ensemble/preview_sh60.h5"),
            ("SH-100", "100", "classic_sh", "Off", "cpu_sh", "outputs/ensemble/preview_sh100.h5"),
            ("ST-LRPS", "surrogate", "st_lrps", "On", "gpu_st_lrps_potential", "outputs/ensemble/preview_stlrps.h5"),
        ]
        self.tbl_backend_compare.setRowCount(len(rows))
        self._backend_compare_meta: list[dict[str, Any]] = []
        for r, (name, deg, mode, gpu, backend, out_path) in enumerate(rows):
            for c, text in enumerate((name, deg, mode, gpu, out_path)):
                item = QtWidgets.QTableWidgetItem(str(text))
                if c == 0:
                    item.setFont(QtGui.QFont(item.font().family(), item.font().pointSize(), QtGui.QFont.Bold))
                self.tbl_backend_compare.setItem(r, c, item)

            self._backend_compare_meta.append({
                "name": name,
                "degree": deg,
                "mode": mode,
                "gpu_on": gpu.lower() == "on",
                "batch_backend": backend,
                "output_path": out_path,
            })

        # "Copy Command" buttons — inserted into a separate column via cellWidget
        self.tbl_backend_compare.setColumnCount(6)
        self.tbl_backend_compare.setHorizontalHeaderLabels(
            ["Backend", "Degree / Model", "batch Gravity Mode", "GPU", "Output Path", "Action"]
        )
        for r in range(len(rows)):
            btn = QtWidgets.QPushButton("Copy Command")
            btn.setCursor(QtCore.Qt.PointingHandCursor)
            btn.clicked.connect(lambda _checked=False, row=r: self._generate_backend_compare_command(row))
            self.tbl_backend_compare.setCellWidget(r, 5, btn)

        self.tbl_backend_compare.setMinimumHeight(170)
        body_layout.addWidget(self.tbl_backend_compare)

        # Preview text — reuses the global monospace command-preview style.
        self.txt_backend_compare_cmd = QtWidgets.QPlainTextEdit()
        self.txt_backend_compare_cmd.setObjectName("commandPreview")
        self.txt_backend_compare_cmd.setAccessibleName("Backend command preview")
        self.txt_backend_compare_cmd.setReadOnly(True)
        self.txt_backend_compare_cmd.setMinimumHeight(80)
        self.txt_backend_compare_cmd.setPlaceholderText(
            "Click 'Copy Command' on a row — the generated command appears here and is copied to clipboard."
        )
        body_layout.addWidget(self.txt_backend_compare_cmd)

        bottom_row = QtWidgets.QHBoxLayout()

        btn_copy_selected = QtWidgets.QPushButton("Copy Selected Command")
        btn_copy_selected.setIcon(get_icon("fa6s.copy", THEME["fg_main"]))
        btn_copy_selected.clicked.connect(self._copy_selected_backend_command)
        bottom_row.addWidget(btn_copy_selected)

        btn_copy_all = QtWidgets.QPushButton("Copy All Commands")
        btn_copy_all.setIcon(get_icon("fa6s.clipboard-list", THEME["fg_main"]))
        btn_copy_all.clicked.connect(self._copy_all_backend_commands)
        bottom_row.addWidget(btn_copy_all)

        bottom_row.addStretch(1)
        body_layout.addLayout(bottom_row)

        outer.addWidget(self._backend_compare_body)
        self._backend_compare_body.setVisible(False)
        return gb

    def _toggle_backend_comparison(self, checked: bool) -> None:
        try:
            self._backend_compare_body.setVisible(bool(checked))
            self.btn_backend_compare_toggle.setArrowType(
                QtCore.Qt.DownArrow if checked else QtCore.Qt.RightArrow
            )
            self.btn_backend_compare_toggle.setText(
                "Hide command matrix" if checked else "Show command matrix"
            )
        except Exception:
            pass

    def _generate_backend_compare_command(self, row: int) -> None:
        """Render a single-row batch_runner.py command into the preview box."""
        try:
            if row < 0 or row >= len(self._backend_compare_meta):
                return
            meta = self._backend_compare_meta[row]
        except Exception:
            return

        runner = str((Path(__file__).resolve().parents[2] / "cli" / "batch_runner.py").resolve())
        try:
            python_exec = sys.executable
        except Exception:
            python_exec = "python"

        gravity_mode = str(meta.get("mode", "classic_sh"))
        degree = meta.get("degree", "20")
        gpu_on = bool(meta.get("gpu_on", True))
        out_path = str(meta.get("output_path", "outputs/ensemble/backend_compare.h5"))

        n_samples = "100"
        try:
            n_samples = str(int(float(self.ent_n_samples.text())))
        except Exception:
            pass

        cmd: list[str] = [python_exec, runner]
        cmd.extend(["--n-samples", n_samples])
        cmd.extend(["--sampling-method", str(self.cb_sampling_method.currentData() or "random")])
        cmd.extend(["--batch-gravity-mode", gravity_mode])
        cmd.extend(["--batch-backend", str(meta.get("batch_backend", "auto"))])
        cmd.extend(["--enable-sh", "on"])
        if gravity_mode == "classic_sh":
            try:
                cmd.extend(["--degree", str(int(degree))])
                cmd.extend(["--sh-degree", str(int(degree))])
            except Exception:
                pass
        cmd.extend(["--use-gpu", "on" if gpu_on else "off"])
        cmd.extend(["--batch-output-format", "hdf5"])
        cmd.extend(["--batch-output-path", out_path])

        if os.name == "nt":
            rendered = subprocess.list2cmdline(cmd)
        else:
            rendered = shlex.join(cmd)

        rendered = f"# {meta.get('name', 'Preview')} Command\n" + rendered

        self.txt_backend_compare_cmd.setPlainText(rendered)

        try:
            QtWidgets.QApplication.clipboard().setText(rendered)
        except Exception:
            pass

    def _copy_selected_backend_command(self) -> None:
        """Copy the currently previewed command to clipboard."""
        try:
            text = self.txt_backend_compare_cmd.toPlainText().strip()
            if text:
                QtWidgets.QApplication.clipboard().setText(text)
        except Exception:
            pass

    def _copy_all_backend_commands(self) -> None:
        """Generate and copy all backend preview commands to clipboard."""
        runner = str((Path(__file__).resolve().parents[2] / "cli" / "batch_runner.py").resolve())
        python_exec = sys.executable
        n_samples = "100"
        try:
            n_samples = str(int(float(self.ent_n_samples.text())))
        except Exception:
            pass
        all_cmds: list[str] = []
        for meta in getattr(self, "_backend_compare_meta", []):
            gravity_mode = str(meta.get("mode", "classic_sh"))
            degree = meta.get("degree", "20")
            gpu_on = bool(meta.get("gpu_on", True))
            out_path = str(meta.get("output_path", "outputs/ensemble/preview.h5"))
            cmd: list[str] = [python_exec, runner]
            cmd.extend(["--n-samples", n_samples])
            cmd.extend(["--sampling-method", str(self.cb_sampling_method.currentData() or "random")])
            cmd.extend(["--batch-gravity-mode", gravity_mode])
            cmd.extend(["--batch-backend", str(meta.get("batch_backend", "auto"))])
            cmd.extend(["--enable-sh", "on"])
            if gravity_mode == "classic_sh":
                try:
                    cmd.extend(["--degree", str(int(degree))])
                    cmd.extend(["--sh-degree", str(int(degree))])
                except Exception:
                    pass
            cmd.extend(["--use-gpu", "on" if gpu_on else "off"])
            cmd.extend(["--batch-output-format", "hdf5"])
            cmd.extend(["--batch-output-path", out_path])
            if os.name == "nt":
                rendered = subprocess.list2cmdline(cmd)
            else:
                rendered = shlex.join(cmd)
            all_cmds.append(f"# --- {meta.get('name', '?')} ---\n{rendered}")
        joined = "\n\n".join(all_cmds)
        try:
            QtWidgets.QApplication.clipboard().setText(joined)
            self.txt_backend_compare_cmd.setPlainText(joined)
        except Exception:
            pass

    # -------------------------------------------------------------------------
    # Configuration cards
    # -------------------------------------------------------------------------

    def _card_ensemble(self) -> QtWidgets.QGroupBox:
        gb = _card("Ensemble")
        grid = QtWidgets.QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setVerticalSpacing(10)
        grid.setHorizontalSpacing(12)

        grid.addWidget(_label("Number of Samples"), 0, 0)
        self.ent_n_samples = NumericDragLineEdit(
            str(self.batch_cfg.n_samples),
            step=50, min_value=2, max_value=100_000, decimals=0,
        )
        self.ent_n_samples.setToolTip("Total number of ensemble trajectories (N >= 2)")
        grid.addWidget(self.ent_n_samples, 0, 1)

        grid.addWidget(_label("Sampling"), 1, 0)
        self.cb_sampling_method = NoWheelComboBox()
        self.cb_sampling_method.setAccessibleName("Ensemble sampling method")
        self.cb_sampling_method.addItem("Random (Monte Carlo)", "random")
        self.cb_sampling_method.addItem("Latin Hypercube (LHS)", "lhs")
        self.cb_sampling_method.addItem("Sobol", "sobol")
        self.cb_sampling_method.addItem("Sobol scrambled", "sobol_scrambled")
        self.cb_sampling_method.setToolTip(
            "Select the ensemble design.\n"
            "Random is the classical Monte Carlo option; LHS and Sobol are "
            "space-filling designs for validation coverage."
        )
        sampling_idx = self.cb_sampling_method.findData(self.batch_cfg.sampling_method)
        self.cb_sampling_method.setCurrentIndex(sampling_idx if sampling_idx >= 0 else 0)
        grid.addWidget(self.cb_sampling_method, 1, 1)

        grid.addWidget(_label("Seed"), 2, 0)
        self.ent_seed = NumericDragLineEdit(
            str(self.batch_cfg.seed),
            step=1, min_value=0, max_value=2**31 - 1, decimals=0,
        )
        self.ent_seed.setToolTip("Seed for random, LHS, and scrambled Sobol reproducibility")
        grid.addWidget(self.ent_seed, 2, 1)

        gb.content_layout.addLayout(grid)
        return gb

    def _card_state_uncertainty(self) -> QtWidgets.QGroupBox:
        gb = _card("Injection State Dispersion  (1-σ, Isotropic)")
        layout = gb.content_layout
        layout.setSpacing(10)

        desc = _label(
            "Gaussian perturbations are applied to the nominal initial state\n"
            "via Y₀ = nominal + L·z,  z~N(0,I),  L = chol(P₀).",
            muted=True,
        )
        desc.setWordWrap(True)
        layout.addWidget(desc)

        grid = QtWidgets.QGridLayout()
        grid.setVerticalSpacing(8)
        grid.setHorizontalSpacing(12)

        grid.addWidget(_label("Position σᵣ  [m]"), 0, 0)
        self.ent_sigma_r = NumericDragLineEdit(
            str(self.batch_cfg.sigma_r_m),
            step=100, min_value=0, max_value=1e7, decimals=1,
        )
        self.ent_sigma_r.setToolTip("1-sigma injection position dispersion (isotropic, all axes)")
        grid.addWidget(self.ent_sigma_r, 0, 1)

        grid.addWidget(_label("Velocity σ_v  [m/s]"), 1, 0)
        self.ent_sigma_v = NumericDragLineEdit(
            str(self.batch_cfg.sigma_v_m_s),
            step=0.1, min_value=0, max_value=1e4, decimals=3,
        )
        self.ent_sigma_v.setToolTip("1-sigma injection velocity dispersion (isotropic, all axes)")
        grid.addWidget(self.ent_sigma_v, 1, 1)

        layout.addLayout(grid)

        self.lbl_sigma_summary = _label("", muted=True)
        layout.addWidget(self.lbl_sigma_summary)
        self._update_sigma_summary()

        self.ent_sigma_r.value_changed.connect(lambda _: self._update_sigma_summary())
        self.ent_sigma_v.value_changed.connect(lambda _: self._update_sigma_summary())

        return gb

    def _update_sigma_summary(self) -> None:
        try:
            r = float(self.ent_sigma_r.text())
            v = float(self.ent_sigma_v.text())
            self.lbl_sigma_summary.setText(
                f"Δr ≈ {r/1000:.3f} km   Δv ≈ {v:.3f} m/s"
            )
        except Exception:
            pass

    def _card_spacecraft_uncertainty(self) -> QtWidgets.QGroupBox:
        gb = _card("Spacecraft Property Dispersion  (optional)")
        layout = gb.content_layout
        layout.setSpacing(10)

        desc = _label("Zero σ = deterministic (no perturbation). Sampling uses truncated-normal (positive values only).", muted=True)
        desc.setWordWrap(True)
        layout.addWidget(desc)

        grid = QtWidgets.QGridLayout()
        grid.setVerticalSpacing(8)
        grid.setHorizontalSpacing(12)

        grid.addWidget(_label("σ Mass  [kg]"), 0, 0)
        self.ent_sigma_mass = NumericDragLineEdit(str(self.batch_cfg.sigma_mass_kg), step=1, min_value=0, decimals=2)
        grid.addWidget(self.ent_sigma_mass, 0, 1)

        grid.addWidget(_label("σ Area  [m²]"), 1, 0)
        self.ent_sigma_area = NumericDragLineEdit(str(self.batch_cfg.sigma_area_m2), step=0.01, min_value=0, decimals=3)
        grid.addWidget(self.ent_sigma_area, 1, 1)

        grid.addWidget(_label("σ C_D  [-]"), 2, 0)
        self.ent_sigma_cd = NumericDragLineEdit(str(self.batch_cfg.sigma_cd), step=0.01, min_value=0, decimals=3)
        grid.addWidget(self.ent_sigma_cd, 2, 1)

        grid.addWidget(_label("σ C_R  [-]"), 3, 0)
        self.ent_sigma_cr = NumericDragLineEdit(str(self.batch_cfg.sigma_cr), step=0.01, min_value=0, decimals=3)
        grid.addWidget(self.ent_sigma_cr, 3, 1)

        layout.addLayout(grid)
        return gb

    def _card_backend(self) -> QtWidgets.QGroupBox:
        gb = _card("Physics Backend")
        layout = gb.content_layout
        layout.setSpacing(12)

        # GPU / CPU toggle row
        toggle_row = QtWidgets.QHBoxLayout()
        toggle_row.addWidget(_label("Use GPU Acceleration"))
        self.toggle_gpu = ToggleSwitch()
        self.toggle_gpu.setAccessibleName("Use GPU acceleration")
        self.toggle_gpu.setChecked(self.batch_cfg.use_gpu)
        self.toggle_gpu.toggled.connect(self._on_backend_changed)
        toggle_row.addWidget(self.toggle_gpu)
        toggle_row.addStretch(1)
        layout.addLayout(toggle_row)

        gravity_row = QtWidgets.QHBoxLayout()
        gravity_row.addWidget(_label("Central Gravity Source"))
        self.cb_batch_gravity_mode = NoWheelComboBox()
        self.cb_batch_gravity_mode.setAccessibleName("Central gravity source")
        self.cb_batch_gravity_mode.addItem("Follow Mission Setup", "follow_mission")
        self.cb_batch_gravity_mode.addItem("Force Classical Gravity", "classic_sh")
        self.cb_batch_gravity_mode.addItem("Force ST-LRPS Gravity", "st_lrps")
        self.cb_batch_gravity_mode.currentIndexChanged.connect(self._on_gravity_mode_changed)
        gravity_row.addWidget(self.cb_batch_gravity_mode, 1)
        layout.addLayout(gravity_row)

        gravity_hint = _label(
            "Use this when you want batch propagation to reuse the mission gravity setup "
            "or explicitly force the classical SH model versus the ST-LRPS model.",
            muted=True,
        )
        gravity_hint.setWordWrap(True)
        layout.addWidget(gravity_hint)

        backend_row = QtWidgets.QHBoxLayout()
        backend_row.addWidget(_label("Batch Backend"))
        self.cb_batch_backend = NoWheelComboBox()
        self.cb_batch_backend.setAccessibleName("Batch propagation backend")
        self.cb_batch_backend.addItem("Auto Policy", "auto")
        self.cb_batch_backend.addItem("CPU Spherical Harmonics", "cpu_sh")
        self.cb_batch_backend.addItem("Numba CUDA SH — degree ≤ 24, low-degree screening", "numba_cuda_sh")
        self.cb_batch_backend.addItem("Torch CUDA SH — high-degree GPU, gravity-only", "torch_cuda_sh")
        self.cb_batch_backend.addItem("Torch CPU SH — validation, no CUDA needed", "torch_cpu_sh")
        self.cb_batch_backend.addItem("GPU ST-LRPS Potential", "gpu_st_lrps_potential")
        self.cb_batch_backend.addItem("GPU ST-LRPS + Third Body", "gpu_st_lrps_third_body")
        self.cb_batch_backend.setToolTip(
            "Explicit backend selector recorded verbatim in ensemble metadata.\n"
            "Numba CUDA SH: degree ≤ 24 (kernel-workspace limit). Torch CUDA SH: "
            "arbitrary degree on PyTorch CUDA, gravity-only.\n"
            "GPU ST-LRPS + Third Body keeps Earth/Sun third-body terms on the torch path.\n"
            "Auto uses safe GPU paths when available and records any fallback."
        )
        self.cb_batch_backend.currentIndexChanged.connect(self._on_batch_backend_changed)
        backend_row.addWidget(self.cb_batch_backend, 1)
        layout.addLayout(backend_row)

        backend_hint = _label(
            "Numba CUDA SH is a true CUDA path through degree 24 (kernel-workspace "
            "limit). Degrees above 24 use Torch CUDA SH (PyTorch, gravity-only) when "
            "available, otherwise fall back to CPU SH — the requested degree is never "
            "clipped.",
            muted=True,
        )
        backend_hint.setWordWrap(True)
        layout.addWidget(backend_hint)

        # ST-LRPS surrogate selection is only relevant when batch explicitly forces
        # the surrogate backend.  Leaving it blank intentionally falls back to
        # the global Force Models page setting.
        # Warning-tinted sub-panel reuses the global inlineNotice surface so the
        # emphasis is token-driven; inner inputs pick up the global QLineEdit style.
        self.st_lrps_config_frame = QtWidgets.QFrame()
        self.st_lrps_config_frame.setObjectName("inlineNotice")
        self.st_lrps_config_frame.setProperty("kind", "warning")
        st_lrps_layout = QtWidgets.QVBoxLayout(self.st_lrps_config_frame)
        st_lrps_layout.setContentsMargins(12, 10, 12, 12)
        st_lrps_layout.setSpacing(8)

        st_lrps_title_row = QtWidgets.QHBoxLayout()
        st_lrps_title = _label("ST-LRPS Model Run")
        st_lrps_title.setObjectName("sectionTitle")
        st_lrps_title_row.addWidget(st_lrps_title)
        st_lrps_title_row.addStretch(1)
        st_lrps_layout.addLayout(st_lrps_title_row)

        st_lrps_help = _label(
            "Select the trained ST-LRPS run used only by this batch propagation run. "
            "If left empty, batch falls back to the main Force Models ST-LRPS directory.",
            muted=True,
        )
        st_lrps_help.setWordWrap(True)
        st_lrps_layout.addWidget(st_lrps_help)

        st_lrps_path_row = QtWidgets.QHBoxLayout()
        self.ent_batch_st_lrps_model_dir = QtWidgets.QLineEdit(self.batch_cfg.st_lrps_model_dir)
        self.ent_batch_st_lrps_model_dir.setAccessibleName("ST-LRPS model directory")
        self.ent_batch_st_lrps_model_dir.setPlaceholderText(
            str(ST_LRPS_RUNS_DIR / "<trained_run>")
        )
        self.ent_batch_st_lrps_model_dir.setToolTip(
            "Path to a trained ST-LRPS run directory. Usually one folder under "
            "st_lrps/runs."
        )
        btn_st_lrps_browse = QtWidgets.QPushButton("Browse...")
        btn_st_lrps_browse.setIcon(get_icon("fa6s.folder-open", THEME["fg_muted"]))
        btn_st_lrps_browse.setCursor(QtCore.Qt.PointingHandCursor)
        btn_st_lrps_browse.clicked.connect(self._browse_st_lrps_model_dir)
        btn_st_lrps_latest = QtWidgets.QPushButton("Use Latest")
        btn_st_lrps_latest.setIcon(get_icon("fa6s.clock-rotate-left", THEME["fg_muted"]))
        btn_st_lrps_latest.setCursor(QtCore.Qt.PointingHandCursor)
        btn_st_lrps_latest.clicked.connect(self._use_latest_st_lrps_model_dir)
        st_lrps_path_row.addWidget(self.ent_batch_st_lrps_model_dir, 1)
        st_lrps_path_row.addWidget(btn_st_lrps_browse)
        st_lrps_path_row.addWidget(btn_st_lrps_latest)
        st_lrps_layout.addLayout(st_lrps_path_row)

        layout.addWidget(self.st_lrps_config_frame)

        # GPU-specific frame (hidden when CPU) — reuses the global section surface.
        self.gpu_frame = QtWidgets.QFrame()
        self.gpu_frame.setObjectName("section")
        gpu_grid = QtWidgets.QGridLayout(self.gpu_frame)
        gpu_grid.setContentsMargins(12, 12, 12, 12)
        gpu_grid.setVerticalSpacing(8)
        gpu_grid.setHorizontalSpacing(12)

        gpu_grid.addWidget(_label("Requested SH Degree"), 0, 0)
        self.ent_sh_degree = NumericDragLineEdit(
            str(self.batch_cfg.sh_degree),
            step=1, min_value=0, max_value=200, decimals=0,
        )
        self.ent_sh_degree.setToolTip(
            "Requested spherical-harmonic degree.\n"
            "The current true GPU classic-SH kernel supports degree <= 24.\n"
            "Higher values fall back to CPU SH with metadata, not silent clipping."
        )
        gpu_grid.addWidget(self.ent_sh_degree, 0, 1)

        gpu_grid.addWidget(_label("Threads/Block"), 1, 0)
        self.ent_tpb = NumericDragLineEdit(
            str(self.batch_cfg.gpu_threads_per_block),
            step=32, min_value=32, max_value=1024, decimals=0,
        )
        self.ent_tpb.setToolTip(
            "CUDA launch width hint.\n"
            "The runtime aligns this value to the active device warp size and hardware limits."
        )
        gpu_grid.addWidget(self.ent_tpb, 1, 1)

        gpu_grid.addWidget(_label("GPU Device ID"), 2, 0)
        self.ent_gpu_dev = NumericDragLineEdit(
            str(self.batch_cfg.gpu_device_id),
            step=1, min_value=0, max_value=7, decimals=0,
        )
        gpu_grid.addWidget(self.ent_gpu_dev, 2, 1)

        # Torch-path tuning (torch_cuda_sh / GPU ST-LRPS) — CLI parity for
        # --torch-dtype and --torch-sh-chunk-size.
        gpu_grid.addWidget(_label("Torch dtype"), 3, 0)
        self.cb_torch_dtype = NoWheelComboBox()
        self.cb_torch_dtype.setAccessibleName("Torch floating-point precision")
        self.cb_torch_dtype.addItem("float64 (reference precision)", "float64")
        self.cb_torch_dtype.addItem("float32 (faster, lower precision)", "float32")
        dtype_idx = self.cb_torch_dtype.findData(self.batch_cfg.torch_dtype)
        self.cb_torch_dtype.setCurrentIndex(dtype_idx if dtype_idx >= 0 else 0)
        self.cb_torch_dtype.setToolTip(
            "Floating-point dtype for the torch_cuda_sh and GPU ST-LRPS paths."
        )
        gpu_grid.addWidget(self.cb_torch_dtype, 3, 1)

        gpu_grid.addWidget(_label("Torch SH Chunk"), 4, 0)
        self.ent_torch_chunk = NumericDragLineEdit(
            str(self.batch_cfg.torch_sh_chunk_size),
            step=1024, min_value=0, max_value=10_000_000, decimals=0,
        )
        self.ent_torch_chunk.setAccessibleName("Torch SH chunk size")
        self.ent_torch_chunk.setToolTip(
            "Samples per GPU chunk on the torch_cuda_sh path. 0 = automatic "
            "(VRAM-aware) chunking."
        )
        gpu_grid.addWidget(self.ent_torch_chunk, 4, 1)

        # GPU-only warning banner
        warn_lbl = _label(
            "- Classic-SH GPU uses Numba CUDA and supports true SH through degree 24.\n"
            "- ST-LRPS Potential is gravity-only; ST-LRPS + Third Body adds analytic Sun/Earth terms.\n"
            "- Full-fidelity non-gravity perturbations can force CPU fallback depending on selected physics.",
            muted=True,
        )
        warn_lbl.setWordWrap(True)
        gpu_grid.addWidget(warn_lbl, 5, 0, 1, 2)

        layout.addWidget(self.gpu_frame)
        gravity_mode_index = self.cb_batch_gravity_mode.findData(self.batch_cfg.gravity_mode_override)
        if gravity_mode_index < 0:
            gravity_mode_index = 0
        self.cb_batch_gravity_mode.setCurrentIndex(gravity_mode_index)
        self.cb_batch_backend.setCurrentIndex(
            self._batch_backend_combo_index(str(getattr(self.batch_cfg, "batch_backend", "auto") or "auto"))
        )
        self._on_gravity_mode_changed()
        self._on_backend_changed(self.batch_cfg.use_gpu)

        # CPU hint
        self.cpu_hint = _label(
            "• CPU mode is slower but uses the full-fidelity propagation path.",
            muted=True,
        )
        self.cpu_hint.setWordWrap(True)
        layout.addWidget(self.cpu_hint)
        self.cpu_hint.setVisible(not self.batch_cfg.use_gpu)

        return gb

    def _on_backend_changed(self, gpu_on: bool) -> None:
        self.gpu_frame.setVisible(gpu_on)
        if hasattr(self, "cpu_hint"):
            self.cpu_hint.setVisible(not gpu_on)

    def _on_gravity_mode_changed(self, *_args: Any) -> None:
        """
        Show batch-specific ST-LRPS controls only when the surrogate backend is forced.

        The global Force Models page remains the default source of truth.  This
        panel is an explicit per-run override for experiments where the
        operator wants to compare different trained surrogate runs.
        """

        backend = str(self.cb_batch_backend.currentData() or "auto") if hasattr(self, "cb_batch_backend") else "auto"
        is_st_lrps = (
            str(self.cb_batch_gravity_mode.currentData() or "") == "st_lrps"
            or backend in {"gpu_st_lrps_potential", "gpu_st_lrps_third_body"}
        )
        if hasattr(self, "st_lrps_config_frame"):
            self.st_lrps_config_frame.setVisible(is_st_lrps)

    def _batch_backend_combo_index(self, value: str) -> int:
        """Resolve a stored batch_backend value to a combo index."""
        requested = str(value or "auto")
        idx = self.cb_batch_backend.findData(requested)
        if idx < 0:
            raise ValueError(f"Unknown batch backend in saved UI state: {requested!r}")
        return idx

    def _on_batch_backend_changed(self, *_args: Any) -> None:
        if hasattr(self, "cb_batch_backend"):
            self.batch_cfg.batch_backend = str(self.cb_batch_backend.currentData() or "auto")
        self._on_gravity_mode_changed()

    def _browse_st_lrps_model_dir(self) -> None:
        """Open a folder chooser rooted at the surrogate run directory."""

        current = self.ent_batch_st_lrps_model_dir.text().strip()
        if current:
            start_path = Path(current).expanduser()
            if start_path.is_file():
                start_path = start_path.parent
            if not start_path.exists():
                start_path = ST_LRPS_RUNS_DIR
        else:
            start_path = ST_LRPS_RUNS_DIR

        path = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Select ST-LRPS Run Directory",
            str(start_path),
        )
        if path:
            self.ent_batch_st_lrps_model_dir.setText(str(Path(path).expanduser().resolve()))

    def _use_latest_st_lrps_model_dir(self) -> None:
        """Fill the batch ST-LRPS directory with the newest valid trained ST-LRPS run."""

        runs = list_st_lrps_model_dirs(ST_LRPS_RUNS_DIR)
        if not runs:
            QtWidgets.QMessageBox.information(
                self,
                "No ST-LRPS Runs Found",
                "No valid lunar ST-LRPS run directory was found under:\n"
                f"{ST_LRPS_RUNS_DIR}",
            )
            return
        self.ent_batch_st_lrps_model_dir.setText(str(runs[0]))

    def _card_integration(self) -> QtWidgets.QGroupBox:
        gb = _card("Integration  (GPU RK4 and batching)")
        grid = QtWidgets.QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setVerticalSpacing(10)
        grid.setHorizontalSpacing(12)

        grid.addWidget(_label("RK4 Step  dt [s]"), 0, 0)
        self.ent_dt = NumericDragLineEdit(
            str(self.batch_cfg.dt_s),
            step=10, min_value=0.1, max_value=3600, decimals=1,
        )
        self.ent_dt.setToolTip(
            "Fixed time-step for the GPU RK4 integrator.\n"
            "60 s is adequate for LEO/LLO; reduce for high-eccentricity orbits."
        )
        grid.addWidget(self.ent_dt, 0, 1)
        grid.addWidget(_label("s"), 0, 2)

        grid.addWidget(_label("VRAM Budget  [GB]"), 1, 0)
        self.ent_vram = NumericDragLineEdit(
            str(self.batch_cfg.max_vram_gb),
            step=0.5, min_value=0.5, max_value=80.0, decimals=1,
        )
        self.ent_vram.setToolTip(
            "Maximum GPU memory used per sub-batch.\n"
            "Large ensembles are tiled to stay within this budget."
        )
        grid.addWidget(self.ent_vram, 1, 1)
        grid.addWidget(_label("GB"), 1, 2)

        gb.content_layout.addLayout(grid)
        return gb

    def _card_output(self) -> QtWidgets.QGroupBox:
        gb = _card("Output")
        layout = gb.content_layout
        layout.setSpacing(10)

        # Format
        fmt_row = QtWidgets.QHBoxLayout()
        fmt_row.addWidget(_label("Format"))
        self.cb_format = NoWheelComboBox()
        self.cb_format.setAccessibleName("Output archive format")
        self.cb_format.addItems(["hdf5", "npz"])
        self.cb_format.setCurrentText(self.batch_cfg.output_format)
        self.cb_format.currentTextChanged.connect(self._on_output_format_changed)
        fmt_row.addWidget(self.cb_format)
        fmt_row.addStretch(1)
        layout.addLayout(fmt_row)

        # Output path
        path_row = QtWidgets.QHBoxLayout()
        self.ent_output = QtWidgets.QLineEdit(self.batch_cfg.output_path)
        self.ent_output.setAccessibleName("Output archive path")
        self.ent_output.setPlaceholderText("outputs/ensemble/batch_output.h5")
        btn_browse = QtWidgets.QPushButton("Browse…")
        btn_browse.setFixedHeight(DESIGN_TOKENS.controls.compact_height)
        btn_browse.clicked.connect(self._browse_output)
        path_row.addWidget(self.ent_output, 1)
        path_row.addWidget(btn_browse)
        layout.addLayout(path_row)

        # UQ covariance report (CLI parity for --uq-report-dir)
        uq_row = QtWidgets.QHBoxLayout()
        uq_row.addWidget(_label("UQ Covariance Report"))
        self.toggle_uq_report = ToggleSwitch()
        self.toggle_uq_report.setAccessibleName("Generate UQ covariance report")
        self.toggle_uq_report.setChecked(bool(self.batch_cfg.uq_report_dir))
        self.toggle_uq_report.setToolTip(
            "Write a provenance-stamped UQ report (covariance history, RIC "
            "sigmas, error-ellipsoid figures, manifest) after the run."
        )
        self.toggle_uq_report.toggled.connect(self._on_uq_report_toggled)
        uq_row.addWidget(self.toggle_uq_report)
        uq_row.addStretch(1)
        layout.addLayout(uq_row)

        uq_path_row = QtWidgets.QHBoxLayout()
        self.ent_uq_report_dir = QtWidgets.QLineEdit(self.batch_cfg.uq_report_dir)
        self.ent_uq_report_dir.setAccessibleName("UQ report directory")
        self.ent_uq_report_dir.setPlaceholderText("outputs/ensemble/uq_report")
        self.btn_uq_browse = QtWidgets.QPushButton("Browse…")
        self.btn_uq_browse.setFixedHeight(DESIGN_TOKENS.controls.compact_height)
        self.btn_uq_browse.clicked.connect(self._browse_uq_report_dir)
        uq_path_row.addWidget(self.ent_uq_report_dir, 1)
        uq_path_row.addWidget(self.btn_uq_browse)
        layout.addLayout(uq_path_row)
        self._on_uq_report_toggled(self.toggle_uq_report.isChecked())

        self._on_output_format_changed(self.cb_format.currentText())

        return gb

    def _on_uq_report_toggled(self, enabled: bool) -> None:
        self.ent_uq_report_dir.setVisible(enabled)
        self.btn_uq_browse.setVisible(enabled)

    def _browse_uq_report_dir(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self, "UQ Report Directory", self.ent_uq_report_dir.text() or "outputs/ensemble"
        )
        if path:
            self.ent_uq_report_dir.setText(path)

    def _browse_output(self) -> None:
        fmt = self.cb_format.currentText()
        ext_filter = "HDF5 Files (*.h5 *.hdf5)" if fmt == "hdf5" else "NumPy Files (*.npz)"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Batch Propagation Output", self.ent_output.text(), ext_filter
        )
        if path:
            self.ent_output.setText(path)

    def _on_output_format_changed(self, fmt: str) -> None:
        """
        Keep the output placeholder and default path consistent with format changes.

        This is intentionally lightweight: user-chosen custom basenames are
        preserved, but legacy/default extensions are rewritten to the newly
        selected archive format.
        """

        normalized = _normalize_output_path_for_format(self.ent_output.text(), fmt)
        self.ent_output.setPlaceholderText(_normalize_output_path_for_format("", fmt))
        if normalized != self.ent_output.text().strip():
            self.ent_output.setText(normalized)

    def _card_impact(self) -> QtWidgets.QGroupBox:
        gb = _card("Impact Detection")
        grid = QtWidgets.QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setVerticalSpacing(10)
        grid.setHorizontalSpacing(12)

        grid.addWidget(_label("Impact Altitude Threshold"), 0, 0)
        self.ent_impact_alt = NumericDragLineEdit(
            str(self.batch_cfg.impact_alt_km),
            step=1, min_value=0, max_value=100, decimals=1,
        )
        self.ent_impact_alt.setToolTip(
            "Samples crossing below this altitude above the mean lunar surface\n"
            "are flagged as impacted and removed from further propagation."
        )
        grid.addWidget(self.ent_impact_alt, 0, 1)
        grid.addWidget(_label("km"), 0, 2)

        gb.content_layout.addLayout(grid)
        return gb

    # -------------------------------------------------------------------------
    # Run controls + metrics panels
    # -------------------------------------------------------------------------

    def _card_run_controls(self) -> QtWidgets.QGroupBox:
        gb = _card("Run Batch Propagation")
        layout = gb.content_layout
        layout.setSpacing(10)

        # Status validation label — global inline-notice text style, kind-driven.
        self.lbl_validation = _label("Configuration looks ready.")
        self.lbl_validation.setObjectName("inlineNoticeLabel")
        self.lbl_validation.setProperty("kind", "ok")
        self.lbl_validation.setWordWrap(True)
        layout.addWidget(self.lbl_validation)

        # Status badge row
        status_row = QtWidgets.QHBoxLayout()
        self.badge_batch = QtWidgets.QLabel("IDLE")
        self.badge_batch.setObjectName("statusBadge")
        self.badge_batch.setAlignment(QtCore.Qt.AlignCenter)
        self.badge_batch.setFixedHeight(DESIGN_TOKENS.controls.status_badge_height)
        self.badge_batch.setContentsMargins(10, 4, 10, 4)
        self.badge_batch.setProperty("kind", "info")
        status_row.addWidget(self.badge_batch)
        status_row.addStretch(1)
        layout.addLayout(status_row)

        # Progress bar
        self.progress_batch = QtWidgets.QProgressBar()
        self.progress_batch.setRange(0, 100)
        self.progress_batch.setValue(0)
        self.progress_batch.setTextVisible(True)
        self.progress_batch.setFixedHeight(16)
        # Bar/chunk styling comes from the global QProgressBar rules. Hidden
        # while idle: the summary/meta labels below already carry the waiting
        # state, so an empty bar would only add noise.
        self.progress_batch.setVisible(False)
        layout.addWidget(self.progress_batch)

        self.lbl_progress_summary = _label("Waiting for run", muted=False)
        self.lbl_progress_summary.setObjectName("statusValue")
        self.lbl_progress_summary.setWordWrap(True)
        layout.addWidget(self.lbl_progress_summary)

        self.lbl_progress_meta = _label("No active batch run", muted=True)
        self.lbl_progress_meta.setWordWrap(True)
        layout.addWidget(self.lbl_progress_meta)

        # Live log (last few lines) — reuses the global console surface style.
        self.txt_progress = QtWidgets.QPlainTextEdit()
        self.txt_progress.setObjectName("logConsole")
        self.txt_progress.setAccessibleName("Batch engine output log")
        self.txt_progress.setReadOnly(True)
        # Minimum (not fixed) height so the mini-log never clips its text under
        # larger fonts, HiDPI scaling, or localized strings; capped to keep the
        # control card compact.
        self.txt_progress.setMinimumHeight(80)
        self.txt_progress.setMaximumHeight(120)
        self.txt_progress.setPlaceholderText("Batch engine output appears here...")
        layout.addWidget(self.txt_progress)

        # Buttons row
        btn_row = QtWidgets.QHBoxLayout()

        self.btn_run_batch = QtWidgets.QPushButton("  Run Batch")
        self.btn_run_batch.setObjectName("primaryBtn")
        self.btn_run_batch.setIcon(get_icon("fa6s.dice", THEME["fg_main"]))
        self.btn_run_batch.setCursor(QtCore.Qt.PointingHandCursor)
        self.btn_run_batch.setFixedHeight(DESIGN_TOKENS.controls.primary_height)
        self.btn_run_batch.clicked.connect(self._on_run_clicked)

        self.btn_open_folder = QtWidgets.QPushButton("  Open Folder")
        self.btn_open_folder.setIcon(get_icon("fa6s.folder-open", THEME["fg_muted"]))
        self.btn_open_folder.setCursor(QtCore.Qt.PointingHandCursor)
        self.btn_open_folder.setFixedHeight(DESIGN_TOKENS.controls.primary_height)
        self.btn_open_folder.clicked.connect(self._open_output_folder)

        # Primary action on its own full-width row, the secondary action
        # below: side by side they exceed the rail's minimum width and the
        # layout clipped the secondary label ("Open Fold…").
        layout.addWidget(self.btn_run_batch)
        btn_row.addWidget(self.btn_open_folder)
        layout.addLayout(btn_row)

        return gb

    def _on_run_clicked(self) -> None:
        self._set_running(True)
        self.clear_results()
        self.txt_progress.clear()
        self.txt_progress.appendPlainText("[BATCH] Queuing batch propagation run...")
        self.run_requested.emit()

    def _open_output_folder(self) -> None:
        path = Path(self.ent_output.text()).expanduser().resolve().parent
        if path.exists():
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(path)))
        else:
            QtWidgets.QMessageBox.information(
                self, "Folder Not Found",
                f"Output directory does not exist yet:\n{path}"
            )

    def _card_metrics(self) -> QtWidgets.QGroupBox:
        gb = _card("Results  —  Last Run")
        layout = gb.content_layout
        layout.setSpacing(6)

        self._metric_labels: dict[str, QtWidgets.QLabel] = {}
        self._metric_order: list[tuple[str, QtWidgets.QLabel]] = []

        def _add(key: str, label: str) -> None:
            row_layout, val_lbl = _metric_row(label)
            layout.addLayout(row_layout)
            self._metric_labels[key] = val_lbl
            self._metric_order.append((label, val_lbl))

        _add("n_samples",      "N Samples")
        _add("sampling_method", "Sampling")
        _add("n_impacts",      "N Impacts")
        _add("p_impact",       "Impact Probability")
        _add("p_impact_ci95",  "95% CI")
        _add("t_impact_mean",  "Mean Impact Time")
        _add("alt_mean_0",     "Initial Mean Altitude")
        _add("alt_std_0",      "Initial Alt 1-σ")
        _add("alt_mean_f",     "Final Mean Altitude")
        _add("alt_std_f",      "Final Alt 1-σ")
        _add("wall_time",      "Wall Time")
        _add("backend",        "Backend")

        sep = QtWidgets.QFrame()
        sep.setObjectName("formDivider")
        sep.setFrameShape(QtWidgets.QFrame.HLine)
        sep.setFixedHeight(1)
        layout.addWidget(sep)

        self.btn_open_report = QtWidgets.QPushButton("  Open PDF Report")
        self.btn_open_report.setIcon(get_icon("fa6s.file-pdf", THEME["fg_muted"]))
        self.btn_open_report.setCursor(QtCore.Qt.PointingHandCursor)
        self.btn_open_report.setFixedHeight(DESIGN_TOKENS.controls.minimum_height)
        self.btn_open_report.setEnabled(False)
        self.btn_open_report.clicked.connect(self._open_report)
        layout.addWidget(self.btn_open_report)

        self.btn_copy_metrics = QtWidgets.QPushButton("  Copy Metrics (CSV)")
        self.btn_copy_metrics.setIcon(get_icon("fa6s.table-list", THEME["fg_muted"]))
        self.btn_copy_metrics.setCursor(QtCore.Qt.PointingHandCursor)
        self.btn_copy_metrics.setFixedHeight(DESIGN_TOKENS.controls.minimum_height)
        self.btn_copy_metrics.clicked.connect(self._copy_metrics_csv)
        layout.addWidget(self.btn_copy_metrics)

        self._last_report_path: str | None = None
        return gb

    def _metrics_to_csv(self) -> str:
        """Render the last-run metrics as ``Metric,Value`` CSV text."""
        import csv
        import io

        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(["Metric", "Value"])
        for label, val_lbl in getattr(self, "_metric_order", []):
            writer.writerow([label, val_lbl.text()])
        return buffer.getvalue()

    def _copy_metrics_csv(self) -> None:
        """Copy the last-run metrics (Metric,Value) to the clipboard as CSV."""
        try:
            QtWidgets.QApplication.clipboard().setText(self._metrics_to_csv())
        except Exception:
            pass

    def _open_report(self) -> None:
        if self._last_report_path and Path(self._last_report_path).exists():
            QtGui.QDesktopServices.openUrl(
                QtCore.QUrl.fromLocalFile(self._last_report_path)
            )

    # -------------------------------------------------------------------------
    # State management
    # -------------------------------------------------------------------------

    def _set_running(self, running: bool) -> None:
        """
        Toggle the page between idle and active-run states.

        The structured progress payload is the authoritative source once the
        backend starts emitting it.  Until then, the page shows a restrained
        warm-up state so the user immediately sees that the request was queued.
        """

        self.btn_run_batch.setEnabled(not running)
        if running:
            total = max(1, self._parse_int(self.ent_n_samples.text(), self.batch_cfg.n_samples))
            self._last_progress_payload = {}
            self._set_badge("RUNNING", "running")
            self.progress_batch.setVisible(True)
            # Reduced motion: keep the bar determinate instead of a marquee.
            if prefers_reduced_motion():
                self.progress_batch.setRange(0, 1000)
            else:
                self.progress_batch.setRange(0, 0)   # indeterminate until first structured payload
            self.progress_batch.setValue(0)
            self.progress_batch.setFormat("Preparing…")
            self.lbl_progress_summary.setText("Preparing ensemble samples")
            self.lbl_progress_meta.setText(f"0 / {total} scenarios | Waiting for backend")
        else:
            self._update_validation()
            if self.progress_batch.maximum() == 0:
                self.progress_batch.setRange(0, 1000)

    def _set_badge(self, text: str, kind: str = "info") -> None:
        """Update the status badge text + semantic kind (styled by global QSS)."""
        self.badge_batch.setText(text)
        self.badge_batch.setProperty("kind", kind)
        _repolish(self.badge_batch)

    def update_progress(self, line: str) -> None:
        """
        Append a human-readable batch log line to the page-local mini log.

        Structured progress payloads are handled by ``update_progress_payload``.
        This method only keeps the operator-facing narrative lines visible and
        retains a lightweight fallback progress parser for legacy output.
        """

        stripped = line.rstrip()
        if stripped.startswith("[BATCH_PROGRESS]") or stripped.startswith("[BATCH_METRICS]"):
            return

        self.txt_progress.appendPlainText(stripped)
        sb = self.txt_progress.verticalScrollBar()
        sb.setValue(sb.maximum())

        # Keep a minimal legacy fallback so older batch-only output still moves
        # the progress bar in a sensible way during development/debug runs.
        low = stripped.lower()
        if "batch" in low and "/" in stripped:
            try:
                parts = stripped.split()[1].split("/")
                done, total = int(parts[0]), int(parts[1])
                pct = float(done) / float(max(total, 1))
                self.progress_batch.setRange(0, 1000)
                self.progress_batch.setValue(int(round(pct * 1000.0)))
                self.progress_batch.setFormat(f"{pct * 100.0:.1f}%")
                self.lbl_progress_summary.setText("Propagating scenarios")
                self.lbl_progress_meta.setText(f"Batch {done}/{total}")
            except Exception:
                pass

    def update_progress_payload(self, payload: dict[str, Any]) -> None:
        """
        Render a structured backend progress payload in the batch control card.

        The backend emits machine-readable progress updates so the page can show
        a professional progress experience: phase label, overall percent,
        scenario-total context, batch position, and ETA.  This avoids brittle
        parsing of free-form log text.
        """

        self._last_progress_payload = dict(payload)

        stage = str(payload.get("stage", "propagating")).strip().lower()
        percent = max(0.0, min(100.0, float(payload.get("percent", 0.0) or 0.0)))
        fraction = max(0.0, min(1.0, float(payload.get("fraction", percent / 100.0) or 0.0)))
        total_samples = max(1, int(payload.get("total_samples", self._parse_int(self.ent_n_samples.text(), 1)) or 1))
        done_samples_raw = float(payload.get("done_samples", 0.0) or 0.0)
        done_samples_raw = max(0.0, min(float(total_samples), done_samples_raw))
        done_samples = int(math.floor(done_samples_raw + 1.0e-9))
        approx_done = abs(done_samples_raw - round(done_samples_raw)) > 1.0e-6
        batch_index = int(payload["batch_index"]) if "batch_index" in payload and payload.get("batch_index") is not None else None
        batch_count = int(payload["batch_count"]) if "batch_count" in payload and payload.get("batch_count") is not None else None
        eta_s = payload.get("eta_s")
        elapsed_s = payload.get("elapsed_s")
        backend = str(payload.get("backend", "") or "").strip().upper()
        detail = str(payload.get("detail", "") or "").strip()

        stage_summary = {
            "sampling": "Preparing ensemble samples",
            "propagating": "Propagating scenarios",
            "writing": "Writing ensemble results",
            "finalizing": "Finalizing ensemble archive",
        }.get(stage, "Running batch propagation")

        badge_text = {
            "sampling": "PREPARING",
            "propagating": "RUNNING",
            "writing": "WRITING",
            "finalizing": "FINALIZING",
        }.get(stage, "RUNNING")
        self._set_badge(badge_text, "running")

        self.progress_batch.setRange(0, 1000)
        self.progress_batch.setValue(int(round(fraction * 1000.0)))
        self.progress_batch.setFormat(f"{percent:.1f}%")

        summary_suffix = f" ({backend})" if backend and backend != "PENDING" else ""
        self.lbl_progress_summary.setText(stage_summary + summary_suffix)

        scenario_prefix = "~" if approx_done and stage == "propagating" else ""
        scenario_text = f"{scenario_prefix}{done_samples} / {total_samples} scenarios"
        meta_parts: list[str] = [scenario_text]
        if batch_index is not None and batch_count is not None and batch_index >= 1:
            meta_parts.append(f"Batch {batch_index}/{batch_count}")
        if eta_s is not None:
            meta_parts.append(f"ETA {_format_clock_span(float(eta_s))}")
        elif elapsed_s is not None:
            meta_parts.append(f"Elapsed {_format_clock_span(float(elapsed_s))}")
        if detail and stage != "propagating":
            meta_parts.append(detail)
        self.lbl_progress_meta.setText(" | ".join(meta_parts))

    def on_run_finished(
        self,
        exit_code: int,
        output_path: str,
        report_path: str | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        """
        Called by MainWindow when the batch subprocess exits.

        Parameters
        ----------
        exit_code : int
            Process exit code (0 = success).
        output_path : str
            Path to the HDF5/NPZ output file.
        report_path : str, optional
            Path to a generated PDF report (if available).
        metrics : dict, optional
            Pre-computed summary metrics to display.
        """
        self._set_running(False)
        if exit_code == 0:
            self._set_badge("DONE", "completed")
            total_samples = int((metrics or {}).get("n_samples", self._parse_int(self.ent_n_samples.text(), self.batch_cfg.n_samples)))
            self.progress_batch.setRange(0, 1000)
            self.progress_batch.setValue(1000)
            self.progress_batch.setFormat("100.0%")
            self.lbl_progress_summary.setText("Batch propagation completed")
            self.lbl_progress_meta.setText(f"{total_samples} / {total_samples} scenarios | Results ready")
            if metrics:
                self.update_results(metrics)
            if output_path:
                self.analysis_panel.set_result_path(output_path, auto_analyze=True)
                self.tabs.setCurrentWidget(self.analysis_panel)
            if report_path and Path(report_path).exists():
                self._last_report_path = report_path
                self.btn_open_report.setEnabled(True)
        else:
            self._set_badge("FAILED", "failed")
            if self.progress_batch.maximum() == 0:
                self.progress_batch.setRange(0, 1000)
            self.progress_batch.setFormat("Failed")
            self.lbl_progress_summary.setText("Batch propagation failed")
            if self._last_progress_payload:
                total_samples = max(1, int(self._last_progress_payload.get("total_samples", 1)))
                done_samples = int(min(total_samples, max(0.0, float(self._last_progress_payload.get("done_samples", 0.0) or 0.0))))
                self.lbl_progress_meta.setText(f"{done_samples} / {total_samples} scenarios | Review the execution log")
            else:
                self.lbl_progress_meta.setText("Review the execution log for the failure cause")

    def shutdown(self) -> None:
        """
        Stop background sub-components owned by the batch propagation page.

        The main window calls this during shutdown so the analysis workspace
        does not keep a background worker alive while the application exits.
        """

        if hasattr(self, "analysis_panel"):
            self.analysis_panel.shutdown()

    def update_results(self, metrics: dict[str, Any]) -> None:
        """
        Populate the metrics panel from a dict returned by the batch engine.

        Expected keys (all optional — missing keys show '—'):
            n_samples, n_impacts, p_impact, p_impact_ci95,
            t_impact_mean_days, alt_mean_0_km, alt_std_0_km,
            alt_mean_f_km, alt_std_f_km, wall_time_s, backend
        """
        def _set(key: str, value: str) -> None:
            lbl = self._metric_labels.get(key)
            if lbl is not None:
                lbl.setText(value)

        _set("n_samples",     str(metrics.get("n_samples", "—")))
        _set("sampling_method", str(metrics.get("sampling_method", "random")))
        _set("n_impacts",     str(metrics.get("n_impacts", "—")))

        p = metrics.get("p_impact")
        _set("p_impact", f"{p:.4f}" if p is not None else "—")

        ci = metrics.get("p_impact_ci95")
        if ci and len(ci) == 2:
            _set("p_impact_ci95", f"[{ci[0]:.4f}, {ci[1]:.4f}]")
        else:
            _set("p_impact_ci95", "—")

        t_d = metrics.get("t_impact_mean_days")
        _set("t_impact_mean", f"{t_d:.3f} d" if t_d is not None else "—")

        _set("alt_mean_0", f"{metrics.get('alt_mean_0_km', '—'):.2f} km" if "alt_mean_0_km" in metrics else "—")
        _set("alt_std_0",  f"{metrics.get('alt_std_0_km',  '—'):.2f} km" if "alt_std_0_km"  in metrics else "—")
        _set("alt_mean_f", f"{metrics.get('alt_mean_f_km', '—'):.2f} km" if "alt_mean_f_km" in metrics else "—")
        _set("alt_std_f",  f"{metrics.get('alt_std_f_km',  '—'):.2f} km" if "alt_std_f_km"  in metrics else "—")

        wt = metrics.get("wall_time_s")
        _set("wall_time", f"{wt:.1f} s" if wt is not None else "—")

        # Backend provenance: surface requested-vs-actual so a silent GPU->CPU
        # fallback is visible instead of hidden behind one label. Both keys are
        # emitted by batch_runner from the run diagnostics; when they differ the row
        # is marked as a warning and carries the fallback reason as a tooltip.
        backend_lbl = self._metric_labels.get("backend")
        if backend_lbl is not None:
            actual = str(metrics.get("actual_batch_backend") or metrics.get("backend") or "—")
            requested = str(metrics.get("requested_batch_backend") or "").strip()
            fell_back = bool(requested) and requested.lower() != actual.lower()
            if fell_back:
                backend_lbl.setText(f"{actual}  (requested: {requested})")
                reason = str(metrics.get("fallback_reason") or metrics.get("backend_note") or "").strip()
                backend_lbl.setToolTip(reason or f"Requested {requested} but ran on {actual}.")
                backend_lbl.setProperty("kind", "warning")
            else:
                backend_lbl.setText(actual)
                backend_lbl.setToolTip("")
                backend_lbl.setProperty("kind", "")
            _repolish(backend_lbl)

    def clear_results(self) -> None:
        """Reset all metric labels to '—'."""
        for lbl in self._metric_labels.values():
            lbl.setText("—")
        # Drop any backend-fallback warning styling/tooltip from a previous run.
        backend_lbl = self._metric_labels.get("backend")
        if backend_lbl is not None:
            backend_lbl.setToolTip("")
            backend_lbl.setProperty("kind", "")
            _repolish(backend_lbl)
        self.btn_open_report.setEnabled(False)
        self._last_report_path = None

    # -------------------------------------------------------------------------
    # Serialization (session persistence + command builder)
    # -------------------------------------------------------------------------

    def get_data(self) -> dict[str, Any]:
        """Return current UI state as a plain dict (JSON-serializable)."""
        return {
            "n_samples":             self._parse_int(self.ent_n_samples.text(), 500),
            "seed":                  self._parse_int(self.ent_seed.text(), 42),
            "sampling_method":        str(self.cb_sampling_method.currentData() or "random"),
            "sigma_r_m":             self._parse_float(self.ent_sigma_r.text(), 500.0),
            "sigma_v_m_s":           self._parse_float(self.ent_sigma_v.text(), 0.5),
            "sigma_mass_kg":         self._parse_float(self.ent_sigma_mass.text(), 0.0),
            "sigma_area_m2":         self._parse_float(self.ent_sigma_area.text(), 0.0),
            "sigma_cd":              self._parse_float(self.ent_sigma_cd.text(), 0.0),
            "sigma_cr":              self._parse_float(self.ent_sigma_cr.text(), 0.0),
            "use_gpu":               bool(self.toggle_gpu.isChecked()),
            "batch_backend":            str(self.cb_batch_backend.currentData() or "auto"),
            "gpu_device_id":         self._parse_int(self.ent_gpu_dev.text(), 0),
            "sh_degree":         self._parse_int(self.ent_sh_degree.text(), 10),
            "gpu_threads_per_block": self._parse_int(self.ent_tpb.text(), 128),
            "gravity_mode_override": str(self.cb_batch_gravity_mode.currentData() or "follow_mission"),
            "st_lrps_model_dir":     self.ent_batch_st_lrps_model_dir.text().strip(),
            "torch_dtype":           str(self.cb_torch_dtype.currentData() or "float64"),
            "torch_sh_chunk_size":   self._parse_int(self.ent_torch_chunk.text(), 0),
            "dt_s":                  self._parse_float(self.ent_dt.text(), 60.0),
            "max_vram_gb":           self._parse_float(self.ent_vram.text(), 4.0),
            "output_format":         self.cb_format.currentText(),
            "output_path":           _normalize_output_path_for_format(
                self.ent_output.text(),
                self.cb_format.currentText(),
            ),
            "result_storage_mode":   self.batch_cfg.result_storage_mode,
            "max_result_memory_gb":  self.batch_cfg.max_result_memory_gb,
            "detect_impact":         self.batch_cfg.detect_impact,
            "compute_impact_statistics": self.batch_cfg.compute_impact_statistics,
            "impact_alt_km":         self._parse_float(self.ent_impact_alt.text(), 0.0),
            "uq_report_dir":         (
                self.ent_uq_report_dir.text().strip()
                if self.toggle_uq_report.isChecked()
                else ""
            ),
        }

    def load_data(self, data: dict[str, Any]) -> None:
        """Restore UI state from a plain dict (e.g., loaded from JSON session)."""
        def _s(key: str, default) -> str:
            return str(data.get(key, default))

        self.ent_n_samples.setText(_s("n_samples", 500))
        self.ent_seed.setText(_s("seed", 42))
        sampling_method = str(data.get("sampling_method", "random") or "random")
        self.batch_cfg.sampling_method = sampling_method
        sampling_idx = self.cb_sampling_method.findData(sampling_method)
        self.cb_sampling_method.setCurrentIndex(sampling_idx if sampling_idx >= 0 else 0)
        self.ent_sigma_r.setText(_s("sigma_r_m", 500.0))
        self.ent_sigma_v.setText(_s("sigma_v_m_s", 0.5))
        self.ent_sigma_mass.setText(_s("sigma_mass_kg", 0.0))
        self.ent_sigma_area.setText(_s("sigma_area_m2", 0.0))
        self.ent_sigma_cd.setText(_s("sigma_cd", 0.0))
        self.ent_sigma_cr.setText(_s("sigma_cr", 0.0))
        self.toggle_gpu.setChecked(bool(data.get("use_gpu", True)))
        self.batch_cfg.batch_backend = str(data.get("batch_backend", "auto") or "auto")
        self.cb_batch_backend.setCurrentIndex(self._batch_backend_combo_index(self.batch_cfg.batch_backend))
        self.ent_gpu_dev.setText(_s("gpu_device_id", 0))
        self.ent_sh_degree.setText(_s("sh_degree", 10))
        self.ent_tpb.setText(_s("gpu_threads_per_block", 128))
        gravity_mode = str(data.get("gravity_mode_override", "follow_mission") or "follow_mission")
        gravity_idx = self.cb_batch_gravity_mode.findData(gravity_mode)
        if gravity_idx < 0:
            gravity_idx = 0
        self.cb_batch_gravity_mode.setCurrentIndex(gravity_idx)
        self.ent_batch_st_lrps_model_dir.setText(str(data.get("st_lrps_model_dir", "") or ""))
        self._on_gravity_mode_changed()
        torch_dtype = str(data.get("torch_dtype", "float64") or "float64")
        dtype_idx = self.cb_torch_dtype.findData(torch_dtype)
        self.cb_torch_dtype.setCurrentIndex(dtype_idx if dtype_idx >= 0 else 0)
        self.ent_torch_chunk.setText(_s("torch_sh_chunk_size", 0))
        self.ent_dt.setText(_s("dt_s", 60.0))
        self.ent_vram.setText(_s("max_vram_gb", 4.0))
        fmt = str(data.get("output_format", "hdf5"))
        idx = self.cb_format.findText(fmt)
        if idx >= 0:
            self.cb_format.setCurrentIndex(idx)
        self.ent_output.setText(
            _normalize_output_path_for_format(
                str(data.get("output_path", "outputs/ensemble/batch_output.h5")),
                fmt,
            )
        )
        self.batch_cfg.result_storage_mode = str(
            data.get("result_storage_mode", "auto") or "auto"
        )
        self.batch_cfg.max_result_memory_gb = float(
            data.get("max_result_memory_gb", 1.0)
        )
        self.batch_cfg.detect_impact = bool(data.get("detect_impact", True))
        self.batch_cfg.compute_impact_statistics = bool(
            data.get("compute_impact_statistics", True)
        )
        self.ent_impact_alt.setText(_s("impact_alt_km", 0.0))
        uq_dir = str(data.get("uq_report_dir", "") or "")
        self.ent_uq_report_dir.setText(uq_dir)
        self.toggle_uq_report.setChecked(bool(uq_dir))
        self._on_uq_report_toggled(bool(uq_dir))
        self._update_sigma_summary()

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _parse_float(text: str, default: float) -> float:
        try:
            return float(text)
        except Exception:
            return default

    @staticmethod
    def _parse_int(text: str, default: int) -> int:
        try:
            return int(float(text))
        except Exception:
            return default

    def validate_page_inputs(self) -> tuple[bool, list[str], list[str]]:
        ok = True
        errors: list[str] = []
        warnings: list[str] = []

        # 1. Ensemble
        n_samples = self._parse_int(self.ent_n_samples.text(), 0)
        seed = self._parse_int(self.ent_seed.text(), -1)

        if n_samples < 2:
            errors.append("Ensemble must have at least 2 samples.")
            ok = False
        if seed < 0:
            errors.append("Sampling seed must be non-negative.")
            ok = False

        # 2. Injection state dispersion
        sigma_r = self._parse_float(self.ent_sigma_r.text(), -1.0)
        sigma_v = self._parse_float(self.ent_sigma_v.text(), -1.0)

        if sigma_r < 0:
            errors.append("Injection position dispersion (σ_r) must be non-negative.")
            ok = False
        if sigma_v < 0:
            errors.append("Injection velocity dispersion (σ_v) must be non-negative.")
            ok = False

        # 3. Spacecraft property dispersion
        sigma_mass = self._parse_float(self.ent_sigma_mass.text(), -1.0)
        sigma_area = self._parse_float(self.ent_sigma_area.text(), -1.0)
        sigma_cd = self._parse_float(self.ent_sigma_cd.text(), -1.0)
        sigma_cr = self._parse_float(self.ent_sigma_cr.text(), -1.0)

        if any(v < 0 for v in (sigma_mass, sigma_area, sigma_cd, sigma_cr)):
            errors.append("Spacecraft property dispersions must be non-negative.")
            ok = False

        # 4. Backend
        gpu_enabled = self.toggle_gpu.isChecked()
        gravity_mode = self.cb_batch_gravity_mode.currentData() or "follow_mission"
        batch_backend = str(self.cb_batch_backend.currentData() or "auto")
        st_lrps_dir = self.ent_batch_st_lrps_model_dir.text().strip()

        if (
            gpu_enabled
            and batch_backend != "torch_cuda_sh"  # torch handles degree > 24 natively
            and (gravity_mode == "classic_sh" or batch_backend == "numba_cuda_sh")
        ):
            sh_deg = self._parse_int(self.ent_sh_degree.text(), 0)
            if sh_deg > 24:
                warnings.append(
                    "Requested SH degree > 24: numba_cuda_sh tops out at degree 24 "
                    "(kernel-workspace limit). Degrees above 24 route to torch_cuda_sh "
                    "(PyTorch CUDA, gravity-only) when available, else CPU SH — never clipped."
                )

        if not gpu_enabled:
            warnings.append("GPU disabled: CPU full-fidelity mode may be slower.")
            if batch_backend.startswith("gpu_") or batch_backend in {"numba_cuda_sh", "torch_cuda_sh"}:
                warnings.append("Explicit GPU batch backend selected; backend policy will record the resolved fallback or GPU override.")

        if (
            gravity_mode == "st_lrps"
            or batch_backend in {"gpu_st_lrps_potential", "gpu_st_lrps_third_body"}
        ) and not st_lrps_dir:
            warnings.append("ST-LRPS model dir is blank. batch will fall back to main Force Models setting.")

        # 5. Integration
        dt_s = self._parse_float(self.ent_dt.text(), 0.0)
        if dt_s <= 0:
            errors.append("Integration step size (dt) must be positive.")
            ok = False
        elif dt_s > 300:
            warnings.append("Large dt (> 300s) may reduce accuracy or cause numerical instability.")
        elif dt_s < 1:
            warnings.append("Small dt (< 1s) will produce heavy output and increase runtime.")

        # 6. Output
        out_path = self.ent_output.text().strip()
        fmt = self.cb_format.currentText()

        if not out_path:
            errors.append("Output path must not be empty.")
            ok = False
        if fmt not in ("hdf5", "npz"):
            errors.append(f"Invalid output format selected: {fmt}")
            ok = False
        else:
            suffix = _preferred_output_suffix(fmt)
            lower_name = Path(out_path).name.lower()
            if not lower_name.endswith(suffix):
                if fmt == "hdf5" and (lower_name.endswith(".h5") or lower_name.endswith(".hdf5")):
                    pass
                elif not lower_name.endswith((".h5", ".hdf5", ".npz")):
                    pass
                else:
                    warnings.append(f"Output path suffix does not match selected format '{fmt}'.")

        return ok, errors, warnings

    def _setup_validation_signals(self) -> None:
        def trigger(*args):
            self._update_validation()

        self.ent_n_samples.textChanged.connect(trigger)
        self.ent_seed.textChanged.connect(trigger)
        self.cb_sampling_method.currentIndexChanged.connect(trigger)
        self.ent_sigma_r.textChanged.connect(trigger)
        self.ent_sigma_v.textChanged.connect(trigger)
        self.ent_sigma_mass.textChanged.connect(trigger)
        self.ent_sigma_area.textChanged.connect(trigger)
        self.ent_sigma_cd.textChanged.connect(trigger)
        self.ent_sigma_cr.textChanged.connect(trigger)
        self.toggle_gpu.toggled.connect(trigger)
        self.cb_batch_gravity_mode.currentIndexChanged.connect(trigger)
        self.cb_batch_backend.currentIndexChanged.connect(trigger)
        self.ent_batch_st_lrps_model_dir.textChanged.connect(trigger)
        self.ent_sh_degree.textChanged.connect(trigger)
        self.ent_dt.textChanged.connect(trigger)
        self.ent_output.textChanged.connect(trigger)
        self.cb_format.currentIndexChanged.connect(trigger)

    def _update_validation(self) -> None:
        ok, errors, warnings = self.validate_page_inputs()
        if not ok:
            self.lbl_validation.setText("Errors:\n" + "\n".join(errors))
            self.lbl_validation.setProperty("kind", "error")
            self.btn_run_batch.setEnabled(False)
        elif warnings:
            self.lbl_validation.setText("Warnings:\n" + "\n".join(warnings))
            self.lbl_validation.setProperty("kind", "warn")
            self.btn_run_batch.setEnabled(True)
        else:
            self.lbl_validation.setText("Configuration looks ready.")
            self.lbl_validation.setProperty("kind", "ok")
            self.btn_run_batch.setEnabled(True)
        _repolish(self.lbl_validation)


# =============================================================================
# 4.                      STANDALONE TEST
# =============================================================================

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")

    win = QtWidgets.QMainWindow()
    win.setWindowTitle("Batch Ensemble Page - Test")
    win.resize(1100, 750)
    win.setStyleSheet(f"background-color: {THEME['bg_space']}; color: {THEME['fg_main']};")

    scroll = QtWidgets.QScrollArea()
    scroll.setWidgetResizable(True)
    page = BatchPropagationPage()

    def _on_run():
        print("[Test] run_requested signal received")
        print("[Test] get_data() =", page.get_data())

    page.run_requested.connect(_on_run)
    scroll.setWidget(page)
    win.setCentralWidget(scroll)
    win.show()
    sys.exit(app.exec())
