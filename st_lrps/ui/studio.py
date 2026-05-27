# -*- coding: utf-8 -*-
"""
st_lrps.ui.studio  -  v3

PyQt6 dashboard for the lunar scalar potential surrogate codebase.
(Thin launcher for structural refactored UI)
"""

import os
import sys
from pathlib import Path

# Add repo root to sys.path so standalone script execution works
_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from st_lrps.ui.studio_parts.qt_common import (
    QApplication,
    QGuiApplication,
    Qt,
    SCRIPT_DIR,
    apply_premium_dark_theme,
)
from st_lrps.ui.studio_parts.common_widgets import _NoWheelOnSpinFilter
from st_lrps.ui.studio_parts.main_window import MainWindow

from st_lrps.ui.studio_parts.training_pages import STLRPSTrainTab
from st_lrps.ui.studio_parts.runtime_pages import STLRPSProfilingTab
from st_lrps.ui.studio_parts.evaluation_pages import STLRPSEvalTab
from st_lrps.ui.studio_parts.orbit_benchmark_pages import (
    OrbitBenchmarkTab,
    OrbitBenchmarkPage,
    BENCHMARK_CLI_MODULE,
)
from st_lrps.ui.studio_parts.qt_common import TRAIN_CLI_MODULE, PROFILE_CLI_MODULE

def main() -> None:
    try:
        QGuiApplication.setHighDpiScaleFactorRoundingPolicy(
            Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
        )
    except Exception:
        pass
    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")
    os.chdir(str(SCRIPT_DIR))
    app = QApplication(sys.argv)
    apply_premium_dark_theme(app)
    _wheel_guard = _NoWheelOnSpinFilter(app)
    app.installEventFilter(_wheel_guard)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
