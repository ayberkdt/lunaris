# -*- coding: utf-8 -*-
"""
ST-LRPS Studio.

PyQt6 dashboard for the lunar scalar-potential surrogate codebase. This module is
a thin launcher that wires together the structured UI components.
"""

import os
import sys

from lunaris.common.paths import project_root_from_file

from lunaris.surrogate.st_lrps.ui.studio_parts.qt_common import (
    QApplication,
    QGuiApplication,
    Qt,
    SCRIPT_DIR,
    apply_premium_dark_theme,
)
from lunaris.surrogate.st_lrps.ui.studio_parts.common_widgets import _NoWheelOnSpinFilter
from lunaris.surrogate.st_lrps.ui.studio_parts.main_window import MainWindow

from lunaris.surrogate.st_lrps.ui.studio_parts.training_pages import STLRPSTrainTab
from lunaris.surrogate.st_lrps.ui.studio_parts.runtime_pages import STLRPSProfilingTab
from lunaris.surrogate.st_lrps.ui.studio_parts.evaluation_pages import STLRPSEvalTab
from lunaris.surrogate.st_lrps.ui.studio_parts.orbit_benchmark_pages import (
    OrbitBenchmarkTab,
    OrbitBenchmarkPage,
    OrbitBenchmarkPlotsTab,
    OrbitBenchmarkPlotsPage,
    BENCHMARK_CLI_MODULE,
)
from lunaris.surrogate.st_lrps.ui.studio_parts.qt_common import TRAIN_CLI_MODULE, PROFILE_CLI_MODULE

def main() -> None:
    try:
        QGuiApplication.setHighDpiScaleFactorRoundingPolicy(
            Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
        )
    except Exception:
        pass
    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")
    os.chdir(str(project_root_from_file(__file__)))
    app = QApplication(sys.argv)
    apply_premium_dark_theme(app)
    _wheel_guard = _NoWheelOnSpinFilter(app)
    app.installEventFilter(_wheel_guard)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
