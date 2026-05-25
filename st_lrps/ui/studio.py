# -*- coding: utf-8 -*-
"""
st_lrps.ui.studio  -  v3

PyQt6 dashboard for the lunar scalar potential surrogate codebase.
(Thin launcher for structural refactored UI)
"""

import sys
from pathlib import Path

# Add repo root to sys.path so standalone script execution works
_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

# Re-export all functional parts
from st_lrps.ui.studio_parts.qt_common import *
from st_lrps.ui.studio_parts.common_widgets import *
from st_lrps.ui.studio_parts.data_pages import *
from st_lrps.ui.studio_parts.training_pages import *
from st_lrps.ui.studio_parts.evaluation_pages import *
from st_lrps.ui.studio_parts.runtime_pages import *
from st_lrps.ui.studio_parts.main_window import *
from st_lrps.ui.studio_parts.qt_common import _USE_PYSIDE
from st_lrps.ui.studio_parts.common_widgets import _HAS_PYQTGRAPH, _HAS_H5PY, _HAS_DASHBOARD_V2
from st_lrps.ui.studio_parts.common_widgets import _tune_form, _tune_inputs, _row_lineedit_with_button, _scroll_wrap, _settings, _read_json_if_exists, _split_cli_args, _format_command, _send_os_notification, _apply_status_tips, _cfg_value, _norm_path, _timestamp_slug, _safe_slug, _default_training_output_dir, _default_runtime_output_dir, _default_dataset_report_dir, _output_standard_text, _mono_font, _inspect_run_artifacts, _NoWheelOnSpinFilter
from st_lrps.ui.studio_parts.data_pages import _introspect_h5
from st_lrps.ui.studio_parts.training_pages import _base_preset, _load_user_presets, _save_user_preset, _delete_user_preset, _BUILTIN_PRESETS


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
