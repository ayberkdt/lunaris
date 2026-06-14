"""Qt binding compatibility and shared ST-LRPS application theme setup."""

from __future__ import annotations

import os
import sys
from pathlib import Path

def _pyside6_available() -> bool:
    try:
        import PySide6  # noqa: F401
    except ImportError:
        return False
    return True


# Prefer PySide6 (the binding the rest of Lunaris uses) whenever it is installed,
# independent of import order. The previous ``"PyQt6" not in sys.modules`` heuristic
# made the binding order-dependent: a test that imported PyQt6 first flipped the
# studio to PyQt6, which then clashed with the PySide6 main-app QApplication
# (e.g. ``QApplication.setFont`` rejecting a PyQt6 ``QFont``). PyQt6 stays as a
# fallback only for environments where PySide6 is genuinely absent.
_USE_PYSIDE = _pyside6_available()

try:
    if not _USE_PYSIDE:
        raise ImportError
    from PySide6.QtCore import (
        QEasingCurve,
        QEvent,
        QObject,
        QProcess,
        QProcessEnvironment,
        QPropertyAnimation,
        QSettings,
        QSize,
        Qt,
        QTimer,
        QUrl,
        Signal as pyqtSignal,
    )
    from PySide6.QtGui import (
        QColor,
        QDesktopServices,
        QFont,
        QGuiApplication,
        QIcon,
        QPalette,
        QPixmap,
        QSyntaxHighlighter,
        QTextCharFormat,
        QTextDocument,
    )
    from PySide6.QtWidgets import (
        QAbstractSpinBox,
        QApplication,
        QBoxLayout,
        QCheckBox,
        QComboBox,
        QDoubleSpinBox,
        QFileDialog,
        QFormLayout,
        QFrame,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QInputDialog,
        QLabel,
        QLineEdit,
        QListWidget,
        QListWidgetItem,
        QMainWindow,
        QMessageBox,
        QPlainTextEdit,
        QProgressBar,
        QPushButton,
        QScrollArea,
        QSizePolicy,
        QSpinBox,
        QSplitter,
        QStackedWidget,
        QSystemTrayIcon,
        QTabWidget,
        QToolButton,
        QVBoxLayout,
        QWidget,
    )
except ImportError:
    from PyQt6.QtCore import (
        QEasingCurve,
        QEvent,
        QObject,
        QProcess,
        QProcessEnvironment,
        QPropertyAnimation,
        QSettings,
        QSize,
        Qt,
        QTimer,
        QUrl,
        pyqtSignal,
    )
    from PyQt6.QtGui import (
        QColor,
        QDesktopServices,
        QFont,
        QGuiApplication,
        QIcon,
        QPalette,
        QPixmap,
        QSyntaxHighlighter,
        QTextCharFormat,
        QTextDocument,
    )
    from PyQt6.QtWidgets import (
        QAbstractSpinBox,
        QApplication,
        QBoxLayout,
        QCheckBox,
        QComboBox,
        QDoubleSpinBox,
        QFileDialog,
        QFormLayout,
        QFrame,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QInputDialog,
        QLabel,
        QLineEdit,
        QListWidget,
        QListWidgetItem,
        QMainWindow,
        QMessageBox,
        QPlainTextEdit,
        QProgressBar,
        QPushButton,
        QScrollArea,
        QSizePolicy,
        QSpinBox,
        QSplitter,
        QStackedWidget,
        QSystemTrayIcon,
        QTabWidget,
        QToolButton,
        QVBoxLayout,
        QWidget,
    )

from lunaris.ui_foundation import LOG_COLORS, THEME, build_app_stylesheet, with_alpha

QT_BINDING_NAME = "PySide6" if _USE_PYSIDE else "PyQt6"
if "pyqtgraph" not in sys.modules:
    os.environ.setdefault("PYQTGRAPH_QT_LIB", QT_BINDING_NAME)

SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_CLI_MODULE = "lunaris.surrogate.st_lrps.training.cli"
PROFILE_CLI_MODULE = "lunaris.surrogate.st_lrps.runtime.profiling"


def pyqtgraph_matches_qt(pg_module) -> bool:
    """Return False when pyqtgraph uses a different Qt binding."""
    qt_lib = getattr(getattr(pg_module, "Qt", None), "QT_LIB", None)
    return qt_lib in (None, QT_BINDING_NAME)


class NoScrollComboBox(QComboBox):
    def wheelEvent(self, event):
        event.ignore()


def apply_premium_dark_theme(app: QApplication) -> None:
    """Apply the shared Lunaris palette and global stylesheet."""
    app.setStyle("Fusion")
    app.setFont(QFont("Segoe UI", 10))
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(THEME["bg_space"]))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(THEME["fg_main"]))
    palette.setColor(QPalette.ColorRole.Base, QColor(THEME["bg_space"]))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(THEME["bg_card"]))
    palette.setColor(QPalette.ColorRole.Text, QColor(THEME["fg_main"]))
    palette.setColor(QPalette.ColorRole.Button, QColor(THEME["bg_card_alt"]))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(THEME["fg_main"]))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(THEME["bg_card"]))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(THEME["fg_main"]))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(THEME["accent"]))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(THEME["fg_inverse"]))
    palette.setColor(QPalette.ColorRole.Link, QColor(THEME["accent"]))
    app.setPalette(palette)
    app.setStyleSheet(build_app_stylesheet(THEME, LOG_COLORS))
