"""Qt binding compatibility and shared ST-LRPS application theme setup."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Lunaris standardizes on PySide6 for all Qt UI (mission app + ST-LRPS studio).
# PyQt6 support was removed in the Phase 3 consolidation: a single binding
# eliminates the order-dependent binding-pollution bug class (mixed PySide6/PyQt6
# QApplication / QFont / QIcon TypeErrors). ``Signal`` is aliased to ``pyqtSignal``
# so existing studio code that uses that name reads unchanged.
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
)
from PySide6.QtCore import (
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

from lunaris.ui_foundation import LOG_COLORS, THEME, build_app_stylesheet, with_alpha

QT_BINDING_NAME = "PySide6"

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
    # Same font pipeline as Mission Studio (registers platform font files, so
    # offscreen capture renders real glyphs instead of tofu). ui_foundation is
    # the sanctioned shared layer; lunaris.ui is contract-forbidden from here.
    from lunaris.ui_foundation.fonts import load_app_font

    app.setStyle("Fusion")
    app.setFont(load_app_font())
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


# Public hub surface. Studio page modules import these names explicitly from
# ``qt_common`` (no ``import *``); listing them here documents the shared surface
# and marks the re-exported Qt/theme symbols as intentional exports.
__all__ = [
    "LOG_COLORS",
    "NoScrollComboBox",
    "PROFILE_CLI_MODULE",
    "Path",
    "QAbstractSpinBox",
    "QApplication",
    "QBoxLayout",
    "QCheckBox",
    "QColor",
    "QComboBox",
    "QDesktopServices",
    "QDoubleSpinBox",
    "QEasingCurve",
    "QEvent",
    "QFileDialog",
    "QFont",
    "QFormLayout",
    "QFrame",
    "QGridLayout",
    "QGroupBox",
    "QGuiApplication",
    "QHBoxLayout",
    "QIcon",
    "QInputDialog",
    "QLabel",
    "QLineEdit",
    "QListWidget",
    "QListWidgetItem",
    "QMainWindow",
    "QMessageBox",
    "QObject",
    "QPalette",
    "QPixmap",
    "QPlainTextEdit",
    "QProcess",
    "QProcessEnvironment",
    "QProgressBar",
    "QPropertyAnimation",
    "QPushButton",
    "QScrollArea",
    "QSettings",
    "QSize",
    "QSizePolicy",
    "QSpinBox",
    "QSplitter",
    "QStackedWidget",
    "QSyntaxHighlighter",
    "QSystemTrayIcon",
    "QT_BINDING_NAME",
    "QTabWidget",
    "QTextCharFormat",
    "QTextDocument",
    "QTimer",
    "QToolButton",
    "QUrl",
    "QVBoxLayout",
    "QWidget",
    "Qt",
    "SCRIPT_DIR",
    "THEME",
    "TRAIN_CLI_MODULE",
    "apply_premium_dark_theme",
    "build_app_stylesheet",
    "pyqtSignal",
    "pyqtgraph_matches_qt",
    "with_alpha",
]
