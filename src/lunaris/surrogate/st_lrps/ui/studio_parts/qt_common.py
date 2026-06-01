import os
import sys
_USE_PYSIDE = "PyQt6" not in sys.modules

try:
    if _USE_PYSIDE:
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
    else:
        raise ImportError
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

QT_BINDING_NAME = "PySide6" if _USE_PYSIDE else "PyQt6"
if "pyqtgraph" not in sys.modules:
    os.environ.setdefault("PYQTGRAPH_QT_LIB", QT_BINDING_NAME)


def pyqtgraph_matches_qt(pg_module) -> bool:
    """Return False when pyqtgraph was imported against a different Qt binding."""
    qt_lib = getattr(getattr(pg_module, "Qt", None), "QT_LIB", None)
    return qt_lib in (None, QT_BINDING_NAME)


class NoScrollComboBox(QComboBox):
    def wheelEvent(self, e):
        e.ignore()

from pathlib import Path
SCRIPT_DIR = Path(__file__).resolve().parent

def apply_premium_dark_theme(app: QApplication) -> None:
    app.setStyle("Fusion")
    app.setFont(QFont("Segoe UI", 10))
    pal = QPalette()
    pal.setColor(QPalette.ColorRole.Window, QColor("#0b1020"))
    pal.setColor(QPalette.ColorRole.WindowText, QColor("#e8ecf8"))
    pal.setColor(QPalette.ColorRole.Base, QColor("#070b14"))
    pal.setColor(QPalette.ColorRole.AlternateBase, QColor("#0f1830"))
    pal.setColor(QPalette.ColorRole.Text, QColor("#e8ecf8"))
    pal.setColor(QPalette.ColorRole.Button, QColor("#121a33"))
    pal.setColor(QPalette.ColorRole.ButtonText, QColor("#e8ecf8"))
    pal.setColor(QPalette.ColorRole.ToolTipBase, QColor("#141e3a"))
    pal.setColor(QPalette.ColorRole.ToolTipText, QColor("#e8ecf8"))
    pal.setColor(QPalette.ColorRole.Highlight, QColor("#35d0ff"))
    pal.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
    pal.setColor(QPalette.ColorRole.Link, QColor("#35d0ff"))
    app.setPalette(pal)

    app.setStyleSheet("""
        QWidget { font-family: 'Segoe UI', Arial, sans-serif; font-size: 13px; color: #e8ecf8; }
        QMainWindow, QWidget {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #0b1020, stop:1 #070a12);
        }
        QToolTip {
            background-color: #141e3a; color: #e8ecf8;
            border: 1px solid rgba(53, 208, 255, 0.35);
            border-radius: 8px; padding: 8px 10px; font-size: 12px;
        }
        QGroupBox {
            background-color: rgba(12, 18, 34, 0.62);
            border: 1px solid rgba(185, 194, 221, 0.12);
            border-radius: 12px; margin-top: 18px; padding-top: 12px;
        }
        QGroupBox::title {
            subcontrol-origin: margin; left: 14px; padding: 2px 10px;
            color: #d8e1f7; font-weight: 750; font-size: 12px;
            background-color: #0b1020;
            border: none;
        }
        QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
            background-color: rgba(7, 11, 20, 0.92);
            border: 1px solid rgba(185, 194, 221, 0.22);
            border-radius: 10px; padding: 0px 12px;
            min-height: 38px; selection-background-color: #35d0ff;
        }
        QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {
            border: 1px solid rgba(53, 208, 255, 0.75);
        }
        QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {
            color: rgba(185, 194, 221, 0.35); background-color: rgba(12, 16, 30, 0.6);
        }
        QSpinBox, QDoubleSpinBox { padding-right: 40px; }
        QAbstractSpinBox::up-button, QAbstractSpinBox::down-button {
            subcontrol-origin: border; width: 30px;
            background: rgba(18, 26, 51, 0.9);
            border-left: 1px solid rgba(185, 194, 221, 0.18);
        }
        QAbstractSpinBox::up-button { subcontrol-position: top right; border-top-right-radius: 10px; }
        QAbstractSpinBox::down-button { subcontrol-position: bottom right; border-bottom-right-radius: 10px; }
        QPlainTextEdit {
            background-color: rgba(7, 11, 20, 0.92);
            border: 1px solid rgba(185, 194, 221, 0.22);
            border-radius: 10px; padding: 10px 12px;
            selection-background-color: #35d0ff;
        }
        QTabWidget::pane {
            border: 1px solid rgba(185, 194, 221, 0.18);
            border-radius: 14px; background-color: rgba(15, 24, 48, 0.35); top: -1px;
        }
        QTabBar {
            alignment: left;
        }
        QTabBar::tab {
            background: rgba(16, 24, 48, 0.65);
            border: 1px solid rgba(185, 194, 221, 0.14); border-bottom: none;
            padding: 9px 18px; margin-right: 4px;
            border-top-left-radius: 12px; border-top-right-radius: 12px;
            color: #8892b0; font-weight: 500; font-size: 13px;
            min-width: 80px; max-width: 240px;
        }
        QTabBar::tab:selected {
            background: rgba(15, 24, 48, 0.95);
            border-color: rgba(53, 208, 255, 0.38);
            border-top: 2px solid rgba(53, 208, 255, 0.75);
            color: #e8ecf8; font-weight: 600;
        }
        QTabBar::tab:hover:!selected { color: #d7e1f7; background: rgba(20, 30, 58, 0.8); }
        QTabBar::scroller { width: 24px; }
        QTabBar QToolButton {
            background: rgba(16, 24, 48, 0.8);
            border: 1px solid rgba(185, 194, 221, 0.18);
            border-radius: 6px;
        }
        QTabBar QToolButton:hover { background: rgba(26, 36, 70, 0.95); }
        QProgressBar {
            background-color: rgba(7, 11, 20, 0.92);
            border: 1px solid rgba(185, 194, 221, 0.22);
            border-radius: 9px; height: 18px; text-align: center; font-size: 11px;
        }
        QProgressBar::chunk {
            background: #35d0ff;
            border-radius: 9px;
        }
        QPushButton {
            border-radius: 10px; padding: 8px 16px;
            border: 1px solid rgba(185, 194, 221, 0.18);
            background-color: rgba(18, 26, 51, 0.9); font-weight: 500;
        }
        QPushButton:hover { background-color: rgba(26, 36, 70, 0.95); }
        QPushButton:pressed { background-color: rgba(14, 20, 40, 0.95); }
        QPushButton:disabled { color: rgba(232, 236, 248, 0.35); background-color: rgba(16, 24, 48, 0.35); }
        QPushButton[kind="primary"] {
            border: 1px solid rgba(53, 208, 255, 0.48);
            background: rgba(53, 208, 255, 0.18);
            color: #effbff; font-weight: 700;
        }
        QPushButton[kind="primary"]:hover {
            background: rgba(53, 208, 255, 0.26);
            border-color: rgba(53, 208, 255, 0.72);
        }
        QPushButton[kind="danger"] {
            border: 1px solid rgba(248, 113, 113, 0.50);
            background-color: rgba(248, 113, 113, 0.14);
            color: #fca5a5;
        }
        QPushButton[kind="danger"]:hover {
            background-color: rgba(248, 113, 113, 0.26);
            border-color: rgba(248, 113, 113, 0.70);
        }
        QPushButton[kind="ghost"] {
            background-color: rgba(16, 24, 48, 0.30);
            border-color: rgba(185, 194, 221, 0.12);
            color: #9aa7c7;
        }
        QPushButton[kind="ghost"]:hover {
            background-color: rgba(26, 36, 70, 0.55);
            color: #d7e1f7;
            border-color: rgba(185, 194, 221, 0.22);
        }
        QCheckBox { spacing: 10px; }
        QCheckBox::indicator {
            width: 17px; height: 17px; border-radius: 5px;
            border: 1px solid rgba(185, 194, 221, 0.22);
            background: rgba(7, 11, 20, 0.92);
        }
        QCheckBox::indicator:hover { border-color: rgba(53, 208, 255, 0.55); }
        QCheckBox::indicator:checked {
            background: rgba(53, 208, 255, 0.75);
            border-color: rgba(53, 208, 255, 0.92);
        }
        QCheckBox:disabled { color: rgba(185, 194, 221, 0.35); }
        QLabel { color: #b9c2dd; font-size: 12px; }
        QScrollBar:vertical { background: transparent; width: 10px; }
        QScrollBar::handle:vertical { background: rgba(185, 194, 221, 0.2); min-height: 28px; border-radius: 5px; }
        QScrollBar::handle:vertical:hover { background: rgba(185, 194, 221, 0.35); }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
        QScrollBar:horizontal { background: transparent; height: 10px; }
        QScrollBar::handle:horizontal { background: rgba(185, 194, 221, 0.2); min-width: 28px; border-radius: 5px; }
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0px; }
        QSplitter::handle { background: rgba(185, 194, 221, 0.07); }
        QSplitter::handle:horizontal { width: 5px; }
        QSplitter::handle:vertical   { height: 5px; }
        QSplitter::handle:hover      { background: rgba(53, 208, 255, 0.18); }
        QListWidget {
            background-color: rgba(7, 11, 20, 0.92);
            border: 1px solid rgba(185, 194, 221, 0.18);
            border-radius: 12px; padding: 6px; font-size: 12px;
        }
        QListWidget::item { padding: 7px 10px; border-radius: 7px; }
        QListWidget::item:selected {
            background-color: rgba(53, 208, 255, 0.18); color: #ffffff;
        }
        QListWidget::item:hover:!selected { background-color: rgba(53, 208, 255, 0.08); }
        QStatusBar {
            background: rgba(7, 11, 20, 0.95);
            border-top: 1px solid rgba(185, 194, 221, 0.10);
            color: #6f7ca8; font-size: 11px;
        }
        QStatusBar::item { border: none; }
        QFrame#navSidebar QPushButton {
            border-radius: 0;
            border-left: 3px solid transparent;
        }
        QInputDialog { background-color: #0f1830; }
    """)




TRAIN_CLI_MODULE = "lunaris.surrogate.st_lrps.training.cli"
PROFILE_CLI_MODULE = "lunaris.surrogate.st_lrps.runtime.profiling"
