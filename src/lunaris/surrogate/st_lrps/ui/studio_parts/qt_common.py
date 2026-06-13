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

from lunaris.ui.core.ui_commons import THEME, with_alpha

def apply_premium_dark_theme(app: QApplication) -> None:
    app.setStyle("Fusion")
    app.setFont(QFont("Segoe UI", 10))
    pal = QPalette()
    pal.setColor(QPalette.ColorRole.Window, QColor(THEME["bg_shell"]))
    pal.setColor(QPalette.ColorRole.WindowText, QColor(THEME["fg_main"]))
    pal.setColor(QPalette.ColorRole.Base, QColor(THEME["bg_space"]))
    pal.setColor(QPalette.ColorRole.AlternateBase, QColor(THEME["bg_card"]))
    pal.setColor(QPalette.ColorRole.Text, QColor(THEME["fg_main"]))
    pal.setColor(QPalette.ColorRole.Button, QColor(THEME["bg_card_alt"]))
    pal.setColor(QPalette.ColorRole.ButtonText, QColor(THEME["fg_main"]))
    pal.setColor(QPalette.ColorRole.ToolTipBase, QColor(THEME["bg_card"]))
    pal.setColor(QPalette.ColorRole.ToolTipText, QColor(THEME["fg_main"]))
    pal.setColor(QPalette.ColorRole.Highlight, QColor(THEME["accent"]))
    pal.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
    pal.setColor(QPalette.ColorRole.Link, QColor(THEME["accent"]))
    app.setPalette(pal)

    app.setStyleSheet(f"""
        QWidget {{ font-family: 'Segoe UI', Arial, sans-serif; font-size: 13px; color: {THEME["fg_main"]}; }}
        QMainWindow, QWidget {{ background: {THEME["bg_space"]}; }}
        QToolTip {{
            background-color: {THEME["bg_card"]}; color: {THEME["fg_main"]};
            border: 1px solid {with_alpha(THEME["accent"], 0.35)};
            border-radius: 8px; padding: 8px 10px; font-size: 12px;
        }}
        QGroupBox {{
            background-color: {THEME["bg_card"]};
            border: 1px solid {THEME["border"]};
            border-radius: 12px; margin-top: 18px; padding-top: 12px;
        }}
        QGroupBox::title {{
            subcontrol-origin: margin; left: 14px; padding: 2px 10px;
            color: {THEME["fg_soft"]}; font-weight: 750; font-size: 12px;
            background-color: {THEME["bg_space"]};
            border: none;
        }}
        QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
            background-color: {THEME["bg_entry"]};
            border: 1px solid {THEME["border"]};
            border-radius: 10px; padding: 0px 12px;
            min-height: 38px; selection-background-color: {THEME["accent"]};
        }}
        QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
            border: 1px solid {THEME["accent"]};
        }}
        QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {{
            color: {THEME["fg_muted"]}; background-color: {THEME["bg_card"]};
        }}
        QSpinBox, QDoubleSpinBox {{ padding-right: 40px; }}
        QAbstractSpinBox::up-button, QAbstractSpinBox::down-button {{
            subcontrol-origin: border; width: 30px;
            background: {THEME["bg_card_alt"]};
            border-left: 1px solid {THEME["border"]};
        }}
        QAbstractSpinBox::up-button {{ subcontrol-position: top right; border-top-right-radius: 10px; }}
        QAbstractSpinBox::down-button {{ subcontrol-position: bottom right; border-bottom-right-radius: 10px; }}
        QPlainTextEdit {{
            background-color: {THEME["bg_log"]};
            border: 1px solid {THEME["border"]};
            border-radius: 10px; padding: 10px 12px;
            selection-background-color: {THEME["accent"]};
        }}
        QTabWidget::pane {{
            border: 1px solid {THEME["border"]};
            border-radius: 14px; background-color: {THEME["bg_card"]}; top: -1px;
        }}
        QTabBar {{
            alignment: left;
        }}
        QTabBar::tab {{
            background: {THEME["bg_shell"]};
            border: 1px solid {THEME["border"]}; border-bottom: none;
            padding: 9px 18px; margin-right: 4px;
            border-top-left-radius: 12px; border-top-right-radius: 12px;
            color: {THEME["fg_muted"]}; font-weight: 500; font-size: 13px;
            min-width: 80px; max-width: 240px;
        }}
        QTabBar::tab:selected {{
            background: {THEME["bg_card"]};
            border-color: {with_alpha(THEME["accent"], 0.38)};
            border-top: 2px solid {THEME["accent"]};
            color: {THEME["fg_main"]}; font-weight: 600;
        }}
        QTabBar::tab:hover:!selected {{ color: {THEME["fg_soft"]}; background: {THEME["bg_card_alt"]}; }}
        QTabBar::scroller {{ width: 24px; }}
        QTabBar QToolButton {{
            background: {THEME["bg_shell"]};
            border: 1px solid {THEME["border"]};
            border-radius: 6px;
        }}
        QTabBar QToolButton:hover {{ background: {THEME["bg_card_alt"]}; }}
        QProgressBar {{
            background-color: {THEME["bg_entry"]};
            border: 1px solid {THEME["border"]};
            border-radius: 9px; height: 18px; text-align: center; font-size: 11px;
        }}
        QProgressBar::chunk {{
            background: {THEME["accent"]};
            border-radius: 9px;
        }}
        QPushButton {{
            border-radius: 10px; padding: 8px 16px;
            border: 1px solid {THEME["border"]};
            background-color: {THEME["bg_card_alt"]}; font-weight: 500;
        }}
        QPushButton:hover {{ background-color: {THEME["bg_shell"]}; }}
        QPushButton:pressed {{ background-color: {THEME["bg_space"]}; }}
        QPushButton:disabled {{ color: {THEME["fg_muted"]}; background-color: {THEME["bg_card"]}; }}
        QPushButton[kind="primary"] {{
            border: 1px solid {THEME["accent"]};
            background: {with_alpha(THEME["accent"], 0.18)};
            color: {THEME["fg_main"]}; font-weight: 700;
        }}
        QPushButton[kind="primary"]:hover {{
            background: {with_alpha(THEME["accent"], 0.26)};
            border-color: {THEME["accent"]};
        }}
        QPushButton[kind="danger"] {{
            border: 1px solid {THEME["error"]};
            background-color: {with_alpha(THEME["error"], 0.14)};
            color: {THEME["error"]};
        }}
        QPushButton[kind="danger"]:hover {{
            background-color: {with_alpha(THEME["error"], 0.26)};
            border-color: {THEME["error"]};
        }}
        QPushButton[kind="ghost"] {{
            background-color: transparent;
            border-color: transparent;
            color: {THEME["fg_muted"]};
        }}
        QPushButton[kind="ghost"]:hover {{
            background-color: {THEME["bg_card_alt"]};
            color: {THEME["fg_main"]};
            border-color: {THEME["border"]};
        }}
        QCheckBox {{ spacing: 10px; }}
        QCheckBox::indicator {{
            width: 17px; height: 17px; border-radius: 5px;
            border: 1px solid {THEME["border"]};
            background: {THEME["bg_entry"]};
        }}
        QCheckBox::indicator:hover {{ border-color: {with_alpha(THEME["accent"], 0.55)}; }}
        QCheckBox::indicator:checked {{
            background: {with_alpha(THEME["accent"], 0.75)};
            border-color: {THEME["accent"]};
        }}
        QCheckBox:disabled {{ color: {THEME["fg_muted"]}; }}
        QLabel {{ color: {THEME["fg_soft"]}; font-size: 12px; }}
        QScrollBar:vertical {{ background: transparent; width: 10px; }}
        QScrollBar::handle:vertical {{ background: {THEME["border"]}; min-height: 28px; border-radius: 5px; }}
        QScrollBar::handle:vertical:hover {{ background: {THEME["border_soft"]}; }}
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0px; }}
        QScrollBar:horizontal {{ background: transparent; height: 10px; }}
        QScrollBar::handle:horizontal {{ background: {THEME["border"]}; min-width: 28px; border-radius: 5px; }}
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0px; }}
        QSplitter::handle {{ background: {with_alpha(THEME["border"], 0.5)}; }}
        QSplitter::handle:horizontal {{ width: 5px; }}
        QSplitter::handle:vertical   {{ height: 5px; }}
        QSplitter::handle:hover      {{ background: {with_alpha(THEME["accent"], 0.18)}; }}
        QListWidget {{
            background-color: {THEME["bg_entry"]};
            border: 1px solid {THEME["border"]};
            border-radius: 12px; padding: 6px; font-size: 12px;
        }}
        QListWidget::item {{ padding: 7px 10px; border-radius: 7px; }}
        QListWidget::item:selected {{
            background-color: {with_alpha(THEME["accent"], 0.18)}; color: {THEME["fg_main"]};
        }}
        QListWidget::item:hover:!selected {{ background-color: {with_alpha(THEME["accent"], 0.08)}; }}
        QStatusBar {{
            background: {THEME["bg_shell"]};
            border-top: 1px solid {THEME["border"]};
            color: {THEME["fg_muted"]}; font-size: 11px;
        }}
        QStatusBar::item {{ border: none; }}
        QFrame#navSidebar QPushButton {{
            border-radius: 0;
            border-left: 3px solid transparent;
        }}
        QInputDialog {{ background-color: {THEME["bg_card"]}; }}
    """)




TRAIN_CLI_MODULE = "lunaris.surrogate.st_lrps.training.cli"
PROFILE_CLI_MODULE = "lunaris.surrogate.st_lrps.runtime.profiling"
