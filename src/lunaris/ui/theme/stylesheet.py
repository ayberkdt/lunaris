# -*- coding: utf-8 -*-
"""
Global application stylesheet builder for the Lunar Graphite theme.

The interface aims for a calm, professional, engineering-oriented dark theme:
flat surfaces, subtle borders, restrained hover states, and a single primary
accent (orbital blue).  Gradients are deliberately reserved for the primary
"Run" action and the progress-bar chunk; every other surface is flat.

All colors are routed through the ``THEME`` / ``LOG_COLORS`` palettes so there
are no page-local hard-coded values here.
"""

from __future__ import annotations

from typing import Dict

from lunaris.ui.core.ui_commons import with_alpha


def build_app_stylesheet(theme: Dict[str, str], log_colors: Dict[str, str]) -> str:
    """Return the full application QSS string for the given palettes.

    Parameters
    ----------
    theme:
        The ``THEME`` token dictionary (Qt widget colors).
    log_colors:
        The ``LOG_COLORS`` token dictionary (used for the console default text).
    """

    # Derived translucent accent variants — computed once so the QSS below never
    # hard-codes a raw ``rgba(...)`` literal.
    acc = theme["accent"]
    acc_hover_border = theme["accent_deep"]   # calm, deep-blue hover edge
    acc_06 = with_alpha(acc, 0.06)            # faint nav hover wash
    acc_dim = theme["accent_dim"]             # ~0.12 tinted background
    acc_18 = with_alpha(acc, 0.18)
    acc_20 = with_alpha(acc, 0.20)
    acc_30 = with_alpha(acc, 0.30)
    acc_35 = with_alpha(acc, 0.35)
    acc_40 = with_alpha(acc, 0.40)

    # Semantic badge tints.
    info_bg, info_bd = with_alpha(theme["accent"], 0.12), with_alpha(theme["accent"], 0.35)
    ok_bg, ok_bd = with_alpha(theme["success"], 0.12), with_alpha(theme["success"], 0.32)
    err_bg, err_bd = with_alpha(theme["error"], 0.12), with_alpha(theme["error"], 0.30)
    warn_bg, warn_bd = with_alpha(theme["warning"], 0.12), with_alpha(theme["warning"], 0.32)

    return f"""
        /* GLOBAL FOUNDATION — flat space-black canvas */
        QMainWindow,
        QWidget#centralRoot {{
            background: {theme['bg_space']};
            color: {theme['fg_main']};
        }}
        QWidget {{
            background: transparent;
            color: {theme['fg_main']};
            font-family: "Segoe UI", "Inter", "Noto Sans", sans-serif;
            font-size: 10pt;
        }}
        QLabel {{
            background: transparent;
        }}
        QWidget#contentRoot,
        QStackedWidget#pages,
        QScrollArea,
        QScrollArea > QWidget > QWidget {{
            background: transparent;
            border: none;
        }}
        QToolTip {{
            background: {theme['bg_card_alt']};
            color: {theme['fg_main']};
            border: 1px solid {theme['border']};
            padding: 6px 8px;
        }}

        /* MENUS */
        QMenuBar {{
            background: {theme['bg_shell']};
            color: {theme['fg_main']};
            border-bottom: 1px solid {theme['border_soft']};
            padding: 4px 8px;
        }}
        QMenuBar::item {{
            background: transparent;
            padding: 6px 12px;
            border-radius: 7px;
        }}
        QMenuBar::item:selected {{
            background: {acc_dim};
            color: {theme['fg_soft']};
        }}
        QMenu {{
            background: {theme['bg_card']};
            color: {theme['fg_main']};
            border: 1px solid {theme['border']};
            padding: 6px;
        }}
        QMenu::item {{
            padding: 7px 22px 7px 12px;
            border-radius: 7px;
        }}
        QMenu::item:selected {{
            background: {acc_dim};
            color: {theme['fg_soft']};
        }}
        QMenu::separator {{
            height: 1px;
            background: {theme['border_soft']};
            margin: 6px 8px;
        }}

        /* CONTAINERS — flat shell surfaces, no gradients */
        QFrame#header, QFrame#logHeader {{
            background: {theme['bg_shell']};
            border: 1px solid {theme['border_soft']};
            border-radius: 12px;
        }}
        QFrame#stateFrame {{
            background: {acc_dim};
            border: 1px solid {acc_20};
            border-radius: 8px;
            padding: 2px 8px;
        }}

        /* TEXT */
        QLabel#title {{
            font-size: 15pt;
            font-weight: 700;
            color: {theme['fg_soft']};
            letter-spacing: 0.3px;
        }}
        QLabel#runState {{
            color: {theme['fg_soft']};
            font-weight: 600;
        }}
        QLabel#progressText {{
            color: {theme['fg_muted']};
            font-size: 9pt;
        }}

        /* NAVIGATION — flat shell; active = left border + tinted background */
        QListWidget#navDrawer {{
            background: {theme['bg_shell']};
            border: 1px solid {theme['border_soft']};
            border-radius: 12px;
            padding: 10px;
            outline: none;
        }}
        QListWidget#navDrawer::item {{
            background: transparent;
            padding: 11px 14px;
            margin-bottom: 6px;
            border-radius: 10px;
            color: {theme['fg_muted']};
        }}
        QListWidget#navDrawer::item:hover {{
            background: {acc_06};
            color: {theme['fg_main']};
        }}
        QListWidget#navDrawer::item:selected {{
            background: {acc_dim};
            color: {theme['fg_main']};
            font-weight: 700;
            border-left: 3px solid {theme['accent']};
        }}

        /* INPUTS — clean, legible, subtle states */
        QLineEdit, QPlainTextEdit, QComboBox, QDoubleSpinBox, QDateTimeEdit, QSpinBox {{
            background: {theme['bg_entry']};
            color: {theme['fg_main']};
            border: 1px solid {theme['border']};
            border-radius: 8px;
            padding: 7px 10px;
            selection-background-color: {theme['accent']};
            selection-color: {theme['bg_space']};
        }}
        QLineEdit:hover, QComboBox:hover, QPlainTextEdit:hover,
        QDoubleSpinBox:hover, QDateTimeEdit:hover, QSpinBox:hover {{
            border: 1px solid {acc_hover_border};
        }}
        QLineEdit:focus, QComboBox:focus, QPlainTextEdit:focus,
        QDoubleSpinBox:focus, QDateTimeEdit:focus, QSpinBox:focus {{
            border: 1px solid {theme['accent']};
            background: {theme['bg_card_alt']};
        }}
        QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled,
        QDoubleSpinBox:disabled, QDateTimeEdit:disabled {{
            color: {theme['text_disabled']};
            background: {theme['bg_card']};
            border-color: {theme['border_soft']};
        }}
        QComboBox::drop-down,
        QDateTimeEdit::drop-down {{
            border: none;
            width: 24px;
        }}
        QComboBox QAbstractItemView {{
            background: {theme['bg_card_alt']};
            color: {theme['fg_main']};
            border: 1px solid {theme['border']};
            selection-background-color: {acc_dim};
        }}

        /* CARDS — subtle border + surface contrast */
        QGroupBox {{
            background: {theme['bg_card']};
            border: 1px solid {theme['border_soft']};
            border-radius: 12px;
            margin-top: 24px;
            font-weight: 600;
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 0 8px;
            margin-left: 12px;
            color: {theme['fg_main']};
            font-size: 10.4pt;
            font-weight: 700;
        }}

        /* BUTTONS (default / secondary) — flat */
        QPushButton {{
            background: {theme['bg_card_alt']};
            color: {theme['fg_main']};
            border: 1px solid {theme['border']};
            border-radius: 8px;
            padding: 7px 16px;
            font-weight: 600;
        }}
        QPushButton:hover {{
            border-color: {acc_hover_border};
            background: {theme['bg_entry']};
            color: {theme['fg_main']};
        }}
        QPushButton:pressed {{
            background: {theme['border_soft']};
        }}
        QPushButton:disabled {{
            background: {theme['bg_card']};
            border-color: {theme['border_soft']};
            color: {theme['text_disabled']};
        }}
        QPushButton#quickChip {{
            background: {acc_dim};
            border: 1px solid {acc_20};
            color: {theme['fg_soft']};
            border-radius: 8px;
            padding: 5px 12px;
        }}
        QPushButton#quickChip:hover {{
            background: {acc_18};
            border-color: {theme['accent']};
        }}

        /* PRIMARY BUTTON (RUN) — the one place a gradient is welcome */
        QPushButton#primaryBtn {{
            background: qlineargradient(
                x1: 0, y1: 0, x2: 1, y2: 0,
                stop: 0 {theme['accent']},
                stop: 1 {theme['secondary']}
            );
            border: 1px solid {theme['accent']};
            color: {theme['bg_space']};
            font-weight: 700;
        }}
        QPushButton#primaryBtn:hover {{
            background: {theme['accent_hov']};
            border-color: {theme['accent_hov']};
            color: {theme['bg_space']};
        }}
        QPushButton#primaryBtn:disabled {{
            background: {theme['bg_entry']};
            border-color: {theme['border']};
            color: {theme['text_disabled']};
        }}

        /* DANGER BUTTON (STOP) */
        QPushButton#dangerBtn {{
            background: {with_alpha(theme['error'], 0.10)};
            border: 1px solid {with_alpha(theme['error'], 0.26)};
            color: {theme['fg_main']};
        }}
        QPushButton#dangerBtn:hover {{
            background: {with_alpha(theme['error'], 0.20)};
            border-color: {theme['error']};
        }}
        QPushButton#dangerBtn:disabled {{
            background: {theme['bg_entry']};
            border-color: {theme['border']};
            color: {theme['text_disabled']};
        }}

        /* TOOL + CHECK CONTROLS */
        QToolButton {{
            background: transparent;
            border: none;
            padding: 4px;
        }}
        QToolButton:hover {{
            background: {acc_dim};
            border-radius: 7px;
        }}
        QCheckBox {{
            spacing: 8px;
            color: {theme['fg_muted']};
        }}
        QCheckBox::indicator {{
            width: 14px;
            height: 14px;
            border-radius: 4px;
            border: 1px solid {theme['border']};
            background: {theme['bg_entry']};
        }}
        QCheckBox::indicator:checked {{
            background: {theme['accent']};
            border-color: {theme['accent']};
        }}
        QCheckBox::indicator:hover {{
            border-color: {acc_40};
        }}

        /* STATUS BADGE */
        QLabel#statusBadge {{
            border-radius: 10px;
            border: 1px solid {theme['border']};
            font-weight: 700;
            padding: 0 8px;
        }}
        QLabel#statusBadge[kind="info"] {{ background: {info_bg}; color: {theme['accent_hov']}; border-color: {info_bd}; }}
        QLabel#statusBadge[kind="success"] {{ background: {ok_bg}; color: {theme['success']}; border-color: {ok_bd}; }}
        QLabel#statusBadge[kind="error"] {{ background: {err_bg}; color: {theme['error']}; border-color: {err_bd}; }}
        QLabel#statusBadge[kind="warning"] {{ background: {warn_bg}; color: {theme['warning']}; border-color: {warn_bd}; }}

        /* RUN DOT */
        QFrame#runDot {{ border-radius: 6px; }}
        QFrame#runDot[kind="idle"] {{ background: {theme['fg_muted']}; }}
        QFrame#runDot[kind="running"] {{ background: {theme['success']}; }}
        QFrame#runDot[kind="error"] {{ background: {theme['error']}; }}
        QFrame#runDot[kind="warning"] {{ background: {theme['warning']}; }}

        /* TABS */
        QTabWidget::pane {{
            border: 1px solid {theme['border_soft']};
            background: {theme['bg_card']};
            border-radius: 12px;
            top: -1px;
        }}
        QTabBar::tab {{
            background: {theme['bg_card_alt']};
            color: {theme['fg_muted']};
            border: 1px solid {theme['border_soft']};
            padding: 8px 16px;
            margin-right: 6px;
            border-top-left-radius: 10px;
            border-top-right-radius: 10px;
        }}
        QTabBar::tab:selected {{
            background: {acc_dim};
            color: {theme['fg_soft']};
            border-color: {acc_30};
            border-bottom: 2px solid {theme['accent']};
        }}
        QTabBar::tab:hover:!selected {{
            color: {theme['fg_main']};
            background: {theme['bg_entry']};
        }}

        /* LOGGING */
        QTextEdit#log {{
            background: {theme['bg_log']};
            color: {log_colors['default']};
            font-family: "Consolas", "Courier New", monospace;
            font-size: 10pt;
            border: 1px solid {theme['border_soft']};
            border-radius: 10px;
            padding: 8px;
        }}

        /* SCROLLBARS */
        QScrollBar:vertical, QScrollBar:horizontal {{
            background: transparent;
            width: 8px;
            height: 8px;
            margin: 0;
        }}
        QScrollBar::handle:vertical, QScrollBar::handle:horizontal {{
            background: {theme['border']};
            border-radius: 4px;
            min-height: 20px;
            min-width: 20px;
        }}
        QScrollBar::handle:vertical:hover, QScrollBar::handle:horizontal:hover {{
            background: {acc_40};
        }}
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ height: 0px; width: 0px; }}

        /* SPLITTER */
        QSplitter::handle {{
            background: {theme['border_soft']};
        }}
        QSplitter::handle:vertical {{
            height: 8px;
            margin: 0 4px;
            border-radius: 3px;
            image: none;
        }}
        QSplitter::handle:vertical:hover {{
            background: {acc_35};
        }}
        QSplitter#mainSplit::handle:vertical {{
            background: {theme['border']};
            height: 8px;
            border-radius: 3px;
        }}
        QSplitter#mainSplit::handle:vertical:hover {{
            background: {acc_40};
        }}

        /* TREE / LIST WIDGETS */
        QTreeWidget, QListWidget {{
            background: {theme['bg_entry']};
            color: {theme['fg_main']};
            border: 1px solid {theme['border']};
            border-radius: 10px;
            outline: none;
            alternate-background-color: {theme['bg_card_alt']};
        }}
        QTreeWidget::item, QListWidget::item {{
            padding: 4px 6px;
            border-radius: 5px;
        }}
        QTreeWidget::item:selected, QListWidget::item:selected {{
            background: {acc_dim};
            color: {theme['fg_main']};
        }}
        QTreeWidget::item:hover, QListWidget::item:hover {{
            background: {acc_06};
        }}
        QHeaderView::section {{
            background: {theme['bg_card']};
            color: {theme['fg_soft']};
            border: none;
            border-bottom: 1px solid {theme['border']};
            padding: 5px 8px;
            font-weight: 600;
        }}
    """
