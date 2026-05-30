# ST_LRPS/ui_parts/ui_commons.py

"""
Core UI Utilities and Shared Resources for ST-LRPS Studio.

This module serves as the foundational layer for the user interface, providing
centralized access to:

1. Global Constants: Application-wide paths, physics constants (e.g., R_MOON_KM),
   and visual theme definitions (color palettes, window settings).
   
2. Utility Functions: Robust helpers for asset loading (fonts, icons), path 
   normalization, and project root detection.

3. Custom UI Primitives: Reusable, stylized PySide6 widgets (e.g., 
   NumericDragLineEdit, ToggleSwitch, StatusBadge) designed to maintain 
   visual consistency and interactivity across all application pages.

Dependencies:
    - PySide6 (Core UI)
    - qtawesome (Optional: for vector icons)
"""


# =============================================================================
# 0.                                    IMPORTS 
# =============================================================================
from __future__ import annotations

import os
import math
from pathlib import Path
from typing import Optional

from PySide6 import QtGui, QtCore, QtWidgets

# Derive lunar constants from the backend SSOT (common.constants).
# UI code works in km, so we convert here once and export aliases.
# Fallback literals keep the UI loadable without the backend on PYTHONPATH.
try:
    from lunaris.common.constants import R_MOON_MEAN as _R_MOON_MEAN_M, MU_MOON as _MU_MOON_SI
    R_MOON_KM: float = _R_MOON_MEAN_M / 1000.0       # 1737.4 km
    MU_MOON_KM3_S2: float = _MU_MOON_SI / 1e9         # ~4902.87 km³/s²
except ImportError:
    R_MOON_KM = 1737.4
    MU_MOON_KM3_S2 = 4902.8695

# Modern Icon Library
try:
    import qtawesome as qta
    HAS_QTAWESOME = True
except ImportError:
    HAS_QTAWESOME = False
    print("[Warning] qtawesome not installed. Icons will be disabled.")




# =============================================================================
# 1.                            UI CONFIGURATION
# =============================================================================

# Application metadata is centralized here so wrapper entry points and saved
# session metadata still share one authoritative version value, while the live
# UI can choose whether or not to display it.
APP_NAME = "ST-LRPS Studio"
APP_VERSION = "13.0"


# Theme configuration using a dictionary for easier QSS (Qt Style Sheet) integration
# Lunar Aurora palette — deep-space, mission-control aesthetic.
# Primary accent: ion cyan (#35D0FF).  Secondary highlight: restrained violet (#8B7CFF).
# No dominant gold/champagne.  All tokens below are the single source of truth.
THEME = {
    # ── Backgrounds ──────────────────────────────────────────────────────────
    "bg_space":    "#070B14",   # deepest canvas
    "bg_shell":    "#0B1220",   # header / sidebar shell
    "bg_card":     "#101A2B",   # primary cards
    "bg_card_alt": "#17243A",   # elevated cards / hover surfaces
    "bg_entry":    "#0D1626",   # input fields / selectors
    "bg_log":      "#050812",   # terminal / log panel

    # ── Foreground ────────────────────────────────────────────────────────────
    "fg_main":     "#E6EDF7",   # primary text
    "fg_soft":     "#BFD2EA",   # secondary headings
    "fg_muted":    "#7F91AC",   # helper / muted text

    # ── Primary accent (cyan / ion-blue) ─────────────────────────────────────
    "accent":      "#35D0FF",   # primary cyan — progress, active, primary CTAs
    "accent_hov":  "#7CE7FF",   # hover / brighter cyan
    "accent_dim":  "rgba(53,208,255,0.13)",  # subtle tinted background

    # ── Secondary accent (violet) ─────────────────────────────────────────────
    "secondary":       "#8B7CFF",   # selected state, secondary UI
    "secondary_hov":   "#B0A7FF",
    "secondary_dim":   "rgba(139,124,255,0.13)",

    # ── Semantic colors ───────────────────────────────────────────────────────
    "success":     "#2DD4BF",   # teal — success states
    "warning":     "#F6C177",   # amber — warnings (not used as dominant accent)
    "error":       "#FF6B7A",   # soft coral — error / danger
    "info":        "#60A5FA",   # sky blue — info chips

    # ── Borders ───────────────────────────────────────────────────────────────
    "border":      "#26364F",   # card / input borders
    "border_soft": "#1A2940",   # quieter separators

    # ── Semantic aliases (for callers that use these token names) ─────────────
    "primary":        "#35D0FF",   # → accent
    "primary_hover":  "#7CE7FF",   # → accent_hov
    "selected_bg":    "rgba(139,124,255,0.13)",   # → secondary_dim
    "panel_shadow":   "rgba(0,0,0,0.40)",
    "plot_bg":        "#050812",   # → bg_log
    "grid_color":     "rgba(38,54,79,0.55)",
    "text_disabled":  "rgba(127,145,172,0.45)",

    # ── Backward-compat keys (kept so older pages still resolve) ─────────────
    # accent_deep was used for darker gold edges; map to a deep cyan instead.
    "accent_deep": "#1A607A",
}

# Rich Text Log Colors (HTML) — aligned with Lunar Aurora palette
LOG_COLORS = {
    "error":     "#FF8B96",   # soft coral — readable on dark bg
    "warning":   "#F6C177",   # amber
    "success":   "#2DD4BF",   # teal
    "system":    "#BFD2EA",   # fg_soft
    "info":      "#7CE7FF",   # cyan highlight
    "debug":     "#7F91AC",   # fg_muted
    "timestamp": "#4A6080",   # dimmed
    "default":   "#E6EDF7",   # fg_main
}

# Window and Navigation Constants
WINDOW_SETTINGS = {
    "title": APP_NAME,
    "size": (1200, 900),
    "min_size": (1000, 840),
}



# =============================================================================
# 2.                          FONT LOADING
# =============================================================================

def find_project_root() -> Path:
    """
    Find the project root directory robustly.

    Strategy
    --------
    1) If env var STLRPS_PROJECT_ROOT is set and valid -> use it.
    2) Walk up from this file's directory, checking for common root markers.
    3) Fallback to the parent of this file.
    """
    # 1) Environment override
    env = os.environ.get("STLRPS_PROJECT_ROOT", "").strip()
    if env:
        p = Path(env).expanduser().resolve()
        if p.exists():
            return p

    # 2) Walk up
    markers = [
        "pyproject.toml",
        "setup.cfg",
        "requirements.txt",
        ".git",
        "data",  # project data folder
    ]

    start_dir = Path(__file__).resolve().parent
    current = start_dir

    # go deeper than 5; monorepo / nested app layouts break otherwise
    for _ in range(30):
        if any((current / m).exists() for m in markers):
            return current
        if current.parent == current:
            break
        current = current.parent

    return start_dir


PROJECT_ROOT = find_project_root()
DATA_DIR = PROJECT_ROOT / "data"
ASSETS_DIR = PROJECT_ROOT / "assets"
if not ASSETS_DIR.exists():
    # The current repository stores UI imagery under data/assets. Falling back
    # here keeps legacy code working while allowing a future top-level assets/
    # directory without further changes.
    fallback_assets_dir = DATA_DIR / "assets"
    if fallback_assets_dir.exists():
        ASSETS_DIR = fallback_assets_dir


def load_fonts() -> QtGui.QFont:
    """
    Load fonts from assets/fonts and return a preferred app font.
    """
    fonts_dir = ASSETS_DIR / "fonts"
    preferred = ["Segoe UI", "Inter", "Noto Sans", "Roboto", "DejaVu Sans", "Arial"]

    loaded_families = []
    if fonts_dir.exists():
        for pattern in ("*.ttf", "*.otf", "*.ttc"):
            for font_file in fonts_dir.glob(pattern):
                font_id = QtGui.QFontDatabase.addApplicationFont(str(font_file))
                if font_id != -1:
                    loaded_families.extend(QtGui.QFontDatabase.applicationFontFamilies(font_id))

    # Build a set of available families after loading
    available = set(QtGui.QFontDatabase.families())

    # 1) Choose preferred if present
    for fam in preferred:
        if fam in available:
            return QtGui.QFont(fam, 10)

    # 2) Otherwise choose first newly-loaded family (if any)
    for fam in loaded_families:
        if fam in available:
            return QtGui.QFont(fam, 10)

    # 3) Fallback
    return QtGui.QFont("Segoe UI", 10)



# =============================================================================
# 3.                          ICON UTILITIES
# =============================================================================

def get_icon(icon_name: str, color: Optional[str] = None) -> QtGui.QIcon:
    """
    Returns a FontAwesome icon using qtawesome.
    Falls back to a colored square if qtawesome is unavailable.
    """
    if not HAS_QTAWESOME:
        # Fallback: Create a simple colored square pixmap
        pixmap = QtGui.QPixmap(16, 16)
        pixmap.fill(QtGui.QColor(color or THEME['accent']))
        return QtGui.QIcon(pixmap)
    
    try:
        options = {'color': color or THEME['fg_main']}
        return qta.icon(icon_name, **options)
    except Exception as e:
        print(f"[Warning] Icon '{icon_name}' not found: {e}")
        # Return empty icon
        return QtGui.QIcon()


# =============================================================================
# 4.                          UTILITY HELPERS
# =============================================================================

def normalize_path(path_str: str) -> str:
    """Standardizes path formatting for the current OS."""
    if not path_str:
        return ""
    return str(Path(path_str).expanduser().resolve())


def is_valid_float(value: str) -> bool:
    """Returns True if the string can be cast to a float."""
    try:
        float(str(value))
        return True
    except (ValueError, TypeError):
        return False


def bool_to_onoff(value: bool) -> str:
    """Converts boolean to 'on'/'off' for CLI compatibility."""
    return "on" if value else "off"


def card_stylesheet() -> str:
    """Standard card GroupBox QSS used across all pages."""
    return f"""
        QGroupBox {{
            border: 1px solid {THEME['border']};
            border-radius: 10px;
            margin-top: 16px;
            padding-top: 8px;
            background: {THEME['bg_card']};
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            subcontrol-position: top left;
            left: 14px;
            padding: 0 8px;
            color: {THEME['fg_soft']};
            font-weight: 700;
            font-size: 10pt;
        }}
    """


def input_stylesheet() -> str:
    """Standard QLineEdit / QComboBox input field QSS."""
    return f"""
        QLineEdit, QComboBox {{
            background: {THEME['bg_entry']};
            color: {THEME['fg_main']};
            border: 1px solid {THEME['border']};
            border-radius: 6px;
            padding: 5px 9px;
            selection-background-color: {THEME['accent']};
            min-height: 28px;
        }}
        QLineEdit:focus, QComboBox:focus {{
            border: 1px solid {THEME['accent']};
        }}
        QLineEdit:hover, QComboBox:hover {{
            border: 1px solid {THEME['accent_hov']};
        }}
        QLineEdit:disabled, QComboBox:disabled {{
            color: {THEME['fg_muted']};
            background: {THEME['bg_card']};
        }}
        QComboBox::drop-down {{
            border: none;
            padding-right: 6px;
        }}
        QComboBox QAbstractItemView {{
            background: {THEME['bg_entry']};
            color: {THEME['fg_main']};
            border: 1px solid {THEME['border']};
            selection-background-color: {THEME['accent']};
        }}
    """


def section_label(text: str, parent=None) -> "QtWidgets.QLabel":
    """A styled section header label."""
    lbl = QtWidgets.QLabel(text, parent)
    lbl.setStyleSheet(
        f"color: {THEME['fg_soft']}; font-size: 10pt; font-weight: 700; "
        f"border-bottom: 1px solid {THEME['border_soft']}; padding-bottom: 4px;"
    )
    return lbl


def path_validity_badge(parent=None) -> "StatusBadge":
    """A StatusBadge pre-configured for path validation state."""
    badge = StatusBadge("NOT SET", kind="error", parent=parent)
    badge.setFixedWidth(90)
    return badge


# =============================================================================
# 5.                        CUSTOM UI PRIMITIVES
# =============================================================================

class StatusBadge(QtWidgets.QLabel):
    """
    A stylized label to show status (e.g., 'READY', 'RUNNING', 'ERROR').
    Colors are controlled via the 'kind' dynamic property in QSS.
    """
    def __init__(self, text: str = "WAITING", kind: str = "info", parent=None):
        super().__init__(text, parent)
        self.setObjectName("statusBadge")
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setContentsMargins(10, 4, 10, 4)
        self.setFixedHeight(24)
        self.set_status(kind, text)

    def set_status(self, kind: str, text: str):
        self.setProperty("kind", kind.lower())
        self.setText(text.upper())
        # Refresh styling
        self.style().unpolish(self)
        self.style().polish(self)


class QuickChip(QtWidgets.QPushButton):
    """Small clickable 'preset' buttons like [12h], [3 days] etc."""
    def __init__(self, text: str, parent=None):
        super().__init__(text, parent)
        self.setObjectName("quickChip")
        self.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.setFixedWidth(65)
        self.setFixedHeight(26)



# =============================================================================
# 6.                       ADVANCED UI CONTROLS
# =============================================================================

class NumericDragLineEdit(QtWidgets.QLineEdit):
    """
    Blender/Unity-style numeric input.
    - Click & Drag horizontally to change values.
    - Hold Ctrl for 0.1x steps, Shift for 10x steps.
    - Double-click to type manually.
    - Glow effect on focus/hover (styled via QSS).
    """
    
    value_changed = QtCore.Signal(float) # Custom signal for cleaner connection
    
    def __init__(self, value: float = 0.0, *, step: float = 1.0,
                 min_value: Optional[float] = None, max_value: Optional[float] = None,
                 decimals: int = 2, parent=None):
        super().__init__(parent)
        
        # Logic State
        try:
            if value is None:
                self._val = 0.0
            elif isinstance(value, str) and value.strip() == "":
                self._val = 0.0
            else:
                self._val = float(value)
        except Exception:
            self._val = 0.0

        self._step = float(step)
        self._min = min_value
        self._max = max_value
        self._decimals = int(decimals)
        
        # Drag State
        self._dragging = False
        self._drag_start_x = 0
        self._drag_start_val = 0.0
        
        # Setup
        self.setText(self._format(self._val))
        self.setMouseTracking(True) # Required for hover detection
        self.setObjectName("numericDrag")  # For QSS targeting
        self.setMinimumHeight(38)

        # Styling — uses Lunar Aurora cyan accent
        self.setStyleSheet(f"""
            QLineEdit#numericDrag {{
                background-color: {THEME['bg_entry']};
                color: {THEME['fg_main']};
                border: 1px solid {THEME['border']};
                border-radius: 9px;
                padding: 7px 10px;
                selection-background-color: {THEME['secondary']};
            }}
            QLineEdit#numericDrag:hover {{
                border: 1px solid rgba(53,208,255,0.45);
            }}
            QLineEdit#numericDrag:focus {{
                border: 1px solid {THEME['accent']};
                background-color: {THEME['bg_card_alt']};
            }}
        """)
    
    def _format(self, v: float) -> str:
        """
        Format the numeric value for display without hiding tiny tolerances.

        Fixed-point formatting works well for ordinary lengths, masses, and
        durations, but it turns solver tolerances like `1e-10` into `0` when the
        widget is configured with `decimals=0`. Switching to scientific notation
        for very small or very large magnitudes keeps the field honest while
        preserving the simple fixed-point look for everyday values.
        """

        val = float(v)
        if not math.isfinite(val):
            return "0"

        abs_val = abs(val)
        if abs_val != 0.0 and (abs_val < 1e-3 or abs_val >= 1e5):
            precision = max(0, int(self._decimals))
            return f"{val:.{precision}e}"

        return f"{val:.{self._decimals}f}"
    
    def _parse_and_set(self, text: str):
        try:
            val = float(text)
            # Clamp if needed
            if self._min is not None: val = max(self._min, val)
            if self._max is not None: val = min(self._max, val)
            
            self._val = val
            self.value_changed.emit(val)
        except ValueError:
            pass # Keep old value on invalid input
    
    def mousePressEvent(self, e: QtGui.QMouseEvent):
        if e.button() == QtCore.Qt.LeftButton:
            # Check if user clicked in the "drag zone" (right side or Alt key)
            is_right_edge = (self.width() - e.pos().x()) < 20
            is_alt = bool(e.modifiers() & QtCore.Qt.AltModifier)
            
            if is_right_edge or is_alt:
                self._dragging = True
                self._drag_start_x = int(e.globalPosition().x())
                self._parse_and_set(self.text()) # Sync state before drag
                self._drag_start_val = self._val
                
                self.setCursor(QtCore.Qt.SizeHorCursor)
                e.accept()
                return
                
        super().mousePressEvent(e)
    
    def mouseMoveEvent(self, e: QtGui.QMouseEvent):
        if self._dragging:
            # Calculate Delta
            dx = int(e.globalPosition().x()) - self._drag_start_x
            
            # Apply Modifiers
            multiplier = 1.0
            if e.modifiers() & QtCore.Qt.ControlModifier:
                multiplier = 0.1
            elif e.modifiers() & QtCore.Qt.ShiftModifier:
                multiplier = 10.0
            
            # Update Value
            new_val = self._drag_start_val + (dx * self._step * multiplier)
            
            # Clamp
            if self._min is not None: new_val = max(self._min, new_val)
            if self._max is not None: new_val = min(self._max, new_val)
            
            self._val = new_val
            
            # Update UI without triggering textEdited loop if needed
            self.setText(self._format(self._val))
            self.value_changed.emit(self._val)
            
            e.accept()
            return
            
        super().mouseMoveEvent(e)
    
    def mouseReleaseEvent(self, e: QtGui.QMouseEvent):
        if self._dragging:
            self._dragging = False
            self.unsetCursor()
            e.accept()
            return
        super().mouseReleaseEvent(e)
    
    def focusOutEvent(self, e: QtGui.QFocusEvent):
        # Validate text on finish
        self._parse_and_set(self.text())
        self.setText(self._format(self._val)) # Reformat to clean up
        super().focusOutEvent(e)


class ToggleSwitch(QtWidgets.QAbstractButton):
    """
    Modern On/Off Switch.
    Replaces QCheckBox with a mobile-style toggle.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setCheckable(True)
        self.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.setFixedSize(44, 24)
    
    def paintEvent(self, _):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing, True)
        
        is_on = self.isChecked()
        is_enabled = self.isEnabled()
        
        # Colors from THEME (cyan when on, dark entry when off)
        bg_color = QtGui.QColor(THEME['accent'] if is_on else THEME['bg_entry'])
        knob_color = QtGui.QColor(THEME['fg_main'])
        
        if not is_enabled:
            bg_color.setAlpha(100)
            knob_color.setAlpha(150)
        
        # Draw Track
        rect = self.rect()
        radius = rect.height() / 2
        
        p.setPen(QtCore.Qt.NoPen)
        p.setBrush(bg_color)
        p.drawRoundedRect(rect, radius, radius)
        
        # Draw Border (if off)
        if not is_on:
            p.setBrush(QtCore.Qt.NoBrush)
            p.setPen(QtGui.QPen(QtGui.QColor(THEME['border']), 1))
            p.drawRoundedRect(rect, radius, radius)
        
        # Draw Knob
        p.setPen(QtCore.Qt.NoPen)
        p.setBrush(knob_color)
        
        margin = 3
        knob_dia = rect.height() - (2 * margin)
        
        # Calculate Position (Left vs Right)
        if is_on:
            x_pos = rect.width() - margin - knob_dia
        else:
            x_pos = margin
            
        p.drawEllipse(QtCore.QRectF(x_pos, margin, knob_dia, knob_dia))


class CostIndicator(QtWidgets.QWidget):
    """
    Visual indicator of computational cost (Low / Medium / High).
    Used to warn users about heavy settings (e.g. 1000x1000 gravity).
    """
    _LEVELS = {"low": 1, "medium": 2, "high": 3}
    _COLORS = {"low": "success", "medium": "warning", "high": "error"} # Keys in THEME
    
    def __init__(self, level: str = "low", parent=None):
        super().__init__(parent)
        self._level = "low"
        self.set_level(level)
        self.setFixedSize(50, 14)
        self.setToolTip("Estimated CPU Load")
    
    def set_level(self, level: str):
        self._level = (level or "low").lower()
        self.update()
    
    def paintEvent(self, _):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing, True)
        
        active_bars = self._LEVELS.get(self._level, 1)
        color_key = self._COLORS.get(self._level, "info")
        active_color = QtGui.QColor(THEME.get(color_key, THEME['accent']))
        inactive_color = QtGui.QColor(THEME['bg_entry'])
        
        bar_width = 12
        bar_height = 8
        gap = 4
        y_pos = (self.height() - bar_height) / 2
        
        for i in range(3):
            x_pos = i * (bar_width + gap)

            if i < active_bars:
                p.setBrush(active_color)
            else:
                p.setBrush(inactive_color)

            p.setPen(QtCore.Qt.NoPen)
            p.drawRoundedRect(QtCore.QRectF(x_pos, y_pos, bar_width, bar_height), 2, 2)


# =============================================================================
# 7.                       FACTORY HELPERS (theming aids)
# =============================================================================

def create_metric_card(parent: Optional["QtWidgets.QWidget"] = None) -> "tuple[QtWidgets.QGroupBox, QtWidgets.QGridLayout]":
    """
    Return a ``(card, grid)`` tuple for rendering compact key/value metrics.

    The grid is intended for label-value pairs arranged into two columns. The
    card itself uses the project-wide dark-card visual language.
    """

    card = QtWidgets.QGroupBox(parent)
    card.setStyleSheet(card_stylesheet())
    grid = QtWidgets.QGridLayout(card)
    grid.setContentsMargins(16, 22, 16, 16)
    grid.setHorizontalSpacing(18)
    grid.setVerticalSpacing(6)
    return card, grid


def create_empty_state(message: str, parent: Optional["QtWidgets.QWidget"] = None) -> "QtWidgets.QLabel":
    """
    Return a centered, muted label suitable for empty-state placeholders.

    Empty states should remain unobtrusive so the page never feels broken when
    the underlying data source is simply not populated yet.
    """

    lbl = QtWidgets.QLabel(message, parent)
    lbl.setAlignment(QtCore.Qt.AlignCenter)
    lbl.setWordWrap(True)
    lbl.setStyleSheet(
        f"color: {THEME['fg_muted']}; font-style: italic; padding: 12px;"
    )
    return lbl


def create_path_row(
    label_text: str,
    placeholder: str = "",
    parent: Optional["QtWidgets.QWidget"] = None,
) -> "tuple[QtWidgets.QWidget, QtWidgets.QLineEdit, QtWidgets.QPushButton]":
    """
    Return a reusable label + line-edit + browse-button row.

    Callers are expected to connect the returned button's ``clicked`` signal to
    a host-owned file/directory dialog handler.
    """

    row = QtWidgets.QWidget(parent)
    layout = QtWidgets.QHBoxLayout(row)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(8)

    label = QtWidgets.QLabel(label_text)
    label.setStyleSheet(f"color: {THEME['fg_muted']};")
    layout.addWidget(label)

    line_edit = QtWidgets.QLineEdit()
    if placeholder:
        line_edit.setPlaceholderText(placeholder)
    layout.addWidget(line_edit, 1)

    button = QtWidgets.QPushButton("Browse")
    button.setIcon(get_icon("fa6s.folder-open", THEME["fg_main"]))
    layout.addWidget(button)

    return row, line_edit, button


def style_primary_button(btn: "QtWidgets.QPushButton") -> None:
    """
    Apply the project-wide primary accent style to ``btn``.

    The application QSS already targets ``QPushButton#primaryBtn`` so this
    helper simply assigns the object name and re-polishes the widget.
    """

    btn.setObjectName("primaryBtn")
    btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
    btn.setStyleSheet(
        f"""
        QPushButton#primaryBtn {{
            background: qlineargradient(
                x1: 0, y1: 0, x2: 1, y2: 0,
                stop: 0 {THEME['accent']},
                stop: 1 {THEME['secondary']}
            );
            color: #05090F;
            border: 1px solid {THEME['accent']};
            border-radius: 8px;
            padding: 7px 16px;
            font-weight: 700;
        }}
        QPushButton#primaryBtn:hover {{
            background: {THEME['accent_hov']};
            border-color: {THEME['accent_hov']};
            color: #05090F;
        }}
        QPushButton#primaryBtn:disabled {{
            background: {THEME['bg_entry']};
            border-color: {THEME['border']};
            color: {THEME['text_disabled']};
        }}
        """
    )
    try:
        btn.style().unpolish(btn)
        btn.style().polish(btn)
    except Exception:
        pass


def style_secondary_button(btn: "QtWidgets.QPushButton") -> None:
    """Apply the project's quieter, neutral button style to ``btn``."""

    btn.setObjectName("secondaryBtn")
    btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
    btn.setStyleSheet(
        f"""
        QPushButton#secondaryBtn {{
            background: {THEME['bg_card_alt']};
            color: {THEME['fg_main']};
            border: 1px solid {THEME['border']};
            border-radius: 8px;
            padding: 7px 16px;
            font-weight: 600;
        }}
        QPushButton#secondaryBtn:hover {{
            background: {THEME['bg_entry']};
            border-color: {THEME['accent_hov']};
        }}
        QPushButton#secondaryBtn:disabled {{
            background: {THEME['bg_entry']};
            border-color: {THEME['border']};
            color: {THEME['fg_muted']};
        }}
        """
    )
    try:
        btn.style().unpolish(btn)
        btn.style().polish(btn)
    except Exception:
        pass


# =============================================================================
# 8.                   NEW FACTORY HELPERS (Task 1 additions)
# =============================================================================


def create_card(
    title: str,
    parent: Optional["QtWidgets.QWidget"] = None,
) -> "QtWidgets.QGroupBox":
    """
    Return a styled QGroupBox card with the project-wide dark card language.

    Drop-in replacement for the inline card-factory closures that individual
    pages previously defined locally.
    """
    gb = QtWidgets.QGroupBox(title, parent)
    gb.setStyleSheet(card_stylesheet())
    return gb


def create_section_header(
    title: str,
    subtitle: Optional[str] = None,
    parent: Optional["QtWidgets.QWidget"] = None,
) -> "QtWidgets.QWidget":
    """
    Return a labelled section header widget (title + optional subtitle row).
    """
    container = QtWidgets.QWidget(parent)
    lay = QtWidgets.QVBoxLayout(container)
    lay.setContentsMargins(0, 0, 0, 4)
    lay.setSpacing(2)

    lbl_title = QtWidgets.QLabel(title)
    lbl_title.setStyleSheet(
        f"color: {THEME['fg_soft']}; font-size: 10pt; font-weight: 700;"
        f" border-bottom: 1px solid {THEME['border_soft']}; padding-bottom: 4px;"
    )
    lay.addWidget(lbl_title)

    if subtitle:
        lbl_sub = QtWidgets.QLabel(subtitle)
        lbl_sub.setStyleSheet(
            f"color: {THEME['fg_muted']}; font-size: 9pt;"
        )
        lay.addWidget(lbl_sub)

    return container


def create_hint_label(
    text: str,
    kind: str = "info",
    parent: Optional["QtWidgets.QWidget"] = None,
) -> "QtWidgets.QLabel":
    """
    Return a small inline hint / callout label.

    *kind* can be ``"info"``, ``"warning"``, ``"success"``, or ``"error"``.
    """
    color_map = {
        "info":    THEME["info"],
        "warning": THEME["warning"],
        "success": THEME["success"],
        "error":   THEME["error"],
    }
    color = color_map.get(kind, THEME["fg_muted"])
    lbl = QtWidgets.QLabel(text, parent)
    lbl.setWordWrap(True)
    lbl.setStyleSheet(
        f"color: {color}; font-size: 9pt; font-style: italic; padding: 4px 0;"
    )
    return lbl


def create_primary_button(
    text: str,
    icon: Optional["QtGui.QIcon"] = None,
    parent: Optional["QtWidgets.QWidget"] = None,
) -> "QtWidgets.QPushButton":
    """Return a styled primary (cyan) action button."""
    btn = QtWidgets.QPushButton(text, parent)
    if icon:
        btn.setIcon(icon)
    style_primary_button(btn)
    return btn


def create_secondary_button(
    text: str,
    icon: Optional["QtGui.QIcon"] = None,
    parent: Optional["QtWidgets.QWidget"] = None,
) -> "QtWidgets.QPushButton":
    """Return a styled secondary (neutral) action button."""
    btn = QtWidgets.QPushButton(text, parent)
    if icon:
        btn.setIcon(icon)
    style_secondary_button(btn)
    return btn


def create_danger_button(
    text: str,
    icon: Optional["QtGui.QIcon"] = None,
    parent: Optional["QtWidgets.QWidget"] = None,
) -> "QtWidgets.QPushButton":
    """Return a styled danger (coral) action button."""
    btn = QtWidgets.QPushButton(text, parent)
    if icon:
        btn.setIcon(icon)
    btn.setObjectName("dangerBtn")
    btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
    btn.setStyleSheet(
        f"""
        QPushButton#dangerBtn {{
            background: rgba(255,107,122,0.12);
            color: {THEME['fg_main']};
            border: 1px solid rgba(255,107,122,0.30);
            border-radius: 8px;
            padding: 7px 16px;
            font-weight: 600;
        }}
        QPushButton#dangerBtn:hover {{
            background: rgba(255,107,122,0.22);
            border-color: {THEME['error']};
        }}
        QPushButton#dangerBtn:disabled {{
            background: {THEME['bg_entry']};
            border-color: {THEME['border']};
            color: {THEME['fg_muted']};
        }}
        """
    )
    return btn


def create_metric_chip(
    title: str,
    value: str = "—",
    subtitle: str = "",
    parent: Optional["QtWidgets.QWidget"] = None,
) -> "QtWidgets.QFrame":
    """
    Return a compact metric card: dim title on top, bold value, optional
    subtitle.  Suitable for header metric rows in monitoring pages.
    """
    frame = QtWidgets.QFrame(parent)
    frame.setObjectName("metricChip")
    frame.setStyleSheet(
        f"QFrame#metricChip {{"
        f"  background: {THEME['bg_card_alt']};"
        f"  border: 1px solid {THEME['border_soft']};"
        f"  border-radius: 10px;"
        f"  padding: 8px 14px;"
        f"}}"
    )
    lay = QtWidgets.QVBoxLayout(frame)
    lay.setContentsMargins(10, 8, 10, 8)
    lay.setSpacing(2)

    lbl_title = QtWidgets.QLabel(title)
    lbl_title.setStyleSheet(
        f"color: {THEME['fg_muted']}; font-size: 9pt; font-weight: 500;"
    )
    lbl_title.setAlignment(QtCore.Qt.AlignCenter)
    lay.addWidget(lbl_title)

    lbl_value = QtWidgets.QLabel(value)
    lbl_value.setObjectName("metricChipValue")
    lbl_value.setStyleSheet(
        f"color: {THEME['accent']}; font-size: 13pt; font-weight: 700;"
        f" font-family: Consolas, monospace;"
    )
    lbl_value.setAlignment(QtCore.Qt.AlignCenter)
    lay.addWidget(lbl_value)

    if subtitle:
        lbl_sub = QtWidgets.QLabel(subtitle)
        lbl_sub.setStyleSheet(f"color: {THEME['fg_muted']}; font-size: 8pt;")
        lbl_sub.setAlignment(QtCore.Qt.AlignCenter)
        lay.addWidget(lbl_sub)

    frame._value_label = lbl_value  # type: ignore[attr-defined]
    return frame


def apply_tree_style(tree: "QtWidgets.QTreeWidget") -> None:
    """Apply project-wide styling to a QTreeWidget."""
    tree.setStyleSheet(
        f"""
        QTreeWidget {{
            background: {THEME['bg_entry']};
            color: {THEME['fg_main']};
            border: 1px solid {THEME['border']};
            border-radius: 8px;
            alternate-background-color: {THEME['bg_card_alt']};
        }}
        QTreeWidget::item {{
            padding: 4px 6px;
        }}
        QTreeWidget::item:selected {{
            background: {THEME['secondary_dim']};
            color: {THEME['fg_main']};
        }}
        QTreeWidget::item:hover {{
            background: {THEME['accent_dim']};
        }}
        QHeaderView::section {{
            background: {THEME['bg_card']};
            color: {THEME['fg_soft']};
            border: none;
            border-bottom: 1px solid {THEME['border']};
            padding: 5px 8px;
            font-weight: 600;
        }}
        """
    )


def apply_plot_theme(widget: "QtWidgets.QWidget") -> None:
    """
    Apply the project-wide pyqtgraph/canvas background color to *widget*.

    This helper is intentionally thin — it only configures the background so
    that callers that construct their own plots stay in control of axis colors,
    pen widths, etc.
    """
    try:
        widget.setBackground(THEME["plot_bg"])  # type: ignore[attr-defined]
    except AttributeError:
        widget.setStyleSheet(f"background: {THEME['plot_bg']};")


def status_text(text: str, kind: str = "info") -> str:
    """
    Return an HTML-styled status string suitable for QLabel.setText().

    Handy for inline validation labels that need color-coded feedback.
    """
    color_map = {
        "info":    THEME["accent"],
        "success": THEME["success"],
        "warning": THEME["warning"],
        "error":   THEME["error"],
        "muted":   THEME["fg_muted"],
    }
    color = color_map.get(kind, THEME["fg_muted"])
    return f"<span style='color:{color};font-weight:600'>{text}</span>"

