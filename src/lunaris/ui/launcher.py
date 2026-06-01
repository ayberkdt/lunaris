# -*- coding: utf-8 -*-
"""
Lunaris Launcher / Welcome Hub.

Top-level parent window and single entry point for the desktop app. It lets the
user choose one of two workspaces:

1. **Lunar Propagation** — the classic mission-analysis desktop UI
   (:class:`lunaris.ui.app.MainWindow`).
2. **ST-LRPS Studio** — the surrogate-gravity training/evaluation suite
   (:class:`lunaris.surrogate.st_lrps.ui.studio_parts.main_window.MainWindow`).

Behind the navigation cards sits an optional, offline, interactive 3D Moon
(:class:`lunaris.ui.widgets.showcase_embed.ShowcaseEmbedWidget`). The web side is
a pure visual engine; all navigation lives in the PySide6 glassmorphic overlay.

Design rules
------------
- **Lazy loading.** Module-level imports are limited to PySide6, the
  dependency-light ``ui_commons`` helpers, and the UI-only ``showcase_embed``
  (whose QtWebEngine dependency is itself optional). The two workspaces — and
  their heavy transitive dependencies (PyTorch, h5py, pyqtgraph, the ST-LRPS
  training stack) — are imported *inside* the button callbacks, so choosing one
  workspace never pulls in the other's stack.
- **Single shared QApplication, in-process windows.** The Studio prefers PySide6
  (see ``studio_parts/qt_common.py``), the same binding as the classic UI, so
  both workspaces can be hosted in one process. The launcher hides itself while a
  workspace is open and reappears when that window is closed.
- **Never fails to open.** The 3D embed degrades to a dark background when the
  web build, QtWebEngine, or WebGL is unavailable.
"""

from __future__ import annotations

import os
import sys

from PySide6 import QtCore, QtGui, QtWidgets

from lunaris.ui.widgets.showcase_embed import ShowcaseEmbedWidget
from lunaris.ui.widgets.ui_commons import (
    ASSETS_DIR,
    THEME,
    find_project_root,
    get_icon,
    load_fonts,
)

GITHUB_URL = "https://github.com/ayberkdt/lunaris"
SHOWCASE_URL = "https://lunaris-showcase.vercel.app/"

# Delay between showing the "Opening …" overlay and starting the (blocking)
# workspace build. Long enough that the overlay paints and the Moon visibly
# keeps moving for a few frames; short enough to feel responsive.
_OPENING_BUILD_DELAY_MS = 280


# =============================================================================
# Card primitive
# =============================================================================

class _LaunchCard(QtWidgets.QFrame):
    """
    A clickable workspace card.

    Layout: a tinted rounded icon badge and the title on the top row (with a
    quiet accent "open" affordance on the right), and a wrapped description
    below. The whole card is the click target — there is no heavy CTA bar, which
    keeps the cards compact and gives a calmer, more technical feel.
    """

    clicked = QtCore.Signal()

    def __init__(
        self,
        *,
        title: str,
        description: str,
        icon_name: str,
        accent: str,
        action_text: str,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._accent = accent
        self.setObjectName("launchCard")
        self.setProperty("accent", accent)
        self.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.setMinimumHeight(132)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 22, 18)
        layout.setSpacing(12)

        # --- Top row: icon badge + title + open affordance --------------------
        header = QtWidgets.QHBoxLayout()
        header.setSpacing(14)

        badge = QtWidgets.QLabel()
        badge.setObjectName("cardIconBadge")
        badge.setProperty("accent", accent)
        badge.setFixedSize(44, 44)
        badge.setAlignment(QtCore.Qt.AlignCenter)
        badge.setPixmap(get_icon(icon_name, accent).pixmap(QtCore.QSize(22, 22)))
        header.addWidget(badge, 0, QtCore.Qt.AlignVCenter)

        title_lbl = QtWidgets.QLabel(title)
        title_lbl.setObjectName("cardTitle")
        header.addWidget(title_lbl, 0, QtCore.Qt.AlignVCenter)
        header.addStretch(1)

        # Quiet "open" cue: short label + arrow, accent-colored.
        self._open_cue = QtWidgets.QLabel()
        self._open_cue.setObjectName("cardOpenCue")
        self._open_cue.setText(action_text)
        self._open_cue.setToolTip(action_text)
        self._open_cue.setAlignment(QtCore.Qt.AlignVCenter | QtCore.Qt.AlignRight)
        header.addWidget(self._open_cue, 0, QtCore.Qt.AlignVCenter)

        arrow = QtWidgets.QLabel()
        arrow.setFixedSize(16, 16)
        arrow.setPixmap(get_icon("fa6s.arrow-right", accent).pixmap(QtCore.QSize(14, 14)))
        header.addWidget(arrow, 0, QtCore.Qt.AlignVCenter)

        layout.addLayout(header)

        # --- Description ------------------------------------------------------
        desc_lbl = QtWidgets.QLabel(description)
        desc_lbl.setObjectName("cardDesc")
        desc_lbl.setWordWrap(True)
        desc_lbl.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum
        )
        layout.addWidget(desc_lbl)

        # Back-compat handle for callers/tests expecting `action_btn` semantics.
        self.action_btn = self

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.LeftButton and self.rect().contains(event.pos()):
            self.clicked.emit()
            event.accept()
            return
        super().mouseReleaseEvent(event)


# =============================================================================
# Launcher window
# =============================================================================

class LauncherWindow(QtWidgets.QWidget):
    """Welcome hub: a 3D Moon background with glassmorphic workspace cards."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)

        # Anchor for the currently open workspace so it is not garbage collected.
        self._active_workspace: QtWidgets.QWidget | None = None

        self.setWindowTitle("Lunaris")
        # A comfortable default that fits the full layout without scrolling;
        # the minimum is smaller because the panel content scrolls if needed.
        self.setMinimumSize(900, 600)
        self.resize(1280, 800)
        self.setAttribute(QtCore.Qt.WA_StyledBackground, True)

        icon_path = ASSETS_DIR / "icon.ico"
        if icon_path.exists():
            self.setWindowIcon(QtGui.QIcon(str(icon_path)))

        # Background 3D Moon embed (degrades to a dark background on its own).
        self.embed = ShowcaseEmbedWidget(self)
        self.embed.lower()

        # Foreground glassmorphic panel. The navigation lives inside a scroll
        # area so the content can never collide on a short window: it simply
        # scrolls instead of overlapping.
        self.panel = QtWidgets.QFrame(self)
        self.panel.setObjectName("glassPanel")

        self._scroll = QtWidgets.QScrollArea(self.panel)
        self._scroll.setObjectName("panelScroll")
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        self._scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self._scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)

        self._content = QtWidgets.QWidget()
        self._content.setObjectName("panelContent")
        self._scroll.setWidget(self._content)

        panel_lay = QtWidgets.QVBoxLayout(self.panel)
        panel_lay.setContentsMargins(0, 0, 0, 0)
        panel_lay.addWidget(self._scroll)

        self._build_ui()
        self._apply_theme()
        self._position_overlay()

    # ------------------------------------------------------------------ UI ---
    def _build_ui(self) -> None:
        root = QtWidgets.QVBoxLayout(self._content)
        root.setContentsMargins(40, 34, 40, 26)
        root.setSpacing(0)

        # --- Masthead: kicker + wordmark + divider ---------------------------
        kicker = QtWidgets.QLabel("MISSION CONTROL")
        kicker.setObjectName("launchKicker")
        root.addWidget(kicker)

        root.addSpacing(6)

        title = QtWidgets.QLabel("Lunaris")
        title.setObjectName("launchTitle")
        root.addWidget(title)

        root.addSpacing(10)

        subtitle = QtWidgets.QLabel(
            "Lunar orbit propagation and neural gravity surrogate research environment."
        )
        subtitle.setObjectName("launchSubtitle")
        subtitle.setWordWrap(True)
        root.addWidget(subtitle)

        root.addSpacing(20)

        divider = QtWidgets.QFrame()
        divider.setObjectName("headerDivider")
        divider.setFrameShape(QtWidgets.QFrame.HLine)
        divider.setFixedHeight(1)
        root.addWidget(divider)

        root.addSpacing(22)

        # Cards — stacked vertically so the 3D Moon stays visible on the right.
        self.card_propagation = _LaunchCard(
            title="Lunar Propagation",
            description=(
                "Configure lunar orbits, select force models, run high-precision "
                "propagation, Monte Carlo studies, telemetry, and result exports."
            ),
            icon_name="fa6s.rocket",
            accent=THEME["accent"],
            action_text="Open Mission Analysis",
        )
        self.card_propagation.clicked.connect(self._open_propagation)
        root.addWidget(self.card_propagation)

        root.addSpacing(16)

        self.card_studio = _LaunchCard(
            title="ST-LRPS Studio",
            description=(
                "Generate spatial gravity datasets, train Sobolev residual-potential "
                "surrogates, evaluate field accuracy, benchmark orbit-level "
                "performance, and prepare research outputs."
            ),
            icon_name="fa6s.brain",
            accent=THEME["secondary"],
            action_text="Open Surrogate Suite",
        )
        self.card_studio.clicked.connect(self._open_studio)
        root.addWidget(self.card_studio)

        # Push the footer links to the bottom with clear separation from cards.
        root.addStretch(1)
        root.addSpacing(22)

        # Footer: just the external links, right-aligned.
        footer = QtWidgets.QHBoxLayout()
        footer.setSpacing(18)
        footer.addStretch(1)

        self.btn_showcase = QtWidgets.QPushButton("Open Web Showcase")
        self.btn_showcase.setObjectName("footerLink")
        self.btn_showcase.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.btn_showcase.clicked.connect(self._open_web_showcase)
        footer.addWidget(self.btn_showcase)

        btn_github = QtWidgets.QPushButton("Docs / GitHub")
        btn_github.setObjectName("footerLink")
        btn_github.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        btn_github.clicked.connect(
            lambda: QtGui.QDesktopServices.openUrl(QtCore.QUrl(GITHUB_URL))
        )
        footer.addWidget(btn_github)

        root.addLayout(footer)

        # Scene toggles float over the Moon (they control it), not inside the
        # panel. Parented to the window and positioned in _position_overlay().
        self._build_scene_toggles()

    def _build_scene_toggles(self) -> None:
        """
        Floating Visual / Gravity / Orbit controls.

        These drive the 3D scene, so they live over the Moon on the right rather
        than inside the navigation panel. Shown only when the WebGL view is live.
        """
        self.scene_controls = QtWidgets.QFrame(self)
        self.scene_controls.setObjectName("sceneControls")
        row = QtWidgets.QHBoxLayout(self.scene_controls)
        row.setContentsMargins(8, 8, 8, 8)
        row.setSpacing(6)

        self.btn_visual = QtWidgets.QPushButton("Visual")
        self.btn_gravity = QtWidgets.QPushButton("Gravity")
        for btn in (self.btn_visual, self.btn_gravity):
            btn.setObjectName("sceneToggle")
            btn.setCheckable(True)
            btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.btn_visual.setChecked(True)

        self._texture_group = QtWidgets.QButtonGroup(self.scene_controls)
        self._texture_group.setExclusive(True)
        self._texture_group.addButton(self.btn_visual)
        self._texture_group.addButton(self.btn_gravity)
        self.btn_visual.clicked.connect(lambda: self.embed.set_texture_mode("visual"))
        self.btn_gravity.clicked.connect(lambda: self.embed.set_texture_mode("gravity"))

        # Thin separator between the texture toggle pair and the orbit toggle.
        sep = QtWidgets.QFrame(self.scene_controls)
        sep.setFrameShape(QtWidgets.QFrame.VLine)
        sep.setObjectName("toggleSep")

        self.btn_orbit = QtWidgets.QPushButton("Orbit")
        self.btn_orbit.setObjectName("sceneToggle")
        self.btn_orbit.setCheckable(True)
        self.btn_orbit.setChecked(True)
        self.btn_orbit.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.btn_orbit.toggled.connect(self.embed.set_orbit_visible)

        # Second separator before the relief slider.
        sep2 = QtWidgets.QFrame(self.scene_controls)
        sep2.setFrameShape(QtWidgets.QFrame.VLine)
        sep2.setObjectName("toggleSep")

        # Minimal surface-relief slider (displacement exaggeration, 0..100%).
        relief_lbl = QtWidgets.QLabel("Relief")
        relief_lbl.setObjectName("toggleLabel")

        self.slider_relief = QtWidgets.QSlider(QtCore.Qt.Horizontal, self.scene_controls)
        self.slider_relief.setObjectName("reliefSlider")
        self.slider_relief.setMinimum(0)
        self.slider_relief.setMaximum(100)
        self.slider_relief.setValue(50)
        self.slider_relief.setFixedWidth(96)
        self.slider_relief.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.slider_relief.valueChanged.connect(
            lambda v: self.embed.set_relief(v / 100.0)
        )

        row.addWidget(self.btn_visual)
        row.addWidget(self.btn_gravity)
        row.addWidget(sep)
        row.addWidget(self.btn_orbit)
        row.addWidget(sep2)
        row.addWidget(relief_lbl)
        row.addWidget(self.slider_relief)

        # Only meaningful when the interactive WebGL view is actually running.
        self.scene_controls.setVisible(self.embed.is_live())

    def _apply_theme(self) -> None:
        """Instance-level stylesheet so workspace windows are never restyled."""
        self.setStyleSheet(
            f"""
            LauncherWindow {{
                background: #000000;
            }}
            QWidget {{
                color: {THEME['fg_main']};
                font-family: "Segoe UI", "Inter", "Noto Sans", sans-serif;
            }}
            QFrame#glassPanel {{
                background: rgba(7, 11, 20, 0.85);
                border: 1px solid rgba(120, 150, 190, 0.16);
                border-radius: 22px;
            }}
            QFrame#openingOverlay {{
                background: rgba(3, 6, 12, 0.55);
            }}
            QLabel#openingCard {{
                background: rgba(7, 11, 20, 0.94);
                color: {THEME['fg_soft']};
                border: 1px solid rgba(53, 208, 255, 0.35);
                border-radius: 14px;
                padding: 20px 38px;
                font-size: 12pt;
                font-weight: 600;
            }}
            QScrollArea#panelScroll {{
                background: transparent;
                border: none;
            }}
            QWidget#panelContent {{
                background: transparent;
            }}
            QScrollArea#panelScroll QScrollBar:vertical {{
                background: transparent;
                width: 7px;
                margin: 14px 4px 14px 0;
            }}
            QScrollArea#panelScroll QScrollBar::handle:vertical {{
                background: rgba(120, 150, 190, 0.30);
                border-radius: 3px;
                min-height: 28px;
            }}
            QScrollArea#panelScroll QScrollBar::handle:vertical:hover {{
                background: rgba(53, 208, 255, 0.45);
            }}
            QScrollArea#panelScroll QScrollBar::add-line:vertical,
            QScrollArea#panelScroll QScrollBar::sub-line:vertical {{
                height: 0;
            }}
            QLabel#launchKicker {{
                font-size: 8.5pt;
                font-weight: 700;
                color: {THEME['accent']};
                letter-spacing: 3px;
                background: transparent;
            }}
            QLabel#launchTitle {{
                font-size: 40pt;
                font-weight: 800;
                color: {THEME['fg_main']};
                letter-spacing: -0.5px;
                background: transparent;
            }}
            QLabel#launchSubtitle {{
                font-size: 11pt;
                color: {THEME['fg_muted']};
                line-height: 152%;
                background: transparent;
            }}
            QFrame#headerDivider {{
                border: none;
                background: rgba(120, 150, 190, 0.14);
            }}
            QFrame#launchCard {{
                background: rgba(15, 23, 39, 0.72);
                border: 1px solid {THEME['border']};
                border-radius: 14px;
            }}
            QFrame#launchCard:hover {{
                border: 1px solid {THEME['accent']};
                background: rgba(20, 31, 52, 0.86);
            }}
            QFrame#launchCard[accent="{THEME['secondary']}"]:hover {{
                border: 1px solid {THEME['secondary']};
            }}
            QLabel#cardIconBadge {{
                background: rgba(53, 208, 255, 0.10);
                border: 1px solid rgba(53, 208, 255, 0.22);
                border-radius: 12px;
            }}
            QLabel#cardIconBadge[accent="{THEME['secondary']}"] {{
                background: rgba(139, 124, 255, 0.10);
                border: 1px solid rgba(139, 124, 255, 0.24);
            }}
            QLabel#cardTitle {{
                font-size: 14pt;
                font-weight: 700;
                color: {THEME['fg_main']};
                background: transparent;
            }}
            QLabel#cardOpenCue {{
                font-size: 8.5pt;
                font-weight: 600;
                color: {THEME['fg_muted']};
                letter-spacing: 0.3px;
                background: transparent;
            }}
            QLabel#cardDesc {{
                font-size: 9.5pt;
                color: {THEME['fg_muted']};
                line-height: 150%;
                background: transparent;
            }}
            QPushButton#footerLink {{
                background: transparent;
                color: {THEME['fg_muted']};
                border: none;
                padding: 4px 4px;
                font-size: 9pt;
            }}
            QPushButton#footerLink:hover {{
                color: {THEME['accent']};
            }}
            QFrame#sceneControls {{
                background: rgba(7, 11, 20, 0.78);
                border: 1px solid rgba(120, 150, 190, 0.18);
                border-radius: 13px;
            }}
            QFrame#toggleSep {{
                color: {THEME['border']};
                max-width: 1px;
                margin: 3px 4px;
            }}
            QPushButton#sceneToggle {{
                background: transparent;
                color: {THEME['fg_muted']};
                border: 1px solid transparent;
                border-radius: 8px;
                padding: 6px 14px;
                font-size: 9pt;
                font-weight: 600;
            }}
            QPushButton#sceneToggle:hover {{
                color: {THEME['fg_main']};
                border-color: rgba(53, 208, 255, 0.35);
            }}
            QPushButton#sceneToggle:checked {{
                background: {THEME['accent_dim']};
                color: {THEME['fg_soft']};
                border-color: {THEME['accent']};
            }}
            QLabel#toggleLabel {{
                color: {THEME['fg_muted']};
                font-size: 8.5pt;
                font-weight: 600;
                padding: 0 2px 0 4px;
                background: transparent;
            }}
            QSlider#reliefSlider::groove:horizontal {{
                height: 3px;
                background: rgba(120, 150, 190, 0.30);
                border-radius: 2px;
            }}
            QSlider#reliefSlider::sub-page:horizontal {{
                height: 3px;
                background: {THEME['accent']};
                border-radius: 2px;
            }}
            QSlider#reliefSlider::handle:horizontal {{
                width: 12px;
                height: 12px;
                margin: -5px 0;
                border-radius: 6px;
                background: {THEME['fg_main']};
                border: 1px solid {THEME['accent']};
            }}
            QSlider#reliefSlider::handle:horizontal:hover {{
                background: {THEME['accent_hov']};
            }}
            """
        )

    # ------------------------------------------------------------ geometry ---
    def _position_overlay(self) -> None:
        """
        Keep the embed full-bleed and the glass panel anchored to the LEFT.

        The 3D Moon is rendered toward the right of the canvas (see
        ``LauncherScene3D``), so a left-anchored panel leaves the Moon clearly
        visible on the right. On narrow windows the panel widens toward center so
        the cards stay readable.
        """
        self.embed.setGeometry(self.rect())

        margin = 40
        # Panel takes a little under half the width on wide windows; more on
        # narrow ones so the stacked cards never get cramped.
        if self.width() >= 1180:
            panel_w = min(int(self.width() * 0.46), 600)
        else:
            panel_w = min(max(int(self.width() * 0.62), 420), self.width() - 2 * margin)

        # Height is DYNAMIC: size to the content the panel actually needs (so
        # the header/cards/footer never collide), clamped to the window. If the
        # window is too short, the panel fills the available height and the inner
        # scroll area takes over — content scrolls instead of overlapping.
        self.panel.setFixedWidth(panel_w)
        # Use the content's full preferred height (plus a couple of px so the
        # scroll viewport fully contains it, accounting for the panel border) so
        # that, when the window is tall enough, no scrollbar appears. When the
        # window is too short, the panel is clamped and the scroll area engages.
        content_h = self._content.sizeHint().height() + 4
        avail_h = self.height() - 2 * margin
        panel_h = max(min(content_h, avail_h), min(avail_h, 360))

        x = margin
        y = (self.height() - panel_h) // 2
        self.panel.setGeometry(x, y, panel_w, panel_h)
        self.panel.raise_()

        # Float the scene toggles over the Moon: horizontally centered in the
        # right-hand (Moon) region, near the bottom.
        controls = getattr(self, "scene_controls", None)
        if controls is not None:
            controls.adjustSize()
            cw = controls.sizeHint().width()
            ch = controls.sizeHint().height()
            moon_left = x + panel_w
            moon_region_w = max(self.width() - moon_left, cw)
            cx = moon_left + (moon_region_w - cw) // 2
            cy = self.height() - margin - ch
            controls.setGeometry(cx, cy, cw, ch)
            controls.raise_()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self._position_overlay()

    # -------------------------------------------------------- workspaces ---
    def _open_propagation(self) -> None:
        """Open the classic Lunar Propagation workspace (lazy import)."""
        self._open_workspace("Lunar Propagation", self._build_propagation_window)

    def _open_studio(self) -> None:
        """Open the ST-LRPS Studio workspace (lazy import + Studio theme)."""
        self._open_workspace("ST-LRPS Studio", self._build_studio_window)

    def _build_propagation_window(self) -> QtWidgets.QWidget:
        from lunaris.ui.app import MainWindow as PropagationMainWindow

        return PropagationMainWindow()

    def _build_studio_window(self) -> QtWidgets.QWidget:
        from lunaris.surrogate.st_lrps.ui.studio_parts.qt_common import (
            apply_premium_dark_theme,
        )
        from lunaris.surrogate.st_lrps.ui.studio_parts.common_widgets import (
            _NoWheelOnSpinFilter,
        )
        from lunaris.surrogate.st_lrps.ui.studio_parts.main_window import (
            MainWindow as StudioMainWindow,
        )

        app = QtWidgets.QApplication.instance()
        if app is not None:
            apply_premium_dark_theme(app)
            # Install the Studio's wheel guard once for the shared application.
            if not getattr(self, "_studio_wheel_guard", None):
                self._studio_wheel_guard = _NoWheelOnSpinFilter(app)
                app.installEventFilter(self._studio_wheel_guard)

        return StudioMainWindow()

    def _open_workspace(self, name: str, builder) -> None:
        """
        Open a workspace with smooth "opening" feedback.

        Flow (matches the desired UX):

        1. An ``Opening {name} …`` overlay appears instantly *over* the launcher.
           The 3D Moon is NOT torn down — it stays behind the overlay and keeps
           spinning while the overlay is up.
        2. After a short beat (so the overlay paints and the Moon visibly keeps
           moving), the heavy ``builder()`` runs. Building a big ``MainWindow``
           briefly blocks the UI thread, but the overlay makes that read as
           intentional loading rather than a freeze.
        3. Once built, the launcher hides (so the Moon is NOT left behind the
           workspace) and the workspace is shown.
        4. Closing the workspace shows the launcher again — instantly, because the
           embed was only hidden, never reloaded.
        """
        if getattr(self, "_opening", False):
            return
        self._opening = True
        self._show_opening_overlay(name)
        # Defer the blocking build so the overlay paints first and the Moon gets
        # a few frames of motion before the build pause.
        QtCore.QTimer.singleShot(_OPENING_BUILD_DELAY_MS, lambda: self._finish_open(name, builder))

    def _finish_open(self, name: str, builder) -> None:
        """Build the workspace, then hand off from the launcher to it."""
        try:
            window = builder()
        except Exception as exc:  # pragma: no cover - environment dependent
            self._hide_opening_overlay()
            self._opening = False
            self._report_launch_error(name, exc)
            return
        self._hide_opening_overlay()
        self._opening = False
        self._launch_workspace(window)

    def _show_opening_overlay(self, name: str) -> None:
        """A light scrim + centered card over the launcher (Moon stays behind it)."""
        overlay = QtWidgets.QFrame(self)
        overlay.setObjectName("openingOverlay")
        overlay.setGeometry(self.rect())

        lay = QtWidgets.QVBoxLayout(overlay)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addStretch(1)

        card = QtWidgets.QLabel(f"Opening {name} …")
        card.setObjectName("openingCard")
        card.setAlignment(QtCore.Qt.AlignCenter)
        lay.addWidget(card, 0, QtCore.Qt.AlignHCenter)

        lay.addStretch(1)

        overlay.show()
        overlay.raise_()
        overlay.repaint()  # paint immediately so feedback is instant
        self._opening_overlay = overlay

    def _hide_opening_overlay(self) -> None:
        overlay = getattr(self, "_opening_overlay", None)
        if overlay is not None:
            try:
                overlay.hide()
                overlay.deleteLater()
            except Exception:
                pass
            self._opening_overlay = None

    def _launch_workspace(self, window: QtWidgets.QWidget) -> None:
        """
        Show *window* and hide the launcher so the Moon is not behind it.

        The launcher is only HIDDEN (not destroyed) and the embed is never torn
        down, so returning to it on close is instant — the Moon resumes rather
        than reloading.
        """
        self._active_workspace = window
        window.setAttribute(QtCore.Qt.WA_DeleteOnClose, True)
        window.destroyed.connect(self._on_workspace_closed)
        window.show()
        window.raise_()
        window.activateWindow()
        self.hide()

    def _on_workspace_closed(self, _obj: object = None) -> None:
        """Bring the launcher (and its still-loaded Moon) back to the front."""
        self._active_workspace = None
        self.show()
        self.raise_()
        self.activateWindow()

    # -------------------------------------------------------------- misc ---
    def _open_web_showcase(self) -> None:
        """Open the hosted web showcase in the system browser."""
        QtGui.QDesktopServices.openUrl(QtCore.QUrl(SHOWCASE_URL))

    def _report_launch_error(self, name: str, exc: Exception) -> None:
        QtWidgets.QMessageBox.critical(
            self,
            "Launch Error",
            f"Could not open {name}:\n\n{type(exc).__name__}: {exc}",
        )

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        try:
            self.embed.shutdown()
        except Exception:
            pass
        super().closeEvent(event)


# =============================================================================
# Entry point
# =============================================================================

def main() -> None:
    """Application entry point for the Lunaris launcher (``lunaris-launcher``)."""
    # High-DPI setup must happen before the QApplication is created. This mirrors
    # the ST-LRPS Studio entry point so the Studio renders identically whether it
    # is launched directly or via the hub.
    try:
        QtGui.QGuiApplication.setHighDpiScaleFactorRoundingPolicy(
            QtCore.Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
        )
    except Exception:
        pass
    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")

    try:
        os.chdir(str(find_project_root()))
    except Exception:
        pass

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    app.setApplicationName("Lunaris")
    app.setFont(load_fonts())

    window = LauncherWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
