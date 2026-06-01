# -*- coding: utf-8 -*-
"""
Offline 3D-Moon showcase embed for the Lunaris launcher.

This widget renders the interactive 3D Moon (the Next.js / Three.js ``/embed``
route, statically exported) **behind** the launcher's navigation cards. It is a
pure UI-layer convenience: the launcher works perfectly without it.

Robustness is the headline requirement — the launcher must never fail to open
because of a missing web build, a missing QtWebEngine, or a GPU without WebGL.
The widget therefore degrades through a four-level fallback chain:

1. QtWebEngine available **and** a static export is found
       -> serve the export from a local loopback HTTP server and load
          ``http://127.0.0.1:<port>/embed/`` in a ``QWebEngineView``.
          A watchdog checks ``window.lunarisReady``; if WebGL never comes up,
          it swaps to the dark fallback.
2. QtWebEngine available but no export build
       -> dark gradient + a small "3D preview not built" note.
3. QtWebEngine missing
       -> static fallback image if present, else a dark gradient.
4. Anything raises
       -> a plain solid dark background.

Offline by design: the loopback server binds ``127.0.0.1`` and serves only local
files, so no internet connection is ever required.

Architectural note: ``PySide6.QtWebEngineWidgets`` is imported lazily inside a
``try/except`` so importing this module (and therefore the launcher) never fails
when QtWebEngine is not installed.
"""

from __future__ import annotations

import os
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional

from PySide6 import QtCore, QtGui, QtWidgets

from lunaris.ui.core.ui_commons import THEME, find_project_root

# --- Optional QtWebEngine import (must never break module import) ------------
try:
    from PySide6.QtWebEngineWidgets import QWebEngineView  # type: ignore

    HAS_WEBENGINE = True
except Exception:  # pragma: no cover - depends on the install
    QWebEngineView = None  # type: ignore
    HAS_WEBENGINE = False


# Background used by every non-WebGL fallback. Matches the web embed's #000000 /
# the project's deep-space canvas so the transition is seamless.
_FALLBACK_BG = "#05050A"
_READY_TIMEOUT_MS = 9000


def _web_engine_is_safe() -> bool:
    """
    Decide whether it is safe to instantiate a ``QWebEngineView``.

    QtWebEngine starts a Chromium render process that needs a real windowing
    system and GL context. On non-GUI Qt platform plugins (``offscreen`` /
    ``minimal``, used in CI, tests, and headless HPC nodes) that initialization
    can hard-crash the process — a C++ segfault that Python ``try/except``
    cannot catch. Skipping the web engine there keeps the launcher's
    "never fails to open" guarantee intact (it falls back to a dark background).

    An explicit ``LUNARIS_DISABLE_WEB_EMBED=1`` opt-out is also honored.
    """
    if os.environ.get("LUNARIS_DISABLE_WEB_EMBED", "").strip() not in ("", "0", "false", "False"):
        return False
    platform = os.environ.get("QT_QPA_PLATFORM", "").strip().lower()
    if platform in {"offscreen", "minimal", "vnc"}:
        return False
    return True


# =============================================================================
# Embed-dir resolution
# =============================================================================

def resolve_web_embed_dir() -> Optional[Path]:
    """
    Locate a built static export that contains ``embed/index.html``.

    Resolution order:

    1. ``LUNARIS_WEB_EMBED_DIR`` environment variable.
    2. A build vendored into the package at
       ``lunaris/ui/assets/web_embed/``.
    3. The in-repo build at
       ``<repo>/desktop/website/lunaris-web/out/``.

    Returns the first directory that actually contains ``embed/index.html``,
    otherwise ``None``.
    """

    candidates: list[Path] = []

    env = os.environ.get("LUNARIS_WEB_EMBED_DIR", "").strip()
    if env:
        candidates.append(Path(env).expanduser())

    # Vendored build inside the installed package.
    candidates.append(Path(__file__).resolve().parents[1] / "assets" / "web_embed")

    # In-repo build output.
    try:
        repo_root = find_project_root()
        candidates.append(repo_root / "web" / "out")
    except Exception:
        pass

    for candidate in candidates:
        try:
            if (candidate / "embed" / "index.html").is_file():
                return candidate
        except Exception:
            continue
    return None


# =============================================================================
# Loopback static server
# =============================================================================

class _LoopbackServer:
    """
    A tiny localhost-only static file server for the static export.

    Binds ``127.0.0.1:0`` (ephemeral port) and serves ``root_dir`` from a daemon
    thread. Used instead of ``file://`` because a Next static export references
    assets with absolute ``/_next/...`` paths that do not resolve under the
    ``file://`` scheme.
    """

    def __init__(self, root_dir: Path) -> None:
        self._root = Path(root_dir)
        self._httpd: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self.port: Optional[int] = None

    def start(self) -> int:
        handler = partial(_QuietHandler, directory=str(self._root))
        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.port = int(self._httpd.server_address[1])
        self._thread = threading.Thread(
            target=self._httpd.serve_forever,
            name="lunaris-embed-httpd",
            daemon=True,
        )
        self._thread.start()
        return self.port

    def stop(self) -> None:
        if self._httpd is not None:
            try:
                self._httpd.shutdown()
                self._httpd.server_close()
            except Exception:
                pass
        self._httpd = None
        self._thread = None
        self.port = None


class _QuietHandler(SimpleHTTPRequestHandler):
    """SimpleHTTPRequestHandler that does not spam stderr with request logs."""

    def log_message(self, *_args, **_kwargs) -> None:  # noqa: D401 - silence
        return


# =============================================================================
# Embed widget
# =============================================================================

class ShowcaseEmbedWidget(QtWidgets.QWidget):
    """
    Background 3D-Moon embed with a graceful fallback chain.

    Construction never raises: any failure collapses to a dark background. Use
    :meth:`is_live` to learn whether the interactive WebGL view is actually
    running (the launcher uses this to decide whether to show its scene
    toggles).
    """

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("showcaseEmbed")
        self.setAttribute(QtCore.Qt.WA_StyledBackground, True)
        self.setStyleSheet(f"QWidget#showcaseEmbed {{ background: {_FALLBACK_BG}; }}")

        self._server: Optional[_LoopbackServer] = None
        self._view: Optional[QWebEngineView] = None  # type: ignore[assignment]
        self._live = False

        self._layout = QtWidgets.QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(0)

        try:
            self._build()
        except Exception:
            # Level 4: never let the launcher fail because of the embed.
            self._show_fallback(note="")

    # ----------------------------------------------------------------- build --
    def _build(self) -> None:
        embed_dir = resolve_web_embed_dir()
        engine_ok = HAS_WEBENGINE and _web_engine_is_safe()

        if engine_ok and embed_dir is not None:
            self._build_webengine(embed_dir)
            return

        if engine_ok and embed_dir is None:
            # Level 2: engine present, build missing.
            self._show_fallback(
                note="3D preview not built — run `npm run build` in "
                "desktop/website/lunaris-web",
            )
            return

        # Level 3: no usable QtWebEngine (absent, or unsafe headless platform).
        self._show_fallback(note="")

    def _build_webengine(self, embed_dir: Path) -> None:
        """Level 1: loopback server + QWebEngineView, with a readiness watchdog."""
        self._server = _LoopbackServer(embed_dir)
        port = self._server.start()

        self._view = QWebEngineView(self)  # type: ignore[call-arg]
        self._view.setUrl(QtCore.QUrl(f"http://127.0.0.1:{port}/embed/"))
        self._layout.addWidget(self._view)
        self._live = True

        # Watchdog: if window.lunarisReady is never true (WebGL failed), fall
        # back to the dark background so the user does not stare at a white/blank
        # canvas.
        self._ready_timer = QtCore.QTimer(self)
        self._ready_timer.setSingleShot(True)
        self._ready_timer.timeout.connect(self._check_ready)
        self._ready_timer.start(_READY_TIMEOUT_MS)

    def _check_ready(self) -> None:
        view = self._view
        if view is None:
            return

        def _on_result(value: object) -> None:
            if not value:
                # WebGL never came up — degrade to the dark fallback.
                self._teardown_view()
                self._show_fallback(note="")

        try:
            view.page().runJavaScript("window.lunarisReady === true", _on_result)
        except Exception:
            pass

    # -------------------------------------------------------------- fallback --
    def _show_fallback(self, *, note: str) -> None:
        """Render a dark gradient (optionally with a small note) and stop the view."""
        self._live = False

        container = QtWidgets.QFrame(self)
        container.setObjectName("embedFallback")
        container.setStyleSheet(
            f"""
            QFrame#embedFallback {{
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 1,
                    stop: 0 #000000,
                    stop: 0.5 {_FALLBACK_BG},
                    stop: 1 #0A0F1E
                );
            }}
            """
        )
        lay = QtWidgets.QVBoxLayout(container)
        lay.setContentsMargins(0, 0, 0, 0)

        if note:
            lbl = QtWidgets.QLabel(note, container)
            lbl.setAlignment(QtCore.Qt.AlignCenter)
            lbl.setWordWrap(True)
            lbl.setStyleSheet(
                f"color: {THEME['fg_muted']}; font-size: 9pt; padding: 16px;"
            )
            lay.addStretch(1)
            lay.addWidget(lbl)
            lay.addSpacing(24)

        self._layout.addWidget(container)

    # ------------------------------------------------------------ public API --
    def is_live(self) -> bool:
        """True only while the interactive WebGL view is active."""
        return bool(self._live and self._view is not None)

    def set_texture_mode(self, mode: str) -> None:
        """Switch the Moon texture: ``"visual"`` (aesthetic) or ``"gravity"``."""
        normalized = "gravity" if str(mode).lower() == "gravity" else "visual"
        self._run_js(f'window.lunarisSetTextureMode && window.lunarisSetTextureMode("{normalized}")')

    def set_orbit_visible(self, visible: bool) -> None:
        """Show or hide the demo orbit + satellite."""
        flag = "true" if visible else "false"
        self._run_js(f"window.lunarisSetOrbitVisible && window.lunarisSetOrbitVisible({flag})")

    def set_performance_mode(self, mode: str) -> None:
        """Set render quality: ``"quality"`` | ``"balanced"`` | ``"low"``."""
        normalized = str(mode).lower()
        if normalized not in {"quality", "balanced", "low"}:
            normalized = "quality"
        self._run_js(
            f'window.lunarisSetPerformanceMode && window.lunarisSetPerformanceMode("{normalized}")'
        )

    def set_relief(self, value: float) -> None:
        """Set surface-relief (displacement) exaggeration, normalized ``0..1``."""
        try:
            v = max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return
        self._run_js(f"window.lunarisSetRelief && window.lunarisSetRelief({v:.4f})")

    def shutdown(self) -> None:
        """Stop the loopback server and release the web view. Idempotent/safe."""
        self._teardown_view()
        if self._server is not None:
            try:
                self._server.stop()
            except Exception:
                pass
            self._server = None

    # --------------------------------------------------------------- helpers --
    def _run_js(self, script: str) -> None:
        if not self.is_live():
            return
        try:
            self._view.page().runJavaScript(script)  # type: ignore[union-attr]
        except Exception:
            pass

    def _teardown_view(self) -> None:
        self._live = False
        if getattr(self, "_view", None) is not None:
            try:
                self._view.setParent(None)
                self._view.deleteLater()
            except Exception:
                pass
            self._view = None
