#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Capture a screenshot of the Lunaris desktop main window (Lunar Graphite theme).

Renders :class:`lunaris.ui.app.MainWindow`, lets it settle for a moment so the
orbit-preview timers fire, then grabs the window to a PNG. Forces Qt's
``offscreen`` platform by default so it runs without a display server (e.g. in
CI or over SSH); override by exporting ``QT_QPA_PLATFORM`` yourself.

Usage
-----
    python tools/ui/capture_main_window.py
    python tools/ui/capture_main_window.py --out outputs/ui/custom.png --delay 1.2
    python tools/ui/capture_main_window.py --page Forces --state validating
    python tools/ui/capture_main_window.py --dialog gravity

Output (default): ``outputs/ui/main_window_lunar_graphite.png``
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Default to headless rendering unless the caller asked for a real display.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

REPO_ROOT = Path(os.path.abspath(__file__)).parents[2]
DEFAULT_OUT = REPO_ROOT / "outputs" / "ui" / "main_window_lunar_graphite.png"
os.environ.setdefault(
    "LUNARIS_APP_DATA_DIR",
    str(REPO_ROOT / "outputs" / "ui" / "capture_runtime"),
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Output PNG path (default: {DEFAULT_OUT}).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.8,
        help="Seconds to let the window settle before capture (default: 0.8).",
    )
    parser.add_argument(
        "--width", type=int, default=1280, help="Window width in px (default: 1280)."
    )
    parser.add_argument(
        "--height", type=int, default=860, help="Window height in px (default: 860)."
    )
    parser.add_argument(
        "--target",
        choices=("mission", "st-lrps"),
        default="mission",
        help="Desktop window to capture (default: mission).",
    )
    parser.add_argument(
        "--page",
        default="",
        help="Visible workspace page (for example Orbit, Forces, Data, or Training Monitor).",
    )
    parser.add_argument(
        "--state",
        choices=("idle", "validating", "running"),
        default="idle",
        help="Visible lifecycle state to render (default: idle).",
    )
    parser.add_argument(
        "--dialog",
        choices=("none", "gravity"),
        default="none",
        help="Optional Mission Studio dialog to capture (default: none).",
    )
    return parser.parse_args(argv)


def _select_page(window, target: str, page: str) -> None:
    requested = page.strip().casefold()
    if not requested:
        return
    if target == "st-lrps":
        titles = list(getattr(window, "_page_titles", []))
        lookup = {title.casefold(): index for index, title in enumerate(titles)}
        if requested not in lookup:
            raise ValueError(f"Unknown ST-LRPS page {page!r}; choose from {titles}")
        window._navigate(lookup[requested])
        return

    page_map = {str(key).casefold(): str(key) for key in getattr(window, "_page_map", {})}
    if requested not in page_map:
        raise ValueError(
            f"Unknown Mission Studio page {page!r}; choose from {list(window._page_map)}"
        )
    window._switch_page(page_map[requested])


def _set_visual_state(window, target: str, state: str) -> None:
    if target == "st-lrps":
        header = getattr(window, "_experiment_header", None)
        if header is None:
            return
        header.set_status("IDLE" if state == "idle" else "TRAINING")
        if state == "running":
            header.set_run("phase2-preview")
            header.set_elapsed("00:04:18")
            header._elapsed.setVisible(True)
            header.set_remaining("00:11:42")
        return

    if state == "validating":
        window._set_preflight_state("validating")
    else:
        window._set_run_state(state)
        if state == "running":
            window.progress_bar.show()
            window.progress_bar.setRange(0, 100)
            window.progress_bar.setValue(38)
            window.progress_bar.setFormat("%p%")
            window.progress_bar.setTextVisible(True)
            window.lbl_progress.show()
            window.lbl_progress.setText("Propagating trajectory...")


def capture(
    out_path: Path,
    *,
    delay: float,
    width: int,
    height: int,
    target: str = "mission",
    page: str = "",
    state: str = "idle",
    dialog: str = "none",
) -> Path:
    """Render the main window and save a PNG; returns the output path."""
    from PySide6 import QtCore, QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    if target == "st-lrps":
        from lunaris.surrogate.st_lrps.ui.studio_parts.main_window import MainWindow
        from lunaris.surrogate.st_lrps.ui.studio_parts.qt_common import (
            apply_premium_dark_theme,
        )

        apply_premium_dark_theme(app)
    else:
        from lunaris.ui.app import MainWindow

    window = MainWindow()
    window.resize(width, height)
    _select_page(window, target, page)
    _set_visual_state(window, target, state)
    window.show()

    capture_widget = window
    if dialog != "none":
        if target != "mission":
            raise ValueError("Dialog capture is only available for Mission Studio.")
        if dialog == "gravity":
            from lunaris.ui.pages.force_models_page import GravitySettingsDialog

            capture_widget = GravitySettingsDialog(window, window.gravity_cfg)
            capture_widget.show()

    # Run the event loop briefly so deferred timers (e.g. the 100 ms orbit-preview
    # draw) fire and the theme fully paints before we grab the frame.
    QtCore.QTimer.singleShot(int(max(0.0, delay) * 1000), app.quit)
    app.exec()
    app.processEvents()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pixmap = capture_widget.grab()
    if not pixmap.save(str(out_path)):
        raise RuntimeError(f"Failed to write screenshot to {out_path}")

    if capture_widget is not window:
        capture_widget.deleteLater()
    window.deleteLater()
    return out_path


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        out = capture(
            args.out,
            delay=args.delay,
            width=args.width,
            height=args.height,
            target=args.target,
            page=args.page,
            state=args.state,
            dialog=args.dialog,
        )
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"[capture] Failed: {exc}", file=sys.stderr)
        return 1
    print(
        f"[capture] Saved {out} ({args.target}, page={args.page or 'default'}, "
        f"state={args.state}, dialog={args.dialog}, {args.width}x{args.height}, "
        f"platform={os.environ.get('QT_QPA_PLATFORM', 'default')})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
