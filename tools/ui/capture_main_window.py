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

Output (default): ``outputs/ui/main_window_lunar_graphite.png``
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Default to headless rendering unless the caller asked for a real display.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_ROOT / "outputs" / "ui" / "main_window_lunar_graphite.png"


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
    return parser.parse_args(argv)


def capture(out_path: Path, *, delay: float, width: int, height: int) -> Path:
    """Render the main window and save a PNG; returns the output path."""
    from PySide6 import QtCore, QtWidgets

    from lunaris.ui.app import MainWindow

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

    window = MainWindow()
    window.resize(width, height)
    window.show()

    # Run the event loop briefly so deferred timers (e.g. the 100 ms orbit-preview
    # draw) fire and the theme fully paints before we grab the frame.
    QtCore.QTimer.singleShot(int(max(0.0, delay) * 1000), app.quit)
    app.exec()
    app.processEvents()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pixmap = window.grab()
    if not pixmap.save(str(out_path)):
        raise RuntimeError(f"Failed to write screenshot to {out_path}")

    window.deleteLater()
    return out_path


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        out = capture(
            args.out, delay=args.delay, width=args.width, height=args.height
        )
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"[capture] Failed: {exc}", file=sys.stderr)
        return 1
    print(f"[capture] Saved {out} ({args.width}x{args.height}, "
          f"platform={os.environ.get('QT_QPA_PLATFORM', 'default')})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
