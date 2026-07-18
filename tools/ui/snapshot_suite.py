#!/usr/bin/env python3
"""
Snapshot-diff harness for the Lunaris Mission Studio UI.

Captures a deterministic set of pages to PNGs and compares a *current* run
against a stored *baseline*, reporting a per-page changed-pixel ratio plus a
diff-map PNG that highlights the pixels that moved. This is the safety net for
mechanical visual changes (spacing normalisation, component migrations): make a
change, re-capture, and eyeball only the pages whose diff exceeds a threshold.

Design notes
------------
* Reuses :func:`tools.ui.capture_main_window.capture` so there is exactly one
  code path that renders the window (no second capture implementation to drift).
* Comparison is pure NumPy over ``QImage`` RGB data — no extra dependency.
* The 3D orbit preview renders blank offscreen (pyqtgraph OpenGL has no
  framebuffer without a display), so its pixels are not a reliable diff signal;
  the Orbit page is captured for chrome coverage but excluded from the
  pass/fail gate by default (``--skip-gl-pages``, on by default).
* Runs headless (``QT_QPA_PLATFORM=offscreen``) like the capture tool.

Usage
-----
    # Write/refresh the baseline set (do this on a known-good tree):
    python tools/ui/snapshot_suite.py --baseline

    # Capture current and diff against the baseline:
    python tools/ui/snapshot_suite.py --compare

    # Determinism check: capture the gate pages twice, assert diff == 0:
    python tools/ui/snapshot_suite.py --self-test

Exit codes: 0 = within threshold / clean, 1 = a page exceeded the threshold or a
baseline was missing, 2 = bad invocation.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

REPO_ROOT = Path(os.path.abspath(__file__)).parents[2]
if str(REPO_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "tools"))

SNAP_ROOT = REPO_ROOT / "outputs" / "ui" / "snapshots"

# Mission-studio pages captured in a fixed order. ``gl`` marks pages whose main
# content is the OpenGL preview (blank/unreliable offscreen) — captured for
# chrome coverage but kept out of the default gate.
PAGES: tuple[tuple[str, bool], ...] = (
    ("Orbit", True),
    ("Forces", False),
    ("Propagation", False),
    ("Output", False),
    ("Telemetry", False),
    ("Data", False),
    ("BatchPropagation", False),
    ("FrozenSearch", False),
)

ST_LRPS_PAGES: tuple[tuple[str, str], ...] = (
    ("Data", "idle"),
    ("Training Setup", "idle"),
    ("Training Monitor", "running"),
)

# A page counts as "changed" when more than this fraction of pixels differ.
# Offscreen 2D rendering is deterministic on a fixed machine, so the intended
# gate is effectively 0; the small floor absorbs any single-pixel antialias
# jitter without hiding a real layout shift.
DEFAULT_THRESHOLD = 0.001


@dataclass
class PageDiff:
    page: str
    ratio: float
    changed: int
    total: int
    note: str = ""


@dataclass(frozen=True)
class SnapshotSpec:
    target: str
    page: str
    state: str = "idle"
    is_gl: bool = False

    @property
    def key(self) -> str:
        return f"{self.target}/{self.page}"

    @property
    def relative_path(self) -> Path:
        directory = "mission" if self.target == "mission" else "st_lrps"
        return Path(directory) / f"{self.page}.png"


SNAPSHOTS: tuple[SnapshotSpec, ...] = (
    *(SnapshotSpec("mission", page, is_gl=is_gl) for page, is_gl in PAGES),
    *(SnapshotSpec("st-lrps", page, state=state) for page, state in ST_LRPS_PAGES),
)


def _qimage_to_array(path: Path):
    """Load *path* as an ``(H, W, 3)`` uint8 NumPy array (RGB, no alpha)."""
    import numpy as np
    from PySide6.QtGui import QImage

    img = QImage(str(path))
    if img.isNull():
        raise FileNotFoundError(f"could not read image: {path}")
    img = img.convertToFormat(QImage.Format.Format_RGB32)
    width, height = img.width(), img.height()
    bytes_per_line = img.bytesPerLine()
    buffer = img.constBits()
    # PySide6 returns a sized memoryview for constBits(); reshape by the true
    # stride (bytesPerLine may be padded), then crop to width*4.
    raw = np.frombuffer(memoryview(buffer), dtype=np.uint8, count=bytes_per_line * height)
    raw = raw.reshape((height, bytes_per_line))[:, : width * 4].reshape((height, width, 4))
    # Format_RGB32 is 0xffRRGGBB little-endian -> byte order B,G,R,A.
    return raw[:, :, 2::-1].copy()


def _diff_pages(baseline: Path, current: Path, diff_out: Path) -> PageDiff:
    import numpy as np
    from PySide6.QtGui import QImage

    base = _qimage_to_array(baseline)
    cur = _qimage_to_array(current)
    if base.shape != cur.shape:
        return PageDiff(
            current.stem, 1.0, -1, -1,
            note=f"size changed {base.shape[1]}x{base.shape[0]} -> "
            f"{cur.shape[1]}x{cur.shape[0]}",
        )

    changed_mask = np.any(base != cur, axis=2)
    changed = int(changed_mask.sum())
    total = int(changed_mask.size)
    ratio = changed / total if total else 0.0

    if changed:
        # Dim the unchanged base to grey and paint changed pixels in warning
        # amber so a reviewer sees exactly what moved. Built as a packed RGB32
        # buffer and wrapped in a QImage in one shot (a per-pixel setPixel loop
        # over ~1M px is far too slow).
        height, width = changed_mask.shape
        grey = (base.mean(axis=2) * 0.35).astype(np.uint8)
        flat = np.stack([grey, grey, grey], axis=2)
        flat[changed_mask] = (0xF5, 0xB4, 0x3C)  # R, G, B
        packed = (
            np.uint32(0xFF000000)
            | (flat[..., 0].astype(np.uint32) << 16)
            | (flat[..., 1].astype(np.uint32) << 8)
            | flat[..., 2].astype(np.uint32)
        ).astype("<u4")
        buffer = packed.tobytes()
        out = QImage(buffer, width, height, QImage.Format.Format_RGB32)
        diff_out.parent.mkdir(parents=True, exist_ok=True)
        # copy() detaches from the soon-to-be-freed Python buffer before save.
        out.copy().save(str(diff_out))

    return PageDiff(current.stem, ratio, changed, total)


def _capture_set(dest: Path, *, delay: float, width: int, height: int) -> list[Path]:
    """Capture every page into *dest*; returns the written paths in order."""
    from ui.capture_main_window import capture  # type: ignore

    dest.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for spec in SNAPSHOTS:
        out = dest / spec.relative_path
        capture(
            out, delay=delay, width=width, height=height,
            target=spec.target, page=spec.page, state=spec.state, dialog="none",
        )
        written.append(out)
    return written


def _gate_pages(include_gl: bool) -> set[str]:
    return {spec.key for spec in SNAPSHOTS if include_gl or not spec.is_gl}


def run_baseline(delay: float, width: int, height: int) -> int:
    dest = SNAP_ROOT / "baseline"
    written = _capture_set(dest, delay=delay, width=width, height=height)
    print(f"[snapshot] wrote {len(written)} baseline page(s) to {dest}")
    return 0


def run_compare(
    delay: float, width: int, height: int, threshold: float, include_gl: bool
) -> int:
    baseline_dir = SNAP_ROOT / "baseline"
    if not baseline_dir.is_dir():
        print(
            "[snapshot] no baseline found; run --baseline on a known-good tree first.",
            file=sys.stderr,
        )
        return 1

    current_dir = SNAP_ROOT / "current"
    diff_dir = SNAP_ROOT / "diff"
    _capture_set(current_dir, delay=delay, width=width, height=height)

    gate = _gate_pages(include_gl)
    failures = 0
    print(f"[snapshot] comparing (threshold {threshold:.3%}, "
          f"{'incl' if include_gl else 'excl'} GL pages)")
    for spec in SNAPSHOTS:
        base_png = baseline_dir / spec.relative_path
        cur_png = current_dir / spec.relative_path
        if not base_png.exists():
            print(f"  MISSING baseline  {spec.key}")
            failures += 1
            continue
        result = _diff_pages(
            base_png,
            cur_png,
            diff_dir / spec.relative_path.with_name(f"{spec.page}_diff.png"),
        )
        gated = spec.key in gate
        flag = "gl-skip" if not gated else ("FAIL" if result.ratio > threshold else "ok")
        detail = result.note or f"{result.changed}/{result.total} px"
        print(f"  {flag:>7}  {spec.key:<30} {result.ratio:8.4%}  {detail}")
        if gated and result.ratio > threshold:
            failures += 1
    if failures:
        print(f"[snapshot] {failures} page(s) exceeded threshold; inspect "
              f"{diff_dir}")
        return 1
    print("[snapshot] all gated pages within threshold")
    return 0


def run_self_test(delay: float, width: int, height: int) -> int:
    """Capture the gate pages twice and assert the two runs are identical."""
    a_dir = SNAP_ROOT / "selftest_a"
    b_dir = SNAP_ROOT / "selftest_b"
    diff_dir = SNAP_ROOT / "selftest_diff"
    _capture_set(a_dir, delay=delay, width=width, height=height)
    _capture_set(b_dir, delay=delay, width=width, height=height)

    gate = _gate_pages(include_gl=False)
    worst = 0.0
    nondeterministic = 0
    print("[snapshot] determinism self-test (2 runs, gate pages)")
    for spec in SNAPSHOTS:
        if spec.key not in gate:
            continue
        result = _diff_pages(
            a_dir / spec.relative_path,
            b_dir / spec.relative_path,
            diff_dir / spec.relative_path.with_name(f"{spec.page}_diff.png"),
        )
        worst = max(worst, result.ratio)
        status = "stable" if result.ratio == 0.0 else "DRIFT"
        if result.ratio:
            nondeterministic += 1
        print(f"  {status:>7}  {spec.key:<30} {result.ratio:8.4%}")
    if nondeterministic:
        print(f"[snapshot] {nondeterministic} page(s) render non-deterministically "
              f"(worst {worst:.4%}); use a diff threshold >= {worst:.4%} in --compare")
        return 1
    print("[snapshot] offscreen 2D rendering is deterministic (diff 0); "
          "threshold 0 is safe")
    return 0


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--baseline", action="store_true",
                      help="Capture the baseline page set.")
    mode.add_argument("--compare", action="store_true",
                      help="Capture current pages and diff against the baseline.")
    mode.add_argument("--self-test", action="store_true",
                      help="Capture the gate pages twice and check determinism.")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help=f"Changed-pixel ratio gate (default: {DEFAULT_THRESHOLD}).")
    parser.add_argument("--include-gl-pages", action="store_true",
                        help="Include OpenGL-preview pages in the pass/fail gate.")
    parser.add_argument("--delay", type=float, default=0.8)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=860)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.baseline:
        return run_baseline(args.delay, args.width, args.height)
    if args.self_test:
        return run_self_test(args.delay, args.width, args.height)
    return run_compare(args.delay, args.width, args.height, args.threshold,
                       args.include_gl_pages)


if __name__ == "__main__":
    raise SystemExit(main())
