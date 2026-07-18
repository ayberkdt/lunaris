"""Unit tests for the UI snapshot-diff harness (tools/ui/snapshot_suite.py).

These exercise the comparison core with tiny synthetic PNGs so they run in
milliseconds — the expensive full-page capture path is covered by the manual
``--self-test`` / ``--compare`` invocations, not here.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

pytest.importorskip("PySide6.QtGui")
import numpy as np  # noqa: E402
from PySide6.QtGui import QImage  # noqa: E402


def _load_suite():
    """Import the harness module by path (it lives under tools/, not a package)."""
    if str(REPO_ROOT / "tools") not in sys.path:
        sys.path.insert(0, str(REPO_ROOT / "tools"))
    spec = importlib.util.spec_from_file_location(
        "lunaris_snapshot_suite", REPO_ROOT / "tools" / "ui" / "snapshot_suite.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    # Register before exec so the @dataclass in the module can resolve its own
    # __module__ via sys.modules during class creation.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


suite = _load_suite()


def _write_png(path: Path, rgb: tuple[int, int, int], size: int = 16) -> None:
    img = QImage(size, size, QImage.Format.Format_RGB32)
    img.fill(0xFF000000 | (rgb[0] << 16) | (rgb[1] << 8) | rgb[2])
    assert img.save(str(path))


def test_array_roundtrip_reads_rgb(tmp_path: Path) -> None:
    png = tmp_path / "solid.png"
    _write_png(png, (0x6A, 0xA9, 0xFF))
    arr = suite._qimage_to_array(png)
    assert arr.shape == (16, 16, 3)
    # Every pixel is the fill color in R, G, B order.
    assert tuple(int(c) for c in arr[0, 0]) == (0x6A, 0xA9, 0xFF)


def test_identical_images_have_zero_diff(tmp_path: Path) -> None:
    a, b = tmp_path / "a.png", tmp_path / "b.png"
    _write_png(a, (20, 30, 40))
    _write_png(b, (20, 30, 40))
    result = suite._diff_pages(a, b, tmp_path / "d.png")
    assert result.ratio == 0.0
    assert result.changed == 0
    assert not (tmp_path / "d.png").exists()  # no diff map when nothing changed


def test_changed_pixels_are_counted_and_mapped(tmp_path: Path) -> None:
    a, b = tmp_path / "a.png", tmp_path / "b.png"
    _write_png(a, (0, 0, 0), size=10)
    # Flip a 10x10 image entirely: 100/100 pixels differ.
    _write_png(b, (255, 255, 255), size=10)
    diff = tmp_path / "d.png"
    result = suite._diff_pages(a, b, diff)
    assert result.changed == 100
    assert result.total == 100
    assert result.ratio == 1.0
    assert diff.exists()
    # The diff map paints changed pixels amber (F5B43C).
    mapped = suite._qimage_to_array(diff)
    assert tuple(int(c) for c in mapped[0, 0]) == (0xF5, 0xB4, 0x3C)


def test_size_change_is_full_diff(tmp_path: Path) -> None:
    a, b = tmp_path / "a.png", tmp_path / "b.png"
    _write_png(a, (10, 10, 10), size=8)
    _write_png(b, (10, 10, 10), size=12)
    result = suite._diff_pages(a, b, tmp_path / "d.png")
    assert result.ratio == 1.0
    assert "size changed" in result.note


def test_partial_change_ratio(tmp_path: Path) -> None:
    a, b = tmp_path / "a.png", tmp_path / "b.png"
    _write_png(a, (0, 0, 0), size=10)
    # Load a, change one row (10 px), save as b.
    arr = suite._qimage_to_array(a).copy()
    arr[0, :] = (255, 0, 0)
    packed = (
        np.uint32(0xFF000000)
        | (arr[..., 0].astype(np.uint32) << 16)
        | (arr[..., 1].astype(np.uint32) << 8)
        | arr[..., 2].astype(np.uint32)
    ).astype("<u4")
    img = QImage(packed.tobytes(), 10, 10, QImage.Format.Format_RGB32)
    img.copy().save(str(b))
    result = suite._diff_pages(a, b, tmp_path / "d.png")
    assert result.changed == 10
    assert result.ratio == pytest.approx(0.1)


def test_gate_pages_excludes_gl_by_default() -> None:
    gated = suite._gate_pages(include_gl=False)
    assert "mission/Orbit" not in gated  # GL-preview page excluded from the gate
    assert "mission/Forces" in gated
    with_gl = suite._gate_pages(include_gl=True)
    assert "mission/Orbit" in with_gl
    assert with_gl >= gated
