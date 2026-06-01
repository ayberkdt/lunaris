# -*- coding: utf-8 -*-
"""
Tests for the offline 3D-Moon showcase embed.

The embed is a best-effort visual: it must never prevent the launcher from
opening. These tests verify the embed-dir resolver and that the widget
constructs (and shuts down) safely across the fallback chain — with QtWebEngine
present or simulated absent, and with or without a built static export.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6 import QtWidgets  # noqa: E402

from lunaris.ui.widgets import showcase_embed  # noqa: E402
from lunaris.ui.widgets.showcase_embed import (  # noqa: E402
    ShowcaseEmbedWidget,
    _web_engine_is_safe,
    resolve_web_embed_dir,
)


def _app() -> QtWidgets.QApplication:
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


# ---------------------------------------------------------------------------
# Resolver
# ---------------------------------------------------------------------------

def test_resolver_returns_none_without_a_build(monkeypatch, tmp_path: Path) -> None:
    # No env override and a project root with no out/ build.
    monkeypatch.delenv("LUNARIS_WEB_EMBED_DIR", raising=False)
    monkeypatch.setattr(showcase_embed, "find_project_root", lambda: tmp_path)
    # Also neutralize the packaged-build candidate by pointing __file__-derived
    # path at an empty tree is unnecessary: tmp_path has neither location.
    assert resolve_web_embed_dir() is None


def test_resolver_finds_env_build(monkeypatch, tmp_path: Path) -> None:
    build = tmp_path / "custom_out"
    (build / "embed").mkdir(parents=True)
    (build / "embed" / "index.html").write_text("<html></html>", encoding="utf-8")

    monkeypatch.setenv("LUNARIS_WEB_EMBED_DIR", str(build))
    assert resolve_web_embed_dir() == build


def test_resolver_ignores_env_build_without_index(monkeypatch, tmp_path: Path) -> None:
    build = tmp_path / "empty_out"
    build.mkdir()
    monkeypatch.setenv("LUNARIS_WEB_EMBED_DIR", str(build))
    monkeypatch.setattr(showcase_embed, "find_project_root", lambda: tmp_path)
    assert resolve_web_embed_dir() is None


# ---------------------------------------------------------------------------
# Headless-platform guard
# ---------------------------------------------------------------------------

def test_web_engine_unsafe_on_offscreen(monkeypatch) -> None:
    """QtWebEngine must be skipped on non-GUI Qt platforms to avoid hard crashes."""
    monkeypatch.delenv("LUNARIS_DISABLE_WEB_EMBED", raising=False)
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    assert _web_engine_is_safe() is False

    monkeypatch.setenv("QT_QPA_PLATFORM", "minimal")
    assert _web_engine_is_safe() is False


def test_web_engine_disable_opt_out(monkeypatch) -> None:
    """LUNARIS_DISABLE_WEB_EMBED=1 forces the dark fallback even on a GUI platform."""
    monkeypatch.setenv("QT_QPA_PLATFORM", "windows")
    monkeypatch.setenv("LUNARIS_DISABLE_WEB_EMBED", "1")
    assert _web_engine_is_safe() is False


# ---------------------------------------------------------------------------
# Widget fallback safety
# ---------------------------------------------------------------------------

def test_widget_builds_without_a_build(monkeypatch, tmp_path: Path) -> None:
    """No export build -> widget still constructs and is not live."""
    _app()
    monkeypatch.delenv("LUNARIS_WEB_EMBED_DIR", raising=False)
    monkeypatch.setattr(showcase_embed, "find_project_root", lambda: tmp_path)

    widget = ShowcaseEmbedWidget()
    try:
        assert widget.is_live() is False
        # Public API is a no-op (must not raise) when not live.
        widget.set_texture_mode("gravity")
        widget.set_texture_mode("visual")
        widget.set_orbit_visible(False)
        widget.set_performance_mode("low")
        widget.set_relief(0.0)
        widget.set_relief(1.0)
        widget.set_relief(0.42)
        widget.set_relief("nonsense")  # type: ignore[arg-type]  # must not raise
    finally:
        widget.shutdown()
        widget.shutdown()  # idempotent
        widget.deleteLater()


def test_widget_builds_with_webengine_absent(monkeypatch, tmp_path: Path) -> None:
    """Simulated QtWebEngine-absent -> widget still constructs and is not live."""
    _app()
    monkeypatch.setattr(showcase_embed, "HAS_WEBENGINE", False)
    monkeypatch.delenv("LUNARIS_WEB_EMBED_DIR", raising=False)
    monkeypatch.setattr(showcase_embed, "find_project_root", lambda: tmp_path)

    widget = ShowcaseEmbedWidget()
    try:
        assert widget.is_live() is False
        widget.set_orbit_visible(True)  # no-op, must not raise
    finally:
        widget.shutdown()
        widget.deleteLater()
