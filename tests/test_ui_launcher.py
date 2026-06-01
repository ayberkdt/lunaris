# -*- coding: utf-8 -*-
"""
Smoke tests for the Lunaris launcher / welcome hub.

These guard the two contracts that make the hub useful:

1. The launcher module imports cleanly and exposes its public surface.
2. Lazy loading is real — importing the launcher (or the classic propagation
   app) must not drag in the *other* workspace's heavy stack (PyTorch, the
   ST-LRPS training pipeline, or the Studio window). The lazy-import checks run
   in a clean subprocess so they are not polluted by modules other tests import.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")


def _run_isolated(body: str) -> subprocess.CompletedProcess:
    """Run *body* in a fresh interpreter with an offscreen Qt platform."""
    env = dict(os.environ)
    env["QT_QPA_PLATFORM"] = "offscreen"
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(body)],
        capture_output=True,
        text=True,
        env=env,
    )


def test_launcher_module_imports_and_exposes_surface() -> None:
    import lunaris.ui.launcher as launcher

    assert hasattr(launcher, "LauncherWindow")
    assert callable(launcher.main)


def test_launcher_open_flow_shows_overlay_then_hides_launcher() -> None:
    """
    The open flow must:
      1. show an "Opening …" overlay immediately and defer the heavy build
         (so the Moon keeps spinning instead of freezing on click),
      2. NOT tear down the embed (so return is instant),
      3. after the build, hide the launcher (so the Moon is not left behind the
         workspace),
      4. show the launcher again when the workspace closes.
    """
    from PySide6 import QtCore, QtWidgets

    import lunaris.ui.launcher as launcher

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = launcher.LauncherWindow()
    win.show()
    try:
        built = {"count": 0}

        def _fake_builder() -> QtWidgets.QWidget:
            built["count"] += 1
            return QtWidgets.QWidget()

        win._open_workspace("Test", _fake_builder)

        # Immediately after click: overlay is up, build NOT yet run, launcher
        # still visible (Moon spinning behind the overlay).
        assert built["count"] == 0
        assert getattr(win, "_opening_overlay", None) is not None
        assert win.isHidden() is False

        # Pump events past the deferred-build delay until the build completes.
        deadline = QtCore.QElapsedTimer()
        deadline.start()
        while built["count"] == 0 and deadline.elapsed() < 3000:
            app.processEvents(QtCore.QEventLoop.AllEvents, 50)

        assert built["count"] == 1
        assert win._active_workspace is not None
        # Overlay cleared and launcher hidden behind the workspace.
        assert getattr(win, "_opening_overlay", None) is None
        assert win.isHidden() is True

        # Closing the workspace shows the launcher again (instant; embed kept).
        win._on_workspace_closed()
        assert win._active_workspace is None
        assert win.isVisible() is True
    finally:
        win.close()
        win.deleteLater()
    _ = app


def test_importing_launcher_does_not_eagerly_load_workspaces() -> None:
    """Importing the hub must not import either workspace or PyTorch."""
    proc = _run_isolated(
        """
        import sys
        import lunaris.ui.launcher  # noqa: F401

        forbidden = [
            "torch",
            "h5py",
            "lunaris.ui.app",
            "lunaris.surrogate.st_lrps.ui.studio_parts.main_window",
            "lunaris.surrogate.st_lrps.training",
        ]
        leaked = sorted(
            m for m in sys.modules
            if any(m == f or m.startswith(f + ".") or m == f for f in forbidden)
        )
        if leaked:
            print("LEAKED:" + ",".join(leaked))
            raise SystemExit(1)
        raise SystemExit(0)
        """
    )
    assert proc.returncode == 0, f"Launcher import leaked modules:\n{proc.stdout}\n{proc.stderr}"


def test_importing_propagation_app_does_not_load_training_stack() -> None:
    """Importing the classic propagation UI must not pull the ST-LRPS/PyTorch side."""
    proc = _run_isolated(
        """
        import sys
        import lunaris.ui.app  # noqa: F401

        forbidden = [
            "torch",
            "lunaris.surrogate.st_lrps.training",
            "lunaris.surrogate.st_lrps.ui.studio_parts.main_window",
        ]
        leaked = sorted(
            m for m in sys.modules
            if any(m == f or m.startswith(f + ".") for f in forbidden)
        )
        if leaked:
            print("LEAKED:" + ",".join(leaked))
            raise SystemExit(1)
        raise SystemExit(0)
        """
    )
    assert proc.returncode == 0, f"app import leaked training stack:\n{proc.stdout}\n{proc.stderr}"


# ---------------------------------------------------------------------------
# Architectural boundary: core / physics / cli must stay UI-free
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "module",
    [
        "lunaris.core.dynamics",
        "lunaris.physics.spherical_harmonics",
        "lunaris.cli.main",
    ],
)
def test_core_layers_do_not_import_qt(module: str) -> None:
    """Importing a core/physics/cli module must not pull PySide6 or QtWebEngine."""
    proc = _run_isolated(
        f"""
        import sys
        import {module}  # noqa: F401

        forbidden = ["PySide6", "PySide6.QtWebEngineWidgets", "PyQt6"]
        leaked = sorted(
            m for m in sys.modules
            if any(m == f or m.startswith(f + ".") for f in forbidden)
        )
        if leaked:
            print("LEAKED:" + ",".join(leaked))
            raise SystemExit(1)
        raise SystemExit(0)
        """
    )
    assert proc.returncode == 0, f"{module} leaked a Qt/UI dependency:\n{proc.stdout}\n{proc.stderr}"


@pytest.mark.parametrize(
    "module",
    [
        "lunaris.surrogate.st_lrps.training.cli",
        "lunaris.surrogate.st_lrps.evaluation.cli",
    ],
)
def test_training_eval_cli_help_is_headless(module: str) -> None:
    """The training/evaluation CLIs must run --help without importing any UI."""
    env = dict(os.environ)
    env.pop("QT_QPA_PLATFORM", None)
    proc = subprocess.run(
        [sys.executable, "-m", module, "--help"],
        capture_output=True,
        text=True,
        env=env,
    )
    assert proc.returncode == 0, f"{module} --help failed:\n{proc.stderr}"
    assert "usage" in (proc.stdout + proc.stderr).lower()
    # The --help text itself must not have required a GUI toolkit.
    assert "QtWebEngine" not in proc.stderr
