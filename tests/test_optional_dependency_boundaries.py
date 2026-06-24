"""Import-time contracts for optional Lunaris dependencies."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
OPTIONAL_ROOTS = ("torch", "h5py", "PySide6", "PyQt6", "spiceypy")


def _blocked_import_script(body: str) -> str:
    prefix = textwrap.dedent(
        f"""\
        import importlib.abc
        import sys

        blocked = {set(OPTIONAL_ROOTS)!r}

        class BlockOptional(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                root = fullname.split(".", 1)[0]
                if root in blocked:
                    exc = ModuleNotFoundError(f"blocked optional dependency: {{fullname}}")
                    exc.name = fullname
                    raise exc
                return None

        sys.meta_path.insert(0, BlockOptional())
        """
    )
    return prefix + textwrap.dedent(body)


def _run_blocked(body: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", _blocked_import_script(body)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )


def test_bare_lunaris_import_does_not_load_optional_dependencies() -> None:
    proc = _run_blocked(
        """
import lunaris
loaded = sorted(root for root in blocked if root in sys.modules)
assert loaded == [], loaded
"""
    )
    assert proc.returncode == 0, proc.stderr


def test_lunaris_api_import_does_not_load_optional_dependencies() -> None:
    proc = _run_blocked(
        """
import lunaris.api
loaded = sorted(root for root in blocked if root in sys.modules)
assert loaded == [], loaded
"""
    )
    assert proc.returncode == 0, proc.stderr


def test_retired_physics_surrogate_module_does_not_load_optional_dependencies() -> None:
    proc = _run_blocked(
        """
import importlib
legacy = importlib.import_module("lunaris.physics.surrogate_gravity")
assert legacy.__all__ == (), legacy.__all__
loaded = sorted(root for root in blocked if root in sys.modules)
assert loaded == [], loaded
try:
    getattr(legacy, "SurrogateGravityModel")
except AttributeError as exc:
    assert "lunaris.surrogate.runtime_adapter" in str(exc), exc
else:
    raise AssertionError("legacy physics surrogate path unexpectedly re-exported adapter")
"""
    )
    assert proc.returncode == 0, proc.stderr


def _console_scripts() -> dict[str, str]:
    scripts: dict[str, str] = {}
    in_scripts = False
    for raw_line in (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line == "[project.scripts]":
            in_scripts = True
            continue
        if in_scripts and line.startswith("["):
            break
        if in_scripts and line and not line.startswith("#"):
            name, target = line.split("=", 1)
            scripts[name.strip()] = target.strip().strip('"')
    return scripts


@pytest.mark.parametrize(("command", "target"), sorted(_console_scripts().items()))
def test_every_console_script_help_survives_without_optional_dependencies(
    command: str,
    target: str,
) -> None:
    module_name, callable_name = target.split(":", 1)
    body = f"""
import importlib
sys.argv = [{command!r}, "--help"]
target = getattr(importlib.import_module({module_name!r}), {callable_name!r})
try:
    result = target()
except SystemExit as exc:
    code = int(exc.code or 0)
else:
    code = int(result or 0)
raise SystemExit(code)
"""
    proc = _run_blocked(body)
    assert proc.returncode == 0, (
        f"{command} --help imported an optional dependency or failed.\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
