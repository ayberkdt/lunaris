"""The frozen-orbit search records the package checkout's commit, not the cwd's.

Regression test: ``_git_commit`` ran ``git rev-parse HEAD`` in the process's
working directory, so a search launched from inside another git repository
recorded that repository's commit in its manifest.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

import lunaris.analysis.frozen.search as search_mod

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=True
    ).stdout.strip()


def test_git_commit_ignores_the_working_directory(tmp_path: Path, monkeypatch) -> None:
    package_dir = Path(search_mod.__file__).resolve().parent
    probe = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=package_dir, capture_output=True, text=True, check=False
    )
    if probe.returncode != 0:
        pytest.skip("lunaris is not running from a git checkout")
    expected = probe.stdout.strip()

    other = tmp_path / "other_repo"
    other.mkdir()
    _git(other, "init", "-q")
    _git(other, "-c", "user.email=t@example.invalid", "-c", "user.name=t",
         "commit", "-q", "--allow-empty", "-m", "unrelated")
    other_head = _git(other, "rev-parse", "HEAD")
    assert other_head != expected

    monkeypatch.chdir(other)
    assert search_mod._git_commit() == expected
