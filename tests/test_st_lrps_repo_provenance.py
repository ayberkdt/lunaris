"""The provenance commit comes from the lunaris checkout, never the caller's cwd.

``artifacts.manager.resolve_git_commit`` used to run ``git rev-parse HEAD`` in
the process cwd, so a run started from inside another git checkout recorded
that checkout's commit. ``dataset_contract._repo_commit_sha`` used a
``parents[5]`` offset that is only right for an editable source-tree install.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

import lunaris
from lunaris.common.provenance import package_source_dir, resolve_repo_commit
from lunaris.surrogate.st_lrps.data import dataset_contract


def _git_head(cwd: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=str(cwd), capture_output=True, text=True, check=False
    ).stdout.strip()


@pytest.fixture
def unrelated_repo_cwd(tmp_path, monkeypatch):
    """chdir into a fresh git repo with one commit; return (its HEAD, lunaris HEAD)."""
    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    real_head = _git_head(package_source_dir())
    if not real_head:
        pytest.skip("the lunaris package directory is not inside a git checkout here")
    other = tmp_path / "unrelated_repo"
    other.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=str(other), check=True)
    subprocess.run(
        ["git", "-c", "user.name=test", "-c", "user.email=test@example.invalid",
         "commit", "--allow-empty", "-q", "-m", "x"],
        cwd=str(other),
        check=True,
    )
    other_head = _git_head(other)
    assert other_head and other_head != real_head
    monkeypatch.delenv("GIT_COMMIT", raising=False)
    monkeypatch.chdir(other)
    return other_head, real_head


def test_package_source_dir_is_the_imported_lunaris_package():
    assert package_source_dir() == Path(lunaris.__file__).resolve().parent


def test_repo_commit_ignores_the_callers_cwd(unrelated_repo_cwd):
    _other_head, real_head = unrelated_repo_cwd
    assert resolve_repo_commit() == real_head
    assert dataset_contract._repo_commit_sha() == real_head


def test_manager_resolve_git_commit_ignores_the_callers_cwd(unrelated_repo_cwd):
    pytest.importorskip("torch")
    from lunaris.surrogate.st_lrps.artifacts.manager import resolve_git_commit

    _other_head, real_head = unrelated_repo_cwd
    assert resolve_git_commit() == real_head


def test_git_commit_env_override_wins(monkeypatch):
    monkeypatch.setenv("GIT_COMMIT", "  deadbeef  ")
    assert resolve_repo_commit() == "deadbeef"
    assert dataset_contract._repo_commit_sha() == "deadbeef"
