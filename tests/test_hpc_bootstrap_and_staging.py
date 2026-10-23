"""Fresh-clone HPC setup: ``hpc/bootstrap.sh`` and ``hpc/stage_artifacts.sh``.

- bootstrap never overwrites an existing ``hpc/cluster.env``, and it does not
  call a cluster ready while a file of the data preset is missing.
- stage_artifacts proves every copy with SHA-256 on the receiving side, and it
  refuses to touch an artifact that is already there.

The scripts run against a scratch copy of ``hpc/`` with a stand-in ``ssh`` that
runs the remote command locally, so nothing here needs a cluster or a network.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_HPC = _REPO / "hpc"

pytestmark = pytest.mark.skipif(
    sys.platform == "win32" or shutil.which("bash") is None,
    reason="the HPC scripts run under a POSIX bash",
)


def _bash(script: Path, *args: str, env: dict[str, str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(script), *args], cwd=cwd, env=env, capture_output=True, text=True, timeout=300, check=False,
    )


def _hpc_copy(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    shutil.copytree(_HPC, repo / "hpc", ignore=shutil.ignore_patterns("cluster.env"))
    return repo / "hpc"


def _write_executable(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)


def test_scripts_parse() -> None:
    for name in ("bootstrap.sh", "stage_artifacts.sh"):
        assert subprocess.run(["bash", "-n", str(_HPC / name)], capture_output=True, check=False).returncode == 0, name


def test_bootstrap_names_the_profiles_when_cluster_env_is_missing(tmp_path: Path) -> None:
    hpc = _hpc_copy(tmp_path)
    run = _bash(hpc / "bootstrap.sh", "--dry-run", env={**os.environ, "HOME": str(tmp_path)}, cwd=tmp_path)
    assert run.returncode == 64
    assert "--profile uhem_altay" in run.stderr


def test_bootstrap_dry_run_copies_nothing(tmp_path: Path) -> None:
    hpc = _hpc_copy(tmp_path)
    run = _bash(hpc / "bootstrap.sh", "--dry-run", "--profile", "uhem_altay",
                env={**os.environ, "HOME": str(tmp_path)}, cwd=tmp_path)
    assert run.returncode == 0, run.stdout + run.stderr
    assert not (hpc / "cluster.env").exists()
    assert "+ lunaris-data download --preset full-gravity" in run.stdout


def test_bootstrap_keeps_an_existing_cluster_env(tmp_path: Path) -> None:
    hpc = _hpc_copy(tmp_path)
    own = 'LUNARIS_PARTITION="mine"\n'
    (hpc / "cluster.env").write_text(own, encoding="utf-8")
    run = _bash(hpc / "bootstrap.sh", "--dry-run", "--profile", "uhem_altay",
                env={**os.environ, "HOME": str(tmp_path)}, cwd=tmp_path)
    assert run.returncode == 0, run.stdout + run.stderr
    assert (hpc / "cluster.env").read_text(encoding="utf-8") == own
    assert "existing cluster.env kept" in run.stdout


def test_bootstrap_refuses_a_crlf_cluster_env(tmp_path: Path) -> None:
    hpc = _hpc_copy(tmp_path)
    (hpc / "cluster.env").write_bytes(b'LUNARIS_PARTITION="a100q"\r\n')
    run = _bash(hpc / "bootstrap.sh", "--dry-run", env={**os.environ, "HOME": str(tmp_path)}, cwd=tmp_path)
    assert run.returncode == 1
    assert "CRLF" in run.stdout + run.stderr


def test_bootstrap_is_not_ready_while_preset_data_is_missing(tmp_path: Path) -> None:
    hpc = _hpc_copy(tmp_path)
    venv = tmp_path / "venv"
    bin_dir = venv / "bin"
    bin_dir.mkdir(parents=True)
    # A stand-in venv: this interpreter (it imports lunaris) and a download that
    # fetches nothing, so the preset check must find every file missing.
    (bin_dir / "activate").write_text(f'export PATH="{bin_dir}:$PATH"\n', encoding="utf-8")
    _write_executable(bin_dir / "python", f'#!/usr/bin/env bash\nexec "{sys.executable}" "$@"\n')
    _write_executable(bin_dir / "lunaris-train", "#!/usr/bin/env bash\nexit 0\n")
    _write_executable(bin_dir / "lunaris-data", "#!/usr/bin/env bash\nexit 0\n")
    data = tmp_path / "data"
    (hpc / "cluster.env").write_text(f'LUNARIS_VENV="{venv}"\nLUNARIS_DATA_DIR="{data}"\n', encoding="utf-8")
    run = _bash(hpc / "bootstrap.sh", env={**os.environ, "HOME": str(tmp_path)}, cwd=tmp_path)
    assert run.returncode == 1, run.stdout + run.stderr
    assert "FAIL grail_gravity_jggrx" in run.stdout
    assert "FAIL  data" in run.stdout
    assert "Not ready." in run.stdout


def _fake_ssh(tmp_path: Path, *, corrupt: bool = False) -> dict[str, str]:
    """`ssh <host> <command>` that runs <command> in a local "remote home"."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    home = tmp_path / "remote_home"
    home.mkdir()
    remote_path = os.environ["PATH"]
    if corrupt:
        # The remote tar damages a file after unpacking: sha256sum -c must notice.
        remote_bin = tmp_path / "remote_bin"
        remote_bin.mkdir()
        real_tar = shutil.which("tar")
        _write_executable(
            remote_bin / "tar",
            f'#!/usr/bin/env bash\n"{real_tar}" "$@" || exit $?\n'
            'if [[ "$1" == "-xf" ]]; then find "$4" -name "*.h5" -exec sh -c \'echo x >> "$1"\' _ {} \\;; fi\n',
        )
        remote_path = f"{remote_bin}:{remote_path}"
    _write_executable(
        bin_dir / "ssh",
        f'#!/usr/bin/env bash\ncd "{home}" && PATH="{remote_path}" bash -c "$2"\n',
    )
    return {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"}


def _suite(tmp_path: Path) -> Path:
    suite = tmp_path / "local" / "suite_20260704"
    (suite / "nested").mkdir(parents=True)
    (suite / "train.h5").write_bytes(b"\x89HDF-train")
    (suite / "nested" / "meta.json").write_text("{}", encoding="utf-8")
    return suite


def test_staging_copies_and_checks_on_the_remote_side(tmp_path: Path) -> None:
    env = _fake_ssh(tmp_path)
    suite = _suite(tmp_path)
    run = _bash(_HPC / "stage_artifacts.sh", "me@login", "--suite", f"{suite}/", env=env, cwd=tmp_path)
    assert run.returncode == 0, run.stdout + run.stderr
    dest = tmp_path / "remote_home" / "lunaris_data" / "datasets" / "cloud_suites"
    assert (dest / "suite_20260704" / "train.h5").read_bytes() == b"\x89HDF-train"
    assert (dest / "suite_20260704" / "nested" / "meta.json").exists()
    assert len((dest / "suite_20260704.sha256").read_text(encoding="utf-8").splitlines()) == 2


def test_staging_never_overwrites_a_remote_artifact(tmp_path: Path) -> None:
    env = _fake_ssh(tmp_path)
    suite = _suite(tmp_path)
    existing = tmp_path / "remote_home" / "lunaris_outputs" / "training" / "suite_20260704"
    existing.mkdir(parents=True)
    (existing / "keep.txt").write_text("theirs", encoding="utf-8")
    run = _bash(_HPC / "stage_artifacts.sh", "me@login", "--model", str(suite), env=env, cwd=tmp_path)
    assert run.returncode == 1  # not made by this script: no hash list to vouch for it
    assert "incomplete, unverified" in run.stdout
    assert sorted(p.name for p in existing.iterdir()) == ["keep.txt"]


def test_a_rerun_accepts_a_complete_copy_and_flags_a_partial_one(tmp_path: Path) -> None:
    env = _fake_ssh(tmp_path)
    suite = _suite(tmp_path)
    args = ("me@login", "--suite", str(suite))
    assert _bash(_HPC / "stage_artifacts.sh", *args, env=env, cwd=tmp_path).returncode == 0
    rerun = _bash(_HPC / "stage_artifacts.sh", *args, env=env, cwd=tmp_path)
    assert rerun.returncode == 0 and "not copied again" in rerun.stdout
    remote = tmp_path / "remote_home" / "lunaris_data" / "datasets" / "cloud_suites" / "suite_20260704"
    (suite / "nested" / "new_eval.json").write_text("{}", encoding="utf-8")  # the run changed locally
    changed = _bash(_HPC / "stage_artifacts.sh", *args, env=env, cwd=tmp_path)
    assert changed.returncode == 1 and "not this copy" in changed.stdout
    (suite / "nested" / "new_eval.json").unlink()
    (remote / "train.h5").unlink()  # a transfer cut off half way
    partial = _bash(_HPC / "stage_artifacts.sh", *args, env=env, cwd=tmp_path)
    assert partial.returncode == 1 and "incomplete, unverified" in partial.stdout


def test_staging_reports_a_copy_that_fails_its_hash(tmp_path: Path) -> None:
    env = _fake_ssh(tmp_path, corrupt=True)
    suite = _suite(tmp_path)
    run = _bash(_HPC / "stage_artifacts.sh", "me@login", "--suite", str(suite), env=env, cwd=tmp_path)
    assert run.returncode == 1
    assert "FAIL" in run.stdout


def test_staging_dry_run_connects_to_nothing(tmp_path: Path) -> None:
    suite = _suite(tmp_path)
    env = {**os.environ, "PATH": f"{tmp_path / 'no_ssh_here'}:{os.environ['PATH']}"}
    run = _bash(_HPC / "stage_artifacts.sh", "me@login", "--dry-run", "--suite", str(suite), env=env, cwd=tmp_path)
    assert run.returncode == 0, run.stdout + run.stderr
    assert "DRY" in run.stdout
    assert not (tmp_path / "remote_home").exists()
