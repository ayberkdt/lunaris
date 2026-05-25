# -*- coding: utf-8 -*-
"""
Regression tests for the ST-LRPS package rename + artifact cleanup (Part 1).

Guards:
- ``import st_lrps`` stays lightweight (no torch / training side-effects),
- the old ``surrogate_gravity_model`` package path is gone everywhere,
- generated run/eval artifacts (gececi_kod, eval CSVs, manifests, ...) are not
  committed back into the source tree,
- the ST-LRPS CLI modules remain runnable via ``python -m``.

The old package token is built dynamically below so this test file does not
itself trip the "old name absent" scan.
"""

from __future__ import annotations

import fnmatch
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# Built dynamically so the literal does not appear in this file's own source.
OLD_PKG = "surrogate" + "_gravity_model"
# The CLI argument dest (``--surrogate-gravity-model-dir``) legitimately contains
# the old token as a substring; it is a public CLI contract, not the package.
ALLOWED_CLI_ARG = OLD_PKG + "_dir"

# Directories we must never descend into while scanning the source tree.
SKIP_DIR_PARTS = {".git", ".claude", "__pycache__", ".pytest_cache", ".mypy_cache", "node_modules"}

# Generated artifact basenames that must never be committed into the tree.
ARTIFACT_NAMES = {
    "history.jsonl",
    "run_manifest.json",
    "command.txt",
    "scaler.json",
    "eval_report.json",
    "eval_manifest.json",
    "evaluate_metrics.json",
    "evaluate_summary.txt",
    "metrics_summary.csv",
    "ood_metrics.csv",
    "topk_worst.csv",
    "altitude_binned_metrics.csv",
    "angular_error_by_altitude.csv",
    "angular_error_by_accel_norm.csv",
    "acceleration_decomposition.csv",
}
ARTIFACT_GLOBS = ("spatial_rmse_*.csv", "spatial_mape_*.csv")
# Tiny intentional fixtures are allowed only under these roots.
FIXTURE_PREFIXES = ("tests/fixtures/", "examples/st_lrps_minimal_artifact/")


def _iter_source_files() -> list[Path]:
    files: list[Path] = []
    for path in REPO_ROOT.rglob("*"):
        if path.suffix not in (".py", ".md"):
            continue
        if any(part in SKIP_DIR_PARTS for part in path.relative_to(REPO_ROOT).parts):
            continue
        if path.resolve() == Path(__file__).resolve():
            continue  # skip this test (it references the token by construction)
        files.append(path)
    return files


def _tracked_files(pathspec: str) -> list[str]:
    proc = subprocess.run(
        ["git", "ls-files", pathspec],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        pytest.skip(f"git ls-files unavailable: {proc.stderr}")
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# 1) Import safety
# ---------------------------------------------------------------------------

def test_import_st_lrps_is_lightweight() -> None:
    """Importing the package must not pull torch/h5py or training modules."""
    code = (
        "import sys, st_lrps; "
        "heavy = [m for m in ('torch', 'h5py') if m in sys.modules]; "
        "sub = [m for m in sys.modules if m.startswith('st_lrps.')]; "
        "print(repr(heavy)); print(repr(sub)); print(st_lrps.__version__)"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    lines = proc.stdout.strip().splitlines()
    assert lines[0] == "[]", f"import st_lrps pulled heavy modules: {lines[0]}"
    assert lines[1] == "[]", f"import st_lrps imported submodules: {lines[1]}"
    assert lines[2]  # __version__ is truthy


def test_st_lrps_all_is_minimal_and_truthful() -> None:
    import st_lrps

    assert st_lrps.__all__ == ["__version__"]
    for name in st_lrps.__all__:
        assert hasattr(st_lrps, name)
    assert isinstance(st_lrps.__version__, str) and st_lrps.__version__


# ---------------------------------------------------------------------------
# 2) Old package path removed
# ---------------------------------------------------------------------------

def test_old_package_directory_is_gone() -> None:
    assert not (REPO_ROOT / OLD_PKG).exists(), f"{OLD_PKG}/ directory must not exist"
    assert (REPO_ROOT / "st_lrps").is_dir(), "st_lrps/ package must exist"
    assert (REPO_ROOT / "st_lrps" / "__init__.py").is_file()


def test_old_package_name_absent_from_sources() -> None:
    offenders: list[str] = []
    for path in _iter_source_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        # The CLI-arg dest is an allowed contract; strip it before scanning.
        residual = text.replace(ALLOWED_CLI_ARG, "")
        if OLD_PKG in residual:
            offenders.append(str(path.relative_to(REPO_ROOT)))
    assert not offenders, f"Old package token still present in: {offenders}"


# ---------------------------------------------------------------------------
# 3) Generated artifact cleanup
# ---------------------------------------------------------------------------

def test_no_gececi_kod_directory_anywhere() -> None:
    offenders: list[str] = []
    for path in REPO_ROOT.rglob("gececi_kod"):
        rel = path.relative_to(REPO_ROOT)
        if any(part in SKIP_DIR_PARTS for part in rel.parts):
            continue
        offenders.append(str(rel))
    assert not offenders, f"gececi_kod directory must not exist: {offenders}"


def test_no_committed_generated_artifacts_under_st_lrps() -> None:
    offenders: list[str] = []
    for rel_path in _tracked_files("st_lrps"):
        norm = rel_path.replace("\\", "/")
        if any(norm.startswith(prefix) for prefix in FIXTURE_PREFIXES):
            continue
        name = norm.rsplit("/", 1)[-1]
        if "gececi_kod" in norm.split("/"):
            offenders.append(rel_path)
        elif name in ARTIFACT_NAMES:
            offenders.append(rel_path)
        elif any(fnmatch.fnmatch(name, g) for g in ARTIFACT_GLOBS):
            offenders.append(rel_path)
    assert not offenders, f"Generated artifacts committed under st_lrps/: {offenders}"


# ---------------------------------------------------------------------------
# 4) Module help smoke tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("module", ["st_lrps.st_lrps_train", "st_lrps.st_lrps_evaluate"])
def test_module_help_exits_zero(module: str) -> None:
    pytest.importorskip("torch")
    proc = subprocess.run(
        [sys.executable, "-m", module, "--help"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"{module} --help failed:\n{proc.stderr}"
    assert "usage" in (proc.stdout + proc.stderr).lower()


# ---------------------------------------------------------------------------
# 5) UI command path references the new package
# ---------------------------------------------------------------------------

def test_ui_surrogate_studio_uses_new_package_path() -> None:
    src = (REPO_ROOT / "ui_parts" / "surrogate_studio_page.py").read_text(encoding="utf-8")
    assert "st_lrps/st_lrps_train.py" in src
    assert OLD_PKG not in src.replace(ALLOWED_CLI_ARG, "")
