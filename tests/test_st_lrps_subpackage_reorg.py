# -*- coding: utf-8 -*-
"""
Regression tests for the ST-LRPS Part 2 subpackage reorganization.

Guards:
- the new subpackages import,
- top-level ``import lunaris.surrogate.st_lrps`` stays lightweight (no torch / training engine),
- canonical public imports resolve from their new homes,
- old flat module imports fail (no compatibility wrappers),
- old flat module files are gone,
- the CLI modules run via ``python -m``,
- no source/docs reference the old flat module paths.

Old flat module tokens are built dynamically below so this test file does not
itself trip the "old names absent" repo scan.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# Built dynamically so the literal old paths never appear in this file's source.
_PKG = "st_lrps"
_OLD_LEAVES = (
    "st_lrps_train", "st_lrps_config", "st_lrps_engine", "st_lrps_models",
    "st_lrps_losses", "st_lrps_metrics", "st_lrps_scaling", "st_lrps_artifacts",
    "st_lrps_data", "st_lrps_evaluate", "st_lrps_force_model",
    "run_ablation_matrix", "ui_st_lrps",
)
OLD_FLAT_MODULES = tuple(f"{_PKG}.{leaf}" for leaf in _OLD_LEAVES)
OLD_FLAT_FILES = tuple(f"{_PKG}/{leaf}.py" for leaf in _OLD_LEAVES) + (
    f"{_PKG}/dataset_parameters.py",
    f"{_PKG}/spatial_cloud_parameters.py",
    f"{_PKG}/spatial_cloud_generator.py",
    f"{_PKG}/spatial_cloud_analysis.py",
)

NEW_SUBPACKAGES = (
    "lunaris.surrogate.st_lrps.data", "lunaris.surrogate.st_lrps.training", "lunaris.surrogate.st_lrps.networks", "lunaris.surrogate.st_lrps.artifacts",
    "lunaris.surrogate.st_lrps.evaluation", "lunaris.surrogate.st_lrps.runtime", "lunaris.surrogate.st_lrps.ui", "lunaris.surrogate.st_lrps.shared",
)

SKIP_DIR_PARTS = {".git", ".claude", "__pycache__", ".pytest_cache", ".mypy_cache", "node_modules"}


def _iter_source_files() -> list[Path]:
    files: list[Path] = []
    for path in REPO_ROOT.rglob("*"):
        if path.suffix not in (".py", ".md"):
            continue
        if any(part in SKIP_DIR_PARTS for part in path.relative_to(REPO_ROOT).parts):
            continue
        if path.resolve() == Path(__file__).resolve():
            continue
        files.append(path)
    return files


# ---------------------------------------------------------------------------
# 1) New subpackages import
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("module", ("lunaris.surrogate.st_lrps",) + NEW_SUBPACKAGES)
def test_new_subpackages_import(module: str) -> None:
    assert importlib.import_module(module) is not None


# ---------------------------------------------------------------------------
# 2) Lightweight top-level import
# ---------------------------------------------------------------------------

def test_top_level_import_is_lightweight() -> None:
    code = (
        "import sys, lunaris.surrogate.st_lrps as st_lrps; "
        "heavy = [m for m in ('torch', 'h5py') if m in sys.modules]; "
        "eng = [m for m in sys.modules if m.startswith('lunaris.surrogate.st_lrps.training') "
        "or m.startswith('lunaris.surrogate.st_lrps.networks') or m.startswith('lunaris.surrogate.st_lrps.runtime')]; "
        "print(repr(heavy)); print(repr(eng)); print(st_lrps.__version__)"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], cwd=str(REPO_ROOT), capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stderr
    lines = proc.stdout.strip().splitlines()
    assert lines[0] == "[]", f"import lunaris.surrogate.st_lrps pulled heavy modules: {lines[0]}"
    assert lines[1] == "[]", f"import lunaris.surrogate.st_lrps imported heavy submodules: {lines[1]}"
    assert lines[2]


# ---------------------------------------------------------------------------
# 3) Canonical public imports
# ---------------------------------------------------------------------------

def test_canonical_public_imports() -> None:
    from lunaris.surrogate.st_lrps.training.config import TrainConfig
    from lunaris.surrogate.st_lrps.training.engine import STLRPSTrainer
    from lunaris.surrogate.st_lrps.training.losses import SobolevLoss
    from lunaris.surrogate.st_lrps.training.metrics import normalize_best_metric
    from lunaris.surrogate.st_lrps.networks.models import PhysicsNet, build_model_from_config
    from lunaris.surrogate.st_lrps.shared.scaling import ScalerPack
    from lunaris.surrogate.st_lrps.artifacts.manager import make_run_layout
    from lunaris.surrogate.st_lrps.data.datasets import DatasetMeta
    from lunaris.surrogate.st_lrps.runtime.force_model import load_surrogate_force_model

    for obj in (
        TrainConfig, STLRPSTrainer, SobolevLoss, normalize_best_metric,
        PhysicsNet, build_model_from_config, ScalerPack, make_run_layout,
        DatasetMeta, load_surrogate_force_model,
    ):
        assert obj is not None


def test_runtime_does_not_depend_on_training() -> None:
    """The runtime inference boundary must not pull the training package."""
    code = (
        "import sys; from lunaris.surrogate.st_lrps.runtime.force_model import load_surrogate_force_model; "
        "bad = [m for m in sys.modules if m.startswith('lunaris.surrogate.st_lrps.training')]; "
        "print(repr(bad))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], cwd=str(REPO_ROOT), capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "[]", f"runtime leaked training deps: {proc.stdout!r}"


# ---------------------------------------------------------------------------
# 4) Old flat imports fail
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("module", OLD_FLAT_MODULES)
def test_old_flat_imports_fail(module: str) -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module)


# ---------------------------------------------------------------------------
# 5) Old flat files absent
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rel_path", OLD_FLAT_FILES)
def test_old_flat_files_absent(rel_path: str) -> None:
    assert not (REPO_ROOT / rel_path).exists(), f"stale flat module still present: {rel_path}"


# ---------------------------------------------------------------------------
# 6) CLI help smoke tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "module",
    ["lunaris.surrogate.st_lrps.training.cli", "lunaris.surrogate.st_lrps.evaluation.cli", "lunaris.surrogate.st_lrps.evaluation.ablation"],
)
def test_cli_help_exits_zero(module: str) -> None:
    pytest.importorskip("torch")
    proc = subprocess.run(
        [sys.executable, "-m", module, "--help"],
        cwd=str(REPO_ROOT), capture_output=True, text=True,
    )
    assert proc.returncode == 0, f"{module} --help failed:\n{proc.stderr}"
    assert "usage" in (proc.stdout + proc.stderr).lower()


# ---------------------------------------------------------------------------
# 7) Repo search guard
# ---------------------------------------------------------------------------

def test_no_old_flat_module_paths_in_sources() -> None:
    blocked = set(OLD_FLAT_MODULES)
    blocked.add(f"{_PKG}/ui_st_lrps.py")
    offenders: dict[str, list[str]] = {}
    for path in _iter_source_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        hits = [tok for tok in blocked if tok in text]
        if hits:
            offenders[str(path.relative_to(REPO_ROOT))] = hits
    assert not offenders, f"Old flat module paths still referenced: {offenders}"
