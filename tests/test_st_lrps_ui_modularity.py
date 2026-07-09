"""Phase 2.5 contracts for the ST-LRPS Studio internal module boundary."""

from __future__ import annotations

import pytest

try:
    import torch
    import torch.nn
    _ = torch.cuda
    _ = getattr(torch, "compile", None)
except (ImportError, AttributeError, ModuleNotFoundError):
    pytest.skip("PyTorch not installed or missing required attributes like cuda/compile", allow_module_level=True)


import pytest

pytest.importorskip('PySide6.QtWidgets')


import ast
from pathlib import Path

from lunaris.surrogate.st_lrps.ui.studio_parts.dataset_introspection import (
    inspect_h5_metadata,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
STUDIO_PARTS = REPO_ROOT / "src/lunaris/surrogate/st_lrps/ui/studio_parts"


def test_non_data_pages_do_not_wildcard_import_the_data_workspace() -> None:
    offenders: list[str] = []
    for name in (
        "training_pages.py",
        "runtime_pages.py",
        "evaluation_pages.py",
        "main_window.py",
    ):
        path = STUDIO_PARTS / name
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or node.module != "data_pages":
                continue
            if any(alias.name == "*" for alias in node.names):
                offenders.append(f"{name}:{node.lineno}")
    assert offenders == []


def test_dataset_introspection_surface_is_qt_independent() -> None:
    path = STUDIO_PARTS / "dataset_introspection.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imported_roots = {
        alias.name.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported_roots.update(
        str(node.module).split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    )
    assert imported_roots.isdisjoint({"PySide6", "PyQt6", "pyqtgraph"})


def test_dataset_introspection_missing_file_is_non_fatal(tmp_path: Path) -> None:
    assert inspect_h5_metadata(str(tmp_path / "missing.h5")) is None
