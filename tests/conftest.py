# tests/conftest.py
"""
Shared pytest configuration for the Lunaris test suite.

Primary job: make the suite honest about its dependence on large external
scientific data (SPICE kernels, lunar gravity coefficients). Tests that genuinely
need those files are marked ``@pytest.mark.requires_data``. This hook then:

* runs them normally when the data IS available (e.g. a developer checkout with
  the kernels/gravity model downloaded), and
* skips them cleanly when the data is absent (e.g. CPU-only CI), instead of
  letting them raise ``FileNotFoundError`` mid-test.

CI additionally deselects them via ``-m "not requires_data"``; this hook is the
belt-and-suspenders that keeps an *unfiltered* no-data run green too.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import os
from functools import lru_cache

# Pin the Qt binding for the whole test process BEFORE any Qt-aware library is
# imported. qtawesome / qtpy / pyqtgraph otherwise bind to whichever Qt wrapper
# was imported first, so a test that imports PyQt6 could make icon and theme code
# return PyQt6 objects that then fail against the PySide6 main-app widgets
# (order-dependent ``setIcon`` / ``setFont`` TypeErrors). Lunaris standardizes on
# PySide6; this keeps the binding deterministic regardless of test order.
os.environ.setdefault("QT_API", "pyside6")
os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")

from pathlib import Path

import pytest


def _lunaris_data_available() -> bool:
    """Return True only when both the lunar gravity model and SPICE kernels exist.

    Path resolution honours ``LUNARIS_DATA_DIR`` (and the repo ``data/`` fallback)
    exactly like production code, so this reflects what the tests would actually
    find at runtime.
    """
    try:
        from lunaris.surrogate.st_lrps.data.dataset_parameters import (
            DEFAULT_LUNAR_GRAVITY_PATH,
        )
    except Exception:
        return False

    gravity_ok = Path(DEFAULT_LUNAR_GRAVITY_PATH).is_file()

    kernels_ok = False
    try:
        from lunaris.core.config import KERNEL_DIR

        kernel_dir = Path(KERNEL_DIR)
        kernels_ok = kernel_dir.is_dir() and any(kernel_dir.glob("*.bsp"))
    except Exception:
        kernels_ok = False

    return bool(gravity_ok and kernels_ok)


def _pyshtools_available() -> bool:
    """Return True only when the optional ``pyshtools`` library is importable."""
    try:
        import pyshtools  # noqa: F401
    except Exception:
        return False
    return True


def _module_available(module_name: str) -> bool:
    """Return True when an optional dependency can be imported."""

    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ValueError):
        return False


@lru_cache(maxsize=1)
def _torch_available() -> bool:
    """Return True only when the optional ``torch`` package is functional.

    A broken/partial install can expose a namespace package named ``torch``
    without core symbols such as ``Tensor`` or ``torch.nn``. Treat that the same
    as "not installed" so optional torch tests skip instead of failing during
    collection or first use.
    """

    if not _module_available("torch"):
        return False
    try:
        torch_mod = importlib.import_module("torch")
    except Exception:
        return False
    required_attrs = ("Tensor", "as_tensor", "device", "float32", "float64")
    if not all(hasattr(torch_mod, attr) for attr in required_attrs):
        return False
    return _module_available("torch.nn")


def _h5py_available() -> bool:
    """Return True only when the optional ``h5py`` package is importable."""

    return _module_available("h5py")


def _tudatpy_available() -> bool:
    """Return True only when the optional ``tudatpy`` library is importable."""
    try:
        import tudatpy  # noqa: F401
    except Exception:
        return False
    return True


def _top_level_import_roots(path: Path) -> set[str]:
    """Return module roots imported at test module top level.

    This is used only as a pre-collection guard for optional-dependency tests:
    tests that import torch/h5py before pytest can see their markers otherwise
    fail as collection errors on a core-only install.
    """

    try:
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text, filename=str(path))
    except Exception:
        return set()

    roots: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".", 1)[0])
    return roots


def _source_mentions(path: Path, needles: tuple[str, ...]) -> bool:
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return False
    return any(needle in text for needle in needles)


def pytest_ignore_collect(collection_path: Path, config) -> bool:
    """Pre-collection optional-dependency guard for core-only environments."""

    path = Path(collection_path)
    if path.suffix != ".py" or path.name == "conftest.py":
        return False

    roots = _top_level_import_roots(path)

    if not _torch_available() and (
        "torch" in roots
        or _source_mentions(
            path,
            (
                "import torch",
                "from torch",
                "pytest.mark.requires_torch",
                "lunaris.surrogate.st_lrps.runtime.profiling",
            ),
        )
    ):
        return True

    if not _h5py_available() and (
        "h5py" in roots
        or _source_mentions(
            path,
            (
                "import h5py",
                "from h5py",
                "dataset_pipeline_test_utils",
                "lunaris.surrogate.st_lrps.evaluation.validation_suite",
                "lunaris.surrogate.st_lrps.shared.scaling",
                "lunaris.surrogate.st_lrps.training.config",
            ),
        )
    ):
        return True

    return False


def pytest_collection_modifyitems(config, items):
    """Auto-skip tests when their prerequisites are unavailable."""
    data_available = _lunaris_data_available()
    torch_available = _torch_available()
    h5py_available = _h5py_available()
    pyshtools_available = _pyshtools_available()
    tudatpy_available = _tudatpy_available()
    skip_no_data = pytest.mark.skip(
        reason="external data unavailable (SPICE kernels / gravity coefficients); "
        "set LUNARIS_DATA_DIR or run `lunaris-data download`."
    )
    skip_no_pyshtools = pytest.mark.skip(
        reason="optional dependency 'pyshtools' is not installed."
    )
    skip_no_torch = pytest.mark.skip(
        reason="optional dependency 'torch' is not installed."
    )
    skip_no_h5py = pytest.mark.skip(
        reason="optional dependency 'h5py' is not installed."
    )
    skip_no_tudatpy = pytest.mark.skip(
        reason="optional dependency 'tudatpy' is not installed."
    )
    for item in items:
        if not data_available and "requires_data" in item.keywords:
            item.add_marker(skip_no_data)
        if not torch_available and "requires_torch" in item.keywords:
            item.add_marker(skip_no_torch)
        if not h5py_available and "requires_h5py" in item.keywords:
            item.add_marker(skip_no_h5py)
        if not pyshtools_available and "requires_pyshtools" in item.keywords:
            item.add_marker(skip_no_pyshtools)
        if not tudatpy_available and "requires_tudatpy" in item.keywords:
            item.add_marker(skip_no_tudatpy)
