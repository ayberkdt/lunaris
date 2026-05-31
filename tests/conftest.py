# tests/conftest.py
# -*- coding: utf-8 -*-
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


def pytest_collection_modifyitems(config, items):
    """Auto-skip ``requires_data`` tests when external data is unavailable."""
    if _lunaris_data_available():
        return
    skip_no_data = pytest.mark.skip(
        reason="external data unavailable (SPICE kernels / gravity coefficients); "
        "set LUNARIS_DATA_DIR or run `lunaris-data download`."
    )
    for item in items:
        if "requires_data" in item.keywords:
            item.add_marker(skip_no_data)
