# -*- coding: utf-8 -*-
"""Package metadata regression tests."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version


def _expected_version() -> str:
    try:
        return version("lunaris")
    except PackageNotFoundError:
        return "0+unknown"


def test_public_package_versions_share_metadata_source() -> None:
    import lunaris
    import lunaris.core as core
    import lunaris.surrogate.st_lrps as st_lrps

    expected = _expected_version()

    assert lunaris.__version__ == expected
    assert core.__version__ == expected
    assert st_lrps.__version__ == expected
