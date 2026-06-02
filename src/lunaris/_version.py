"""Package version helpers for Lunaris."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

_PACKAGE_NAME = "lunaris"
_FALLBACK_VERSION = "0+unknown"


def get_version() -> str:
    """Return the installed package version without importing heavy modules."""
    try:
        return version(_PACKAGE_NAME)
    except PackageNotFoundError:
        return _FALLBACK_VERSION


__version__ = get_version()

__all__ = ["__version__", "get_version"]
