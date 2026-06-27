"""Compatibility shim for the historical import path.

The canonical implementation lives in ``lunaris.core.propagation.propagator``.
"""

from __future__ import annotations

import importlib as _importlib
import sys as _sys

_legacy_name = __name__
_legacy_package = __package__
_impl = _importlib.import_module('lunaris.core.propagation.propagator')
globals().update(_impl.__dict__)
_sys.modules[_legacy_name] = _impl

_parent = _sys.modules.get(_legacy_package)
if _parent is not None:
    _parent.propagator = _impl
