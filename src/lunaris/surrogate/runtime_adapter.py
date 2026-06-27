"""Compatibility shim for the historical import path.

The canonical implementation lives in ``lunaris.surrogate.runtime``. The original
``lunaris.surrogate.runtime_adapter`` module exposed a single flat namespace, so
this shim reconstructs that surface by folding every split submodule's public and
private names into the canonical ``runtime.adapter`` module. Legacy imports and
monkeypatch targets continue to resolve after the split. The dynamic fold is
intentionally not a static ``from x import y`` block so import-pruning linters
cannot strip the re-exports.
"""

from __future__ import annotations

import importlib as _importlib
import sys as _sys

_legacy_name = __name__
_legacy_package = __package__

_impl = _importlib.import_module("lunaris.surrogate.runtime.adapter")

# Submodules that, together with adapter.py, made up the original flat module.
_SUBMODULES = (
    "lunaris.surrogate.runtime.device",
    "lunaris.surrogate.runtime.artifact",
    "lunaris.surrogate.runtime.metadata",
    "lunaris.surrogate.runtime.scalers",
    "lunaris.surrogate.runtime.networks",
    "lunaris.surrogate.runtime.gravity_provider",
    "lunaris.surrogate.runtime.force_runtime",
)
for _modname in _SUBMODULES:
    _sub = _importlib.import_module(_modname)
    for _k, _v in vars(_sub).items():
        if not _k.startswith("__") and not hasattr(_impl, _k):
            setattr(_impl, _k, _v)

globals().update(_impl.__dict__)
_sys.modules[_legacy_name] = _impl

_parent = _sys.modules.get(_legacy_package)
if _parent is not None:
    _parent.runtime_adapter = _impl
