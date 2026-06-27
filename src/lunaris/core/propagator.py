"""Compatibility shim for the historical ``lunaris.core.propagator`` import path.

The canonical implementation lives in ``lunaris.core.propagation``. The original
module exposed a single flat namespace, so this shim:

1. folds every split submodule's public + private names onto the canonical
   ``propagation.propagator`` module (so legacy private accesses such as
   ``lunaris.core.propagator._norm_method`` keep resolving), and
2. aliases the old import path to the canonical module object via ``sys.modules``
   so that ``lunaris.core.propagator is lunaris.core.propagation.propagator`` and
   monkeypatching the legacy path updates the canonical module.

The fold is intentionally dynamic (not a static ``from x import y`` block) so
import-pruning linters (``ruff --fix``) cannot strip the re-exports.
"""

from __future__ import annotations

import importlib as _importlib
import sys as _sys

_legacy_name = __name__
_legacy_package = __package__

_impl = _importlib.import_module("lunaris.core.propagation.propagator")

# Submodules that, together with propagator.py, made up the original flat module.
_SUBMODULES = (
    "lunaris.core.propagation.result",
    "lunaris.core.propagation.time_grid",
    "lunaris.core.propagation.telemetry",
    "lunaris.core.propagation.checkpoint",
    "lunaris.core.propagation.events",
    "lunaris.core.propagation.integrators.scipy",
    "lunaris.core.propagation.integrators.rk",
    "lunaris.core.propagation.integrators.symplectic",
    "lunaris.core.propagation.integrators.fixed_step",
)
for _modname in _SUBMODULES:
    _sub = _importlib.import_module(_modname)
    for _k, _v in vars(_sub).items():
        if not _k.startswith("__") and not hasattr(_impl, _k):
            setattr(_impl, _k, _v)

# Alias the legacy import path to the canonical module object (identity), so
# downstream monkeypatching and ``is`` checks see one module.
globals().update(_impl.__dict__)
_sys.modules[_legacy_name] = _impl

_parent = _sys.modules.get(_legacy_package)
if _parent is not None:
    _parent.propagator = _impl  # type: ignore[attr-defined]
