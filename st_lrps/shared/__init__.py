# st_lrps/shared/__init__.py
"""ST-LRPS shared utilities used by both training and runtime.

Houses the scaling single-source-of-truth (``st_lrps.shared.scaling``) so the
runtime inference path never has to depend on the training package.

Import-safe: importing this package must not pull heavy dependencies. Import
the submodule explicitly: ``from st_lrps.shared.scaling import ScalerPack``.
"""

from __future__ import annotations

__all__: list[str] = []
