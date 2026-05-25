# st_lrps/runtime/__init__.py
"""ST-LRPS runtime inference subpackage: the surrogate force model.

The runtime boundary is intentionally clean: it depends only on
``st_lrps.artifacts``, ``st_lrps.shared``, ``st_lrps.networks``, and
``st_lrps.data.dataset_parameters`` — never on ``st_lrps.training``.

Import-safe: importing this package must not pull torch eagerly. Import the
force model explicitly:

    from st_lrps.runtime.force_model import load_surrogate_force_model
"""

from __future__ import annotations

__all__: list[str] = []
