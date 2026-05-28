# st_lrps/networks/__init__.py
"""ST-LRPS neural network architectures.

The architecture implementations live in ``st_lrps.networks.models`` and pull
torch on import. This package initializer stays lightweight and does NOT import
``models`` eagerly, so ``import lunaris.surrogate.st_lrps.networks`` does not force a torch import.
Import architectures explicitly:

    from lunaris.surrogate.st_lrps.networks.models import PhysicsNet, build_model_from_config
"""

from __future__ import annotations

__all__: list[str] = []
