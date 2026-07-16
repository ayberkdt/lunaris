#!/usr/bin/env python3
"""
st_lrps.training.cli
====================

Thin CLI entry point for the lunar scalar potential surrogate trainer.
Canonical invocation: ``python -m lunaris.surrogate.st_lrps.training.cli``.

This file intentionally stays small. The project no longer implements the
training stack here; it re-exports the public symbols that callers import
while delegating real work to the focused modules below:

``st_lrps.training.config``
    Owns ``TrainConfig`` and the command-line interface.

``st_lrps.networks.models``
    Owns SIREN/MLP/Fourier model construction.

``st_lrps.shared.scaling``
    Owns origin-fixed coordinate scaling and target normalization.

``st_lrps.data.datasets``
    Owns HDF5 loading, splits, and strict lunar metadata validation.

``st_lrps.training.losses``
    Owns Sobolev, direction, radial/cross, altitude-balanced, and sparse
    Laplacian losses.

``st_lrps.training.engine``
    Owns the training loop, checkpoints, metrics, and history plots.

Physics convention
------------------
The model learns a scalar residual potential dU(x). Residual acceleration da is
obtained by differentiating that scalar field with autograd. It is therefore a
Sobolev-trained lunar residual potential surrogate, not a classical q,p
state-space model.
"""

from __future__ import annotations

import sys
from importlib import import_module
from typing import Any

_EXPORT_MODULES = {
    "TrainConfig": "lunaris.surrogate.st_lrps.training.config",
    "parse_args": "lunaris.surrogate.st_lrps.training.config",
    "train": "lunaris.surrogate.st_lrps.training.engine",
    "STLRPSTrainer": "lunaris.surrogate.st_lrps.training.engine",
    "Sine": "lunaris.surrogate.st_lrps.networks.models",
    "SirenMLP": "lunaris.surrogate.st_lrps.networks.models",
    "MLP": "lunaris.surrogate.st_lrps.networks.models",
    "FourierInputEmbedding": "lunaris.surrogate.st_lrps.networks.models",
    "PhysicsNet": "lunaris.surrogate.st_lrps.networks.models",
    "build_model_from_config": "lunaris.surrogate.st_lrps.networks.models",
    "IsometricScaleParams": "lunaris.surrogate.st_lrps.shared.scaling",
    "ScalerPack": "lunaris.surrogate.st_lrps.shared.scaling",
    "OnlineIsometricStats": "lunaris.surrogate.st_lrps.shared.scaling",
    "fit_scaler_streaming": "lunaris.surrogate.st_lrps.shared.scaling",
    "DatasetMeta": "lunaris.surrogate.st_lrps.data.datasets",
    "H5BlockDataset": "lunaris.surrogate.st_lrps.data.datasets",
    "TensorMemoryDataset": "lunaris.surrogate.st_lrps.data.datasets",
    "BlockShuffleSampler": "lunaris.surrogate.st_lrps.data.datasets",
    "collate_h5": "lunaris.surrogate.st_lrps.data.datasets",
    "build_train_val_indices": "lunaris.surrogate.st_lrps.data.datasets",
    "find_latest_dataset": "lunaris.surrogate.st_lrps.data.datasets",
    "resolve_loader_worker_count": "lunaris.surrogate.st_lrps.data.datasets",
    "resolve_lunar_dataset_contract": "lunaris.surrogate.st_lrps.data.datasets",
    # compat: retired underscore spellings of the dataset helpers above.
    # Resolved through _COMPAT_RENAMES; remove in the MINOR release after
    # the rename ships (docs/PUBLIC_API.md "Naming And Boundary Policy").
    "_build_train_val_indices": "lunaris.surrogate.st_lrps.data.datasets",
    "_find_latest_dataset": "lunaris.surrogate.st_lrps.data.datasets",
    "_resolve_loader_worker_count": "lunaris.surrogate.st_lrps.data.datasets",
    "_resolve_lunar_dataset_contract": "lunaris.surrogate.st_lrps.data.datasets",
    "SobolevLoss": "lunaris.surrogate.st_lrps.training.losses",
    "LossCurriculum": "lunaris.surrogate.st_lrps.training.losses",
    "GradNormWeights": "lunaris.surrogate.st_lrps.training.losses",
}

__all__ = list(_EXPORT_MODULES.keys())

# Old exported name -> current name in the defining module (compat only).
_COMPAT_RENAMES = {
    "_build_train_val_indices": "build_train_val_indices",
    "_find_latest_dataset": "find_latest_dataset",
    "_resolve_loader_worker_count": "resolve_loader_worker_count",
    "_resolve_lunar_dataset_contract": "resolve_lunar_dataset_contract",
}


def __getattr__(name: str) -> Any:
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    attr_name = _COMPAT_RENAMES.get(name, name)
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def main() -> None:
    from lunaris.surrogate.st_lrps.training.config import parse_args

    cfg = parse_args()
    from lunaris.surrogate.st_lrps.training.engine import train

    train(cfg)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\n[STOP] Training interrupted by user. Exiting safely...')
        sys.exit(0)
    except Exception as exc:
        print(f'\n[FATAL ERROR] {exc}')
        import traceback

        traceback.print_exc()
        sys.exit(1)
