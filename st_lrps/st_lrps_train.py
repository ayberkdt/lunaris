#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
st_lrps_train
=========

Thin CLI entry point for the lunar scalar potential surrogate trainer.

This file intentionally stays small. The project no longer implements the
training stack here; it re-exports the public symbols that older scripts import
while delegating real work to the focused modules below:

``st_lrps_config``
    Owns ``TrainConfig`` and the command-line interface.

``st_lrps_models``
    Owns SIREN/MLP/Fourier model construction.

``st_lrps_scaling``
    Owns origin-fixed coordinate scaling and target normalization.

``st_lrps_data``
    Owns HDF5 loading, splits, and strict lunar metadata validation.

``st_lrps_losses``
    Owns Sobolev, direction, radial/cross, altitude-balanced, and sparse
    Laplacian losses.

``st_lrps_engine``
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

try:
    from .st_lrps_config import TrainConfig, parse_args
    from .st_lrps_data import (
        BlockShuffleSampler, DatasetMeta, H5BlockDataset, TensorMemoryDataset,
        collate_h5, _build_train_val_indices, _find_latest_dataset,
        _resolve_loader_worker_count, _resolve_lunar_dataset_contract,
    )
    from .st_lrps_engine import STLRPSTrainer, train
    from .st_lrps_losses import GradNormWeights, LossCurriculum, SobolevLoss
    from .st_lrps_models import (
        FourierInputEmbedding, MLP, PhysicsNet, Sine, SirenMLP,
        build_model_from_config,
    )
    from .st_lrps_scaling import (
        IsometricScaleParams, OnlineIsometricStats, ScalerPack, fit_scaler_streaming,
    )
except ImportError:  # pragma: no cover - direct script execution
    from st_lrps_config import TrainConfig, parse_args
    from st_lrps_data import (
        BlockShuffleSampler, DatasetMeta, H5BlockDataset, TensorMemoryDataset,
        collate_h5, _build_train_val_indices, _find_latest_dataset,
        _resolve_loader_worker_count, _resolve_lunar_dataset_contract,
    )
    from st_lrps_engine import STLRPSTrainer, train
    from st_lrps_losses import GradNormWeights, LossCurriculum, SobolevLoss
    from st_lrps_models import (
        FourierInputEmbedding, MLP, PhysicsNet, Sine, SirenMLP,
        build_model_from_config,
    )
    from st_lrps_scaling import (
        IsometricScaleParams, OnlineIsometricStats, ScalerPack, fit_scaler_streaming,
    )

__all__ = [
    'TrainConfig', 'parse_args', 'train', 'STLRPSTrainer',
    'Sine', 'SirenMLP', 'MLP', 'FourierInputEmbedding', 'PhysicsNet',
    'build_model_from_config', 'IsometricScaleParams', 'ScalerPack',
    'OnlineIsometricStats', 'fit_scaler_streaming', 'DatasetMeta',
    'H5BlockDataset', 'TensorMemoryDataset', 'BlockShuffleSampler', 'collate_h5',
    '_resolve_loader_worker_count', 'SobolevLoss', 'LossCurriculum', 'GradNormWeights',
]


def main() -> None:
    cfg = parse_args()
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
