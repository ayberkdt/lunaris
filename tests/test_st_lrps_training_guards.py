"""Training-engine guards: invalid programmatic settings fail instead of being replaced."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from lunaris.surrogate.st_lrps.shared.scaling import IsometricScaleParams, ScalerPack
from lunaris.surrogate.st_lrps.training.config import TrainConfig
from lunaris.surrogate.st_lrps.training.engine import STLRPSTrainer
from lunaris.surrogate.st_lrps.training.losses import GradNormWeights, SobolevLoss


def _trainer(cfg: TrainConfig) -> STLRPSTrainer:
    sp = ScalerPack(
        x=IsometricScaleParams(mean=[0.0, 0.0, 0.0], scale=2e6),
        u=IsometricScaleParams(mean=[0.0], scale=1.0),
        a=IsometricScaleParams(mean=[0.0, 0.0, 0.0], scale=1e-3),
    ).to_tensors(torch.device("cpu"), torch.float32)
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Tanh(), torch.nn.Linear(4, 1))
    opt = torch.optim.SGD(model.parameters(), lr=0.0)
    return STLRPSTrainer(
        model, SobolevLoss(sp, a_sign=1.0), opt, GradNormWeights(mode="fixed"),
        torch.device("cpu"), cfg,
    )


def test_trainer_rejects_an_invalid_laplacian_mode():
    # "trian" used to become "diagnostic" silently: the run asked for a trained
    # Laplacian penalty and got diagnostics only.
    cfg = TrainConfig(data="x.h5", out="o", epochs=1, batch_size=8, laplacian_mode="trian", amp=False)
    with pytest.raises(ValueError, match="laplacian_mode='trian' is not valid"):
        _trainer(cfg)


@pytest.mark.parametrize("mode", ["off", "diagnostic", "train", " Train "])
def test_trainer_accepts_the_valid_laplacian_modes(mode):
    cfg = TrainConfig(data="x.h5", out="o", epochs=1, batch_size=8, laplacian_mode=mode, amp=False)
    assert _trainer(cfg).laplacian_mode == mode.strip().lower()
