"""Focused regression tests for the ST-LRPS training-improvement pass."""

from __future__ import annotations

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")
torch = pytest.importorskip("torch")

from lunaris.surrogate.st_lrps.evaluation.cli import iter_h5_batches
from lunaris.surrogate.st_lrps.shared.scaling import (
    IsometricScaleParams,
    ScalerPack,
)
from lunaris.surrogate.st_lrps.training.losses import (
    SobolevLoss,
    _altitude_balanced_mean_square,
    _direction_loss_factor,
    collocation_laplacian_loss,
)
from lunaris.surrogate.st_lrps.training.periodic_eval import (
    resolve_eval_exclude_train_indices,
)


def _scaler() -> ScalerPack:
    return ScalerPack(
        x=IsometricScaleParams(mean=[0.0, 0.0, 0.0], scale=1.0),
        u=IsometricScaleParams(mean=[0.0], scale=1.0),
        a=IsometricScaleParams(mean=[0.0, 0.0, 0.0], scale=1.0),
    )


def test_altitude_balanced_mean_square_matches_nonempty_bin_reference() -> None:
    # Bins [0, 100), [100, 200], plus one outside bucket.
    r_ref = 1_000_000.0
    alt_km = np.array([0.0, 50.0, 100.0, 250.0])
    x = torch.from_numpy(np.column_stack([r_ref + alt_km * 1000.0, np.zeros(4), np.zeros(4)])).float()
    sample_sq = torch.tensor([1.0, 3.0, 10.0, 7.0])

    result = _altitude_balanced_mean_square(
        sample_sq,
        x,
        r_ref_m=r_ref,
        altitude_min_km=0.0,
        altitude_max_km=200.0,
        altitude_bin_width_km=100.0,
    )
    expected = torch.tensor(((1.0 + 3.0) / 2.0 + 10.0 + 7.0) / 3.0)
    assert result == pytest.approx(float(expected))


def test_three_dimensional_laplacian_diagnostic_is_exact_trace() -> None:
    scaler = _scaler()
    loss = SobolevLoss(scaler, a_sign=1.0)
    x = torch.randn(12, 3, requires_grad=True)
    # x^2 + y^2 - 2 z^2 is harmonic, despite having nonzero off-diagonal-free
    # Hessian entries. A squared Hutchinson estimate would not be zero here.
    u = (x[:, 0:1] ** 2) + (x[:, 1:2] ** 2) - 2.0 * (x[:, 2:3] ** 2)
    grad_u = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True, retain_graph=True)[0]

    result = loss._laplacian_penalty(
        grad_u,
        x,
        subset_size=x.shape[0],
        n_hutchinson_samples=4,
        laplacian_mode="diagnostic",
    )
    assert float(result) == pytest.approx(0.0, abs=1e-10)


def test_collocation_three_dimensional_laplacian_is_exact_trace() -> None:
    class HarmonicPotential(torch.nn.Module):
        def forward(self, values: torch.Tensor) -> torch.Tensor:
            return values[:, 0:1] ** 2 + values[:, 1:2] ** 2 - 2.0 * values[:, 2:3] ** 2

    result = collocation_laplacian_loss(
        HarmonicPotential(),
        _scaler(),
        r_min_m=1.0,
        r_max_m=2.0,
        n_points=16,
        device=torch.device("cpu"),
        mode="diagnostic",
    )
    assert float(result) == pytest.approx(0.0, abs=1e-10)


def test_collocation_penalty_invariant_under_grad_accumulation(monkeypatch) -> None:
    """One full-weight collocation penalty per optimizer step: the collocation-only
    gradient contribution must not shrink by 1/grad_accumulation_steps."""
    import copy

    from torch.utils.data import DataLoader, TensorDataset

    from lunaris.surrogate.st_lrps.training import engine as engine_mod
    from lunaris.surrogate.st_lrps.training.config import TrainConfig
    from lunaris.surrogate.st_lrps.training.engine import STLRPSTrainer
    from lunaris.surrogate.st_lrps.training.losses import GradNormWeights

    # Deterministic stand-in for the (randomly sampled) collocation loss so the
    # accum=1 vs accum=4 runs are exactly comparable.
    def fake_collocation_loss(model, loss_fn, **kwargs):
        return (model[0].weight ** 2).sum()

    monkeypatch.setattr(engine_mod, "collocation_laplacian_loss", fake_collocation_loss)

    sp = ScalerPack(
        x=IsometricScaleParams(mean=[0.0, 0.0, 0.0], scale=2e6),
        u=IsometricScaleParams(mean=[0.0], scale=1.0),
        a=IsometricScaleParams(mean=[0.0, 0.0, 0.0], scale=1e-3),
    ).to_tensors(torch.device("cpu"), torch.float32)

    base_model = torch.nn.Sequential(
        torch.nn.Linear(3, 16), torch.nn.Tanh(), torch.nn.Linear(16, 1)
    )
    gen = torch.Generator().manual_seed(0)
    x_raw = torch.randn(16, 3, generator=gen) * 1.85e6
    u_raw = torch.randn(16, 1, generator=gen)
    a_raw = torch.randn(16, 3, generator=gen) * 1e-3

    def epoch_grads(grad_accum: int, laplacian_on: bool) -> list:
        model = copy.deepcopy(base_model)
        cfg = TrainConfig(
            data="/tmp/fake.h5", out="/tmp/fake_out",
            epochs=1, batch_size=4,
            grad_accumulation_steps=grad_accum,
            laplacian_mode="train" if laplacian_on else "off",
            collocation_laplacian_weight=0.5 if laplacian_on else 0.0,
            collocation_laplacian_every=1,
            collocation_laplacian_samples=8,
            collocation_laplacian_hutchinson_samples=2,
            max_grad_norm=0.0,  # no clipping: gradients must stay comparable
            amp=False,
        )
        opt = torch.optim.SGD(model.parameters(), lr=0.0)  # lr=0 -> params never move
        trainer = STLRPSTrainer(
            model, SobolevLoss(sp, a_sign=1.0), opt, GradNormWeights(mode="fixed"),
            torch.device("cpu"), cfg,
            collocation_r_min_m=1.837e6, collocation_r_max_m=1.937e6,
        )
        loader = DataLoader(TensorDataset(x_raw, u_raw, a_raw), batch_size=4)
        trainer.run_epoch(loader, is_train=True, epoch=0)
        return [p.grad.detach().clone() for p in model.parameters()]

    # The gradient left on the model after the epoch belongs to the LAST
    # accumulation group; differencing laplacian-on vs laplacian-off isolates
    # the collocation contribution added to that optimizer step.
    col_only_1 = [
        a - b for a, b in zip(epoch_grads(1, True), epoch_grads(1, False), strict=True)
    ]
    col_only_4 = [
        a - b for a, b in zip(epoch_grads(4, True), epoch_grads(4, False), strict=True)
    ]

    norm_1 = float(torch.sqrt(sum((g ** 2).sum() for g in col_only_1)))
    norm_4 = float(torch.sqrt(sum((g ** 2).sum() for g in col_only_4)))
    assert norm_1 > 0.0, "collocation penalty must contribute a gradient"
    # A regression back to per-group scaling would give norm_4 ~= norm_1 / 4;
    # the tolerance only needs to absorb float32 cancellation noise in the
    # base-gradient differencing.
    assert norm_4 == pytest.approx(norm_1, rel=1e-2), (
        "collocation-only gradient must not scale with grad_accumulation_steps "
        f"(accum=1 -> {norm_1:.6e}, accum=4 -> {norm_4:.6e})"
    )


def test_direction_loss_ramp_starts_on_configured_epoch() -> None:
    cfg = type(
        "Cfg",
        (),
        {
            "direction_loss_start_epoch": 10,
            "direction_loss_ramp_epochs": 4,
            "direction_loss_weight": 2.0,
        },
    )()
    assert _direction_loss_factor(9, cfg) == pytest.approx(0.0)
    assert _direction_loss_factor(10, cfg) == pytest.approx(0.5)
    assert _direction_loss_factor(13, cfg) == pytest.approx(2.0)


def test_periodic_eval_excludes_persisted_train_rows(tmp_path) -> None:
    path = tmp_path / "cloud.h5"
    values = np.arange(70, dtype=np.float32).reshape(10, 7)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("data", data=values)

    batches = list(iter_h5_batches(path, "data", batch_size=4, exclude_indices=np.array([0, 2, 4, 6])))
    observed = np.concatenate(batches, axis=0)
    assert observed.shape[0] == 6
    assert observed[:, 0].tolist() == [7.0, 21.0, 35.0, 49.0, 56.0, 63.0]

    capped = list(
        iter_h5_batches(
            path,
            "data",
            batch_size=4,
            exclude_indices=np.array([0, 2, 4, 6]),
            limit_rows=3,
        )
    )
    assert np.concatenate(capped, axis=0).shape[0] == 3

    run_dir = tmp_path / "run"
    split_dir = run_dir / "provenance"
    split_dir.mkdir(parents=True)
    np.savez(split_dir / "split_indices.npz", train=np.array([0, 2, 4, 6]), val=np.array([1, 3, 5, 7, 8, 9]))
    cfg = type("Cfg", (), {"data": str(path), "val_data": None})()
    assert resolve_eval_exclude_train_indices(cfg, "val", run_dir) is not None
    assert resolve_eval_exclude_train_indices(cfg, "test", run_dir) is None
