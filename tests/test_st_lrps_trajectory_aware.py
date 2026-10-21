"""Trajectory-aware training options: low-degree leakage penalty, low-altitude
potential metric, uncentred potential scaler, warmup-stable-decay schedule and
the first-layer band initialisation of the multi-scale SIREN."""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from dataset_pipeline_test_utils import write_toy_contract_h5
from lunaris.surrogate.st_lrps.data.dataset_parameters import MU_MOON_SI, R_MOON_SI
from lunaris.surrogate.st_lrps.data.datasets import DatasetMeta
from lunaris.surrogate.st_lrps.networks.models import (
    MultiScaleSirenMLP,
    RealSHBasisEncoding,
    real_sh_basis_4pi,
)
from lunaris.surrogate.st_lrps.shared.scaling import (
    IsometricScaleParams,
    ScalerPack,
    fit_scaler_streaming,
)
from lunaris.surrogate.st_lrps.training.engine import _lr_multiplier_for_epoch
from lunaris.surrogate.st_lrps.training.losses import (
    GradNormWeights,
    SobolevLoss,
    low_degree_leakage,
)
from lunaris.surrogate.st_lrps.training.metrics import HISTORY_FIELDNAMES, flatten_epoch_metrics


def _unit(n: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    u = torch.randn(n, 3, generator=g, dtype=torch.float64)
    return u / u.norm(dim=-1, keepdim=True)


# --------------------------------------------------------------------------- #
# Real SH basis beyond the encoder's degree cap
# --------------------------------------------------------------------------- #


def test_real_sh_basis_is_orthonormal_to_degree_25_by_quadrature():
    # Gauss-Legendre in cos(theta) x uniform longitude is exact for products of
    # two degree-25 harmonics (degree 50 polynomials on the sphere).
    lmax = 25
    t, w = np.polynomial.legendre.leggauss(lmax + 2)
    lon = np.arange(2 * lmax + 3) * 2.0 * np.pi / (2 * lmax + 3)
    tt, ll = np.meshgrid(t, lon, indexing="ij")
    weights = (np.repeat(w[:, None], lon.size, axis=1) * (2.0 * np.pi / lon.size)).reshape(-1, 1)
    st = np.sqrt(1.0 - tt**2)
    n = np.stack([st * np.cos(ll), st * np.sin(ll), tt], axis=-1).reshape(-1, 3)
    y = real_sh_basis_4pi(torch.tensor(n, dtype=torch.float64), lmax).numpy()
    gram = (y * weights).T @ y
    assert gram.shape == ((lmax + 1) ** 2,) * 2
    np.testing.assert_allclose(gram, 4.0 * np.pi * np.eye(gram.shape[0]), atol=1e-11)


def test_encoder_output_is_unchanged_by_the_basis_refactor():
    x = torch.randn(7, 3, dtype=torch.float64)
    r = x.norm(dim=-1, keepdim=True)
    out = RealSHBasisEncoding(4, append_raw=True, include_radial=True)(x)
    assert torch.equal(out, torch.cat([r, real_sh_basis_4pi(x / r, 4), x], dim=-1))


# --------------------------------------------------------------------------- #
# Low-degree leakage estimator
# --------------------------------------------------------------------------- #


def _low_field(unit: torch.Tensor, lmax: int) -> tuple[torch.Tensor, float]:
    coeff = torch.zeros((lmax + 1) ** 2, dtype=torch.float64)
    coeff[[0, 3, 10, 200, 600]] = torch.tensor([0.3, -0.2, 0.1, 0.05, 0.02], dtype=torch.float64)
    return real_sh_basis_4pi(unit, lmax) @ coeff, float(coeff.square().sum())


def test_leakage_recovers_the_low_degree_power_exactly():
    unit = _unit(4000, 0)
    field, power = _low_field(unit, 25)
    assert float(low_degree_leakage(field, unit, 25)) == pytest.approx(power, rel=1e-6)


def test_leakage_is_unbiased_for_non_uniform_directions():
    unit = _unit(4000, 1)
    unit[:, 2] = unit[:, 2].abs().sqrt() * torch.sign(unit[:, 2])  # crowd the poles
    unit = unit / unit.norm(dim=-1, keepdim=True)
    field, power = _low_field(unit, 25)
    assert float(low_degree_leakage(field, unit, 25)) == pytest.approx(power, rel=1e-6)


def test_leakage_of_a_pure_high_degree_field_is_the_sampling_floor():
    # A degree-40 field has no degree <= 25 content; the least-squares fit on
    # M samples picks up only aliasing of order (K / M) * variance.
    unit = _unit(6000, 2)
    y40 = real_sh_basis_4pi(unit, 40)[:, 40**2:]
    g = torch.Generator().manual_seed(3)
    field = y40 @ (0.05 * torch.randn(y40.shape[1], generator=g, dtype=torch.float64))
    k_over_m = 26**2 / 6000
    assert float(low_degree_leakage(field, unit, 25)) < 2.0 * k_over_m * float(field.var())


def test_leakage_needs_more_samples_than_basis_functions_and_is_differentiable():
    unit = _unit(4000, 4)
    assert float(low_degree_leakage(torch.ones(10, dtype=torch.float64), unit[:10], 25)) == 0.0
    field, _ = _low_field(unit, 25)
    field = field.clone().requires_grad_(True)
    low_degree_leakage(field, unit, 25).backward()
    assert field.grad is not None and torch.isfinite(field.grad).all() and field.grad.abs().sum() > 0


# --------------------------------------------------------------------------- #
# SobolevLoss plumbing
# --------------------------------------------------------------------------- #


def _loss_fixture(n: int = 1200):
    sc = ScalerPack(
        x=IsometricScaleParams(mean=[0.0, 0.0, 0.0], scale=2.7e6),
        u=IsometricScaleParams(mean=[0.0], scale=10.0),
        a=IsometricScaleParams(mean=[0.0, 0.0, 0.0], scale=1.0e-4),
    ).to_tensors(torch.device("cpu"), torch.float64)
    loss = SobolevLoss(
        sc, a_sign=1.0, r_ref_m=float(R_MOON_SI),
        target_mode="residual", degree_min=25, degree_max=200,
    )
    unit = _unit(n, 5)
    g = torch.Generator().manual_seed(6)
    alt_km = 100.0 + 900.0 * torch.rand(n, 1, generator=g, dtype=torch.float64)
    x = unit * (float(R_MOON_SI) + 1000.0 * alt_km)
    u = torch.zeros(n, 1, dtype=torch.float64)
    a = torch.zeros(n, 3, dtype=torch.float64)
    return loss, GradNormWeights(mode="fixed"), x, u, a, alt_km.squeeze(-1)


class _Bias(torch.nn.Module):
    """Model whose scaled potential is a constant: the error is pure degree 0."""

    def __init__(self, value: float) -> None:
        super().__init__()
        self.c = torch.nn.Parameter(torch.tensor(float(value), dtype=torch.float64))

    def forward(self, x):
        return (x[:, :1] * 0.0) + self.c


def test_low_altitude_potential_rmse_is_physical_and_altitude_limited():
    loss, weights, x, u, a, alt = _loss_fixture()
    # The fixture targets are zero for a mean-zero scaler, so the physical
    # potential error is c * u_scale everywhere.
    _, stats = loss(_Bias(0.5), x, u, a, weights, is_train=False, potential_metric_alt_max_km=300.0)
    assert stats["u_low_count"] == float((alt < 300.0).sum())
    assert math.sqrt(stats["u_low_sq_sum"] / stats["u_low_count"]) == pytest.approx(5.0, rel=1e-12)


def test_leakage_is_logged_on_validation_and_enters_training_only_when_weighted():
    loss, weights, x, u, a, _ = _loss_fixture()
    model = _Bias(0.5)
    _, val = loss(model, x, u, a, weights, is_train=False, low_degree_alt_max_km=1000.0)
    # Constant scaled error 0.5 without radial normalisation: the degree-0
    # projection is exactly 0.5 (4pi-normalised Y_00 = 1), so the power is 0.25.
    assert val["loss_low_degree"] == pytest.approx(0.25, rel=1e-6)

    off, train_off = loss(model, x, u, a, weights, is_train=True, low_degree_alt_max_km=1000.0)
    assert train_off["loss_low_degree"] == 0.0
    on, train_on = loss(
        model, x, u, a, weights, is_train=True,
        low_degree_lambda=2.0, low_degree_alt_max_km=1000.0,
    )
    assert train_on["loss_low_degree"] == pytest.approx(0.25, rel=1e-6)
    assert float(on - off) == pytest.approx(2.0 * 0.25, rel=1e-6)
    on.backward()
    assert model.c.grad is not None and float(model.c.grad) != 0.0


def test_leakage_radial_power_normalises_by_the_envelope():
    loss, weights, x, u, a, alt = _loss_fixture()
    kw = {"is_train": False, "low_degree_alt_max_km": 150.0, "low_degree_lmax": 2}
    _, plain = loss(_Bias(0.5), x, u, a, weights, **kw)
    _, normed = loss(_Bias(0.5), x, u, a, weights, low_degree_radial_power=26.0, **kw)
    assert (alt < 150.0).sum() > 9 and plain["loss_low_degree"] == pytest.approx(0.25, rel=1e-6)
    # Below 150 km rho**-26 lies between 1.27e1 and 4.25e1 (factor > 10 in power).
    assert normed["loss_low_degree"] > 10.0 * plain["loss_low_degree"]


def test_history_rows_carry_the_new_metrics():
    row = flatten_epoch_metrics(
        0,
        {"u_low_rmse_m2_s2": 1.0e-3, "loss_low_degree": 2.0e-6},
        {"u_low_rmse_m2_s2": 3.0e-3, "loss_low_degree": 4.0e-6},
        {"score": 1.0},
        cfg=None,
    )
    assert row["val_u_low_rmse_m2_s2"] == 3.0e-3
    assert row["val_loss_low_degree"] == 4.0e-6
    assert {
        "train_u_low_rmse_m2_s2", "val_u_low_rmse_m2_s2",
        "train_loss_low_degree", "val_loss_low_degree",
    } <= set(HISTORY_FIELDNAMES)


# --------------------------------------------------------------------------- #
# Uncentred potential scaler
# --------------------------------------------------------------------------- #


def _rows_with_potential_offset(n: int) -> np.ndarray:
    rng = np.random.default_rng(7)
    dirs = rng.normal(size=(n, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    xyz = dirs * (float(R_MOON_SI) + rng.uniform(100.0, 500.0, size=(n, 1)) * 1000.0)
    u = 3.0 + rng.normal(size=(n, 1))
    a = dirs * 1.0e-4
    return np.concatenate([xyz, u, a], axis=1).astype(np.float32)


@pytest.mark.parametrize("center", ["mean", "zero"])
def test_u_center_controls_only_the_potential_mean(tmp_path, center):
    path = write_toy_contract_h5(tmp_path / "rows.h5", rows=_rows_with_potential_offset(2000))
    sc = fit_scaler_streaming(
        h5_path=path, dset_name="data", meta=DatasetMeta.from_h5(path), use_si=True,
        mu_si=float(MU_MOON_SI), a_sign=1.0, degree_min=2, degree_max=4,
        target_mode="residual", u_center=center,
    )
    assert sc.provenance["u_center"] == center
    if center == "zero":
        assert sc.u.mean == [0.0]
    else:
        assert sc.u.mean[0] == pytest.approx(3.0, abs=0.1)
    assert sc.u.scale > 0.0


def test_u_center_rejects_unknown_values(tmp_path):
    path = write_toy_contract_h5(tmp_path / "rows.h5", rows=_rows_with_potential_offset(200))
    with pytest.raises(ValueError, match="u_center"):
        fit_scaler_streaming(
            h5_path=path, dset_name="data", meta=DatasetMeta.from_h5(path), use_si=True,
            mu_si=float(MU_MOON_SI), a_sign=1.0, degree_min=2, degree_max=4,
            target_mode="residual", u_center="median",
        )


# --------------------------------------------------------------------------- #
# Warmup-stable-decay schedule
# --------------------------------------------------------------------------- #


def test_wsd_holds_then_cools_down_to_the_floor_on_the_last_epoch():
    kw = {"total_epochs": 40, "warmup_epochs": 4, "min_lr_ratio": 0.01, "t_max": 999,
          "schedule": "wsd", "decay_fraction": 0.25}
    lrs = [_lr_multiplier_for_epoch(e, **kw) for e in range(40)]
    assert lrs[:4] == pytest.approx([0.25, 0.5, 0.75, 1.0])
    assert lrs[4:31] == [1.0] * 27            # 36 post-warm-up epochs, 9 in the cooldown
    cooldown = lrs[31:]
    assert all(b < a for a, b in itertools.pairwise(cooldown))
    assert lrs[-1] == pytest.approx(0.01)
    # 1 - sqrt profile: one ninth of the way in, the multiplier is 1 - 1/3 of the span.
    assert lrs[31] == pytest.approx(0.01 + 0.99 * (1.0 - math.sqrt(1.0 / 9.0)))


def test_cosine_schedule_is_the_unchanged_default():
    kw = {"total_epochs": 800, "warmup_epochs": 5, "min_lr_ratio": 0.01, "t_max": 780}
    assert _lr_multiplier_for_epoch(100, **kw) == _lr_multiplier_for_epoch(100, schedule="cosine", **kw)
    with pytest.raises(ValueError, match="schedule"):
        _lr_multiplier_for_epoch(100, schedule="step", **kw)


# --------------------------------------------------------------------------- #
# Multi-scale band initialisation
# --------------------------------------------------------------------------- #


def _first_layer_frequency(model: MultiScaleSirenMLP, band: int) -> float:
    return model.w0_bands[band] * float(model.band_layers[band].weight.abs().max())


def test_legacy_band_init_starts_every_later_band_at_the_same_low_frequency():
    torch.manual_seed(0)
    model = MultiScaleSirenMLP(in_dim=11, hidden=64, depth=3, w0_bands=[15.3, 42.4, 80.0])
    bound = math.sqrt(6.0 / 11.0)
    assert _first_layer_frequency(model, 1) == pytest.approx(bound, rel=0.05)
    assert _first_layer_frequency(model, 2) == pytest.approx(bound, rel=0.05)
    assert _first_layer_frequency(model, 1) < _first_layer_frequency(model, 0)


def test_siren_first_band_init_scales_with_each_band_w0():
    torch.manual_seed(0)
    model = MultiScaleSirenMLP(
        in_dim=11, hidden=64, depth=3, w0_bands=[15.3, 42.4, 80.0], band_init="siren_first"
    )
    for band, w0 in enumerate((15.3, 42.4, 80.0)):
        assert _first_layer_frequency(model, band) == pytest.approx(w0 / 11.0, rel=0.05)
    with pytest.raises(ValueError, match="band_init"):
        MultiScaleSirenMLP(in_dim=11, hidden=64, depth=3, w0_bands=[15.3, 42.4], band_init="x")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def test_new_flags_reach_the_train_config(monkeypatch):
    import sys

    from lunaris.surrogate.st_lrps.training.config import parse_args

    monkeypatch.setattr(sys, "argv", [
        "lunaris-train", "--train-data", "train.h5", "--val-data", "val.h5",
        "--lr-schedule", "wsd", "--lr-decay-fraction", "0.3",
        "--low-degree-loss-weight", "0.5", "--low-degree-lmax", "12",
        "--low-degree-alt-max-km", "350", "--potential-metric-alt-max-km", "250",
        "--low-degree-radial-power", "26",
        "--u-center", "zero", "--band-init", "siren_first",
    ])
    cfg = parse_args()
    assert (cfg.lr_schedule_mode, cfg.lr_decay_fraction) == ("wsd", 0.3)
    assert (cfg.low_degree_loss_weight, cfg.low_degree_lmax) == (0.5, 12)
    assert (cfg.low_degree_alt_max_km, cfg.potential_metric_alt_max_km) == (350.0, 250.0)
    assert cfg.low_degree_radial_power == 26.0
    assert (cfg.u_center, cfg.band_init) == ("zero", "siren_first")


def test_defaults_keep_the_shipped_recipe(monkeypatch):
    import sys

    from lunaris.surrogate.st_lrps.training.config import parse_args

    monkeypatch.setattr(sys, "argv", ["lunaris-train", "--train-data", "t.h5", "--val-data", "v.h5"])
    cfg = parse_args()
    assert (cfg.lr_schedule_mode, cfg.u_center, cfg.band_init) == ("cosine", "mean", "legacy")
    assert (cfg.low_degree_loss_weight, cfg.low_degree_radial_power) == (0.0, 0.0)
    assert cfg.low_degree_alt_max_km == cfg.potential_metric_alt_max_km == 300.0


def test_potential_low_checkpoint_metric_scores_the_low_altitude_potential_rmse():
    from lunaris.surrogate.st_lrps.training.metrics import (
        checkpoint_formula,
        compute_checkpoint_score,
    )

    stats = {"val_total_loss": 1.0, "val_loss_u": 0.1, "val_loss_a": 0.2, "u_low_rmse_m2_s2": 7.5e-4}
    score, report = compute_checkpoint_score(stats, {"best_metric": "potential_low"})
    assert score == 7.5e-4 and report["formula"] == checkpoint_formula("potential_low")
    with pytest.raises(ValueError, match="not finite"):
        compute_checkpoint_score({**stats, "u_low_rmse_m2_s2": float("nan")}, {"best_metric": "potential_low"})
    with pytest.raises(KeyError, match="potential_low"):
        compute_checkpoint_score({k: v for k, v in stats.items() if k != "u_low_rmse_m2_s2"},
                                 {"best_metric": "potential_low"})
