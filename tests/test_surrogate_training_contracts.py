# -*- coding: utf-8 -*-
"""
Regression tests for the lunar surrogate training / analysis contract.

These tests protect the exact areas that previously caused silent mistakes:

- training auto-discovery must not prefer a newer Earth-era dataset
- altitude filtering must not invent a reference radius for SI datasets
- runtime inference must understand the newer Fourier-augmented lunar runs
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

from common.constants import MU_MOON, R_MOON
from models.surrogate_gravity import _build_model_from_config
from surrogate_gravity_model import spatial_cloud_generator as scg
from surrogate_gravity_model.st_lrps_evaluate import _build_ood_region_masks, compute_metrics, evaluate
from surrogate_gravity_model.st_lrps_train import (
    LossCurriculum,
    _build_train_val_indices,
    _find_latest_dataset,
    _resolve_loader_worker_count,
    parse_args,
)
from surrogate_gravity_model.spatial_cloud_analysis import _apply_region_filter


def _write_cloud(path: Path, *, body: str, mu_si: float, r_ref_m: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("data", data=np.zeros((32, 7), dtype=np.float32))
        handle.attrs["central_body"] = body
        handle.attrs["mu_si"] = float(mu_si)
        handle.attrs["r_ref_m"] = float(r_ref_m)
        handle.attrs["unit_system"] = "si"
        handle.attrs["degree_min"] = 20
        handle.attrs["degree_max"] = 100
        handle.attrs["target_mode"] = "residual"
        handle.attrs["columns"] = "[x,y,z,dU,dax,day,daz]"


def test_find_latest_dataset_prefers_lunar_cloud_over_newer_earth_cloud(tmp_path: Path) -> None:
    lunar = tmp_path / "data" / "potential_cloud_moon.h5"
    earth = tmp_path / "data" / "potential_cloud_earth.h5"

    _write_cloud(lunar, body="moon", mu_si=float(MU_MOON), r_ref_m=float(R_MOON))
    _write_cloud(earth, body="earth", mu_si=3.986004418e14, r_ref_m=6_378_137.0)

    os.utime(lunar, (1_700_000_000, 1_700_000_000))
    os.utime(earth, (1_800_000_000, 1_800_000_000))

    discovered = _find_latest_dataset(tmp_path)

    assert discovered == lunar


def test_altitude_filter_requires_reference_radius_for_si_datasets() -> None:
    X = np.array([[float(R_MOON + 150_000.0), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float64)

    with pytest.raises(ValueError, match="r_ref_m"):
        _apply_region_filter(
            X,
            r_ref_m=None,
            DU_m=None,
            unit_system="si",
            alt_min_km=100.0,
            alt_max_km=500.0,
        )


def test_runtime_model_builder_supports_fourier_raw_skip_path() -> None:
    torch = pytest.importorskip("torch")

    model = _build_model_from_config(
        {
            "activation": "sine",
            "hidden": 16,
            "depth": 2,
            "use_fourier": True,
            "fourier_n_features": 8,
            "fourier_sigma": 1.0,
            "fourier_seed": 7,
            "fourier_append_raw": True,
            "w0_first": 30.0,
            "w0_hidden": 30.0,
        }
    )

    x = torch.zeros((4, 3), dtype=torch.float32)
    y = model(x)

    assert tuple(y.shape) == (4, 1)


def test_loss_curriculum_ramps_acceleration_term_monotonically() -> None:
    curriculum = LossCurriculum(potential_only_epochs=3, accel_ramp_epochs=4)

    factors = [curriculum.accel_factor(epoch) for epoch in range(8)]

    assert factors[:3] == pytest.approx([0.05, 0.05, 0.05])
    assert factors[3:] == pytest.approx([0.2875, 0.525, 0.7625, 1.0, 1.0])


def test_hdf5_loader_workers_are_forced_safe_on_windows() -> None:
    assert _resolve_loader_worker_count(Path("cloud.h5"), 4, os_name="nt") == 0
    assert _resolve_loader_worker_count(Path("cloud.pt"), 4, os_name="nt") == 4
    assert _resolve_loader_worker_count(Path("cloud.h5"), 4, os_name="posix") == 4


def test_compute_metrics_uses_bounded_relative_error_for_near_zero_residuals() -> None:
    ref = np.array([10.0, 9.0, 0.0, 0.0], dtype=np.float64)
    err = np.array([0.1, 0.1, 0.1, 0.1], dtype=np.float64)

    metrics = compute_metrics(err, ref)

    assert metrics.rel_floor_abs > 0.0
    assert metrics.rel_mean_pct < 1000.0
    assert metrics.nrmse_pct > 0.0


def test_load_coeffs_from_ssot_uses_loaded_gfc_constants(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_loader(*, file_path: str, max_degree: int, expected_norm: str, strict: bool):
        c = np.zeros((max_degree + 1, max_degree + 1), dtype=np.float64)
        s = np.zeros_like(c)
        meta = {
            "mu_si": float(MU_MOON) * 1.001,
            "r_ref_m": float(R_MOON) + 10.0,
            "degree": int(max_degree),
            "central_body": "moon",
            "modelname": "fake_moon.gfc",
            "norm": expected_norm,
        }
        return c, s, meta

    monkeypatch.setattr(scg, "load_icgem_gfc", _fake_loader)
    C, S, meta = scg.load_coeffs_from_ssot(degree_max=8, gfc_path="fake.gfc")

    assert C.shape == (9, 9)
    assert S.shape == (9, 9)
    assert float(meta["mu_si"]) == pytest.approx(float(MU_MOON) * 1.001)
    assert float(meta["r_ref_m"]) == pytest.approx(float(R_MOON) + 10.0)


def test_load_coeffs_from_ssot_rejects_non_lunar_signature(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_loader(*, file_path: str, max_degree: int, expected_norm: str, strict: bool):
        c = np.zeros((max_degree + 1, max_degree + 1), dtype=np.float64)
        s = np.zeros_like(c)
        meta = {
            "mu_si": 3.986004418e14,
            "r_ref_m": 6_378_137.0,
            "degree": int(max_degree),
            "central_body": "earth",
        }
        return c, s, meta

    monkeypatch.setattr(scg, "load_icgem_gfc", _fake_loader)

    with pytest.raises(ValueError, match="not lunar-compatible"):
        scg.load_coeffs_from_ssot(degree_max=8, gfc_path="fake_earth.gfc")


def test_train_val_split_is_reproducible_and_not_tail_contiguous() -> None:
    train_idx, val_idx = _build_train_val_indices(100, 0.2, seed=7)

    assert train_idx.shape == (80,)
    assert val_idx.shape == (20,)
    assert np.array_equal(train_idx, np.sort(train_idx))
    assert np.array_equal(val_idx, np.sort(val_idx))
    assert not np.array_equal(val_idx, np.arange(80, 100, dtype=np.int64))

    train_idx_2, val_idx_2 = _build_train_val_indices(100, 0.2, seed=7)
    assert np.array_equal(train_idx, train_idx_2)
    assert np.array_equal(val_idx, val_idx_2)


def test_train_parse_args_defaults_are_safe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data_path = tmp_path / "cloud.h5"
    _write_cloud(data_path, body="moon", mu_si=float(MU_MOON), r_ref_m=float(R_MOON))
    with h5py.File(data_path, "a") as handle:
        handle.attrs["requested_degree"] = 100
        handle.attrs["alt_min_km"] = 100.0
        handle.attrs["alt_max_km"] = 500.0

    out_dir = tmp_path / "run"
    monkeypatch.setattr(
        sys,
        "argv",
        ["st_lrps_train.py", "--data", str(data_path), "--out", str(out_dir)],
    )

    cfg = parse_args()

    assert cfg.activation == "sine"
    assert cfg.use_fourier is False
    assert cfg.dynamic_weights is False
    assert cfg.gradnorm_mode == "ntk_init"
    assert cfg.amp is False
    # Production defaults as of the AI/ML training-system upgrade.
    assert cfg.accel_ramp_epochs == 40
    assert cfg.warmup_epochs == 5
    assert cfg.min_lr_ratio == pytest.approx(0.05)
    assert cfg.depth == 6
    assert cfg.use_residual_blocks is True
    assert cfg.n_bands == 3
    assert cfg.use_altitude_balanced_loss is True
    assert cfg.use_radial_cross_loss is True
    assert cfg.best_metric == "hybrid"


def test_ood_region_masks_match_immediate_shell_outside_training_band() -> None:
    altitudes = np.array([50.0, 80.0, 100.0, 250.0, 500.0, 520.0, 550.0], dtype=np.float64)
    masks = _build_ood_region_masks(altitudes, alt_lo=100.0, alt_hi=500.0, margin_fraction=0.10)

    assert masks["lower_bounds_km"] == pytest.approx([60.0, 100.0])
    assert masks["in_band_bounds_km"] == pytest.approx([100.0, 500.0])
    assert masks["upper_bounds_km"] == pytest.approx([500.0, 540.0])
    assert masks["lower_ood"].tolist() == [False, True, False, False, False, False, False]
    assert masks["in_band"].tolist() == [False, False, True, True, True, False, False]
    assert masks["upper_ood"].tolist() == [False, False, False, False, False, True, False]


def test_evaluate_rejects_degree_max_mismatch_early(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model_dir = tmp_path / "model"
    (model_dir / "checkpoints").mkdir(parents=True)
    (model_dir / "config.json").write_text(
        """
        {
          "resolved_mu_si": 4904869500000.0,
          "resolved_r_ref_m": 1738000.0,
          "resolved_a_sign": 1.0,
          "degree_min": 20,
          "degree_max": 100,
          "central_body": "moon",
          "dataset_meta": {
            "degree_min": 20,
            "degree_max": 100,
            "target_mode": "residual",
            "mu_si": 4904869500000.0,
            "r_ref_m": 1738000.0,
            "central_body": "moon",
            "alt_min_km": 100.0,
            "alt_max_km": 500.0
          }
        }
        """.strip(),
        encoding="utf-8",
    )
    (model_dir / "scaler.json").write_text("{}", encoding="utf-8")
    (model_dir / "checkpoints" / "ckpt_best.pt").write_text("placeholder", encoding="utf-8")

    data_path = tmp_path / "eval_cloud.h5"
    _write_cloud(data_path, body="moon", mu_si=float(MU_MOON), r_ref_m=float(R_MOON))
    with h5py.File(data_path, "a") as handle:
        handle.attrs["requested_degree"] = 180
        handle.attrs["degree_max"] = 180
        handle.attrs["alt_min_km"] = 100.0
        handle.attrs["alt_max_km"] = 500.0

    class _DummyScaler:
        pass

    monkeypatch.setattr("surrogate_gravity_model.st_lrps_evaluate.ScalerPack.load", lambda *args, **kwargs: _DummyScaler())

    with pytest.raises(ValueError, match="degree_max"):
        evaluate(
            model_dir=model_dir,
            data_path=data_path,
            out_dir=tmp_path / "eval_out",
            device=pytest.importorskip("torch").device("cpu"),
            batch_size=32,
            a_sign=1.0,
            r_ref_m=float(R_MOON),
            alt_bin_km=50.0,
            dataset_name="data",
            start=0,
            end=None,
            max_points_for_plots=1000,
        )
