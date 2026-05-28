# -*- coding: utf-8 -*-
"""
Reload-safety / evaluation-correctness regression tests for the ST-LRPS surrogate.

These guard the failure mode that motivated this work: training metrics looked
good but a reloaded evaluator was catastrophically bad, because a MultiScale
SIREN was silently reconstructed with different per-band frequencies (w0_bands)
than it was trained with — the state_dict matched by shape while the functional
model was wrong. They also lock in the vector-error acceleration metrics, the
GradNorm robustness fix, the robust scaler modes, the dataset-convention guard,
and the collocation-Laplacian / active-refinement safety changes.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from lunaris.surrogate.st_lrps.networks.models import (
    build_model_from_config,
    compute_architecture_signature,
    reconstruct_model_from_artifacts,
)
from lunaris.surrogate.st_lrps.shared.scaling import (
    IsometricScaleParams,
    ScalerPack,
    fit_scaler_streaming,
    OnlineIsometricStats,
)
from lunaris.surrogate.st_lrps.evaluation.cli import (
    _StreamingMetrics,
    _TopKErrors,
    _build_eval_warnings,
    predict_residual_u_a,
    reload_model_from_run_dir,
)

R_REF = 1.737e6
MU = 4.902800066e12

ARCH_FIELDS = (
    "activation", "hidden", "depth", "dropout", "use_residual_blocks", "n_bands",
    "degree_min", "degree_max", "w0_bands",
    "use_sh_encoding", "sh_encoding_degree", "sh_append_raw",
    "use_radial_separation", "radial_append_raw", "use_fourier",
)


def _tiny_scaler() -> ScalerPack:
    return ScalerPack(
        x=IsometricScaleParams(mean=[0.0, 0.0, 0.0], scale=2.0e6),
        u=IsometricScaleParams(mean=[0.0], scale=1.0e4),
        a=IsometricScaleParams(mean=[0.0, 0.0, 0.0], scale=1.0e-3),
        provenance={"alt_min_km": 50.0, "alt_max_km": 300.0},
    )


def _base_cfg(**overrides) -> dict:
    cfg = {
        "activation": "sine",
        "hidden": 16,
        "depth": 3,
        "dropout": 0.0,
        "use_residual_blocks": True,
        "n_bands": 1,
        "degree_min": 10,
        "degree_max": 60,
        "w0_bands": None,
        "use_sh_encoding": False,
        "sh_encoding_degree": 4,
        "sh_append_raw": True,
        "use_radial_separation": False,
        "radial_append_raw": False,
        "use_fourier": False,
        "fourier_n_features": 256,
        "fourier_sigma": 1.0,
        "fourier_seed": 42,
        "resolved_mu_si": MU,
        "resolved_a_sign": 1.0,
        "resolved_r_ref_m": R_REF,
        "target_mode": "residual",
        "central_body": "moon",
        "dataset_name": "data",
    }
    cfg.update(overrides)
    return cfg


def _make_run_dir(tmp_path: Path, build_cfg: dict, *, config_json_overrides: dict | None = None) -> Path:
    """Create a run dir (config.json + scaler.json + checkpoint) mimicking the engine.

    The checkpoint's ``config`` block is the AUTHORITATIVE architecture (what the
    weights were trained with). ``config_json_overrides`` lets a test deliberately
    desynchronise config.json from the checkpoint to exercise mismatch handling.
    """
    run_dir = tmp_path / "run"
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    model = build_model_from_config(build_cfg, device=torch.device("cpu"), dtype=torch.float32)
    scaler = _tiny_scaler()

    ckpt_config = dict(build_cfg)
    ckpt_config["w0_bands"] = list(getattr(model, "w0_bands", []) or []) or None
    ckpt_config["input_feature_dim"] = int(getattr(model, "input_feature_dim", 3))
    ckpt_config["embedding_type"] = str(getattr(model, "embedding_type", "raw"))
    ckpt_config["model_builder_version"] = str(getattr(model, "model_builder_version", "v3"))
    ckpt_config["architecture_signature"] = compute_architecture_signature(ckpt_config)

    # config.json defaults to matching the checkpoint, unless a test overrides it.
    cfg_json = dict(ckpt_config)
    if config_json_overrides:
        cfg_json.update(config_json_overrides)

    (run_dir / "config.json").write_text(json.dumps(cfg_json, indent=2), encoding="utf-8")
    scaler.save_json(run_dir / "scaler.json")
    torch.save(
        {
            "model": model.state_dict(),
            "config": ckpt_config,
            "scaler": asdict(scaler),
            "epoch": 7,
            "best_val": 1.234e-3,
            "resolved_mu_si": MU,
            "resolved_a_sign": 1.0,
            "resolved_r_ref_m": R_REF,
            "degree_min": build_cfg["degree_min"],
            "degree_max": build_cfg["degree_max"],
        },
        run_dir / "checkpoints" / "ckpt_best.pt",
    )
    return run_dir


def _rand_positions(n: int, seed: int = 0) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    r = R_REF + rng.uniform(60e3, 280e3, n)
    d = rng.standard_normal((n, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    return torch.tensor((r[:, None] * d), dtype=torch.float32)


# ---------------------------------------------------------------------------
# P0 — MultiScale SIREN reconstruction
# ---------------------------------------------------------------------------

def test_multiscale_siren_requires_degree_metadata_when_no_w0_bands():
    cfg = _base_cfg(n_bands=3, w0_bands=None)
    cfg.pop("degree_min")
    cfg.pop("degree_max")
    with pytest.raises(ValueError, match="w0_bands"):
        build_model_from_config(cfg)


def test_multiscale_siren_uses_explicit_w0_bands():
    bands = [12.0, 30.0, 77.0]
    cfg = _base_cfg(n_bands=3, w0_bands=bands)
    model = build_model_from_config(cfg)
    assert [round(b, 4) for b in model.w0_bands] == [round(b, 4) for b in bands]
    # explicit w0_bands of wrong length must fail
    with pytest.raises(ValueError, match="length"):
        build_model_from_config(_base_cfg(n_bands=2, w0_bands=[1.0, 2.0, 3.0]))


def test_checkpoint_reconstructs_same_w0_bands(tmp_path):
    cfg = _base_cfg(n_bands=3, degree_min=20, degree_max=200, w0_bands=None)
    # resolve the bands the engine would compute and bake them into the artifacts
    model0 = build_model_from_config(cfg)
    cfg = _base_cfg(n_bands=3, degree_min=20, degree_max=200, w0_bands=list(model0.w0_bands))
    run_dir = _make_run_dir(tmp_path, cfg)
    model, scaler, merged, report = reload_model_from_run_dir(run_dir, torch.device("cpu"))
    assert [round(b, 4) for b in model.w0_bands] == [round(b, 4) for b in model0.w0_bands]
    assert report["n_bands"] == 3
    assert report["model_w0_bands"] == [round(b, 4) for b in model0.w0_bands] or \
        [round(b, 4) for b in report["model_w0_bands"]] == [round(b, 4) for b in model0.w0_bands]


def test_evaluator_rejects_config_checkpoint_architecture_mismatch(tmp_path):
    # checkpoint trained with degrees 20/200 (so a 3-band spectrum); config.json
    # lies and says 0/50 with no w0_bands → reconstruction must fail loudly.
    model_ref = build_model_from_config(_base_cfg(n_bands=3, degree_min=20, degree_max=200))
    cfg = _base_cfg(n_bands=3, degree_min=20, degree_max=200, w0_bands=list(model_ref.w0_bands))
    run_dir = _make_run_dir(
        tmp_path, cfg,
        config_json_overrides={"degree_min": 0, "degree_max": 50, "w0_bands": None},
    )
    cfg_json = json.loads((run_dir / "config.json").read_text())
    ckpt = torch.load(run_dir / "checkpoints" / "ckpt_best.pt", weights_only=False)
    with pytest.raises(RuntimeError, match="architecture"):
        reconstruct_model_from_artifacts(cfg_json, ckpt, torch.device("cpu"))
    # ...but the override flag allows it (uses the checkpoint architecture).
    model, merged, report = reconstruct_model_from_artifacts(
        cfg_json, ckpt, torch.device("cpu"), allow_config_mismatch=True
    )
    assert [round(b, 4) for b in model.w0_bands] == [round(b, 4) for b in model_ref.w0_bands]
    assert report["architecture_mismatch_fields"]


def test_force_model_rejects_config_checkpoint_architecture_mismatch(tmp_path):
    from lunaris.surrogate.st_lrps.runtime.force_model import load_surrogate_force_model

    model_ref = build_model_from_config(_base_cfg(n_bands=2, degree_min=15, degree_max=150))
    cfg = _base_cfg(n_bands=2, degree_min=15, degree_max=150, w0_bands=list(model_ref.w0_bands))
    run_dir = _make_run_dir(
        tmp_path, cfg,
        config_json_overrides={"degree_min": 0, "degree_max": 40, "w0_bands": None},
    )
    with pytest.raises(RuntimeError, match="architecture"):
        load_surrogate_force_model(run_dir, device="cpu")
    fm = load_surrogate_force_model(run_dir, device="cpu", allow_config_mismatch=True)
    assert fm is not None


# ---------------------------------------------------------------------------
# P0 — reload parity
# ---------------------------------------------------------------------------

def test_reload_prediction_matches_pre_save_prediction(tmp_path):
    cfg = _base_cfg(n_bands=3, degree_min=20, degree_max=180)
    model0 = build_model_from_config(cfg)
    cfg = _base_cfg(n_bands=3, degree_min=20, degree_max=180, w0_bands=list(model0.w0_bands))
    run_dir = _make_run_dir(tmp_path, cfg)

    # Pre-save prediction using the freshly built artifacts.
    scaler = ScalerPack.load_json(run_dir / "scaler.json").to_tensors(torch.device("cpu"), torch.float32)
    pre_model = build_model_from_config(cfg, device=torch.device("cpu"), dtype=torch.float32)
    ckpt = torch.load(run_dir / "checkpoints" / "ckpt_best.pt", weights_only=False)
    pre_model.load_state_dict(ckpt["model"], strict=False)
    pre_model.eval()
    x = _rand_positions(64, seed=1)
    du_pre, da_pre = predict_residual_u_a(pre_model, scaler, x, a_sign=1.0)

    # Reload via the canonical evaluator path.
    model, scaler2, merged, report = reload_model_from_run_dir(run_dir, torch.device("cpu"))
    du_post, da_post = predict_residual_u_a(model, scaler2, x, a_sign=1.0)

    assert torch.allclose(du_pre, du_post, atol=1e-5, rtol=1e-4)
    assert torch.allclose(da_pre, da_post, atol=1e-7, rtol=1e-4)

    # No 90-degree directional collapse between the two predictions.
    cos = torch.nn.functional.cosine_similarity(da_pre, da_post, dim=-1)
    assert float(cos.mean()) > 0.999


def test_eval_on_training_batch_matches_loss_function_prediction(tmp_path):
    # The residual dU predicted by the evaluation utility must equal a direct
    # loss-style forward (model(scale_x(x)) then unscale) — same code path.
    cfg = _base_cfg(n_bands=1, degree_min=10, degree_max=60)
    run_dir = _make_run_dir(tmp_path, cfg)
    model, scaler, merged, report = reload_model_from_run_dir(run_dir, torch.device("cpu"))
    x = _rand_positions(48, seed=3)

    du_util, _ = predict_residual_u_a(model, scaler, x, a_sign=1.0)
    with torch.no_grad():
        du_manual = scaler.unscale_u(model(scaler.scale_x(x)))
    assert torch.allclose(du_util, du_manual, atol=1e-6, rtol=1e-5)


def test_eval_val_subset_does_not_use_wrong_architecture(tmp_path):
    # Reconstruction must reflect the checkpoint architecture, not config.json,
    # when they disagree on a non-fatal-by-shape field.
    cfg = _base_cfg(n_bands=3, degree_min=20, degree_max=200)
    model0 = build_model_from_config(cfg)
    cfg = _base_cfg(n_bands=3, degree_min=20, degree_max=200, w0_bands=list(model0.w0_bands))
    run_dir = _make_run_dir(tmp_path, cfg)
    model, scaler, merged, report = reload_model_from_run_dir(run_dir, torch.device("cpu"))
    # merged config carries the checkpoint's spectrum
    assert [round(b, 4) for b in merged["w0_bands"]] == [round(b, 4) for b in model0.w0_bands]
    assert report["checkpoint_config_source"] == "checkpoint"


# ---------------------------------------------------------------------------
# P0 — acceleration evaluation metrics
# ---------------------------------------------------------------------------

def test_streaming_metrics_vector_error_detects_orthogonal_vectors():
    sm = _StreamingMetrics(n_alt_bins=4, alt_min_km=0.0, alt_max_km=400.0)
    x = np.array([[R_REF + 100e3, 0.0, 0.0]], dtype=np.float64)
    a_true = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
    a_pred = np.array([[0.0, 1.0, 0.0]], dtype=np.float64)  # orthogonal, same magnitude
    sm.update(x, a_true, a_pred, np.array([0.0]), np.array([0.0]), R_REF)
    res = sm.finalize()
    # magnitude error is ~0 (both unit vectors) but vector error is sqrt(2), 90 deg.
    assert res["mae_a_mag"] < 1e-9
    assert abs(res["mae_a_vec"] - np.sqrt(2.0)) < 1e-9
    assert abs(res["mae_a"] - res["mae_a_vec"]) < 1e-12     # alias points at vector
    assert abs(res["mean_ang_deg"] - 90.0) < 1e-6


def test_streaming_metrics_mag_error_can_be_zero_when_vector_error_nonzero():
    sm = _StreamingMetrics(n_alt_bins=2, alt_min_km=0.0, alt_max_km=400.0)
    rng = np.random.default_rng(0)
    n = 50
    x = np.tile([[R_REF + 100e3, 0.0, 0.0]], (n, 1)).astype(np.float64)
    dirs = rng.standard_normal((n, 3)); dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    a_true = dirs * 1e-3
    # same magnitude, random direction → magnitude error ~0, vector error large
    pred_dirs = rng.standard_normal((n, 3)); pred_dirs /= np.linalg.norm(pred_dirs, axis=1, keepdims=True)
    a_pred = pred_dirs * 1e-3
    sm.update(x, a_true, a_pred, np.zeros(n), np.zeros(n), R_REF)
    res = sm.finalize()
    assert res["mae_a_mag"] < 1e-12
    assert res["mae_a_vec"] > 1e-4


def test_altitude_bins_use_vector_error():
    sm = _StreamingMetrics(n_alt_bins=3, alt_min_km=0.0, alt_max_km=300.0)
    x = np.array([[R_REF + 50e3, 0.0, 0.0]], dtype=np.float64)
    a_true = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
    a_pred = np.array([[0.0, 1.0, 0.0]], dtype=np.float64)
    sm.update(x, a_true, a_pred, np.array([0.0]), np.array([0.0]), R_REF)
    res = sm.finalize()
    # default alt_bin metric == vector metric, distinct from magnitude metric
    assert res["alt_bin_rmse_a"] == res["alt_bin_rmse_a_vec"]
    nz = [v for v in res["alt_bin_rmse_a_vec"] if v > 0]
    assert nz and abs(nz[0] - np.sqrt(2.0)) < 1e-9
    assert all(v == 0 for v in res["alt_bin_rmse_a_mag"])


def test_topk_errors_include_angle_and_cosine():
    tk = _TopKErrors(5)
    rng = np.random.default_rng(2)
    n = 40
    x = _rand_positions(n, seed=5).numpy().astype(np.float64)
    a_true = rng.standard_normal((n, 3)) * 1e-3
    a_pred = a_true + rng.standard_normal((n, 3)) * 1e-4
    tk.update_batch(x, np.zeros(n), np.zeros(n), a_true, a_pred, R_REF)
    arr = tk.to_array()
    assert arr.shape[1] == 16
    cos_col, ang_col = arr[:, 14], arr[:, 15]
    assert np.all(np.abs(cos_col) <= 1.0 + 1e-6)
    assert np.all((ang_col >= 0.0) & (ang_col <= 180.0 + 1e-6))


# ---------------------------------------------------------------------------
# P1 — GradNorm robustness
# ---------------------------------------------------------------------------

def _gradnorm_setup():
    from lunaris.surrogate.st_lrps.training.losses import GradNormWeights
    p = torch.nn.Parameter(torch.randn(8, 4))
    return GradNormWeights(mode="ntk_init", w_a=1.0, w_a_min=0.35, w_a_max=4.0), p


def test_gradnorm_all_none_grad_a_keeps_current_weight():
    gn, p = _gradnorm_setup()
    loss_u = (p ** 2).sum()                # depends on p
    loss_a = (torch.randn(3, requires_grad=True) ** 2).sum()  # disconnected from p
    out = gn._compute_grad_norm_ratio(loss_u, loss_a, [p])
    assert gn.last_gradnorm_status == "empty_grad_a"
    assert out == pytest.approx(1.0)       # unchanged current w_a


def test_gradnorm_zero_norm_a_keeps_current_weight():
    gn, p = _gradnorm_setup()
    loss_u = (p ** 2).sum()
    loss_a = (p * 0.0).sum()               # connected but zero gradient
    out = gn._compute_grad_norm_ratio(loss_u, loss_a, [p])
    assert gn.last_gradnorm_status == "zero_norm_a"
    assert out == pytest.approx(1.0)


def test_gradnorm_valid_ratio_still_clamps_normally():
    gn, p = _gradnorm_setup()
    # large loss_u gradient, tiny loss_a gradient → raw ratio huge → clamp to max
    loss_u = (p ** 2).sum() * 1e6
    loss_a = (p ** 2).sum() * 1e-6
    out = gn._compute_grad_norm_ratio(loss_u, loss_a, [p])
    assert gn.last_gradnorm_status == "ok"
    assert out == pytest.approx(gn.w_a_max)
    assert gn.w_a_min <= out <= gn.w_a_max


# ---------------------------------------------------------------------------
# P1 — scaler robustness
# ---------------------------------------------------------------------------

def _write_scaler_h5(tmp_path: Path, *, outlier: bool) -> Path:
    import h5py
    rng = np.random.default_rng(0)
    n = 2000
    r = R_REF + rng.uniform(50e3, 200e3, n)
    d = rng.standard_normal((n, 3)); d /= np.linalg.norm(d, axis=1, keepdims=True)
    x = (r[:, None] * d).astype(np.float32)
    u = (rng.standard_normal(n) * 10.0).astype(np.float32)
    a = (rng.standard_normal((n, 3)) * 1e-4).astype(np.float32)
    if outlier:
        u[0] = 1.0e6      # single huge outlier
        a[0] = [1.0, 1.0, 1.0]
    data = np.concatenate([x, u[:, None], a], axis=1).astype(np.float32)
    p = tmp_path / ("cloud_out.h5" if outlier else "cloud.h5")
    with h5py.File(p, "w") as hf:
        hf.create_dataset("data", data=data)
        hf.attrs["unit_system"] = "si"
        hf.attrs["mu_si"] = MU
        hf.attrs["r_ref_m"] = R_REF
        hf.attrs["alt_min_km"] = 50.0
        hf.attrs["alt_max_km"] = 200.0
    return p


def test_scaler_hybrid_less_sensitive_to_outlier_than_max(tmp_path):
    from lunaris.surrogate.st_lrps.data.datasets import DatasetMeta
    p = _write_scaler_h5(tmp_path, outlier=True)
    meta = DatasetMeta.from_h5(p)
    sc_max = fit_scaler_streaming(p, "data", meta, use_si=False, mu_si=MU, a_sign=1.0,
                                  n_fit=2000, seed=0, degree_min=-1,
                                  u_scale_mode="max", a_scale_mode="max")
    sc_hyb = fit_scaler_streaming(p, "data", meta, use_si=False, mu_si=MU, a_sign=1.0,
                                  n_fit=2000, seed=0, degree_min=-1,
                                  u_scale_mode="hybrid", a_scale_mode="hybrid",
                                  target_scale_multiplier=6.0)
    # the single huge outlier inflates the max-norm scale; hybrid resists it.
    assert sc_hyb.u.scale < sc_max.u.scale


def test_scaler_provenance_records_modes(tmp_path):
    from lunaris.surrogate.st_lrps.data.datasets import DatasetMeta
    p = _write_scaler_h5(tmp_path, outlier=False)
    meta = DatasetMeta.from_h5(p)
    sc = fit_scaler_streaming(p, "data", meta, use_si=False, mu_si=MU, a_sign=1.0,
                              n_fit=2000, seed=0, degree_min=-1,
                              u_scale_mode="hybrid", a_scale_mode="rms",
                              target_scale_multiplier=5.0)
    assert sc.provenance["u_scale_mode"] == "hybrid"
    assert sc.provenance["a_scale_mode"] == "rms"
    assert sc.provenance["target_scale_multiplier"] == pytest.approx(5.0)


def test_fit_scaler_streaming_uses_requested_scale_modes(tmp_path):
    from lunaris.surrogate.st_lrps.data.datasets import DatasetMeta
    p = _write_scaler_h5(tmp_path, outlier=True)
    meta = DatasetMeta.from_h5(p)
    sc_rms = fit_scaler_streaming(p, "data", meta, use_si=False, mu_si=MU, a_sign=1.0,
                                  n_fit=2000, seed=0, degree_min=-1,
                                  u_scale_mode="rms", a_scale_mode="rms",
                                  target_scale_multiplier=6.0)
    sc_max = fit_scaler_streaming(p, "data", meta, use_si=False, mu_si=MU, a_sign=1.0,
                                  n_fit=2000, seed=0, degree_min=-1,
                                  u_scale_mode="max", a_scale_mode="max")
    assert sc_rms.provenance["u_scale_mode"] == "rms"
    assert sc_max.provenance["u_scale_mode"] == "max"
    # rms scale (bulk-based) is much smaller than max scale on outlier data
    assert sc_rms.u.scale < sc_max.u.scale


# ---------------------------------------------------------------------------
# P1 — dataset convention validation
# ---------------------------------------------------------------------------

def _write_conv_h5(tmp_path, *, deriv_conv, degree_min=20, degree_max=100, name="c.h5"):
    import h5py
    p = tmp_path / name
    with h5py.File(p, "w") as hf:
        hf.create_dataset("data", data=np.zeros((8, 7), dtype=np.float32))
        hf.attrs["unit_system"] = "si"
        hf.attrs["mu_si"] = MU
        hf.attrs["r_ref_m"] = R_REF
        hf.attrs["central_body"] = "moon"
        hf.attrs["degree_min"] = degree_min
        hf.attrs["degree_max"] = degree_max
        hf.attrs["target_mode"] = "residual"
        if deriv_conv is not None:
            hf.attrs["derivative_convention_version"] = deriv_conv
    return p


def test_missing_derivative_convention_raises(tmp_path):
    from lunaris.surrogate.st_lrps.data.datasets import (
        DatasetMeta, validate_training_dataset_convention,
    )
    p = _write_conv_h5(tmp_path, deriv_conv=None)
    meta = DatasetMeta.from_h5(p)
    with pytest.raises(ValueError, match="derivative_convention"):
        validate_training_dataset_convention(meta, data_path=p)


def test_wrong_derivative_convention_raises(tmp_path):
    from lunaris.surrogate.st_lrps.data.datasets import (
        DatasetMeta, validate_training_dataset_convention,
    )
    p = _write_conv_h5(tmp_path, deriv_conv="legacy_v0")
    meta = DatasetMeta.from_h5(p)
    with pytest.raises(ValueError, match="derivative_convention"):
        validate_training_dataset_convention(meta, data_path=p)


def test_allow_legacy_derivative_convention_only_with_flag(tmp_path):
    from lunaris.surrogate.st_lrps.data.datasets import (
        DatasetMeta, validate_training_dataset_convention,
    )
    p = _write_conv_h5(tmp_path, deriv_conv=None)
    meta = DatasetMeta.from_h5(p)
    with pytest.raises(ValueError):
        validate_training_dataset_convention(meta, data_path=p,
                                             allow_legacy_derivative_convention=False)
    # with the flag it must NOT raise (inspection mode)
    validate_training_dataset_convention(meta, data_path=p,
                                         allow_legacy_derivative_convention=True)


def test_degree_zero_parse_does_not_become_silent_none(tmp_path):
    import h5py
    from lunaris.surrogate.st_lrps.data.datasets import DatasetMeta
    p = tmp_path / "deg0.h5"
    with h5py.File(p, "w") as hf:
        hf.create_dataset("data", data=np.zeros((4, 7), dtype=np.float32))
        hf.attrs["unit_system"] = "si"
        hf.attrs["cloud_config_json"] = json.dumps({"degree_max": 0, "degree_min": -1})
    meta = DatasetMeta.from_h5(p)
    # degree_max=0 must survive as 0, not collapse to None via `int(v) or None`.
    assert meta.requested_degree == 0
    assert meta.degree_max == 0


def test_degree_max_not_greater_than_min_raises(tmp_path):
    from lunaris.surrogate.st_lrps.data.datasets import (
        DatasetMeta, validate_training_dataset_convention,
    )
    p = _write_conv_h5(tmp_path, deriv_conv="dP_dphi_corrected_v1",
                       degree_min=50, degree_max=50)
    meta = DatasetMeta.from_h5(p)
    with pytest.raises(ValueError, match="degree_max"):
        validate_training_dataset_convention(meta, data_path=p)


# ---------------------------------------------------------------------------
# P1 — evaluation reporting
# ---------------------------------------------------------------------------

def test_report_warns_when_magnitude_good_but_direction_bad():
    sm_res = {"mean_cos_sim": 0.02, "mean_ang_deg": 89.0,
              "mae_a_vec": 1.0e-3, "mae_a_mag": 1.0e-9}
    warnings = _build_eval_warnings(sm_res, {"architecture_mismatch_fields": []})
    joined = " ".join(warnings).lower()
    assert "cosine" in joined or "direction" in joined
    assert any("magnitude" in w.lower() and "vector" in w.lower() for w in warnings)


def test_evaluation_report_contains_architecture_and_metric_blocks(tmp_path):
    import h5py
    from lunaris.surrogate.st_lrps.evaluation.cli import evaluate
    # tiny single-scale residual run + matching tiny dataset
    cfg = _base_cfg(n_bands=1, degree_min=10, degree_max=60)
    run_dir = _make_run_dir(tmp_path, cfg)

    # dataset must agree with model degree_min/degree_max (eval contract checks).
    n = 256
    rng = np.random.default_rng(0)
    r = R_REF + rng.uniform(60e3, 280e3, n)
    d = rng.standard_normal((n, 3)); d /= np.linalg.norm(d, axis=1, keepdims=True)
    x = (r[:, None] * d).astype(np.float64)
    u = (rng.standard_normal(n) * 1e3).astype(np.float64)
    a = (rng.standard_normal((n, 3)) * 1e-5).astype(np.float64)
    data = np.concatenate([x, u[:, None], a], axis=1)
    ds = tmp_path / "eval.h5"
    with h5py.File(ds, "w") as hf:
        hf.create_dataset("data", data=data)
        hf.attrs["unit_system"] = "si"
        hf.attrs["mu_si"] = MU
        hf.attrs["r_ref_m"] = R_REF
        hf.attrs["central_body"] = "moon"
        hf.attrs["degree_min"] = 10
        hf.attrs["degree_max"] = 60
        hf.attrs["target_mode"] = "residual"

    out_dir = tmp_path / "eval_out"
    evaluate(
        model_dir=run_dir, data_path=ds, out_dir=out_dir,
        device=torch.device("cpu"), batch_size=64, a_sign=1.0, r_ref_m=R_REF,
        alt_bin_km=50.0, dataset_name="data", streaming=True,
    )
    report = json.loads((out_dir / "eval_report.json").read_text())
    m = report["metrics"]
    for block in ("residual_vector_metrics", "residual_magnitude_metrics",
                  "residual_angular_metrics", "total_approx_metrics"):
        assert block in m, f"missing block: {block}"
    assert "architecture_signature" in m
    assert "checkpoint_path" in m
    assert m["checkpoint_config_source"] in ("checkpoint", "config_json")
    assert "warnings" in m
    assert m["|a|"]["error_kind"] == "vector"


# ---------------------------------------------------------------------------
# P2 — collocation Laplacian failure accounting
# ---------------------------------------------------------------------------

def _make_lap_trainer(mode):
    from lunaris.surrogate.st_lrps.training.engine import STLRPSTrainer
    from lunaris.surrogate.st_lrps.training.config import TrainConfig
    from lunaris.surrogate.st_lrps.training.losses import SobolevLoss, GradNormWeights
    sp = _tiny_scaler().to_tensors(torch.device("cpu"), torch.float32)
    model = torch.nn.Sequential(torch.nn.Linear(3, 8), torch.nn.Tanh(), torch.nn.Linear(8, 1))
    cfg = TrainConfig(data="/tmp/x.h5", out="/tmp/o", epochs=1, batch_size=8,
                      laplacian_mode=mode, collocation_laplacian_weight=1e-2,
                      collocation_laplacian_every=1, collocation_laplacian_samples=8,
                      collocation_laplacian_hutchinson_samples=2, amp=False)
    trainer = STLRPSTrainer(model, SobolevLoss(sp, a_sign=1.0),
                            torch.optim.SGD(model.parameters(), lr=0.0),
                            GradNormWeights(mode="fixed"), torch.device("cpu"), cfg,
                            collocation_r_min_m=1.837e6, collocation_r_max_m=1.937e6)
    return trainer


def _lap_loader():
    from torch.utils.data import DataLoader, TensorDataset
    g = torch.Generator(); g.manual_seed(0)
    x = torch.randn(8, 3, generator=g) * 1.85e6
    u = torch.randn(8, 1, generator=g)
    a = torch.randn(8, 3, generator=g) * 1e-3
    return DataLoader(TensorDataset(x, u, a), batch_size=8)


def test_collocation_laplacian_failure_count_recorded(monkeypatch):
    import lunaris.surrogate.st_lrps.training.engine as eng

    def _boom(*args, **kwargs):
        raise RuntimeError("synthetic collocation failure")

    monkeypatch.setattr(eng, "collocation_laplacian_loss", _boom)
    trainer = _make_lap_trainer("diagnostic")
    res = trainer.run_epoch(_lap_loader(), is_train=True, epoch=0)
    assert res["collocation_laplacian_attempt_count"] >= 1
    assert res["collocation_laplacian_fail_count"] >= 1
    assert res["collocation_laplacian_success_count"] == 0


def test_laplacian_train_mode_failure_raises(monkeypatch):
    import lunaris.surrogate.st_lrps.training.engine as eng

    def _boom(*args, **kwargs):
        raise RuntimeError("synthetic collocation failure")

    monkeypatch.setattr(eng, "collocation_laplacian_loss", _boom)
    trainer = _make_lap_trainer("train")
    with pytest.raises(RuntimeError, match="train mode"):
        trainer.run_epoch(_lap_loader(), is_train=True, epoch=0)


# ---------------------------------------------------------------------------
# P2 — active refinement altitude safety
# ---------------------------------------------------------------------------

def _error_csv(tmp_path: Path, n=5) -> Path:
    import csv
    rng = np.random.default_rng(0)
    header = ("x,y,z,u_true,u_pred,ax_true,ay_true,az_true,ax_pred,ay_pred,az_pred,"
              "abs_a_error,rel_a_error,altitude_km")
    p = tmp_path / "errors.csv"
    with open(p, "w", newline="") as f:
        f.write(header + "\n")
        w = csv.writer(f)
        for _ in range(n):
            r = R_REF + rng.uniform(60e3, 250e3)
            d = rng.standard_normal(3); d /= np.linalg.norm(d)
            x, y, z = (r * d).tolist()
            w.writerow([x, y, z] + [0.0] * 8 + [1e-4, 0.01, (r - R_REF) / 1000.0])
    return p


def _active_ns(tmp_path, **overrides):
    import argparse
    ns = argparse.Namespace(
        active_from_error_points=str(_error_csv(tmp_path)),
        active_jitter_radial_km=5.0, active_jitter_tangent_km=10.0,
        active_samples_per_point=4, active_max_source_points=5,
        active_gfc_file=None, active_degree_max=None, active_degree_min=None,
        active_out=None, active_seed=42,
        active_clip_to_alt_range=False, active_reject_outside_alt_range=False,
        active_save_positions_only=True, out=str(tmp_path),
        degree_max=10, degree_min=-1, format="h5",
    )
    for k, v in overrides.items():
        setattr(ns, k, v)
    return ns


class _AP:
    @staticmethod
    def error(msg):
        raise SystemExit(f"error: {msg}")


def test_active_refinement_requires_altitude_bounds(tmp_path):
    from lunaris.surrogate.st_lrps.data.spatial_cloud_generator import _run_active_refinement
    ns = _active_ns(tmp_path, active_reject_outside_alt_range=True)  # no alt bounds anywhere
    with pytest.raises(ValueError, match="altitude"):
        _run_active_refinement(ns, _AP())


def test_active_refinement_uses_dataset_altitude_metadata(tmp_path):
    import h5py
    from lunaris.surrogate.st_lrps.data.spatial_cloud_generator import _resolve_active_alt_bounds
    src = tmp_path / "src.h5"
    with h5py.File(src, "w") as hf:
        hf.create_dataset("data", data=np.zeros((4, 7), dtype=np.float32))
        hf.attrs["alt_min_km"] = 40.0
        hf.attrs["alt_max_km"] = 260.0
    import argparse
    ns = argparse.Namespace(active_source_dataset=str(src))
    lo, hi, src_label = _resolve_active_alt_bounds(ns)
    assert lo == pytest.approx(40.0)
    assert hi == pytest.approx(260.0)
    assert src_label.startswith("dataset:")


def test_active_refinement_rejects_out_of_shell_points(tmp_path):
    # With explicit bounds + reject, the saved positions must lie inside the shell.
    from lunaris.surrogate.st_lrps.data.spatial_cloud_generator import _run_active_refinement
    ns = _active_ns(
        tmp_path,
        active_reject_outside_alt_range=True,
        altitude_min_km=80.0, altitude_max_km=200.0,
        active_jitter_radial_km=1.0, active_jitter_tangent_km=1.0,
    )
    _run_active_refinement(ns, _AP())
    data = np.load(tmp_path / "active_refinement_positions.npz")
    x = data["x"]; data.close()
    alt_km = (np.linalg.norm(x, axis=1) - R_REF) / 1000.0
    # allow a tiny numerical margin around the shell
    assert alt_km.min() >= 80.0 - 1.0
    assert alt_km.max() <= 200.0 + 1.0
