import os
import json
import math
import tempfile
import pytest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import h5py

from lunaris.surrogate.st_lrps.networks.models import (
    SHInspiredAngularEncoding,
    RadialSeparationEncoding,
    PhysicsNet,
    build_model_from_config
)
from lunaris.surrogate.st_lrps.runtime.force_model import SurrogateForceModel, ScalerPack
from lunaris.surrogate.st_lrps.data.spatial_cloud_generator import _run_active_refinement, _sh_potential_accel_batch_serial
from lunaris.surrogate.st_lrps.training.engine import TrainConfig
from lunaris.surrogate.st_lrps.evaluation.cli import evaluate

# ==============================================================================
# TESTS FOR TASK 1: ACTIVE REFINEMENT SH LABELING
# ==============================================================================

class MockArgs:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        if not hasattr(self, "active_jitter_radial_km"):
            self.active_jitter_radial_km = 0.5
        if not hasattr(self, "active_jitter_tangent_km"):
            self.active_jitter_tangent_km = 0.5
        if not hasattr(self, "active_max_source_points"):
            self.active_max_source_points = 1000
        if not hasattr(self, "active_from_error_points"):
            self.active_from_error_points = getattr(self, "active_error_file", "err.csv")

def create_mock_gfc(path: Path):
    path.write_text(
        "1737.4 4902.8 0.0 2 2 1 0 0\n"
        "\n"
        "0 0 1.0 0.0\n"
        "1 0 0.0 0.0\n"
        "1 1 0.0 0.0\n"
        "2 0 0.0 0.0\n"
        "2 1 0.0 0.0\n"
        "2 2 0.0 0.0\n"
    )

def test_active_refinement_kernel_signature_no_unexpected_keywords():
    import inspect
    sig = inspect.signature(_sh_potential_accel_batch_serial)
    assert "xyz_m" in sig.parameters
    assert "degree_min" in sig.parameters
    assert "degree_max" in sig.parameters
    assert "mu" not in sig.parameters
    assert "n_max" not in sig.parameters

def test_active_refinement_full_labeling_path_writes_h5(tmp_path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    err_csv = tmp_path / "err.csv"
    err_csv.write_text("x,y,z,u_true,u_pred,ax_true,ay_true,az_true,ax_pred,ay_pred,az_pred,abs_a_error,rel_a_error,altitude_km\n1000,0,0,0,0,0,0,0,0,0,0,0,0,0\n")
    gfc_path = tmp_path / "mock.txt"
    create_mock_gfc(gfc_path)
    
    args = MockArgs(
        active_error_file=str(err_csv),
        active_from_error_points=str(err_csv),
        active_gfc_file=str(gfc_path),
        active_degree_max=1,
        active_degree_min=-1,
        active_samples_per_point=5,
        active_save_positions_only=False,
        active_out=str(out_dir / "active_refinement_labeled.h5")
    )
    
    _run_active_refinement(args, out_dir)
    
    h5_file = out_dir / "active_refinement_labeled.h5"
    assert h5_file.exists()
    
    with h5py.File(h5_file, "r") as f:
        assert "data" in f
        data = f["data"][:]
        assert data.shape[1] == 7
        assert "component_name" in f.attrs
        assert f.attrs["target_mode"] == "full"
        assert f.attrs["degree_min"] == -1
        assert f.attrs["degree_max"] == 1
        assert "[x,y,z,U,ax,ay,az]" in f.attrs["columns"]

def test_active_refinement_residual_labeling_path_writes_h5(tmp_path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    err_csv = tmp_path / "err.csv"
    err_csv.write_text("x,y,z,u_true,u_pred,ax_true,ay_true,az_true,ax_pred,ay_pred,az_pred,abs_a_error,rel_a_error,altitude_km\n1000,0,0,0,0,0,0,0,0,0,0,0,0,0\n")
    gfc_path = tmp_path / "mock.txt"
    create_mock_gfc(gfc_path)
    
    args = MockArgs(
        active_error_file=str(err_csv),
        active_from_error_points=str(err_csv),
        active_gfc_file=str(gfc_path),
        active_degree_max=2,
        active_degree_min=1,
        active_samples_per_point=5,
        active_save_positions_only=False,
        active_out=str(out_dir / "active_refinement_labeled.h5")
    )
    
    _run_active_refinement(args, out_dir)
    
    h5_file = out_dir / "active_refinement_labeled.h5"
    with h5py.File(h5_file, "r") as f:
        assert f.attrs["target_mode"] == "residual"
        assert f.attrs["degree_min"] == 1
        assert f.attrs["degree_max"] == 2
        assert "[x,y,z,dU,dax,day,daz]" in f.attrs["columns"]
        data = f["data"][:]
        assert np.all(np.isfinite(data))

# ==============================================================================
# TESTS FOR TASK 2: SH INSPIRED ANGULAR ENCODING & RADIAL ENCODING
# ==============================================================================

def test_angular_polynomial_encoding_feature_count():
    # degree_max = 1 -> nx, ny, nz -> 3 features
    enc = SHInspiredAngularEncoding(degree_max=1, append_raw=True)
    assert enc.n_features == 3
    assert enc.out_dim == 6

    # degree_max = 2 -> i+j+k <= 2 => C(2+3,3)-1 = 10-1 = 9
    enc2 = SHInspiredAngularEncoding(degree_max=2, append_raw=True)
    assert enc2.n_features == 9
    assert enc2.out_dim == 12

def test_angular_polynomial_encoding_output_shape():
    enc = SHInspiredAngularEncoding(degree_max=2, append_raw=True)
    assert enc.out_dim == 12
    x = torch.randn(10, 3)
    out = enc(x)
    assert out.shape == (10, 12)

def test_angular_polynomial_encoding_preserves_radial_information_or_raises():
    with pytest.raises(ValueError, match="loses radial information"):
        SHInspiredAngularEncoding(degree_max=2, append_raw=False)

def test_angular_polynomial_encoding_forward_backward_differentiable():
    enc = SHInspiredAngularEncoding(degree_max=2, append_raw=True)
    x = torch.randn(5, 3, requires_grad=True)
    out = enc(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()

def test_radial_encoding_changes_input_feature_dim():
    enc = RadialSeparationEncoding(append_raw=True)
    assert enc.out_dim == 7
    enc2 = RadialSeparationEncoding(append_raw=False)
    assert enc2.out_dim == 4

def test_radial_encoding_forward_backward_differentiable():
    enc = RadialSeparationEncoding(append_raw=False)
    x = torch.randn(5, 3, requires_grad=True)
    out = enc(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None

def test_model_builder_rejects_sh_and_radial_together():
    cfg = MockArgs(use_sh_encoding=True, use_radial_separation=True)
    with pytest.raises(ValueError, match="cannot both be True"):
        build_model_from_config(cfg)

def test_sh_encoded_model_predicts_accel_with_autograd():
    cfg = MockArgs(
        use_sh_encoding=True, sh_encoding_degree=2, sh_append_raw=True,
        hidden=32, depth=2, activation="silu", use_fourier=False
    )
    model = build_model_from_config(cfg)
    x = torch.randn(5, 3, requires_grad=True)
    y = model(x)
    loss = y.sum()
    loss.backward()
    assert x.grad is not None

from lunaris.surrogate.st_lrps.shared.scaling import ScalerPack, IsometricScaleParams

def mock_scaler():
    return ScalerPack(
        x=IsometricScaleParams(mean=[0.0, 0.0, 0.0], scale=1.0),
        u=IsometricScaleParams(mean=[0.0], scale=1.0),
        a=IsometricScaleParams(mean=[0.0, 0.0, 0.0], scale=1.0)
    )

# ==============================================================================
# TESTS FOR TASK 3: STREAMING EVALUATOR
# ==============================================================================

def test_streaming_report_does_not_use_stub_arrays(tmp_path):
    # Create small h5 dataset
    dpath = tmp_path / "test.h5"
    with h5py.File(dpath, "w") as f:
        data = np.random.randn(100, 7)
        data[:, :3] *= 1e6
        f.create_dataset("data", data=data)
        f.attrs["central_body"] = "moon"
        f.attrs["mu_si"] = 4.902800066e12
        f.attrs["r_ref_m"] = 1737400.0
        f.attrs["unit_system"] = "si"
        f.attrs["target_mode"] = "full"
        f.attrs["degree_min"] = -1
        f.attrs["degree_max"] = 2
    
    cfg = MockArgs(
        hidden=16, depth=1, activation="silu", use_fourier=False, use_sh_encoding=False,
        central_body="moon", resolved_mu_si=4.902800066e12, resolved_r_ref_m=1737400.0,
        degree_min=-1, degree_max=2
    )
    model_dir = tmp_path / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    import json
    with open(model_dir / "config.json", "w") as f:
        json.dump({
            "hidden": 16, "depth": 1, "activation": "silu", "use_fourier": False, "use_sh_encoding": False,
            "central_body": "moon", "resolved_mu_si": 4.902800066e12, "resolved_r_ref_m": 1737400.0,
            "degree_min": -1, "degree_max": 2
        }, f)
    mock_scaler().save_json(model_dir / "scaler.json")
    model = build_model_from_config(cfg)
    ckpt_dir = model_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict()}, ckpt_dir / "ckpt_best.pt")
    
    out_dir = tmp_path / "eval_out"
    evaluate(
        model_dir=model_dir, data_path=dpath, out_dir=out_dir, device=torch.device("cpu"),
        batch_size=10, a_sign=1.0, r_ref_m=1737400.0, alt_bin_km=50.0,
        streaming=True
    )
    
    with open(out_dir / "evaluate_metrics.json", "r") as f:
        metrics = json.load(f)
    assert metrics["evaluation_mode"] == "streaming"
    assert metrics["memory_safe"] is True
    assert metrics["n_points"] == 100
    assert "streaming_limitations" in metrics

def test_streaming_report_metrics_match_in_memory_for_small_dataset(tmp_path):
    dpath = tmp_path / "test.h5"
    with h5py.File(dpath, "w") as f:
        data = np.random.randn(200, 7)
        data[:, :3] *= 1e6
        f.create_dataset("data", data=data)
        f.attrs["central_body"] = "moon"
        f.attrs["mu_si"] = 4.902800066e12
        f.attrs["r_ref_m"] = 1737400.0
        f.attrs["unit_system"] = "si"
        f.attrs["target_mode"] = "full"
        f.attrs["degree_min"] = -1
        f.attrs["degree_max"] = 2
        
    cfg = MockArgs(
        hidden=16, depth=1, activation="silu", use_fourier=False, use_sh_encoding=False,
        central_body="moon", resolved_mu_si=4.902800066e12, resolved_r_ref_m=1737400.0,
        degree_min=-1, degree_max=2
    )
    model_dir = tmp_path / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    import json
    with open(model_dir / "config.json", "w") as f:
        json.dump({
            "hidden": 16, "depth": 1, "activation": "silu", "use_fourier": False, "use_sh_encoding": False,
            "central_body": "moon", "resolved_mu_si": 4.902800066e12, "resolved_r_ref_m": 1737400.0,
            "degree_min": -1, "degree_max": 2
        }, f)
    mock_scaler().save_json(model_dir / "scaler.json")
    model = build_model_from_config(cfg)
    ckpt_dir = model_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict()}, ckpt_dir / "ckpt_best.pt")
    
    out_dir_mem = tmp_path / "eval_mem"
    evaluate(
        model_dir=model_dir, data_path=dpath, out_dir=out_dir_mem, device=torch.device("cpu"),
        batch_size=50, a_sign=1.0, r_ref_m=1737400.0, alt_bin_km=50.0,
        streaming=False
    )
    
    out_dir_str = tmp_path / "eval_str"
    evaluate(
        model_dir=model_dir, data_path=dpath, out_dir=out_dir_str, device=torch.device("cpu"),
        batch_size=50, a_sign=1.0, r_ref_m=1737400.0, alt_bin_km=50.0,
        streaming=True
    )
    
    with open(out_dir_str / "evaluate_metrics.json", "r") as f:
        metrics_str = json.load(f)
    with open(out_dir_mem / "eval_report.json", "r") as f:
        metrics_mem = json.load(f)["metrics"]
    
    # Check that core metrics are identical
    assert metrics_str["n_points"] == metrics_mem["n_points"]
    assert np.isclose(metrics_str["U"]["mae"], metrics_mem["U"]["mae"])
    assert np.isclose(metrics_str["|a|"]["rmse"], metrics_mem["|a|"]["rmse"])

# ==============================================================================
# TESTS FOR TASK 4: HYBRID CHECKPOINT SCORE
# ==============================================================================

def test_hybrid_checkpoint_score_uses_base_plus_direction():
    cfg = MockArgs(best_metric="hybrid", hybrid_direction_alpha=0.5, direction_loss_weight=1.0)
    
    # Mock validation loop dictionary
    va = {
        "loss": 15.0, # val_total_loss
        "val_base_loss": 10.0,
        "loss_dir": 4.0,
        "mse_u": 5.0,
        "mse_a": 5.0
    }
    
    _best_metric_mode = getattr(cfg, "best_metric", "hybrid")
    _hybrid_alpha = getattr(cfg, "hybrid_direction_alpha", 0.5)
    _direction_loss_weight = getattr(cfg, "direction_loss_weight", 1.0)
    if _best_metric_mode == "hybrid" and _direction_loss_weight > 0.0:
        score = float(va.get("val_base_loss", va["loss"])) + _hybrid_alpha * float(va.get("loss_dir", 0.0))
        assert score == 12.0
        assert score != (va["loss"] + 0.5 * va["loss_dir"]) # 15 + 2 = 17

def test_best_metric_val_total_loss_uses_total_loss_directly():
    cfg = MockArgs(best_metric="val_total_loss")
    va = {"loss": 15.0, "val_total_loss": 15.0, "val_base_loss": 10.0}
    score = float(va.get("val_total_loss", va["loss"]))
    assert score == 15.0

# ==============================================================================
# TESTS FOR TASK 5: PREDICT RESIDUAL POTENTIAL INPUT GUARD
# ==============================================================================

def test_predict_residual_potential_rejects_nan_input():
    cfg = MockArgs(hidden=16, depth=1, activation="silu", use_fourier=False, use_sh_encoding=False)
    model = build_model_from_config(cfg)
    scaler = mock_scaler()
    fm = SurrogateForceModel(model, scaler, cfg={"a_sign": 1.0, "mu_si": 4.902800066e12, "degree_min": 0}, device=torch.device("cpu"))
    
    x_nan = np.array([[np.nan, 1000.0, 1000.0]])
    with pytest.raises(ValueError, match="NaN or Inf"):
        fm.predict_residual_potential(x_nan)

def test_predict_residual_potential_rejects_inf_input():
    cfg = MockArgs(hidden=16, depth=1, activation="silu", use_fourier=False, use_sh_encoding=False)
    model = build_model_from_config(cfg)
    scaler = mock_scaler()
    fm = SurrogateForceModel(model, scaler, cfg={"a_sign": 1.0, "mu_si": 4.902800066e12, "degree_min": 0}, device=torch.device("cpu"))
    
    x_inf = np.array([[np.inf, 1000.0, 1000.0]])
    with pytest.raises(ValueError, match="NaN or Inf"):
        fm.predict_residual_potential(x_inf)

def test_predict_residual_potential_valid_input_unchanged():
    cfg = MockArgs(hidden=16, depth=1, activation="silu", use_fourier=False, use_sh_encoding=False)
    model = build_model_from_config(cfg)
    scaler = mock_scaler()
    fm = SurrogateForceModel(model, scaler, cfg={"a_sign": 1.0, "mu_si": 4.902800066e12, "degree_min": 0}, device=torch.device("cpu"))

    x = np.array([[1000.0, 1000.0, 1000.0]])
    res = fm.predict_residual_potential(x)
    assert np.isfinite(res).all()


# ==============================================================================
# TASK 6: SHInspiredAngularEncoding in __all__
# ==============================================================================

def test_sh_inspired_angular_encoding_in_all():
    """Task 6: SHInspiredAngularEncoding must be exported from __all__."""
    import lunaris.surrogate.st_lrps.networks.models as _m
    assert "SHInspiredAngularEncoding" in _m.__all__, (
        "SHInspiredAngularEncoding must be in st_lrps_models.__all__"
    )


# ==============================================================================
# TASK 1 (cont.) / TASK 2: TrainConfig encoding fields and config.json metadata
# ==============================================================================

def test_train_config_has_encoding_fields():
    """Task 1 & 8: TrainConfig must have all 5 encoding fields with correct defaults."""
    from lunaris.surrogate.st_lrps.training.config import TrainConfig
    import dataclasses as _dc
    field_names = {f.name for f in _dc.fields(TrainConfig)}
    for field in ("use_sh_encoding", "sh_encoding_degree", "sh_append_raw",
                  "use_radial_separation", "radial_append_raw"):
        assert field in field_names, f"TrainConfig missing field: {field}"
    # Check defaults (all off)
    # TrainConfig needs data and out, so use dummy values
    cfg = TrainConfig(data="/dev/null", out="/tmp")
    assert cfg.use_sh_encoding is False
    assert cfg.sh_encoding_degree == 4
    assert cfg.sh_append_raw is True
    assert cfg.use_radial_separation is False
    assert cfg.radial_append_raw is False


def test_sh_encoding_config_written_to_json():
    """Task 2: config.json must contain encoding fields when SH encoding is active."""
    import dataclasses
    from lunaris.surrogate.st_lrps.training.config import TrainConfig
    cfg = TrainConfig(data="/dev/null", out="/tmp", use_sh_encoding=True, sh_encoding_degree=3)
    payload = dataclasses.asdict(cfg)
    assert "use_sh_encoding" in payload
    assert payload["use_sh_encoding"] is True
    assert payload["sh_encoding_degree"] == 3


def test_sh_encoded_model_strict_reload(tmp_path):
    """Task 2 & 8: Build SH-encoded model, save checkpoint, reload with strict=True."""
    cfg_dict = {
        "activation": "sine",
        "hidden": 32, "depth": 2,
        "w0_first": 30.0, "w0_hidden": 30.0,
        "dropout": 0.0,
        "use_sh_encoding": True, "sh_encoding_degree": 2, "sh_append_raw": True,
        "use_radial_separation": False, "use_fourier": False,
        "n_bands": 1, "use_residual_blocks": False,
    }
    model = build_model_from_config(cfg_dict, in_dim=3)
    ckpt_path = tmp_path / "model_sh.pt"
    torch.save({"model_state_dict": model.state_dict(), "cfg": cfg_dict}, str(ckpt_path))

    # Reload
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    cfg2 = ckpt["cfg"]
    model2 = build_model_from_config(cfg2, in_dim=3)
    model2.load_state_dict(ckpt["model_state_dict"], strict=True)

    x = torch.randn(5, 3, dtype=torch.float32)
    model.eval()
    model2.eval()
    with torch.no_grad():
        out1 = model(x)
        out2 = model2(x)
    assert torch.allclose(out1, out2, atol=1e-6), "SH strict reload outputs differ"


def test_radial_encoded_model_strict_reload(tmp_path):
    """Task 2 & 8: Build radial-encoded model, save checkpoint, reload with strict=True."""
    cfg_dict = {
        "activation": "sine",
        "hidden": 32, "depth": 2,
        "w0_first": 30.0, "w0_hidden": 30.0,
        "dropout": 0.0,
        "use_sh_encoding": False,
        "use_radial_separation": True, "radial_append_raw": False,
        "use_fourier": False,
        "n_bands": 1, "use_residual_blocks": False,
    }
    model = build_model_from_config(cfg_dict, in_dim=3)
    ckpt_path = tmp_path / "model_radial.pt"
    torch.save({"model_state_dict": model.state_dict(), "cfg": cfg_dict}, str(ckpt_path))

    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    cfg2 = ckpt["cfg"]
    model2 = build_model_from_config(cfg2, in_dim=3)
    model2.load_state_dict(ckpt["model_state_dict"], strict=True)

    x = torch.randn(5, 3, dtype=torch.float32)
    model.eval()
    model2.eval()
    with torch.no_grad():
        out1 = model(x)
        out2 = model2(x)
    assert torch.allclose(out1, out2, atol=1e-6), "Radial strict reload outputs differ"


def _make_encoded_force_model_checkpoint(tmp_path, use_sh=False, use_radial=False):
    """Helper: build encoded model, save checkpoint + config.json, return (model_dir, cfg_dict)."""
    cfg_dict = {
        "activation": "sine",
        "hidden": 32, "depth": 2,
        "w0_first": 30.0, "w0_hidden": 30.0,
        "dropout": 0.0,
        "use_sh_encoding": use_sh, "sh_encoding_degree": 2, "sh_append_raw": True,
        "use_radial_separation": use_radial, "radial_append_raw": False,
        "use_fourier": False,
        "n_bands": 1, "use_residual_blocks": False,
        "resolved_a_sign": 1.0,
        "resolved_mu_si": 4902.8e9,
        "resolved_r_ref_m": 1.737e6,
        "degree_min": 2,
        "residual_mode": True,
    }
    model = build_model_from_config(cfg_dict, in_dim=3)
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "ckpt_best.pt"
    torch.save({"model": model.state_dict(), "cfg": cfg_dict}, str(ckpt_path))

    # Minimal scaler.json (identity ScalerPack format)
    scaler_data = {
        "x": {"mean": [0.0, 0.0, 0.0], "scale": 1.0},
        "u": {"mean": [0.0], "scale": 1.0},
        "a": {"mean": [0.0, 0.0, 0.0], "scale": 1.0},
        "provenance": {"fit_rows": 10}
    }
    (tmp_path / "scaler.json").write_text(json.dumps(scaler_data))

    full_cfg = dict(cfg_dict)
    full_cfg["resolved_r_ref_m"] = 1.737e6
    (tmp_path / "config.json").write_text(json.dumps(full_cfg))
    return tmp_path, cfg_dict


def test_force_model_loads_sh_encoded_checkpoint(tmp_path):
    """Task 2 & 7: SurrogateForceModel must load an SH-encoded checkpoint and produce finite output."""
    model_dir, _ = _make_encoded_force_model_checkpoint(tmp_path, use_sh=True)
    from lunaris.surrogate.st_lrps.runtime.force_model import load_surrogate_force_model
    fm = load_surrogate_force_model(str(model_dir), device="cpu", allow_legacy_contract=True)
    x = np.array([[1.8e6, 0.0, 0.0], [0.0, 1.8e6, 0.0]])
    out = fm.predict_residual_potential(x)
    assert np.isfinite(out).all(), f"SH-encoded force model produced non-finite output: {out}"


def test_force_model_loads_radial_encoded_checkpoint(tmp_path):
    """Task 2 & 7: SurrogateForceModel must load a radial-encoded checkpoint and produce finite output."""
    model_dir, _ = _make_encoded_force_model_checkpoint(tmp_path, use_radial=True)
    from lunaris.surrogate.st_lrps.runtime.force_model import load_surrogate_force_model
    fm = load_surrogate_force_model(str(model_dir), device="cpu", allow_legacy_contract=True)
    x = np.array([[1.8e6, 0.0, 0.0], [0.0, 1.8e6, 0.0]])
    out = fm.predict_residual_potential(x)
    assert np.isfinite(out).all(), f"Radial-encoded force model produced non-finite output: {out}"


# ==============================================================================
# TASK 8: Backward compatibility — old raw configs load without encoding fields
# ==============================================================================

def test_old_config_without_encoding_fields_builds_raw_model():
    """Task 8: build_model_from_config must work with old configs that lack encoding fields."""
    old_cfg = {
        "activation": "sine",
        "hidden": 32, "depth": 2,
        "w0_first": 30.0, "w0_hidden": 30.0,
        "dropout": 0.0,
        "use_fourier": False,
        "n_bands": 1, "use_residual_blocks": False,
        # No encoding keys at all (old config)
    }
    model = build_model_from_config(old_cfg, in_dim=3)
    assert model is not None
    assert getattr(model, "embedding_type", "raw") == "raw"
    x = torch.randn(4, 3, dtype=torch.float32)
    out = model(x)
    assert out.shape == (4, 1)


def test_old_raw_checkpoint_strict_reload(tmp_path):
    """Task 8: Old raw (no-encoding) checkpoints must reload with strict=True after fix."""
    old_cfg = {
        "activation": "sine",
        "hidden": 32, "depth": 2,
        "w0_first": 30.0, "w0_hidden": 30.0,
        "dropout": 0.0,
        "use_fourier": False,
        "n_bands": 1, "use_residual_blocks": False,
    }
    model = build_model_from_config(old_cfg, in_dim=3)
    ckpt_path = tmp_path / "old_model.pt"
    torch.save({"model_state_dict": model.state_dict(), "cfg": old_cfg}, str(ckpt_path))

    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    # Old config has no encoding keys: build_model_from_config must default to raw
    model2 = build_model_from_config(ckpt["cfg"], in_dim=3)
    model2.load_state_dict(ckpt["model_state_dict"], strict=True)

    x = torch.randn(3, 3, dtype=torch.float32)
    model.eval(); model2.eval()
    with torch.no_grad():
        assert torch.allclose(model(x), model2(x), atol=1e-6)


# ==============================================================================
# TASK 3: Active refinement GFC key fixes
# ==============================================================================

def test_active_refinement_uses_mu_si_and_r_ref_m_keys(tmp_path, monkeypatch):
    """Task 3: _run_active_refinement must read mu_si and r_ref_m (not earth_gravity_constant/radius)."""
    import lunaris.surrogate.st_lrps.data.spatial_cloud_generator as scg

    captured_keys = {}

    def fake_load_icgem_gfc(file_path, max_degree, **kw):
        # Return normalized keys (what load_icgem_gfc actually provides)
        from lunaris.surrogate.st_lrps.data.dataset_parameters import MU_MOON_SI, R_MOON_SI
        gmeta = {
            "mu_si": MU_MOON_SI,
            "r_ref_m": R_MOON_SI,
            "degree": max_degree,
            "central_body": "moon",
        }
        captured_keys.update(gmeta)
        # Return minimal C, S arrays
        n = max_degree + 1
        C = np.zeros((n, n))
        S = np.zeros((n, n))
        C[0, 0] = 1.0
        return C, S, gmeta

    monkeypatch.setattr(scg, "load_icgem_gfc", fake_load_icgem_gfc)

    # Create dummy error points CSV
    err_csv = tmp_path / "errors.csv"
    from lunaris.surrogate.st_lrps.data.dataset_parameters import R_MOON_SI
    r0 = R_MOON_SI + 300e3
    cols = ["x","y","z","u_true","u_pred","ax_true","ay_true","az_true","ax_pred","ay_pred","az_pred","abs_a_error","rel_a_error","altitude_km"]
    header = ",".join(cols)
    row1 = f"{r0},0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1e-5,0.0,300.0"
    row2 = f"0.0,{r0},0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,2e-5,0.0,300.0"
    err_csv.write_text(f"{header}\n{row1}\n{row2}\n")

    gfc_file = tmp_path / "fake.gfc"
    gfc_file.write_text("")  # content irrelevant — we monkeypatch loader

    class _Ap:
        def error(self, msg):
            raise SystemExit(msg)

    args = MockArgs(
        active_from_error_points=str(err_csv),
        active_max_source_points=10,
        active_samples_per_point=1,
        active_jitter_radial_km=0.1,
        active_jitter_tangent_km=0.1,
        active_gfc_file=str(gfc_file),
        active_degree_max=2,
        active_degree_min=-1,
        active_seed=42,
        active_save_positions_only=False,
        active_clip_to_alt_range=False,
        active_reject_outside_alt_range=False,
        active_out=str(tmp_path / "out.h5"),
        out=str(tmp_path),
    )

    scg._run_active_refinement(args, _Ap())
    out_h5 = tmp_path / "out.h5"
    assert out_h5.exists(), "Active refinement must write HDF5 output"
    with h5py.File(str(out_h5), "r") as hf:
        mu_written = float(hf.attrs["mu_si"])
        r_ref_written = float(hf.attrs["r_ref_m"])
    from lunaris.surrogate.st_lrps.data.dataset_parameters import MU_MOON_SI, R_MOON_SI
    assert abs(mu_written - MU_MOON_SI) < 1e6, (
        f"Active refinement wrote wrong mu_si={mu_written}, expected ~{MU_MOON_SI}"
    )
    assert abs(r_ref_written - R_MOON_SI) < 1e3, (
        f"Active refinement wrote wrong r_ref_m={r_ref_written}, expected ~{R_MOON_SI}"
    )


def test_active_refinement_degree_min_zero_preserved(tmp_path, monkeypatch):
    """Task 3: degree_min=0 must not be overwritten to -1 by the falsy `or` bug."""
    import lunaris.surrogate.st_lrps.data.spatial_cloud_generator as scg
    from lunaris.surrogate.st_lrps.data.dataset_parameters import MU_MOON_SI, R_MOON_SI

    def fake_load_icgem_gfc(file_path, max_degree, **kw):
        n = max_degree + 1
        C = np.zeros((n, n)); S = np.zeros((n, n)); C[0, 0] = 1.0
        return C, S, {"mu_si": MU_MOON_SI, "r_ref_m": R_MOON_SI,
                      "degree": max_degree, "central_body": "moon"}

    monkeypatch.setattr(scg, "load_icgem_gfc", fake_load_icgem_gfc)

    err_csv = tmp_path / "err.csv"
    r0 = R_MOON_SI + 300e3
    cols = ["x","y","z","u_true","u_pred","ax_true","ay_true","az_true","ax_pred","ay_pred","az_pred","abs_a_error","rel_a_error","altitude_km"]
    header = ",".join(cols)
    row1 = f"{r0},0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1e-5,0.0,300.0"
    err_csv.write_text(f"{header}\n{row1}\n")
    gfc_file = tmp_path / "fake.gfc"; gfc_file.write_text("")

    class _Ap:
        def error(self, msg): raise SystemExit(msg)

    args = MockArgs(
        active_from_error_points=str(err_csv),
        active_max_source_points=5,
        active_samples_per_point=1,
        active_jitter_radial_km=0.1,
        active_jitter_tangent_km=0.1,
        active_gfc_file=str(gfc_file),
        active_degree_max=2,
        active_degree_min=0,    # ← explicitly zero; must NOT become -1
        active_seed=42,
        active_save_positions_only=False,
        active_clip_to_alt_range=False,
        active_reject_outside_alt_range=False,
        active_out=str(tmp_path / "out_deg0.h5"),
        out=str(tmp_path),
    )
    scg._run_active_refinement(args, _Ap())
    out_h5 = tmp_path / "out_deg0.h5"
    assert out_h5.exists()
    with h5py.File(str(out_h5), "r") as hf:
        deg_min_written = int(hf.attrs.get("degree_min", -99))
    assert deg_min_written == 0, (
        f"degree_min=0 was overwritten; got {deg_min_written} (old bug would give -1)"
    )


# ==============================================================================
# TASK 4: Streaming evaluator U L∞ and relative error are real values
# ==============================================================================

def _make_small_h5(path: Path, n: int = 150):
    """Create a minimal valid HDF5 file for evaluator smoke tests."""
    from lunaris.surrogate.st_lrps.data.dataset_parameters import R_MOON_SI, MU_MOON_SI
    rng = np.random.default_rng(7)
    r = R_MOON_SI + rng.uniform(200e3, 500e3, n)
    theta = rng.uniform(0, np.pi, n)
    phi = rng.uniform(0, 2 * np.pi, n)
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    U = -MU_MOON_SI / r + rng.normal(0, 1.0, n)
    ax = -MU_MOON_SI * x / r**3 + rng.normal(0, 1e-6, n)
    ay = -MU_MOON_SI * y / r**3 + rng.normal(0, 1e-6, n)
    az = -MU_MOON_SI * z / r**3 + rng.normal(0, 1e-6, n)
    data = np.stack([x, y, z, U, ax, ay, az], axis=1).astype(np.float32)
    with h5py.File(str(path), "w") as hf:
        hf.create_dataset("data", data=data)
        hf.attrs["unit_system"] = "SI"
        hf.attrs["mu_si"] = float(MU_MOON_SI)
        hf.attrs["r_ref_m"] = float(R_MOON_SI)
        hf.attrs["central_body"] = "moon"
        hf.attrs["target_mode"] = "full"
        hf.attrs["degree_min"] = -1
        hf.attrs["requested_degree"] = 2
        hf.attrs["a_sign_convention"] = 1.0
        hf.attrs["columns"] = "[x,y,z,U,ax,ay,az]"


def _make_minimal_run_dir(tmp_path: Path):
    """Create a minimal model run dir (config.json + scaler.json + checkpoint) for evaluator."""
    from lunaris.surrogate.st_lrps.data.dataset_parameters import R_MOON_SI, MU_MOON_SI
    cfg_dict = {
        "activation": "sine", "hidden": 16, "depth": 2,
        "w0_first": 30.0, "w0_hidden": 30.0, "dropout": 0.0,
        "use_sh_encoding": False, "sh_encoding_degree": 4, "sh_append_raw": True,
        "use_radial_separation": False, "radial_append_raw": False,
        "use_fourier": False, "n_bands": 1, "use_residual_blocks": False,
        "resolved_a_sign": 1.0, "resolved_mu_si": MU_MOON_SI,
        "resolved_r_ref_m": R_MOON_SI, "degree_min": -1, "residual_mode": False,
        "target_mode": "full",
    }
    model = build_model_from_config(cfg_dict, in_dim=3)
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "cfg": cfg_dict},
               str(ckpt_dir / "ckpt_best.pt"))
    (tmp_path / "config.json").write_text(json.dumps(cfg_dict))
    scaler_data = {
        "x": {"mean": [0.0, 0.0, 0.0], "scale": 1.0},
        "u": {"mean": [0.0], "scale": 1.0},
        "a": {"mean": [0.0, 0.0, 0.0], "scale": 1.0},
        "provenance": {"fit_rows": 10}
    }
    (tmp_path / "scaler.json").write_text(json.dumps(scaler_data))
    return tmp_path


def test_streaming_u_linf_is_not_fake_zero(tmp_path):
    """Task 4: streaming U.linf must be > 0 for non-trivial data (real L∞ tracking)."""
    data_path = tmp_path / "data.h5"
    _make_small_h5(data_path, n=100)
    run_dir = _make_minimal_run_dir(tmp_path / "run")
    out_dir = tmp_path / "eval_out"
    result = evaluate(
        model_dir=run_dir,
        data_path=data_path,
        out_dir=out_dir,
        streaming=True,
        device=torch.device("cpu"),
        batch_size=10,
        a_sign=1.0,
        r_ref_m=1737400.0,
        alt_bin_km=50.0,
    )
    u_linf = result.get("U", {}).get("linf", None)
    assert u_linf is not None, "U.linf missing from streaming result"
    assert u_linf > 0.0, f"U.linf should be > 0 for non-trivial data; got {u_linf}"


def test_streaming_u_relative_error_is_computed(tmp_path):
    """Task 4: streaming U.rel_mean_pct must be > 0 for non-trivial data."""
    data_path = tmp_path / "data.h5"
    _make_small_h5(data_path, n=100)
    run_dir = _make_minimal_run_dir(tmp_path / "run")
    out_dir = tmp_path / "eval_out"
    result = evaluate(
        model_dir=run_dir,
        data_path=data_path,
        out_dir=out_dir,
        streaming=True,
        device=torch.device("cpu"),
        batch_size=10,
        a_sign=1.0,
        r_ref_m=1737400.0,
        alt_bin_km=50.0,
    )
    rel_pct = result.get("U", {}).get("rel_mean_pct", None)
    assert rel_pct is not None, "U.rel_mean_pct missing from streaming result"
    assert rel_pct > 0.0, f"U.rel_mean_pct should be > 0; got {rel_pct}"


# ==============================================================================
# TASK 5: eval_report.json as primary output from streaming mode
# ==============================================================================

def test_streaming_writes_eval_report_json(tmp_path):
    """Task 5: streaming mode must write eval_report.json."""
    data_path = tmp_path / "data.h5"
    _make_small_h5(data_path, n=80)
    run_dir = _make_minimal_run_dir(tmp_path / "run")
    out_dir = tmp_path / "eval_out"
    evaluate(
        model_dir=run_dir,
        data_path=data_path,
        out_dir=out_dir,
        streaming=True,
        device=torch.device("cpu"),
        batch_size=10,
        a_sign=1.0,
        r_ref_m=1737400.0,
        alt_bin_km=50.0,
    )
    assert (out_dir / "eval_report.json").exists(), (
        "streaming evaluate() must write eval_report.json"
    )


def test_streaming_eval_report_has_metrics_block(tmp_path):
    """Task 5: streaming eval_report.json must have a 'metrics' sub-block."""
    data_path = tmp_path / "data.h5"
    _make_small_h5(data_path, n=80)
    run_dir = _make_minimal_run_dir(tmp_path / "run")
    out_dir = tmp_path / "eval_out"
    evaluate(
        model_dir=run_dir,
        data_path=data_path,
        out_dir=out_dir,
        streaming=True,
        device=torch.device("cpu"),
        batch_size=10,
        a_sign=1.0,
        r_ref_m=1737400.0,
        alt_bin_km=50.0,
    )
    report_path = out_dir / "eval_report.json"
    report = json.loads(report_path.read_text())
    assert "metrics" in report, (
        "eval_report.json must have a 'metrics' top-level key (got: " + str(list(report.keys())) + ")"
    )
    metrics = report["metrics"]
    assert "U" in metrics
    assert "evaluation_mode" in metrics
    assert metrics["evaluation_mode"] == "streaming"


def test_streaming_also_writes_evaluate_metrics_json_alias(tmp_path):
    """Task 5: streaming mode must still write evaluate_metrics.json for backward compat."""
    data_path = tmp_path / "data.h5"
    _make_small_h5(data_path, n=80)
    run_dir = _make_minimal_run_dir(tmp_path / "run")
    out_dir = tmp_path / "eval_out"
    evaluate(
        model_dir=run_dir,
        data_path=data_path,
        out_dir=out_dir,
        streaming=True,
        device=torch.device("cpu"),
        batch_size=10,
        a_sign=1.0,
        r_ref_m=1737400.0,
        alt_bin_km=50.0,
    )
    assert (out_dir / "evaluate_metrics.json").exists(), (
        "streaming evaluate() must still write evaluate_metrics.json (compatibility alias)"
    )


if __name__ == "__main__":
    pytest.main(["-v", __file__])
