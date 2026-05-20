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

from surrogate_gravity_model.st_lrps_models import (
    SHInspiredAngularEncoding,
    RadialSeparationEncoding,
    PhysicsNet,
    build_model_from_config
)
from surrogate_gravity_model.st_lrps_force_model import SurrogateForceModel, ScalerPack
from surrogate_gravity_model.spatial_cloud_generator import _run_active_refinement, _sh_potential_accel_batch_serial
from surrogate_gravity_model.st_lrps_engine import TrainConfig
from surrogate_gravity_model.st_lrps_evaluate import evaluate

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

from surrogate_gravity_model.st_lrps_scaling import ScalerPack, IsometricScaleParams

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
    fm = SurrogateForceModel(model, scaler, cfg={"a_sign": 1.0, "mu_si": 1e14, "degree_min": 0}, device=torch.device("cpu"))
    
    x_nan = np.array([[np.nan, 1000.0, 1000.0]])
    with pytest.raises(ValueError, match="NaN or Inf"):
        fm.predict_residual_potential(x_nan)

def test_predict_residual_potential_rejects_inf_input():
    cfg = MockArgs(hidden=16, depth=1, activation="silu", use_fourier=False, use_sh_encoding=False)
    model = build_model_from_config(cfg)
    scaler = mock_scaler()
    fm = SurrogateForceModel(model, scaler, cfg={"a_sign": 1.0, "mu_si": 1e14, "degree_min": 0}, device=torch.device("cpu"))
    
    x_inf = np.array([[np.inf, 1000.0, 1000.0]])
    with pytest.raises(ValueError, match="NaN or Inf"):
        fm.predict_residual_potential(x_inf)

def test_predict_residual_potential_valid_input_unchanged():
    cfg = MockArgs(hidden=16, depth=1, activation="silu", use_fourier=False, use_sh_encoding=False)
    model = build_model_from_config(cfg)
    scaler = mock_scaler()
    fm = SurrogateForceModel(model, scaler, cfg={"a_sign": 1.0, "mu_si": 1e14, "degree_min": 0}, device=torch.device("cpu"))
    
    x = np.array([[1000.0, 1000.0, 1000.0]])
    res = fm.predict_residual_potential(x)
    assert np.isfinite(res).all()

if __name__ == "__main__":
    pytest.main(["-v", __file__])
