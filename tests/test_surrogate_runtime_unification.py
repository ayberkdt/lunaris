from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from st_lrps.data.dataset_parameters import MU_MOON_SI, R_MOON_SI
from st_lrps.networks.models import build_model_from_config, compute_architecture_signature
from st_lrps.runtime.force_model import DirectForceRuntime, load_surrogate_force_model
from st_lrps.shared.contracts import TargetContract
from st_lrps.shared.scaling import IsometricScaleParams, ScalerPack
from models.surrogate_gravity import SurrogateGravityModel


def _make_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    (run_dir / "checkpoints").mkdir(parents=True)
    contract = TargetContract(
        central_body="moon",
        target_mode="residual",
        base_degree=0,
        target_degree=50,
        baseline_kind="spherical_harmonics",
        unit_system="si",
        frame="moon_fixed_cartesian",
        derivative_convention_version="dP_dphi_corrected_v1",
        a_sign=1.0,
        mu_si=MU_MOON_SI,
        r_ref_m=R_MOON_SI,
    )
    cfg = {
        "activation": "tanh",
        "hidden": 8,
        "depth": 1,
        "dropout": 0.0,
        "n_bands": 1,
        "degree_min": 0,
        "degree_max": 50,
        "target_mode": "residual",
        "central_body": "moon",
        "resolved_mu_si": MU_MOON_SI,
        "resolved_r_ref_m": R_MOON_SI,
        "resolved_a_sign": 1.0,
        "runtime_model_kind": "potential_autograd",
        "target_contract": contract.to_dict(),
        "model_preset": "custom",
    }
    model = build_model_from_config(cfg, device=torch.device("cpu"), dtype=torch.float32)
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
    cfg["input_feature_dim"] = int(model.input_feature_dim)
    cfg["embedding_type"] = str(model.embedding_type)
    cfg["model_builder_version"] = str(model.model_builder_version)
    cfg["architecture_signature"] = compute_architecture_signature(cfg)
    scaler = ScalerPack(
        x=IsometricScaleParams(mean=[0.0, 0.0, 0.0], scale=2.0e6),
        u=IsometricScaleParams(mean=[0.0], scale=1.0),
        a=IsometricScaleParams(mean=[0.0, 0.0, 0.0], scale=1.0),
        provenance={"target_contract": contract.to_dict()},
    )
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    scaler.save_json(run_dir / "scaler.json")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model": model.state_dict(),
            "config": cfg,
            "scaler": asdict(scaler),
            "kind": "best",
            "epoch": 0,
        },
        run_dir / "checkpoints" / "ckpt_best.pt",
    )
    return run_dir


def test_force_model_and_legacy_adapter_return_same_total_accel(tmp_path: Path) -> None:
    run_dir = _make_run(tmp_path)
    force = load_surrogate_force_model(run_dir, device="cpu")
    legacy = SurrogateGravityModel.from_model_dir(run_dir, device_preference="cpu")
    x = np.array(
        [
            [R_MOON_SI + 100_000.0, 0.0, 0.0],
            [0.0, R_MOON_SI + 150_000.0, 0.0],
        ],
        dtype=np.float64,
    )
    assert np.allclose(force.predict_total_accel(x), legacy.acceleration_fixed_batch(x), rtol=1e-6, atol=1e-12)
    assert force.degree_min == legacy.degree_min == 0
    assert force.degree_max == legacy.degree_max == 50
    assert force.target_contract.target_mode == "residual"


def test_force_direct_placeholder_raises() -> None:
    with pytest.raises(NotImplementedError, match="force_direct"):
        DirectForceRuntime()
