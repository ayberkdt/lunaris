from __future__ import annotations
import pytest
try:
    import torch
    import torch.nn
    _ = torch.cuda
    _ = getattr(torch, "compile", None)
except (ImportError, AttributeError, ModuleNotFoundError):
    pytest.skip("PyTorch not installed or missing required attributes like cuda/compile", allow_module_level=True)



import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from lunaris.surrogate.runtime import SurrogateGravityModel
from lunaris.surrogate.st_lrps.data.dataset_parameters import MU_MOON_SI, R_MOON_SI
from lunaris.surrogate.st_lrps.networks.models import (
    build_model_from_config,
    compute_architecture_signature,
)
from lunaris.surrogate.st_lrps.runtime.force_model import (
    load_surrogate_force_model,
)
from lunaris.surrogate.st_lrps.shared.contracts import TargetContract
from lunaris.surrogate.st_lrps.shared.scaling import IsometricScaleParams, ScalerPack


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
        "dataset": {
            "target_mode": "residual",
            "degree_min": 0,
            "degree_max": 50,
            "altitude_min_km": 50.0,
            "altitude_max_km": 300.0,
        },
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
    from lunaris.surrogate.st_lrps.shared.contracts import ArtifactContract
    ac = ArtifactContract.from_resolved_config(cfg, scaler_payload=asdict(scaler))
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model": model.state_dict(),
            "config": cfg,
            "scaler": asdict(scaler),
            "kind": "best",
            "epoch": 0,
            "artifact_contract": ac.to_dict(),
        },
        run_dir / "checkpoints" / "ckpt_best.pt",
    )
    return run_dir


def test_force_model_and_gravity_provider_return_same_total_accel(tmp_path: Path) -> None:
    run_dir = _make_run(tmp_path)
    force = load_surrogate_force_model(run_dir, device="cpu")
    provider = SurrogateGravityModel.from_model_dir(run_dir, device_preference="cpu")
    x = np.array(
        [
            [R_MOON_SI + 100_000.0, 0.0, 0.0],
            [0.0, R_MOON_SI + 150_000.0, 0.0],
        ],
        dtype=np.float64,
    )
    assert np.allclose(force.predict_total_accel(x), provider.acceleration_fixed_batch(x), rtol=1e-6, atol=1e-12)
    assert force.degree_min == provider.degree_min == 0
    assert force.degree_max == provider.degree_max == 50
    assert force.target_contract.target_mode == "residual"


def test_retired_physics_import_is_removed() -> None:
    import importlib

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("lunaris.physics.surrogate_gravity")
