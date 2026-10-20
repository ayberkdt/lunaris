"""Regression tests for the surrogate gravity-provider runtime fixes.

1. ``acceleration_fixed`` / ``acceleration_fixed_batch`` no longer compute the
   discarded potential, and return the same acceleration bit for bit.
2. ``strict_domain`` / ``paper_safe`` reach the canonical runtime, so a strict
   provider hard-fails on the CPU path below the training envelope (it only
   warned before), while a non-strict provider keeps warning.
3. ``import lunaris.surrogate.runtime`` does not import torch.
4. Empty ``(0, 3)`` batches pass the domain guards; a ``mu_override`` that
   disagrees with the artifact's own GM is refused instead of ignored.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from lunaris.common.constants import MU_MOON, R_MOON
from lunaris.surrogate.runtime import SurrogateGravityModel
from lunaris.surrogate.runtime.networks import _build_model_from_config
from lunaris.surrogate.st_lrps.runtime import canonical_runtime
from lunaris.surrogate.st_lrps.shared.contracts import ArtifactContract, TargetContract
from lunaris.surrogate.st_lrps.shared.scaling import IsometricScaleParams, ScalerPack

_ALT_MIN_KM = 50.0
_ALT_MAX_KM = 300.0


def _make_contract_run(tmp_path: Path, *, seed: int = 3) -> Path:
    """Contract-era run (loads through the canonical runtime) with non-zero weights."""
    from lunaris.surrogate.st_lrps.data.dataset_parameters import MU_MOON_SI, R_MOON_SI
    from lunaris.surrogate.st_lrps.networks.models import (
        build_model_from_config,
        compute_architecture_signature,
    )

    run_dir = tmp_path / "contract_run"
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
        "hidden": 16,
        "depth": 2,
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
            "altitude_min_km": _ALT_MIN_KM,
            "altitude_max_km": _ALT_MAX_KM,
        },
    }
    torch.manual_seed(seed)
    model = build_model_from_config(cfg, device=torch.device("cpu"), dtype=torch.float32)
    cfg["input_feature_dim"] = int(model.input_feature_dim)
    cfg["embedding_type"] = str(model.embedding_type)
    cfg["model_builder_version"] = str(model.model_builder_version)
    cfg["architecture_signature"] = compute_architecture_signature(cfg)
    scaler = ScalerPack(
        x=IsometricScaleParams(mean=[0.0, 0.0, 0.0], scale=2.0e6),
        u=IsometricScaleParams(mean=[0.0], scale=1.0e3),
        a=IsometricScaleParams(mean=[0.0, 0.0, 0.0], scale=1.0),
        provenance={"target_contract": contract.to_dict()},
    )
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    scaler.save_json(run_dir / "scaler.json")
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


def _make_legacy_run(tmp_path: Path, *, resolved_mu: float = float(MU_MOON)) -> Path:
    """Pre-contract run (legacy local path) with non-zero weights."""
    run_dir = tmp_path / "legacy_run"
    (run_dir / "checkpoints").mkdir(parents=True)
    config = {
        "hidden": 16, "depth": 2, "activation": "tanh", "dropout": 0.0,
        "resolved_mu_si": float(resolved_mu), "resolved_a_sign": 1.0,
        "scaler_kind": "isometric", "degree_min": 0, "degree_max": 50,
    }
    scaler = {
        "x": {"mean": [0.0, 0.0, 0.0], "scale": 2_000_000.0},
        "u": {"mean": [0.0], "scale": 1.0e3},
        "a": {"mean": [0.0, 0.0, 0.0], "scale": 1.0},
    }
    torch.manual_seed(5)
    model = _build_model_from_config(config)
    (run_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (run_dir / "scaler.json").write_text(json.dumps(scaler), encoding="utf-8")
    torch.save(
        {"model": model.state_dict(), "config": config, "scaler": scaler},
        run_dir / "checkpoints" / "ckpt_best.pt",
    )
    return run_dir


def _points_in_envelope(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    dirs = rng.standard_normal((n, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    radii = float(R_MOON) + rng.uniform(60e3, 290e3, (n, 1))
    return dirs * radii


# ---------------------------------------------------------------------------
# 1) acceleration-only CPU path
# ---------------------------------------------------------------------------

def test_acceleration_only_path_is_bit_identical_canonical(tmp_path, monkeypatch):
    provider = SurrogateGravityModel.from_model_dir(
        _make_contract_run(tmp_path), device_preference="cpu"
    )
    assert provider._force_runtime is not None
    pts = _points_in_envelope(64)
    _u, a_full = provider.predict_potential_and_acceleration_fixed(pts)

    calls = {"potential": 0}
    original = type(provider._force_runtime).predict_residual_potential

    def _spy(self, x):
        calls["potential"] += 1
        return original(self, x)

    monkeypatch.setattr(type(provider._force_runtime), "predict_residual_potential", _spy)
    a_batch = provider.acceleration_fixed_batch(pts)
    a_single = provider.acceleration_fixed(pts[7])

    assert np.any(a_batch != 0.0)
    assert np.array_equal(a_batch, a_full)
    assert np.array_equal(a_single, a_full[7])
    assert a_batch.shape == (64, 3) and a_single.shape == (3,)
    assert calls["potential"] == 0  # the discarded potential is no longer computed


def test_acceleration_only_path_is_bit_identical_legacy(tmp_path, monkeypatch):
    monkeypatch.setenv("LUNARIS_ALLOW_LEGACY_ARTIFACT", "1")
    with pytest.warns(RuntimeWarning, match="legacy"):
        provider = SurrogateGravityModel.from_model_dir(
            _make_legacy_run(tmp_path), device_preference="cpu"
        )
    assert provider._force_runtime is None
    pts = _points_in_envelope(32, seed=1)
    _u, a_full = provider.predict_potential_and_acceleration_fixed(pts)
    a_batch = provider.acceleration_fixed_batch(pts)
    assert np.any(a_batch != 0.0)
    assert np.array_equal(a_batch, a_full)
    assert np.array_equal(provider.acceleration_fixed(pts[3]), a_full[3])


# ---------------------------------------------------------------------------
# 2) strict_domain / paper_safe forwarding
# ---------------------------------------------------------------------------

def _below_envelope() -> np.ndarray:
    return np.array([[float(R_MOON) + 10e3, 0.0, 0.0]])  # 10 km < 50 km floor


def test_strict_provider_raises_on_cpu_path_below_envelope(tmp_path):
    provider = SurrogateGravityModel.from_model_dir(
        _make_contract_run(tmp_path), device_preference="cpu", strict_domain=True
    )
    assert provider._force_runtime.strict_domain is True
    with pytest.raises(RuntimeError, match="strict_domain=True"):
        provider.acceleration_fixed_batch(_below_envelope())
    with pytest.raises(RuntimeError, match="strict_domain=True"):
        provider.acceleration_fixed(_below_envelope()[0])
    # The GPU tensor path agrees (it already did).
    with pytest.raises(RuntimeError, match="training envelope"):
        provider.predict_total_accel_torch(torch.as_tensor(_below_envelope()))
    # Inside the envelope nothing raises.
    provider.acceleration_fixed_batch(_points_in_envelope(4))


def test_nonstrict_provider_keeps_warning_on_cpu_path(tmp_path, caplog):
    provider = SurrogateGravityModel.from_model_dir(
        _make_contract_run(tmp_path), device_preference="cpu", strict_domain=False
    )
    assert provider._force_runtime.strict_domain is False
    with caplog.at_level(logging.WARNING, logger=canonical_runtime.logger.name):
        a = provider.acceleration_fixed_batch(_below_envelope())
    assert np.all(np.isfinite(a))
    assert any("outside training range" in r.getMessage() for r in caplog.records)
    with pytest.warns(RuntimeWarning, match="extrapolation"):
        provider.predict_total_accel_torch(torch.as_tensor(_below_envelope()))


def test_paper_safe_is_forwarded_to_canonical_loader(tmp_path, monkeypatch):
    seen: dict[str, object] = {}
    original = canonical_runtime.load_surrogate_force_model

    def _spy(*args, **kwargs):
        seen.update(kwargs)
        return original(*args, **kwargs)

    # Patch the name the bridge actually imports (the force_model shim). A
    # monkeypatch undo elsewhere can leave a concrete attribute on the shim, so
    # patching only canonical_runtime would be order dependent.
    from lunaris.surrogate.st_lrps.runtime import force_model

    monkeypatch.setattr(force_model, "load_surrogate_force_model", _spy, raising=False)
    with pytest.raises(RuntimeError, match="paper_safe"):
        # The synthetic artifact lacks paper-safe metadata, so the canonical
        # R26 gate refuses it; being contract-era, no legacy fallback exists.
        SurrogateGravityModel.from_model_dir(
            _make_contract_run(tmp_path), device_preference="cpu", paper_safe=True
        )
    assert seen["paper_safe"] is True
    assert seen["strict_domain"] is False


def test_paper_safe_refuses_the_legacy_fallback(tmp_path, monkeypatch):
    # The legacy local path has no paper-safe metadata gate, so the research
    # opt-in must not let a paper_safe load degrade onto it.
    monkeypatch.setenv("LUNARIS_ALLOW_LEGACY_ARTIFACT", "1")
    with pytest.raises(RuntimeError, match="paper_safe=True"):
        SurrogateGravityModel.from_model_dir(
            _make_legacy_run(tmp_path), device_preference="cpu", paper_safe=True
        )


# ---------------------------------------------------------------------------
# 3) lazy torch import
# ---------------------------------------------------------------------------

def test_runtime_package_import_does_not_import_torch():
    code = (
        "import sys\n"
        "import lunaris.surrogate.runtime as rt\n"
        "from lunaris.surrogate.runtime import DEFAULT_ST_LRPS_RUNS_DIR\n"
        "assert 'torch' not in sys.modules, 'torch imported eagerly'\n"
        "assert set(rt.__all__) <= set(dir(rt))\n"
        "rt.SurrogateGravityModel\n"
        "assert 'torch' in sys.modules\n"
    )
    env = dict(os.environ)
    src = str(Path(__file__).resolve().parents[1] / "src")
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=300, check=False
    )
    assert proc.returncode == 0, proc.stderr


def test_runtime_package_unknown_attribute_raises():
    import lunaris.surrogate.runtime as rt

    with pytest.raises(AttributeError):
        _ = rt.not_a_real_name


# ---------------------------------------------------------------------------
# 4) empty batches and mu_override
# ---------------------------------------------------------------------------

def test_empty_batch_passes_cpu_and_tensor_domain_guards(tmp_path):
    provider = SurrogateGravityModel.from_model_dir(
        _make_contract_run(tmp_path), device_preference="cpu", strict_domain=True
    )
    empty = np.empty((0, 3), dtype=np.float64)
    assert provider.acceleration_fixed_batch(empty).shape == (0, 3)
    assert provider.predict_total_accel_torch(torch.as_tensor(empty)).shape == (0, 3)
    status = provider._force_runtime.domain_status(empty)
    assert status["recommended_fallback"] is False


def test_mu_override_disagreeing_with_artifact_gm_is_refused(tmp_path, monkeypatch):
    monkeypatch.setenv("LUNARIS_ALLOW_LEGACY_ARTIFACT", "1")
    earth_gm = 3.986004418e14
    run = _make_legacy_run(tmp_path, resolved_mu=earth_gm)
    with pytest.raises(ValueError, match="mu_override"):
        SurrogateGravityModel.from_model_dir(
            run, mu_override=float(MU_MOON), device_preference="cpu"
        )


def test_mu_override_matching_artifact_gm_loads(tmp_path, monkeypatch):
    monkeypatch.setenv("LUNARIS_ALLOW_LEGACY_ARTIFACT", "1")
    run = _make_legacy_run(tmp_path, resolved_mu=float(MU_MOON) * 1.0001)
    with pytest.warns(RuntimeWarning, match="legacy"):
        provider = SurrogateGravityModel.from_model_dir(
            run, mu_override=float(MU_MOON), device_preference="cpu"
        )
    assert provider.GM_m3s2 == pytest.approx(float(MU_MOON) * 1.0001)  # artifact GM wins


@pytest.mark.parametrize("factor", [1.01, 0.99, 1.0006])
def test_mu_override_outside_lunar_gm_tolerance_is_refused(tmp_path, monkeypatch, factor):
    """Ported from Sidera 4a8784c1: the override-vs-artifact GM check uses the
    strict lunar tolerance (LUNAR_GM_REL_TOL = 5e-4), not a 20% envelope."""
    from lunaris.common.lunar_data import LUNAR_GM_REL_TOL

    assert LUNAR_GM_REL_TOL == 5.0e-4
    monkeypatch.setenv("LUNARIS_ALLOW_LEGACY_ARTIFACT", "1")
    run = _make_legacy_run(tmp_path, resolved_mu=float(MU_MOON) * factor)
    with pytest.raises(ValueError, match="mu_override"):
        SurrogateGravityModel.from_model_dir(
            run, mu_override=float(MU_MOON), device_preference="cpu"
        )


def test_mu_override_inside_lunar_gm_tolerance_loads(tmp_path, monkeypatch):
    monkeypatch.setenv("LUNARIS_ALLOW_LEGACY_ARTIFACT", "1")
    run = _make_legacy_run(tmp_path, resolved_mu=float(MU_MOON) * 1.0004)
    with pytest.warns(RuntimeWarning, match="legacy"):
        provider = SurrogateGravityModel.from_model_dir(
            run, mu_override=float(MU_MOON), device_preference="cpu"
        )
    assert provider.GM_m3s2 == pytest.approx(float(MU_MOON) * 1.0004)
