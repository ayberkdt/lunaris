from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from lunaris.analysis.monte_carlo.result_audit import TRUSTED, classify_st_lrps_run
from lunaris.surrogate.runtime import SurrogateGravityModel
from lunaris.surrogate.st_lrps.data.dataset_contract import (
    GRAVITY_LABEL_ENGINE_VERSION,
    REQUIRED_SH_PHASE_CONVENTION,
)
from lunaris.surrogate.st_lrps.data.dataset_parameters import R_MOON_SI
from lunaris.surrogate.st_lrps.evaluation.force_direct_eval import (
    curl_diagnostics,
    evaluate_force_direct,
)
from lunaris.surrogate.st_lrps.networks.models import (
    build_model_from_config,
    compute_architecture_signature,
)
from lunaris.surrogate.st_lrps.runtime.force_model import (
    DirectForceRuntime,
    SurrogateForceModel,
    load_surrogate_force_model,
)
from lunaris.surrogate.st_lrps.training.force_direct_cli import train_force_direct
from st_lrps_contract_test_utils import make_contract_run, tiny_training_cfg

pytestmark = pytest.mark.requires_torch


def _force_cfg(**overrides):
    cfg = tiny_training_cfg(
        runtime_model_kind="force_direct",
        prediction_kind="residual_force",
        output_dim=3,
        **overrides,
    )
    return cfg


def test_model_output_dim_and_architecture_signature_guard():
    potential_cfg = tiny_training_cfg(output_dim=1)
    force_cfg = _force_cfg()

    potential = build_model_from_config(potential_cfg, device=torch.device("cpu"), dtype=torch.float32)
    direct = build_model_from_config(force_cfg, device=torch.device("cpu"), dtype=torch.float32)

    x = torch.zeros((4, 3), dtype=torch.float32)
    assert potential(x).shape == (4, 1)
    assert direct(x).shape == (4, 3)
    assert compute_architecture_signature(potential_cfg) != compute_architecture_signature(force_cfg)

    with pytest.raises(RuntimeError, match="size mismatch"):
        direct.load_state_dict(potential.state_dict())


def test_force_direct_loader_and_runtime_methods(tmp_path, monkeypatch):
    run = make_contract_run(
        tmp_path,
        cfg_overrides={"runtime_model_kind": "force_direct", "prediction_kind": "residual_force", "output_dim": 3},
    )
    fm = load_surrogate_force_model(run["run_dir"], device="cpu")

    assert isinstance(fm, DirectForceRuntime)
    x = np.array([R_MOON_SI + 200_000.0, 0.0, 0.0])

    def _boom(*args, **kwargs):  # pragma: no cover - called only on regression
        raise AssertionError("force_direct inference must not call torch.autograd.grad")

    monkeypatch.setattr(torch.autograd, "grad", _boom)
    da = fm.predict_residual_accel_fixed(x)
    assert da.shape == (3,)
    assert np.all(np.isfinite(da))

    batch = np.vstack([x, [0.0, R_MOON_SI + 250_000.0, 0.0]])
    assert fm.predict_residual_accel_fixed(batch).shape == (2, 3)
    np.testing.assert_allclose(
        fm.predict_residual_accel_inertial(batch, [1.0, 0.0, 0.0, 0.0]),
        fm.predict_residual_accel_fixed(batch),
        atol=1e-12,
    )
    with pytest.raises(NotImplementedError, match="scalar residual potential|DeltaU"):
        fm.predict_residual_potential_fixed(x)
    with pytest.raises(ValueError, match="base_accel_fn"):
        fm.predict_total_accel_fixed(x)
    total = fm.predict_total_accel_fixed(x, lambda arr: np.zeros((np.asarray(arr).reshape(-1, 3).shape[0], 3)))
    assert total.shape == (3,)


def test_force_direct_strict_domain(tmp_path):
    run = make_contract_run(
        tmp_path,
        alt_min_km=100.0,
        alt_max_km=300.0,
        cfg_overrides={"runtime_model_kind": "force_direct", "prediction_kind": "residual_force", "output_dim": 3},
    )
    fm = load_surrogate_force_model(run["run_dir"], device="cpu", strict_domain=True)

    with pytest.raises(RuntimeError, match="strict_domain"):
        fm.predict_residual_accel_fixed(np.array([R_MOON_SI + 800_000.0, 0.0, 0.0]))


def test_force_direct_surrogate_gravity_torch_path_avoids_autograd(tmp_path, monkeypatch):
    run = make_contract_run(
        tmp_path,
        degree_min=0,
        cfg_overrides={"runtime_model_kind": "force_direct", "prediction_kind": "residual_force", "output_dim": 3},
    )
    sg = SurrogateGravityModel.from_model_dir(run["run_dir"], device_preference="cpu")
    x = torch.tensor([[R_MOON_SI + 150_000.0, 0.0, 0.0]], dtype=torch.float32)

    def _boom(*args, **kwargs):
        raise AssertionError("force_direct tensor inference must not call torch.autograd.grad")

    monkeypatch.setattr(torch.autograd, "grad", _boom)
    residual = sg.predict_residual_accel_torch(x)
    total = sg.predict_total_accel_torch(x)
    assert residual.shape == (1, 3)
    assert total.shape == (1, 3)
    assert torch.isfinite(residual).all()
    assert torch.isfinite(total).all()


def test_loader_preserves_potential_runtime(tmp_path):
    run = make_contract_run(tmp_path)
    fm = load_surrogate_force_model(run["run_dir"], device="cpu")

    assert isinstance(fm, SurrogateForceModel)
    assert not isinstance(fm, DirectForceRuntime)


def test_force_direct_training_roundtrip(tmp_path):
    h5py = pytest.importorskip("h5py")
    rng = np.random.default_rng(123)
    n = 64
    dirs = rng.normal(size=(n, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    radii = R_MOON_SI + rng.uniform(120_000.0, 220_000.0, size=(n, 1))
    x = radii * dirs
    residual_a = 2.5e-7 * dirs + 1.0e-13 * x
    u = np.zeros((n, 1))
    arr = np.concatenate([x, u, residual_a], axis=1)
    data_path = tmp_path / "toy_force_direct.h5"
    with h5py.File(data_path, "w") as handle:
        handle.create_dataset("data", data=arr)
        handle.attrs["unit_system"] = "si"
        handle.attrs["target_mode"] = "residual"
        handle.attrs["degree_min"] = 0
        handle.attrs["degree_max"] = 2
        handle.attrs["requested_degree"] = 2
        handle.attrs["mu_si"] = 4.9048695e12
        handle.attrs["r_ref_m"] = R_MOON_SI
        handle.attrs["alt_min_km"] = 100.0
        handle.attrs["alt_max_km"] = 250.0
        handle.attrs["derivative_convention_version"] = "dP_dphi_corrected_v1"
        handle.attrs["spherical_harmonic_convention"] = REQUIRED_SH_PHASE_CONVENTION
        handle.attrs["gravity_label_engine_version"] = GRAVITY_LABEL_ENGINE_VERSION

    args = SimpleNamespace(
        data=str(data_path),
        out=str(tmp_path / "force_run"),
        dataset_name="data",
        epochs=20,
        batch_size=16,
        max_samples=None,
        val_fraction=0.2,
        lr=5e-3,
        weight_decay=0.0,
        max_grad_norm=1.0,
        hidden=16,
        depth=2,
        activation="tanh",
        dropout=0.0,
        w0_first=30.0,
        w0_hidden=30.0,
        n_bands=1,
        use_residual_blocks=False,
        target_mode="residual",
        degree_min=0,
        degree_max=2,
        mu_si=None,
        r_ref_m=None,
        seed=7,
        device="cpu",
        timestamped=False,
        quiet=True,
    )
    run_dir = train_force_direct(args)
    history = [
        json.loads(line)
        for line in (run_dir / "history.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert history[-1]["val_loss"] <= history[0]["val_loss"]

    # B3: the minimal trainer now records determinism + a pinned split, so the
    # artifact is reproducibility-verifiable and the auditor TRUSTs it.
    assert (run_dir / "provenance" / "split_manifest.json").is_file()
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    prov = config["run_provenance"]
    assert prov["model_kind"] == "force_direct"
    assert prov["determinism"]
    assert prov["split_manifest_sha256"]
    status, reasons = classify_st_lrps_run(config, has_checkpoint=True)
    assert status == TRUSTED, reasons

    fm = load_surrogate_force_model(run_dir, device="cpu")
    assert isinstance(fm, DirectForceRuntime)
    pred = fm.predict_residual_accel_fixed(x[:3])
    assert pred.shape == (3, 3)
    assert np.all(np.isfinite(pred))

    report = evaluate_force_direct(run_dir, data_path, device="cpu", max_samples=16)
    assert report["potential_metrics"]["available"] is False
    assert report["potential_metrics"]["rmse_u"] is None
    assert report["acceleration_metrics"]["rmse_a_vec"] >= 0.0
    assert "mean_deg" in report["angular_metrics"]

    # force_direct now ships a measured non-conservativeness diagnostic.
    cons = report["conservativeness_metrics"]
    assert cons["n_points"] >= 1
    ratio = cons["nonconservative_ratio"]
    assert 0.0 <= ratio <= 1.0 + 1e-9
    assert np.isfinite(cons["curl_abs_rms"])
    assert any("nonconservative_ratio=" in w for w in report["warnings"])


def test_force_direct_training_rejects_missing_gravity_label_contract(tmp_path):
    h5py = pytest.importorskip("h5py")
    data_path = tmp_path / "legacy_force_direct.h5"
    arr = np.zeros((8, 7), dtype=np.float64)
    arr[:, 0] = R_MOON_SI + 150_000.0
    with h5py.File(data_path, "w") as handle:
        handle.create_dataset("data", data=arr)
        handle.attrs["unit_system"] = "si"
        handle.attrs["target_mode"] = "residual"
        handle.attrs["degree_min"] = 0
        handle.attrs["degree_max"] = 2
        handle.attrs["requested_degree"] = 2
        handle.attrs["mu_si"] = 4.9048695e12
        handle.attrs["r_ref_m"] = R_MOON_SI
        handle.attrs["alt_min_km"] = 100.0
        handle.attrs["alt_max_km"] = 250.0
        handle.attrs["derivative_convention_version"] = "dP_dphi_corrected_v1"

    args = SimpleNamespace(
        data=str(data_path),
        dataset_name="data",
        max_samples=None,
        seed=7,
    )
    with pytest.raises(ValueError, match="unsafe gravity label contract"):
        train_force_direct(args)


# --------------------------------------------------------------------------- #
# curl / non-conservativeness diagnostic (model-independent analytic checks)
# --------------------------------------------------------------------------- #
def _sample_shell(n: int, *, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    dirs = rng.normal(size=(n, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    radii = R_MOON_SI + rng.uniform(100_000.0, 300_000.0, size=(n, 1))
    return radii * dirs


def test_curl_diagnostics_conservative_linear_field():
    # a = -k r = -grad(0.5 k |r|^2): symmetric Jacobian -> curl == 0 exactly
    # (finite differences are exact for a linear field at any step).
    k = 1.0e-12
    pts = _sample_shell(256, seed=1)
    out = curl_diagnostics(lambda r: -k * np.asarray(r), pts, step_m=1.0)
    assert out["nonconservative_ratio"] < 1e-9
    assert out["curl_abs_rms"] < 1e-20


def test_curl_diagnostics_conservative_point_mass():
    # a = -mu r / |r|^3 = -grad(-mu/|r|): conservative, curl == 0 analytically.
    mu = 4.9048695e12
    pts = _sample_shell(256, seed=2)

    def accel(r):
        r = np.asarray(r, dtype=np.float64)
        rn = np.linalg.norm(r, axis=1, keepdims=True)
        return -mu * r / rn ** 3

    out = curl_diagnostics(accel, pts, step_m=1.0)
    # Central-difference truncation only; should be far below any real signal.
    assert out["nonconservative_ratio"] < 1e-4


def test_curl_diagnostics_detects_rotational_field():
    # a = (-y, x, 0) is purely rotational: curl = (0, 0, 2), Jacobian is fully
    # antisymmetric, so the non-conservative ratio is exactly 1.
    pts = _sample_shell(128, seed=3)

    def accel(r):
        r = np.asarray(r, dtype=np.float64)
        out = np.zeros_like(r)
        out[:, 0] = -r[:, 1]
        out[:, 1] = r[:, 0]
        return out

    out = curl_diagnostics(accel, pts, step_m=1.0)
    np.testing.assert_allclose(out["curl_abs_rms"], 2.0, rtol=1e-6)
    np.testing.assert_allclose(out["nonconservative_ratio"], 1.0, rtol=1e-6)


def test_curl_diagnostics_rejects_bad_input():
    pts = _sample_shell(4, seed=4)
    with pytest.raises(ValueError, match="step_m must be positive"):
        curl_diagnostics(lambda r: np.asarray(r), pts, step_m=0.0)
    with pytest.raises(ValueError, match="at least one point"):
        curl_diagnostics(lambda r: np.asarray(r), np.zeros((0, 3)), step_m=1.0)
