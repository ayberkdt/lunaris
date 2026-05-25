# -*- coding: utf-8 -*-
"""
Tests for the ST-LRPS surrogate AI/ML training-system upgrade.

Covers the targeted production-quality changes:
  * TensorMemoryDataset returns torch tensors; generalized collate handles both backends
  * single-source-of-truth production defaults
  * Laplacian is off by default; diagnostic mode never enters the objective
  * trainable vs diagnostic Laplacian gradient flow
  * radial decay + real spherical-harmonic input encodings (experimental)
  * encoding mutual exclusion
  * multi-scale SIREN w0_bands persistence through a state_dict round-trip
  * force-model strict_domain behaviour
  * ablation matrix dry-run command/manifest generation
  * residual-mag streaming (weighted reservoir) sampling smoke test

These are intentionally lightweight: tiny synthetic tensors / HDF5 files only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from st_lrps.data.datasets import (
    TensorMemoryDataset,
    collate_h5,
    collate_xyz_u_a,
)
from st_lrps.training.config import TrainConfig, parse_args
from st_lrps.training.engine import _laplacian_requested
from st_lrps.training.losses import (
    collocation_laplacian_loss,
    GradNormWeights,
    SobolevLoss,
)
from st_lrps.networks.models import (
    build_model_from_config,
    compute_architecture_signature,
    RadialDecayEncoding,
    RealSHBasisEncoding,
    _compute_harmonic_w0_bands,
)
from st_lrps.shared.scaling import IsometricScaleParams, ScalerPack

R_REF = 1.737e6
MU = 4.902800066e12


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _write_min_cloud(path: Path, *, degree_max: int = 100) -> None:
    import h5py
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h:
        h.create_dataset("data", data=np.zeros((16, 7), dtype=np.float32))
        h.attrs["central_body"] = "moon"
        h.attrs["mu_si"] = float(MU)
        h.attrs["r_ref_m"] = float(R_REF)
        h.attrs["unit_system"] = "si"
        h.attrs["degree_min"] = 20
        h.attrs["degree_max"] = degree_max
        h.attrs["requested_degree"] = degree_max
        h.attrs["target_mode"] = "residual"
        h.attrs["alt_min_km"] = 100.0
        h.attrs["alt_max_km"] = 500.0


def _ms_cfg(**overrides) -> dict:
    cfg = {
        "activation": "sine",
        "hidden": 24,
        "depth": 4,
        "dropout": 0.0,
        "use_residual_blocks": True,
        "n_bands": 3,
        "degree_min": 20,
        "degree_max": 200,
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
    }
    cfg.update(overrides)
    return cfg


def _tiny_scaler_tensors():
    sc = ScalerPack(
        x=IsometricScaleParams(mean=[0.0, 0.0, 0.0], scale=2.0e6),
        u=IsometricScaleParams(mean=[0.0], scale=1.0e4),
        a=IsometricScaleParams(mean=[0.0, 0.0, 0.0], scale=1.0e-3),
    )
    return sc.to_tensors(torch.device("cpu"), torch.float32)


# ---------------------------------------------------------------------------
# Item 3 — dataset throughput / collate
# ---------------------------------------------------------------------------

def test_tensor_memory_dataset_returns_tensors():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((10, 3)).astype(np.float32)
    u = rng.standard_normal((10, 1)).astype(np.float32)
    a = rng.standard_normal((10, 3)).astype(np.float32)
    ds = TensorMemoryDataset(x, u, a)
    item = ds[0]
    assert all(isinstance(t, torch.Tensor) for t in item)
    assert tuple(item[0].shape) == (3,)
    assert tuple(item[1].shape) == (1,)
    assert tuple(item[2].shape) == (3,)


def test_general_collate_handles_numpy_and_tensors():
    # collate_h5 must remain a usable alias for backward compatibility.
    assert collate_h5 is collate_xyz_u_a

    # numpy items (H5BlockDataset style)
    nb = [
        (np.zeros(3, np.float32), np.zeros(1, np.float32), np.zeros(3, np.float32))
        for _ in range(5)
    ]
    xn, un, an = collate_xyz_u_a(nb)
    assert isinstance(xn, torch.Tensor) and xn.dtype == torch.float32
    assert tuple(xn.shape) == (5, 3) and tuple(un.shape) == (5, 1) and tuple(an.shape) == (5, 3)

    # tensor items (TensorMemoryDataset style)
    ds = TensorMemoryDataset(
        np.ones((5, 3), np.float32), np.ones((5, 1), np.float32), np.ones((5, 3), np.float32)
    )
    tb = [ds[i] for i in range(5)]
    xt, ut, at = collate_xyz_u_a(tb)
    assert isinstance(xt, torch.Tensor) and xt.dtype == torch.float32
    assert tuple(xt.shape) == (5, 3)

    # DataLoader end-to-end produces torch tensors of the right shape.
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=4, collate_fn=collate_xyz_u_a)
    xb, ub, ab = next(iter(loader))
    assert all(isinstance(t, torch.Tensor) for t in (xb, ub, ab))
    assert tuple(xb.shape) == (4, 3)


# ---------------------------------------------------------------------------
# Item 1 / 2 — single-source-of-truth defaults; removed preset rejected
# ---------------------------------------------------------------------------

def test_no_legacy_defaults_flag_exists(tmp_path, monkeypatch):
    # The removed default preset flag, --legacy-defaults, must be
    # rejected by the parser (there is exactly one default configuration).
    data = tmp_path / "cloud.h5"
    _write_min_cloud(data)
    out = tmp_path / "run"
    monkeypatch.setattr(
        sys, "argv",
        ["st_lrps_train.py", "--data", str(data), "--out", str(out), "--legacy-defaults"],
    )
    with pytest.raises(SystemExit):
        parse_args()
    # The helper module must not retain removed preset machinery.
    import st_lrps.training.config as cfgmod
    assert not hasattr(cfgmod, "_LEGACY_DEFAULTS")
    assert not hasattr(cfgmod, "_apply_legacy_defaults")


def test_current_defaults_are_single_source_of_truth():
    cfg = TrainConfig(data="x.h5", out="o")
    assert cfg.depth == 6
    assert cfg.use_residual_blocks is True
    assert cfg.n_bands == 3
    assert cfg.accel_ramp_epochs == 40
    assert cfg.accel_min_factor == pytest.approx(0.15)
    assert cfg.direction_loss_weight == pytest.approx(0.20)
    assert cfg.direction_loss_start_epoch == 10
    assert cfg.direction_loss_floor_abs == pytest.approx(1e-7)
    assert cfg.use_altitude_balanced_loss is True
    assert cfg.use_radial_cross_loss is True
    assert cfg.radial_loss_weight == pytest.approx(0.05)
    assert cfg.cross_loss_weight == pytest.approx(0.10)
    assert cfg.best_metric == "hybrid"
    assert cfg.hybrid_direction_alpha == pytest.approx(0.30)
    assert cfg.auto_preload_mb == pytest.approx(2048.0)
    assert cfg.preload_policy == "auto"
    assert cfg.multiscale_mode == "concat_shared"
    # Experimental encodings off by default.
    assert cfg.use_radial_decay_encoding is False
    assert cfg.use_real_sh_basis is False
    # Laplacian off by default.
    assert cfg.use_laplacian_regularization is False
    assert cfg.collocation_laplacian_weight == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Item 6 — Laplacian diagnostic vs train gradient flow
# ---------------------------------------------------------------------------

def test_laplacian_train_mode_has_gradients():
    sc = _tiny_scaler_tensors()
    model = torch.nn.Sequential(torch.nn.Linear(3, 8), torch.nn.Tanh(), torch.nn.Linear(8, 1))

    # diagnostic: finite scalar, but DETACHED (no grad path into the model).
    ld = collocation_laplacian_loss(
        model, sc, r_min_m=1.8e6, r_max_m=1.9e6, n_points=16,
        device=torch.device("cpu"), n_hutchinson=2, mode="diagnostic",
    )
    assert torch.isfinite(ld)
    assert not ld.requires_grad

    # train: requires grad AND backward populates a nonzero model-parameter grad.
    lt = collocation_laplacian_loss(
        model, sc, r_min_m=1.8e6, r_max_m=1.9e6, n_points=16,
        device=torch.device("cpu"), n_hutchinson=2, mode="train",
    )
    assert lt.requires_grad
    model.zero_grad(set_to_none=True)
    lt.backward()
    nonzero = any(
        p.grad is not None and torch.isfinite(p.grad).all() and float(p.grad.abs().sum()) > 0.0
        for p in model.parameters()
    )
    assert nonzero


# ---------------------------------------------------------------------------
# Item 3 / 4 — Laplacian gating and objective purity
# ---------------------------------------------------------------------------

def test_laplacian_not_requested_by_default():
    cfg = TrainConfig(data="x.h5", out="o")
    assert _laplacian_requested(cfg) is False
    # Any explicit request flips it on.
    assert _laplacian_requested(TrainConfig(data="x", out="o", laplacian_mode="train")) is True
    assert _laplacian_requested(TrainConfig(data="x", out="o", use_laplacian_regularization=True)) is True
    assert _laplacian_requested(TrainConfig(data="x", out="o", collocation_laplacian_weight=1e-12)) is True
    assert _laplacian_requested(TrainConfig(data="x", out="o", laplacian_weight=1e-6)) is True


def _sobolev_setup():
    sc = _tiny_scaler_tensors()
    model = torch.nn.Sequential(torch.nn.Linear(3, 8), torch.nn.Tanh(), torch.nn.Linear(8, 1))
    loss = SobolevLoss(sc, a_sign=1.0)
    weights = GradNormWeights(mode="fixed")
    g = torch.Generator().manual_seed(0)
    x = torch.randn(16, 3, generator=g) * 2.0e6
    u = torch.randn(16, 1, generator=g)
    a = torch.randn(16, 3, generator=g) * 1.0e-3
    return loss, model, weights, x, u, a


def test_diagnostic_laplacian_does_not_modify_objective():
    loss, model, weights, x, u, a = _sobolev_setup()
    # Baseline: no laplacian.
    l_base, s_base = loss(model, x, u, a, weights, is_train=True, apply_laplacian=False)
    # Diagnostic in-batch laplacian with a large lambda: must NOT change the objective.
    l_diag, s_diag = loss(
        model, x, u, a, weights, is_train=True,
        apply_laplacian=True, laplacian_lambda=1.0, laplacian_mode="diagnostic",
    )
    assert s_diag["laplacian_applied"] is True
    assert s_diag["laplacian_mode"] == "diagnostic"
    assert s_diag["loss_laplacian_diag"] > 0.0      # reported as a metric
    assert s_diag["loss_laplacian_train"] == 0.0
    # Objective (loss_opt / loss_ref) identical to baseline -> diagnostic is metric-only.
    assert float(l_diag) == pytest.approx(float(l_base), abs=0.0, rel=0.0)
    assert s_diag["loss_opt"] == pytest.approx(s_base["loss_opt"])
    assert s_diag["loss_ref"] == pytest.approx(s_base["loss_ref"])


def test_train_laplacian_modifies_objective_and_flows_grad():
    loss, model, weights, x, u, a = _sobolev_setup()
    l_train, s_train = loss(
        model, x, u, a, weights, is_train=True,
        apply_laplacian=True, laplacian_lambda=1.0, laplacian_mode="train",
    )
    assert s_train["laplacian_mode"] == "train"
    assert s_train["loss_laplacian_train"] > 0.0
    assert l_train.requires_grad
    model.zero_grad(set_to_none=True)
    l_train.backward()
    assert any(
        p.grad is not None and float(p.grad.abs().sum()) > 0.0 for p in model.parameters()
    )


# ---------------------------------------------------------------------------
# Item 6 / 7 — radial decay and real SH encodings
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("append_raw", [True, False])
def test_radial_decay_encoding_shape_and_finite(append_raw):
    enc = RadialDecayEncoding(max_power=4, append_raw=append_raw)
    expected = 1 + 3 + 4 + (3 if append_raw else 0)
    assert enc.out_dim == expected
    x = torch.randn(13, 3) * 2.0e6
    y = enc(x)
    assert tuple(y.shape) == (13, expected)
    assert torch.isfinite(y).all()
    # Degenerate (origin) input must not produce NaN/Inf thanks to the eps clamp.
    y0 = enc(torch.zeros(2, 3))
    assert torch.isfinite(y0).all()


def test_radial_decay_encoding_registered_in_architecture_signature():
    base = {"activation": "sine", "hidden": 16, "depth": 3, "n_bands": 1,
            "use_radial_decay_encoding": True, "radial_decay_max_power": 4,
            "radial_decay_append_raw": True}
    sig4 = compute_architecture_signature(base)
    sig3 = compute_architecture_signature({**base, "radial_decay_max_power": 3})
    assert sig4 != sig3   # changing the power changes the architecture signature


def test_real_sh_basis_shape_and_finite():
    enc = RealSHBasisEncoding(degree_max=4, append_raw=True, include_radial=True)
    expected = (4 + 1) ** 2 + 1 + 3
    assert enc.out_dim == expected
    # Points at the poles, equator, and a generic direction must all be finite.
    x = torch.tensor(
        [[0.0, 0.0, 2.0e6], [0.0, 0.0, -2.0e6], [2.0e6, 0.0, 0.0],
         [0.0, 2.0e6, 0.0], [1.0e6, 1.0e6, 1.0e6]],
        dtype=torch.float32,
    )
    y = enc(x)
    assert tuple(y.shape) == (5, expected)
    assert torch.isfinite(y).all()
    # Angular-only variant.
    enc2 = RealSHBasisEncoding(degree_max=3, append_raw=False, include_radial=False)
    assert enc2.out_dim == (3 + 1) ** 2
    assert torch.isfinite(enc2(x)).all()


def test_encoding_mutual_exclusion():
    with pytest.raises(ValueError, match="one input encoding"):
        build_model_from_config({
            "activation": "sine", "hidden": 8, "depth": 2, "n_bands": 1,
            "use_radial_decay_encoding": True, "use_real_sh_basis": True,
        })
    with pytest.raises(ValueError, match="one input encoding"):
        build_model_from_config({
            "activation": "sine", "hidden": 8, "depth": 2, "n_bands": 1,
            "use_radial_separation": True, "use_real_sh_basis": True,
        })


# ---------------------------------------------------------------------------
# Item 1 / reload-safety — multi-scale SIREN w0_bands persistence
# ---------------------------------------------------------------------------

def _inner_multiscale(module):
    """Return the MultiScaleSirenMLP backbone (the module owning w0_bands_tensor)."""
    return next(m for m in module.modules() if hasattr(m, "w0_bands_tensor"))


def test_multiscale_siren_w0_bands_are_persisted():
    bands = _compute_harmonic_w0_bands(3, 20, 200)
    cfg = _ms_cfg(w0_bands=bands)
    model = build_model_from_config(cfg, device=torch.device("cpu"), dtype=torch.float32)
    assert len(model.w0_bands) == 3
    assert [round(b, 4) for b in model.w0_bands] == [round(b, 4) for b in bands]

    # The spectrum is stored in the state_dict as a persistent buffer.
    sd = model.state_dict()
    assert any(k.endswith("w0_bands_tensor") for k in sd)

    # Canonical reload: rebuild from the same cfg and load → identical predictions.
    model_same = build_model_from_config(cfg, device=torch.device("cpu"), dtype=torch.float32)
    model_same.load_state_dict(sd)
    x = torch.randn(8, 3)
    with torch.no_grad():
        assert torch.allclose(model(x), model_same(x), atol=1e-6)

    # Adversarial reload: reconstruct with DIFFERENT placeholder bands, then load.
    # The persistent w0_bands_tensor buffer must drive the backbone's functional
    # spectrum back to the trained values, so predictions still match.
    model_bad = build_model_from_config(
        _ms_cfg(w0_bands=[15.0, 15.0, 15.0]), device=torch.device("cpu"), dtype=torch.float32
    )
    inner_before = _inner_multiscale(model_bad)
    assert [round(b, 4) for b in inner_before.w0_bands] != [round(b, 4) for b in bands]
    model_bad.load_state_dict(sd)
    inner_after = _inner_multiscale(model_bad)
    assert [round(b, 4) for b in inner_after.w0_bands] == [round(b, 4) for b in bands]
    with torch.no_grad():
        assert torch.allclose(model(x), model_bad(x), atol=1e-6)


# ---------------------------------------------------------------------------
# Item 9 — force-model strict_domain
# ---------------------------------------------------------------------------

def test_force_model_strict_domain_flag_logic():
    from st_lrps.runtime.force_model import SurrogateForceModel

    scaler = _tiny_scaler_tensors()
    model = build_model_from_config(
        _ms_cfg(n_bands=1, w0_bands=None, degree_min=10, degree_max=60),
        device=torch.device("cpu"), dtype=torch.float32,
    )
    cfg = {
        "resolved_mu_si": MU,
        "resolved_a_sign": 1.0,
        "resolved_r_ref_m": R_REF,
        "degree_min": -1,
        "altitude_min_km": 100.0,
        "altitude_max_km": 500.0,
    }

    fm_lax = SurrogateForceModel(model, scaler, cfg, torch.device("cpu"), strict_domain=False)
    fm_strict = SurrogateForceModel(model, scaler, cfg, torch.device("cpu"), strict_domain=True)

    # In-domain point: ~200 km altitude, well within the scaler radius.
    x_ok = np.array([R_REF + 200e3, 0.0, 0.0], dtype=np.float64)
    assert fm_lax.domain_status(x_ok)["recommended_fallback"] is False
    # Both modes succeed in-domain.
    _ = fm_lax.predict_residual_accel(x_ok)
    _ = fm_strict.predict_residual_accel(x_ok)

    # Far out-of-domain point (radius beyond the scaler radius → fallback recommended).
    x_bad = np.array([6.0e6, 0.0, 0.0], dtype=np.float64)
    assert fm_lax.domain_status(x_bad)["recommended_fallback"] is True
    # Lax mode returns a value (extrapolation, warn-once); strict mode raises.
    out = fm_lax.predict_residual_accel(x_bad)
    assert out.shape == (3,)
    with pytest.raises(RuntimeError, match="strict_domain"):
        fm_strict.predict_residual_accel(x_bad)
    with pytest.raises(RuntimeError, match="strict_domain"):
        fm_strict.predict_total_accel(x_bad)


# ---------------------------------------------------------------------------
# Item 8 — ablation matrix dry-run
# ---------------------------------------------------------------------------

def test_ablation_command_generation_dry_run(tmp_path):
    from st_lrps.evaluation import ablation as ram

    out_root = tmp_path / "ablations"
    rc = ram.main([
        "--train-data", "train.h5",
        "--val-data", "val.h5",
        "--out-root", str(out_root),
        "--seed", "7",
        "--dry-run",
    ])
    assert rc == 0

    commands_txt = out_root / "ablation_commands.txt"
    manifest_json = out_root / "ablation_manifest.json"
    assert commands_txt.exists()
    assert manifest_json.exists()

    manifest = json.loads(manifest_json.read_text(encoding="utf-8"))
    assert manifest["execute"] is False
    # The manifest documents that the default = the recommended production architecture.
    assert "note" in manifest and "recommended production" in manifest["note"]
    names = [a["name"] for a in manifest["ablations"]]
    expected = [
        "baseline_single_siren", "multiscale_siren", "multiscale_no_resblocks",
        "multiscale_no_direction", "multiscale_no_altitude_balance",
        "multiscale_no_radial_cross", "radial_decay_encoding",
        "real_sh_basis_encoding_optional", "additive_multiband",
    ]
    assert names == expected
    for ab in manifest["ablations"]:
        assert isinstance(ab["command"], list) and ab["command"]
        assert ab["seed"] == 7

    assert all("expected_purpose" in ab for ab in manifest["ablations"])
    assert all("experimental" in ab for ab in manifest["ablations"])

    # Dry-run must NOT create any per-run directory (only the root + files).
    assert not (out_root / "baseline_single_siren").exists()


def test_ablation_matrix_contains_radial_decay_and_real_sh(tmp_path):
    from st_lrps.evaluation import ablation as ram

    out_root = tmp_path / "ablations2"
    rc = ram.main(["--train-data", "train.h5", "--out-root", str(out_root), "--dry-run"])
    assert rc == 0
    manifest = json.loads((out_root / "ablation_manifest.json").read_text(encoding="utf-8"))
    by_name = {a["name"]: a for a in manifest["ablations"]}
    assert "radial_decay_encoding" in by_name
    assert "real_sh_basis_encoding_optional" in by_name
    assert "additive_multiband" in by_name
    assert "--use-radial-decay-encoding" in by_name["radial_decay_encoding"]["flags"]
    assert "--use-real-sh-basis" in by_name["real_sh_basis_encoding_optional"]["flags"]
    assert "--multiscale-mode" in by_name["additive_multiband"]["flags"]


# ---------------------------------------------------------------------------
# Item 7 — residual-mag streaming sampling
# ---------------------------------------------------------------------------

def test_residual_mag_streaming_sampling(monkeypatch):
    import st_lrps.data.spatial_cloud_generator as scg

    def _fake_sample(cnt, lo, hi, rng):
        r = rng.uniform(lo, hi, (cnt, 1))
        d = rng.standard_normal((cnt, 3))
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        return r * d

    def _fake_labels(xyz, blob):
        n = xyz.shape[0]
        out = np.zeros((n, 7), dtype=np.float64)
        out[:, 0:3] = xyz
        # deterministic pseudo-residual acceleration from position norm
        rnorm = np.linalg.norm(xyz, axis=1)
        out[:, 4] = np.sin(rnorm * 1e-5) * 1e-4
        out[:, 5] = np.cos(rnorm * 1e-5) * 1e-4
        out[:, 6] = (rnorm % 1000.0) * 1e-8
        return out

    monkeypatch.setattr(scg, "sample_uniform_shell_xyz", _fake_sample)
    monkeypatch.setattr(scg, "_compute_labels_for_xyz", _fake_labels)

    r_ref = R_REF
    r_min = r_ref + 200e3
    r_max = r_ref + 600e3

    a = scg._generate_residual_mag_component(
        400, r_min, r_max, r_ref, 7, {}, 128, candidate_multiplier=4, streaming=True
    )
    b = scg._generate_residual_mag_component(
        400, r_min, r_max, r_ref, 7, {}, 128, candidate_multiplier=4, streaming=True
    )
    c = scg._generate_residual_mag_component(
        400, r_min, r_max, r_ref, 11, {}, 128, candidate_multiplier=4, streaming=True
    )

    # exact requested count, finite, reproducible with same seed, different across seeds
    assert a.shape == (400, 7)
    assert np.all(np.isfinite(a))
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)

    # exact path also returns exactly the requested count
    exact = scg._generate_residual_mag_component(
        400, r_min, r_max, r_ref, 7, {}, 128, candidate_multiplier=4, streaming=False
    )
    assert exact.shape == (400, 7)
    assert np.all(np.isfinite(exact))
