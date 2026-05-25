# -*- coding: utf-8 -*-
"""
Architecture + Laplacian-cleanup validation tests for the ST-LRPS surrogate.

This is a self-contained, CPU-only suite covering the post-upgrade state:
  * removed ``--legacy-defaults`` preset is rejected; no dynamic-weight alias
  * single-source-of-truth recommended defaults
  * Laplacian off by default; diagnostic mode never enters the objective;
    train mode backpropagates
  * RadialDecayEncoding + RealSHBasisEncoding (shape, finiteness, signature)
  * encoding mutual exclusion
  * additive vs concat-shared multi-scale forward shapes
  * ablation matrix exposes the new experimental architectures

All tests use tiny synthetic tensors / a tiny HDF5 file.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from st_lrps.training.config import TrainConfig, parse_args
from st_lrps.training.engine import _laplacian_requested
from st_lrps.training.losses import (
    GradNormWeights,
    SobolevLoss,
    collocation_laplacian_loss,
)
from st_lrps.networks.models import (
    RadialDecayEncoding,
    RealSHBasisEncoding,
    build_model_from_config,
    compute_architecture_signature,
)
from st_lrps.shared.scaling import IsometricScaleParams, ScalerPack

R_REF = 1.737e6
MU = 4.902800066e12


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _write_tiny_cloud(path: Path) -> None:
    import h5py
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h:
        h.create_dataset("data", data=np.zeros((16, 7), dtype=np.float32))
        h.attrs["central_body"] = "moon"
        h.attrs["mu_si"] = float(MU)
        h.attrs["r_ref_m"] = float(R_REF)
        h.attrs["unit_system"] = "si"
        h.attrs["degree_min"] = 20
        h.attrs["degree_max"] = 100
        h.attrs["requested_degree"] = 100
        h.attrs["target_mode"] = "residual"
        h.attrs["alt_min_km"] = 100.0
        h.attrs["alt_max_km"] = 500.0


def _tiny_scaler_tensors() -> ScalerPack:
    return ScalerPack(
        x=IsometricScaleParams(mean=[0.0, 0.0, 0.0], scale=2.0e6),
        u=IsometricScaleParams(mean=[0.0], scale=1.0e4),
        a=IsometricScaleParams(mean=[0.0, 0.0, 0.0], scale=1.0e-3),
    ).to_tensors(torch.device("cpu"), torch.float32)


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


# ---------------------------------------------------------------------------
# 4.1 / 4.2 — removed preset rejected; single-source-of-truth defaults
# ---------------------------------------------------------------------------

def test_no_legacy_defaults_flag_exists(tmp_path, monkeypatch):
    data = tmp_path / "cloud.h5"
    _write_tiny_cloud(data)
    monkeypatch.setattr(
        sys, "argv", ["st_lrps_train.py", "--data", str(data), "--legacy-defaults"]
    )
    with pytest.raises(SystemExit):
        parse_args()
    # And the deprecated dynamic-weight alias is gone too.
    monkeypatch.setattr(
        sys, "argv", ["st_lrps_train.py", "--data", str(data), "--dynamic-weights"]
    )
    with pytest.raises(SystemExit):
        parse_args()


def test_current_defaults_are_single_source_of_truth(tmp_path, monkeypatch):
    data = tmp_path / "cloud.h5"
    _write_tiny_cloud(data)
    out = tmp_path / "run"
    monkeypatch.setattr(
        sys, "argv", ["st_lrps_train.py", "--data", str(data), "--out", str(out)]
    )
    cfg = parse_args()
    assert cfg.activation == "sine"
    assert cfg.depth == 6
    assert cfg.use_residual_blocks is True
    assert cfg.n_bands == 3
    assert cfg.use_altitude_balanced_loss is True
    assert cfg.use_radial_cross_loss is True
    assert cfg.best_metric == "hybrid"
    assert cfg.preload_policy == "auto"
    assert cfg.auto_preload_mb == pytest.approx(2048.0)
    assert cfg.use_radial_decay_encoding is False
    assert cfg.use_real_sh_basis is False
    assert cfg.multiscale_mode == "concat_shared"
    removed_attr = "dynamic" + "_weights"
    assert not hasattr(cfg, removed_attr)


# ---------------------------------------------------------------------------
# 4.3 — Laplacian not requested by default
# ---------------------------------------------------------------------------

def test_laplacian_not_requested_by_default():
    cfg = TrainConfig(data="x.h5", out="o")
    assert _laplacian_requested(cfg) is False
    assert _laplacian_requested(TrainConfig(data="x", out="o", use_laplacian_regularization=True)) is True
    assert _laplacian_requested(TrainConfig(data="x", out="o", laplacian_weight=1e-12)) is True
    assert _laplacian_requested(TrainConfig(data="x", out="o", collocation_laplacian_weight=1e-12)) is True
    assert _laplacian_requested(TrainConfig(data="x", out="o", laplacian_mode="train")) is True


# ---------------------------------------------------------------------------
# 4.4 — diagnostic Laplacian is metric-only, never in the objective
# ---------------------------------------------------------------------------

def test_diagnostic_laplacian_does_not_modify_objective():
    loss, model, weights, x, u, a = _sobolev_setup()

    torch.manual_seed(0)
    l_base, s_base = loss(model, x, u, a, weights, is_train=True, apply_laplacian=False)
    torch.manual_seed(0)
    l_diag, s_diag = loss(
        model, x, u, a, weights, is_train=True,
        apply_laplacian=True, laplacian_lambda=1.0, laplacian_mode="diagnostic",
    )

    assert "loss_laplacian_diag" in s_diag
    assert s_diag["laplacian_applied"] is True
    assert s_diag["loss_laplacian_diag"] > 0.0
    assert s_diag["loss_laplacian_train"] == 0.0
    # The non-laplacian objective terms are unchanged...
    assert s_diag["mse_u"] == pytest.approx(s_base["mse_u"])
    assert s_diag["mse_a"] == pytest.approx(s_base["mse_a"])
    # ...and crucially the optimization/reference objective itself is unchanged,
    # i.e. it does NOT include laplacian_lambda * loss_laplacian.
    assert s_diag["loss_opt"] == pytest.approx(s_base["loss_opt"])
    assert s_diag["loss_ref"] == pytest.approx(s_base["loss_ref"])
    assert float(l_diag) == pytest.approx(float(l_base))


# ---------------------------------------------------------------------------
# 4.5 — train Laplacian backpropagates; diagnostic does not require grad
# ---------------------------------------------------------------------------

def test_train_laplacian_requires_grad():
    sc = _tiny_scaler_tensors()
    model = torch.nn.Sequential(torch.nn.Linear(3, 8), torch.nn.Tanh(), torch.nn.Linear(8, 1))

    diag = collocation_laplacian_loss(
        model, sc, r_min_m=1.8e6, r_max_m=1.9e6, n_points=16,
        device=torch.device("cpu"), n_hutchinson=2, mode="diagnostic",
    )
    assert torch.isfinite(diag)
    assert diag.requires_grad is False

    train = collocation_laplacian_loss(
        model, sc, r_min_m=1.8e6, r_max_m=1.9e6, n_points=16,
        device=torch.device("cpu"), n_hutchinson=2, mode="train",
    )
    assert train.requires_grad is True
    model.zero_grad(set_to_none=True)
    train.backward()
    assert any(
        p.grad is not None and torch.isfinite(p.grad).all() and float(p.grad.abs().sum()) > 0.0
        for p in model.parameters()
    )


# ---------------------------------------------------------------------------
# 4.6 — RadialDecayEncoding shape + finiteness
# ---------------------------------------------------------------------------

def test_radial_decay_encoding_shape_and_finite():
    x = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
         [1.0, 2.0, 3.0], [-2.0, 0.5, -1.0]],
        dtype=torch.float32,
    ) * 1.0e6

    enc = RadialDecayEncoding(max_power=4, append_raw=True)
    assert enc.out_dim == 1 + 3 + 4 + 3
    y = enc(x)
    assert tuple(y.shape) == (6, 1 + 3 + 4 + 3)
    assert torch.isfinite(y).all()

    enc2 = RadialDecayEncoding(max_power=4, append_raw=False)
    assert enc2.out_dim == 1 + 3 + 4
    y2 = enc2(x)
    assert tuple(y2.shape) == (6, 1 + 3 + 4)
    assert torch.isfinite(y2).all()


# ---------------------------------------------------------------------------
# 4.7 — RadialDecayEncoding participates in the architecture signature
# ---------------------------------------------------------------------------

def test_radial_decay_encoding_registered_in_architecture_signature():
    base = {
        "activation": "sine", "hidden": 16, "depth": 3, "n_bands": 1,
        "use_radial_decay_encoding": True, "radial_decay_max_power": 4,
        "radial_decay_append_raw": True,
    }
    sig4 = compute_architecture_signature(base)
    sig3 = compute_architecture_signature({**base, "radial_decay_max_power": 3})
    sig_noraw = compute_architecture_signature({**base, "radial_decay_append_raw": False})
    assert sig4 != sig3
    assert sig4 != sig_noraw


# ---------------------------------------------------------------------------
# 4.8 — RealSHBasisEncoding shape + finiteness (incl. poles)
# ---------------------------------------------------------------------------

def test_real_sh_basis_shape_and_finite():
    x = torch.tensor(
        [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
         [0.3, -0.7, 0.5], [2.0, 1.0, -3.0]],
        dtype=torch.float32,
    ) * 1.0e6

    enc = RealSHBasisEncoding(degree_max=4, append_raw=True, include_radial=True)
    assert enc.out_dim == (4 + 1) ** 2 + 1 + 3
    y = enc(x)
    assert tuple(y.shape) == (6, (4 + 1) ** 2 + 1 + 3)
    assert torch.isfinite(y).all()

    enc2 = RealSHBasisEncoding(degree_max=4, append_raw=False, include_radial=False)
    assert enc2.out_dim == (4 + 1) ** 2
    y2 = enc2(x)
    assert tuple(y2.shape) == (6, (4 + 1) ** 2)
    assert torch.isfinite(y2).all()


# ---------------------------------------------------------------------------
# 4.9 — RealSHBasisEncoding low-degree sanity (no overfit to exact ordering)
# ---------------------------------------------------------------------------

def test_real_sh_basis_low_degree_sanity():
    # degree 1, angular only: documented per-(l,m) order is
    #   [Y_{0,0}, Y_{1,0}, Y_{1,1}^cos, Y_{1,1}^sin]
    enc = RealSHBasisEncoding(degree_max=1, append_raw=False, include_radial=False)
    assert enc.out_dim == 4

    north = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32)
    south = torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32)
    xax = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
    yax = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float32)

    yN = enc(north)[0]
    yS = enc(south)[0]
    yX = enc(xax)[0]
    yY = enc(yax)[0]

    # All finite.
    for v in (yN, yS, yX, yY):
        assert torch.isfinite(v).all()

    # Y_{0,0} (first term) is constant across all directions.
    const_vals = torch.stack([yN[0], yS[0], yX[0], yY[0]])
    assert torch.allclose(const_vals, const_vals[0].expand_as(const_vals), atol=1e-5)

    # At the poles the m>0 (sectoral) terms vanish because nx=ny=0.
    assert abs(float(yN[2])) < 1e-5 and abs(float(yN[3])) < 1e-5
    assert abs(float(yS[2])) < 1e-5 and abs(float(yS[3])) < 1e-5

    # Off the poles at least one m=1 term is nonzero.
    assert max(abs(float(yX[2])), abs(float(yX[3]))) > 1e-2
    assert max(abs(float(yY[2])), abs(float(yY[3]))) > 1e-2


# ---------------------------------------------------------------------------
# 4.10 — encoding mutual exclusion
# ---------------------------------------------------------------------------

def test_encoding_mutual_exclusion():
    with pytest.raises(ValueError, match="one input encoding"):
        build_model_from_config({
            "activation": "sine", "hidden": 8, "depth": 2, "n_bands": 1,
            "use_radial_decay_encoding": True, "use_real_sh_basis": True,
        })
    # SIREN + Fourier remains a hard error.
    with pytest.raises(ValueError):
        build_model_from_config({
            "activation": "sine", "hidden": 8, "depth": 2, "n_bands": 1,
            "use_fourier": True,
        })


# ---------------------------------------------------------------------------
# 4.11 — additive vs concat-shared multi-scale forward shapes
# ---------------------------------------------------------------------------

def test_additive_multiband_forward_shape():
    common = {
        "activation": "sine", "hidden": 32, "depth": 3, "n_bands": 3,
        "degree_min": 20, "degree_max": 100, "use_residual_blocks": True,
    }
    x = torch.randn(5, 3)

    m_add = build_model_from_config(
        {**common, "multiscale_mode": "additive"},
        device=torch.device("cpu"), dtype=torch.float32,
    )
    y_add = m_add(x)
    assert tuple(y_add.shape) == (5, 1)
    assert torch.isfinite(y_add).all()

    m_cat = build_model_from_config(
        {**common, "multiscale_mode": "concat_shared"},
        device=torch.device("cpu"), dtype=torch.float32,
    )
    y_cat = m_cat(x)
    assert tuple(y_cat.shape) == (5, 1)
    assert torch.isfinite(y_cat).all()


# ---------------------------------------------------------------------------
# 4.12 — ablation matrix exposes the new experimental architectures
# ---------------------------------------------------------------------------

def test_ablation_matrix_contains_new_experimental_architectures(tmp_path):
    from st_lrps.evaluation import ablation as ram

    names = [a["name"] for a in ram.ABLATIONS]
    for required in ("radial_decay_encoding", "real_sh_basis_encoding_optional", "additive_multiband"):
        assert required in names

    # Generate the manifest (dry-run) and check the note + flag wiring.
    out_root = tmp_path / "abl"
    rc = ram.main(["--train-data", "train.h5", "--out-root", str(out_root), "--dry-run"])
    assert rc == 0
    manifest = json.loads((out_root / "ablation_manifest.json").read_text(encoding="utf-8"))
    assert "note" in manifest and "recommended" in manifest["note"].lower()
    by_name = {a["name"]: a for a in manifest["ablations"]}
    assert "--use-radial-decay-encoding" in by_name["radial_decay_encoding"]["flags"]
    assert "--use-real-sh-basis" in by_name["real_sh_basis_encoding_optional"]["flags"]
    assert "--multiscale-mode" in by_name["additive_multiband"]["flags"]


# ---------------------------------------------------------------------------
# 5 (optional) — cross-check RealSHBasisEncoding against scipy if available
# ---------------------------------------------------------------------------

def test_real_sh_basis_matches_scipy_subspace_if_available():
    """Convention-robust scipy cross-check.

    Real SH are real linear combinations of complex SH. We confirm that every
    column of our basis lies in the span of scipy's complex SH (Re + Im) at the
    same points. This is invariant to normalization, sign, m-ordering, and the
    cos/sin phase convention, so it is robust across scipy versions. Skips if
    scipy (or a usable spherical-harmonic function) is unavailable.
    """
    sp = pytest.importorskip("scipy.special")

    L = 2
    rng = np.random.default_rng(0)
    n = 600
    v = rng.standard_normal((n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    nx, ny, nz = v[:, 0], v[:, 1], v[:, 2]
    colat = np.arccos(np.clip(nz, -1.0, 1.0))   # polar angle θ ∈ [0, π]
    az = np.arctan2(ny, nx)                      # azimuth φ ∈ (-π, π]

    # Build scipy complex SH for all (l, m), l = 0..L, using whichever API exists.
    cols = []
    try:
        if hasattr(sp, "sph_harm_y"):
            sph = lambda m, l: sp.sph_harm_y(l, m, colat, az)      # (n, m, theta=polar, phi=azimuth)
        else:  # pragma: no cover - older scipy
            sph = lambda m, l: sp.sph_harm(m, l, az, colat)        # (m, n, theta=azimuth, phi=polar)
        for l in range(L + 1):
            for m in range(-l, l + 1):
                yc = np.asarray(sph(m, l), dtype=np.complex128)
                cols.append(yc.real)
                cols.append(yc.imag)
    except Exception:  # pragma: no cover - scipy API mismatch
        pytest.skip("scipy spherical-harmonic API not usable in this environment")

    B = np.stack(cols, axis=1)   # (n, 2*(L+1)^2) real spanning matrix

    enc = RealSHBasisEncoding(degree_max=L, append_raw=False, include_radial=False)
    A = enc(torch.tensor(v, dtype=torch.float64)).detach().numpy()  # (n, (L+1)^2)

    # Least-squares projection of A onto span(B); residual must be ~0.
    coef, *_ = np.linalg.lstsq(B, A, rcond=None)
    resid = A - B @ coef
    rel = float(np.linalg.norm(resid) / max(np.linalg.norm(A), 1e-12))
    assert rel < 1e-3, f"RealSH basis not in scipy SH span (rel residual={rel:.2e})"
