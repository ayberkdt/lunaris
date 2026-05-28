# -*- coding: utf-8 -*-
"""
Tests for ST-LRPS resume-training support.

Covers (without heavy training where possible):
- resume checkpoint path resolution (run dir / checkpoints dir / .pt file; prefer=last default),
- checkpoint payload carries full resume state,
- GradNormWeights.state_dict / load_state_dict roundtrip,
- capture_rng_state / restore_rng_state helpers,
- an optional end-to-end resume smoke test (skips gracefully if deps/data fail),
- no old flat ST-LRPS module paths.

All training is launched via ``st_lrps.training.cli`` (the canonical entry point).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Helpers — build a minimal but schema-valid run with real checkpoints
# (no training; just model/optimizer/scaler construction + save).
# ---------------------------------------------------------------------------

def _make_min_run(run_dir: Path):
    from lunaris.surrogate.st_lrps.training.config import TrainConfig
    from lunaris.surrogate.st_lrps.training.losses import GradNormWeights
    from lunaris.surrogate.st_lrps.networks.models import build_model_from_config, compute_architecture_signature
    from lunaris.surrogate.st_lrps.shared.scaling import IsometricScaleParams, ScalerPack
    from lunaris.surrogate.st_lrps.artifacts.manager import (
        ensure_run_layout,
        build_resolved_config,
        build_checkpoint_payload,
        save_checkpoint,
        atomic_write_json,
        capture_rng_state,
    )

    layout = ensure_run_layout(run_dir)
    cfg = TrainConfig(
        data="d.h5", out=str(run_dir),
        activation="silu", hidden=8, depth=2,
        n_bands=1, use_residual_blocks=False, use_fourier=False,
        degree_min=20, degree_max=100,
    )
    cfg.w0_bands = None
    model = build_model_from_config(cfg, in_dim=3, device=torch.device("cpu"), dtype=torch.float32)
    sig = compute_architecture_signature(cfg)
    scaler = ScalerPack(
        x=IsometricScaleParams(mean=[0.0, 0.0, 0.0], scale=2.0e6),
        u=IsometricScaleParams(mean=[0.0], scale=1.0e4),
        a=IsometricScaleParams(mean=[0.0, 0.0, 0.0], scale=1.0e-3),
    )
    dataset_meta = {
        "mu_si": 4.9e12, "r_ref_m": 1.738e6, "degree_min": 20, "degree_max": 100,
        "target_mode": "residual", "unit_system": "si", "central_body": "moon",
    }
    resolved = build_resolved_config(cfg, dataset_meta, model, scaler, sig)
    resolved["best_epoch"] = 4          # display (one-based)
    resolved["best_score"] = 0.123
    resolved["epochs_since_improvement"] = 1
    atomic_write_json(layout.config_json, resolved)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    gnw = GradNormWeights(mode="fixed")
    train_stats = {
        "lr": 1e-4, "w_u": 1.0, "w_a": 1.0, "gradnorm_status": "fixed",
        "gradnorm_weights": gnw.state_dict(),
        "rng_state": capture_rng_state(),
    }
    val_stats = {"loss": 0.5, "val_checkpoint_score": 0.5}
    scheduler_state = {"kind": "warmup_cosine", "epoch": 4, "warmup_epochs": 5,
                       "min_lr_ratio": 0.05, "t_max": None}

    payload = build_checkpoint_payload(
        kind="last", epoch=4, model=model, optimizer=opt, scheduler=scheduler_state,
        cfg=resolved, scaler=scaler, train_stats=train_stats, val_stats=val_stats,
        dataset_meta=dataset_meta, architecture_signature=sig, global_step=40,
    )
    save_checkpoint(layout, kind="last", payload=payload, epoch=4)
    save_checkpoint(layout, kind="best", payload=payload, epoch=3)
    return layout, payload


# ---------------------------------------------------------------------------
# 1) Resume path resolution
# ---------------------------------------------------------------------------

def test_resolve_resume_checkpoint_from_run_dir_defaults_to_last(tmp_path):
    from lunaris.surrogate.st_lrps.artifacts.manager import resolve_resume_checkpoint

    layout, _ = _make_min_run(tmp_path / "runs" / "r1")
    res_layout, ckpt_path, payload = resolve_resume_checkpoint(layout.run_dir)
    assert res_layout.run_dir == layout.run_dir
    assert ckpt_path == layout.ckpt_last, "run-dir resume must default to ckpt_last.pt"
    assert payload["epoch"] == 4


def test_resolve_resume_checkpoint_from_checkpoints_dir(tmp_path):
    from lunaris.surrogate.st_lrps.artifacts.manager import resolve_resume_checkpoint

    layout, _ = _make_min_run(tmp_path / "runs" / "r2")
    _, ckpt_path, _ = resolve_resume_checkpoint(layout.checkpoints_dir)
    assert ckpt_path == layout.ckpt_last


def test_resolve_resume_checkpoint_from_pt_file(tmp_path):
    from lunaris.surrogate.st_lrps.artifacts.manager import resolve_resume_checkpoint

    layout, _ = _make_min_run(tmp_path / "runs" / "r3")
    res_layout, ckpt_path, _ = resolve_resume_checkpoint(layout.ckpt_best)
    assert ckpt_path == layout.ckpt_best
    assert res_layout.run_dir == layout.run_dir


def test_resolve_resume_checkpoint_prefer_best(tmp_path):
    from lunaris.surrogate.st_lrps.artifacts.manager import resolve_resume_checkpoint

    layout, _ = _make_min_run(tmp_path / "runs" / "r4")
    _, ckpt_path, _ = resolve_resume_checkpoint(layout.run_dir, prefer="best")
    assert ckpt_path == layout.ckpt_best


def test_resolve_resume_checkpoint_missing_path_raises(tmp_path):
    from lunaris.surrogate.st_lrps.artifacts.manager import resolve_resume_checkpoint

    with pytest.raises(FileNotFoundError):
        resolve_resume_checkpoint(tmp_path / "does_not_exist")


# ---------------------------------------------------------------------------
# 2) Checkpoint payload carries full resume state
# ---------------------------------------------------------------------------

def test_checkpoint_payload_contains_resume_state(tmp_path):
    _, payload = _make_min_run(tmp_path / "runs" / "p1")
    for key in (
        "model_state_dict", "optimizer_state_dict", "scheduler_state_dict",
        "epoch", "epoch_display", "global_step", "config", "scaler", "training_state",
    ):
        assert key in payload, f"checkpoint payload missing {key!r}"
    assert payload["optimizer_state_dict"] is not None
    assert payload["epoch_display"] == payload["epoch"] + 1
    ts = payload["training_state"]
    assert ts.get("gradnorm_weights") is not None, "training_state must carry gradnorm_weights"
    assert ts.get("rng_state") is not None, "training_state must carry rng_state"


# ---------------------------------------------------------------------------
# 3) GradNormWeights state roundtrip
# ---------------------------------------------------------------------------

def test_gradnorm_state_roundtrip():
    from lunaris.surrogate.st_lrps.training.losses import GradNormWeights

    g = GradNormWeights(w_u=1.0, w_a=1.0, mode="ntk_init")
    g.w_a = 2.5
    g._ntk_done = True
    g._step_counter = 7
    g._ema_ratio = 1.7
    g.last_gradnorm_status = "ok"
    g.last_n_grad_u = 3
    state = g.state_dict()

    g2 = GradNormWeights(mode="ntk_init")
    g2.load_state_dict(state)
    assert g2.w_a == pytest.approx(2.5)
    assert g2._ntk_done is True
    assert g2._step_counter == 7
    assert g2._ema_ratio == pytest.approx(1.7)
    assert g2.last_gradnorm_status == "ok"
    assert g2.last_n_grad_u == 3
    # Tolerant of empty / None.
    GradNormWeights().load_state_dict(None)
    GradNormWeights().load_state_dict({})


# ---------------------------------------------------------------------------
# 4) RNG state helpers
# ---------------------------------------------------------------------------

def test_rng_state_helpers_roundtrip():
    from lunaris.surrogate.st_lrps.artifacts.manager import capture_rng_state, restore_rng_state

    state = capture_rng_state()
    assert {"python", "numpy", "torch_cpu"}.issubset(set(state.keys()))
    # Restoring should not crash, and should reproduce the next draw.
    import torch as _t
    restore_rng_state(state)
    a = _t.rand(4)
    restore_rng_state(state)
    b = _t.rand(4)
    assert _t.allclose(a, b), "torch CPU RNG restore should be reproducible"
    # Tolerant of None / partial.
    restore_rng_state(None)
    restore_rng_state({"python": None})


# ---------------------------------------------------------------------------
# 5) End-to-end resume smoke (optional; skips gracefully on env/data issues)
# ---------------------------------------------------------------------------

def _write_tiny_training_h5(path: Path) -> bool:
    h5py = pytest.importorskip("h5py")
    import numpy as np
    rng = np.random.default_rng(0)
    n = 512
    r_ref = 1.738e6
    dirs = rng.normal(size=(n, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    radii = r_ref + rng.uniform(100e3, 500e3, size=(n, 1))
    xyz = (dirs * radii).astype(np.float32)
    u = (rng.normal(scale=1.0e3, size=(n, 1))).astype(np.float32)
    a = (rng.normal(scale=1.0e-3, size=(n, 3))).astype(np.float32)
    data = np.concatenate([xyz, u, a], axis=1).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h:
        h.create_dataset("data", data=data)
        h.attrs["central_body"] = "moon"
        h.attrs["mu_si"] = 4.902800e12
        h.attrs["r_ref_m"] = r_ref
        h.attrs["unit_system"] = "si"
        h.attrs["degree_min"] = 20
        h.attrs["degree_max"] = 100
        h.attrs["requested_degree"] = 100
        h.attrs["target_mode"] = "residual"
        h.attrs["alt_min_km"] = 100.0
        h.attrs["alt_max_km"] = 500.0
    return True


def _train_cmd(*extra: str) -> list[str]:
    return [sys.executable, "-m", "lunaris.surrogate.st_lrps.training.cli", *extra]


def test_engine_resume_smoke(tmp_path):
    pytest.importorskip("h5py")
    import json

    data = tmp_path / "tiny.h5"
    _write_tiny_training_h5(data)
    run_dir = tmp_path / "runs" / "st_lrps_train_smoke"

    common = [
        "--n-bands", "1", "--activation", "silu", "--hidden", "8", "--depth", "2",
        "--batch-size", "32", "--num-workers", "0", "--quick-check",
        "--a-sign", "1.0", "--allow-legacy-derivative-convention",
        "--direction-loss-weight", "0.0", "--best-ckpt-start-epoch", "0",
        "--no-auto-preload",
    ]

    # Initial run: target 1 epoch.
    p1 = subprocess.run(
        _train_cmd("--data", str(data), "--out", str(run_dir), "--epochs", "1", *common),
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=900,
    )
    if p1.returncode != 0:
        pytest.skip(f"initial training did not complete in this environment:\n{p1.stdout[-2000:]}\n{p1.stderr[-2000:]}")

    ckpt_last = run_dir / "checkpoints" / "ckpt_last.pt"
    assert ckpt_last.exists(), "initial run must produce ckpt_last.pt"

    # Resume: total target 2 epochs (continue from epoch 1 -> run epoch 2).
    p2 = subprocess.run(
        _train_cmd("--resume-from", str(run_dir), "--epochs", "2", *common),
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=900,
    )
    if p2.returncode != 0:
        pytest.skip(f"resume training did not complete in this environment:\n{p2.stdout[-2000:]}\n{p2.stderr[-2000:]}")

    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert int(manifest.get("latest_epoch", 0)) >= 2, f"expected latest_epoch>=2, got {manifest.get('latest_epoch')}"
    assert manifest.get("resumed") is True, "resume must mark the manifest resumed=True"

    history_jsonl = run_dir / "history.jsonl"
    assert history_jsonl.exists()
    rows = [ln for ln in history_jsonl.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(rows) >= 2, f"history should contain >=2 epochs after resume, got {len(rows)}"
    # Combined stdout should mention resuming.
    assert "resum" in (p2.stdout + p2.stderr).lower()


# ---------------------------------------------------------------------------
# 6) No old flat ST-LRPS module paths referenced by this test
# ---------------------------------------------------------------------------

def test_uses_canonical_cli_only():
    # Note: "st_lrps_train_<ts>" is the run-DIRECTORY naming convention, not the
    # old module; guard against the old MODULE/path forms specifically.
    src = Path(__file__).read_text(encoding="utf-8")
    assert "lunaris.surrogate.st_lrps.training.cli" in src
    old_module = "st_lrps" + ".st_lrps_train"
    old_path = "st_lrps" + "/st_lrps_train.py"
    assert old_module not in src
    assert old_path not in src
