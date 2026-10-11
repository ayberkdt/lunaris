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

from lunaris.common.constants import MU_MOON

torch = pytest.importorskip("torch")

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Helpers — build a minimal but schema-valid run with real checkpoints
# (no training; just model/optimizer/scaler construction + save).
# ---------------------------------------------------------------------------

def _make_min_run(run_dir: Path):
    from lunaris.surrogate.st_lrps.artifacts.manager import (
        atomic_write_json,
        build_checkpoint_payload,
        build_resolved_config,
        capture_rng_state,
        ensure_run_layout,
        save_checkpoint,
    )
    from lunaris.surrogate.st_lrps.networks.models import (
        build_model_from_config,
        compute_architecture_signature,
    )
    from lunaris.surrogate.st_lrps.shared.scaling import IsometricScaleParams, ScalerPack
    from lunaris.surrogate.st_lrps.training.config import TrainConfig
    from lunaris.surrogate.st_lrps.training.losses import GradNormWeights

    layout = ensure_run_layout(run_dir)
    cfg = TrainConfig(
        data="d.h5", out=str(run_dir),
        activation="silu", hidden=8, depth=2,
        n_bands=1, use_residual_blocks=False, use_fourier=False,
        use_physical_radial_decay_encoding=False,
        degree_min=20, degree_max=100,
        x_scale_m=2.0e6, resolved_r_ref_m=1.738e6,
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
        "mu_si": float(MU_MOON), "r_ref_m": 1.738e6, "degree_min": 20, "degree_max": 100,
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

    from lunaris.surrogate.st_lrps.data.dataset_contract import DatasetContract

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
        DatasetContract(
            dataset_id="tiny-resume-smoke",
            n_samples=n,
            target_mode="residual",
            baseline_kind="spherical_harmonics",
            degree_min=20,
            degree_max=100,
            mu_si=4.902800e12,
            r_ref_m=r_ref,
            altitude_min_km=100.0,
            altitude_max_km=500.0,
            source_gravity_model="toy",
            source_gravity_file_path="toy.gfc",
            source_gravity_file_sha256="a" * 64,
            spherical_harmonic_convention="4pi_geodesy_no_condon_shortley_v1",
            gravity_label_engine_version="lunaris_sh_v2",
            dataset_layout={"dataset_name": "data", "shape": [n, 7]},
        ).write_hdf5_attrs(h)
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
        "--a-sign", "1.0",
        "--direction-loss-weight", "0.0", "--best-ckpt-start-epoch", "0",
        "--no-auto-preload", "--save-epoch-snapshots",
    ]

    # Initial run: target 1 epoch. A non-zero exit is a failure, not a skip:
    # the skip hid two resume bugs (scaler provenance recorded without the
    # dataset hash, and training_config_hash listed as a critical field).
    p1 = subprocess.run(
        _train_cmd("--data", str(data), "--out", str(run_dir), "--epochs", "1", *common),
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=900, check=False,
    )
    assert p1.returncode == 0, f"initial training failed:\n{p1.stdout[-2000:]}\n{p1.stderr[-2000:]}"

    ckpt_last = run_dir / "checkpoints" / "ckpt_last.pt"
    assert ckpt_last.exists(), "initial run must produce ckpt_last.pt"
    scaler_prov = json.loads((run_dir / "scaler.json").read_text(encoding="utf-8"))["provenance"]
    assert scaler_prov.get("dataset_content_sha256"), "a new scaler must record the dataset hash it is checked against"

    # Resume: total target 2 epochs (continue from epoch 1 -> run epoch 2).
    p2 = subprocess.run(
        _train_cmd("--resume-from", str(run_dir), "--epochs", "2", *common),
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=900, check=False,
    )
    assert p2.returncode == 0, f"resume training failed:\n{p2.stdout[-2000:]}\n{p2.stderr[-2000:]}"

    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert int(manifest.get("latest_epoch", 0)) >= 2, f"expected latest_epoch>=2, got {manifest.get('latest_epoch')}"
    assert manifest.get("resumed") is True, "resume must mark the manifest resumed=True"

    history_jsonl = run_dir / "history.jsonl"
    assert history_jsonl.exists()
    rows = [ln for ln in history_jsonl.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(rows) >= 2, f"history should contain >=2 epochs after resume, got {len(rows)}"
    # Combined stdout should mention resuming.
    assert "resum" in (p2.stdout + p2.stderr).lower()

    # Resume from the epoch-1 snapshot (older than ckpt_last): epoch 2 is
    # re-trained and its stale history row must be replaced, not duplicated.
    snapshot = run_dir / "checkpoints" / "ckpt_epoch_000001.pt"
    assert snapshot.exists(), "--save-epoch-snapshots must write ckpt_epoch_000001.pt"
    p3 = subprocess.run(
        _train_cmd("--resume-from", str(snapshot), "--epochs", "2", *common),
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=900, check=False,
    )
    assert p3.returncode == 0, f"snapshot resume failed:\n{p3.stdout[-2000:]}\n{p3.stderr[-2000:]}"
    epochs = [
        json.loads(ln)["epoch"]
        for ln in history_jsonl.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    assert epochs == [0, 1], f"history epochs must be unique and ascending, got {epochs}"


def _tiny_cli(data: Path, run_dir: Path, *extra: str) -> list[str]:
    return _train_cmd(
        "--data", str(data), "--out", str(run_dir), "--epochs", "1",
        "--n-bands", "1", "--activation", "silu", "--hidden", "8", "--depth", "2",
        "--num-workers", "0", "--quick-check", "--a-sign", "1.0",
        "--direction-loss-weight", "0.0", "--best-ckpt-start-epoch", "0",
        "--no-auto-preload", *extra,
    )


def test_engine_rejects_train_split_smaller_than_one_batch(tmp_path):
    # The train loader drops its partial last batch: with fewer train rows than
    # one batch every epoch ran zero batches and "finished" with loss 0.
    pytest.importorskip("h5py")
    import json

    data = tmp_path / "tiny.h5"
    _write_tiny_training_h5(data)  # 512 rows -> train split below the batch size
    run_dir = tmp_path / "runs" / "st_lrps_train_no_batch"
    p = subprocess.run(
        _tiny_cli(data, run_dir, "--batch-size", "4096"),
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=900, check=False,
    )
    assert p.returncode != 0, f"training must fail closed:\n{p.stdout[-2000:]}"
    assert "yields no batch" in (p.stdout + p.stderr)
    manifest_path = run_dir / "run_manifest.json"
    if manifest_path.exists():
        assert json.loads(manifest_path.read_text(encoding="utf-8")).get("status") != "completed"


def test_each_run_in_one_process_gets_its_own_train_log(tmp_path):
    # logging.basicConfig is a no-op once the root logger has handlers, so a
    # second run in the same process (notebook, sweep, test session) wrote no
    # train.log. Each run now attaches and removes its own file handler.
    import logging

    from lunaris.surrogate.st_lrps.training.engine import _attach_run_log, _detach_run_logs

    root = logging.getLogger()
    sentinel = logging.NullHandler()
    root.addHandler(sentinel)            # logging already configured by the host
    log = logging.getLogger("lunaris.surrogate.st_lrps.training.engine")
    try:
        for name in ("run1", "run2"):
            path = tmp_path / f"{name}.log"
            _attach_run_log(path)
            try:
                log.info("hello from %s", name)
            finally:
                _detach_run_logs()
            assert f"hello from {name}" in path.read_text(encoding="utf-8")
        assert "run2" not in (tmp_path / "run1.log").read_text(encoding="utf-8")
        assert sentinel in root.handlers
        assert not any(getattr(h, "_lunaris_run_log", False) for h in root.handlers)
    finally:
        root.removeHandler(sentinel)


def test_nan_validation_loss_writes_a_failure_manifest(tmp_path):
    # Only the training loss was checked: a NaN validation loss reached
    # compute_checkpoint_score() and raised, leaving no failure manifest and
    # the run manifest at status "running".
    h5py = pytest.importorskip("h5py")
    import json

    train = tmp_path / "train.h5"
    val = tmp_path / "val.h5"
    _write_tiny_training_h5(train)
    _write_tiny_training_h5(val)
    with h5py.File(val, "r+") as h:
        h["data"][:, 3] = float("nan")
    run_dir = tmp_path / "runs" / "st_lrps_train_nan_val"
    p = subprocess.run(
        _train_cmd(
            "--train-data", str(train), "--val-data", str(val), "--out", str(run_dir), "--epochs", "1",
            "--n-bands", "1", "--activation", "silu", "--hidden", "8", "--depth", "2",
            "--batch-size", "32", "--num-workers", "0", "--quick-check", "--a-sign", "1.0",
            "--direction-loss-weight", "0.0", "--best-ckpt-start-epoch", "0",
            "--no-auto-preload", "--allow-dataset-validation-fail", "--skip-preflight",
        ),
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=900, check=False,
    )
    out = p.stdout[-3000:] + p.stderr[-3000:]
    failure = run_dir / "failure_manifest.json"
    assert failure.exists(), f"no failure manifest:\n{out}"
    assert json.loads(failure.read_text(encoding="utf-8"))["reason"] == "nan_val_loss"
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest.get("status") == "failed", out


def test_independent_val_file_is_validated_before_training(tmp_path):
    h5py = pytest.importorskip("h5py")
    from lunaris.surrogate.st_lrps.artifacts.manager import ensure_run_layout
    from lunaris.surrogate.st_lrps.training.config import TrainConfig
    from lunaris.surrogate.st_lrps.training.engine import _load_dataset_context

    train = tmp_path / "train.h5"
    val = tmp_path / "val.h5"
    _write_tiny_training_h5(train)
    _write_tiny_training_h5(val)
    layout = ensure_run_layout(tmp_path / "run")
    cfg = TrainConfig(data=str(train), train_data=str(train), val_data=str(val), out=str(layout.run_dir))

    assert _load_dataset_context(cfg, layout=layout).independent_val  # a clean val file passes

    with h5py.File(val, "r+") as h:
        h["data"][3, 3] = float("nan")
    with pytest.raises(ValueError, match="Validation dataset failed validation"):
        _load_dataset_context(cfg, layout=layout)


def test_independent_split_manifest_records_file_hashes(tmp_path):
    h5py = pytest.importorskip("h5py")
    from lunaris.surrogate.st_lrps.artifacts.manager import ensure_run_layout
    from lunaris.surrogate.st_lrps.data.dataset_contract import content_sha256_for_hdf5_dataset
    from lunaris.surrogate.st_lrps.training.config import TrainConfig
    from lunaris.surrogate.st_lrps.training.engine import (
        _load_dataset_context,
        _resolve_data_splits,
    )

    train = tmp_path / "train.h5"
    val = tmp_path / "val.h5"
    _write_tiny_training_h5(train)
    _write_tiny_training_h5(val)
    with h5py.File(val, "r+") as h:
        h["data"][0, 3] = h["data"][0, 3] + 1.0   # distinct content, still valid
    layout = ensure_run_layout(tmp_path / "run")
    cfg = TrainConfig(data=str(train), train_data=str(train), val_data=str(val), out=str(layout.run_dir))
    ctx = _load_dataset_context(cfg, layout=layout)
    split = _resolve_data_splits(
        cfg, meta=ctx.meta, N=ctx.N, independent_val=True,
        N_train_file=ctx.N_train_file, N_val_file=ctx.N_val_file,
        primary_path=ctx.primary_path, val_data_path=ctx.val_data_path, dset_name=ctx.dset_name,
        dataset_contract_obj=ctx.dataset_contract_obj, validation_report=ctx.validation_report,
        layout=layout,
    )
    manifest = split.split_manifest
    assert manifest["train_file_sha256"] == content_sha256_for_hdf5_dataset(train, dataset_name=ctx.dset_name)
    assert manifest["val_file_sha256"] == content_sha256_for_hdf5_dataset(val, dataset_name=ctx.dset_name)
    assert manifest["train_file_sha256"] != manifest["val_file_sha256"]


# ---------------------------------------------------------------------------
# 5b) Resume compatibility gate with realistic payloads
# ---------------------------------------------------------------------------

def _rebuilt_config(run_dir: Path, **cfg_overrides):
    """The config a resume rebuilds: same model/data, different epochs/resume source."""
    from lunaris.surrogate.st_lrps.artifacts.manager import build_resolved_config
    from lunaris.surrogate.st_lrps.networks.models import (
        build_model_from_config,
        compute_architecture_signature,
    )
    from lunaris.surrogate.st_lrps.shared.scaling import IsometricScaleParams, ScalerPack
    from lunaris.surrogate.st_lrps.training.config import TrainConfig

    params = dict(
        data="d.h5", out=str(run_dir),
        activation="silu", hidden=8, depth=2,
        n_bands=1, use_residual_blocks=False, use_fourier=False,
        use_physical_radial_decay_encoding=False,
        degree_min=20, degree_max=100,
        x_scale_m=2.0e6, resolved_r_ref_m=1.738e6,
        epochs=50, resume_from=str(run_dir),
    )
    params.update(cfg_overrides)
    cfg = TrainConfig(**params)
    cfg.w0_bands = None
    model = build_model_from_config(cfg, in_dim=3, device=torch.device("cpu"), dtype=torch.float32)
    scaler = ScalerPack(
        x=IsometricScaleParams(mean=[0.0, 0.0, 0.0], scale=2.0e6),
        u=IsometricScaleParams(mean=[0.0], scale=1.0e4),
        a=IsometricScaleParams(mean=[0.0, 0.0, 0.0], scale=1.0e-3),
    )
    dataset_meta = {
        "mu_si": float(MU_MOON), "r_ref_m": 1.738e6, "degree_min": 20, "degree_max": 100,
        "target_mode": "residual", "unit_system": "si", "central_body": "moon",
    }
    return build_resolved_config(cfg, dataset_meta, model, scaler, compute_architecture_signature(cfg))


def test_resume_gate_accepts_its_own_checkpoint_despite_new_epoch_target(tmp_path):
    from lunaris.surrogate.st_lrps.artifacts.manager import verify_critical_config_fields_match

    run_dir = tmp_path / "runs" / "gate1"
    _, payload = _make_min_run(run_dir)
    rebuilt = _rebuilt_config(run_dir)
    # The whole-config identity differs (epochs, resume_from) ...
    assert rebuilt["training_config_hash"] != payload["config"]["training_config_hash"]
    # ... but nothing the weights depend on does, so the resume is compatible.
    verify_critical_config_fields_match(rebuilt, payload["config"])


def test_resume_gate_treats_absent_envelope_as_off_but_keeps_real_mismatches(tmp_path):
    from lunaris.surrogate.st_lrps.artifacts.manager import verify_critical_config_fields_match

    run_dir = tmp_path / "runs" / "gate2"
    _, payload = _make_min_run(run_dir)
    pre_envelope = dict(payload["config"])
    pre_envelope.pop("output_envelope_power")      # a checkpoint from before the envelope

    # Rebuilt config with the envelope off (default 0.0): compatible.
    verify_critical_config_fields_match(_rebuilt_config(run_dir), pre_envelope)
    verify_critical_config_fields_match(pre_envelope, _rebuilt_config(run_dir))

    # Envelope switched on against a pre-envelope checkpoint: a real change.
    with pytest.raises(RuntimeError, match="output_envelope_power"):
        verify_critical_config_fields_match(_rebuilt_config(run_dir, output_envelope_power=21.0), pre_envelope)

    # An architecture change is still refused.
    with pytest.raises(RuntimeError, match="hidden"):
        verify_critical_config_fields_match(_rebuilt_config(run_dir, hidden=16), payload["config"])


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
