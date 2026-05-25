import json
from dataclasses import asdict
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from st_lrps.st_lrps_artifacts import (
    CHECKPOINT_SCHEMA_VERSION,
    CRITICAL_CONFIG_FIELDS,
    atomic_write_json,
    build_checkpoint_payload,
    build_resolved_config,
    compute_file_sha256,
    compute_payload_sha256,
    default_eval_output_dir,
    ensure_run_layout,
    load_best_or_last,
    load_checkpoint,
    load_scaler_for_run,
    normalize_legacy_checkpoint,
    read_run_manifest,
    save_checkpoint,
    update_run_manifest,
    validate_checkpoint_schema,
    verify_critical_config_fields_match,
    write_run_manifest,
    write_scaler_json,
)
from st_lrps.st_lrps_evaluate import evaluate, predict_residual_u_a
from st_lrps.st_lrps_force_model import load_surrogate_force_model
from st_lrps.st_lrps_models import (
    build_model_from_config,
    compute_architecture_signature,
)
from st_lrps.st_lrps_scaling import IsometricScaleParams, ScalerPack


def _make_scaler() -> ScalerPack:
    return ScalerPack(
        x=IsometricScaleParams(mean=[0.0, 0.0, 0.0], scale=1.0),
        u=IsometricScaleParams(mean=[0.0], scale=1.0),
        a=IsometricScaleParams(mean=[0.0, 0.0, 0.0], scale=1.0),
        provenance={"fit_rows": 32, "target_mode": "residual"},
    )


def _write_dataset(path: Path, n_rows: int = 32) -> None:
    rng = np.random.default_rng(7)
    r_ref_m = 1_737_400.0
    xyz = rng.normal(size=(n_rows, 3)).astype(np.float64)
    xyz /= np.linalg.norm(xyz, axis=1, keepdims=True)
    xyz *= (r_ref_m + rng.uniform(50_000.0, 250_000.0, size=(n_rows, 1)))
    du = rng.normal(scale=10.0, size=(n_rows, 1))
    da = rng.normal(scale=1e-6, size=(n_rows, 3))
    data = np.concatenate([xyz, du, da], axis=1).astype(np.float32)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("data", data=data)
        handle.attrs["unit_system"] = "si"
        handle.attrs["mu_si"] = 4.902800066e12
        handle.attrs["r_ref_m"] = r_ref_m
        handle.attrs["central_body"] = "moon"
        handle.attrs["target_mode"] = "residual"
        handle.attrs["degree_min"] = 2
        handle.attrs["degree_max"] = 4
        handle.attrs["a_sign_convention"] = "1.0"
        handle.attrs["derivative_convention_version"] = "test_v1"
        handle.attrs["columns"] = "[x,y,z,dU,dax,day,daz]"


def _base_cfg(data_path: Path) -> dict:
    return {
        "data": str(data_path),
        "train_data_path": str(data_path),
        "val_data_path": str(data_path),
        "test_data_path": str(data_path),
        "ood_data_path": None,
        "dataset_name": "data",
        "central_body": "moon",
        "target_mode": "residual",
        "degree_min": 2,
        "degree_max": 4,
        "unit_system": "si",
        "resolved_mu_si": 4.902800066e12,
        "resolved_r_ref_m": 1_737_400.0,
        "resolved_a_sign": 1.0,
        "activation": "sine",
        "hidden": 16,
        "depth": 2,
        "dropout": 0.0,
        "use_fourier": False,
        "fourier_append_raw": True,
        "fourier_n_features": 0,
        "fourier_sigma": 0.0,
        "fourier_seed": 0,
        "use_sh_encoding": False,
        "sh_encoding_degree": 4,
        "sh_append_raw": True,
        "use_radial_separation": False,
        "radial_append_raw": False,
        "use_residual_blocks": False,
        "n_bands": 2,
        "w0_bands": [30.0, 6.0],
        "w0_first": 30.0,
        "w0_hidden": 30.0,
        "best_metric": "val_total_loss",
        "run_name": "artifact_test_run",
    }


def _create_canonical_run(tmp_path: Path) -> dict:
    torch.manual_seed(0)
    run_dir = tmp_path / "run"
    dataset_path = tmp_path / "dataset.h5"
    _write_dataset(dataset_path)
    layout = ensure_run_layout(run_dir)
    scaler = _make_scaler()
    cfg = _base_cfg(dataset_path)
    dataset_meta = {
        "mu_si": cfg["resolved_mu_si"],
        "r_ref_m": cfg["resolved_r_ref_m"],
        "central_body": cfg["central_body"],
        "target_mode": cfg["target_mode"],
        "degree_min": cfg["degree_min"],
        "degree_max": cfg["degree_max"],
        "unit_system": cfg["unit_system"],
        "alt_min_km": 50.0,
        "alt_max_km": 250.0,
        "a_sign_convention": "1.0",
        "derivative_convention_version": "test_v1",
    }
    model = build_model_from_config(cfg)
    seed_cfg = build_resolved_config(cfg, dataset_meta, model, scaler, "pending")
    arch_sig = compute_architecture_signature(seed_cfg)
    resolved_cfg = build_resolved_config(cfg, dataset_meta, model, scaler, arch_sig)
    atomic_write_json(layout.config_json, resolved_cfg)
    scaler_info = write_scaler_json(layout, scaler)
    write_run_manifest(
        layout,
        {
            "run_id": layout.run_dir.name,
            "created_at_utc": "2026-05-21T00:00:00Z",
            "status": "running",
            "config_path": str(layout.config_json),
            "scaler_path": str(layout.scaler_json),
            "best_checkpoint_path": str(layout.ckpt_best),
            "last_checkpoint_path": str(layout.ckpt_last),
            "architecture_signature": arch_sig,
            "w0_bands": resolved_cfg.get("w0_bands"),
            "scaler_hash": scaler_info["scaler_hash"],
            "evaluations": [],
            "warnings": [],
        },
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_stats = {
        "lr": 1e-3,
        "w_u": 1.0,
        "w_a": 1.0,
        "gradnorm_status": "fixed",
        "accel_factor": 1.0,
        "lambda_dir_eff": 0.1,
    }
    best_val_stats = {
        "loss": 0.25,
        "val_base_loss": 0.20,
        "val_total_loss": 0.25,
        "val_physics_loss": 0.01,
        "loss_dir": 0.02,
        "cossim_mean": 0.95,
        "angular_mean_deg": 4.0,
        "val_checkpoint_score": 0.25,
    }
    last_val_stats = dict(best_val_stats)
    last_val_stats["loss"] = 0.30
    last_val_stats["val_total_loss"] = 0.30
    last_val_stats["val_checkpoint_score"] = 0.30
    best_payload = build_checkpoint_payload(
        kind="best",
        epoch=1,
        model=model,
        optimizer=optimizer,
        scheduler=None,
        cfg=resolved_cfg,
        scaler=scaler,
        train_stats=train_stats,
        val_stats=best_val_stats,
        dataset_meta=dataset_meta,
        architecture_signature=arch_sig,
        global_step=10,
    )
    last_payload = build_checkpoint_payload(
        kind="last",
        epoch=2,
        model=model,
        optimizer=optimizer,
        scheduler=None,
        cfg=resolved_cfg,
        scaler=scaler,
        train_stats=train_stats,
        val_stats=last_val_stats,
        dataset_meta=dataset_meta,
        architecture_signature=arch_sig,
        global_step=20,
    )
    save_checkpoint(layout, kind="best", payload=best_payload, epoch=1)
    save_checkpoint(layout, kind="last", payload=last_payload, epoch=2, write_epoch_snapshot=True)
    update_run_manifest(
        layout,
        {
            "status": "completed",
            "best_epoch": 2,
            "best_score": 0.25,
            "latest_epoch": 3,
            "checkpoint_hashes": {
                "best": compute_file_sha256(layout.ckpt_best),
                "last": compute_file_sha256(layout.ckpt_last),
            },
        },
    )
    return {
        "run_dir": run_dir,
        "layout": layout,
        "dataset_path": dataset_path,
        "cfg": resolved_cfg,
        "dataset_meta": dataset_meta,
        "scaler": scaler,
        "scaler_info": scaler_info,
        "best_payload": best_payload,
        "last_payload": last_payload,
        "architecture_signature": arch_sig,
    }


@pytest.fixture
def canonical_run(tmp_path: Path) -> dict:
    return _create_canonical_run(tmp_path)


def test_run_layout_paths_are_stable(tmp_path: Path) -> None:
    layout = ensure_run_layout(tmp_path / "stable_run")
    assert layout.config_json == layout.run_dir / "config.json"
    assert layout.scaler_json == layout.run_dir / "scaler.json"
    assert layout.run_manifest_json == layout.run_dir / "run_manifest.json"
    assert layout.checkpoints_dir == layout.run_dir / "checkpoints"
    assert layout.ckpt_best == layout.checkpoints_dir / "ckpt_best.pt"
    assert layout.ckpt_last == layout.checkpoints_dir / "ckpt_last.pt"
    assert layout.history_csv == layout.run_dir / "history.csv"
    assert layout.history_jsonl == layout.run_dir / "history.jsonl"
    assert layout.evals_dir == layout.run_dir / "evals"
    assert layout.provenance_dir == layout.run_dir / "provenance"


def test_checkpoint_payload_has_canonical_schema(canonical_run: dict) -> None:
    payload = canonical_run["best_payload"]
    validated = validate_checkpoint_schema(payload, strict=True)
    assert validated["schema_version"] == CHECKPOINT_SCHEMA_VERSION
    assert validated["kind"] == "best"
    assert validated["model_state_dict"]
    assert validated["config"]["architecture_signature"] == canonical_run["architecture_signature"]
    assert validated["architecture"]["signature"] == canonical_run["architecture_signature"]
    assert validated["scaler_hash"] == compute_payload_sha256(validated["scaler"])


def test_ckpt_best_and_last_share_same_top_level_keys(canonical_run: dict) -> None:
    layout = canonical_run["layout"]
    best = load_checkpoint(layout.ckpt_best, torch.device("cpu"))
    last = load_checkpoint(layout.ckpt_last, torch.device("cpu"))
    assert set(best.keys()) == set(last.keys())


def test_legacy_checkpoint_model_key_normalizes_to_model_state_dict(canonical_run: dict) -> None:
    legacy = {
        "model": canonical_run["best_payload"]["model_state_dict"],
        "config": canonical_run["cfg"],
        "scaler": canonical_run["best_payload"]["scaler"],
        "epoch": 0,
        "best_val": 0.5,
    }
    normalized = normalize_legacy_checkpoint(legacy)
    assert "model_state_dict" in normalized
    assert normalized["model_state_dict"] == legacy["model"]
    assert normalized["model"] == legacy["model"]


def test_config_json_and_checkpoint_config_critical_fields_match(canonical_run: dict) -> None:
    layout = canonical_run["layout"]
    config_payload = json.loads(layout.config_json.read_text(encoding="utf-8"))
    ckpt = load_checkpoint(layout.ckpt_best, torch.device("cpu"))
    verify_critical_config_fields_match(config_payload, ckpt["config"])
    for field in CRITICAL_CONFIG_FIELDS:
        if field in config_payload or field in ckpt["config"]:
            assert config_payload.get(field) == ckpt["config"].get(field)


def test_evaluator_and_force_model_use_same_artifact_loader(canonical_run: dict) -> None:
    run_dir = canonical_run["run_dir"]
    from st_lrps.st_lrps_artifacts import reload_model_from_run_dir

    model, scaler, cfg, report = reload_model_from_run_dir(run_dir, torch.device("cpu"))
    force_model = load_surrogate_force_model(run_dir, device="cpu")
    x_np = np.array([[1_760_000.0, 2_000.0, -3_000.0]], dtype=np.float32)
    x_t = torch.from_numpy(x_np)
    u_ref, a_ref = predict_residual_u_a(model, scaler, x_t, a_sign=float(cfg["resolved_a_sign"]))
    u_force = force_model.predict_residual_potential(x_np)
    a_force = force_model.predict_residual_accel(x_np)
    assert report["checkpoint_path"] == force_model.checkpoint_path
    assert report["architecture_signature"] == force_model.architecture_signature
    assert force_model.run_manifest["architecture_signature"] == report["architecture_signature"]
    assert np.allclose(u_ref.detach().cpu().numpy().reshape(-1), np.asarray(u_force).reshape(-1), atol=1e-6)
    assert np.allclose(a_ref.detach().cpu().numpy(), np.asarray(a_force), atol=1e-6)


def test_test_st_lrps_uses_canonical_reload_path() -> None:
    source = (Path(__file__).resolve().parent / "test_st_lrps.py").read_text(encoding="utf-8")
    assert "reload_model_from_run_dir(" in source
    assert "load_best_or_last(" in source
    assert "predict_residual_u_a(" in source
    assert "strict=False" not in source
    assert "_build_model_from_ckpt" not in source
    assert "_load_scaler_from_ckpt" not in source


def test_checkpoint_rejects_wrong_w0_bands_after_reload(canonical_run: dict) -> None:
    layout = canonical_run["layout"]
    broken = json.loads(layout.config_json.read_text(encoding="utf-8"))
    broken["w0_bands"] = [99.0, 33.0]
    atomic_write_json(layout.config_json, broken)
    from st_lrps.st_lrps_artifacts import reload_model_from_run_dir

    with pytest.raises(RuntimeError, match="architecture-critical fields|w0_bands"):
        reload_model_from_run_dir(layout.run_dir, torch.device("cpu"))


def test_scaler_json_and_checkpoint_scaler_match(canonical_run: dict) -> None:
    layout = canonical_run["layout"]
    _, ckpt = load_best_or_last(layout, prefer="best", device=torch.device("cpu"))
    scaler, report = load_scaler_for_run(layout, ckpt, device=torch.device("cpu"))
    assert isinstance(scaler, ScalerPack)
    assert report["scaler_hash"] == canonical_run["scaler_info"]["scaler_hash"]
    assert compute_payload_sha256(ckpt["scaler"]) == canonical_run["scaler_info"]["scaler_hash"]


def test_run_manifest_updates_after_best_checkpoint(canonical_run: dict) -> None:
    layout = canonical_run["layout"]
    manifest = read_run_manifest(layout)
    assert manifest["best_epoch"] == 2
    assert manifest["best_score"] == pytest.approx(0.25)
    assert manifest["architecture_signature"] == canonical_run["architecture_signature"]
    assert manifest["w0_bands"] == canonical_run["cfg"]["w0_bands"]
    assert manifest["scaler_hash"] == canonical_run["scaler_info"]["scaler_hash"]
    assert manifest["checkpoint_hashes"]["best"] == compute_file_sha256(layout.ckpt_best)
    assert manifest["checkpoint_hashes"]["last"] == compute_file_sha256(layout.ckpt_last)


def test_eval_manifest_written_to_evals_dir(canonical_run: dict) -> None:
    layout = canonical_run["layout"]
    data_path = canonical_run["dataset_path"]
    out_dir = default_eval_output_dir(layout, data_path, timestamp="20260521_120000")
    evaluate(
        model_dir=layout.run_dir,
        data_path=data_path,
        out_dir=out_dir,
        device=torch.device("cpu"),
        batch_size=8,
        a_sign=1.0,
        r_ref_m=canonical_run["cfg"]["resolved_r_ref_m"],
        alt_bin_km=50.0,
        streaming=True,
    )
    assert out_dir.parent == layout.evals_dir
    assert (out_dir / "eval_manifest.json").exists()
    assert (out_dir / "evaluate_metrics.json").exists()
    assert (out_dir / "evaluate_summary.txt").exists()
    manifest = json.loads((out_dir / "eval_manifest.json").read_text(encoding="utf-8"))
    run_manifest = read_run_manifest(layout)
    assert manifest["metrics_path"].endswith("evaluate_metrics.json")
    assert run_manifest["evaluations"][-1]["out_dir"] == str(out_dir)


def test_no_silent_strict_false_load_for_architecture_mismatch(canonical_run: dict) -> None:
    layout = canonical_run["layout"]
    broken = json.loads(layout.config_json.read_text(encoding="utf-8"))
    broken["hidden"] = int(broken["hidden"]) + 4
    atomic_write_json(layout.config_json, broken)
    from st_lrps.st_lrps_artifacts import reload_model_from_run_dir

    with pytest.raises(RuntimeError, match="architecture-critical fields|disagree"):
        reload_model_from_run_dir(layout.run_dir, torch.device("cpu"))
