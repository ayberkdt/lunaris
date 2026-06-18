"""Minimal direct-force ST-LRPS trainer.

This module intentionally does not reuse the Sobolev potential training loop.
It trains a 3-output student that predicts scaled residual acceleration directly
from Moon-fixed Cartesian positions:

    NN(scale_x(r_fixed_m)) -> scale_a(Delta a_fixed_m_s2)

The saved artifact uses ``runtime_model_kind='force_direct'`` and is loadable by
``load_surrogate_force_model``. It does not predict scalar potential.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lunaris.surrogate.st_lrps.artifacts.manager import (
    build_checkpoint_payload,
    build_resolved_config,
    capture_environment_snapshot,
    ensure_run_layout,
    write_command_txt,
    write_run_manifest,
)
from lunaris.surrogate.st_lrps.data.dataset_contract import (
    DATASET_CONTRACT_ATTR,
    GRAVITY_LABEL_ENGINE_VERSION,
    REQUIRED_SH_PHASE_CONVENTION,
)
from lunaris.surrogate.st_lrps.data.dataset_parameters import MU_MOON_SI, R_MOON_SI
from lunaris.surrogate.st_lrps.data.datasets import DatasetMeta
from lunaris.surrogate.st_lrps.data.splits import build_split_manifest, write_split_manifest
from lunaris.surrogate.st_lrps.networks.models import (
    build_model_from_config,
    compute_architecture_signature,
)
from lunaris.surrogate.st_lrps.shared.scaling import IsometricScaleParams, ScalerPack
from lunaris.surrogate.st_lrps.training.engine import set_seed


def _json_text(payload: Mapping[str, Any]) -> str:
    return json.dumps(dict(payload), indent=2, sort_keys=True, ensure_ascii=True, default=str) + "\n"


def _read_dataset(path: Path, dataset_name: str, max_samples: int | None, seed: int) -> tuple[np.ndarray, DatasetMeta]:
    import h5py  # type: ignore

    meta = DatasetMeta.from_h5(path)
    with h5py.File(path, "r") as handle:
        if dataset_name in handle:
            name = dataset_name
        else:
            name = next((key for key in handle.keys() if hasattr(handle[key], "shape")), None)
            if name is None:
                raise ValueError(f"No HDF5 dataset found in {path}")
        ds = handle[name]
        if len(ds.shape) != 2 or int(ds.shape[1]) != 7:
            raise ValueError(f"Direct-force training expects rows [x,y,z,U,ax,ay,az], got shape {ds.shape}")
        n_total = int(ds.shape[0])
        if max_samples is not None and int(max_samples) < n_total:
            rng = np.random.default_rng(int(seed))
            idx = np.sort(rng.choice(n_total, size=int(max_samples), replace=False).astype(np.int64))
            arr = np.asarray(ds[idx, :], dtype=np.float64)
        else:
            arr = np.asarray(ds[:, :], dtype=np.float64)
    if meta.unit_system == "canonical":
        x, u, a = meta.convert_xyz_U_a_to_si(arr[:, 0:3], arr[:, 3:4], arr[:, 4:7])
        arr = np.concatenate([x, u, a], axis=1)
    if not np.all(np.isfinite(arr)):
        raise ValueError("Dataset contains non-finite values.")
    return arr, meta


def _as_contract_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except UnicodeDecodeError:
            return None
    text = str(value).strip()
    return text or None


def _parse_contract_payload(raw: Any) -> dict[str, Any] | None:
    text = _as_contract_text(raw)
    if not text:
        return None
    try:
        payload = json.loads(text)
    except Exception:
        return None
    return dict(payload) if isinstance(payload, Mapping) else None


def _iter_contract_payloads(
    meta: DatasetMeta,
    *,
    data_path: Path,
    invalid_sources: list[str] | None = None,
) -> list[tuple[str, dict[str, Any]]]:
    payloads: list[tuple[str, dict[str, Any]]] = []
    for key in (DATASET_CONTRACT_ATTR, "contract_json", "dataset_contract"):
        source = f"attrs.{key}"
        raw = meta.raw_attrs.get(key)
        text = _as_contract_text(raw)
        if not text:
            continue
        payload = _parse_contract_payload(text)
        if payload is not None:
            payloads.append((source, payload))
        elif invalid_sources is not None:
            invalid_sources.append(source)

    try:
        import h5py  # type: ignore

        with h5py.File(data_path, "r") as handle:
            if "metadata" in handle and "contract_json" in handle["metadata"]:
                source = "metadata.contract_json"
                raw = handle["metadata"]["contract_json"][()]
                text = _as_contract_text(raw)
                payload = _parse_contract_payload(text)
                if payload is not None:
                    payloads.append((source, payload))
                elif text and invalid_sources is not None:
                    invalid_sources.append(source)
    except Exception:
        pass
    return payloads


def _declared_gravity_label_values(
    meta: DatasetMeta,
    *,
    data_path: Path,
    key: str,
    payloads: list[tuple[str, dict[str, Any]]] | None = None,
) -> list[tuple[str, str]]:
    values: list[tuple[str, str]] = []
    top_level = _as_contract_text(meta.raw_attrs.get(key))
    if top_level is not None:
        values.append((f"attrs.{key}", top_level))
    if payloads is None:
        payloads = _iter_contract_payloads(meta, data_path=data_path)
    for source, payload in payloads:
        value = _as_contract_text(payload.get(key))
        if value is not None:
            values.append((f"{source}.{key}", value))
    return values


def _validate_input_gravity_label_contract(meta: DatasetMeta, *, data_path: Path) -> None:
    expected = {
        "spherical_harmonic_convention": REQUIRED_SH_PHASE_CONVENTION,
        "gravity_label_engine_version": GRAVITY_LABEL_ENGINE_VERSION,
    }
    errors: list[str] = []
    invalid_sources: list[str] = []
    payloads = _iter_contract_payloads(meta, data_path=data_path, invalid_sources=invalid_sources)
    for source in invalid_sources:
        errors.append(f"{source} could not be parsed as JSON")
    for key, expected_value in expected.items():
        values = _declared_gravity_label_values(meta, data_path=data_path, key=key, payloads=payloads)
        if not values:
            errors.append(f"{key} is missing")
            continue
        mismatches = [(source, value) for source, value in values if value != expected_value]
        if mismatches:
            source, value = mismatches[0]
            errors.append(f"{source}={value!r}, expected {expected_value!r}")
    if errors:
        details = "; ".join(errors)
        raise ValueError(
            f"Dataset {str(data_path)!r} has an unsafe gravity label contract: {details}. "
            "Regenerate the dataset with the current Lunaris spherical-harmonic label engine before training."
        )


def _point_mass_accel(x_m: np.ndarray, mu_si: float) -> np.ndarray:
    r = np.linalg.norm(x_m, axis=1, keepdims=True)
    r = np.maximum(r, 1.0)
    return -float(mu_si) * x_m / (r ** 3)


def _target_accel(arr: np.ndarray, *, target_mode: str, baseline_kind: str, mu_si: float) -> np.ndarray:
    x = arr[:, 0:3]
    a = arr[:, 4:7]
    mode = str(target_mode).strip().lower()
    baseline = str(baseline_kind).strip().lower()
    if mode == "residual" or baseline == "none":
        return a
    if baseline == "point_mass":
        return a - _point_mass_accel(x, mu_si)
    raise NotImplementedError(
        "Direct-force CLI cannot subtract a spherical-harmonics full-field baseline. "
        "Use residual datasets, or provide a direct residual-acceleration dataset."
    )


def _fit_scaler(arr: np.ndarray, target_a: np.ndarray, meta: DatasetMeta, *, fit_scope: str) -> ScalerPack:
    x = arr[:, 0:3]
    u = arr[:, 3:4]
    x_scale = float(np.max(np.linalg.norm(x, axis=1)))
    a_mean = np.mean(target_a, axis=0)
    a_centered = target_a - a_mean
    a_scale = float(max(np.sqrt(np.mean(np.sum(a_centered * a_centered, axis=1))), 1e-12))
    u_mean = np.mean(u, axis=0)
    u_centered = u - u_mean
    u_scale = float(max(np.sqrt(np.mean(u_centered * u_centered)), 1e-12))
    return ScalerPack(
        x=IsometricScaleParams(mean=[0.0, 0.0, 0.0], scale=max(x_scale, 1e-12)),
        u=IsometricScaleParams(mean=u_mean.reshape(-1).tolist(), scale=u_scale),
        a=IsometricScaleParams(mean=a_mean.reshape(-1).tolist(), scale=a_scale),
        provenance={
            "fit_scope": fit_scope,
            "fit_rows": int(arr.shape[0]),
            "target": "residual_acceleration",
            "runtime_model_kind": "force_direct",
            "alt_min_km": meta.alt_min_km,
            "alt_max_km": meta.alt_max_km,
            "degree_min": meta.degree_min,
            "degree_max": meta.degree_max or meta.requested_degree,
        },
    )


def _dataset_meta_block(
    meta: DatasetMeta,
    *,
    dataset_name: str,
    target_mode: str,
    degree_min: int,
    degree_max: int,
    mu_si: float,
    r_ref_m: float,
    alt_min_km: float | None,
    alt_max_km: float | None,
    n_samples: int,
) -> dict[str, Any]:
    baseline_kind = "spherical_harmonics" if target_mode == "residual" else "point_mass"
    contract = {
        "schema_version": 1,
        "dataset_kind": "st_lrps_direct_force_training",
        "dataset_name": dataset_name,
        "target_mode": target_mode,
        "baseline_kind": baseline_kind,
        "degree_min": int(degree_min),
        "degree_max": int(degree_max),
        "mu_si": float(mu_si),
        "r_ref_m": float(r_ref_m),
        "a_sign": 1.0,
        "altitude_min_km": alt_min_km,
        "altitude_max_km": alt_max_km,
        "n_samples": int(n_samples),
        "coordinate_frame": "moon_fixed_cartesian",
        "units": {"position": "m", "potential": "m^2/s^2", "acceleration": "m/s^2"},
        "derivative_convention_version": meta.derivative_convention_version or "dP_dphi_corrected_v1",
        "spherical_harmonic_convention": REQUIRED_SH_PHASE_CONVENTION,
        "gravity_label_engine_version": GRAVITY_LABEL_ENGINE_VERSION,
    }
    if target_mode == "residual":
        # Residual targets are Δa against the spherical-harmonics baseline; record
        # which model that is so the contract validates without a legacy bypass.
        contract["source_gravity_model"] = f"spherical_harmonics_deg{int(degree_min)}_{int(degree_max)}"
    return {
        "schema_version": 1,
        "dataset_name": dataset_name,
        "target_mode": target_mode,
        "degree_min": int(degree_min),
        "degree_max": int(degree_max),
        "requested_degree": int(degree_max),
        "mu_si": float(mu_si),
        "r_ref_m": float(r_ref_m),
        "central_body": meta.central_body or "moon",
        "unit_system": "si",
        "alt_min_km": alt_min_km,
        "alt_max_km": alt_max_km,
        "a_sign": 1.0,
        "a_sign_convention": "+1",
        "derivative_convention_version": contract["derivative_convention_version"],
        "spherical_harmonic_convention": REQUIRED_SH_PHASE_CONVENTION,
        "gravity_label_engine_version": GRAVITY_LABEL_ENGINE_VERSION,
        "coordinate_frame": "moon_fixed_cartesian",
        "dataset_contract": contract,
    }


def train_force_direct(args: argparse.Namespace) -> Path:
    # Canonical determinism setup (same flags the main engine records) so the
    # artifact carries a real run_provenance.determinism block.
    determinism_flags = set_seed(int(args.seed))

    data_path = Path(args.data).expanduser().resolve()
    arr, meta = _read_dataset(data_path, args.dataset_name, args.max_samples, int(args.seed))
    _validate_input_gravity_label_contract(meta, data_path=data_path)
    mu_si = float(args.mu_si or meta.mu_si or MU_MOON_SI)
    r_ref_m = float(args.r_ref_m or meta.r_ref_m or R_MOON_SI)
    degree_min = int(args.degree_min if args.degree_min is not None else (meta.degree_min if meta.degree_min is not None else 0))
    degree_max = int(
        args.degree_max
        if args.degree_max is not None
        else (meta.degree_max if meta.degree_max is not None else (meta.requested_degree if meta.requested_degree is not None else degree_min + 1))
    )
    target_mode = str(args.target_mode or meta.target_mode or "residual").strip().lower()
    if target_mode not in {"residual", "full"}:
        raise ValueError("--target-mode must be residual or full")
    baseline_kind = "spherical_harmonics" if target_mode == "residual" else "point_mass"
    target_a = _target_accel(arr, target_mode=target_mode, baseline_kind=baseline_kind, mu_si=mu_si)

    rng = np.random.default_rng(int(args.seed))
    n = int(arr.shape[0])
    perm = rng.permutation(n)
    n_val = max(1, int(round(n * float(args.val_fraction)))) if n > 1 else 0
    val_idx = perm[:n_val]
    train_idx = perm[n_val:] if n_val else perm
    if train_idx.size == 0:
        train_idx = perm
        val_idx = perm[:0]

    scaler = _fit_scaler(arr[train_idx], target_a[train_idx], meta, fit_scope="train_only")
    device = torch.device(args.device)
    scaler.to_tensors(device=device, dtype=torch.float32)

    cfg: dict[str, Any] = {
        "data": str(data_path),
        "train_data_path": str(data_path),
        "dataset_name": args.dataset_name,
        "central_body": "moon",
        "target_mode": target_mode,
        "degree_min": degree_min,
        "degree_max": degree_max,
        "unit_system": "si",
        "resolved_mu_si": mu_si,
        "resolved_r_ref_m": r_ref_m,
        "resolved_a_sign": 1.0,
        "activation": args.activation,
        "hidden": int(args.hidden),
        "depth": int(args.depth),
        "dropout": float(args.dropout),
        "use_residual_blocks": bool(args.use_residual_blocks),
        "n_bands": int(args.n_bands),
        "w0_bands": None,
        "w0_first": float(args.w0_first),
        "w0_hidden": float(args.w0_hidden),
        "runtime_model_kind": "force_direct",
        "prediction_kind": "residual_force",
        "output_dim": 3,
        "model_preset": str(getattr(args, "model_preset", "baseline_raw") or "baseline_raw"),
        "use_fourier": False,
        "fourier_append_raw": True,
        "fourier_n_features": 0,
        "fourier_sigma": 0.0,
        "fourier_seed": int(args.seed),
        "use_sh_encoding": False,
        "sh_encoding_degree": 4,
        "sh_append_raw": True,
        "use_radial_separation": False,
        "radial_append_raw": False,
        "use_radial_decay_encoding": False,
        "radial_decay_max_power": 4,
        "radial_decay_append_raw": True,
        "use_physical_radial_decay_encoding": False,
        "physical_radial_decay_max_power": 4,
        "physical_radial_decay_append_raw": True,
        "physical_radial_decay_include_unit": True,
        "physical_radial_decay_include_r_scaled": True,
        "altitude_min_km": meta.alt_min_km,
        "altitude_max_km": meta.alt_max_km,
        "best_metric": "val_force_mse",
        "run_name": "force_direct",
        # Origin-fixed isometric x scale (meters). Needed by physically-informed
        # encodings (e.g. recommended_physical_radial_decay) to recover r_phys;
        # harmless for the raw-xyz baseline.
        "x_scale_m": float(scaler.x.scale),
    }
    model = build_model_from_config(cfg, device=device, dtype=torch.float32)
    cfg["input_feature_dim"] = int(getattr(model, "input_feature_dim", 3))
    cfg["embedding_type"] = str(getattr(model, "embedding_type", "raw"))
    cfg["model_builder_version"] = str(getattr(model, "model_builder_version", "unknown"))
    cfg["output_dim"] = int(getattr(model, "output_dim", 3))
    arch_sig = compute_architecture_signature(cfg)

    dataset_meta = _dataset_meta_block(
        meta,
        dataset_name=args.dataset_name,
        target_mode=target_mode,
        degree_min=degree_min,
        degree_max=degree_max,
        mu_si=mu_si,
        r_ref_m=r_ref_m,
        alt_min_km=meta.alt_min_km,
        alt_max_km=meta.alt_max_km,
        n_samples=int(arr.shape[0]),
    )
    # Pin the train/val split so the artifact is reproducibility-verifiable
    # (TRUSTED-eligible) rather than quarantined for a missing split digest.
    split_indices = {
        "train": np.asarray(train_idx, dtype=np.int64),
        "val": np.asarray(val_idx, dtype=np.int64),
        "test": np.asarray([], dtype=np.int64),
        "ood": np.asarray([], dtype=np.int64),
    }
    split_manifest = build_split_manifest(
        dataset_contract=dataset_meta["dataset_contract"],
        splits=split_indices,
        split_policy="random_holdout",
        split_seed=int(args.seed),
        xyz=np.asarray(arr[:, 0:3], dtype=np.float64),
    )
    dataset_meta["split_manifest"] = split_manifest

    resolved_cfg = build_resolved_config(
        cfg, dataset_meta, model, scaler, arch_sig, determinism=determinism_flags
    )

    x_t = torch.as_tensor(arr[:, 0:3], dtype=torch.float32, device=device)
    y_t = torch.as_tensor(target_a, dtype=torch.float32, device=device)
    x_scaled = scaler.scale_x(x_t)
    y_scaled = scaler.scale_a(y_t)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    batch_size = max(1, int(args.batch_size))
    best_val = float("inf")
    history: list[dict[str, float | int]] = []
    train_idx_t = torch.as_tensor(train_idx, dtype=torch.long, device=device)
    val_idx_t = torch.as_tensor(val_idx, dtype=torch.long, device=device) if val_idx.size else None

    for epoch in range(int(args.epochs)):
        model.train()
        order = train_idx_t[torch.randperm(int(train_idx_t.numel()), device=device)]
        losses: list[float] = []
        for start in range(0, int(order.numel()), batch_size):
            idx = order[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            pred = model(x_scaled[idx])
            loss = torch.mean((pred - y_scaled[idx]) ** 2)
            loss.backward()
            if float(args.max_grad_norm) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.max_grad_norm))
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            if val_idx_t is not None and int(val_idx_t.numel()) > 0:
                val_loss = float(torch.mean((model(x_scaled[val_idx_t]) - y_scaled[val_idx_t]) ** 2).cpu())
            else:
                val_loss = float(np.mean(losses))
        train_loss = float(np.mean(losses)) if losses else val_loss
        best_val = min(best_val, val_loss)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if not args.quiet:
            print(f"[force-direct] epoch {epoch + 1}/{args.epochs} train={train_loss:.6e} val={val_loss:.6e}")

    run_dir = Path(args.out).expanduser().resolve()
    if args.timestamped:
        run_dir = run_dir / f"force_direct_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    layout = ensure_run_layout(run_dir)
    write_split_manifest(layout.provenance_dir / "split_manifest.json", split_manifest)
    layout.config_json.write_text(_json_text(resolved_cfg), encoding="utf-8")
    scaler.save_json(layout.scaler_json)
    layout.history_jsonl.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in history),
        encoding="utf-8",
    )
    layout.history_csv.write_text(
        "epoch,train_loss,val_loss\n"
        + "".join(f"{row['epoch']},{row['train_loss']},{row['val_loss']}\n" for row in history),
        encoding="utf-8",
    )
    payload = build_checkpoint_payload(
        kind="best",
        epoch=max(0, int(args.epochs) - 1),
        model=model,
        optimizer=optimizer,
        scheduler=None,
        cfg=resolved_cfg,
        scaler=scaler,
        train_stats={
            "lr": float(args.lr),
            "w_u": 0.0,
            "w_a": 1.0,
            "gradnorm_status": "not_used_direct_force",
            "accel_factor": 1.0,
            "lambda_dir_eff": 0.0,
        },
        val_stats={"loss": best_val, "val_total_loss": best_val, "val_checkpoint_score": best_val},
        dataset_meta=dataset_meta,
        architecture_signature=arch_sig,
        global_step=len(history),
    )
    torch.save(payload, layout.ckpt_best)
    torch.save(payload, layout.ckpt_last)
    write_command_txt(layout)
    write_run_manifest(
        layout,
        {
            "schema_version": "st_lrps_run_manifest_v1",
            "run_id": layout.run_dir.name,
            "status": "completed",
            "runtime_model_kind": "force_direct",
            "prediction_kind": "residual_force",
            "output_dim": 3,
            "data": str(data_path),
            "epochs": int(args.epochs),
            "best_val_loss": best_val,
            "notes": [
                "Direct-force artifact: predicts residual acceleration directly.",
                "No scalar potential or conservative-field guarantee is provided.",
            ],
        },
    )
    capture_environment_snapshot(layout, extra={"runtime_model_kind": "force_direct"})
    if not args.quiet:
        print(f"[force-direct] wrote artifact: {layout.run_dir}")
    return layout.run_dir


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a minimal ST-LRPS force_direct residual-acceleration artifact.")
    parser.add_argument("--data", required=True, help="HDF5 dataset with rows [x,y,z,U,ax,ay,az].")
    parser.add_argument("--out", required=True, help="Output run directory, or parent when --timestamped is set.")
    parser.add_argument("--dataset-name", default="data")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--activation", choices=["sine", "silu", "tanh", "softplus"], default="silu")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--w0-first", type=float, default=30.0)
    parser.add_argument("--w0-hidden", type=float, default=30.0)
    parser.add_argument("--n-bands", type=int, default=1)
    parser.add_argument("--use-residual-blocks", action="store_true")
    parser.add_argument(
        "--model-preset",
        choices=["baseline_raw", "recommended_physical_radial_decay"],
        default="baseline_raw",
        help=(
            "Input-encoding preset for the student. 'baseline_raw' keeps raw xyz "
            "inputs (default, current behaviour); 'recommended_physical_radial_decay' "
            "enables the true R_ref/r radial-decay features. Same encoding semantics "
            "as the scalar-potential trainer; force_direct still predicts residual "
            "acceleration directly (no autograd, no scalar potential)."
        ),
    )
    parser.add_argument("--target-mode", choices=["residual", "full"], default=None)
    parser.add_argument("--degree-min", type=int, default=None)
    parser.add_argument("--degree-max", type=int, default=None)
    parser.add_argument("--mu-si", type=float, default=None)
    parser.add_argument("--r-ref-m", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--timestamped", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    train_force_direct(args)


if __name__ == "__main__":  # pragma: no cover
    try:
        main()
    except KeyboardInterrupt:
        print("\n[STOP] Direct-force training interrupted.")
        sys.exit(130)
