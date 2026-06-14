"""Field-level evaluation for ST-LRPS force_direct artifacts."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from lunaris.surrogate.st_lrps.runtime.force_model import (
    DirectForceRuntime,
    load_surrogate_force_model,
)


def _json_text(payload: Mapping[str, Any]) -> str:
    return json.dumps(dict(payload), indent=2, sort_keys=True, ensure_ascii=True, default=str) + "\n"


def _read_h5(path: Path, dataset_name: str, max_samples: int | None, seed: int) -> np.ndarray:
    import h5py  # type: ignore

    with h5py.File(path, "r") as handle:
        name = dataset_name if dataset_name in handle else next(
            (key for key in handle.keys() if hasattr(handle[key], "shape")),
            None,
        )
        if name is None:
            raise ValueError(f"No HDF5 dataset found in {path}")
        ds = handle[name]
        if len(ds.shape) != 2 or int(ds.shape[1]) < 7:
            raise ValueError(f"Expected HDF5 rows [x,y,z,U,ax,ay,az], got shape {ds.shape}")
        n = int(ds.shape[0])
        if max_samples is not None and int(max_samples) < n:
            rng = np.random.default_rng(int(seed))
            idx = np.sort(rng.choice(n, size=int(max_samples), replace=False).astype(np.int64))
            return np.asarray(ds[idx, :], dtype=np.float64)
        return np.asarray(ds[:, :], dtype=np.float64)


def _angular_deg(a_true: np.ndarray, a_pred: np.ndarray) -> np.ndarray:
    true_norm = np.linalg.norm(a_true, axis=1).clip(1e-30)
    pred_norm = np.linalg.norm(a_pred, axis=1).clip(1e-30)
    cos = np.sum(a_true * a_pred, axis=1) / (true_norm * pred_norm)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def _radial_cross(err: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r_hat = x / np.linalg.norm(x, axis=1, keepdims=True).clip(1e-30)
    radial = np.sum(err * r_hat, axis=1)
    cross = np.linalg.norm(err - radial[:, None] * r_hat, axis=1)
    return radial, cross


def evaluate_force_direct(
    model_dir: str | Path,
    data: str | Path,
    *,
    dataset_name: str = "data",
    out: str | Path | None = None,
    device: str = "auto",
    batch_size: int = 8192,
    max_samples: int | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    runtime = load_surrogate_force_model(model_dir, device=device, chunk_size=batch_size)
    if not isinstance(runtime, DirectForceRuntime):
        raise ValueError(
            "force_direct evaluation requires a DirectForceRuntime artifact; "
            f"loaded runtime_model_kind={getattr(runtime, 'runtime_model_kind', None)!r}."
        )
    arr = _read_h5(Path(data).expanduser().resolve(), dataset_name, max_samples, seed)
    x = arr[:, 0:3]
    a_true = arr[:, 4:7]
    a_pred = np.asarray(runtime.predict_residual_accel_fixed(x), dtype=np.float64)
    err = a_pred - a_true
    err_norm = np.linalg.norm(err, axis=1)
    true_norm = np.linalg.norm(a_true, axis=1).clip(1e-30)
    ang = _angular_deg(a_true, a_pred)
    radial, cross = _radial_cross(err, x)
    status = runtime.domain_status(x)
    report = {
        "schema_version": 1,
        "runtime_model_kind": "force_direct",
        "model_dir": str(Path(model_dir).expanduser()),
        "data": str(Path(data).expanduser()),
        "n_samples": int(x.shape[0]),
        "potential_metrics": {
            "available": False,
            "rmse_u": None,
            "mae_u": None,
            "note": "force_direct artifacts predict residual acceleration directly and do not predict DeltaU.",
        },
        "acceleration_metrics": {
            "rmse_a_vec": float(np.sqrt(np.mean(err_norm ** 2))),
            "mae_a_vec": float(np.mean(err_norm)),
            "max_abs_a_vec": float(np.max(err_norm)),
            "robust_rel_err": float(np.sum(err_norm) / max(float(np.sum(true_norm)), 1e-30)),
        },
        "angular_metrics": {
            "mean_deg": float(np.mean(ang)),
            "median_deg": float(np.median(ang)),
            "p90_deg": float(np.percentile(ang, 90)),
            "p95_deg": float(np.percentile(ang, 95)),
        },
        "directional_metrics": {
            "radial_rmse": float(np.sqrt(np.mean(radial ** 2))),
            "cross_radial_rmse": float(np.sqrt(np.mean(cross ** 2))),
            "radial_mae": float(np.mean(np.abs(radial))),
            "cross_radial_mae": float(np.mean(cross)),
        },
        "domain_status": status,
        "warnings": [
            "force_direct is not a scalar-potential model and is not conservative by construction.",
            "Use curl and orbit-level validation before scientific claims.",
        ],
    }
    if out is not None:
        out_path = Path(out).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(_json_text(report), encoding="utf-8")
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a force_direct ST-LRPS residual-acceleration artifact.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--dataset-name", default="data")
    parser.add_argument("--out", default="outputs/force_direct_eval/metrics.json")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    report = evaluate_force_direct(
        args.model_dir,
        args.data,
        dataset_name=args.dataset_name,
        out=args.out,
        device=args.device,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        seed=args.seed,
    )
    print(_json_text(report))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
