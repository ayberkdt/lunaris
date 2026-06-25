"""Field-level evaluation for ST-LRPS force_direct artifacts."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Mapping, Sequence
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


def curl_diagnostics(
    accel_fn: Callable[[np.ndarray], np.ndarray],
    points: np.ndarray,
    *,
    step_m: float = 1.0,
) -> dict[str, Any]:
    """Finite-difference non-conservativeness (curl) diagnostic for a force field.

    A conservative acceleration field ``a = -grad U`` has zero curl and a
    symmetric Jacobian everywhere. ``force_direct`` artifacts predict the
    residual acceleration directly, with no scalar potential, so they carry no
    structural guarantee of conservativeness. The antisymmetric part of the
    field Jacobian (equivalently ``curl a``) measures how far the predicted
    field is from *any* potential model.

    The Jacobian is estimated with central differences,
    ``J[i, j] = d a_i / d r_j``, using ``+/- step_m`` along each Cartesian axis
    (6 field evaluations per point). The diagnostic returns:

    * ``curl_abs_*`` -- magnitude statistics of ``curl a`` [1/s^2];
    * ``nonconservative_ratio`` -- pooled ``||antisym(J)||_F / ||J||_F`` in
      ``[0, 1]`` (``0`` == conservative), a scale-free measure that is
      independent of the absolute field strength;
    * ``nonconservative_ratio_per_point_*`` -- the same ratio summarised across
      points.

    Parameters
    ----------
    accel_fn:
        Maps ``(N, 3)`` Moon-fixed positions [m] to ``(N, 3)`` accelerations
        [m/s^2]. The runtime's ``predict_residual_accel_fixed`` satisfies this.
    points:
        ``(N, 3)`` evaluation positions [m].
    step_m:
        Central-difference step [m]. Must be positive.
    """
    if step_m <= 0.0:
        raise ValueError(f"step_m must be positive, got {step_m!r}.")
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    n = int(pts.shape[0])
    if n == 0:
        raise ValueError("curl_diagnostics requires at least one point.")

    # J[:, i, j] = d a_i / d r_j via central differences along each axis j.
    jac = np.empty((n, 3, 3), dtype=np.float64)
    for j in range(3):
        offset = np.zeros(3, dtype=np.float64)
        offset[j] = step_m
        a_plus = np.asarray(accel_fn(pts + offset), dtype=np.float64).reshape(n, 3)
        a_minus = np.asarray(accel_fn(pts - offset), dtype=np.float64).reshape(n, 3)
        jac[:, :, j] = (a_plus - a_minus) / (2.0 * step_m)

    # curl a = (dz/dy - dy/dz, dx/dz - dz/dx, dy/dx - dx/dy).
    curl = np.stack(
        [
            jac[:, 2, 1] - jac[:, 1, 2],
            jac[:, 0, 2] - jac[:, 2, 0],
            jac[:, 1, 0] - jac[:, 0, 1],
        ],
        axis=1,
    )
    curl_mag = np.linalg.norm(curl, axis=1)

    # Antisymmetric part A = (J - J^T)/2; ||A||_F^2 = ||curl||^2 / 2 per point.
    antisym = 0.5 * (jac - np.transpose(jac, (0, 2, 1)))
    antisym_fro = np.linalg.norm(antisym.reshape(n, 9), axis=1)
    jac_fro = np.linalg.norm(jac.reshape(n, 9), axis=1)

    # Pooled, energy-weighted ratio (robust to near-zero-gradient points), plus
    # a per-point ratio summary that ignores points with no gradient signal.
    pooled = float(
        np.sqrt(np.sum(antisym_fro ** 2) / max(float(np.sum(jac_fro ** 2)), 1e-300))
    )
    valid = jac_fro > 1e-30
    if np.any(valid):
        per_point = antisym_fro[valid] / jac_fro[valid]
        per_point_mean = float(np.mean(per_point))
        per_point_max = float(np.max(per_point))
    else:
        per_point_mean = 0.0
        per_point_max = 0.0

    return {
        "n_points": n,
        "step_m": float(step_m),
        "curl_abs_rms": float(np.sqrt(np.mean(curl_mag ** 2))),
        "curl_abs_median": float(np.median(curl_mag)),
        "curl_abs_max": float(np.max(curl_mag)),
        "nonconservative_ratio": pooled,
        "nonconservative_ratio_per_point_mean": per_point_mean,
        "nonconservative_ratio_per_point_max": per_point_max,
        "note": (
            "nonconservative_ratio = ||antisym(J)||_F / ||J||_F in [0, 1]; "
            "0 indicates a conservative (potential) field."
        ),
    }


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
    curl_step_m: float = 1.0,
    curl_max_points: int | None = 4096,
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

    # Non-conservativeness (curl) diagnostic on a subsample of the eval points.
    # force_direct has no potential, so this quantifies the warning rather than
    # only asserting it. Computed on the same predicted field used above.
    if curl_max_points is not None and int(curl_max_points) < x.shape[0]:
        rng = np.random.default_rng(int(seed))
        curl_idx = rng.choice(x.shape[0], size=int(curl_max_points), replace=False)
        curl_points = x[curl_idx]
    else:
        curl_points = x
    conservativeness = curl_diagnostics(
        lambda pts: np.asarray(runtime.predict_residual_accel_fixed(pts), dtype=np.float64),
        curl_points,
        step_m=curl_step_m,
    )

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
        "conservativeness_metrics": conservativeness,
        "warnings": [
            "force_direct is not a scalar-potential model and is not conservative by construction.",
            "nonconservative_ratio="
            f"{conservativeness['nonconservative_ratio']:.3e} (0==conservative); "
            "complete orbit-level drift validation before scientific claims.",
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
    parser.add_argument(
        "--curl-step-m", type=float, default=1.0,
        help="Central-difference step [m] for the curl/non-conservativeness diagnostic.",
    )
    parser.add_argument(
        "--curl-max-points", type=int, default=4096,
        help="Subsample size for the curl diagnostic (6 field evals per point).",
    )
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
        curl_step_m=args.curl_step_m,
        curl_max_points=args.curl_max_points,
    )
    print(_json_text(report))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
