from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np

from lunaris.surrogate.st_lrps.data.dataset_contract import DatasetContract
from lunaris.surrogate.st_lrps.data.dataset_parameters import MU_MOON_SI, R_MOON_SI


def toy_truth_fn(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(xyz, dtype=np.float64)
    u = 2.0e-7 * np.sum(x * x, axis=1)
    a = 4.0e-7 * x
    return u, a


def toy_baseline_fn(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(xyz, dtype=np.float64)
    u = 0.5e-7 * np.sum(x * x, axis=1)
    a = 1.0e-7 * x
    return u, a


def make_toy_residual_rows(
    *,
    n: int = 32,
    alt_min_km: float = 100.0,
    alt_max_km: float = 500.0,
    seed: int = 7,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    dirs = rng.normal(size=(n, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    alt_km = np.linspace(float(alt_min_km), float(alt_max_km), int(n), dtype=np.float64)
    xyz = dirs * (float(R_MOON_SI) + alt_km[:, None] * 1000.0)
    u_truth, a_truth = toy_truth_fn(xyz)
    u_base, a_base = toy_baseline_fn(xyz)
    residual = np.concatenate(
        [
            xyz,
            (u_truth - u_base).reshape(-1, 1),
            a_truth - a_base,
        ],
        axis=1,
    )
    return residual.astype(np.float32)


def make_toy_dataset_contract(
    *,
    n: int = 32,
    alt_min_km: float = 100.0,
    alt_max_km: float = 500.0,
    **overrides: Any,
) -> DatasetContract:
    payload: dict[str, Any] = {
        "dataset_id": "toy_residual_cloud",
        "created_at_utc": "2026-05-31T00:00:00Z",
        "generator_version": "test",
        "repo_commit_sha": "test",
        "random_seed": 7,
        "n_samples": int(n),
        "target_mode": "residual",
        "baseline_kind": "spherical_harmonics",
        "degree_min": 2,
        "degree_max": 4,
        "mu_si": float(MU_MOON_SI),
        "r_ref_m": float(R_MOON_SI),
        "a_sign": 1.0,
        "altitude_min_km": float(alt_min_km),
        "altitude_max_km": float(alt_max_km),
        "sampling_policy": {"name": "toy_shell"},
        "split_policy": {"name": "seeded_random"},
        "source_gravity_model": "toy_spherical_harmonics",
        "source_gravity_file_path": "toy.gfc",
        "source_gravity_file_sha256": "a" * 64,
        "content_sha256": "b" * 64,
        "dataset_layout": {"dataset_name": "data", "shape": [int(n), 7]},
    }
    payload.update(overrides)
    return DatasetContract(**payload)


def write_toy_contract_h5(
    path: Path,
    *,
    n: int = 32,
    alt_min_km: float = 100.0,
    alt_max_km: float = 500.0,
    rows: np.ndarray | None = None,
    contract_overrides: dict[str, Any] | None = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (
        np.asarray(rows, dtype=np.float32)
        if rows is not None
        else make_toy_residual_rows(n=n, alt_min_km=alt_min_km, alt_max_km=alt_max_km)
    )
    contract = make_toy_dataset_contract(
        n=int(data.shape[0]),
        alt_min_km=alt_min_km,
        alt_max_km=alt_max_km,
        **(contract_overrides or {}),
    )
    with h5py.File(path, "w") as handle:
        handle.create_dataset("data", data=data)
        handle.attrs["unit_system"] = "si"
        handle.attrs["central_body"] = "moon"
        handle.attrs["mu_si"] = float(MU_MOON_SI)
        handle.attrs["r_ref_m"] = float(R_MOON_SI)
        handle.attrs["requested_degree"] = int(contract.degree_max or 0)
        handle.attrs["degree_min"] = int(contract.degree_min or 0)
        handle.attrs["degree_max"] = int(contract.degree_max or 0)
        handle.attrs["target_mode"] = str(contract.target_mode)
        handle.attrs["baseline_kind"] = str(contract.baseline_kind)
        handle.attrs["alt_min_km"] = float(contract.altitude_min_km or alt_min_km)
        handle.attrs["alt_max_km"] = float(contract.altitude_max_km or alt_max_km)
        handle.attrs["a_sign_convention"] = "+1"
        handle.attrs["derivative_convention_version"] = str(contract.derivative_convention)
        handle.attrs["columns"] = "[x,y,z,dU,dax,day,daz]"
        handle.attrs["source_gravity_model"] = str(contract.source_gravity_model or "")
        handle.attrs["source_gravity_file_path"] = str(contract.source_gravity_file_path or "")
        handle.attrs["source_gravity_file_sha256"] = str(contract.source_gravity_file_sha256 or "")
        contract.write_hdf5_attrs(handle, generation_config={"name": "toy"})
    return path
