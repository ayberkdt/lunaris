from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent
SCENARIO = Path(os.environ.get("LUNARIS_TUDAT_SCENARIO", ROOT / "scenario.json")).resolve()
RESULTS = Path(os.environ.get("LUNARIS_TUDAT_RESULTS", ROOT / "results")).resolve()


def repo_root() -> Path:
    configured = os.environ.get("LUNARIS_REPO_ROOT")
    return Path(configured).resolve() if configured else ROOT.parents[4]


def load_scenario() -> dict[str, Any]:
    scenario = json.loads(SCENARIO.read_text(encoding="utf-8"))
    root = repo_root()
    for key in ("gravity_file", "gravity_label"):
        path = Path(scenario[key])
        scenario[key] = str(path if path.is_absolute() else root / path)
    scenario["kernel_files"] = [
        str(path if path.is_absolute() else root / path)
        for item in scenario["kernel_files"]
        for path in (Path(item),)
    ]
    return scenario


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: str | Path, payload: Any) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_trajectory_csv(
    path: str | Path,
    epochs_rel_s: np.ndarray,
    states: np.ndarray,
    accelerations: np.ndarray,
) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "epoch_rel_s",
        "x_m",
        "y_m",
        "z_m",
        "vx_m_s",
        "vy_m_s",
        "vz_m_s",
        "ax_m_s2",
        "ay_m_s2",
        "az_m_s2",
    ]
    with out.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        for t, state, acceleration in zip(epochs_rel_s, states, accelerations, strict=True):
            writer.writerow(
                [f"{float(t):.9f}"]
                + [f"{float(value):.17e}" for value in state]
                + [f"{float(value):.17e}" for value in acceleration]
            )


def read_trajectory_csv(path: str | Path) -> dict[str, np.ndarray]:
    data = np.genfromtxt(Path(path), delimiter=",", names=True, dtype=np.float64)
    if data.ndim == 0:
        data = np.asarray([data], dtype=data.dtype)
    state = np.column_stack(
        [data[name] for name in ("x_m", "y_m", "z_m", "vx_m_s", "vy_m_s", "vz_m_s")]
    )
    acceleration = np.column_stack([data[name] for name in ("ax_m_s2", "ay_m_s2", "az_m_s2")])
    return {
        "epoch_rel_s": np.asarray(data["epoch_rel_s"], dtype=np.float64),
        "state": np.asarray(state, dtype=np.float64),
        "acceleration": np.asarray(acceleration, dtype=np.float64),
    }


def vector_error_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    error = np.asarray(candidate, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    norms = np.linalg.norm(error, axis=1)
    return {
        "max": float(np.max(norms)),
        "rms": float(np.sqrt(np.mean(norms * norms))),
        "final": float(norms[-1]),
    }


def state_error_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    position = vector_error_metrics(reference[:, :3], candidate[:, :3])
    velocity = vector_error_metrics(reference[:, 3:], candidate[:, 3:])

    ric = np.empty((reference.shape[0], 3), dtype=np.float64)
    delta_r = candidate[:, :3] - reference[:, :3]
    for index, state in enumerate(reference):
        r = state[:3]
        v = state[3:]
        r_hat = r / np.linalg.norm(r)
        c_hat = np.cross(r, v)
        c_hat /= np.linalg.norm(c_hat)
        i_hat = np.cross(c_hat, r_hat)
        ric[index] = np.array(
            [r_hat @ delta_r[index], i_hat @ delta_r[index], c_hat @ delta_r[index]]
        )
    return {
        "position_m": position,
        "velocity_m_s": velocity,
        "ric_max_abs_m": {
            "radial": float(np.max(np.abs(ric[:, 0]))),
            "in_track": float(np.max(np.abs(ric[:, 1]))),
            "cross_track": float(np.max(np.abs(ric[:, 2]))),
        },
        "ric_final_m": {
            "radial": float(ric[-1, 0]),
            "in_track": float(ric[-1, 1]),
            "cross_track": float(ric[-1, 2]),
        },
    }


def assert_same_grid(*datasets: dict[str, np.ndarray]) -> np.ndarray:
    reference = datasets[0]["epoch_rel_s"]
    for dataset in datasets[1:]:
        candidate = dataset["epoch_rel_s"]
        if reference.shape != candidate.shape or not np.allclose(
            reference, candidate, rtol=0.0, atol=1e-8
        ):
            raise ValueError("trajectory epoch grids do not match exactly enough for comparison")
    return reference
