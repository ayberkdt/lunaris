"""Strict trajectory reference CSV parsing."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import numpy as np

REQUIRED_TRAJECTORY_COLUMNS = (
    "epoch_s",
    "x_m",
    "y_m",
    "z_m",
    "vx_m_s",
    "vy_m_s",
    "vz_m_s",
)


def _finite(value: str, column: str) -> float:
    try:
        out = float(value)
    except ValueError as exc:
        raise ValueError(f"Trajectory column {column!r} is not numeric.") from exc
    if not math.isfinite(out):
        raise ValueError(f"Trajectory column {column!r} is not finite.")
    return out


def load_trajectory_csv(path: str | Path) -> dict[str, Any]:
    """Load a trajectory CSV with strictly increasing epochs."""
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("Trajectory reference CSV is missing a header.")
        missing = [c for c in REQUIRED_TRAJECTORY_COLUMNS if c not in reader.fieldnames]
        if missing:
            raise ValueError(f"Trajectory reference CSV missing columns: {missing}")
        rows = list(reader)
    if not rows:
        raise ValueError("Trajectory reference CSV contains no rows.")
    epochs: list[float] = []
    states: list[list[float]] = []
    for row in rows:
        epoch = _finite(row["epoch_s"], "epoch_s")
        state = [
            _finite(row["x_m"], "x_m"),
            _finite(row["y_m"], "y_m"),
            _finite(row["z_m"], "z_m"),
            _finite(row["vx_m_s"], "vx_m_s"),
            _finite(row["vy_m_s"], "vy_m_s"),
            _finite(row["vz_m_s"], "vz_m_s"),
        ]
        epochs.append(epoch)
        states.append(state)
    t = np.asarray(epochs, dtype=np.float64)
    if np.any(np.diff(t) <= 0.0):
        raise ValueError("Trajectory epochs must be strictly increasing.")
    return {"epoch_s": t, "state": np.asarray(states, dtype=np.float64)}

