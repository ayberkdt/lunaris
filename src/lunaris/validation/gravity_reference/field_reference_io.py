"""Read and write pointwise field reference samples."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import numpy as np

REQUIRED_FIELD_COLUMNS = (
    "point_id",
    "x_m",
    "y_m",
    "z_m",
    "potential_m2_s2",
    "ax_m_s2",
    "ay_m_s2",
    "az_m_s2",
)


def _as_finite_float(value: str, *, column: str, row_id: str) -> float:
    try:
        out = float(value)
    except ValueError as exc:
        raise ValueError(f"Field sample {row_id!r} column {column!r} is not numeric.") from exc
    if not math.isfinite(out):
        raise ValueError(f"Field sample {row_id!r} column {column!r} is not finite.")
    return out


def load_field_reference_csv(path: str | Path) -> dict[str, Any]:
    """Load strict field reference CSV rows into arrays."""
    rows: list[dict[str, str]] = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("Field reference CSV is missing a header.")
        missing = [c for c in REQUIRED_FIELD_COLUMNS if c not in reader.fieldnames]
        if missing:
            raise ValueError(f"Field reference CSV missing columns: {missing}")
        rows = list(reader)

    if not rows:
        raise ValueError("Field reference CSV contains no samples.")

    point_ids: list[str] = []
    positions: list[list[float]] = []
    potentials: list[float] = []
    accelerations: list[list[float]] = []
    seen_ids: set[str] = set()
    seen_xyz: set[tuple[float, float, float]] = set()

    for index, row in enumerate(rows, start=1):
        point_id = str(row["point_id"]).strip()
        if not point_id:
            raise ValueError(f"Field reference row {index} has an empty point_id.")
        if point_id in seen_ids:
            raise ValueError(f"Duplicate field point_id: {point_id}")
        seen_ids.add(point_id)
        x = _as_finite_float(row["x_m"], column="x_m", row_id=point_id)
        y = _as_finite_float(row["y_m"], column="y_m", row_id=point_id)
        z = _as_finite_float(row["z_m"], column="z_m", row_id=point_id)
        xyz = (x, y, z)
        if xyz in seen_xyz:
            raise ValueError(f"Duplicate field position: {xyz}")
        seen_xyz.add(xyz)
        potential = _as_finite_float(
            row["potential_m2_s2"], column="potential_m2_s2", row_id=point_id
        )
        ax = _as_finite_float(row["ax_m_s2"], column="ax_m_s2", row_id=point_id)
        ay = _as_finite_float(row["ay_m_s2"], column="ay_m_s2", row_id=point_id)
        az = _as_finite_float(row["az_m_s2"], column="az_m_s2", row_id=point_id)

        point_ids.append(point_id)
        positions.append([x, y, z])
        potentials.append(potential)
        accelerations.append([ax, ay, az])

    return {
        "point_ids": point_ids,
        "positions_m": np.asarray(positions, dtype=np.float64),
        "potential_m2_s2": np.asarray(potentials, dtype=np.float64),
        "acceleration_m_s2": np.asarray(accelerations, dtype=np.float64),
    }


def write_field_samples_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    """Write per-sample comparison rows."""
    dst = Path(path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "point_id",
        "x_m",
        "y_m",
        "z_m",
        "ref_potential_m2_s2",
        "lunaris_potential_m2_s2",
        "potential_abs_error_m2_s2",
        "potential_rel_error",
        "ref_ax_m_s2",
        "ref_ay_m_s2",
        "ref_az_m_s2",
        "lunaris_ax_m_s2",
        "lunaris_ay_m_s2",
        "lunaris_az_m_s2",
        "accel_norm_error_m_s2",
        "accel_rel_norm_error",
        "accel_angle_error_deg",
        "radial_accel_error_m_s2",
        "tangential_accel_error_m_s2",
    ]
    with dst.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

