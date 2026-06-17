"""Field-level comparison metrics."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def _safe_rel(num: float, denom: float) -> float:
    base = abs(float(denom))
    return abs(float(num)) / base if base > 0.0 else abs(float(num))


def _percentiles(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    return {
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
    }


def compute_field_metrics(
    *,
    point_ids: list[str],
    positions_m: np.ndarray,
    reference_potential_m2_s2: np.ndarray,
    reference_acceleration_m_s2: np.ndarray,
    lunaris_potential_m2_s2: np.ndarray,
    lunaris_acceleration_m_s2: np.ndarray,
    reference_radius_m: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Compute per-sample and aggregate field metrics."""
    positions = np.asarray(positions_m, dtype=np.float64)
    ref_u = np.asarray(reference_potential_m2_s2, dtype=np.float64).reshape(-1)
    ref_a = np.asarray(reference_acceleration_m_s2, dtype=np.float64)
    got_u = np.asarray(lunaris_potential_m2_s2, dtype=np.float64).reshape(-1)
    got_a = np.asarray(lunaris_acceleration_m_s2, dtype=np.float64)

    if positions.shape != ref_a.shape or positions.shape != got_a.shape or positions.shape[1] != 3:
        raise ValueError("Field metric arrays must have shape (N,3) for positions/accelerations.")
    n = positions.shape[0]
    if len(point_ids) != n or ref_u.size != n or got_u.size != n:
        raise ValueError("Field metric arrays have inconsistent sample counts.")

    rows: list[dict[str, Any]] = []
    pot_abs = np.zeros(n, dtype=np.float64)
    pot_rel = np.zeros(n, dtype=np.float64)
    accel_norm = np.zeros(n, dtype=np.float64)
    accel_rel = np.zeros(n, dtype=np.float64)
    accel_angle = np.zeros(n, dtype=np.float64)
    radial_err = np.zeros(n, dtype=np.float64)
    tangential_err = np.zeros(n, dtype=np.float64)
    component_abs = np.abs(got_a - ref_a)
    altitudes = np.linalg.norm(positions, axis=1) - float(reference_radius_m)

    for i in range(n):
        du = float(got_u[i] - ref_u[i])
        da = got_a[i] - ref_a[i]
        pos = positions[i]
        r_norm = float(np.linalg.norm(pos))
        r_hat = pos / r_norm if r_norm > 0.0 else np.array([1.0, 0.0, 0.0])
        radial = float(np.dot(da, r_hat))
        tangential_vec = da - radial * r_hat
        ref_norm = float(np.linalg.norm(ref_a[i]))
        got_norm = float(np.linalg.norm(got_a[i]))
        err_norm = float(np.linalg.norm(da))
        dot = float(np.dot(ref_a[i], got_a[i]))
        if ref_norm > 0.0 and got_norm > 0.0:
            cos_angle = max(-1.0, min(1.0, dot / (ref_norm * got_norm)))
            angle_deg = math.degrees(math.acos(cos_angle))
        else:
            angle_deg = 0.0

        pot_abs[i] = abs(du)
        pot_rel[i] = _safe_rel(du, ref_u[i])
        accel_norm[i] = err_norm
        accel_rel[i] = err_norm / ref_norm if ref_norm > 0.0 else err_norm
        accel_angle[i] = angle_deg
        radial_err[i] = abs(radial)
        tangential_err[i] = float(np.linalg.norm(tangential_vec))

        rows.append({
            "point_id": point_ids[i],
            "x_m": f"{positions[i, 0]:.17e}",
            "y_m": f"{positions[i, 1]:.17e}",
            "z_m": f"{positions[i, 2]:.17e}",
            "ref_potential_m2_s2": f"{ref_u[i]:.17e}",
            "lunaris_potential_m2_s2": f"{got_u[i]:.17e}",
            "potential_abs_error_m2_s2": f"{pot_abs[i]:.17e}",
            "potential_rel_error": f"{pot_rel[i]:.17e}",
            "ref_ax_m_s2": f"{ref_a[i, 0]:.17e}",
            "ref_ay_m_s2": f"{ref_a[i, 1]:.17e}",
            "ref_az_m_s2": f"{ref_a[i, 2]:.17e}",
            "lunaris_ax_m_s2": f"{got_a[i, 0]:.17e}",
            "lunaris_ay_m_s2": f"{got_a[i, 1]:.17e}",
            "lunaris_az_m_s2": f"{got_a[i, 2]:.17e}",
            "accel_norm_error_m_s2": f"{accel_norm[i]:.17e}",
            "accel_rel_norm_error": f"{accel_rel[i]:.17e}",
            "accel_angle_error_deg": f"{accel_angle[i]:.17e}",
            "radial_accel_error_m_s2": f"{radial_err[i]:.17e}",
            "tangential_accel_error_m_s2": f"{tangential_err[i]:.17e}",
        })

    metrics = {
        "n_points": int(n),
        "potential_abs_error_m2_s2": _percentiles(pot_abs),
        "potential_rel_error": _percentiles(pot_rel),
        "acceleration_component_abs_error_m_s2": {
            "x": _percentiles(component_abs[:, 0]),
            "y": _percentiles(component_abs[:, 1]),
            "z": _percentiles(component_abs[:, 2]),
            "max_any_component": float(np.max(component_abs)) if component_abs.size else 0.0,
        },
        "acceleration_norm_error_m_s2": _percentiles(accel_norm),
        "acceleration_rel_norm_error": _percentiles(accel_rel),
        "acceleration_angle_error_deg": _percentiles(accel_angle),
        "radial_acceleration_error_m_s2": _percentiles(radial_err),
        "tangential_acceleration_error_m_s2": _percentiles(tangential_err),
        "altitude_m": {
            "min": float(np.min(altitudes)),
            "max": float(np.max(altitudes)),
        },
        "worst_point": {
            "point_id": point_ids[int(np.argmax(accel_norm))],
            "acceleration_norm_error_m_s2": float(np.max(accel_norm)),
        },
    }
    return metrics, rows

