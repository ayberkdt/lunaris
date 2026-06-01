"""Representative orbit-state sampling and RIC decomposition."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, pi, sin, sqrt
from typing import Iterable, List, Tuple

import numpy as np

from lunaris.common.constants import AU, MU_MOON, R_MOON_MEAN

from .config import PerturbationBudgetConfig


@dataclass(frozen=True, slots=True)
class SampleState:
    sample_id: str
    altitude_km: float
    inclination_deg: float
    true_anomaly_deg: float
    epoch_utc: str
    r_m: np.ndarray
    v_m_s: np.ndarray
    sun_m: np.ndarray
    earth_m: np.ndarray
    geometry_source: str


def _deg2rad(value: float) -> float:
    return float(value) * pi / 180.0


def _synthetic_sun_earth(epoch_index: int, epoch_utc: str) -> Tuple[np.ndarray, np.ndarray]:
    """Return deterministic representative Moon-centered Sun/Earth vectors."""
    # Spread epochs through geometry without depending on SPICE kernels.
    phase = 2.0 * pi * (epoch_index % 12) / 12.0
    if epoch_utc:
        phase += (sum(ord(c) for c in epoch_utc) % 360) * pi / 180.0 / 17.0
    earth_distance_m = 384_400_000.0
    sun = np.array([AU * cos(phase), AU * sin(phase), 0.18 * AU * sin(0.5 * phase)], dtype=np.float64)
    earth = np.array(
        [
            earth_distance_m * cos(phase + 1.1),
            earth_distance_m * sin(phase + 1.1),
            0.12 * earth_distance_m * sin(phase),
        ],
        dtype=np.float64,
    )
    return sun, earth


def circular_state(altitude_km: float, inclination_deg: float, true_anomaly_deg: float) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a simple circular lunar orbit state in an inertial frame."""
    radius_m = float(R_MOON_MEAN) + float(altitude_km) * 1000.0
    inc = _deg2rad(inclination_deg)
    nu = _deg2rad(true_anomaly_deg)
    r_pf = np.array([radius_m * cos(nu), radius_m * sin(nu), 0.0], dtype=np.float64)
    v_mag = sqrt(float(MU_MOON) / radius_m)
    v_pf = np.array([-v_mag * sin(nu), v_mag * cos(nu), 0.0], dtype=np.float64)

    # Rotation by inclination about x-axis, with RAAN=argp=0 for deterministic sampling.
    rot = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cos(inc), -sin(inc)],
            [0.0, sin(inc), cos(inc)],
        ],
        dtype=np.float64,
    )
    return rot @ r_pf, rot @ v_pf


def generate_sample_states(config: PerturbationBudgetConfig) -> List[SampleState]:
    samples: List[SampleState] = []
    geometry_source = "synthetic_geometry"
    for epoch_index, epoch in enumerate(config.epochs_utc):
        sun, earth = _synthetic_sun_earth(epoch_index, epoch)
        for altitude in config.altitudes_km:
            for inc in config.inclinations_deg:
                for anomaly in config.true_anomalies_deg:
                    r, v = circular_state(altitude, inc, anomaly)
                    sample_id = (
                        f"alt{altitude:g}_inc{inc:g}_nu{anomaly:g}_epoch{epoch_index}"
                        .replace(".", "p")
                        .replace("-", "m")
                    )
                    samples.append(
                        SampleState(
                            sample_id=sample_id,
                            altitude_km=float(altitude),
                            inclination_deg=float(inc),
                            true_anomaly_deg=float(anomaly),
                            epoch_utc=str(epoch),
                            r_m=r,
                            v_m_s=v,
                            sun_m=sun,
                            earth_m=earth,
                            geometry_source=geometry_source,
                        )
                    )
    return samples


def _unit(vec: np.ndarray, name: str) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError(f"{name} vector is degenerate")
    return np.asarray(vec, dtype=np.float64) / norm


def ric_frame(r_m: Iterable[float], v_m_s: Iterable[float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return radial, along-track/transverse, and cross-track unit vectors."""
    r = np.asarray(r_m, dtype=np.float64)
    v = np.asarray(v_m_s, dtype=np.float64)
    if r.shape != (3,) or v.shape != (3,):
        raise ValueError("r_m and v_m_s must have shape (3,)")
    r_hat = _unit(r, "radial")
    h_hat = _unit(np.cross(r, v), "orbit angular momentum")
    t_hat = _unit(np.cross(h_hat, r_hat), "along-track")
    return r_hat, t_hat, h_hat


def decompose_ric(accel_m_s2: Iterable[float], r_m: Iterable[float], v_m_s: Iterable[float]) -> Tuple[float, float, float]:
    a = np.asarray(accel_m_s2, dtype=np.float64)
    if a.shape != (3,):
        raise ValueError("accel_m_s2 must have shape (3,)")
    r_hat, t_hat, h_hat = ric_frame(r_m, v_m_s)
    return float(a @ r_hat), float(a @ t_hat), float(a @ h_hat)
