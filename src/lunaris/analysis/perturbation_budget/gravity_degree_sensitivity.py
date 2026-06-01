"""Spherical-harmonic degree sensitivity tables."""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping

import numpy as np

from .acceleration_budget import _norm, _safe_ratio, force_norm, sh_increment_vectors
from .config import PerturbationBudgetConfig
from .sampling import SampleState, decompose_ric


def compute_gravity_degree_sensitivity(
    config: PerturbationBudgetConfig,
    samples: Iterable[SampleState],
    sh_by_sample: Mapping[str, Mapping[int, np.ndarray]],
    forces_by_sample: Mapping[str, Mapping[str, np.ndarray]],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    sample_by_id = {s.sample_id: s for s in samples}

    for sample_id, sh_vectors in sh_by_sample.items():
        sample = sample_by_id[sample_id]
        forces = forces_by_sample[sample_id]
        increments = sh_increment_vectors(sh_vectors)
        srp = _norm(forces.get("SRP", np.zeros(3, dtype=np.float64)))
        albedo = _norm(forces.get("Lunar Albedo", np.zeros(3, dtype=np.float64)))
        thermal = _norm(forces.get("Thermal IR", np.zeros(3, dtype=np.float64)))
        tides = force_norm(forces, "Solid Tides")
        third_body = force_norm(forces, "Third Body Earth", "Third Body Sun", "Earth J2 Differential")

        for band, vec in increments.items():
            norm = _norm(vec)
            radial, along, cross = decompose_ric(vec, sample.r_m, sample.v_m_s)
            lo_hi = band.replace("Delta SH", "").split("->")
            rows.append(
                {
                    "sample_id": sample.sample_id,
                    "altitude_km": sample.altitude_km,
                    "inclination_deg": sample.inclination_deg,
                    "true_anomaly_deg": sample.true_anomaly_deg,
                    "epoch_utc": sample.epoch_utc,
                    "band": band,
                    "degree_low": int(lo_hi[0]),
                    "degree_high": int(lo_hi[1]),
                    "increment_norm_m_s2": norm,
                    "radial_m_s2": radial,
                    "along_track_m_s2": along,
                    "cross_track_m_s2": cross,
                    "ratio_to_srp": _safe_ratio(norm, srp),
                    "ratio_to_albedo": _safe_ratio(norm, albedo),
                    "ratio_to_thermal_ir": _safe_ratio(norm, thermal),
                    "ratio_to_solid_tides": _safe_ratio(norm, tides),
                    "ratio_to_third_body": _safe_ratio(norm, third_body),
                }
            )
    return rows
