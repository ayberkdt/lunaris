"""Force-model uncertainty estimates and gravity-degree recommendations."""

from __future__ import annotations

from collections import defaultdict
from math import sqrt
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np

from .acceleration_budget import _norm
from .config import PerturbationBudgetConfig
from .sampling import SampleState


def _rss(values: Sequence[float]) -> float:
    return float(sqrt(sum(float(v) * float(v) for v in values)))


def _linear(values: Sequence[float]) -> float:
    return float(sum(abs(float(v)) for v in values))


def _thermal_relative_uncertainty(config: PerturbationBudgetConfig) -> float:
    temp = max(float(config.thermal_temperature_K), 1.0)
    temp_rel = 4.0 * abs(float(config.thermal_temperature_uncertainty_K)) / temp
    return _rss([config.thermal_uncertainty, temp_rel, config.area_uncertainty, config.mass_uncertainty])


def model_relative_uncertainties(config: PerturbationBudgetConfig) -> Dict[str, float]:
    radiation_rel = _rss([config.area_uncertainty, config.mass_uncertainty])
    return {
        "SRP": _rss([config.srp_uncertainty, radiation_rel]),
        "Lunar Albedo": _rss([config.albedo_uncertainty, radiation_rel]),
        "Thermal IR": _thermal_relative_uncertainty(config),
        "Solid Tides": float(config.tide_uncertainty),
        "Third Body": float(config.third_body_uncertainty),
        "1PN Relativity": float(config.relativity_uncertainty),
    }


def compute_uncertainty_budget(
    config: PerturbationBudgetConfig,
    samples: Iterable[SampleState],
    forces_by_sample: Mapping[str, Mapping[str, np.ndarray]],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    rel = model_relative_uncertainties(config)

    for sample in samples:
        forces = forces_by_sample[sample.sample_id]
        model_vectors = {
            "SRP": forces.get("SRP", np.zeros(3, dtype=np.float64)),
            "Lunar Albedo": forces.get("Lunar Albedo", np.zeros(3, dtype=np.float64)),
            "Thermal IR": forces.get("Thermal IR", np.zeros(3, dtype=np.float64)),
            "Solid Tides": forces.get("Solid Tides", np.zeros(3, dtype=np.float64)),
            "Third Body": (
                forces.get("Third Body Earth", np.zeros(3, dtype=np.float64))
                + forces.get("Third Body Sun", np.zeros(3, dtype=np.float64))
                + forces.get("Earth J2 Differential", np.zeros(3, dtype=np.float64))
            ),
            "1PN Relativity": forces.get("1PN Relativity", np.zeros(3, dtype=np.float64)),
        }
        uncertainties: List[float] = []
        for model, vector in model_vectors.items():
            accel_norm = _norm(vector)
            rel_u = float(rel.get(model, 0.0))
            sigma = accel_norm * rel_u
            uncertainties.append(sigma)
            rows.append(
                {
                    "sample_id": sample.sample_id,
                    "altitude_km": sample.altitude_km,
                    "inclination_deg": sample.inclination_deg,
                    "true_anomaly_deg": sample.true_anomaly_deg,
                    "epoch_utc": sample.epoch_utc,
                    "model": model,
                    "acceleration_norm_m_s2": accel_norm,
                    "relative_uncertainty": rel_u,
                    "uncertainty_norm_m_s2": sigma,
                    "combination": "component",
                }
            )

        rows.append(
            {
                "sample_id": sample.sample_id,
                "altitude_km": sample.altitude_km,
                "inclination_deg": sample.inclination_deg,
                "true_anomaly_deg": sample.true_anomaly_deg,
                "epoch_utc": sample.epoch_utc,
                "model": "Combined Non-Gravitational RSS",
                "acceleration_norm_m_s2": float("nan"),
                "relative_uncertainty": float("nan"),
                "uncertainty_norm_m_s2": _rss(uncertainties),
                "combination": "rss",
            }
        )
        rows.append(
            {
                "sample_id": sample.sample_id,
                "altitude_km": sample.altitude_km,
                "inclination_deg": sample.inclination_deg,
                "true_anomaly_deg": sample.true_anomaly_deg,
                "epoch_utc": sample.epoch_utc,
                "model": "Combined Non-Gravitational Linear",
                "acceleration_norm_m_s2": float("nan"),
                "relative_uncertainty": float("nan"),
                "uncertainty_norm_m_s2": _linear(uncertainties),
                "combination": "linear",
            }
        )
    return rows


def _percentile(values: Sequence[float], q: float) -> float:
    finite = [float(v) for v in values if np.isfinite(v)]
    if not finite:
        return float("nan")
    return float(np.percentile(np.asarray(finite, dtype=np.float64), q))


def _combined_uncertainty_by_sample(rows: Iterable[Mapping[str, object]], combination: str) -> Dict[str, float]:
    target = "Combined Non-Gravitational RSS" if combination == "rss" else "Combined Non-Gravitational Linear"
    out: Dict[str, float] = {}
    for row in rows:
        if row.get("model") == target:
            out[str(row["sample_id"])] = float(row["uncertainty_norm_m_s2"])
    return out


def recommend_gravity_degree_by_altitude(
    config: PerturbationBudgetConfig,
    sh_rows: Iterable[Mapping[str, object]],
    uncertainty_rows: Iterable[Mapping[str, object]],
) -> List[Dict[str, object]]:
    sh_by_alt_band: Dict[tuple[float, str], List[Mapping[str, object]]] = defaultdict(list)
    for row in sh_rows:
        sh_by_alt_band[(float(row["altitude_km"]), str(row["band"]))].append(row)

    combination = "rss" if config.rss_uncertainty_combination else "linear"
    unc_by_sample = _combined_uncertainty_by_sample(uncertainty_rows, combination)
    unc_by_alt: Dict[float, List[float]] = defaultdict(list)
    for row in sh_rows:
        sid = str(row["sample_id"])
        if sid in unc_by_sample:
            unc_by_alt[float(row["altitude_km"])].append(unc_by_sample[sid])

    profiles = {
        "quick": {"q": 50.0, "fraction": max(1.0, config.recommendation_uncertainty_fraction)},
        "medium": {"q": 50.0, "fraction": config.recommendation_uncertainty_fraction},
        "high": {"q": 95.0, "fraction": 0.5 * config.recommendation_uncertainty_fraction},
    }

    rows: List[Dict[str, object]] = []
    for altitude in sorted({float(a) for a, _ in sh_by_alt_band}):
        rec: Dict[str, object] = {"altitude_km": altitude}
        limiting: Dict[str, str] = {}
        reason: Dict[str, str] = {}
        unc_stat = _percentile(unc_by_alt.get(altitude, []), 50.0)
        for profile, opts in profiles.items():
            selected = max(config.sh_degrees)
            selected_reason = "No increment met the configured thresholds."
            selected_factor = "max_degree"
            for low, high in zip(config.sh_degrees[:-1], config.sh_degrees[1:]):
                band = f"Delta SH{low}->{high}"
                band_rows = sh_by_alt_band.get((altitude, band), [])
                stat = _percentile([float(r["increment_norm_m_s2"]) for r in band_rows], float(opts["q"]))
                below_abs = bool(np.isfinite(stat) and stat <= config.recommendation_absolute_threshold_m_s2)
                below_unc = bool(np.isfinite(stat) and np.isfinite(unc_stat) and stat <= float(opts["fraction"]) * unc_stat)
                if below_abs or below_unc:
                    selected = int(low)
                    if below_abs:
                        selected_factor = "absolute_threshold"
                        selected_reason = (
                            f"{band} {opts['q']:.0f}th percentile {stat:.3e} m/s^2 is below "
                            f"{config.recommendation_absolute_threshold_m_s2:.3e} m/s^2."
                        )
                    else:
                        selected_factor = "non_grav_uncertainty"
                        selected_reason = (
                            f"{band} {opts['q']:.0f}th percentile {stat:.3e} m/s^2 is below "
                            f"{float(opts['fraction']):.3g} x combined non-grav uncertainty ({unc_stat:.3e} m/s^2)."
                        )
                    break
            rec[f"recommended_degree_{profile}"] = selected
            limiting[profile] = selected_factor
            reason[profile] = selected_reason
        rec["limiting_factor"] = "; ".join(f"{k}:{v}" for k, v in limiting.items())
        rec["reason"] = " | ".join(f"{k}: {v}" for k, v in reason.items())
        rows.append(rec)
    return rows
