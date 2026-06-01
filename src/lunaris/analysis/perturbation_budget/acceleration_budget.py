"""Instantaneous acceleration contribution calculations."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Dict, Iterable, List, Mapping, Tuple

import numpy as np

from lunaris.common.constants import (
    AU,
    MU_EARTH,
    MU_MOON,
    MU_SUN,
    P_SUN_1AU,
    R_EARTH_MEAN,
    R_MOON,
    R_MOON_MEAN,
)
from lunaris.common.type_defs import SpacecraftProps
from lunaris.physics.relativity_effects import calc_schwarzschild_accel
from lunaris.physics.solar_effects import accel_srp
from lunaris.physics.solid_tides import calc_solid_tide_accel
from lunaris.physics.spherical_harmonics import GravityModel, compute_point_mass_acceleration
from lunaris.physics.surface_effects import AlbedoConfig, ThermalConfig, albedo_accel, thermal_accel
from lunaris.physics.third_body_effects import (
    EarthJ2Params,
    calc_3rd_body_accel,
    calc_j2_oblate_diff_accel,
)

from .config import PerturbationBudgetConfig
from .sampling import SampleState, decompose_ric


@dataclass(frozen=True, slots=True)
class GravityModelInfo:
    model: GravityModel
    source: str
    warning: str


def _norm(vec: Iterable[float]) -> float:
    return float(np.linalg.norm(np.asarray(vec, dtype=np.float64)))


def _safe_ratio(num: float, den: float) -> float:
    if not np.isfinite(num) or not np.isfinite(den) or abs(den) <= 0.0:
        return float("nan")
    return float(num / den)


def synthetic_gravity_model(max_degree: int) -> GravityModel:
    """Build deterministic toy harmonics for self-contained smoke runs.

    The coefficients are not a lunar gravity product. They only provide a
    smooth high-degree spectrum so degree-sensitivity machinery can be tested
    without bundled external data.
    """
    degree = max(2, int(max_degree))
    c = np.zeros((degree + 1, degree + 1), dtype=np.float64)
    s = np.zeros_like(c)
    for n in range(2, degree + 1):
        amp = 2.0e-4 / (n * n)
        c[n, 0] = ((-1.0) ** n) * amp
        c[n, min(2, n)] = 0.35 * amp
        if n >= 3:
            s[n, min(3, n)] = -0.2 * amp
    return GravityModel.from_arrays(degree, float(R_MOON), float(MU_MOON), c, s)


def load_gravity_model_for_budget(config: PerturbationBudgetConfig) -> GravityModelInfo:
    max_degree = max(config.sh_degrees)
    if config.gravity_model_path:
        model = GravityModel.from_file(config.gravity_model_path, requested_degree=max_degree)
        return GravityModelInfo(model=model, source=str(config.gravity_model_path), warning="")
    if not config.use_synthetic_geometry_fallback:
        raise ValueError("gravity_model_path is required when synthetic fallback is disabled")
    return GravityModelInfo(
        model=synthetic_gravity_model(max_degree),
        source="synthetic_gravity_coefficients",
        warning=(
            "No gravity_model_path was provided. Synthetic coefficients were used for "
            "self-contained analysis; do not interpret degree recommendations as lunar truth."
        ),
    )


def central_gravity(r_m: np.ndarray) -> np.ndarray:
    ax, ay, az = compute_point_mass_acceleration(float(r_m[0]), float(r_m[1]), float(r_m[2]), float(MU_MOON))
    return np.array((ax, ay, az), dtype=np.float64)


def gravity_vectors_by_degree(model: GravityModel, sample: SampleState, degrees: Iterable[int]) -> Dict[int, np.ndarray]:
    return {int(degree): model.accel_fixed(sample.r_m, degree=int(degree)) for degree in degrees}


def non_gravity_vectors(config: PerturbationBudgetConfig, sample: SampleState) -> Dict[str, np.ndarray]:
    vectors: Dict[str, np.ndarray] = {}
    sc = SpacecraftProps(
        mass_kg=float(config.spacecraft_mass_kg),
        area_m2=float(config.spacecraft_area_m2),
        cr=float(config.srp_coefficient),
    )

    if config.include_third_body:
        vectors["Third Body Earth"] = calc_3rd_body_accel(sample.r_m, sample.earth_m, MU_EARTH)
        vectors["Third Body Sun"] = calc_3rd_body_accel(sample.r_m, sample.sun_m, MU_SUN)
        if config.include_earth_j2:
            vectors["Earth J2 Differential"] = calc_j2_oblate_diff_accel(
                sample.r_m,
                sample.earth_m,
                mu_body=MU_EARTH,
                params=EarthJ2Params(),
            )

    if config.include_srp:
        vectors["SRP"] = np.array(
            accel_srp(
                float(sample.r_m[0]),
                float(sample.r_m[1]),
                float(sample.r_m[2]),
                float(sample.sun_m[0]),
                float(sample.sun_m[1]),
                float(sample.sun_m[2]),
                float(sample.earth_m[0]),
                float(sample.earth_m[1]),
                float(sample.earth_m[2]),
                float(R_MOON_MEAN),
                float(R_EARTH_MEAN),
                float(AU),
                float(P_SUN_1AU),
                float(config.srp_coefficient),
                float(config.spacecraft_area_m2),
                float(config.spacecraft_mass_kg),
                False,
                False,
            ),
            dtype=np.float64,
        )

    if config.include_albedo:
        vectors["Lunar Albedo"] = albedo_accel(
            sample.r_m,
            sample.sun_m,
            sc,
            AlbedoConfig(
                albedo_const=float(config.albedo_const),
                facet_lat_count=int(config.albedo_facet_lat_count),
                facet_lon_count=int(config.albedo_facet_lon_count),
                max_facets=max(1, int(config.albedo_facet_lat_count) * int(config.albedo_facet_lon_count)),
            ),
            enable_eclipse=False,
            R_moon=float(R_MOON_MEAN),
        )

    if config.include_thermal_ir:
        vectors["Thermal IR"] = thermal_accel(
            sample.r_m,
            sample.sun_m,
            sc,
            ThermalConfig(
                temperature_K=float(config.thermal_temperature_K),
                facet_lat_count=int(config.thermal_facet_lat_count),
                facet_lon_count=int(config.thermal_facet_lon_count),
                max_facets=max(1, int(config.thermal_facet_lat_count) * int(config.thermal_facet_lon_count)),
            ),
            enable_eclipse=False,
            R_moon=float(R_MOON_MEAN),
        )

    if config.include_tides:
        tide_earth = calc_solid_tide_accel(sample.r_m, sample.earth_m, mu_body=MU_EARTH, r_ref_m=R_MOON)
        tide_sun = calc_solid_tide_accel(sample.r_m, sample.sun_m, mu_body=MU_SUN, r_ref_m=R_MOON)
        vectors["Solid Tides Earth"] = tide_earth
        vectors["Solid Tides Sun"] = tide_sun
        vectors["Solid Tides"] = tide_earth + tide_sun

    if config.include_relativity:
        vectors["1PN Relativity"] = calc_schwarzschild_accel(sample.r_m, sample.v_m_s, float(MU_MOON))

    return vectors


def sh_increment_vectors(sh_vectors: Mapping[int, np.ndarray]) -> Dict[str, np.ndarray]:
    degrees = sorted(int(d) for d in sh_vectors)
    out: Dict[str, np.ndarray] = {}
    for lo, hi in zip(degrees[:-1], degrees[1:]):
        out[f"Delta SH{lo}->{hi}"] = np.asarray(sh_vectors[hi], dtype=np.float64) - np.asarray(sh_vectors[lo], dtype=np.float64)
    return out


def vector_row(
    sample: SampleState,
    force_name: str,
    vector: np.ndarray,
    *,
    central_norm: float,
    srp_norm: float,
    selected_sh_increment_norm: float,
) -> Dict[str, object]:
    radial, along, cross = decompose_ric(vector, sample.r_m, sample.v_m_s)
    norm = _norm(vector)
    return {
        "sample_id": sample.sample_id,
        "altitude_km": sample.altitude_km,
        "inclination_deg": sample.inclination_deg,
        "true_anomaly_deg": sample.true_anomaly_deg,
        "epoch_utc": sample.epoch_utc,
        "geometry_source": sample.geometry_source,
        "force_name": force_name,
        "acceleration_norm_m_s2": norm,
        "radial_m_s2": radial,
        "along_track_m_s2": along,
        "cross_track_m_s2": cross,
        "ratio_to_central": _safe_ratio(norm, central_norm),
        "ratio_to_srp": _safe_ratio(norm, srp_norm),
        "ratio_to_selected_sh_increment": _safe_ratio(norm, selected_sh_increment_norm),
    }


def compute_acceleration_budget(
    config: PerturbationBudgetConfig,
    samples: Iterable[SampleState],
    gravity_info: GravityModelInfo,
) -> Tuple[List[Dict[str, object]], Dict[str, Dict[int, np.ndarray]], Dict[str, Dict[str, np.ndarray]]]:
    rows: List[Dict[str, object]] = []
    sh_by_sample: Dict[str, Dict[int, np.ndarray]] = {}
    forces_by_sample: Dict[str, Dict[str, np.ndarray]] = {}
    selected_band = ""
    if len(config.sh_degrees) >= 2:
        selected_band = f"Delta SH{config.sh_degrees[-2]}->{config.sh_degrees[-1]}"

    for sample in samples:
        central = central_gravity(sample.r_m)
        sh_vectors = gravity_vectors_by_degree(gravity_info.model, sample, config.sh_degrees)
        increments = sh_increment_vectors(sh_vectors)
        non_grav = non_gravity_vectors(config, sample)
        selected_norm = _norm(increments[selected_band]) if selected_band in increments else float("nan")
        srp_norm = _norm(non_grav.get("SRP", np.zeros(3, dtype=np.float64)))
        central_norm = _norm(central)

        force_vectors: Dict[str, np.ndarray] = {"Central Lunar Gravity": central}
        for degree in config.sh_degrees:
            force_vectors[f"Gravity SH{degree}"] = sh_vectors[int(degree)]
        force_vectors.update(increments)
        force_vectors.update(non_grav)

        sh_by_sample[sample.sample_id] = sh_vectors
        forces_by_sample[sample.sample_id] = force_vectors

        for name, vec in force_vectors.items():
            rows.append(
                vector_row(
                    sample,
                    name,
                    np.asarray(vec, dtype=np.float64),
                    central_norm=central_norm,
                    srp_norm=srp_norm,
                    selected_sh_increment_norm=selected_norm,
                )
            )
    return rows, sh_by_sample, forces_by_sample


def force_norm(forces: Mapping[str, np.ndarray], *names: str) -> float:
    total = np.zeros(3, dtype=np.float64)
    for name in names:
        if name in forces:
            total += np.asarray(forces[name], dtype=np.float64)
    return _norm(total)
