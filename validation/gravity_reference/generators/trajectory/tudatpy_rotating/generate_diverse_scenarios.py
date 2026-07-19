from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import spiceypy as spice
from validation_common import load_scenario, repo_root

SCENARIOS = (
    {
        "filename": "scenario_polar_5day.json",
        "name": "lunaris_tudat_jggrx120_moon_pa_polar_5day",
        "epoch": "2025-04-01 00:00:00 TDB",
        "duration_s": 432000.0,
        "degree": 120,
        "periapsis_altitude_m": 100000.0,
        "apoapsis_altitude_m": 100000.0,
        "inclination_deg": 90.0,
        "raan_deg": 0.0,
        "argument_of_periapsis_deg": 0.0,
        "true_anomaly_deg": 0.0,
        "position_cap_m": 0.05,
        "velocity_cap_m_s": 5.0e-5,
        "coverage": (10000.0, 250000.0, 80.0, 90.1, 30),
    },
    {
        "filename": "scenario_equatorial_retrograde_5day.json",
        "name": "lunaris_tudat_jggrx120_moon_pa_equatorial_retrograde_5day",
        "epoch": "2026-01-15 00:00:00 TDB",
        "duration_s": 432000.0,
        "degree": 120,
        "periapsis_altitude_m": 100000.0,
        "apoapsis_altitude_m": 100000.0,
        "inclination_deg": 180.0,
        "raan_deg": 45.0,
        "argument_of_periapsis_deg": 0.0,
        "true_anomaly_deg": 0.0,
        "position_cap_m": 0.05,
        "velocity_cap_m_s": 5.0e-5,
        "coverage": (10000.0, 250000.0, 0.0, 15.0, 30),
    },
    {
        "filename": "scenario_eccentric_low_periapsis_5day.json",
        "name": "lunaris_tudat_jggrx120_moon_pa_eccentric_low_periapsis_5day",
        "epoch": "2027-06-01 00:00:00 TDB",
        "duration_s": 432000.0,
        "degree": 120,
        "periapsis_altitude_m": 50000.0,
        "apoapsis_altitude_m": 500000.0,
        "inclination_deg": 90.0,
        "raan_deg": 120.0,
        "argument_of_periapsis_deg": 30.0,
        "true_anomaly_deg": 0.0,
        "position_cap_m": 0.05,
        "velocity_cap_m_s": 5.0e-5,
        "coverage": (10000.0, 650000.0, 80.0, 90.1, 30),
    },
    {
        "filename": "scenario_high_altitude_retrograde_5day.json",
        "name": "lunaris_tudat_jggrx120_moon_pa_high_altitude_retrograde_5day",
        "epoch": "2030-09-01 00:00:00 TDB",
        "duration_s": 432000.0,
        "degree": 120,
        "periapsis_altitude_m": 450000.0,
        "apoapsis_altitude_m": 550000.0,
        "inclination_deg": 120.0,
        "raan_deg": 210.0,
        "argument_of_periapsis_deg": 60.0,
        "true_anomaly_deg": 180.0,
        "position_cap_m": 0.05,
        "velocity_cap_m_s": 5.0e-5,
        "coverage": (350000.0, 650000.0, 45.0, 75.0, 30),
    },
    {
        "filename": "scenario_degree360_high_inclination_1day.json",
        "name": "lunaris_tudat_jggrx360_moon_pa_high_inclination_1day",
        "epoch": "2028-02-01 00:00:00 TDB",
        "duration_s": 86400.0,
        "degree": 360,
        "periapsis_altitude_m": 70000.0,
        "apoapsis_altitude_m": 70000.0,
        "inclination_deg": 80.0,
        "raan_deg": 75.0,
        "argument_of_periapsis_deg": 0.0,
        "true_anomaly_deg": 0.0,
        "position_cap_m": 0.02,
        "velocity_cap_m_s": 2.0e-5,
        "coverage": (10000.0, 200000.0, 70.0, 85.0, 30),
    },
)


def rotation_3(angle_rad: float) -> np.ndarray:
    cosine = np.cos(angle_rad)
    sine = np.sin(angle_rad)
    return np.asarray(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def rotation_1(angle_rad: float) -> np.ndarray:
    cosine = np.cos(angle_rad)
    sine = np.sin(angle_rad)
    return np.asarray(
        [[1.0, 0.0, 0.0], [0.0, cosine, -sine], [0.0, sine, cosine]],
        dtype=np.float64,
    )


def keplerian_vectors_in_basis(
    *,
    mu: float,
    periapsis_radius_m: float,
    apoapsis_radius_m: float,
    inclination_deg: float,
    raan_deg: float,
    argument_of_periapsis_deg: float,
    true_anomaly_deg: float,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    semi_major_axis = 0.5 * (periapsis_radius_m + apoapsis_radius_m)
    eccentricity = (apoapsis_radius_m - periapsis_radius_m) / (
        apoapsis_radius_m + periapsis_radius_m
    )
    true_anomaly = np.radians(true_anomaly_deg)
    semilatus_rectum = semi_major_axis * (1.0 - eccentricity * eccentricity)
    radius = semilatus_rectum / (1.0 + eccentricity * np.cos(true_anomaly))
    position_pqw = np.asarray([radius * np.cos(true_anomaly), radius * np.sin(true_anomaly), 0.0])
    velocity_pqw = np.sqrt(mu / semilatus_rectum) * np.asarray(
        [-np.sin(true_anomaly), eccentricity + np.cos(true_anomaly), 0.0]
    )
    pqw_to_fixed = (
        rotation_3(np.radians(raan_deg))
        @ rotation_1(np.radians(inclination_deg))
        @ rotation_3(np.radians(argument_of_periapsis_deg))
    )
    return (
        pqw_to_fixed @ position_pqw,
        pqw_to_fixed @ velocity_pqw,
        semi_major_axis,
        eccentricity,
    )


def build_scenario(base: dict, definition: dict) -> dict:
    scenario = copy.deepcopy(base)
    radius = float(base["reference_radius_m"])
    position_fixed, velocity_fixed, semi_major_axis, eccentricity = keplerian_vectors_in_basis(
        mu=float(base["gravitational_parameter_m3_s2"]),
        periapsis_radius_m=radius + float(definition["periapsis_altitude_m"]),
        apoapsis_radius_m=radius + float(definition["apoapsis_altitude_m"]),
        inclination_deg=float(definition["inclination_deg"]),
        raan_deg=float(definition["raan_deg"]),
        argument_of_periapsis_deg=float(definition["argument_of_periapsis_deg"]),
        true_anomaly_deg=float(definition["true_anomaly_deg"]),
    )
    epoch = float(spice.str2et(definition["epoch"]))
    fixed_to_j2000 = np.asarray(spice.pxform("MOON_PA", "J2000", epoch), dtype=np.float64)
    state = np.concatenate([fixed_to_j2000 @ position_fixed, fixed_to_j2000 @ velocity_fixed])
    scenario.update(
        {
            "schema_version": 2,
            "name": definition["name"],
            "description": "Independent rotating-Moon gravity-only trajectory validation with predeclared diverse geometry",
            "start_epoch_text": definition["epoch"],
            "start_epoch_tdb_j2000_s": epoch,
            "duration_s": float(definition["duration_s"]),
            "gravity_degree": int(definition["degree"]),
            "gravity_order": int(definition["degree"]),
            "initial_state_j2000_m_m_s": [float(value) for value in state],
            "initial_state_construction": {
                "basis": "instantaneous MOON_PA axes",
                "vector_transform": "same SPICE 3x3 MOON_PA-to-J2000 rotation applied to position and inertial velocity components",
                "semi_major_axis_m": semi_major_axis,
                "eccentricity": eccentricity,
                "inclination_deg": float(definition["inclination_deg"]),
                "raan_deg": float(definition["raan_deg"]),
                "argument_of_periapsis_deg": float(definition["argument_of_periapsis_deg"]),
                "true_anomaly_deg": float(definition["true_anomaly_deg"]),
            },
        }
    )
    scenario["tudat"]["coarse_step_s"] = 10.0
    scenario["acceptance"].update(
        {
            "spherical_harmonic_position_cross_max_m": float(definition["position_cap_m"]),
            "spherical_harmonic_velocity_cross_max_m_s": float(definition["velocity_cap_m_s"]),
            "rk4_observed_order_min": 3.2,
            "rk4_observed_order_max": 4.8,
        }
    )
    for obsolete in (
        "trajectory_noise_multiplier",
        "trajectory_position_floor_m",
        "trajectory_velocity_floor_m_s",
    ):
        scenario["acceptance"].pop(obsolete, None)
    altitude_min, altitude_max, latitude_min, latitude_max, longitude_bins = definition["coverage"]
    scenario["coverage_acceptance"] = {
        "altitude_min_m": altitude_min,
        "altitude_max_m": altitude_max,
        "latitude_abs_max_deg_min": latitude_min,
        "latitude_abs_max_deg_max": latitude_max,
        "longitude_10deg_bins_min": longitude_bins,
    }
    return scenario


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=True)

    root = repo_root()
    base = load_scenario()
    base["gravity_file"] = str(Path(base["gravity_file"]).relative_to(root)).replace("\\", "/")
    base["gravity_label"] = str(Path(base["gravity_label"]).relative_to(root)).replace("\\", "/")
    base["kernel_files"] = [
        str(Path(path).relative_to(root)).replace("\\", "/") for path in base["kernel_files"]
    ]
    spice.kclear()
    try:
        for kernel in load_scenario()["kernel_files"]:
            spice.furnsh(str(kernel))
        for definition in SCENARIOS:
            scenario = build_scenario(base, definition)
            (output / definition["filename"]).write_text(
                json.dumps(scenario, indent=2) + "\n", encoding="utf-8"
            )
    finally:
        spice.kclear()


if __name__ == "__main__":
    main()
