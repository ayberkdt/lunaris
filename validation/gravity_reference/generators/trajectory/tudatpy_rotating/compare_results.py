from __future__ import annotations

import json

import numpy as np
from validation_common import (
    RESULTS,
    assert_same_grid,
    load_scenario,
    read_trajectory_csv,
    state_error_metrics,
    write_json,
)


def load_tudat_triplet(force_model: str) -> tuple[dict, dict, dict]:
    return (
        read_trajectory_csv(RESULTS / f"tudat_{force_model}_coarse.csv"),
        read_trajectory_csv(RESULTS / f"tudat_{force_model}_baseline.csv"),
        read_trajectory_csv(RESULTS / f"tudat_{force_model}_tight.csv"),
    )


def load_lunaris_pair(force_model: str) -> tuple[dict, dict]:
    return (
        read_trajectory_csv(RESULTS / f"lunaris_{force_model}_baseline.csv"),
        read_trajectory_csv(RESULTS / f"lunaris_{force_model}_tight.csv"),
    )


def observed_order(coarse_error: float, fine_error: float) -> float:
    if coarse_error <= 0.0 or fine_error <= 0.0:
        raise ValueError("convergence errors must be positive")
    return float(np.log2(coarse_error / fine_error))


def relative_acceleration_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    absolute = np.linalg.norm(candidate - reference, axis=1)
    scale = np.linalg.norm(reference, axis=1)
    relative = absolute / scale
    return {
        "absolute_max_m_s2": float(np.max(absolute)),
        "absolute_rms_m_s2": float(np.sqrt(np.mean(absolute * absolute))),
        "relative_max": float(np.max(relative)),
        "relative_rms": float(np.sqrt(np.mean(relative * relative))),
    }


def rotation_metrics() -> dict[str, float]:
    tudat = json.loads((RESULTS / "tudat_rotation_probes.json").read_text(encoding="utf-8"))
    lunaris = json.loads((RESULTS / "lunaris_rotation_probes.json").read_text(encoding="utf-8"))
    differences = []
    orthogonality = []
    determinants = []
    for tudat_probe, lunaris_probe in zip(tudat, lunaris, strict=True):
        if tudat_probe["epoch_rel_s"] != lunaris_probe["epoch_rel_s"]:
            raise ValueError("rotation probe epochs differ")
        tudat_matrix = np.asarray(tudat_probe["j2000_to_moon_pa"], dtype=np.float64)
        lunaris_matrix = np.asarray(lunaris_probe["j2000_to_moon_pa"], dtype=np.float64)
        differences.append(np.linalg.norm(lunaris_matrix - tudat_matrix, ord="fro"))
        for matrix in (tudat_matrix, lunaris_matrix):
            orthogonality.append(np.linalg.norm(matrix @ matrix.T - np.eye(3), ord="fro"))
            determinants.append(abs(np.linalg.det(matrix) - 1.0))
    return {
        "matrix_frobenius_max": float(max(differences)),
        "orthogonality_error_max": float(max(orthogonality)),
        "determinant_error_max": float(max(determinants)),
    }


def point_mass_invariant_metrics(state: np.ndarray, mu: float) -> dict[str, float]:
    position = state[:, :3]
    velocity = state[:, 3:]
    radius = np.linalg.norm(position, axis=1)
    specific_energy = 0.5 * np.sum(velocity * velocity, axis=1) - mu / radius
    angular_momentum = np.linalg.norm(np.cross(position, velocity), axis=1)

    def drift(values: np.ndarray) -> dict[str, float]:
        scale = abs(float(values[0]))
        return {
            "initial": float(values[0]),
            "final": float(values[-1]),
            "absolute_final_drift": float(abs(values[-1] - values[0])),
            "relative_final_drift": float(abs(values[-1] - values[0]) / scale),
            "relative_peak_to_peak": float(np.ptp(values) / scale),
        }

    return {
        "specific_energy_m2_s2": drift(specific_energy),
        "specific_angular_momentum_m2_s": drift(angular_momentum),
    }


def main() -> None:
    scenario = load_scenario()
    acceptance = scenario["acceptance"]
    metrics = {
        "rotation": rotation_metrics(),
        "coverage": json.loads(
            (RESULTS / "lunaris_trajectory_coverage.json").read_text(encoding="utf-8")
        ),
        "force_models": {},
    }
    for force_model in ("point_mass", "spherical_harmonic"):
        tudat_coarse, tudat_baseline, tudat_tight = load_tudat_triplet(force_model)
        lunaris_baseline, lunaris_tight = load_lunaris_pair(force_model)
        lunaris_probe = read_trajectory_csv(RESULTS / f"lunaris_{force_model}_at_tudat_states.csv")
        assert_same_grid(
            tudat_coarse,
            tudat_baseline,
            tudat_tight,
            lunaris_baseline,
            lunaris_tight,
            lunaris_probe,
        )
        metrics["force_models"][force_model] = {
            "tudat_coarse_to_baseline_convergence": state_error_metrics(
                tudat_baseline["state"], tudat_coarse["state"]
            ),
            "tudat_step_convergence": state_error_metrics(
                tudat_tight["state"], tudat_baseline["state"]
            ),
            "lunaris_tolerance_convergence": state_error_metrics(
                lunaris_tight["state"], lunaris_baseline["state"]
            ),
            "cross_implementation_tight": state_error_metrics(
                tudat_tight["state"], lunaris_tight["state"]
            ),
            "cross_implementation_baseline_vs_lunaris_tight": state_error_metrics(
                tudat_baseline["state"], lunaris_tight["state"]
            ),
            "acceleration_at_identical_tudat_states": relative_acceleration_metrics(
                tudat_tight["acceleration"], lunaris_probe["acceleration"]
            ),
        }

        if force_model == "point_mass":
            mu = float(scenario["gravitational_parameter_m3_s2"])
            metrics["force_models"][force_model]["invariants"] = {
                "tudat_tight": point_mass_invariant_metrics(tudat_tight["state"], mu),
                "lunaris_tight": point_mass_invariant_metrics(lunaris_tight["state"], mu),
            }

    for force_model in ("point_mass", "spherical_harmonic"):
        force_metrics = metrics["force_models"][force_model]
        coarse_to_baseline = force_metrics["tudat_coarse_to_baseline_convergence"]
        baseline_to_tight = force_metrics["tudat_step_convergence"]
        position_order = observed_order(
            coarse_to_baseline["position_m"]["max"],
            baseline_to_tight["position_m"]["max"],
        )
        velocity_order = observed_order(
            coarse_to_baseline["velocity_m_s"]["max"],
            baseline_to_tight["velocity_m_s"]["max"],
        )
        force_metrics["tudat_rk4_observed_order"] = {
            "position_max": position_order,
            "velocity_max": velocity_order,
            "step_triplet_s": [
                float(scenario["tudat"]["coarse_step_s"]),
                float(scenario["tudat"]["baseline_step_s"]),
                float(scenario["tudat"]["tight_step_s"]),
            ],
        }
        force_metrics["tudat_richardson_tight_error_estimate"] = {
            "position_max_m": baseline_to_tight["position_m"]["max"] / (2.0**position_order - 1.0),
            "velocity_max_m_s": baseline_to_tight["velocity_m_s"]["max"]
            / (2.0**velocity_order - 1.0),
        }
        baseline_error = force_metrics["cross_implementation_baseline_vs_lunaris_tight"]
        tight_error = force_metrics["cross_implementation_tight"]
        force_metrics["tudat_rk4_refinement_ratio"] = {
            "position_max": baseline_error["position_m"]["max"] / tight_error["position_m"]["max"],
            "velocity_max": baseline_error["velocity_m_s"]["max"]
            / tight_error["velocity_m_s"]["max"],
            "interpretation": "ratio of dt=5 s and dt=2.5 s Tudat errors against the same tight Lunaris trajectory",
        }

    point = metrics["force_models"]["point_mass"]
    spherical = metrics["force_models"]["spherical_harmonic"]
    noise_position = (
        spherical["tudat_step_convergence"]["position_m"]["max"]
        + spherical["lunaris_tolerance_convergence"]["position_m"]["max"]
    )
    noise_velocity = (
        spherical["tudat_step_convergence"]["velocity_m_s"]["max"]
        + spherical["lunaris_tolerance_convergence"]["velocity_m_s"]["max"]
    )
    position_limit = float(acceptance["spherical_harmonic_position_cross_max_m"])
    velocity_limit = float(acceptance["spherical_harmonic_velocity_cross_max_m_s"])
    derived_limits = {
        "spherical_harmonic_position_hard_max_m": position_limit,
        "spherical_harmonic_velocity_hard_max_m_s": velocity_limit,
        "spherical_harmonic_position_numerical_band_m": noise_position,
        "spherical_harmonic_velocity_numerical_band_m_s": noise_velocity,
        "derivation": "predeclared hard cap plus independent requirement that cross error stay below raw within-tool convergence sum",
    }

    coverage = metrics["coverage"]
    coverage_acceptance = scenario["coverage_acceptance"]
    rk4_min = float(acceptance["rk4_observed_order_min"])
    rk4_max = float(acceptance["rk4_observed_order_max"])
    sh_order = spherical["tudat_rk4_observed_order"]

    checks = {
        "rotation_matrix_parity": metrics["rotation"]["matrix_frobenius_max"]
        <= float(acceptance["rotation_matrix_frobenius_max"]),
        "point_mass_acceleration_parity": point["acceleration_at_identical_tudat_states"][
            "relative_max"
        ]
        <= float(acceptance["point_mass_acceleration_relative_max"]),
        "spherical_harmonic_acceleration_parity": spherical[
            "acceleration_at_identical_tudat_states"
        ]["relative_max"]
        <= float(acceptance["spherical_harmonic_acceleration_relative_max"]),
        "point_mass_position_trajectory": point["cross_implementation_tight"]["position_m"]["max"]
        <= float(acceptance["point_mass_position_cross_max_m"]),
        "point_mass_velocity_trajectory": point["cross_implementation_tight"]["velocity_m_s"]["max"]
        <= float(acceptance["point_mass_velocity_cross_max_m_s"]),
        "spherical_harmonic_position_trajectory": spherical["cross_implementation_tight"][
            "position_m"
        ]["max"]
        <= position_limit,
        "spherical_harmonic_velocity_trajectory": spherical["cross_implementation_tight"][
            "velocity_m_s"
        ]["max"]
        <= velocity_limit,
        "spherical_harmonic_position_within_numerical_band": spherical[
            "cross_implementation_tight"
        ]["position_m"]["max"]
        <= noise_position,
        "spherical_harmonic_velocity_within_numerical_band": spherical[
            "cross_implementation_tight"
        ]["velocity_m_s"]["max"]
        <= noise_velocity,
        "tudat_rk4_position_order": rk4_min <= sh_order["position_max"] <= rk4_max,
        "tudat_rk4_velocity_order": rk4_min <= sh_order["velocity_max"] <= rk4_max,
        "coverage_altitude_floor": coverage["altitude_m"]["min"]
        >= float(coverage_acceptance["altitude_min_m"]),
        "coverage_altitude_ceiling": coverage["altitude_m"]["max"]
        <= float(coverage_acceptance["altitude_max_m"]),
        "coverage_latitude": float(coverage_acceptance["latitude_abs_max_deg_min"])
        <= coverage["body_fixed_latitude_deg"]["max_abs"]
        <= float(coverage_acceptance["latitude_abs_max_deg_max"]),
        "coverage_longitude": coverage["body_fixed_longitude_10deg_bins"]["covered"]
        >= int(coverage_acceptance["longitude_10deg_bins_min"]),
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    summary = {
        "status": status,
        "scenario": scenario["name"],
        "duration_s": scenario["duration_s"],
        "gravity_degree_order": [scenario["gravity_degree"], scenario["gravity_order"]],
        "checks": checks,
        "derived_limits": derived_limits,
        "metrics": metrics,
    }
    write_json(RESULTS / "comparison_summary.json", summary)

    report = [
        "# Lunaris–TudatPy rotating-Moon validation\n",
        f"**Overall status: {status}**\n",
        "## Contract\n",
        f"- Duration: {scenario['duration_s'] / 86400.0:g} day\n",
        f"- Gravity: {scenario['gravity_model_id']} degree/order {scenario['gravity_degree']}/{scenario['gravity_order']}\n",
        f"- Frames: {scenario['integration_frame']} integration, {scenario['gravity_fixed_frame']} coefficients\n",
        f"- GM: {scenario['gravitational_parameter_m3_s2']:.16g} m^3/s^2\n",
        f"- Radius: {scenario['reference_radius_m']:.16g} m\n",
        "- Force set: gravity only (point-mass control and rotating spherical harmonics)\n",
        "\n## Results\n",
        f"- Rotation matrix max Frobenius difference: {metrics['rotation']['matrix_frobenius_max']:.6e}\n",
        f"- Point-mass acceleration max relative difference: {point['acceleration_at_identical_tudat_states']['relative_max']:.6e}\n",
        f"- SH acceleration max relative difference at identical states: {spherical['acceleration_at_identical_tudat_states']['relative_max']:.6e}\n",
        f"- Point-mass trajectory max position difference: {point['cross_implementation_tight']['position_m']['max']:.6e} m\n",
        f"- Rotating-SH trajectory max position difference: {spherical['cross_implementation_tight']['position_m']['max']:.6e} m\n",
        f"- Rotating-SH trajectory final position difference: {spherical['cross_implementation_tight']['position_m']['final']:.6e} m\n",
        f"- Rotating-SH trajectory max velocity difference: {spherical['cross_implementation_tight']['velocity_m_s']['max']:.6e} m/s\n",
        f"- Rotating-SH Tudat RK4 position refinement ratio: {spherical['tudat_rk4_refinement_ratio']['position_max']:.3f}\n",
        f"- Rotating-SH Tudat observed RK4 position/velocity order: {sh_order['position_max']:.4f} / {sh_order['velocity_max']:.4f}\n",
        f"- Coverage: altitude {coverage['altitude_m']['min'] / 1000.0:.3f}..{coverage['altitude_m']['max'] / 1000.0:.3f} km, body-fixed max |latitude| {coverage['body_fixed_latitude_deg']['max_abs']:.3f} deg, longitude bins {coverage['body_fixed_longitude_10deg_bins']['covered']}/{coverage['body_fixed_longitude_10deg_bins']['total']}\n",
        f"- Rotating-SH final R/I/C: {json.dumps(spherical['cross_implementation_tight']['ric_final_m'])}\n",
        f"- Point-mass Tudat tight relative energy drift: {point['invariants']['tudat_tight']['specific_energy_m2_s2']['relative_final_drift']:.6e}\n",
        f"- Point-mass Lunaris tight relative energy drift: {point['invariants']['lunaris_tight']['specific_energy_m2_s2']['relative_final_drift']:.6e}\n",
        f"- Point-mass Tudat tight relative angular-momentum drift: {point['invariants']['tudat_tight']['specific_angular_momentum_m2_s']['relative_final_drift']:.6e}\n",
        f"- Point-mass Lunaris tight relative angular-momentum drift: {point['invariants']['lunaris_tight']['specific_angular_momentum_m2_s']['relative_final_drift']:.6e}\n",
        "\n## Acceptance checks\n",
    ]
    report.extend(f"- {'PASS' if passed else 'FAIL'} — {name}\n" for name, passed in checks.items())
    report.extend(
        [
            "\n## Interpretation boundary\n",
            f"This run qualifies the gravity-only, DE440 MOON_PA rotating-field path for the stated {scenario['duration_s'] / 86400.0:g}-day scenario. ",
            "It does not qualify third bodies, tides, radiation, or relativity.\n",
        ]
    )
    (RESULTS / "VALIDATION_REPORT.md").write_text("".join(report), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
