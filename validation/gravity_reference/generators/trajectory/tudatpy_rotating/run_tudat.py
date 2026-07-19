from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import tudatpy
from tudatpy.dynamics import environment_setup, propagation_setup, simulator
from tudatpy.interface import spice
from validation_common import (
    RESULTS,
    SCENARIO,
    load_scenario,
    sha256_file,
    write_json,
    write_trajectory_csv,
)


def load_jggrx_independently(
    path: str | Path, degree: int
) -> tuple[float, float, np.ndarray, np.ndarray, dict[str, float | int]]:
    """Parse the PDS SHADR file without using any Lunaris loader code."""
    coefficients_path = Path(path)
    cosine = np.zeros((degree + 1, degree + 1), dtype=np.float64)
    sine = np.zeros_like(cosine)
    cosine[0, 0] = 1.0
    count = 0
    with coefficients_path.open("r", encoding="ascii") as stream:
        header = [part.strip() for part in next(stream).split(",")]
        radius_m = float(header[0]) * 1.0e3
        gravitational_parameter_m3_s2 = float(header[1]) * 1.0e9
        file_degree = int(header[3])
        file_order = int(header[4])
        normalization_state = int(header[5])
        for line in stream:
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 4:
                continue
            n = int(parts[0])
            m = int(parts[1])
            if n > degree:
                break
            c_nm = float(parts[2])
            s_nm = float(parts[3])
            if not np.isfinite(c_nm) or not np.isfinite(s_nm):
                raise ValueError(f"non-finite coefficient at degree/order {n}/{m}")
            cosine[n, m] = c_nm
            sine[n, m] = s_nm
            count += 1
    expected_count = (degree + 1) * (degree + 2) // 2 - 1
    if count != expected_count:
        raise ValueError(f"loaded {count} coefficient rows; expected {expected_count}")
    if normalization_state != 1:
        raise ValueError(f"JGGRX normalization state is {normalization_state}, expected 1")
    return (
        gravitational_parameter_m3_s2,
        radius_m,
        cosine,
        sine,
        {
            "file_degree": file_degree,
            "file_order": file_order,
            "normalization_state": normalization_state,
            "loaded_rows": count,
        },
    )


def create_bodies(
    scenario: dict,
    gravitational_parameter: float,
    reference_radius: float,
    cosine: np.ndarray,
    sine: np.ndarray,
):
    body_settings = environment_setup.get_default_body_settings(["Moon"], "Moon", "J2000")
    moon = body_settings.get("Moon")
    moon.gravity_field_settings = environment_setup.gravity_field.spherical_harmonic(
        gravitational_parameter,
        reference_radius,
        cosine,
        sine,
        "MOON_PA",
    )
    moon.rotation_model_settings = environment_setup.rotation_model.spice(
        "J2000", "MOON_PA", "MOON_PA"
    )
    bodies = environment_setup.create_system_of_bodies(body_settings)
    bodies.create_empty_body("Vehicle")
    bodies.get("Vehicle").mass = 100.0
    return bodies


def run_case(
    scenario: dict,
    bodies,
    *,
    force_model: str,
    step_s: float,
    output_name: str,
) -> dict[str, float | int | str]:
    degree = int(scenario["gravity_degree"])
    if force_model == "point_mass":
        acceleration = propagation_setup.acceleration.point_mass_gravity()
    elif force_model == "spherical_harmonic":
        acceleration = propagation_setup.acceleration.spherical_harmonic_gravity(degree, degree)
    else:
        raise ValueError(force_model)

    acceleration_settings = {"Vehicle": {"Moon": [acceleration]}}
    acceleration_models = propagation_setup.create_acceleration_models(
        bodies, acceleration_settings, ["Vehicle"], ["Moon"]
    )
    start_epoch = float(scenario["start_epoch_tdb_j2000_s"])
    duration = float(scenario["duration_s"])
    integrator_settings = propagation_setup.integrator.runge_kutta_fixed_step(
        time_step=float(step_s),
        coefficient_set=propagation_setup.integrator.CoefficientSets.rk_4,
    )
    termination = propagation_setup.propagator.time_termination(
        start_epoch + duration, terminate_exactly_on_final_condition=True
    )
    output_variables = [propagation_setup.dependent_variable.total_acceleration("Vehicle")]
    propagator_settings = propagation_setup.propagator.translational(
        central_bodies=["Moon"],
        acceleration_models=acceleration_models,
        bodies_to_integrate=["Vehicle"],
        initial_states=np.asarray(scenario["initial_state_j2000_m_m_s"], dtype=np.float64),
        initial_time=start_epoch,
        integrator_settings=integrator_settings,
        termination_settings=termination,
        output_variables=output_variables,
    )
    output_step = float(scenario["output_step_s"])
    save_frequency = int(round(output_step / float(step_s)))
    if save_frequency < 1 or not np.isclose(
        save_frequency * float(step_s), output_step, rtol=0.0, atol=1e-12
    ):
        raise ValueError("output_step_s must be an integer multiple of the Tudat RK4 step")
    propagator_settings.processing_settings.results_save_frequency_in_steps = save_frequency
    started = time.perf_counter()
    dynamics_simulator = simulator.create_dynamics_simulator(bodies, propagator_settings)
    runtime_s = time.perf_counter() - started

    state_history = dynamics_simulator.propagation_results.state_history
    dependent_history = dynamics_simulator.propagation_results.dependent_variable_history
    epochs = np.asarray(sorted(state_history), dtype=np.float64)
    states = np.vstack([np.asarray(state_history[epoch], dtype=np.float64) for epoch in epochs])
    accelerations = np.vstack(
        [np.asarray(dependent_history[epoch], dtype=np.float64)[:3] for epoch in epochs]
    )
    relative = epochs - start_epoch
    requested = np.arange(0.0, duration + 0.5 * output_step, output_step, dtype=np.float64)
    if relative.shape != requested.shape or not np.allclose(
        relative, requested, rtol=0.0, atol=2e-7
    ):
        mismatch = (
            float("inf")
            if relative.shape != requested.shape
            else float(np.max(np.abs(relative - requested)))
        )
        raise RuntimeError(f"Tudat output grid mismatch: max {mismatch} s")
    write_trajectory_csv(
        RESULTS / output_name,
        requested,
        states,
        accelerations,
    )
    return {
        "force_model": force_model,
        "integrator": "classical_rk4_fixed_step",
        "step_s": float(step_s),
        "runtime_s": runtime_s,
        "internal_steps_saved": int(epochs.size),
        "output_samples": int(requested.size),
    }


def main() -> None:
    scenario = load_scenario()
    RESULTS.mkdir(parents=True, exist_ok=True)
    gravity_path = Path(scenario["gravity_file"])
    actual_hash = sha256_file(gravity_path)
    if actual_hash != scenario["expected_gravity_sha256"]:
        raise ValueError(f"gravity hash mismatch: {actual_hash}")
    mu, radius, cosine, sine, parser_metadata = load_jggrx_independently(
        gravity_path, int(scenario["gravity_degree"])
    )
    if mu != float(scenario["gravitational_parameter_m3_s2"]):
        raise ValueError(f"gravity GM mismatch: {mu}")
    if radius != float(scenario["reference_radius_m"]):
        raise ValueError(f"gravity radius mismatch: {radius}")

    spice.clear_kernels()
    for kernel in scenario["kernel_files"]:
        spice.load_kernel(str(kernel))
    converted_epoch = float(
        spice.convert_date_string_to_ephemeris_time(scenario["start_epoch_text"])
    )
    if converted_epoch != float(scenario["start_epoch_tdb_j2000_s"]):
        raise ValueError(f"SPICE epoch mismatch: {converted_epoch}")

    probes = []
    for offset in (0.0, float(scenario["duration_s"]) / 2.0, float(scenario["duration_s"])):
        matrix = np.asarray(
            spice.compute_rotation_matrix_between_frames(
                "J2000", "MOON_PA", converted_epoch + offset
            ),
            dtype=np.float64,
        )
        probes.append({"epoch_rel_s": offset, "j2000_to_moon_pa": matrix.tolist()})
    write_json(RESULTS / "tudat_rotation_probes.json", probes)

    bodies = create_bodies(scenario, mu, radius, cosine, sine)
    cases = []
    for force_model in ("point_mass", "spherical_harmonic"):
        cases.append(
            run_case(
                scenario,
                bodies,
                force_model=force_model,
                step_s=float(scenario["tudat"]["baseline_step_s"]),
                output_name=f"tudat_{force_model}_baseline.csv",
            )
        )
        cases.append(
            run_case(
                scenario,
                bodies,
                force_model=force_model,
                step_s=float(scenario["tudat"]["tight_step_s"]),
                output_name=f"tudat_{force_model}_tight.csv",
            )
        )

    provenance = {
        "implementation": "TudatPy",
        "implementation_version": getattr(tudatpy, "__version__", "unknown"),
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "dtype": "float64",
        "backend": "cpu",
        "scenario_path": str(SCENARIO),
        "scenario_sha256": sha256_file(SCENARIO),
        "runner_sha256": sha256_file(Path(__file__)),
        "exact_command": "micromamba run --root-prefix .mamba-root --prefix .tudat-env python run_tudat.py",
        "gravity_file": str(gravity_path),
        "gravity_sha256": actual_hash,
        "gravity_parser": "independent local PDS SHADR parser in run_tudat.py",
        "gravity_parser_metadata": parser_metadata,
        "gravitational_parameter_m3_s2": mu,
        "reference_radius_m": radius,
        "kernel_files": [
            {"path": str(path), "sha256": sha256_file(path)} for path in scenario["kernel_files"]
        ],
        "frame_rotation": "Tudat SPICE rotation model J2000 -> MOON_PA evaluated at integrator stages",
        "cases": cases,
    }
    write_json(RESULTS / "tudat_provenance.json", provenance)
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
