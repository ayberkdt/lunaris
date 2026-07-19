from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numba
import numpy as np
import scipy
from validation_common import (
    RESULTS,
    SCENARIO,
    load_scenario,
    read_trajectory_csv,
    repo_root,
    sha256_file,
    write_json,
    write_trajectory_csv,
)

from lunaris.common.type_defs import (
    PerturbationFlags,
    PropagatorConfig,
    SpacecraftProps,
    TimeConfig,
)
from lunaris.core.dynamics import DynamicsEngine
from lunaris.core.propagation.propagator import propagate
from lunaris.physics.ephemeris import EphemerisManager, build_tables
from lunaris.physics.spherical_harmonics import GravityModel

REPO = repo_root()


def git(args: list[str]) -> str | None:
    process = subprocess.run(
        ["git", *args], cwd=REPO, capture_output=True, text=True, check=False, timeout=10
    )
    return process.stdout.strip() if process.returncode == 0 else None


def quaternion_to_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = (float(value) for value in q)
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def build_engine(scenario: dict, model: GravityModel, force_model: str, ephemeris_step_s: float):
    if force_model == "point_mass":
        return DynamicsEngine(
            sc_props=SpacecraftProps(mass_kg=100.0, area_m2=1.0, cr=1.0),
            flags=PerturbationFlags(enable_sh=False),
            gravity_model=model,
            ephem_manager=None,
            allow_identity_rotation=True,
        ), None
    tables = build_tables(
        start_utc=str(scenario["start_epoch_text"]),
        duration_s=float(scenario["duration_s"]),
        output_dt_s=float(ephemeris_step_s),
        kernels=tuple(str(path) for path in scenario["kernel_files"]),
        inertial_frame="J2000",
        fixed_frame="MOON_PA",
        observer="MOON",
        include_third_body=False,
        clear_kernels_after=True,
        clean_kernels_before=True,
        auto_fix_kernel_paths=False,
        need_moon_fixed_rotation=True,
    )
    manager = EphemerisManager.from_tables(tables)
    return DynamicsEngine(
        sc_props=SpacecraftProps(mass_kg=100.0, area_m2=1.0, cr=1.0),
        flags=PerturbationFlags(enable_sh=True),
        gravity_model=model,
        ephem_manager=manager,
        strict_fixed_degree=True,
        allow_identity_rotation=False,
    ), tables


def run_case(
    scenario: dict,
    model: GravityModel,
    *,
    force_model: str,
    label: str,
    rtol: float,
    atol: float,
    max_step_s: float,
    ephemeris_step_s: float,
) -> tuple[dict, DynamicsEngine, object | None]:
    engine, tables = build_engine(scenario, model, force_model, ephemeris_step_s)
    output_step = float(scenario["output_step_s"])
    config = PropagatorConfig(
        method="DOP853",
        rtol=float(rtol),
        atol=float(atol),
        user_max_step_s=float(max_step_s),
    )
    time_config = TimeConfig(
        duration_s=float(scenario["duration_s"]), output_dt_s=output_step, t0_s=0.0
    )
    started = time.perf_counter()
    result = propagate(
        engine,
        np.asarray(scenario["initial_state_j2000_m_m_s"], dtype=np.float64),
        config,
        time_cfg=time_config,
    )
    runtime_s = time.perf_counter() - started
    epochs = np.asarray(result.t, dtype=np.float64)
    states = np.asarray(result.y, dtype=np.float64)
    if states.shape[0] == 6 and states.shape[1] == epochs.size:
        states = states.T
    if states.shape != (epochs.size, 6):
        raise RuntimeError(f"unexpected Lunaris state shape {states.shape}")
    rhs = engine.build_rhs()
    accelerations = np.vstack(
        [np.asarray(rhs(float(t), state)[3:6]) for t, state in zip(epochs, states, strict=True)]
    )
    write_trajectory_csv(
        RESULTS / f"lunaris_{force_model}_{label}.csv", epochs, states, accelerations
    )
    details = {
        "force_model": force_model,
        "integrator": "SciPy DOP853 via lunaris.core.propagation.propagator.propagate",
        "rtol": float(rtol),
        "atol": float(atol),
        "max_step_s": float(max_step_s),
        "ephemeris_rotation_table_step_s": None
        if force_model == "point_mass"
        else float(ephemeris_step_s),
        "runtime_s": runtime_s,
        "output_samples": int(epochs.size),
    }
    return details, engine, tables


def write_force_probes(scenario: dict, engine: DynamicsEngine, force_model: str) -> None:
    tudat = read_trajectory_csv(RESULTS / f"tudat_{force_model}_tight.csv")
    rhs = engine.build_rhs()
    accelerations = np.vstack(
        [
            np.asarray(rhs(float(t), state)[3:6], dtype=np.float64)
            for t, state in zip(tudat["epoch_rel_s"], tudat["state"], strict=True)
        ]
    )
    write_trajectory_csv(
        RESULTS / f"lunaris_{force_model}_at_tudat_states.csv",
        tudat["epoch_rel_s"],
        tudat["state"],
        accelerations,
    )


def write_trajectory_coverage(scenario: dict, tables: object) -> None:
    trajectory = read_trajectory_csv(RESULTS / "lunaris_spherical_harmonic_tight.csv")
    epochs = trajectory["epoch_rel_s"]
    states = trajectory["state"]
    table_dt = float(tables.dt_s)
    indices = np.rint(epochs / table_dt).astype(np.int64)
    if not np.allclose(indices * table_dt, epochs, rtol=0.0, atol=1.0e-9):
        raise ValueError("coverage epochs do not align with the tight rotation table")
    matrices = np.stack([quaternion_to_matrix(tables.q_i2f_tab[index]) for index in indices])
    fixed_positions = np.einsum("nij,nj->ni", matrices, states[:, :3])
    radii = np.linalg.norm(states[:, :3], axis=1)
    fixed_radii = np.linalg.norm(fixed_positions, axis=1)
    latitudes = np.degrees(np.arcsin(fixed_positions[:, 2] / fixed_radii))
    longitudes = np.mod(np.degrees(np.arctan2(fixed_positions[:, 1], fixed_positions[:, 0])), 360.0)
    angular_momentum = np.cross(states[:, :3], states[:, 3:])
    inclinations = np.degrees(
        np.arccos(
            np.clip(
                angular_momentum[:, 2] / np.linalg.norm(angular_momentum, axis=1),
                -1.0,
                1.0,
            )
        )
    )
    longitude_counts, _ = np.histogram(longitudes, bins=np.arange(0.0, 361.0, 10.0))
    write_json(
        RESULTS / "lunaris_trajectory_coverage.json",
        {
            "samples": int(epochs.size),
            "altitude_m": {
                "min": float(np.min(radii) - float(scenario["reference_radius_m"])),
                "max": float(np.max(radii) - float(scenario["reference_radius_m"])),
            },
            "body_fixed_latitude_deg": {
                "min": float(np.min(latitudes)),
                "max": float(np.max(latitudes)),
                "max_abs": float(np.max(np.abs(latitudes))),
            },
            "body_fixed_longitude_10deg_bins": {
                "covered": int(np.count_nonzero(longitude_counts)),
                "total": int(longitude_counts.size),
            },
            "j2000_inclination_deg": {
                "min": float(np.min(inclinations)),
                "max": float(np.max(inclinations)),
            },
        },
    )
def main() -> None:
    scenario = load_scenario()
    RESULTS.mkdir(parents=True, exist_ok=True)
    gravity_hash = sha256_file(scenario["gravity_file"])
    if gravity_hash != scenario["expected_gravity_sha256"]:
        raise ValueError(f"gravity hash mismatch: {gravity_hash}")
    model = GravityModel.from_file(
        str(scenario["gravity_file"]), requested_degree=int(scenario["gravity_degree"])
    )
    if model.GM_m3s2 != float(scenario["gravitational_parameter_m3_s2"]):
        raise ValueError("Lunaris loaded a different gravity GM")
    if model.R_ref_m != float(scenario["reference_radius_m"]):
        raise ValueError("Lunaris loaded a different reference radius")
    if model.metadata.coefficient_frame != "MOON_PA":
        raise ValueError(f"unexpected coefficient frame {model.metadata.coefficient_frame}")
    if model.metadata.tide_system != "tide_free":
        raise ValueError(f"unexpected tide system {model.metadata.tide_system}")

    settings = scenario["lunaris"]
    cases = []
    tight_engines = {}
    tight_tables = None
    for force_model in ("point_mass", "spherical_harmonic"):
        baseline, _engine, _tables = run_case(
            scenario,
            model,
            force_model=force_model,
            label="baseline",
            rtol=float(settings["baseline_rtol"]),
            atol=float(settings["baseline_atol"]),
            max_step_s=float(settings["baseline_max_step_s"]),
            ephemeris_step_s=float(settings["baseline_ephemeris_step_s"]),
        )
        cases.append(baseline)
        tight, engine, tables = run_case(
            scenario,
            model,
            force_model=force_model,
            label="tight",
            rtol=float(settings["tight_rtol"]),
            atol=float(settings["tight_atol"]),
            max_step_s=float(settings["tight_max_step_s"]),
            ephemeris_step_s=float(settings["tight_ephemeris_step_s"]),
        )
        cases.append(tight)
        tight_engines[force_model] = engine
        if force_model == "spherical_harmonic":
            tight_tables = tables

    for force_model, engine in tight_engines.items():
        write_force_probes(scenario, engine, force_model)

    if tight_tables is None:
        raise RuntimeError("missing spherical-harmonic ephemeris table")
    write_trajectory_coverage(scenario, tight_tables)
    rotation_probes = []
    for offset in (0.0, float(scenario["duration_s"]) / 2.0, float(scenario["duration_s"])):
        index = int(round(offset / float(tight_tables.dt_s)))
        rotation_probes.append(
            {
                "epoch_rel_s": offset,
                "j2000_to_moon_pa": quaternion_to_matrix(tight_tables.q_i2f_tab[index]).tolist(),
            }
        )
    write_json(RESULTS / "lunaris_rotation_probes.json", rotation_probes)

    status_short = git(["status", "--short"])
    provenance = {
        "implementation": "Lunaris",
        "implementation_path": "lunaris.core.propagation.propagator.propagate + DynamicsEngine",
        "git_commit": git(["rev-parse", "HEAD"]),
        "git_dirty": bool(status_short),
        "git_status_short": status_short,
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "numba_version": numba.__version__,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "dtype": "float64",
        "backend": "cpu",
        "scenario_path": str(SCENARIO),
        "scenario_sha256": sha256_file(SCENARIO),
        "runner_sha256": sha256_file(Path(__file__)),
        "exact_command": "<repo-python> run_lunaris.py",
        "gravity_file": str(scenario["gravity_file"]),
        "gravity_sha256": gravity_hash,
        "gravity_metadata": {
            "model_id": model.metadata.model_id,
            "normalization": model.metadata.normalization,
            "coefficient_frame": model.metadata.coefficient_frame,
            "tide_system": model.metadata.tide_system,
            "source_gm_m3_s2": model.metadata.source_gm_m3s2,
            "source_radius_m": model.metadata.source_radius_m,
        },
        "kernel_files": [
            {"path": str(path), "sha256": sha256_file(path)} for path in scenario["kernel_files"]
        ],
        "frame_rotation": "SPICE-sampled J2000 -> MOON_PA quaternion table, SLERP at integrator stages",
        "cases": cases,
    }
    write_json(RESULTS / "lunaris_provenance.json", provenance)
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
