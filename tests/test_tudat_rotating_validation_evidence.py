from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNNER = (
    ROOT / "validation" / "gravity_reference" / "generators" / "trajectory" / "tudatpy_rotating"
)
EVIDENCE = (
    ROOT
    / "validation"
    / "gravity_reference"
    / "evidence"
    / "tudatpy_rotating"
    / "evidence_2026_07_19.json"
)
EVIDENCE_MATRIX = EVIDENCE.with_name("evidence_matrix_2026_07_19.json")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_tudat_rotating_scenarios_are_portable_and_match_evidence() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert evidence["status"] == "PASS"
    assert evidence["lunaris_git_dirty"] is False
    assert [run["duration_days"] for run in evidence["runs"]] == [1, 5, 30]
    assert all(run["status"] == "PASS" for run in evidence["runs"])

    for filename, days in (
        ("scenario.json", 1),
        ("scenario_5day.json", 5),
        ("scenario_30day.json", 30),
    ):
        raw = (RUNNER / filename).read_text(encoding="utf-8")
        scenario = json.loads(raw)
        assert scenario["duration_s"] == days * 86400.0
        assert scenario["gravity_degree"] == scenario["gravity_order"] == 120
        assert scenario["integration_frame"] == "J2000"
        assert scenario["gravity_fixed_frame"] == "MOON_PA"
        assert scenario["gravitational_parameter_m3_s2"] == 4902800306330.2
        assert "C:/" not in raw and "C:\\" not in raw


def test_tudat_rotating_evidence_retains_independent_convergence_margin() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    for run in evidence["runs"]:
        numerical_band = (
            run["tudat_sh_step_convergence_position_max_m"]
            + run["lunaris_sh_tolerance_convergence_position_max_m"]
        )
        assert run["sh_position_max_m"] < numerical_band
        assert run["sh_acceleration_relative_max"] < 5.0e-10
        assert run["tudat_rk4_position_refinement_ratio"] > 10.0


def test_diverse_scenarios_are_predeclared_and_physically_consistent() -> None:
    expected = {
        "scenario_polar_5day.json": (5, 120, 90.0),
        "scenario_equatorial_retrograde_5day.json": (5, 120, 180.0),
        "scenario_eccentric_low_periapsis_5day.json": (5, 120, 90.0),
        "scenario_high_altitude_retrograde_5day.json": (5, 120, 120.0),
        "scenario_degree360_high_inclination_1day.json": (1, 360, 80.0),
    }
    for filename, (days, degree, inclination_deg) in expected.items():
        raw = (RUNNER / filename).read_text(encoding="utf-8")
        scenario = json.loads(raw)
        construction = scenario["initial_state_construction"]
        state = np.asarray(scenario["initial_state_j2000_m_m_s"], dtype=np.float64)
        radius = np.linalg.norm(state[:3])
        speed_squared = float(state[3:] @ state[3:])
        mu = float(scenario["gravitational_parameter_m3_s2"])
        recovered_a = -mu / (speed_squared - 2.0 * mu / radius)

        assert scenario["schema_version"] == 2
        assert scenario["duration_s"] == days * 86400.0
        assert scenario["gravity_degree"] == scenario["gravity_order"] == degree
        assert scenario["tudat"]["coarse_step_s"] == 10.0
        assert construction["inclination_deg"] == inclination_deg
        assert np.isclose(recovered_a, construction["semi_major_axis_m"], rtol=2.0e-15)
        assert scenario["acceptance"]["rk4_observed_order_min"] == 3.2
        assert scenario["acceptance"]["rk4_observed_order_max"] == 4.8
        assert "spherical_harmonic_position_cross_max_m" in scenario["acceptance"]
        assert "C:/" not in raw and "C:\\" not in raw


def test_diverse_evidence_matrix_is_bound_to_exact_runners_and_contracts() -> None:
    evidence = json.loads(EVIDENCE_MATRIX.read_text(encoding="utf-8"))
    source_hashes = evidence["source_hashes"]
    assert evidence["status"] == "PASS"
    assert len(evidence["runs"]) == 5
    assert source_hashes["executed_tudat_runner_sha256"] == _sha256(RUNNER / "run_tudat.py")
    assert source_hashes["executed_lunaris_runner_sha256"] == _sha256(
        RUNNER / "run_lunaris.py"
    )

    for run in evidence["runs"]:
        scenario = json.loads((RUNNER / run["scenario_file"]).read_text(encoding="utf-8"))
        assert run["status"] == "PASS"
        assert run["all_checks_pass"] is True
        assert run["lunaris_git_dirty"] is False
        assert run["gravity_degree_order"] == scenario["gravity_degree"]
        assert run["sh_position_max_m"] < run["sh_position_hard_max_m"]
        assert run["sh_position_max_m"] < run["sh_position_numerical_band_m"]
        assert run["sh_velocity_max_m_s"] < run["sh_velocity_hard_max_m_s"]
        assert run["sh_velocity_max_m_s"] < run["sh_velocity_numerical_band_m_s"]
        assert run["sh_acceleration_relative_max"] < 5.0e-10
        assert 3.2 <= run["tudat_rk4_position_order"] <= 4.8
        assert 3.2 <= run["tudat_rk4_velocity_order"] <= 4.8
        assert run["coverage"]["altitude_min_m"] >= scenario["coverage_acceptance"][
            "altitude_min_m"
        ]
        assert run["coverage"]["altitude_max_m"] <= scenario["coverage_acceptance"][
            "altitude_max_m"
        ]
        assert run["coverage"]["longitude_10deg_bins_covered"] >= scenario[
            "coverage_acceptance"
        ]["longitude_10deg_bins_min"]
        latitude_max_abs = max(
            abs(run["coverage"]["latitude_min_deg"]),
            abs(run["coverage"]["latitude_max_deg"]),
        )
        assert latitude_max_abs >= scenario["coverage_acceptance"][
            "latitude_abs_max_deg_min"
        ]
        assert latitude_max_abs <= scenario["coverage_acceptance"][
            "latitude_abs_max_deg_max"
        ]
        for key in (
            "evidence_scenario_sha256",
            "comparison_summary_sha256",
            "tudat_provenance_sha256",
            "lunaris_provenance_sha256",
            "result_directory_aggregate_sha256",
        ):
            assert len(run[key]) == 64
            int(run[key], 16)


def test_failed_design_trial_is_retained_but_not_counted_as_pass() -> None:
    evidence = json.loads(EVIDENCE_MATRIX.read_text(encoding="utf-8"))
    assert len(evidence["excluded_diagnostic_trials"]) == 1
    trial = evidence["excluded_diagnostic_trials"][0]
    assert trial["status"] == "FAIL"
    assert trial["excluded_from_pass_count"] is True
    assert trial["failed_checks"] == ["coverage_longitude"]
