from __future__ import annotations

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
        "scenario_degree360_high_inclination_1day.json": (1, 360, 90.0),
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
