from __future__ import annotations

import csv
import math
from dataclasses import replace
from pathlib import Path

import numpy as np

from lunaris.analysis.perturbation_budget.acceleration_budget import sh_increment_vectors
from lunaris.analysis.perturbation_budget.cli import main as perturbation_budget_main
from lunaris.analysis.perturbation_budget.config import PerturbationBudgetConfig
from lunaris.analysis.perturbation_budget.reporting import run_perturbation_budget
from lunaris.analysis.perturbation_budget.sampling import decompose_ric
from lunaris.analysis.perturbation_budget.uncertainty_budget import compute_uncertainty_budget
from lunaris.analysis.perturbation_budget.sampling import SampleState


def test_ric_components_reconstruct_vector_norm() -> None:
    r = np.array([1.0, 0.0, 0.0])
    v = np.array([0.0, 2.0, 0.0])
    a = np.array([1.0, 2.0, 3.0])

    radial, along, cross = decompose_ric(a, r, v)

    assert radial == 1.0
    assert along == 2.0
    assert cross == 3.0
    assert math.isclose(math.sqrt(radial**2 + along**2 + cross**2), float(np.linalg.norm(a)))


def test_sh_increment_uses_vector_difference_and_identical_degrees_are_zero() -> None:
    vectors = {
        20: np.array([1.0, 2.0, 3.0]),
        30: np.array([2.0, 5.0, 9.0]),
        60: np.array([2.0, 5.0, 9.0]),
    }

    increments = sh_increment_vectors(vectors)

    np.testing.assert_allclose(increments["Delta SH20->30"], np.array([1.0, 3.0, 6.0]))
    np.testing.assert_allclose(increments["Delta SH30->60"], np.zeros(3))


def _sample() -> SampleState:
    return SampleState(
        sample_id="s0",
        altitude_km=100.0,
        inclination_deg=0.0,
        true_anomaly_deg=0.0,
        epoch_utc="synthetic",
        r_m=np.array([1.0, 0.0, 0.0]),
        v_m_s=np.array([0.0, 1.0, 0.0]),
        sun_m=np.array([1.0, 0.0, 0.0]),
        earth_m=np.array([0.0, 1.0, 0.0]),
        geometry_source="synthetic_geometry",
    )


def _srp_uncertainty(rows: list[dict[str, object]]) -> float:
    for row in rows:
        if row["model"] == "SRP":
            return float(row["uncertainty_norm_m_s2"])
    raise AssertionError("missing SRP row")


def test_uncertainty_budget_scales_and_rss_is_not_linear() -> None:
    base_cfg = PerturbationBudgetConfig(
        altitudes_km=[100],
        inclinations_deg=[0],
        true_anomalies_deg=[0],
        epochs_utc=["synthetic"],
        sh_degrees=[2, 4],
        srp_uncertainty=0.2,
        area_uncertainty=0.0,
        mass_uncertainty=0.0,
    )
    forces = {"s0": {"SRP": np.array([1.0e-8, 0.0, 0.0])}}

    rows = compute_uncertainty_budget(base_cfg, [_sample()], forces)
    rows_doubled = compute_uncertainty_budget(replace(base_cfg, srp_uncertainty=0.4), [_sample()], forces)

    assert math.isclose(_srp_uncertainty(rows), 2.0e-9)
    assert math.isclose(_srp_uncertainty(rows_doubled), 4.0e-9)
    rss = next(float(r["uncertainty_norm_m_s2"]) for r in rows if r["model"] == "Combined Non-Gravitational RSS")
    linear = next(float(r["uncertainty_norm_m_s2"]) for r in rows if r["model"] == "Combined Non-Gravitational Linear")
    assert rss <= linear


def test_run_perturbation_budget_writes_expected_outputs(tmp_path: Path) -> None:
    cfg = PerturbationBudgetConfig(
        altitudes_km=[100],
        inclinations_deg=[0],
        true_anomalies_deg=[0],
        epochs_utc=["synthetic"],
        sh_degrees=[2, 4],
        include_albedo=False,
        include_thermal_ir=False,
        include_tides=False,
        include_third_body=False,
        include_relativity=False,
        output_dir=str(tmp_path),
    )

    result = run_perturbation_budget(cfg)

    expected = [
        result.perturbation_budget_csv,
        result.gravity_degree_sensitivity_csv,
        result.force_model_uncertainty_budget_csv,
        result.recommended_gravity_degree_by_altitude_csv,
        result.summary_md,
    ]
    for path in expected:
        assert path.exists(), path
        assert "nan" not in path.read_text(encoding="utf-8").lower()
        assert "inf" not in path.read_text(encoding="utf-8").lower()

    with result.perturbation_budget_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows
    assert {"sample_id", "force_name", "acceleration_norm_m_s2", "radial_m_s2"}.issubset(rows[0])
    assert any(row["force_name"] == "Central Lunar Gravity" for row in rows)
    assert any(row["force_name"] == "Delta SH2->4" for row in rows)


def test_cli_smoke_writes_outputs(tmp_path: Path) -> None:
    rc = perturbation_budget_main(
        [
            "--altitudes-km",
            "100",
            "--inclinations-deg",
            "0",
            "--true-anomalies-deg",
            "0",
            "--epochs",
            "synthetic",
            "--sh-degrees",
            "2,4",
            "--include-albedo",
            "off",
            "--include-thermal",
            "off",
            "--include-tides",
            "off",
            "--include-third-body",
            "off",
            "--include-relativity",
            "off",
            "--out-dir",
            str(tmp_path),
        ]
    )
    assert rc == 0
    assert (tmp_path / "perturbation_budget.csv").exists()
    assert (tmp_path / "perturbation_budget_summary.md").exists()
