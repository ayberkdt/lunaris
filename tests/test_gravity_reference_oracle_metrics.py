from __future__ import annotations

import numpy as np
import pytest

from lunaris.validation.gravity_reference.field_metrics import compute_field_metrics
from lunaris.validation.gravity_reference.independent_field_oracle import (
    acceleration,
    geopotential,
)
from lunaris.validation.gravity_reference.normalization import full_normalization_factor
from lunaris.validation.gravity_reference.thresholds import PASS, classify_field_metrics
from lunaris.validation.gravity_reference.trajectory_metrics import (
    compute_trajectory_metrics,
    ric_basis,
    specific_energy_drift,
)

MU = 4.9048695e12
R = 1_738_000.0


def test_normalization_known_values() -> None:
    assert full_normalization_factor(0, 0) == pytest.approx(1.0)
    assert full_normalization_factor(1, 0) == pytest.approx(np.sqrt(3.0))
    assert full_normalization_factor(2, 0) == pytest.approx(np.sqrt(5.0))


def test_independent_oracle_point_mass_anchor() -> None:
    c = np.zeros((1, 1), dtype=np.float64)
    s = np.zeros_like(c)
    pos = np.array([2.0e6, -3.0e5, 4.0e5], dtype=np.float64)
    u = geopotential(pos, mu_m3_s2=MU, reference_radius_m=R, c_coeffs=c, s_coeffs=s, degree=0)
    a = acceleration(pos, mu_m3_s2=MU, reference_radius_m=R, c_coeffs=c, s_coeffs=s, degree=0)
    radius = np.linalg.norm(pos)
    assert u == pytest.approx(MU / radius, rel=1e-12)
    np.testing.assert_allclose(a, -MU * pos / radius**3, rtol=1e-8, atol=0.0)


def test_field_metrics_and_threshold_classification_pass() -> None:
    point_ids = ["p0", "p1"]
    positions = np.array([[R + 100.0, 0.0, 0.0], [0.0, R + 200.0, 0.0]])
    ref_u = np.array([1.0, 2.0])
    got_u = ref_u + np.array([1e-12, -1e-12])
    ref_a = np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    got_a = ref_a + np.array([[1e-12, 0.0, 0.0], [0.0, -1e-12, 0.0]])

    metrics, rows = compute_field_metrics(
        point_ids=point_ids,
        positions_m=positions,
        reference_potential_m2_s2=ref_u,
        reference_acceleration_m_s2=ref_a,
        lunaris_potential_m2_s2=got_u,
        lunaris_acceleration_m_s2=got_a,
        reference_radius_m=R,
    )
    assert len(rows) == 2
    status = classify_field_metrics(
        metrics,
        {
            "threshold_origin": "convergence_derived_and_frozen",
            "absolute_tolerances": {"potential_m2_s2": 1e-9, "acceleration_norm_m_s2": 1e-9},
            "relative_tolerances": {"potential": 1e-9, "acceleration_norm": 1e-9},
        },
    )
    assert status["status"] == PASS


def test_specific_energy_drift_detects_nonconservation() -> None:
    r0 = R + 100_000.0
    v = float(np.sqrt(MU / r0))

    def point_mass_potential(rv: np.ndarray) -> float:
        return MU / float(np.linalg.norm(rv))

    # Three points on the same circular orbit share one specific energy.
    circular = np.array(
        [[r0, 0.0, 0.0, 0.0, v, 0.0],
         [0.0, r0, 0.0, -v, 0.0, 0.0],
         [-r0, 0.0, 0.0, 0.0, -v, 0.0]],
        dtype=np.float64,
    )
    drift = specific_energy_drift(circular, point_mass_potential)
    assert drift["initial_specific_energy_m2_s2"] == pytest.approx(-0.5 * MU / r0)
    assert drift["max_abs_relative_drift"] < 1e-14

    # A velocity glitch breaks conservation and must be flagged.
    glitched = circular.copy()
    glitched[1, 3] *= 1.001
    assert specific_energy_drift(glitched, point_mass_potential)["max_abs_relative_drift"] > 1e-4


def test_ric_basis_and_trajectory_metrics() -> None:
    ref = np.array([
        [2.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 2.0, 0.0, -1.0, 0.0, 0.0],
    ])
    got = ref.copy()
    got[:, 0] += 0.1
    basis = ric_basis(ref[0])
    np.testing.assert_allclose(basis, np.eye(3), atol=1e-12)
    metrics = compute_trajectory_metrics(ref, got)
    assert metrics["position_error_m"]["max"] == pytest.approx(0.1)

