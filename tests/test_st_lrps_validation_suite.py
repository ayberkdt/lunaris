"""Task 6 — ST-LRPS validation suite (field metrics across split policies)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from lunaris.surrogate.st_lrps.data.dataset_parameters import R_MOON_SI
from lunaris.surrogate.st_lrps.evaluation.validation_suite import (
    SPLIT_KIND,
    attach_orbit_validation,
    compute_field_metrics,
    write_validation_report,
)


def test_compute_field_metrics_known_error():
    n = 200
    # Points spread along +x at lunar radius so r_hat = +x.
    xyz = np.zeros((n, 3))
    xyz[:, 0] = R_MOON_SI + np.linspace(100e3, 400e3, n)
    a_true = np.tile([1.0, 0.0, 0.0], (n, 1))
    # Constant error purely cross-radial (y), magnitude 1e-3.
    a_pred = a_true + np.tile([0.0, 1.0e-3, 0.0], (n, 1))
    u_true = np.zeros(n)
    u_pred = u_true + 2.0  # constant potential error -> RMSE = 2.0

    m = compute_field_metrics(xyz, u_true, a_true, u_pred, a_pred, r_ref_m=R_MOON_SI)

    assert m["count"] == n
    assert m["residual_potential_rmse_m2_s2"] == pytest.approx(2.0, rel=1e-9)
    assert m["residual_accel_rmse_m_s2"] == pytest.approx(1.0e-3, rel=1e-6)
    assert m["relative_accel_error_pct_median"] == pytest.approx(0.1, rel=1e-6)
    # Error is perpendicular to the radial direction.
    assert m["radial_error_rms_m_s2"] < 1e-12
    assert m["cross_radial_error_rms_m_s2"] == pytest.approx(1.0e-3, rel=1e-6)
    # angle = atan(1e-3 / 1) in degrees.
    assert m["angular_accel_error_deg_median"] == pytest.approx(np.degrees(np.arctan(1e-3)), rel=1e-3)
    assert m["accel_error_p99_m_s2"] == pytest.approx(1.0e-3, rel=1e-6)
    assert isinstance(m["altitude_binned_error"], list) and m["altitude_binned_error"]
    assert isinstance(m["latitude_binned_error"], list)
    assert isinstance(m["longitude_binned_error"], list)


def test_compute_field_metrics_empty():
    empty = np.zeros((0, 3))
    m = compute_field_metrics(empty, np.zeros(0), empty, np.zeros(0), empty)
    assert m["count"] == 0


def test_write_report_separates_generalization_kinds(tmp_path):
    # Hand-built report mixing the split kinds; the markdown must label each
    # section so interpolation is never conflated with generalization.
    report = {
        "schema_version": 1,
        "model_dir": "run",
        "dataset_path": "data.h5",
        "split_seed": 1,
        "n_rows": 100,
        "field_validation": {
            "seeded_random": {"kind": "interpolation", "residual_accel_rmse_m_s2": 1e-6, "count": 10},
            "spatial_block": {"kind": "spatial_generalization", "residual_accel_rmse_m_s2": 2e-6, "count": 10},
            "ood_low_altitude": {"kind": "altitude_extrapolation", "residual_accel_rmse_m_s2": 5e-6, "count": 10},
        },
        "orbit_validation": None,
    }
    paths = write_validation_report(report, tmp_path)
    md = paths["md"].read_text(encoding="utf-8")
    assert "Interpolation" in md
    assert "Spatial generalization" in md
    assert "Altitude extrapolation" in md
    assert "Trajectory propagation" in md
    loaded = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert loaded["field_validation"]["spatial_block"]["kind"] == "spatial_generalization"


def test_attach_orbit_validation():
    report = {"orbit_validation": None}
    attach_orbit_validation(report, {"final_pos_err_km": 1.2, "rms_pos_err_km": 0.8})
    assert report["orbit_validation"]["final_pos_err_km"] == 1.2


def test_split_kind_labels_cover_default_policies():
    for policy in ("seeded_random", "spatial_block", "ood_low_altitude", "ood_high_altitude"):
        assert policy in SPLIT_KIND


# --- Integration smoke (needs torch + a real run dir) ----------------------

torch = pytest.importorskip("torch")

from lunaris.surrogate.st_lrps.evaluation.validation_suite import run_field_validation
from dataset_pipeline_test_utils import write_toy_contract_h5
from st_lrps_contract_test_utils import make_contract_run


def test_run_field_validation_smoke(tmp_path):
    run = make_contract_run(tmp_path, degree_min=20, degree_max=60, alt_min_km=100.0, alt_max_km=500.0)
    dataset = write_toy_contract_h5(tmp_path / "cloud.h5", n=400, alt_min_km=100.0, alt_max_km=500.0)

    report = run_field_validation(
        run["run_dir"], dataset, split_seed=7, val_fraction=0.2, device="cpu"
    )
    field = report["field_validation"]
    for policy in ("seeded_random", "altitude_stratified", "spatial_block", "ood_low_altitude", "ood_high_altitude"):
        assert policy in field, policy
    # Random + spatial + OOD-low should all evaluate without error.
    for policy in ("seeded_random", "spatial_block", "ood_low_altitude"):
        assert "error" not in field[policy], field[policy]
        assert field[policy]["count"] > 0
        assert field[policy]["kind"] == SPLIT_KIND[policy]
        assert np.isfinite(field[policy]["residual_accel_rmse_m_s2"])

    paths = write_validation_report(report, tmp_path / "report")
    assert paths["json"].exists() and paths["md"].exists()
