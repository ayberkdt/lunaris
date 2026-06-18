from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from lunaris.validation.gravity_reference.lunaris_field_runner import run_field_validation
from lunaris.validation.gravity_reference.lunaris_trajectory_runner import (
    _ContractViolation,
    _enforce_trajectory_contract,
    run_trajectory_validation,
)
from lunaris.validation.gravity_reference.thresholds import (
    INCOMPLETE_CONTRACT,
    PASS,
    REFERENCE_GENERATION_REQUIRED,
)

_BENCH = Path(__file__).resolve().parent.parent / "validation" / "gravity_reference" / "benchmarks"
FIELD_MANIFEST = Path("validation/gravity_reference/benchmarks/field/synthetic_degree4_oracle.json")
GRAIL_FIELD_MANIFEST = _BENCH / "field" / "grail_degree120_pyshtools_oracle.json"
SOBOL_FIELD_MANIFEST = _BENCH / "field" / "grail_degree120_pyshtools_sobol.json"
TRAJECTORY_MANIFEST = _BENCH / "trajectory" / "grail_degree32_pyshtools_trajectory.json"


def test_synthetic_field_validation_end_to_end(tmp_path: Path) -> None:
    result = run_field_validation(FIELD_MANIFEST, tmp_path / "field_synthetic")

    assert result["status"] == PASS
    out = Path(result["out_dir"])
    assert (out / "resolved_manifest.json").exists()
    assert (out / "run_provenance.json").exists()
    assert (out / "validation_status.json").exists()
    assert (out / "field_metrics_summary.json").exists()
    assert (out / "field_samples.csv").exists()
    assert (out / "comparison_report.md").exists()


def test_grail_field_validation_end_to_end(tmp_path: Path) -> None:
    result = run_field_validation(GRAIL_FIELD_MANIFEST, tmp_path / "field_grail")

    assert result["status"] == PASS
    out = Path(result["out_dir"])
    assert (out / "resolved_manifest.json").exists()
    assert (out / "run_provenance.json").exists()
    assert (out / "validation_status.json").exists()
    assert (out / "field_metrics_summary.json").exists()
    assert (out / "field_samples.csv").exists()
    assert (out / "comparison_report.md").exists()


def test_grail_sobol_statistical_field_validation(tmp_path: Path) -> None:
    """Statistical field validation over thousands of Sobol + strata points.

    Far stronger than the 8-point smoke: it bounds the engine across the sphere
    and all altitude bands, and emits latitude/altitude-binned error tables.
    """
    result = run_field_validation(SOBOL_FIELD_MANIFEST, tmp_path / "field_sobol")
    assert result["status"] == PASS
    metrics = result["metrics"]
    assert metrics["n_points"] >= 2000
    # Matches pyshtools to ~machine precision even at the worst point/region.
    assert metrics["acceleration_norm_error_m_s2"]["max"] < 1e-8
    assert metrics["acceleration_norm_error_m_s2"]["p99"] < 1e-9
    region = metrics["error_by_region"]
    assert region["by_latitude_deg"] and region["by_altitude_km"]
    # Every populated band must stay within tolerance (no hidden weak region).
    for row in region["by_altitude_km"] + region["by_latitude_deg"]:
        if row["count"]:
            assert row["max"] < 1e-8, f"weak region {row['bin']}: {row['max']:.2e}"


def test_trajectory_runner_fails_closed_for_incomplete_reference(tmp_path: Path) -> None:
    manifest = tmp_path / "trajectory.json"
    manifest.write_text(
        """
{
  "schema_version": "lunaris_gravity_trajectory_reference_v1",
  "benchmark_id": "missing_external_arc",
  "reference_class": "incomplete_reference",
  "title": "Missing external trajectory placeholder",
  "reference_file": {"path": "", "format": "csv", "sha256": "", "size_bytes": 0},
  "time": {
    "initial_epoch": "2027-01-01T00:00:00",
    "time_scale": "UTC",
    "duration_s": 600.0,
    "output_step_s": 60.0
  },
  "frames": {
    "state_center": "MOON",
    "state_frame": "MOON_J2000",
    "gravity_fixed_frame": "MOON_PA",
    "comparison_frame": "MOON_J2000"
  },
  "initial_state": {
    "representation": "cartesian",
    "position_unit": "m",
    "velocity_unit": "m/s",
    "state": [1838000.0, 0.0, 0.0, 0.0, 1633.0, 0.0]
  },
  "dynamics": {"gravity_only": true},
  "comparison": {
    "epoch_alignment_tolerance_s": 0.0,
    "position_thresholds_m": {},
    "velocity_thresholds_mps": {},
    "ric_thresholds_m": {},
    "threshold_origin": "report_only"
  }
}
""",
        encoding="utf-8",
    )
    result = run_trajectory_validation(manifest, tmp_path / "trajectory")
    assert result["status"] == REFERENCE_GENERATION_REQUIRED


def test_trajectory_contract_rejects_missing_final_epoch() -> None:
    payload = {
        "time": {"duration_s": 600.0, "output_step_s": 60.0},
        "frames": {
            "state_center": "MOON",
            "state_frame": "NONROTATING_FROZEN_BODY_FIXED",
            "gravity_fixed_frame": "NONROTATING_FROZEN_BODY_FIXED",
            "comparison_frame": "NONROTATING_FROZEN_BODY_FIXED",
        },
    }
    epochs = np.arange(0.0, 600.0, 60.0, dtype=np.float64)

    with pytest.raises(_ContractViolation, match="sample count"):
        _enforce_trajectory_contract(payload, epochs)


def test_trajectory_validation_end_to_end(tmp_path: Path) -> None:
    """Lunaris must reproduce the independent gravity-only reference arc.

    The reference was generated by an external library (pyshtools) and an
    independent integrator (SciPy DOP853); the runner re-propagates with the
    Lunaris production propagator and compares. A sign/frame/kernel regression
    would diverge by kilometres and fail the committed thresholds.
    """
    result = run_trajectory_validation(TRAJECTORY_MANIFEST, tmp_path / "traj")
    assert result["status"] == PASS
    out = Path(result["out_dir"])
    for name in (
        "resolved_manifest.json",
        "run_provenance.json",
        "validation_status.json",
        "trajectory_metrics_summary.json",
        "trajectory_samples.csv",
        "comparison_report.md",
    ):
        assert (out / name).exists()
    metrics = result["metrics"]
    # Independent-tool agreement is far inside the frozen limits, and Lunaris
    # conserves the gravity-only Hamiltonian to ~machine precision.
    assert metrics["position_error_m"]["max"] < 1.0e-2
    assert metrics["lunaris_energy_drift"]["max_abs_relative_drift"] < 1.0e-10


def test_trajectory_runner_rejects_rotating_frame(tmp_path: Path) -> None:
    """A rotating body-fixed field is out of scope and must fail closed."""
    import json

    repo = _BENCH.parents[2]
    payload = json.loads(TRAJECTORY_MANIFEST.read_text(encoding="utf-8"))
    payload["frames"]["gravity_fixed_frame"] = "MOON_ME"  # != state_frame (MOON_PA)
    # Absolute paths so resolution does not depend on the temp manifest location.
    payload["gravity"]["coefficient_file"] = str(repo / payload["gravity"]["coefficient_file"])
    payload["reference_file"]["path"] = str(repo / payload["reference_file"]["path"])
    manifest = tmp_path / "rotating.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    result = run_trajectory_validation(manifest, tmp_path / "rot")
    assert result["status"] == INCOMPLETE_CONTRACT

