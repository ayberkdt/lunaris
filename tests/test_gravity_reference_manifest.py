from __future__ import annotations

import json
from pathlib import Path

import pytest

from lunaris.validation.gravity_reference.manifest import (
    HashMismatchError,
    ManifestError,
    load_field_manifest,
    load_trajectory_manifest,
)

FIELD_MANIFEST = Path("validation/gravity_reference/benchmarks/field/synthetic_degree4_oracle.json")


def _payload_with_absolute_paths() -> dict:
    payload = json.loads(FIELD_MANIFEST.read_text(encoding="utf-8"))
    payload["reference_file"]["path"] = str(
        Path("validation/gravity_reference/reference_data/field/synthetic_degree4_field_vectors.csv").resolve()
    )
    payload["gravity"]["coefficient_file"] = str(
        Path("validation/gravity_reference/reference_data/field/synthetic_degree4_coefficients.json").resolve()
    )
    return payload


def test_field_manifest_loads_and_resolves_hashes() -> None:
    manifest = load_field_manifest(FIELD_MANIFEST)
    assert manifest.payload["benchmark_id"] == "synthetic_degree4_oracle"
    assert manifest.reference_path.exists()
    assert manifest.coefficient_path.exists()


def test_field_manifest_rejects_hash_mismatch(tmp_path: Path) -> None:
    payload = _payload_with_absolute_paths()
    payload["reference_file"]["sha256"] = "0" * 64
    bad = tmp_path / "bad_manifest.json"
    bad.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(HashMismatchError):
        load_field_manifest(bad)


def test_field_manifest_rejects_missing_frame(tmp_path: Path) -> None:
    payload = _payload_with_absolute_paths()
    payload["coordinates"].pop("frame")
    bad = tmp_path / "missing_frame.json"
    bad.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ManifestError, match="coordinates.frame"):
        load_field_manifest(bad)


def test_incomplete_trajectory_manifest_is_loadable_for_fail_closed_runner(tmp_path: Path) -> None:
    payload = {
        "schema_version": "lunaris_gravity_trajectory_reference_v1",
        "benchmark_id": "missing_external_arc",
        "reference_class": "incomplete_reference",
        "title": "Missing external trajectory placeholder",
        "reference_file": {"path": "", "format": "csv", "sha256": "", "size_bytes": 0},
        "time": {
            "initial_epoch": "2027-01-01T00:00:00",
            "time_scale": "UTC",
            "duration_s": 600.0,
            "output_step_s": 60.0,
        },
        "frames": {
            "state_center": "MOON",
            "state_frame": "MOON_J2000",
            "gravity_fixed_frame": "MOON_PA",
            "comparison_frame": "MOON_J2000",
        },
        "initial_state": {
            "representation": "cartesian",
            "position_unit": "m",
            "velocity_unit": "m/s",
            "state": [1838000.0, 0.0, 0.0, 0.0, 1633.0, 0.0],
        },
        "dynamics": {"gravity_only": True},
        "comparison": {
            "epoch_alignment_tolerance_s": 0.0,
            "position_thresholds_m": {},
            "velocity_thresholds_mps": {},
            "ric_thresholds_m": {},
            "threshold_origin": "report_only",
        },
    }
    path = tmp_path / "trajectory.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    manifest = load_trajectory_manifest(path)
    assert manifest.payload["reference_class"] == "incomplete_reference"
    assert manifest.reference_path is None
