"""Run Lunaris field-level validation from an immutable manifest."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from lunaris.physics.spherical_harmonics import GravityModel
from lunaris.validation.gravity_reference.field_metrics import compute_field_metrics
from lunaris.validation.gravity_reference.field_reference_io import (
    load_field_reference_csv,
    write_field_samples_csv,
)
from lunaris.validation.gravity_reference.independent_field_oracle import coefficients_from_json
from lunaris.validation.gravity_reference.manifest import ResolvedFieldManifest, load_field_manifest
from lunaris.validation.gravity_reference.reporting import field_report_markdown
from lunaris.validation.gravity_reference.source_hashes import (
    atomic_write_json,
    atomic_write_text,
    load_json,
    sha256_file,
    utc_now_iso,
)
from lunaris.validation.gravity_reference.thresholds import classify_field_metrics


def _git_value(args: list[str], cwd: Path) -> str | None:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def _run_provenance(manifest: ResolvedFieldManifest, out_dir: Path) -> dict[str, Any]:
    repo = manifest.path
    for parent in (manifest.path.parent, *manifest.path.parents):
        if (parent / "pyproject.toml").exists():
            repo = parent
            break
    status_short = _git_value(["status", "--short"], repo)
    return {
        "created_at_utc": utc_now_iso(),
        "manifest_path": str(manifest.path),
        "manifest_sha256": sha256_file(manifest.path),
        "reference_file_sha256": sha256_file(manifest.reference_path),
        "coefficient_file_sha256": sha256_file(manifest.coefficient_path),
        "output_dir": str(out_dir),
        "git_commit": _git_value(["rev-parse", "HEAD"], repo),
        "git_dirty": bool(status_short),
        "git_status_short": status_short,
        "python_version": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "implementation_path": "lunaris.physics.spherical_harmonics.GravityModel",
        "dtype": "float64",
        "backend": "cpu",
    }


def _load_gravity_model(manifest: ResolvedFieldManifest) -> GravityModel:
    payload = manifest.payload
    gravity = payload["gravity"]
    degree = int(gravity["degree"])
    coefficient_path = manifest.coefficient_path
    if coefficient_path.suffix.lower() == ".json":
        fixture = load_json(coefficient_path)
        fixture_degree, r_ref, mu, c, s = coefficients_from_json(fixture)
        if fixture_degree < degree:
            raise ValueError(
                f"Coefficient fixture degree {fixture_degree} is smaller than manifest degree {degree}."
            )
        model = GravityModel.from_arrays(degree, r_ref, mu, c, s)
    else:
        model = GravityModel.from_file(str(coefficient_path), requested_degree=degree)

    if model.degree_max != degree:
        raise ValueError(f"Loaded degree {model.degree_max} does not match manifest degree {degree}.")
    if abs(float(model.GM_m3s2) - float(gravity["mu_m3_s2"])) > max(
        1e-6, abs(float(gravity["mu_m3_s2"])) * 1e-12
    ):
        raise ValueError("Loaded gravity GM does not match manifest.")
    if abs(float(model.R_ref_m) - float(gravity["reference_radius_m"])) > 1e-6:
        raise ValueError("Loaded gravity reference radius does not match manifest.")
    return model


def run_field_validation(manifest_path: str | Path, out_dir: str | Path) -> dict[str, Any]:
    """Run a field benchmark and write JSON/CSV/Markdown outputs."""
    manifest = load_field_manifest(manifest_path)
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    reference = load_field_reference_csv(manifest.reference_path)
    model = _load_gravity_model(manifest)

    positions = np.asarray(reference["positions_m"], dtype=np.float64)
    potentials = np.zeros(positions.shape[0], dtype=np.float64)
    accelerations = np.zeros_like(positions)
    degree = int(manifest.payload["gravity"]["degree"])
    for i, pos in enumerate(positions):
        potentials[i] = model.potential_fixed(pos, degree=degree)
        accelerations[i] = model.accel_fixed(pos, degree=degree)

    metrics, rows = compute_field_metrics(
        point_ids=list(reference["point_ids"]),
        positions_m=positions,
        reference_potential_m2_s2=reference["potential_m2_s2"],
        reference_acceleration_m_s2=reference["acceleration_m_s2"],
        lunaris_potential_m2_s2=potentials,
        lunaris_acceleration_m_s2=accelerations,
        reference_radius_m=float(manifest.payload["gravity"]["reference_radius_m"]),
    )
    status = classify_field_metrics(metrics, manifest.payload["comparison"])
    provenance = _run_provenance(manifest, out)

    atomic_write_json(out / "resolved_manifest.json", manifest.payload)
    atomic_write_json(out / "run_provenance.json", provenance)
    atomic_write_json(out / "validation_status.json", status)
    atomic_write_json(out / "field_metrics_summary.json", metrics)
    write_field_samples_csv(out / "field_samples.csv", rows)
    atomic_write_text(
        out / "comparison_report.md",
        field_report_markdown(
            manifest=manifest.payload,
            status=status,
            metrics=metrics,
            provenance=provenance,
        ),
    )
    return {
        "status": status["status"],
        "out_dir": str(out),
        "metrics": metrics,
        "provenance": provenance,
    }


def run_field_validation_json(manifest_path: str | Path, out_dir: str | Path) -> str:
    """Run a field benchmark and return a compact JSON summary."""
    return json.dumps(run_field_validation(manifest_path, out_dir), sort_keys=True, default=str)

