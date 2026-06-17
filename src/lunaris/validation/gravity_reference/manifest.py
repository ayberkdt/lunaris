"""Strict manifests for external gravity-reference validation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .normalization import require_supported_normalization
from .source_hashes import load_json, resolve_manifest_path, sha256_file

FIELD_SCHEMA_VERSION = "lunaris_gravity_field_reference_v1"
TRAJECTORY_SCHEMA_VERSION = "lunaris_gravity_trajectory_reference_v1"

FIELD_REFERENCE_CLASSES = {
    "published_field_vectors",
    "independent_high_precision_field_oracle",
}

TRAJECTORY_REFERENCE_CLASSES = {
    "published_truth_trajectory",
    "external_tool_generated_trajectory",
    "mission_ephemeris_non_gravity_only",
    "incomplete_reference",
}

STATUS_INCOMPLETE_CLASSES = {
    "mission_ephemeris_non_gravity_only",
    "incomplete_reference",
}

THRESHOLD_ORIGINS = {
    "published",
    "official_software_validation",
    "convergence_derived_and_frozen",
    "finite_difference_oracle_truncation_limit",
    "report_only",
}

TIME_SCALES = {"UTC", "TAI", "TT", "TDB", "ET"}


class ManifestError(ValueError):
    """Raised when a reference manifest is scientifically incomplete or invalid."""


class HashMismatchError(ManifestError):
    """Raised when manifest-declared hashes do not match local bytes."""


@dataclass(frozen=True, slots=True)
class ResolvedFieldManifest:
    path: Path
    payload: dict[str, Any]
    reference_path: Path
    coefficient_path: Path


@dataclass(frozen=True, slots=True)
class ResolvedTrajectoryManifest:
    path: Path
    payload: dict[str, Any]
    reference_path: Path | None


def _mapping(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ManifestError(f"Manifest field '{key}' must be an object.")
    return value


def _required(mapping: dict[str, Any], key: str, *, where: str) -> Any:
    if key not in mapping:
        raise ManifestError(f"Missing required manifest field '{where}.{key}'.")
    return mapping[key]


def _required_str(mapping: dict[str, Any], key: str, *, where: str) -> str:
    value = _required(mapping, key, where=where)
    if not isinstance(value, str) or not value.strip():
        raise ManifestError(f"Manifest field '{where}.{key}' must be a non-empty string.")
    return value


def _required_bool(mapping: dict[str, Any], key: str, *, where: str) -> bool:
    value = _required(mapping, key, where=where)
    if not isinstance(value, bool):
        raise ManifestError(f"Manifest field '{where}.{key}' must be boolean.")
    return value


def _required_number(mapping: dict[str, Any], key: str, *, where: str) -> float:
    value = _required(mapping, key, where=where)
    if not isinstance(value, int | float) or not math.isfinite(float(value)):
        raise ManifestError(f"Manifest field '{where}.{key}' must be finite numeric.")
    return float(value)


def _required_int(mapping: dict[str, Any], key: str, *, where: str) -> int:
    value = _required(mapping, key, where=where)
    if not isinstance(value, int) or value < 0:
        raise ManifestError(f"Manifest field '{where}.{key}' must be a non-negative integer.")
    return int(value)


def _verify_hash(path: Path, expected: str | None, *, label: str) -> None:
    if not expected or not isinstance(expected, str):
        raise ManifestError(f"{label} is missing a required SHA-256 hash.")
    actual = sha256_file(path)
    if actual.lower() != expected.lower():
        raise HashMismatchError(f"{label} hash mismatch: expected {expected}, got {actual}")


def _validate_source(payload: dict[str, Any]) -> None:
    source = _mapping(payload, "source")
    for key in ("organization", "project", "license_note"):
        _required_str(source, key, where="source")


def _validate_gravity(payload: dict[str, Any], *, manifest_path: Path) -> Path:
    gravity = _mapping(payload, "gravity")
    for key in (
        "model_name",
        "model_release",
        "coefficient_file",
        "coefficient_sha256",
        "normalization",
        "tide_system",
        "coefficient_frame",
    ):
        _required_str(gravity, key, where="gravity")
    degree = _required_int(gravity, "degree", where="gravity")
    order = _required_int(gravity, "order", where="gravity")
    if order > degree:
        raise ManifestError("gravity.order cannot exceed gravity.degree.")
    require_supported_normalization(str(gravity["normalization"]))
    if _required_number(gravity, "mu_m3_s2", where="gravity") <= 0.0:
        raise ManifestError("gravity.mu_m3_s2 must be positive.")
    if _required_number(gravity, "reference_radius_m", where="gravity") <= 0.0:
        raise ManifestError("gravity.reference_radius_m must be positive.")
    coefficient_path = resolve_manifest_path(
        str(gravity["coefficient_file"]), manifest_path=manifest_path, must_exist=True
    )
    _verify_hash(coefficient_path, str(gravity["coefficient_sha256"]), label="coefficient_file")
    return coefficient_path


def load_field_manifest(
    path: str | Path,
    *,
    verify_hashes: bool = True,
) -> ResolvedFieldManifest:
    """Load and validate a field-reference manifest."""
    manifest_path = Path(path).resolve()
    payload = load_json(manifest_path)
    if not isinstance(payload, dict):
        raise ManifestError("Field manifest must be a JSON object.")
    if payload.get("schema_version") != FIELD_SCHEMA_VERSION:
        raise ManifestError(f"Unsupported field schema_version: {payload.get('schema_version')!r}")
    _required_str(payload, "benchmark_id", where="")
    reference_class = _required_str(payload, "reference_class", where="")
    if reference_class not in FIELD_REFERENCE_CLASSES:
        raise ManifestError(f"Unsupported field reference_class: {reference_class}")
    _required_str(payload, "title", where="")
    _validate_source(payload)

    ref_file = _mapping(payload, "reference_file")
    for key in ("path", "format", "sha256"):
        _required_str(ref_file, key, where="reference_file")
    if ref_file["format"] not in {"csv", "json", "npz"}:
        raise ManifestError(f"Unsupported field reference_file.format: {ref_file['format']}")
    reference_path = resolve_manifest_path(
        str(ref_file["path"]), manifest_path=manifest_path, must_exist=True
    )
    if verify_hashes:
        _verify_hash(reference_path, str(ref_file["sha256"]), label="reference_file")

    coordinates = _mapping(payload, "coordinates")
    if _required_str(coordinates, "center", where="coordinates") != "MOON":
        raise ManifestError("coordinates.center must be MOON.")
    _required_str(coordinates, "frame", where="coordinates")
    if _required_str(coordinates, "position_unit", where="coordinates") != "m":
        raise ManifestError("coordinates.position_unit must be m.")

    coefficient_path = _validate_gravity(payload, manifest_path=manifest_path) if verify_hashes else Path()

    quantities = _mapping(payload, "quantities")
    _required_bool(quantities, "potential", where="quantities")
    _required_bool(quantities, "acceleration", where="quantities")
    _required_bool(quantities, "gradient_or_hessian", where="quantities")
    if _required_str(quantities, "potential_sign_convention", where="quantities") != "positive_geodesy_mu_over_r":
        raise ManifestError("Unsupported potential sign convention.")
    if _required_str(quantities, "acceleration_sign_convention", where="quantities") != "a_equals_positive_gradient_U":
        raise ManifestError("Unsupported acceleration sign convention.")

    comparison = _mapping(payload, "comparison")
    origin = _required_str(comparison, "threshold_origin", where="comparison")
    if origin not in THRESHOLD_ORIGINS:
        raise ManifestError(f"Unsupported comparison.threshold_origin: {origin}")
    for key in ("absolute_tolerances", "relative_tolerances"):
        if not isinstance(comparison.get(key), dict):
            raise ManifestError(f"comparison.{key} must be an object.")

    return ResolvedFieldManifest(
        path=manifest_path,
        payload=payload,
        reference_path=reference_path,
        coefficient_path=coefficient_path,
    )


def load_trajectory_manifest(
    path: str | Path,
    *,
    verify_hashes: bool = True,
) -> ResolvedTrajectoryManifest:
    """Load and validate a trajectory-reference manifest."""
    manifest_path = Path(path).resolve()
    payload = load_json(manifest_path)
    if not isinstance(payload, dict):
        raise ManifestError("Trajectory manifest must be a JSON object.")
    if payload.get("schema_version") != TRAJECTORY_SCHEMA_VERSION:
        raise ManifestError(
            f"Unsupported trajectory schema_version: {payload.get('schema_version')!r}"
        )
    _required_str(payload, "benchmark_id", where="")
    reference_class = _required_str(payload, "reference_class", where="")
    if reference_class not in TRAJECTORY_REFERENCE_CLASSES:
        raise ManifestError(f"Unsupported trajectory reference_class: {reference_class}")
    _required_str(payload, "title", where="")

    reference_path: Path | None = None
    ref_file = _mapping(payload, "reference_file")
    ref_path_value = ref_file.get("path")
    if isinstance(ref_path_value, str) and ref_path_value.strip():
        reference_path = resolve_manifest_path(
            ref_path_value, manifest_path=manifest_path, must_exist=reference_class not in STATUS_INCOMPLETE_CLASSES
        )
        if verify_hashes and reference_path.exists():
            _verify_hash(reference_path, str(ref_file.get("sha256", "")), label="trajectory reference_file")

    time = _mapping(payload, "time")
    _required_str(time, "initial_epoch", where="time")
    scale = _required_str(time, "time_scale", where="time")
    if scale not in TIME_SCALES:
        raise ManifestError(f"Unsupported time.time_scale: {scale}")
    if _required_number(time, "duration_s", where="time") < 0.0:
        raise ManifestError("time.duration_s must be non-negative.")
    if _required_number(time, "output_step_s", where="time") <= 0.0:
        raise ManifestError("time.output_step_s must be positive.")

    frames = _mapping(payload, "frames")
    for key in ("state_center", "state_frame", "gravity_fixed_frame", "comparison_frame"):
        _required_str(frames, key, where="frames")
    if frames["state_center"] != "MOON":
        raise ManifestError("frames.state_center must be MOON.")

    initial_state = _mapping(payload, "initial_state")
    if _required_str(initial_state, "representation", where="initial_state") != "cartesian":
        raise ManifestError("initial_state.representation must be cartesian.")
    if _required_str(initial_state, "position_unit", where="initial_state") != "m":
        raise ManifestError("initial_state.position_unit must be m.")
    if _required_str(initial_state, "velocity_unit", where="initial_state") != "m/s":
        raise ManifestError("initial_state.velocity_unit must be m/s.")
    state = initial_state.get("state")
    if not isinstance(state, list) or len(state) != 6:
        raise ManifestError("initial_state.state must contain six Cartesian values.")
    if not all(isinstance(v, int | float) and math.isfinite(float(v)) for v in state):
        raise ManifestError("initial_state.state contains non-finite values.")

    if reference_class not in STATUS_INCOMPLETE_CLASSES and verify_hashes:
        _validate_gravity(payload, manifest_path=manifest_path)

    dynamics = _mapping(payload, "dynamics")
    if reference_class not in STATUS_INCOMPLETE_CLASSES:
        if _required_bool(dynamics, "gravity_only", where="dynamics") is not True:
            raise ManifestError("Complete trajectory references must be gravity-only.")
        for key in (
            "third_body",
            "solar_radiation_pressure",
            "relativity",
            "solid_tides",
            "albedo",
            "thermal_ir",
            "maneuvers",
            "thrust",
            "empirical_accelerations",
        ):
            if _required_bool(dynamics, key, where="dynamics"):
                raise ManifestError(f"Complete gravity-only trajectory has dynamics.{key}=true.")

    comparison = _mapping(payload, "comparison")
    origin = _required_str(comparison, "threshold_origin", where="comparison")
    if origin not in THRESHOLD_ORIGINS:
        raise ManifestError(f"Unsupported comparison.threshold_origin: {origin}")

    return ResolvedTrajectoryManifest(
        path=manifest_path,
        payload=payload,
        reference_path=reference_path,
    )

