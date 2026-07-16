"""Full-resolution post-propagation analysis for one Lunaris mission run."""

from __future__ import annotations

import importlib.metadata
import math
import platform
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

from lunaris.analysis.contracts import (
    ANALYSIS_SCHEMA_VERSION,
    REPORT_SCHEMA_VERSION,
    AnalysisEvent,
    ForceContribution,
    MetricValue,
    OrbitAnalysisResult,
    OrbitSeries,
)
from lunaris.analysis.postprocess import compute_history
from lunaris.common.constants import MU_MOON, R_MOON
from lunaris.common.hashing import canonical_json_sha256
from lunaris.common.provenance import sha256_file, utc_now_iso

_KNOWN_FORCE_FLAGS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("spherical_harmonics", "Spherical-harmonic gravity residual", ("enable_sh",)),
    ("third_body_sun", "Sun third-body gravity", ("enable_3rd_body_sun",)),
    ("third_body_earth", "Earth third-body gravity", ("enable_3rd_body_earth",)),
    ("earth_j2", "Earth J2 differential gravity", ("enable_earth_j2",)),
    ("srp", "Solar radiation pressure", ("enable_srp",)),
    ("albedo", "Lunar albedo pressure", ("enable_albedo",)),
    ("thermal_ir", "Lunar thermal IR pressure", ("enable_thermal", "enable_thermal_ir")),
    ("solid_tides", "Solid tides", ("enable_tides", "enable_tides_k2", "enable_tides_k3")),
    ("relativity", "Relativity (1PN)", ("enable_relativity_1pn", "enable_relativity")),
)


def _get(obj: Any, *path: str, default: Any = None) -> Any:
    node = obj
    for key in path:
        if isinstance(node, Mapping):
            if key not in node:
                return default
            node = node[key]
        else:
            if not hasattr(node, key):
                return default
            node = getattr(node, key)
    return default if node is None else node


def _jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple | list):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return str(value)


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _linear_slope_per_day(t_s: np.ndarray, values: np.ndarray) -> float | None:
    """Least-squares secular slope on full-resolution samples, per day."""
    t = np.asarray(t_s, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    mask = np.isfinite(t) & np.isfinite(y)
    if np.count_nonzero(mask) < 3 or float(np.ptp(t[mask])) <= 0.0:
        return None
    centered = t[mask] - float(np.mean(t[mask]))
    denominator = float(np.dot(centered, centered))
    if denominator <= 0.0:
        return None
    slope_per_s = float(np.dot(centered, y[mask] - float(np.mean(y[mask]))) / denominator)
    return slope_per_s * 86_400.0


def _metric(
    metric_id: str,
    label: str,
    value: Any,
    unit: str | None,
    *,
    source: str,
    kind: str = "derived",
    status: str = "ok",
    frame: str | None = None,
    time_system: str | None = None,
    interpretation: str | None = None,
    unavailable_reason: str | None = None,
) -> MetricValue:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        value = None
    if value is None:
        return MetricValue(
            metric_id=metric_id,
            label=label,
            value=None,
            unit=unit,
            status="unavailable",
            source=source,
            kind=kind,
            frame=frame,
            time_system=time_system,
            interpretation=interpretation,
            availability_reason=unavailable_reason or "Source value was unavailable.",
        )
    return MetricValue(
        metric_id=metric_id,
        label=label,
        value=value,
        unit=unit,
        status=status,
        source=source,
        kind=kind,
        frame=frame,
        time_system=time_system,
        interpretation=interpretation,
    )


def _extract_state(result: Any) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(getattr(result, "t", []), dtype=np.float64)
    y = np.asarray(getattr(result, "y", []), dtype=np.float64)
    if t.ndim != 1 or t.size < 2:
        raise ValueError("analysis requires a one-dimensional time vector with at least two samples")
    if y.ndim != 2:
        raise ValueError("analysis requires a two-dimensional state history")
    if y.shape[0] == t.size and y.shape[1] >= 6:
        state = y
    elif y.shape[1] == t.size and y.shape[0] >= 6:
        state = y.T
    else:
        raise ValueError("state history does not align with the time vector")
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(state[:, :6])):
        raise ValueError("time/state history contains NaN or Inf")
    if np.any(np.diff(t) < 0.0):
        raise ValueError("time history must be monotonic")
    return t, state


def _epoch_at(start_text: str | None, elapsed_s: float) -> str | None:
    if not start_text:
        return None
    try:
        text = str(start_text).strip()
        parsed = datetime.fromisoformat(text[:-1] + "+00:00" if text.endswith("Z") else text)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        epoch = parsed.astimezone(timezone.utc) + timedelta(seconds=float(elapsed_s))
        return epoch.isoformat(timespec="milliseconds").replace("+00:00", "Z")
    except (TypeError, ValueError, OverflowError):
        return None


def _slug(text: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "_", text.strip().lower()).strip("_")
    return value or "component"


def _active_flags(config: Any) -> dict[str, bool]:
    flags = _get(config, "flags", default={})
    active: dict[str, bool] = {}
    for force_id, _label, names in _KNOWN_FORCE_FLAGS:
        active[force_id] = any(bool(_get(flags, name, default=False)) for name in names)
    return active


def _to_ric(vector: np.ndarray, state: np.ndarray) -> np.ndarray:
    """Project one inertial vector into the state's right-handed local RIC frame.

    ``state`` is ``[r, v]`` with shape ``(6,)`` in one central-body inertial
    frame. The input vector is shape ``(3,)`` in the same axes and keeps its
    original units. Basis rows are ``R=r/|r|``, ``C=(r×v)/|r×v|``, and
    ``I=C×R``; the returned component order is radial, in-track, cross-track.
    Zero radius or angular momentum makes the frame undefined and raises
    ``ValueError`` rather than inventing an orientation.
    """
    r = np.asarray(state[:3], dtype=np.float64)
    v = np.asarray(state[3:6], dtype=np.float64)
    r_norm = float(np.linalg.norm(r))
    h = np.cross(r, v)
    h_norm = float(np.linalg.norm(h))
    if r_norm <= 0.0 or h_norm <= 0.0:
        raise ValueError("RIC frame is undefined for a degenerate state")
    radial = r / r_norm
    cross_track = h / h_norm
    in_track = np.cross(cross_track, radial)
    return np.asarray(
        (
            np.dot(vector, radial),
            np.dot(vector, in_track),
            np.dot(vector, cross_track),
        ),
        dtype=np.float64,
    )


def _noncentral_vector_names(names: Sequence[str]) -> list[str]:
    """Return unique non-central rows, excluding aggregate duplicates."""
    available = set(names)
    has_individual_tides = any(name.startswith("Solid Tides (") for name in names)
    has_individual_relativity = any(
        name in {"Relativity (Moon Schwarzschild)", "Relativity (External 1PN)"}
        for name in names
    )
    selected: list[str] = []
    for name in names:
        if name in {"Gravity (SH)", "Gravity (PM)", "Gravity (ST-LRPS)"}:
            continue
        if name == "Solid Tides" and has_individual_tides:
            continue
        if name == "Relativity (1PN)" and has_individual_relativity:
            continue
        if name == "Total non-central acceleration":
            continue
        if name in available:
            selected.append(name)
    return selected


def _force_budget(
    *,
    t_s: np.ndarray,
    state: np.ndarray,
    ctx: Any,
    config: Any,
    preset: str,
    radius_m: float,
) -> tuple[
    tuple[ForceContribution, ...],
    np.ndarray | None,
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    list[str],
]:
    active = _active_flags(config)
    warnings: list[str] = []
    vector_hook = (
        getattr(ctx, "get_acceleration_vector_breakdown", None)
        if ctx is not None
        else None
    )
    hook = getattr(ctx, "get_acceleration_breakdown", None) if ctx is not None else None
    if callable(vector_hook):
        hook = vector_hook
    if not callable(hook):
        rows = [
            ForceContribution(
                force_id=force_id,
                label=label,
                active=active[force_id],
                available=False,
                minimum_m_s2=None,
                median_m_s2=None,
                p95_m_s2=None,
                maximum_m_s2=None,
                sample_count=0,
                source="DynamicsEngine.get_acceleration_breakdown",
                included_in_noncentral_ranking=False,
                availability_reason=(
                    "Force model disabled for this run."
                    if not active[force_id]
                    else "The propagation context does not expose an acceleration-breakdown hook."
                ),
            )
            for force_id, label, _names in _KNOWN_FORCE_FLAGS
        ]
        rows.extend(
            [
                ForceContribution(
                    force_id="total_noncentral_acceleration",
                    label="Total non-central acceleration",
                    active=any(enabled for key, enabled in active.items() if key != "spherical_harmonics"),
                    available=False,
                    minimum_m_s2=None,
                    median_m_s2=None,
                    p95_m_s2=None,
                    maximum_m_s2=None,
                    sample_count=0,
                    source="DynamicsEngine.get_acceleration_breakdown",
                    included_in_noncentral_ranking=False,
                    availability_reason="Force-vector breakdown is unavailable for this run context.",
                ),
                ForceContribution(
                    force_id="sh_degree_increments",
                    label="Spherical-harmonic degree increments",
                    active=active["spherical_harmonics"],
                    available=False,
                    minimum_m_s2=None,
                    median_m_s2=None,
                    p95_m_s2=None,
                    maximum_m_s2=None,
                    sample_count=0,
                    source="opt-in perturbation-budget workflow",
                    included_in_noncentral_ranking=False,
                    availability_reason="Per-degree vectors were not evaluated for this report.",
                    interpretation="Use the existing detailed perturbation-budget CLI as an explicit opt-in.",
                ),
            ]
        )
        return tuple(rows), None, {}, {}, {}, warnings

    cap = {"quick": 96, "standard": 384, "paper": 1024}[preset]
    if t_s.size > cap:
        indices = np.linspace(0, t_s.size - 1, cap, dtype=np.int64)
    else:
        indices = np.arange(t_s.size, dtype=np.int64)
    sample_t = t_s[indices]
    sample_state = state[indices, :6]

    sample_payloads: list[dict[str, float]] = []
    vector_payloads: list[dict[str, np.ndarray]] = []
    for time_value, state_value in zip(sample_t, sample_state, strict=True):
        try:
            payload = hook(float(time_value), np.asarray(state_value, dtype=np.float64))
        except Exception as exc:
            warnings.append(f"Force-budget sample failed at t={time_value:.6g} s: {exc}")
            sample_payloads.append({})
            vector_payloads.append({})
            continue
        clean: dict[str, float] = {}
        clean_vectors: dict[str, np.ndarray] = {}
        if isinstance(payload, Mapping):
            for key, value in payload.items():
                if callable(vector_hook):
                    vector = np.asarray(value, dtype=np.float64)
                    if vector.shape == (3,) and np.all(np.isfinite(vector)):
                        clean_vectors[str(key)] = vector
                        clean[str(key)] = float(np.linalg.norm(vector))
                else:
                    finite = _finite_float(value)
                    if finite is not None and finite >= 0.0:
                        clean[str(key)] = finite
        sample_payloads.append(clean)
        vector_payloads.append(clean_vectors)

    names = sorted({name for payload in sample_payloads for name in payload})
    series: dict[str, np.ndarray] = {}
    vector_series: dict[str, np.ndarray] = {}
    ric_series: dict[str, np.ndarray] = {}
    if callable(vector_hook):
        for name in names:
            vectors = np.asarray(
                [payload.get(name, np.full(3, np.nan)) for payload in vector_payloads],
                dtype=np.float64,
            )
            vector_series[name] = vectors
            ric_values = np.full_like(vectors, np.nan)
            for index, (vector, state_value) in enumerate(
                zip(vectors, sample_state, strict=True)
            ):
                if not np.all(np.isfinite(vector)):
                    continue
                try:
                    ric_values[index] = _to_ric(vector, state_value)
                except ValueError as exc:
                    warnings.append(
                        f"RIC projection failed at t={sample_t[index]:.6g} s: {exc}"
                    )
            ric_series[name] = ric_values

        noncentral_names = _noncentral_vector_names(names)
        if noncentral_names:
            stacked = np.stack([vector_series[name] for name in noncentral_names], axis=0)
            valid = np.all(np.isfinite(stacked), axis=(0, 2))
            total = np.full((sample_t.size, 3), np.nan, dtype=np.float64)
            total[valid] = np.sum(stacked[:, valid, :], axis=0)
            total_name = "Total non-central acceleration"
            vector_series[total_name] = total
            ric_total = np.full_like(total, np.nan)
            for index, (vector, state_value) in enumerate(
                zip(total, sample_state, strict=True)
            ):
                if np.all(np.isfinite(vector)):
                    ric_total[index] = _to_ric(vector, state_value)
            ric_series[total_name] = ric_total
            magnitudes = np.linalg.norm(total, axis=1)
            magnitudes[~valid] = np.nan
            series[total_name] = magnitudes
    contributions: list[ForceContribution] = []
    has_individual_tides = any(name.startswith("Solid Tides (") for name in names)
    for name in names:
        values = np.asarray([payload.get(name, np.nan) for payload in sample_payloads], dtype=np.float64)
        series[name] = values
        finite_values = values[np.isfinite(values)]
        if finite_values.size == 0:
            continue
        is_total_gravity = name in {"Gravity (SH)", "Gravity (PM)", "Gravity (ST-LRPS)"}
        is_tide_aggregate = name == "Solid Tides" and has_individual_tides
        is_relativity_aggregate = name == "Relativity (1PN)" and any(
            candidate in names
            for candidate in {
                "Relativity (Moon Schwarzschild)",
                "Relativity (External 1PN)",
            }
        )
        interpretation = None
        if name in {"Gravity (SH)", "Gravity (ST-LRPS)"}:
            interpretation = (
                "Total active lunar gravity acceleration. The live engine hook does not expose "
                "a central-field/SH-residual vector split, so no residual is inferred."
            )
        elif is_tide_aggregate:
            interpretation = (
                "Vector aggregate of Earth and Sun tide terms; excluded from rankings when its "
                "individual components are present to prevent double counting."
            )
        elif is_relativity_aggregate:
            interpretation = (
                "Vector aggregate of the Moon and external 1PN terms; excluded from rankings "
                "when constituent vectors are present to prevent double counting."
            )
        if callable(vector_hook):
            interpretation = " ".join(
                part
                for part in (
                    interpretation,
                    "Inertial vectors and signed RIC components are available in the force time series.",
                )
                if part
            )
        contributions.append(
            ForceContribution(
                force_id=_slug(name),
                label=name,
                active=True,
                available=True,
                minimum_m_s2=float(np.min(finite_values)),
                median_m_s2=float(np.median(finite_values)),
                p95_m_s2=float(np.percentile(finite_values, 95.0)),
                maximum_m_s2=float(np.max(finite_values)),
                sample_count=int(finite_values.size),
                source=(
                    "DynamicsEngine.get_acceleration_vector_breakdown over propagated history"
                    if callable(vector_hook)
                    else "DynamicsEngine.get_acceleration_breakdown over propagated history"
                ),
                included_in_noncentral_ranking=not (
                    is_total_gravity or is_tide_aggregate or is_relativity_aggregate
                ),
                interpretation=interpretation,
            )
        )

    returned_ids = {item.force_id for item in contributions}
    for force_id, label, _names in _KNOWN_FORCE_FLAGS:
        if force_id == "spherical_harmonics":
            # The hook reports total SH gravity rather than a residual.  Preserve
            # that distinction instead of subtracting magnitudes (invalid).
            contributions.append(
                ForceContribution(
                    force_id="spherical_harmonic_residual",
                    label=label,
                    active=active[force_id],
                    available=False,
                    minimum_m_s2=None,
                    median_m_s2=None,
                    p95_m_s2=None,
                    maximum_m_s2=None,
                    sample_count=0,
                    source="DynamicsEngine.get_acceleration_breakdown",
                    included_in_noncentral_ranking=False,
                    availability_reason=(
                        "Force model disabled for this run."
                        if not active[force_id]
                        else "The live hook exposes total SH gravity, not the central-field residual vector."
                    ),
                    interpretation="No magnitude subtraction is performed because that would be physically invalid.",
                )
            )
            continue
        aliases = {
            "third_body_sun": ("3rd_body_sun",),
            "third_body_earth": ("3rd_body_earth",),
            "earth_j2": ("3rd_body_earth_j2",),
            "srp": ("srp",),
            "albedo": ("albedo", "lunar_albedo"),
            "thermal_ir": ("thermal_ir", "thermal_radiation"),
            "solid_tides": ("solid_tides",),
            "relativity": ("relativity_1pn", "1pn_relativity"),
        }.get(force_id, ())
        if any(alias in returned_ids for alias in aliases):
            continue
        contributions.append(
            ForceContribution(
                force_id=force_id,
                label=label,
                active=active[force_id],
                available=False,
                minimum_m_s2=None,
                median_m_s2=None,
                p95_m_s2=None,
                maximum_m_s2=None,
                sample_count=0,
                source="DynamicsEngine.get_acceleration_breakdown",
                included_in_noncentral_ranking=False,
                availability_reason=(
                    "Force model disabled for this run."
                    if not active[force_id]
                    else "The active engine did not return this component from its breakdown hook."
                ),
            )
        )

    any_noncentral = any(active_id != "spherical_harmonics" and enabled for active_id, enabled in active.items())
    total_values = series.get("Total non-central acceleration")
    total_finite = (
        np.asarray(total_values, dtype=np.float64)[
            np.isfinite(np.asarray(total_values, dtype=np.float64))
        ]
        if total_values is not None
        else np.asarray([], dtype=np.float64)
    )
    if total_finite.size:
        contributions.append(
            ForceContribution(
                force_id="total_noncentral_acceleration",
                label="Total non-central acceleration",
                active=True,
                available=True,
                minimum_m_s2=float(np.min(total_finite)),
                median_m_s2=float(np.median(total_finite)),
                p95_m_s2=float(np.percentile(total_finite, 95.0)),
                maximum_m_s2=float(np.max(total_finite)),
                sample_count=int(total_finite.size),
                source="vector sum of unique DynamicsEngine force components",
                included_in_noncentral_ranking=False,
                interpretation=(
                    "Vector sum excludes lunar gravity and aggregate duplicates. Signed RIC "
                    "components are preserved in the force time series."
                ),
            )
        )
    else:
        contributions.append(
            ForceContribution(
                force_id="total_noncentral_acceleration",
                label="Total non-central acceleration",
                active=any_noncentral,
                available=False,
                minimum_m_s2=None,
                median_m_s2=None,
                p95_m_s2=None,
                maximum_m_s2=None,
                sample_count=0,
                source="DynamicsEngine.get_acceleration_breakdown",
                included_in_noncentral_ranking=False,
                availability_reason=(
                    "No non-central force model was active."
                    if not any_noncentral
                    else "The engine hook exposes component magnitudes but not vectors; a physically valid vector total cannot be reconstructed."
                ),
            )
        )
    contributions.append(
        ForceContribution(
            force_id="sh_degree_increments",
            label="Spherical-harmonic degree increments",
            active=active["spherical_harmonics"],
            available=False,
            minimum_m_s2=None,
            median_m_s2=None,
            p95_m_s2=None,
            maximum_m_s2=None,
            sample_count=0,
            source="DynamicsEngine.get_acceleration_breakdown",
            included_in_noncentral_ranking=False,
            availability_reason=(
                "Force model disabled for this run."
                if not active["spherical_harmonics"]
                else "The live run hook does not expose per-degree vectors."
            ),
            interpretation=(
                "Degree-band sensitivity requires the existing opt-in perturbation-budget "
                "workflow; it is not inferred from total-gravity magnitudes."
            ),
        )
    )

    eligible = [item for item in contributions if item.available and item.included_in_noncentral_ranking]
    if eligible:
        labels = [item.label for item in eligible]
        matrix = np.column_stack([series[label] for label in labels])
        valid_rows = np.any(np.isfinite(matrix), axis=1)
        safe = np.where(np.isfinite(matrix), matrix, -np.inf)
        dominant_index = np.argmax(safe, axis=1)
        sampled_altitude = np.linalg.norm(sample_state[:, :3], axis=1) - float(radius_m)
        updated: list[ForceContribution] = []
        for item in contributions:
            if item.label not in labels:
                updated.append(item)
                continue
            mask = valid_rows & (dominant_index == labels.index(item.label))
            if not np.any(mask):
                updated.append(item)
                continue
            low = float(np.min(sampled_altitude[mask])) / 1000.0
            high = float(np.max(sampled_altitude[mask])) / 1000.0
            dominance_note = f"Dominant sampled non-central term from {low:.3f} to {high:.3f} km altitude."
            updated.append(
                replace(
                    item,
                    interpretation=" ".join(
                        part for part in (item.interpretation, dominance_note) if part
                    ),
                )
            )
        contributions = updated
    contributions.sort(key=lambda item: item.force_id)
    return tuple(contributions), sample_t, series, vector_series, ric_series, warnings


def _git_snapshot() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[3]

    def run(*args: str) -> str | None:
        try:
            proc = subprocess.run(
                ["git", "-C", str(root), *args],
                capture_output=True,
                text=True,
                check=False,
                timeout=5,
            )
        except Exception:
            return None
        return proc.stdout.strip() if proc.returncode == 0 else None

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    return {"commit": commit, "dirty": None if status is None else bool(status)}


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _asset_record(path_value: Any, label: str) -> dict[str, Any]:
    if path_value is None or str(path_value).strip() == "":
        return {"label": label, "path": None, "sha256": None, "status": "unavailable"}
    path = Path(str(path_value)).expanduser()
    if path.is_dir():
        return {
            "label": label,
            "path": str(path),
            "sha256": None,
            "status": "directory_reference",
        }
    digest = sha256_file(path, missing_ok=True, suppress_errors=True)
    return {
        "label": label,
        "path": str(path),
        "sha256": digest,
        "status": "ok" if digest else "missing",
    }


def _provenance(
    config: Any,
    diagnostics: Mapping[str, Any],
    frame: str,
    meta: Mapping[str, Any],
    effective_initial_state: np.ndarray,
) -> tuple[dict[str, Any], dict[str, Any]]:
    config_snapshot = _jsonable(config)
    if not isinstance(config_snapshot, dict):
        config_snapshot = {"value": config_snapshot}
    initial = np.asarray(effective_initial_state, dtype=np.float64).reshape(-1)
    config_snapshot["effective_initial_state"] = {
        "source": "PropagationResult.y[0]",
        "frame": frame,
        "x_m": float(initial[0]),
        "y_m": float(initial[1]),
        "z_m": float(initial[2]),
        "vx_m_s": float(initial[3]),
        "vy_m_s": float(initial[4]),
        "vz_m_s": float(initial[5]),
    }
    gravity_path = _get(config, "gravity", "file_path", default=None)
    kernel_values = _get(config, "spice", "kernel_paths", default=None)
    if kernel_values is None:
        kernel_values = _get(config, "spice", "kernels", default=())
    if isinstance(kernel_values, str | Path):
        kernel_values = [kernel_values]
    kernels = [_asset_record(item, f"SPICE kernel {index + 1}") for index, item in enumerate(kernel_values or ())]
    git = _git_snapshot()
    provenance = {
        "generated_at_utc": utc_now_iso(),
        "git": git,
        "lunaris_version": _package_version("lunaris"),
        "config_sha256": canonical_json_sha256(config_snapshot),
        "gravity_model": _asset_record(gravity_path, "gravity model"),
        "spice_kernels": kernels,
        "backend": {
            "requested_gravity": _get(config, "gravity", "backend", default=None),
            "effective_rhs": diagnostics.get("rhs_path"),
            "integration": diagnostics.get("integration_backend"),
            "device": diagnostics.get("device", "cpu"),
            "dtype": diagnostics.get("dtype", "float64"),
        },
        "integrator": diagnostics.get("integrator") or _get(config, "propagator", "method", default=None),
        "frame": frame,
        "platform": platform.platform(),
        "python_version": sys.version.replace("\n", " "),
        "packages": {
            "numpy": np.__version__,
            "scipy": _package_version("scipy"),
            "numba": _package_version("numba"),
            "matplotlib": _package_version("matplotlib"),
        },
        "schemas": {
            "analysis": ANALYSIS_SCHEMA_VERSION,
            "report": REPORT_SCHEMA_VERSION,
        },
        "surface_assets": {
            "topography": _asset_record(meta.get("ldem_root"), "topography asset"),
            "albedo": _asset_record(meta.get("albedo_root"), "albedo asset"),
            "thermal": _asset_record(meta.get("thermal_asset"), "thermal asset"),
        },
    }
    return provenance, config_snapshot


def _termination(result: Any) -> tuple[str, str]:
    if bool(getattr(result, "impacted", False)):
        return "impact", "critical"
    if bool(getattr(result, "stopped_early", False)):
        reason = str(getattr(result, "stop_reason", None) or "stopped early")
        return reason, "warning"
    ode = getattr(result, "ode", None)
    if ode is not None and getattr(ode, "success", True) is False:
        return str(getattr(ode, "message", "integrator failure")), "critical"
    return "completed", "ok"


def _event(
    *,
    event_id: str,
    event_type: str,
    index: int,
    t_s: np.ndarray,
    state: np.ndarray,
    altitude_m: np.ndarray,
    start_epoch: str | None,
    frame: str,
    source: str,
    severity: str = "normal",
    note: str | None = None,
) -> AnalysisEvent:
    idx = int(np.clip(index, 0, t_s.size - 1))
    return AnalysisEvent(
        event_id=event_id,
        event_type=event_type,
        simulation_time_s=float(t_s[idx]),
        epoch_utc=_epoch_at(start_epoch, float(t_s[idx] - t_s[0])),
        state_m_mps=tuple(float(value) for value in state[idx, :6]),
        altitude_m=float(altitude_m[idx]),
        frame=frame,
        source=source,
        severity=severity,
        note=note,
    )


def build_orbit_analysis(
    result: Any,
    *,
    config: Any,
    ctx: Any = None,
    meta: Mapping[str, Any] | None = None,
    preset: str = "standard",
    run_id: str | None = None,
) -> OrbitAnalysisResult:
    """Compute the canonical, full-resolution analysis for one propagation."""

    normalized_preset = str(preset).strip().lower()
    if normalized_preset not in {"quick", "standard", "paper"}:
        raise ValueError("preset must be one of: quick, standard, paper")

    t_s, state = _extract_state(result)
    diagnostics = dict(getattr(result, "diagnostics", None) or {})
    meta_map = dict(meta or {})
    mu = _finite_float(meta_map.get("mu_m3s2"))
    if mu is None:
        mu = _finite_float(getattr(ctx, "mu_m3s2", None) if ctx is not None else None)
    if mu is None:
        mu = _finite_float(_get(config, "gravity", "mu_m3s2", default=None))
    if mu is None:
        mu = float(MU_MOON)
    radius_m = _finite_float(getattr(ctx, "R_body_m", None) if ctx is not None else None)
    if radius_m is None:
        radius_m = _finite_float(meta_map.get("body_radius_m"))
    if radius_m is None:
        radius_m = float(R_MOON)

    event_cfg = _get(config, "propagator", "events", default={})
    impact_alt_km = _finite_float(_get(event_cfg, "impact_alt_km", default=0.0))
    compute_eclipse = bool(_get(event_cfg, "detect_eclipse", default=False)) or any(
        _active_flags(config).get(name, False) for name in ("srp", "albedo", "thermal_ir")
    )
    history = compute_history(
        t_s,
        state[:, :6].T,
        mu=mu,
        R_body=radius_m,
        ctx=ctx,
        impact_alt_km=impact_alt_km,
        max_samples=int(t_s.size),
        detect_peri_apo=bool(_get(event_cfg, "enable_peri_apo_events", default=True)),
        detect_impact=bool(_get(event_cfg, "detect_impact", default=True)),
        compute_eclipse=compute_eclipse,
        compute_groundtrack=normalized_preset != "quick",
        compute_accel_breakdown=False,
        strict=False,
    )

    r = state[:, :3]
    v = state[:, 3:6]
    h_vec = np.cross(r, v)
    h_norm = np.linalg.norm(h_vec, axis=1)
    radius = np.linalg.norm(r, axis=1)
    speed = np.linalg.norm(v, axis=1)
    e_vec = np.cross(v, h_vec) / float(mu) - r / radius[:, None]
    e_norm = np.linalg.norm(e_vec, axis=1)
    altitude = radius - float(radius_m)

    groundtrack = history.get("groundtrack") if isinstance(history.get("groundtrack"), Mapping) else {}
    lat = np.radians(np.asarray(groundtrack.get("lat_deg", []), dtype=np.float64)) if groundtrack else None
    lon = np.radians(np.asarray(groundtrack.get("lon_deg", []), dtype=np.float64)) if groundtrack else None
    eclipse = history.get("eclipse")
    eclipse_mask = np.asarray(eclipse, dtype=bool) if eclipse is not None else None

    series = OrbitSeries(
        t_s=np.asarray(history["t_s"], dtype=np.float64),
        state_m_mps=np.asarray(state, dtype=np.float64),
        semi_major_axis_m=np.asarray(history["a_km"], dtype=np.float64) * 1000.0,
        eccentricity=np.asarray(history["e"], dtype=np.float64),
        inclination_rad=np.radians(np.asarray(history["i_deg"], dtype=np.float64)),
        raan_rad=np.radians(np.asarray(history["raan_deg"], dtype=np.float64)),
        argument_of_periapsis_rad=np.radians(np.asarray(history["argp_deg"], dtype=np.float64)),
        true_anomaly_rad=np.radians(np.asarray(history["nu_deg"], dtype=np.float64)),
        altitude_m=np.asarray(altitude, dtype=np.float64),
        radius_m=np.asarray(radius, dtype=np.float64),
        speed_m_s=np.asarray(speed, dtype=np.float64),
        specific_energy_j_kg=np.asarray(history["energy_Jkg"], dtype=np.float64),
        angular_momentum_m2_s=np.asarray(h_norm, dtype=np.float64),
        eccentricity_vector_norm=np.asarray(e_norm, dtype=np.float64),
        eclipse_mask=eclipse_mask,
        latitude_rad=lat,
        longitude_rad=lon,
    )

    frame = str(
        _get(config, "spice", "inertial_frame", default=None)
        or meta_map.get("frame")
        or "J2000"
    )
    start_epoch = str(_get(config, "time", "start_date", default="")) or None
    time_system = "UTC"
    termination_reason, termination_status = _termination(result)

    event_map = history.get("events", {})
    events: list[AnalysisEvent] = []
    min_idx = int(np.argmin(altitude))
    max_idx = int(np.argmax(altitude))
    events.append(
        _event(
            event_id="minimum_altitude",
            event_type="minimum_altitude",
            index=min_idx,
            t_s=t_s,
            state=state,
            altitude_m=altitude,
            start_epoch=start_epoch,
            frame=frame,
            source="full-resolution global extrema",
            severity="critical" if altitude[min_idx] <= 0.0 else "normal",
        )
    )
    events.append(
        _event(
            event_id="maximum_altitude",
            event_type="maximum_altitude",
            index=max_idx,
            t_s=t_s,
            state=state,
            altitude_m=altitude,
            start_epoch=start_epoch,
            frame=frame,
            source="full-resolution global extrema",
        )
    )
    for kind, key in (("periselene", "peri_idx"), ("aposelene", "apo_idx")):
        for number, index in enumerate(np.asarray(event_map.get(key, []), dtype=int), start=1):
            events.append(
                _event(
                    event_id=f"{kind}_{number:04d}",
                    event_type=kind,
                    index=int(index),
                    t_s=t_s,
                    state=state,
                    altitude_m=altitude,
                    start_epoch=start_epoch,
                    frame=frame,
                    source="radial-velocity sign change",
                )
            )
    if eclipse_mask is not None and eclipse_mask.size == t_s.size:
        changes = np.flatnonzero(np.diff(eclipse_mask.astype(np.int8)) != 0) + 1
        for number, index in enumerate(changes, start=1):
            entering = bool(eclipse_mask[index])
            events.append(
                _event(
                    event_id=f"eclipse_{'enter' if entering else 'exit'}_{number:04d}",
                    event_type="eclipse_enter" if entering else "eclipse_exit",
                    index=int(index),
                    t_s=t_s,
                    state=state,
                    altitude_m=altitude,
                    start_epoch=start_epoch,
                    frame=frame,
                    source="postprocessed occultation mask transition",
                )
            )
    impact_idx = event_map.get("impact_idx")
    if impact_idx is not None or bool(getattr(result, "impacted", False)):
        index = int(impact_idx) if impact_idx is not None else int(np.argmin(np.abs(t_s - float(getattr(result, "t_impact_s", t_s[-1])))))
        events.append(
            _event(
                event_id="impact",
                event_type="impact",
                index=index,
                t_s=t_s,
                state=state,
                altitude_m=altitude,
                start_epoch=start_epoch,
                frame=frame,
                source="PropagationResult impact outcome",
                severity="critical",
            )
        )
    if bool(diagnostics.get("fallback_applied") or diagnostics.get("backend_fallback")):
        events.append(
            _event(
                event_id="backend_fallback",
                event_type="backend_fallback",
                index=0,
                t_s=t_s,
                state=state,
                altitude_m=altitude,
                start_epoch=start_epoch,
                frame=frame,
                source="propagation diagnostics",
                severity="warning",
                note=str(diagnostics.get("fallback_reason") or "Backend fallback applied."),
            )
        )
    diagnostic_warning = diagnostics.get("single_run_stlrps_cpu_warning") or diagnostics.get("symplectic_violation")
    if diagnostic_warning:
        events.append(
            _event(
                event_id="numerical_warning",
                event_type="numerical_warning",
                index=t_s.size - 1,
                t_s=t_s,
                state=state,
                altitude_m=altitude,
                start_epoch=start_epoch,
                frame=frame,
                source="propagation diagnostics",
                severity="warning",
                note="Run diagnostics contain a numerical/backend warning; see Numerical Health.",
            )
        )
    events.append(
        _event(
            event_id="terminal",
            event_type="terminal_event" if termination_reason != "completed" else "completed",
            index=t_s.size - 1,
            t_s=t_s,
            state=state,
            altitude_m=altitude,
            start_epoch=start_epoch,
            frame=frame,
            source="PropagationResult termination outcome",
            severity="critical" if termination_status == "critical" else ("warning" if termination_status == "warning" else "normal"),
            note=termination_reason,
        )
    )
    events.sort(key=lambda item: (item.simulation_time_s, item.event_id))

    peri_indices = np.asarray(event_map.get("peri_idx", []), dtype=int)
    period_s: float | None = None
    period_source = ""
    if peri_indices.size >= 2:
        period_s = float(np.median(np.diff(t_s[peri_indices])))
        period_source = "median interval between full-resolution periselene detections"
    elif series.semi_major_axis_m.size and series.semi_major_axis_m[0] > 0.0:
        period_s = float(2.0 * np.pi * math.sqrt(series.semi_major_axis_m[0] ** 3 / mu))
        period_source = "initial osculating two-body period approximation"
    duration_s = float(t_s[-1] - t_s[0])
    orbit_count = duration_s / period_s if period_s is not None and period_s > 0.0 else None

    flags = _active_flags(config)
    noncentral_active = any(flags.values())
    conservation_note = (
        "Diagnostic only - not expected to be conserved under the active force model."
        if noncentral_active
        else "Expected invariant for the active point-mass conservative force model."
    )
    invariant_kind = "diagnostic" if noncentral_active else "invariant"
    energy = series.specific_energy_j_kg
    angmom = series.angular_momentum_m2_s
    rel_energy = (energy - energy[0]) / max(abs(float(energy[0])), 1.0e-30)
    rel_angmom = (angmom - angmom[0]) / max(abs(float(angmom[0])), 1.0e-30)
    dt = np.diff(t_s)
    output_min = float(np.min(dt)) if dt.size else None
    output_median = float(np.median(dt)) if dt.size else None
    output_max = float(np.max(dt)) if dt.size else None
    nfev = _finite_float(diagnostics.get("nfev"))
    wall_s = _finite_float(diagnostics.get("wall_time_s"))
    throughput = float(t_s.size / wall_s) if wall_s is not None and wall_s > 0.0 else None
    accepted_steps = _finite_float(diagnostics.get("accepted_steps"))
    rejected_steps = _finite_float(diagnostics.get("rejected_steps"))
    internal_step_min = _finite_float(diagnostics.get("internal_step_min_s"))
    internal_step_median = _finite_float(diagnostics.get("internal_step_median_s"))
    internal_step_max = _finite_float(diagnostics.get("internal_step_max_s"))
    method = diagnostics.get("integrator") or _get(config, "propagator", "method", default=None)
    integration_backend = diagnostics.get("integration_backend")
    requested_gravity_backend = _get(config, "gravity", "backend", default=None)
    effective_rhs = diagnostics.get("rhs_path")
    degree = _finite_float(diagnostics.get("degree"))
    if degree is None:
        degree = _finite_float(_get(config, "gravity", "degree", default=None))
    recommended_degree = _finite_float(diagnostics.get("recommended_degree"))
    degree_is_low = (
        degree is not None
        and recommended_degree is not None
        and recommended_degree > degree
    )

    e0 = float(series.eccentricity[0])
    ef = float(series.eccentricity[-1])
    i0 = float(series.inclination_rad[0])
    i1 = float(series.inclination_rad[-1])
    circular_start = e0 < 1.0e-8
    circular_end = ef < 1.0e-8
    equatorial_start = abs(math.sin(i0)) < 1.0e-8
    equatorial_end = abs(math.sin(i1)) < 1.0e-8
    alternative_start = float(np.mod(math.atan2(state[0, 1], state[0, 0]), 2.0 * math.pi))
    alternative_final = float(np.mod(math.atan2(state[-1, 1], state[-1, 0]), 2.0 * math.pi))
    singular_start = circular_start or equatorial_start
    singular_end = circular_end or equatorial_end
    secular_note = (
        "Full-history least-squares slope. Osculating periodic terms are not removed; "
        "interpret as a compact trend indicator, not a force attribution."
    )

    metrics: list[MetricValue] = [
        _metric("run.status", "Run status", termination_reason, None, source="PropagationResult", kind="measured", status=termination_status),
        _metric("mission.start_epoch", "Mission start epoch", start_epoch, None, source="TimeConfig.start_date", kind="configuration", time_system=time_system, unavailable_reason="Start epoch was not configured."),
        _metric("mission.duration", "Propagation duration", duration_s, "s", source="full-resolution time history", kind="measured", time_system="simulation elapsed time"),
        _metric("mission.output_samples", "Output sample count", int(t_s.size), "1", source="PropagationResult.t", kind="measured"),
        _metric("orbit.altitude.minimum", "Minimum altitude", float(np.min(altitude)), "m", source="full-resolution state history", frame=frame),
        _metric("orbit.altitude.maximum", "Maximum altitude", float(np.max(altitude)), "m", source="full-resolution state history", frame=frame),
        _metric("orbit.altitude.initial", "Initial altitude", float(altitude[0]), "m", source="full-resolution state history", frame=frame),
        _metric("orbit.altitude.final", "Final altitude", float(altitude[-1]), "m", source="full-resolution state history", frame=frame),
        _metric("orbit.period", "Orbital period", period_s, "s", source=period_source or "unavailable", kind="derived", interpretation="Measured from periselene intervals when possible; otherwise an initial two-body approximation.", unavailable_reason="Too few periselene events and no valid elliptic semi-major axis."),
        _metric("orbit.completed_count", "Completed orbit count", orbit_count, "1", source="duration divided by reported orbital period", kind="derived", interpretation="May be fractional; not a count of solver events.", unavailable_reason="Orbital period was unavailable."),
        _metric("orbit.a.initial", "Initial semi-major axis", float(series.semi_major_axis_m[0]), "m", source="osculating elements from state", frame=frame),
        _metric("orbit.a.final", "Final semi-major axis", float(series.semi_major_axis_m[-1]), "m", source="osculating elements from state", frame=frame),
        _metric("orbit.e.initial", "Initial eccentricity", e0, "1", source="eccentricity vector from state", frame=frame),
        _metric("orbit.e.final", "Final eccentricity", ef, "1", source="eccentricity vector from state", frame=frame),
        _metric("orbit.i.initial", "Initial inclination", i0, "rad", source="osculating elements from state", frame=frame),
        _metric("orbit.i.final", "Final inclination", i1, "rad", source="osculating elements from state", frame=frame),
        _metric("orbit.raan.initial", "Initial RAAN", None if equatorial_start else float(series.raan_rad[0]), "rad", source="osculating elements from state", frame=frame, interpretation="Undefined for equatorial orbits; longitude-based alternatives should be used.", unavailable_reason="Equatorial singularity: RAAN is undefined."),
        _metric("orbit.raan.final", "Final RAAN", None if equatorial_end else float(series.raan_rad[-1]), "rad", source="osculating elements from state", frame=frame, interpretation="Undefined for equatorial orbits; longitude-based alternatives should be used.", unavailable_reason="Equatorial singularity: RAAN is undefined."),
        _metric("orbit.argp.initial", "Initial argument of periapsis", None if circular_start else float(series.argument_of_periapsis_rad[0]), "rad", source="osculating elements from state", frame=frame, interpretation="Undefined for circular orbits; argument of latitude/true longitude should be used.", unavailable_reason="Circular singularity: argument of periapsis is undefined."),
        _metric("orbit.argp.final", "Final argument of periapsis", None if circular_end else float(series.argument_of_periapsis_rad[-1]), "rad", source="osculating elements from state", frame=frame, interpretation="Undefined for circular orbits; argument of latitude/true longitude should be used.", unavailable_reason="Circular singularity: argument of periapsis is undefined."),
        _metric("orbit.alternative_longitude.initial", "Initial inertial longitude (singularity alternative)", alternative_start if singular_start else None, "rad", source="atan2(y, x) from state", frame=frame, interpretation="Wrapped to [0, 2pi); supplied when classical angular elements are singular.", unavailable_reason="Alternative angle is not needed for this nonsingular initial state."),
        _metric("orbit.alternative_longitude.final", "Final inertial longitude (singularity alternative)", alternative_final if singular_end else None, "rad", source="atan2(y, x) from state", frame=frame, interpretation="Wrapped to [0, 2pi); supplied when classical angular elements are singular.", unavailable_reason="Alternative angle is not needed for this nonsingular final state."),
        _metric("orbit.secular.semi_major_axis", "Semi-major-axis linear trend", _linear_slope_per_day(t_s, series.semi_major_axis_m), "m/day", source="full-resolution least-squares fit", kind="derived", frame=frame, interpretation=secular_note, unavailable_reason="Insufficient finite time history for a linear fit."),
        _metric("orbit.secular.eccentricity", "Eccentricity linear trend", _linear_slope_per_day(t_s, series.eccentricity), "1/day", source="full-resolution least-squares fit", kind="derived", frame=frame, interpretation=secular_note, unavailable_reason="Insufficient finite time history for a linear fit."),
        _metric("orbit.secular.inclination", "Inclination linear trend", _linear_slope_per_day(t_s, series.inclination_rad), "rad/day", source="full-resolution least-squares fit", kind="derived", frame=frame, interpretation=secular_note, unavailable_reason="Insufficient finite time history for a linear fit."),
        _metric("numerical.integrator", "Integrator", method, None, source="propagation diagnostics or config", kind="configuration", unavailable_reason="Integrator name was unavailable."),
        _metric("numerical.integration_backend", "Effective integration backend", integration_backend, None, source="propagation diagnostics", kind="measured", unavailable_reason="Integrator did not report a backend."),
        _metric("numerical.rhs_evaluations", "RHS evaluations", int(nfev) if nfev is not None else None, "1", source="integrator diagnostics", kind="measured", unavailable_reason="Unavailable for this integrator."),
        _metric("numerical.accepted_steps", "Accepted internal steps", int(accepted_steps) if accepted_steps is not None else None, "1", source="integrator diagnostics", kind="measured", unavailable_reason="Unavailable for this integrator."),
        _metric("numerical.rejected_steps", "Rejected internal steps", int(rejected_steps) if rejected_steps is not None else None, "1", source="integrator diagnostics", kind="measured", unavailable_reason="Unavailable for this integrator."),
        _metric("numerical.internal_step.minimum", "Minimum internal step", internal_step_min, "s", source="integrator diagnostics", kind="measured", unavailable_reason="Unavailable for this integrator; output cadence is not substituted."),
        _metric("numerical.internal_step.median", "Median internal step", internal_step_median, "s", source="integrator diagnostics", kind="measured", unavailable_reason="Unavailable for this integrator; output cadence is not substituted."),
        _metric("numerical.internal_step.maximum", "Maximum internal step", internal_step_max, "s", source="integrator diagnostics", kind="measured", unavailable_reason="Unavailable for this integrator; output cadence is not substituted."),
        _metric("numerical.output_step.minimum", "Minimum output cadence", output_min, "s", source="PropagationResult.t differences", kind="measured", interpretation="Output sampling cadence, not internal integrator step size.", unavailable_reason="Insufficient output samples."),
        _metric("numerical.output_step.median", "Median output cadence", output_median, "s", source="PropagationResult.t differences", kind="measured", interpretation="Output sampling cadence, not internal integrator step size.", unavailable_reason="Insufficient output samples."),
        _metric("numerical.output_step.maximum", "Maximum output cadence", output_max, "s", source="PropagationResult.t differences", kind="measured", interpretation="Output sampling cadence, not internal integrator step size.", unavailable_reason="Insufficient output samples."),
        _metric("numerical.wall_time", "Wall-clock runtime", wall_s, "s", source="propagation diagnostics", kind="measured", unavailable_reason="Runtime was not reported."),
        _metric("numerical.throughput", "Output throughput", throughput, "sample/s", source="output sample count / wall time", kind="derived", unavailable_reason="Runtime was not reported."),
        _metric("numerical.finite_validation", "Finite-value validation", True, "1", source="analysis input validation", kind="measured"),
        _metric("numerical.termination_reason", "Termination reason", termination_reason, None, source="PropagationResult", kind="measured", status=termination_status),
        _metric("numerical.tolerance.relative", "Relative tolerance", _finite_float(_get(config, "propagator", "rtol", default=None)), "1", source="PropagatorConfig", kind="configuration", unavailable_reason="Not configured for this integrator."),
        _metric("numerical.tolerance.absolute", "Absolute tolerance", _finite_float(_get(config, "propagator", "atol", default=None)), "state-unit", source="PropagatorConfig", kind="configuration", interpretation="Scalar state tolerance; position/velocity-specific tolerances remain in config.json.", unavailable_reason="Not configured for this integrator."),
        _metric("numerical.checkpoint", "Checkpoint configuration", str(_get(config, "propagator", "checkpoint_path", default=None)) if _get(config, "propagator", "checkpoint_path", default=None) else "not configured", None, source="PropagatorConfig", kind="configuration", interpretation="Configuration state only; this does not claim that a checkpoint write or resume occurred."),
        _metric("numerical.event_location_quality", "Event-location quality", "integrator event roots available" if any(np.asarray(item).size for item in (getattr(result, "t_events", None) or ())) else None, None, source="PropagationResult.t_events", kind="measured", unavailable_reason="No integrator event-root quality record was exposed."),
        _metric("physics.gravity_backend.requested", "Requested gravity backend", requested_gravity_backend, None, source="GravityConfig", kind="configuration", unavailable_reason="Requested gravity backend was unavailable."),
        _metric("physics.rhs_path.effective", "Effective RHS path", effective_rhs, None, source="propagation diagnostics", kind="measured", unavailable_reason="Effective RHS path was not reported."),
        _metric("physics.gravity_degree", "Spherical-harmonic degree", int(degree) if degree is not None else None, "1", source="propagation diagnostics or config", kind="configuration", unavailable_reason="SH degree was unavailable."),
        _metric("physics.gravity_degree.recommended", "Recommended SH degree", int(recommended_degree) if recommended_degree is not None else None, "1", source="propagation diagnostics altitude adequacy policy", kind="diagnostic", status="warning" if degree_is_low else "ok", interpretation="A value above the active degree indicates gravity truncation risk at the reported periapsis altitude." if degree_is_low else None, unavailable_reason="No degree recommendation was produced for this run."),
        _metric("physics.active_force_models", "Active force models", ", ".join(force_id for force_id, enabled in flags.items() if enabled) or "point_mass_gravity", None, source="PerturbationFlags", kind="configuration"),
        _metric("diagnostic.energy.initial", "Initial specific orbital energy", float(energy[0]), "J/kg", source="full-resolution state history", kind=invariant_kind, frame=frame, interpretation=conservation_note),
        _metric("diagnostic.energy.final", "Final specific orbital energy", float(energy[-1]), "J/kg", source="full-resolution state history", kind=invariant_kind, frame=frame, interpretation=conservation_note),
        _metric("diagnostic.energy.max_relative_drift", "Maximum relative energy drift", float(np.max(np.abs(rel_energy))), "1", source="full-resolution state history", kind=invariant_kind, frame=frame, interpretation=conservation_note),
        _metric("diagnostic.angular_momentum.initial", "Initial angular momentum norm", float(angmom[0]), "m^2/s", source="full-resolution state history", kind=invariant_kind, frame=frame, interpretation=conservation_note),
        _metric("diagnostic.angular_momentum.final", "Final angular momentum norm", float(angmom[-1]), "m^2/s", source="full-resolution state history", kind=invariant_kind, frame=frame, interpretation=conservation_note),
        _metric("diagnostic.angular_momentum.max_relative_drift", "Maximum relative angular-momentum drift", float(np.max(np.abs(rel_angmom))), "1", source="full-resolution state history", kind=invariant_kind, frame=frame, interpretation=conservation_note),
        _metric("diagnostic.eccentricity_vector.initial", "Initial eccentricity-vector norm", float(e_norm[0]), "1", source="full-resolution state history", kind="diagnostic", frame=frame),
        _metric("diagnostic.eccentricity_vector.final", "Final eccentricity-vector norm", float(e_norm[-1]), "1", source="full-resolution state history", kind="diagnostic", frame=frame),
        _metric("diagnostic.energy.secular_drift", "Specific-energy linear trend", _linear_slope_per_day(t_s, energy), "J/kg/day", source="full-resolution least-squares fit", kind=invariant_kind, frame=frame, interpretation=conservation_note, unavailable_reason="Insufficient finite history for a linear fit."),
        _metric("diagnostic.jacobi_like", "Jacobi-like invariant", None, "1", source="unavailable", kind="diagnostic", frame=frame, interpretation="Only meaningful with an explicitly defined rotating frame and effective potential.", unavailable_reason="The propagation contract does not expose a rotating-frame Jacobi definition."),
    ]

    (
        force_contributions,
        force_time_s,
        force_series,
        force_vectors,
        force_ric,
        force_warnings,
    ) = _force_budget(
        t_s=t_s,
        state=state,
        ctx=ctx,
        config=config,
        preset=normalized_preset,
        radius_m=radius_m,
    )
    warnings: list[str] = list(force_warnings)
    if circular_start or circular_end:
        warnings.append("Circular-orbit singularity detected; argument of periapsis is reported as unavailable where undefined.")
    if equatorial_start or equatorial_end:
        warnings.append("Equatorial-orbit singularity detected; RAAN is reported as unavailable where undefined.")
    if noncentral_active:
        warnings.append(conservation_note)
    if bool(diagnostics.get("symplectic_violation")):
        warnings.append("Propagation diagnostics reported a symplectic-method compatibility violation.")
    if diagnostics.get("single_run_stlrps_cpu_warning"):
        warnings.append("Single-run ST-LRPS used the interpreted CPU/autograd path.")
    if degree_is_low and degree is not None and recommended_degree is not None:
        warnings.append(
            f"Active SH degree {int(degree)} is below the altitude-based recommendation "
            f"of {int(recommended_degree)}; gravity truncation may dominate position error."
        )

    metrics.append(
        _metric(
            "numerical.warning_count",
            "Analysis warning count",
            len(tuple(dict.fromkeys(warnings))),
            "1",
            source="canonical analysis warning list",
            kind="derived",
            status="warning" if warnings else "ok",
        )
    )

    provenance, config_snapshot = _provenance(
        config,
        diagnostics,
        frame,
        meta_map,
        state[0, :6],
    )
    identifier = run_id or f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    return OrbitAnalysisResult(
        run_id=identifier,
        preset=normalized_preset,
        generated_at_utc=utc_now_iso(),
        frame=frame,
        time_system=time_system,
        metrics=tuple(metrics),
        events=tuple(events),
        series=series,
        force_contributions=force_contributions,
        force_time_s=force_time_s,
        force_magnitudes_m_s2=force_series,
        force_vectors_m_s2=force_vectors,
        force_ric_m_s2=force_ric,
        provenance=provenance,
        config_snapshot=config_snapshot,
        diagnostics_snapshot=_jsonable(diagnostics),
        warnings=tuple(dict.fromkeys(warnings)),
    )


__all__ = ["build_orbit_analysis"]
