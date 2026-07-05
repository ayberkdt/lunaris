"""Candidate-family JSON report for the frozen-orbit search (roadmap R30).

One versioned, schema-checked JSON document summarizes the candidate families
that survive screening + validation. The schema is enforced by
:func:`validate_family_report` (pure-Python, no jsonschema dependency), and the
R21 language rule is checked structurally: a family may carry a validated
frozen status (``strict_frozen`` / ``quasi_frozen``) only when its
``validation_backend`` is a classical spherical-harmonics backend.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np

from .classify import (
    QUASI_FROZEN,
    STRICT_FROZEN,
    is_classical_validation_backend,
)

FAMILY_REPORT_SCHEMA_VERSION = 1

_VALIDATED_STATUSES = (STRICT_FROZEN, QUASI_FROZEN)

# Field name -> allowed types. `None` entries in _NULLABLE may also be null.
_FAMILY_REQUIRED_FIELDS: dict[str, tuple[type, ...]] = {
    "family_id": (str,),
    "status": (str,),
    "screening_backend": (str,),
    "validation_backend": (str,),
    "gravity_model": (dict,),
    "third_body": (dict,),
    "validation_days": (int, float),
    "member_count": (int,),
    "member_sample_indices": (list,),
    "element_ranges": (dict,),
    "stability_metrics": (dict,),
    "provenance": (dict,),
}
_FAMILY_NULLABLE = frozenset({"validation_backend", "validation_days"})

_ELEMENT_RANGE_KEYS = ("a_km", "e", "i_deg", "argp_deg")

_TOP_REQUIRED_FIELDS: dict[str, tuple[type, ...]] = {
    "schema_version": (int,),
    "generated_at": (str,),
    "score_definition": (str,),
    "run_id": (str,),
    "families": (list,),
}


def _fail(path: str, message: str) -> None:
    raise ValueError(f"family report schema violation at {path}: {message}")


def _check_fields(
    obj: dict[str, Any],
    required: dict[str, tuple[type, ...]],
    nullable: frozenset[str],
    path: str,
) -> None:
    for key, types in required.items():
        if key not in obj:
            _fail(path, f"missing required field {key!r}")
        value = obj[key]
        if value is None:
            if key not in nullable:
                _fail(f"{path}.{key}", "must not be null")
            continue
        if not isinstance(value, types):
            _fail(
                f"{path}.{key}",
                f"expected {' or '.join(t.__name__ for t in types)}, "
                f"got {type(value).__name__}",
            )


def validate_family_report(payload: dict[str, Any]) -> None:
    """Validate a family report document; raise ``ValueError`` on violation."""
    if not isinstance(payload, dict):
        raise ValueError("family report must be a JSON object")
    _check_fields(payload, _TOP_REQUIRED_FIELDS, frozenset(), "$")
    if int(payload["schema_version"]) != FAMILY_REPORT_SCHEMA_VERSION:
        _fail(
            "$.schema_version",
            f"expected {FAMILY_REPORT_SCHEMA_VERSION}, got {payload['schema_version']}",
        )
    for idx, family in enumerate(payload["families"]):
        path = f"$.families[{idx}]"
        if not isinstance(family, dict):
            _fail(path, "must be an object")
        _check_fields(family, _FAMILY_REQUIRED_FIELDS, _FAMILY_NULLABLE, path)
        for key in _ELEMENT_RANGE_KEYS:
            rng = family["element_ranges"].get(key)
            if (
                not isinstance(rng, (list, tuple))
                or len(rng) != 2
                or not all(isinstance(v, (int, float)) for v in rng)
            ):
                _fail(f"{path}.element_ranges.{key}", "must be a [lo, hi] number pair")
            if float(rng[0]) > float(rng[1]):
                _fail(f"{path}.element_ranges.{key}", f"lo > hi: {rng}")
        if int(family["member_count"]) != len(family["member_sample_indices"]):
            _fail(
                f"{path}.member_count",
                "does not match len(member_sample_indices)",
            )
        # R21 structural rule: validated frozen statuses require classical SH.
        status = str(family["status"])
        if status in _VALIDATED_STATUSES and not is_classical_validation_backend(
            family.get("validation_backend")
        ):
            _fail(
                f"{path}.status",
                f"{status!r} requires a classical SH validation_backend "
                f"(got {family.get('validation_backend')!r})",
            )


def _finite_range(values: list[float]) -> list[float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return [float("nan"), float("nan")]
    return [float(arr.min()), float(arr.max())]


def build_family_report(
    *,
    run_id: str,
    score_definition: str,
    families: list[dict[str, Any]],
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Assemble + validate the top-level family report document."""
    payload = {
        "schema_version": FAMILY_REPORT_SCHEMA_VERSION,
        "generated_at": generated_at
        or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "score_definition": str(score_definition),
        "run_id": str(run_id),
        "families": families,
    }
    validate_family_report(payload)
    return payload


def group_candidates_into_families(
    candidates: list[dict[str, Any]],
    *,
    screening_backend: str,
    gravity_model: dict[str, Any],
    third_body: dict[str, Any],
    provenance: dict[str, Any],
    a_bin_km: float = 100.0,
    i_bin_deg: float = 5.0,
) -> list[dict[str, Any]]:
    """Group validated candidate records into families by (a, i) bins.

    Each candidate record must carry ``elements`` (dict with a_km/e/i_deg/
    argp_deg), ``classification`` (from ``classify_candidate().to_dict()``),
    ``sample_index``, and optionally ``validation`` metadata. The family status
    is the *weakest* member status ordering-wise (a family is only as validated
    as its least-validated member), which keeps the R21 rule conservative.
    """
    if a_bin_km <= 0.0 or i_bin_deg <= 0.0:
        raise ValueError("a_bin_km and i_bin_deg must be positive")

    status_rank = {
        STRICT_FROZEN: 0,
        QUASI_FROZEN: 1,
        "candidate_frozen_orbit": 2,
        "quasi_frozen_candidate": 3,
        "long_lived_not_frozen": 4,
        "unstable_invalid": 5,
    }

    def _bin_key(record: dict[str, Any]) -> tuple[int, int]:
        el = record["elements"]
        return (
            int(np.floor(float(el["a_km"]) / a_bin_km)),
            int(np.floor(float(el["i_deg"]) / i_bin_deg)),
        )

    grouped: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for record in candidates:
        grouped.setdefault(_bin_key(record), []).append(record)

    families: list[dict[str, Any]] = []
    for fam_idx, (_key, members) in enumerate(sorted(grouped.items()), start=1):
        statuses = [str(m["classification"]["status"]) for m in members]
        weakest = max(statuses, key=lambda s: status_rank.get(s, 99))
        backends = {
            m["classification"].get("validation_backend") for m in members
        } - {None}
        validation_backend = sorted(str(b) for b in backends)[0] if backends else None
        validation_days = None
        days = [
            float(m["validation"]["duration_days"])
            for m in members
            if isinstance(m.get("validation"), dict)
            and m["validation"].get("duration_days") is not None
        ]
        if days:
            validation_days = float(min(days))

        elements = {
            key: _finite_range([float(m["elements"][key]) for m in members])
            for key in _ELEMENT_RANGE_KEYS
        }
        scores = [float(m["classification"]["score"]) for m in members]
        metric_pool = [m.get("metrics", {}) for m in members]

        def _metric_max(name: str, pool: list[dict[str, Any]] = metric_pool) -> float:
            vals = [
                float(mp[name])
                for mp in pool
                if name in mp and np.isfinite(float(mp[name]))
            ]
            return float(max(vals)) if vals else float("nan")

        families.append(
            {
                "family_id": f"F{fam_idx:03d}",
                "status": weakest,
                "screening_backend": str(screening_backend),
                "validation_backend": validation_backend,
                "gravity_model": dict(gravity_model),
                "third_body": dict(third_body),
                "validation_days": validation_days,
                "member_count": len(members),
                "member_sample_indices": [int(m["sample_index"]) for m in members],
                "element_ranges": elements,
                "stability_metrics": {
                    "score_min": float(np.nanmin(scores)) if scores else float("nan"),
                    "score_max": float(np.nanmax(scores)) if scores else float("nan"),
                    "e_range_max": _metric_max("e_range"),
                    "h_peri_range_m_max": _metric_max("h_peri_range_m"),
                    "omega_span_rad_max": _metric_max("omega_span_rad"),
                },
                "provenance": dict(provenance),
            }
        )
    return families


__all__ = [
    "FAMILY_REPORT_SCHEMA_VERSION",
    "build_family_report",
    "group_candidates_into_families",
    "validate_family_report",
]
