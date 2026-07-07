"""Benchmark evidence taxonomy (scientific-hardening Phase 5).

An orbit-level position error mixes several independent error sources: the
surrogate gravity *field* error, the *integrator* error, dtype/frame/output
interpolation error, and domain extrapolation. Collapsing them into one generic
``surrogate error`` lets a trajectory number masquerade as a field-accuracy
claim. This module gives the benchmark a small, explicit schema so an artifact
declares *which* error category each metric belongs to, and so paper-safe mode
can require real field-level evidence rather than trajectory error alone.

The five categories (see ``EVIDENCE_CATEGORIES``):

* ``model_error_field``     — gravity acceleration error at fixed Moon-fixed
  points (truth field vs surrogate field, same points). Proves field accuracy.
* ``integrator_error``      — truth-field error from the integrator alone
  (e.g. SH200 RK4 vs SH200 DOP853). Isolates numerics, not the surrogate.
* ``trajectory_error``      — propagated orbit position/velocity error. Mixes
  field + integrator + dtype + frame + interpolation. NOT a field claim.
* ``phase_corrected_error`` — trajectory error after removing along-track phase
  drift. Separates shape error from timing error.
* ``runtime_metrics``       — throughput/timing. Never an accuracy claim.

This module is pure schema + classification; it does not compute metrics.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

MODEL_ERROR_FIELD = "model_error_field"
INTEGRATOR_ERROR = "integrator_error"
TRAJECTORY_ERROR = "trajectory_error"
PHASE_CORRECTED_ERROR = "phase_corrected_error"
RUNTIME_METRICS = "runtime_metrics"

# --- Paper-safe claim taxonomy ------------------------------------------------
# What a paper-safe benchmark is *claiming*. This decides whether field-level
# evidence (model_error_field) is mandatory. A trajectory benchmark that never
# claims field accuracy is legitimate; an ST-LRPS field/full-surrogate claim is
# not defensible on trajectory error alone.
CLAIM_TRAJECTORY_ONLY = "trajectory_only"
CLAIM_FIELD_ACCURACY = "field_accuracy"
CLAIM_FULL_SURROGATE_VALIDATION = "full_surrogate_validation"

PAPER_SAFE_CLAIM_TYPES: frozenset[str] = frozenset(
    {CLAIM_TRAJECTORY_ONLY, CLAIM_FIELD_ACCURACY, CLAIM_FULL_SURROGATE_VALIDATION}
)

#: Default when a paper-safe benchmark does not declare a claim type. The strict
#: default: a paper-safe ST-LRPS benchmark implies a field claim, so field-level
#: evidence is required unless the author explicitly downgrades to
#: ``trajectory_only``.
DEFAULT_PAPER_SAFE_CLAIM_TYPE = CLAIM_FULL_SURROGATE_VALIDATION

#: Claim types that make model_error_field evidence mandatory.
FIELD_REQUIRING_CLAIM_TYPES: frozenset[str] = frozenset(
    {CLAIM_FIELD_ACCURACY, CLAIM_FULL_SURROGATE_VALIDATION}
)


def normalize_claim_type(value: Any) -> str:
    """Resolve/validate a paper-safe claim type, falling back to the default.

    ``None``/empty resolves to :data:`DEFAULT_PAPER_SAFE_CLAIM_TYPE`. An
    unrecognized value raises ``ValueError`` (fail closed: a typo must not
    silently downgrade the field-evidence requirement).
    """
    if value is None or str(value).strip() == "":
        return DEFAULT_PAPER_SAFE_CLAIM_TYPE
    resolved = str(value).strip().lower()
    if resolved not in PAPER_SAFE_CLAIM_TYPES:
        raise ValueError(
            f"unknown paper_safe_claim_type {value!r}; expected one of "
            + ", ".join(sorted(PAPER_SAFE_CLAIM_TYPES))
        )
    return resolved


def claim_type_requires_field_evidence(claim_type: Any) -> bool:
    """True when the claim type makes model_error_field evidence mandatory."""
    return normalize_claim_type(claim_type) in FIELD_REQUIRING_CLAIM_TYPES

#: Category -> human description + whether it constitutes ST-LRPS *field* evidence.
EVIDENCE_CATEGORIES: dict[str, dict[str, Any]] = {
    MODEL_ERROR_FIELD: {
        "proves_field_accuracy": True,
        "description": (
            "Gravity acceleration error at fixed Moon-fixed points (truth field "
            "vs surrogate field). The only category that proves field accuracy."
        ),
    },
    INTEGRATOR_ERROR: {
        "proves_field_accuracy": False,
        "description": (
            "Truth-field error from the integrator alone (same field, RK4 vs "
            "DOP853). Isolates numerical error; independent of the surrogate."
        ),
    },
    TRAJECTORY_ERROR: {
        "proves_field_accuracy": False,
        "description": (
            "Propagated orbit position/velocity error. Mixes field, integrator, "
            "dtype, frame, and interpolation error; not a field-accuracy claim."
        ),
    },
    PHASE_CORRECTED_ERROR: {
        "proves_field_accuracy": False,
        "description": (
            "Trajectory error after removing along-track phase drift. Separates "
            "orbit-shape error from timing error; still an orbit-level metric."
        ),
    },
    RUNTIME_METRICS: {
        "proves_field_accuracy": False,
        "description": "Throughput and timing. Never an accuracy claim.",
    },
}

# Substrings (matched case-insensitively) that map a metric/column/key name to a
# category. Order matters: the first match wins, so more specific field/phase/
# integrator markers are checked before the generic trajectory markers.
_CLASSIFY_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (RUNTIME_METRICS, ("runtime", "throughput", "steps_per_s", "per_second",
                       "samples_per_s", "wall", "eval_time", "cold", "warm")),
    (MODEL_ERROR_FIELD, ("model_error", "field_error", "accel_error", "accel_rms",
                         "acceleration_error", "mgal", "field_rms")),
    (INTEGRATOR_ERROR, ("integrator_error", "rk4_vs_dop853", "integration_error")),
    (PHASE_CORRECTED_ERROR, ("phase_corrected", "phase_removed", "detrended",
                             "phase_drift")),
    (TRAJECTORY_ERROR, ("pos_err", "vel_err", "position_error", "velocity_error",
                        "radial_rms", "along_rms", "cross_rms", "alt_err",
                        "trajectory", "ric")),
)


def classify_metric(name: str) -> str | None:
    """Return the evidence category for a metric/column name, or ``None``.

    ``None`` means the name does not match any known category (e.g. an id or a
    label column); callers should treat unclassified numeric metrics as
    non-evidence rather than silently promoting them.
    """
    lowered = str(name).strip().lower()
    for category, markers in _CLASSIFY_RULES:
        if any(marker in lowered for marker in markers):
            return category
    return None


def summarize_evidence_taxonomy(
    metric_names: Any,
    *,
    synthetic: bool = False,
) -> dict[str, Any]:
    """Bucket metric names by evidence category and describe what they prove.

    Parameters
    ----------
    metric_names:
        Iterable of metric/column names present in the benchmark output.
    synthetic:
        When True the run is a synthetic smoke test; the taxonomy is stamped
        ``scientific_evidence=False`` so a schema-only run can never be read as
        field evidence.

    Returns a self-describing block suitable for embedding in a validation or
    manifest artifact.
    """
    buckets: dict[str, list[str]] = {cat: [] for cat in EVIDENCE_CATEGORIES}
    unclassified: list[str] = []
    for name in metric_names:
        category = classify_metric(name)
        if category is None:
            unclassified.append(str(name))
        else:
            buckets[category].append(str(name))

    has_field_evidence = bool(buckets[MODEL_ERROR_FIELD])
    return {
        "schema_version": 1,
        "categories": {
            cat: {
                "metrics": sorted(buckets[cat]),
                "proves_field_accuracy": EVIDENCE_CATEGORIES[cat]["proves_field_accuracy"],
                "description": EVIDENCE_CATEGORIES[cat]["description"],
            }
            for cat in EVIDENCE_CATEGORIES
        },
        "unclassified_metrics": sorted(unclassified),
        "has_field_level_evidence": has_field_evidence,
        "trajectory_error_only": (
            bool(buckets[TRAJECTORY_ERROR]) and not has_field_evidence
        ),
        "scientific_evidence": bool(not synthetic),
    }


def field_evidence_error_for_paper_safe(
    taxonomy: Mapping[str, Any],
    *,
    paper_safe: bool,
    claim_type: Any = None,
) -> str | None:
    """Return an error string if a paper-safe run's evidence is non-scientific.

    A synthetic/quick run can never be paper-safe evidence — always a hard
    error. Beyond that, whether *field-level* evidence is mandatory depends on
    ``claim_type`` (see :data:`PAPER_SAFE_CLAIM_TYPES`):

    * ``trajectory_only`` — trajectory error is legitimate evidence for
      *trajectory* accuracy; missing field evidence is NOT an error (it is only
      surfaced as the ``trajectory_error_only`` warning). It must simply never
      be *labeled* field accuracy.
    * ``field_accuracy`` / ``full_surrogate_validation`` — the run claims
      ST-LRPS gravity-field accuracy, so ``model_error_field`` evidence is
      mandatory: a field claim on trajectory error alone is a hard error.

    ``claim_type=None`` resolves to :data:`DEFAULT_PAPER_SAFE_CLAIM_TYPE`.
    """
    if not paper_safe:
        return None
    if not taxonomy.get("scientific_evidence", False):
        return (
            "paper_safe run has synthetic/non-scientific evidence taxonomy; "
            "synthetic output can never be paper-safe field evidence"
        )
    resolved_claim = normalize_claim_type(claim_type)
    if resolved_claim in FIELD_REQUIRING_CLAIM_TYPES and not taxonomy.get(
        "has_field_level_evidence", False
    ):
        return (
            f"paper_safe run declares claim_type={resolved_claim!r} but carries no "
            "model_error_field evidence (only orbit-level trajectory metrics); "
            "ST-LRPS field accuracy cannot be claimed from trajectory error alone. "
            "Add gravity-field error metrics or set claim_type='trajectory_only'."
        )
    return None


__all__ = [
    "MODEL_ERROR_FIELD",
    "INTEGRATOR_ERROR",
    "TRAJECTORY_ERROR",
    "PHASE_CORRECTED_ERROR",
    "RUNTIME_METRICS",
    "EVIDENCE_CATEGORIES",
    "CLAIM_TRAJECTORY_ONLY",
    "CLAIM_FIELD_ACCURACY",
    "CLAIM_FULL_SURROGATE_VALIDATION",
    "PAPER_SAFE_CLAIM_TYPES",
    "DEFAULT_PAPER_SAFE_CLAIM_TYPE",
    "FIELD_REQUIRING_CLAIM_TYPES",
    "normalize_claim_type",
    "claim_type_requires_field_evidence",
    "classify_metric",
    "summarize_evidence_taxonomy",
    "field_evidence_error_for_paper_safe",
]
