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
) -> str | None:
    """Return an error string if a paper-safe run's evidence is non-scientific.

    A synthetic/quick run can never be paper-safe evidence. This is a hard
    error. Trajectory-only-vs-field is deliberately NOT an error here: an orbit
    benchmark reporting propagated trajectory error is legitimate evidence for
    *trajectory* accuracy — it just must not be *labeled* field accuracy, which
    the taxonomy block and the ``trajectory_error_only`` warning make explicit.
    """
    if not paper_safe:
        return None
    if not taxonomy.get("scientific_evidence", False):
        return (
            "paper_safe run has synthetic/non-scientific evidence taxonomy; "
            "synthetic output can never be paper-safe field evidence"
        )
    return None


__all__ = [
    "MODEL_ERROR_FIELD",
    "INTEGRATOR_ERROR",
    "TRAJECTORY_ERROR",
    "PHASE_CORRECTED_ERROR",
    "RUNTIME_METRICS",
    "EVIDENCE_CATEGORIES",
    "classify_metric",
    "summarize_evidence_taxonomy",
    "field_evidence_error_for_paper_safe",
]
