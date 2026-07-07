"""ST-LRPS benchmark error-decomposition schema (scientific hardening).

An ST-LRPS benchmark number can come from several independent error sources:
surrogate *field* error, fixed-step integrator error, dtype error, frame
rotation/interpolation error, output resampling error, and along-track phase
drift. A report that collapses them into one vague "surrogate error" lets a
trajectory number masquerade as a field-accuracy claim.

This module defines the versioned decomposition block every paper-safe
benchmark artifact must carry:

* ``field_error``           — truth-field vs surrogate-field acceleration error
  at fixed Moon-fixed points. The only block that proves field accuracy.
* ``orbit_error``           — propagated trajectory position/velocity error
  (mixes field + integrator + dtype + frame + interpolation).
* ``integrator_error``      — truth-field integrator-only error (e.g. RK4 vs
  DOP853 on the same field).
* ``phase_corrected_error`` — trajectory error after removing along-track
  phase drift.
* ``runtime``               — throughput/timing; never an accuracy claim.
* ``provenance``            — requested/actual backend, requested/effective
  dtype, frame mode, identity-rotation flag, synthetic flag, paper-safe flag.

The module is pure schema + validation: it computes no metrics and fabricates
no numbers. Quick/smoke runs may fill placeholder blocks, but they must be
stamped ``synthetic=True`` which forces ``scientific_evidence=False`` and can
never validate as paper-safe. Category descriptions are shared with
:mod:`.benchmark_evidence_taxonomy` so both artifacts tell the same story.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .benchmark_evidence_taxonomy import (
    EVIDENCE_CATEGORIES,
    INTEGRATOR_ERROR,
    MODEL_ERROR_FIELD,
    PHASE_CORRECTED_ERROR,
    RUNTIME_METRICS,
    TRAJECTORY_ERROR,
)

ERROR_DECOMPOSITION_SCHEMA_VERSION = "st_lrps_error_decomposition_v1"

#: Decomposition block name -> evidence-taxonomy category supplying its description.
_BLOCK_CATEGORY: dict[str, str] = {
    "field_error": MODEL_ERROR_FIELD,
    "orbit_error": TRAJECTORY_ERROR,
    "integrator_error": INTEGRATOR_ERROR,
    "phase_corrected_error": PHASE_CORRECTED_ERROR,
    "runtime": RUNTIME_METRICS,
}

REQUIRED_TOP_LEVEL_BLOCKS: tuple[str, ...] = (
    "field_error",
    "orbit_error",
    "integrator_error",
    "phase_corrected_error",
    "runtime",
    "provenance",
)

#: Blocks a paper-safe artifact must actually carry (``present=True``).
_PAPER_SAFE_REQUIRED_PRESENT: tuple[str, ...] = (
    "field_error",
    "orbit_error",
    "runtime",
)

_REQUIRED_PROVENANCE_KEYS: tuple[str, ...] = (
    "requested_backend",
    "actual_backend",
    "requested_dtype",
    "effective_dtype",
    "frame_mode",
    "identity_rotation_used",
    "synthetic",
    "paper_safe",
)


def _metric_block(name: str, metrics: Mapping[str, Any] | None) -> dict[str, Any]:
    return {
        "present": metrics is not None,
        "description": str(EVIDENCE_CATEGORIES[_BLOCK_CATEGORY[name]]["description"]),
        "metrics": dict(metrics) if metrics is not None else {},
    }


def build_error_decomposition_schema(
    *,
    field_error: Mapping[str, Any] | None = None,
    orbit_error: Mapping[str, Any] | None = None,
    integrator_error: Mapping[str, Any] | None = None,
    phase_corrected_error: Mapping[str, Any] | None = None,
    runtime: Mapping[str, Any] | None = None,
    requested_backend: str | None = None,
    actual_backend: str | None = None,
    requested_dtype: str | None = None,
    effective_dtype: str | None = None,
    frame_mode: str | None = None,
    identity_rotation_used: bool = False,
    synthetic: bool = False,
    paper_safe: bool = False,
) -> dict[str, Any]:
    """Build the versioned error-decomposition block.

    Each ``*_error``/``runtime`` argument is the (already computed) metrics
    mapping for that category, or ``None`` when the run did not measure it —
    the block is then emitted with ``present=False`` and empty metrics, so an
    absent measurement is explicit rather than silently missing.

    ``synthetic=True`` marks a quick/smoke schema fill; it forces
    ``scientific_evidence=False`` and the result can never pass
    :func:`validate_paper_safe_error_decomposition`. No default here invents a
    measurement: every ``present=True`` block was supplied by the caller.
    """
    return {
        "schema_version": ERROR_DECOMPOSITION_SCHEMA_VERSION,
        "field_error": _metric_block("field_error", field_error),
        "orbit_error": _metric_block("orbit_error", orbit_error),
        "integrator_error": _metric_block("integrator_error", integrator_error),
        "phase_corrected_error": _metric_block(
            "phase_corrected_error", phase_corrected_error
        ),
        "runtime": _metric_block("runtime", runtime),
        "provenance": {
            "requested_backend": requested_backend,
            "actual_backend": actual_backend,
            "requested_dtype": requested_dtype,
            "effective_dtype": effective_dtype,
            "frame_mode": frame_mode,
            "identity_rotation_used": bool(identity_rotation_used),
            "synthetic": bool(synthetic),
            "paper_safe": bool(paper_safe),
        },
        "scientific_evidence": not bool(synthetic),
    }


def validate_error_decomposition(block: Mapping[str, Any]) -> list[str]:
    """Structural validation for any (paper-safe or quick) decomposition block.

    Returns a list of problems (empty when structurally valid). Quick/smoke
    blocks are structurally valid only when their synthetic marking is honest:
    ``synthetic=True`` requires ``scientific_evidence=False``.
    """
    problems: list[str] = []
    version = block.get("schema_version")
    if version != ERROR_DECOMPOSITION_SCHEMA_VERSION:
        problems.append(
            f"schema_version={version!r}, expected {ERROR_DECOMPOSITION_SCHEMA_VERSION!r}"
        )
    for name in REQUIRED_TOP_LEVEL_BLOCKS:
        sub = block.get(name)
        if not isinstance(sub, Mapping):
            problems.append(f"missing required block {name!r}")
            continue
        if name != "provenance" and "present" not in sub:
            problems.append(f"block {name!r} is missing the 'present' flag")

    provenance = block.get("provenance")
    if isinstance(provenance, Mapping):
        for key in _REQUIRED_PROVENANCE_KEYS:
            if key not in provenance:
                problems.append(f"provenance is missing {key!r}")
        if bool(provenance.get("synthetic", False)) and bool(
            block.get("scientific_evidence", True)
        ):
            problems.append(
                "synthetic=True blocks must carry scientific_evidence=False; a "
                "quick/smoke schema fill can never be labeled scientific evidence"
            )
    return problems


def validate_paper_safe_error_decomposition(block: Mapping[str, Any]) -> None:
    """Reject a decomposition block that cannot back a paper-safe claim.

    Raises ``ValueError`` listing every violation. A paper-safe artifact must
    separate field error from orbit error (both measured), carry runtime and
    full backend/dtype/frame provenance, must not be synthetic, and must not
    have used identity rotation for Moon-fixed gravity.
    """
    problems = validate_error_decomposition(block)

    for name in _PAPER_SAFE_REQUIRED_PRESENT:
        sub = block.get(name)
        if isinstance(sub, Mapping) and not bool(sub.get("present", False)):
            msg = f"paper-safe reports require a measured {name!r} block (present=True)"
            if name == "field_error":
                msg += "; trajectory error alone must not imply field accuracy"
            problems.append(msg)

    provenance = block.get("provenance")
    if isinstance(provenance, Mapping):
        for key in (
            "requested_backend",
            "actual_backend",
            "requested_dtype",
            "effective_dtype",
            "frame_mode",
        ):
            value = provenance.get(key)
            if value is None or str(value).strip() == "":
                problems.append(f"paper-safe provenance must record {key!r}")
        if bool(provenance.get("identity_rotation_used", False)):
            problems.append(
                "identity_rotation_used=True: Moon-fixed gravity evaluated with an "
                "identity rotation can never be paper-safe evidence"
            )
        if bool(provenance.get("synthetic", False)):
            problems.append("synthetic=True output can never be paper-safe evidence")
    if not bool(block.get("scientific_evidence", False)):
        problems.append("scientific_evidence=False blocks are not paper-safe evidence")

    if problems:
        raise ValueError(
            "error decomposition is not paper-safe:\n  - " + "\n  - ".join(problems)
        )


__all__ = [
    "ERROR_DECOMPOSITION_SCHEMA_VERSION",
    "REQUIRED_TOP_LEVEL_BLOCKS",
    "build_error_decomposition_schema",
    "validate_error_decomposition",
    "validate_paper_safe_error_decomposition",
]
