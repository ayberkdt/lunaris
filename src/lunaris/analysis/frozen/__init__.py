"""Frozen-orbit screening, classification, and search-pipeline utilities.

Lightweight members (metrics, classification, domain guard, schema) are
re-exported eagerly. The pipeline/backends/plots members pull heavier optional
dependencies (scipy.optimize, matplotlib, torch) and stay behind their own
modules: import them as ``lunaris.analysis.frozen.search`` etc.
"""

from __future__ import annotations

from .classify import (
    CANDIDATE_FROZEN_ORBIT,
    LONG_LIVED_NOT_FROZEN,
    QUASI_FROZEN,
    QUASI_FROZEN_CANDIDATE,
    STRICT_FROZEN,
    UNSTABLE_INVALID,
    FrozenCandidateClassification,
    FrozenClassificationConfig,
    classify_candidate,
    frozen_score,
    is_classical_validation_backend,
)
from .domain_guard import (
    DomainGuardResult,
    FrozenSearchDomainGuard,
    apply_domain_guard_to_scores,
    assert_candidate_domain_clean,
    evaluate_domain_guard,
)
from .family_report import (
    FAMILY_REPORT_SCHEMA_VERSION,
    build_family_report,
    group_candidates_into_families,
    validate_family_report,
)
from .metrics import FrozenOrbitMetrics, compute_frozen_metrics

__all__ = [
    "CANDIDATE_FROZEN_ORBIT",
    "FAMILY_REPORT_SCHEMA_VERSION",
    "LONG_LIVED_NOT_FROZEN",
    "QUASI_FROZEN",
    "QUASI_FROZEN_CANDIDATE",
    "STRICT_FROZEN",
    "UNSTABLE_INVALID",
    "DomainGuardResult",
    "FrozenCandidateClassification",
    "FrozenClassificationConfig",
    "FrozenOrbitMetrics",
    "FrozenSearchDomainGuard",
    "apply_domain_guard_to_scores",
    "assert_candidate_domain_clean",
    "build_family_report",
    "classify_candidate",
    "compute_frozen_metrics",
    "evaluate_domain_guard",
    "frozen_score",
    "group_candidates_into_families",
    "is_classical_validation_backend",
    "validate_family_report",
]
