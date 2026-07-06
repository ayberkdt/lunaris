"""Frozen-orbit candidate classification.

Validated frozen labels are intentionally harder to obtain than candidate
labels. Good screening metrics without classical-SH long-horizon validation are
reported as candidate language, not as discovered frozen families.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .metrics import FrozenOrbitMetrics

STRICT_FROZEN = "strict_frozen"
QUASI_FROZEN = "quasi_frozen"
CANDIDATE_FROZEN_ORBIT = "candidate_frozen_orbit"
QUASI_FROZEN_CANDIDATE = "quasi_frozen_candidate"
LONG_LIVED_NOT_FROZEN = "long_lived_not_frozen"
UNSTABLE_INVALID = "unstable_invalid"

_LABELS = {
    STRICT_FROZEN: "strict frozen",
    QUASI_FROZEN: "quasi-frozen",
    CANDIDATE_FROZEN_ORBIT: "candidate frozen orbit",
    QUASI_FROZEN_CANDIDATE: "quasi-frozen candidate",
    LONG_LIVED_NOT_FROZEN: "long-lived not frozen",
    UNSTABLE_INVALID: "unstable/invalid",
}


@dataclass(frozen=True, slots=True)
class FrozenClassificationConfig:
    """Threshold config for frozen-orbit classification.

    The classifier consumes this object rather than embedding mission thresholds
    in the decision tree. Use :meth:`for_mission_duration` for a deterministic
    starting point, then override values from a search/validation config.
    """

    mission_duration_s: float
    min_valid_samples: int = 3
    perilune_safety_min_m: float = 0.0
    eccentricity_upper_bound: float = 0.5

    strict_e_range_max: float = 2.0e-3
    quasi_e_range_max: float = 2.0e-2
    strict_abs_de_dt_max: float = 1.0e-10
    quasi_abs_de_dt_max: float = 1.0e-9

    strict_h_peri_range_max_m: float = 5_000.0
    quasi_h_peri_range_max_m: float = 30_000.0
    strict_abs_dh_peri_dt_max_m_per_s: float = 5.0e-3
    quasi_abs_dh_peri_dt_max_m_per_s: float = 5.0e-2

    strict_inclination_range_max_rad: float = np.deg2rad(0.25)
    quasi_inclination_range_max_rad: float = np.deg2rad(2.0)
    strict_abs_dinclination_dt_max_rad_per_s: float = 1.0e-10
    quasi_abs_dinclination_dt_max_rad_per_s: float = 1.0e-9

    strict_omega_span_max_rad: float = np.deg2rad(20.0)
    quasi_omega_span_max_rad: float = np.deg2rad(120.0)
    strict_abs_domega_dt_max_rad_per_s: float = 1.0e-9
    quasi_abs_domega_dt_max_rad_per_s: float = 1.0e-8

    strict_hk_loop_drift_max: float = 2.0e-3
    quasi_hk_loop_drift_max: float = 2.0e-2

    @classmethod
    def for_mission_duration(
        cls,
        mission_duration_s: float,
        *,
        perilune_safety_min_m: float = 0.0,
        eccentricity_upper_bound: float = 0.5,
        strict_e_range_max: float = 2.0e-3,
        quasi_e_range_max: float = 2.0e-2,
        strict_h_peri_range_max_m: float = 5_000.0,
        quasi_h_peri_range_max_m: float = 30_000.0,
        strict_omega_span_max_rad: float = np.deg2rad(20.0),
        quasi_omega_span_max_rad: float = np.deg2rad(120.0),
    ) -> FrozenClassificationConfig:
        duration = float(mission_duration_s)
        if duration <= 0.0 or not np.isfinite(duration):
            raise ValueError("mission_duration_s must be positive and finite")
        return cls(
            mission_duration_s=duration,
            perilune_safety_min_m=float(perilune_safety_min_m),
            eccentricity_upper_bound=float(eccentricity_upper_bound),
            strict_e_range_max=float(strict_e_range_max),
            quasi_e_range_max=float(quasi_e_range_max),
            strict_abs_de_dt_max=float(strict_e_range_max) / duration,
            quasi_abs_de_dt_max=float(quasi_e_range_max) / duration,
            strict_h_peri_range_max_m=float(strict_h_peri_range_max_m),
            quasi_h_peri_range_max_m=float(quasi_h_peri_range_max_m),
            strict_abs_dh_peri_dt_max_m_per_s=float(strict_h_peri_range_max_m)
            / duration,
            quasi_abs_dh_peri_dt_max_m_per_s=float(quasi_h_peri_range_max_m)
            / duration,
            strict_omega_span_max_rad=float(strict_omega_span_max_rad),
            quasi_omega_span_max_rad=float(quasi_omega_span_max_rad),
            strict_abs_domega_dt_max_rad_per_s=float(strict_omega_span_max_rad)
            / duration,
            quasi_abs_domega_dt_max_rad_per_s=float(quasi_omega_span_max_rad)
            / duration,
            strict_hk_loop_drift_max=float(strict_e_range_max),
            quasi_hk_loop_drift_max=float(quasi_e_range_max),
        )

    def __post_init__(self) -> None:
        if not np.isfinite(self.mission_duration_s) or self.mission_duration_s <= 0.0:
            raise ValueError("mission_duration_s must be positive and finite")
        if int(self.min_valid_samples) < 2:
            raise ValueError("min_valid_samples must be at least 2")
        ordered_pairs = (
            ("e_range", self.strict_e_range_max, self.quasi_e_range_max),
            ("abs_de_dt", self.strict_abs_de_dt_max, self.quasi_abs_de_dt_max),
            ("h_peri_range", self.strict_h_peri_range_max_m, self.quasi_h_peri_range_max_m),
            (
                "abs_dh_peri_dt",
                self.strict_abs_dh_peri_dt_max_m_per_s,
                self.quasi_abs_dh_peri_dt_max_m_per_s,
            ),
            (
                "inclination_range",
                self.strict_inclination_range_max_rad,
                self.quasi_inclination_range_max_rad,
            ),
            (
                "abs_dinclination_dt",
                self.strict_abs_dinclination_dt_max_rad_per_s,
                self.quasi_abs_dinclination_dt_max_rad_per_s,
            ),
            ("omega_span", self.strict_omega_span_max_rad, self.quasi_omega_span_max_rad),
            (
                "abs_domega_dt",
                self.strict_abs_domega_dt_max_rad_per_s,
                self.quasi_abs_domega_dt_max_rad_per_s,
            ),
            ("hk_loop_drift", self.strict_hk_loop_drift_max, self.quasi_hk_loop_drift_max),
        )
        for name, strict, quasi in ordered_pairs:
            if strict < 0.0 or quasi < 0.0 or not np.isfinite(strict) or not np.isfinite(quasi):
                raise ValueError(f"{name} thresholds must be non-negative and finite")
            if strict > quasi:
                raise ValueError(f"{name} strict threshold must be <= quasi threshold")
        if self.perilune_safety_min_m < 0.0:
            raise ValueError("perilune_safety_min_m must be non-negative")
        if self.eccentricity_upper_bound <= 0.0:
            raise ValueError("eccentricity_upper_bound must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class FrozenCandidateClassification:
    """Classification result for one candidate."""

    status: str
    display_label: str
    score: float
    validated: bool
    validation_backend: str | None
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def is_classical_validation_backend(validation_backend: str | None) -> bool:
    if validation_backend is None:
        return False
    token = str(validation_backend).strip().lower().replace("-", "_")
    return (
        token.startswith("classical_sh")
        or token.startswith("classic_sh")
        or token.startswith("cpu_sh")
    )


def _finite(value: float) -> bool:
    return bool(np.isfinite(float(value)))


def _ratio(value: float, scale: float) -> float:
    if not _finite(value) or not _finite(scale) or scale <= 0.0:
        return float("inf")
    return abs(float(value)) / float(scale)


def _invalid_reasons(
    metrics: FrozenOrbitMetrics,
    config: FrozenClassificationConfig,
) -> list[str]:
    reasons: list[str] = []
    if metrics.valid_sample_count < int(config.min_valid_samples):
        reasons.append("insufficient valid samples")
    if metrics.has_impact:
        reasons.append("impact detected")
    if metrics.has_domain_exit:
        reasons.append("domain exit detected")
    if metrics.escape:
        reasons.append("escape detected")
    critical = (
        metrics.e_min,
        metrics.e_max,
        metrics.e_range,
        metrics.h_peri_min_m,
        metrics.h_peri_max_m,
        metrics.h_peri_range_m,
    )
    if any(not _finite(v) for v in critical):
        reasons.append("non-finite core metrics")
    if _finite(metrics.e_max) and metrics.e_max > config.eccentricity_upper_bound:
        reasons.append("eccentricity growth limit exceeded")
    if _finite(metrics.h_peri_min_m) and metrics.h_peri_min_m < config.perilune_safety_min_m:
        reasons.append("perilune below safety threshold")
    return reasons


def _bounded_failures(
    metrics: FrozenOrbitMetrics,
    *,
    e_range_max: float,
    abs_de_dt_max: float,
    h_peri_range_max_m: float,
    abs_dh_peri_dt_max_m_per_s: float,
    inclination_range_max_rad: float,
    abs_dinclination_dt_max_rad_per_s: float,
    omega_span_max_rad: float,
    abs_domega_dt_max_rad_per_s: float,
    hk_loop_drift_max: float,
    allow_mixed_omega: bool,
) -> list[str]:
    failures: list[str] = []
    if _ratio(metrics.e_range, e_range_max) > 1.0:
        failures.append("eccentricity envelope too wide")
    if _ratio(metrics.de_dt_per_s, abs_de_dt_max) > 1.0:
        failures.append("eccentricity secular drift too large")
    if _ratio(metrics.h_peri_range_m, h_peri_range_max_m) > 1.0:
        failures.append("perilune envelope too wide")
    if _ratio(metrics.dh_peri_dt_m_per_s, abs_dh_peri_dt_max_m_per_s) > 1.0:
        failures.append("perilune secular drift too large")
    if _ratio(metrics.inclination_range_rad, inclination_range_max_rad) > 1.0:
        failures.append("inclination envelope too wide")
    if _ratio(metrics.dinclination_dt_rad_per_s, abs_dinclination_dt_max_rad_per_s) > 1.0:
        failures.append("inclination secular drift too large")
    if _ratio(metrics.omega_span_rad, omega_span_max_rad) > 1.0:
        failures.append("omega span too wide")
    if _ratio(metrics.domega_dt_rad_per_s, abs_domega_dt_max_rad_per_s) > 1.0:
        failures.append("omega secular drift too large")
    if _ratio(metrics.hk_loop_drift, hk_loop_drift_max) > 1.0:
        failures.append("eccentricity-vector loop not bounded")
    if metrics.omega_behavior == "circulation":
        failures.append("omega circulates")
    if metrics.omega_behavior == "mixed" and not allow_mixed_omega:
        failures.append("omega behavior is mixed")
    if metrics.omega_behavior == "indeterminate":
        failures.append("omega behavior indeterminate")
    return failures


def _strict_failures(
    metrics: FrozenOrbitMetrics,
    config: FrozenClassificationConfig,
) -> list[str]:
    return _bounded_failures(
        metrics,
        e_range_max=config.strict_e_range_max,
        abs_de_dt_max=config.strict_abs_de_dt_max,
        h_peri_range_max_m=config.strict_h_peri_range_max_m,
        abs_dh_peri_dt_max_m_per_s=config.strict_abs_dh_peri_dt_max_m_per_s,
        inclination_range_max_rad=config.strict_inclination_range_max_rad,
        abs_dinclination_dt_max_rad_per_s=config.strict_abs_dinclination_dt_max_rad_per_s,
        omega_span_max_rad=config.strict_omega_span_max_rad,
        abs_domega_dt_max_rad_per_s=config.strict_abs_domega_dt_max_rad_per_s,
        hk_loop_drift_max=config.strict_hk_loop_drift_max,
        allow_mixed_omega=False,
    )


def _quasi_failures(
    metrics: FrozenOrbitMetrics,
    config: FrozenClassificationConfig,
) -> list[str]:
    return _bounded_failures(
        metrics,
        e_range_max=config.quasi_e_range_max,
        abs_de_dt_max=config.quasi_abs_de_dt_max,
        h_peri_range_max_m=config.quasi_h_peri_range_max_m,
        abs_dh_peri_dt_max_m_per_s=config.quasi_abs_dh_peri_dt_max_m_per_s,
        inclination_range_max_rad=config.quasi_inclination_range_max_rad,
        abs_dinclination_dt_max_rad_per_s=config.quasi_abs_dinclination_dt_max_rad_per_s,
        omega_span_max_rad=config.quasi_omega_span_max_rad,
        abs_domega_dt_max_rad_per_s=config.quasi_abs_domega_dt_max_rad_per_s,
        hk_loop_drift_max=config.quasi_hk_loop_drift_max,
        allow_mixed_omega=True,
    )


def frozen_score(
    metrics: FrozenOrbitMetrics,
    config: FrozenClassificationConfig,
) -> float:
    """Return a lower-is-better score suitable for candidate ranking."""

    if _invalid_reasons(metrics, config):
        return float("inf")
    return float(
        _ratio(metrics.e_range, config.quasi_e_range_max)
        + _ratio(metrics.de_dt_per_s, config.quasi_abs_de_dt_max)
        + _ratio(metrics.h_peri_range_m, config.quasi_h_peri_range_max_m)
        + _ratio(metrics.dh_peri_dt_m_per_s, config.quasi_abs_dh_peri_dt_max_m_per_s)
        + _ratio(metrics.omega_span_rad, config.quasi_omega_span_max_rad)
        + _ratio(metrics.domega_dt_rad_per_s, config.quasi_abs_domega_dt_max_rad_per_s)
        + _ratio(metrics.hk_loop_drift, config.quasi_hk_loop_drift_max)
    )


def _validation_ok(
    validation_backend: str | None,
    long_horizon_validation_passed: bool,
) -> bool:
    return bool(long_horizon_validation_passed) and is_classical_validation_backend(
        validation_backend
    )


def _candidate_reason(
    validation_backend: str | None,
    long_horizon_validation_passed: bool,
) -> str:
    if not is_classical_validation_backend(validation_backend):
        return "classical SH long-horizon validation required for frozen status"
    if not long_horizon_validation_passed:
        return "long-horizon validation has not passed"
    return "classical SH long-horizon validation passed"


def _result(
    status: str,
    *,
    score: float,
    validated: bool,
    validation_backend: str | None,
    reasons: list[str] | tuple[str, ...],
) -> FrozenCandidateClassification:
    return FrozenCandidateClassification(
        status=status,
        display_label=_LABELS[status],
        score=float(score),
        validated=bool(validated),
        validation_backend=validation_backend,
        reasons=tuple(dict.fromkeys(str(r) for r in reasons)),
    )


def classify_candidate(
    metrics: FrozenOrbitMetrics,
    config: FrozenClassificationConfig,
    *,
    validation_backend: str | None = None,
    long_horizon_validation_passed: bool = False,
) -> FrozenCandidateClassification:
    """Classify one candidate from metrics and explicit validation evidence."""

    invalid = _invalid_reasons(metrics, config)
    score = frozen_score(metrics, config)
    if invalid:
        return _result(
            UNSTABLE_INVALID,
            score=float("inf"),
            validated=False,
            validation_backend=validation_backend,
            reasons=invalid,
        )

    strict_failures = _strict_failures(metrics, config)
    validation_ok = _validation_ok(validation_backend, long_horizon_validation_passed)
    validation_reason = _candidate_reason(validation_backend, long_horizon_validation_passed)

    if not strict_failures:
        status = STRICT_FROZEN if validation_ok else CANDIDATE_FROZEN_ORBIT
        return _result(
            status,
            score=score,
            validated=validation_ok,
            validation_backend=validation_backend,
            reasons=[validation_reason],
        )

    quasi_failures = _quasi_failures(metrics, config)
    if not quasi_failures:
        status = QUASI_FROZEN if validation_ok else QUASI_FROZEN_CANDIDATE
        return _result(
            status,
            score=score,
            validated=validation_ok,
            validation_backend=validation_backend,
            reasons=[validation_reason, *strict_failures],
        )

    return _result(
        LONG_LIVED_NOT_FROZEN,
        score=score,
        validated=False,
        validation_backend=validation_backend,
        reasons=quasi_failures,
    )


__all__ = [
    "CANDIDATE_FROZEN_ORBIT",
    "LONG_LIVED_NOT_FROZEN",
    "QUASI_FROZEN",
    "QUASI_FROZEN_CANDIDATE",
    "STRICT_FROZEN",
    "UNSTABLE_INVALID",
    "FrozenCandidateClassification",
    "FrozenClassificationConfig",
    "classify_candidate",
    "frozen_score",
    "is_classical_validation_backend",
]
