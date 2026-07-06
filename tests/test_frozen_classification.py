"""Sprint 5 R05 frozen/quasi-frozen metric and label contracts."""

from __future__ import annotations

import numpy as np

from lunaris.analysis.frozen import (
    CANDIDATE_FROZEN_ORBIT,
    LONG_LIVED_NOT_FROZEN,
    QUASI_FROZEN,
    QUASI_FROZEN_CANDIDATE,
    STRICT_FROZEN,
    UNSTABLE_INVALID,
    FrozenClassificationConfig,
    classify_candidate,
    compute_frozen_metrics,
    frozen_score,
)

DURATION_S = 14.0 * 86_400.0


def _config() -> FrozenClassificationConfig:
    return FrozenClassificationConfig(
        mission_duration_s=DURATION_S,
        perilune_safety_min_m=20_000.0,
        eccentricity_upper_bound=0.2,
        strict_e_range_max=5.0e-4,
        quasi_e_range_max=5.0e-3,
        strict_abs_de_dt_max=5.0e-10,
        quasi_abs_de_dt_max=5.0e-9,
        strict_h_peri_range_max_m=800.0,
        quasi_h_peri_range_max_m=8_000.0,
        strict_abs_dh_peri_dt_max_m_per_s=2.0e-3,
        quasi_abs_dh_peri_dt_max_m_per_s=2.0e-2,
        strict_inclination_range_max_rad=1.0e-4,
        quasi_inclination_range_max_rad=5.0e-3,
        strict_abs_dinclination_dt_max_rad_per_s=1.0e-10,
        quasi_abs_dinclination_dt_max_rad_per_s=1.0e-9,
        strict_omega_span_max_rad=5.0e-3,
        quasi_omega_span_max_rad=5.0e-2,
        strict_abs_domega_dt_max_rad_per_s=1.0e-8,
        quasi_abs_domega_dt_max_rad_per_s=1.0e-7,
        strict_hk_loop_drift_max=5.0e-4,
        quasi_hk_loop_drift_max=5.0e-3,
    )


def _metrics(
    *,
    e_drift: float = 0.0,
    h_drift_m: float = 0.0,
    omega_drift_rad: float = 0.0,
    impact_time_s: float | None = None,
    domain_exit_time_s: float | None = None,
    h_base_m: float = 100_000.0,
) -> tuple[object, np.ndarray]:
    t = np.linspace(0.0, DURATION_S, 80)
    phase = 2.0 * np.pi * t / DURATION_S
    e = 0.03 + e_drift * (t / DURATION_S) + 5.0e-5 * np.sin(phase)
    h = h_base_m + h_drift_m * (t / DURATION_S) + 100.0 * np.sin(phase)
    inc = 1.0 + 2.0e-5 * np.cos(phase)
    omega = 0.4 + omega_drift_rad * (t / DURATION_S) + 5.0e-4 * np.sin(phase)
    return (
        compute_frozen_metrics(
            t,
            eccentricity=e,
            h_peri_m=h,
            inclination_rad=inc,
            omega_rad=omega,
            impact_time_s=impact_time_s,
            domain_exit_time_s=domain_exit_time_s,
        ),
        t,
    )


def test_metrics_report_envelopes_slopes_and_hk_loop():
    t = np.linspace(0.0, DURATION_S, 20)
    e = 0.02 + 2.0e-4 * (t / DURATION_S)
    h = 90_000.0 + 500.0 * (t / DURATION_S)
    inc = np.full_like(t, 1.1)
    omega = np.full_like(t, 0.25)

    metrics = compute_frozen_metrics(
        t,
        eccentricity=e,
        h_peri_m=h,
        inclination_rad=inc,
        omega_rad=omega,
    )

    assert metrics.e_min == e[0]
    assert metrics.e_max == e[-1]
    assert np.isclose(metrics.e_range, 2.0e-4)
    assert np.isclose(metrics.de_dt_per_s, 2.0e-4 / DURATION_S)
    assert np.isclose(metrics.h_peri_range_m, 500.0)
    assert metrics.omega_behavior == "libration"
    assert metrics.hk_loop_drift < 3.0e-4


def test_strict_like_candidate_is_not_validated_without_classical_sh():
    metrics, _ = _metrics()
    cfg = _config()

    result = classify_candidate(metrics, cfg)

    assert result.status == CANDIDATE_FROZEN_ORBIT
    assert result.display_label == "candidate frozen orbit"
    assert result.validated is False
    assert "classical SH" in result.reasons[0]
    assert "frozen orbit family discovered" not in str(result.to_dict()).lower()


def test_strict_label_requires_classical_sh_long_horizon_validation():
    metrics, _ = _metrics()
    cfg = _config()

    result = classify_candidate(
        metrics,
        cfg,
        validation_backend="classical_SH500",
        long_horizon_validation_passed=True,
    )

    assert result.status == STRICT_FROZEN
    assert result.validated is True
    assert np.isfinite(result.score)


def test_quasi_frozen_candidate_and_validated_label_are_distinct():
    metrics, _ = _metrics(e_drift=0.002, h_drift_m=3_000.0, omega_drift_rad=0.02)
    cfg = _config()

    candidate = classify_candidate(metrics, cfg)
    validated = classify_candidate(
        metrics,
        cfg,
        validation_backend="cpu_sh",
        long_horizon_validation_passed=True,
    )

    assert candidate.status == QUASI_FROZEN_CANDIDATE
    assert candidate.display_label == "quasi-frozen candidate"
    assert candidate.validated is False
    assert validated.status == QUASI_FROZEN
    assert validated.validated is True


def test_clear_drift_is_long_lived_not_frozen():
    metrics, _ = _metrics(e_drift=0.03, h_drift_m=60_000.0, omega_drift_rad=0.5)
    cfg = _config()

    result = classify_candidate(metrics, cfg, validation_backend="classical_SH500")

    assert result.status == LONG_LIVED_NOT_FROZEN
    assert result.validated is False
    assert "eccentricity envelope too wide" in result.reasons


def test_impact_domain_exit_and_safety_floor_are_unstable_invalid():
    cfg = _config()

    impacted, t = _metrics(impact_time_s=0.5 * DURATION_S)
    domain_exit, _ = _metrics(domain_exit_time_s=0.5 * DURATION_S)
    unsafe_perilune, _ = _metrics(h_base_m=10_000.0)

    assert classify_candidate(impacted, cfg).status == UNSTABLE_INVALID
    assert classify_candidate(domain_exit, cfg).status == UNSTABLE_INVALID
    unsafe = classify_candidate(unsafe_perilune, cfg)
    assert unsafe.status == UNSTABLE_INVALID
    assert "perilune below safety threshold" in unsafe.reasons
    assert np.isinf(frozen_score(domain_exit, cfg))
    assert t.size == impacted.sample_count


def test_duration_factory_derives_rate_thresholds_from_mission_span():
    cfg = FrozenClassificationConfig.for_mission_duration(
        10.0 * 86_400.0,
        strict_e_range_max=1.0e-3,
        quasi_e_range_max=1.0e-2,
        strict_h_peri_range_max_m=1_000.0,
        quasi_h_peri_range_max_m=10_000.0,
    )

    assert np.isclose(cfg.strict_abs_de_dt_max, 1.0e-3 / cfg.mission_duration_s)
    assert np.isclose(
        cfg.quasi_abs_dh_peri_dt_max_m_per_s,
        10_000.0 / cfg.mission_duration_s,
    )
