"""2B regression: the impact-statistics contract distinguishes a genuine 0 %
impact outcome (``evaluated``) from a run that never measured impacts
(``disabled`` when detection was off, ``not_evaluated`` when it could not be
produced). ``MCStatistics.impacts`` is ``None`` in the latter two cases so no
consumer can misread "we did not look" as "0 % chance of impact".
"""

from __future__ import annotations

import numpy as np

from lunaris.analysis.monte_carlo.statistics import (
    IMPACT_DISABLED,
    IMPACT_EVALUATED,
    IMPACT_NOT_EVALUATED,
    compute_mc_statistics,
)
from lunaris.common.montecarlo_defs import MCRunResult


def _result(*, impacts: bool, diagnostics: dict | None = None) -> MCRunResult:
    """Two valid orbiting samples; ``impacts`` toggles whether either hits."""
    Y = np.zeros((2, 2, 6), dtype=np.float64)
    Y[:, 0, 0] = 1.8e6
    Y[:, 1, 0] = 1.9e6
    impact_mask = np.array([0.0, 1.0 if impacts else 0.0])
    t_impact = np.array([np.nan, 1.0 if impacts else np.nan])
    fixed = np.full((2, 3), np.nan)
    if impacts:
        fixed[1] = [0.0, 1.0, 0.0]
    return MCRunResult(
        t=np.array([0.0, 1.0]),
        Y=Y,
        sc_samples=np.ones((2, 4)),
        impact_mask=impact_mask,
        t_impact=t_impact,
        valid_mask=np.array([1.0, 1.0]),
        impact_position_fixed_m=fixed,
        diagnostics=diagnostics or {},
    )


def test_detection_disabled_reports_disabled_not_zero_percent() -> None:
    result = _result(impacts=False, diagnostics={"compute_impact_statistics": False})
    stats = compute_mc_statistics(result, compute_oe=False)
    assert stats.impact_status == IMPACT_DISABLED
    assert stats.impacts is None


def test_detect_impact_flag_alone_disables_when_false() -> None:
    result = _result(impacts=False, diagnostics={"detect_impact": False})
    stats = compute_mc_statistics(result, compute_oe=False)
    assert stats.impact_status == IMPACT_DISABLED
    assert stats.impacts is None


def test_genuine_zero_percent_is_evaluated_with_zero_impacts() -> None:
    # Detection on, no sample hit -> a real 0 %, not "disabled".
    result = _result(impacts=False, diagnostics={"compute_impact_statistics": True})
    stats = compute_mc_statistics(result, compute_oe=False)
    assert stats.impact_status == IMPACT_EVALUATED
    assert stats.impacts is not None
    assert stats.impacts.n_impacts == 0
    assert stats.impacts.p_impact == 0.0


def test_evaluated_run_reports_impacts() -> None:
    result = _result(impacts=True, diagnostics={"compute_impact_statistics": True})
    stats = compute_mc_statistics(result, compute_oe=False)
    assert stats.impact_status == IMPACT_EVALUATED
    assert stats.impacts is not None
    assert stats.impacts.n_impacts == 1


def test_legacy_archive_without_flag_defaults_to_evaluated() -> None:
    result = _result(impacts=True, diagnostics={})
    stats = compute_mc_statistics(result, compute_oe=False)
    assert stats.impact_status == IMPACT_EVALUATED
    assert stats.impacts is not None


def test_enabled_but_impact_analysis_cannot_run_is_not_evaluated(monkeypatch) -> None:
    # Detection on, ensemble succeeds, but the impact analysis itself cannot
    # produce statistics -> "not evaluated", never a fabricated 0 %.
    import lunaris.analysis.monte_carlo.statistics as stats_mod

    def _raise(*_args, **_kwargs):
        raise ValueError("impact statistics are undefined: no valid samples")

    monkeypatch.setattr(stats_mod, "compute_impact_statistics", _raise)
    result = _result(impacts=True, diagnostics={"compute_impact_statistics": True})
    stats = compute_mc_statistics(result, compute_oe=False)
    assert stats.impact_status == IMPACT_NOT_EVALUATED
    assert stats.impacts is None


def test_explicit_compute_impacts_false_overrides_manifest() -> None:
    result = _result(impacts=True, diagnostics={"compute_impact_statistics": True})
    stats = compute_mc_statistics(result, compute_oe=False, compute_impacts=False)
    assert stats.impact_status == IMPACT_DISABLED
    assert stats.impacts is None


def test_string_and_numeric_flag_encodings_are_honored() -> None:
    # HDF5 attrs can round-trip the flag as a numpy scalar or a string.
    for value in (np.bool_(False), 0, "false", "off"):
        result = _result(impacts=False, diagnostics={"compute_impact_statistics": value})
        stats = compute_mc_statistics(result, compute_oe=False)
        assert stats.impact_status == IMPACT_DISABLED, value
