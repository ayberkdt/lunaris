"""The analysis ``time_system`` label must follow the configured start epoch.

``build_orbit_analysis`` hard-coded ``time_system = "UTC"`` although
``TimeConfig.start_date`` may be a SPICE ``"... TDB"`` / ``"... TT"`` string
(passed through to ``str2et`` unchanged), so a TDB epoch was reported as UTC.
"""

from __future__ import annotations

import pytest
from tests.test_orbit_analysis_contract import _circular_result, _config

from lunaris.analysis.orbit_analysis import build_orbit_analysis


@pytest.mark.parametrize(
    ("start", "label"),
    [
        ("2026-01-01T00:00:00Z", "UTC"),
        ("2026-01-01T00:01:09.184 TDB", "TDB"),
        ("2026 JAN 01 00:01:09.184 TT", "TT"),
        ("2026-01-01T00:01:09.184 TDT", "TT"),
    ],
)
def test_time_system_label_follows_start_epoch(start: str, label: str) -> None:
    cfg = _config()
    cfg.time.start_date = start
    result = build_orbit_analysis(_circular_result(), config=cfg)
    assert result.time_system == label
    start_metric = next(m for m in result.metrics if m.metric_id == "mission.start_epoch")
    assert start_metric.time_system == label
