"""Series color+dash cycle contract (W4).

The invariant: pyqtgraph adapters consume the color and dash cycles TOGETHER,
so within one combination period no two series share both color and line
style, and every same-dash color pair stays distinguishable under CVD
simulation (gated by tools/ui/color_audit.py, imported here).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

from lunaris.ui_foundation.tokens import DESIGN_TOKENS

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "ui"))

import color_audit  # noqa: E402  (tools/ui/color_audit.py)


def test_series_cycles_are_declared() -> None:
    viz = DESIGN_TOKENS.visualization
    assert len(viz.series_cycle) >= 4
    assert len(viz.series_dash_cycle) >= 3
    assert len(set(viz.series_cycle)) == len(viz.series_cycle)
    assert len(set(viz.series_dash_cycle)) == len(viz.series_dash_cycle)


def test_role_dashes_are_distinct_and_known() -> None:
    viz = DESIGN_TOKENS.visualization
    roles = (viz.truth_dash, viz.surrogate_dash, viz.comparison_dash)
    assert len(set(roles)) == 3
    for dash in roles:
        assert dash in viz.series_dash_cycle


def test_color_dash_combination_unique_within_period() -> None:
    from lunaris.ui.core.plot_style import series_style

    viz = DESIGN_TOKENS.visualization
    period = math.lcm(len(viz.series_cycle), len(viz.series_dash_cycle))
    combos = [series_style(i) for i in range(period)]
    assert len(set(combos)) == period, "color+dash repeated inside one period"
    # The wrap point repeats the first combination, by construction.
    assert series_style(period) == series_style(0)


def test_color_audit_is_clean() -> None:
    violations = color_audit.audit_series_cycle()
    assert violations == [], "\n".join(violations)


def test_dash_names_map_to_qt_styles() -> None:
    pytest.importorskip("PySide6.QtCore")
    from lunaris.ui.core.plot_style import DASH_STYLES

    viz = DESIGN_TOKENS.visualization
    for dash in viz.series_dash_cycle:
        assert dash in DASH_STYLES
