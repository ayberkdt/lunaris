"""pyqtgraph pen builders that pair the series color and dash cycles (W4).

Series identity must never rest on hue alone: the tokens define a color cycle
and a dash cycle whose lengths differ, and this adapter consumes them
*together*, so the same color always returns with a different line style.
Physical roles carry their identity in the line style itself (truth=solid,
surrogate=dashed, comparison=dotted) and therefore survive grayscale and
color-vision deficiency.

Import safety: pyqtgraph is imported lazily inside the pen builders so this
module can be imported (e.g. by tests or headless tools) without an OpenGL/
pyqtgraph installation.
"""

from __future__ import annotations

import math
from typing import Any

from PySide6 import QtCore

from lunaris.ui_foundation.tokens import DESIGN_TOKENS

__all__ = ["DASH_STYLES", "series_style", "series_pen", "role_pen"]

#: Qt-neutral dash names (tokens) -> Qt pen styles.
DASH_STYLES: dict[str, QtCore.Qt.PenStyle] = {
    "solid": QtCore.Qt.SolidLine,
    "dash": QtCore.Qt.DashLine,
    "dot": QtCore.Qt.DotLine,
    "dashdot": QtCore.Qt.DashDotLine,
}

_VIZ = DESIGN_TOKENS.visualization


def series_style(index: int) -> tuple[str, str]:
    """Return ``(color, dash_name)`` for the *index*-th series of a plot.

    The (color, dash) combination repeats with period ``lcm(len(colors),
    len(dashes))`` (12 for the 6x4 default cycles), so two series only share
    color+dash after 12 siblings.
    """

    colors = _VIZ.series_cycle
    dashes = _VIZ.series_dash_cycle
    i = int(index) % math.lcm(len(colors), len(dashes))
    return colors[i % len(colors)], dashes[i % len(dashes)]


def series_pen(index: int, *, width: float = 2.2, **pen_kwargs: Any):
    """Build a pyqtgraph pen for the *index*-th series (color+dash together)."""

    import pyqtgraph as pg

    color, dash = series_style(index)
    return pg.mkPen(color=color, width=width, style=DASH_STYLES[dash], **pen_kwargs)


def role_pen(role: str, color: str, *, width: float = 2.2, **pen_kwargs: Any):
    """Build a pen whose line style encodes the physical series *role*.

    ``truth`` -> solid, ``surrogate`` -> dashed, ``comparison`` -> dotted
    (from the visualization tokens). The caller picks the color; the role
    decides the dash, so truth-vs-model stays readable without color.
    """

    import pyqtgraph as pg

    dash_by_role = {
        "truth": _VIZ.truth_dash,
        "surrogate": _VIZ.surrogate_dash,
        "comparison": _VIZ.comparison_dash,
    }
    try:
        dash = dash_by_role[str(role)]
    except KeyError:
        raise ValueError(
            f"Unknown series role {role!r}; expected one of {sorted(dash_by_role)}"
        ) from None
    return pg.mkPen(color=color, width=width, style=DASH_STYLES[dash], **pen_kwargs)
