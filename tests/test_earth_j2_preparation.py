"""Production Earth-J2 preparation invariants."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from lunaris.core.dynamics.preparation import prepare_earth_j2


def _params(axis: tuple[float, float, float]) -> SimpleNamespace:
    return SimpleNamespace(
        j2_coeff=1.082_626_68e-3,
        r_eq_m=6_378_137.0,
        spin_axis_i=axis,
    )


def test_prepare_earth_j2_normalizes_equivalent_axis_scales() -> None:
    req = SimpleNamespace(use_earth_j2=True)

    unit = prepare_earth_j2(req, _params((0.0, 0.0, 1.0)))
    scaled = prepare_earth_j2(req, _params((0.0, 0.0, 2.0)))

    assert (scaled.ax, scaled.ay, scaled.az) == pytest.approx(
        (unit.ax, unit.ay, unit.az), rel=0.0, abs=0.0
    )
    assert np.linalg.norm((scaled.ax, scaled.ay, scaled.az)) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "axis",
    [(0.0, 0.0, 0.0), (float("nan"), 0.0, 1.0), (float("inf"), 0.0, 1.0)],
)
def test_prepare_earth_j2_rejects_invalid_axes(axis: tuple[float, float, float]) -> None:
    req = SimpleNamespace(use_earth_j2=True)

    with pytest.raises(ValueError, match="spin_axis_i"):
        prepare_earth_j2(req, _params(axis))
