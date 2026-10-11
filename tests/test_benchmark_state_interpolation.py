"""Benchmark trajectory interpolation: cubic Hermite against an analytic Kepler orbit."""

from __future__ import annotations

import numpy as np
import pytest

from lunaris.common.constants import MU_MOON, R_MOON
from lunaris.surrogate.st_lrps.evaluation.phase_diagnostics import hermite_resample_state

pytest.importorskip("matplotlib")
from lunaris.surrogate.st_lrps.evaluation._gravity_benchmark.types import (  # noqa: E402
    interpolate_state_to_times,
)


def _kepler_state(t: np.ndarray, *, a: float, e: float, inc: float = 0.6) -> np.ndarray:
    """Exact two-body state (inertial, m and m/s) at times t, periapsis at t=0."""
    n = np.sqrt(MU_MOON / a**3)
    mean_anom = n * np.asarray(t, dtype=np.float64)
    ecc_anom = mean_anom.copy()
    for _ in range(50):
        ecc_anom -= (ecc_anom - e * np.sin(ecc_anom) - mean_anom) / (1.0 - e * np.cos(ecc_anom))
    cos_e, sin_e = np.cos(ecc_anom), np.sin(ecc_anom)
    b = a * np.sqrt(1.0 - e * e)
    x, y = a * (cos_e - e), b * sin_e
    e_dot = n / (1.0 - e * cos_e)
    vx, vy = -a * sin_e * e_dot, b * cos_e * e_dot
    ci, si = np.cos(inc), np.sin(inc)
    return np.column_stack([x, y * ci, y * si, vx, vy * ci, vy * si])


A_M = R_MOON + 100e3 / (1 - 0.05) * 1.0  # ~100 km periapsis altitude scale
ECC = 0.05


def _interior_errors(h: float) -> tuple[float, float, float, float]:
    period = 2 * np.pi * np.sqrt(A_M**3 / MU_MOON)
    t_src = np.arange(0.0, period + h, h)
    y_src = _kepler_state(t_src, a=A_M, e=ECC)
    # A generic interior abscissa (s = 0.3); at s = 0.5 the h^3 velocity term
    # cancels by symmetry and would hide the true order.
    t_q = t_src[:-1] + 0.3 * h
    y_true = _kepler_state(t_q, a=A_M, e=ECC)
    herm = interpolate_state_to_times(t_src, y_src, t_q)
    lin = np.column_stack([np.interp(t_q, t_src, y_src[:, k]) for k in range(6)])
    pos = lambda y: float(np.max(np.linalg.norm(y[:, :3] - y_true[:, :3], axis=1)))  # noqa: E731
    vel = lambda y: float(np.max(np.linalg.norm(y[:, 3:] - y_true[:, 3:], axis=1)))  # noqa: E731
    return pos(herm), vel(herm), pos(lin), vel(lin)


def test_hermite_beats_linear_and_converges_at_the_expected_order() -> None:
    p60, v60, lp60, lv60 = _interior_errors(60.0)
    p30, v30, _, _ = _interior_errors(30.0)
    # 60 s output step of a ~100 km lunar orbit: linear interpolation is
    # hundreds of metres off inside a segment, Hermite well below a metre.
    assert lp60 > 100.0
    assert p60 < 1e-2 * lp60
    assert v60 < 1e-2 * lv60
    # Position error O(h^4) -> x16 per halving; velocity (derivative) O(h^3) -> x8.
    assert 12.0 < p60 / p30 < 20.0
    assert 6.0 < v60 / v30 < 10.0


def test_same_grid_is_returned_unchanged_and_knots_are_exact() -> None:
    t = np.arange(0.0, 3600.0, 60.0)
    y = _kepler_state(t, a=A_M, e=ECC)
    assert interpolate_state_to_times(t, y, t.copy()) is y
    # Subset of the grid (different length -> no shortcut): reproduced bit for bit.
    sub = t[::3]
    out = interpolate_state_to_times(t, y, sub)
    assert np.array_equal(out, y[::3])
    r_q, v_q, valid = hermite_resample_state(t, y[:, :3], y[:, 3:], t)
    assert valid.all()
    assert np.array_equal(r_q, y[:, :3])
    assert np.array_equal(v_q, y[:, 3:])


def test_extrapolation_is_refused() -> None:
    t = np.arange(0.0, 600.0, 60.0)  # ends at 540 s
    y = _kepler_state(t, a=A_M, e=ECC)
    with pytest.raises(ValueError, match="outside the source time range"):
        interpolate_state_to_times(t, y, np.linspace(0.0, 600.0, 7))
    # Within the grid-matching tolerance an endpoint query is accepted.
    out = interpolate_state_to_times(t, y, np.array([0.0, 540.0 + 5e-4]))
    assert np.allclose(out[0], y[0]) and np.allclose(out[1], y[-1], rtol=0, atol=1.0)
