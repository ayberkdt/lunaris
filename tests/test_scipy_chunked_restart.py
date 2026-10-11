"""Chunked ``solve_ivp`` must restart each chunk from the chunk end.

Regression for the chunked SciPy path restarting from the last *output* sample
when a chunk end is not on the output grid: the span between that sample and
the chunk end was integrated twice and any event inside it was recorded twice.
A chunked run must reproduce the unchunked run on the output grid (to
integrator tolerance) and record every event exactly once.
"""

from __future__ import annotations

import logging
import math
from types import SimpleNamespace

import numpy as np
import pytest

from lunaris.common.constants import MU_MOON, R_MOON
from lunaris.core.propagation.scipy_runner import run_scipy_propagation

MU = float(MU_MOON)


def _rhs(_t: float, y: np.ndarray) -> np.ndarray:
    r = y[:3]
    dy = np.empty_like(y)
    dy[:3] = y[3:6]
    dy[3:6] = -MU * r / float(np.linalg.norm(r)) ** 3
    return dy


def _periapsis_event(_t: float, y: np.ndarray) -> float:
    return float(np.dot(y[:3], y[3:6]))


_periapsis_event.terminal = False  # type: ignore[attr-defined]
_periapsis_event.direction = 1.0  # type: ignore[attr-defined]


def _orbit() -> tuple[np.ndarray, float, float]:
    """Eccentric orbit started at true anomaly 90 deg (r.v > 0, not on an apsis)."""
    rp = float(R_MOON) + 300e3
    ra = float(R_MOON) + 3000e3
    a = 0.5 * (rp + ra)
    e = (ra - rp) / (ra + rp)
    p = a * (1.0 - e * e)
    nu = 0.5 * math.pi
    r = p / (1.0 + e * math.cos(nu))
    h = math.sqrt(MU * p)
    vr = MU / h * e * math.sin(nu)
    vt = h / r
    y0 = np.array([r * math.cos(nu), r * math.sin(nu), 0.0,
                   vr * math.cos(nu) - vt * math.sin(nu),
                   vr * math.sin(nu) + vt * math.cos(nu), 0.0])
    period = 2.0 * math.pi * math.sqrt(a ** 3 / MU)
    # Time from periapsis at nu = 90 deg (Kepler's equation).
    E = 2.0 * math.atan(math.sqrt((1.0 - e) / (1.0 + e)) * math.tan(0.5 * nu))
    t_since_peri = (E - e * math.sin(E)) / (2.0 * math.pi / period)
    return y0, period, t_since_peri


def _run(y0, t_eval, chunk_s, stop_file=None):
    cfg = SimpleNamespace(rtol=1e-12, atol=1e-6, checkpoint_every_chunk=False, stop_event_in_scipy=False)
    return run_scipy_propagation(
        rhs=_rhs,
        t_eval=t_eval,
        y0=y0,
        t0=float(t_eval[0]),
        tf=float(t_eval[-1]),
        method="DOP853",
        max_step_s=60.0,
        chunk_s=chunk_s,
        cfg=cfg,  # type: ignore[arg-type]
        events=[_periapsis_event],
        stop_file=stop_file,
        checkpoint_path=None,
        checkpoint_metadata={},
        output_dt_s=float(t_eval[1] - t_eval[0]),
        verbose=False,
        logger=logging.getLogger(__name__),
    )


def test_chunked_run_matches_unchunked_and_records_each_event_once():
    y0, period, t_since_peri = _orbit()
    duration = 3.0 * period
    # Output step and chunk length chosen so chunk ends fall off the grid and
    # periapses fall between the last grid sample of a chunk and its end.
    t_eval = np.linspace(0.0, duration, 22)
    chunk_s = 0.55 * period

    ref = _run(y0, t_eval, None)
    chunked = _run(y0, t_eval, chunk_s)

    np.testing.assert_array_equal(chunked.t, ref.t)
    # Position to 1 m and velocity to 1 mm/s: rtol=1e-12 over three periods.
    np.testing.assert_allclose(chunked.y[:, :3], ref.y[:, :3], rtol=0.0, atol=1.0)
    np.testing.assert_allclose(chunked.y[:, 3:6], ref.y[:, 3:6], rtol=0.0, atol=1e-3)

    expected = np.array([period * k - t_since_peri for k in (1, 2, 3)])
    expected = expected[expected <= duration]
    np.testing.assert_allclose(ref.t_events[0], expected, rtol=0.0, atol=1e-3)
    np.testing.assert_allclose(chunked.t_events[0], expected, rtol=0.0, atol=1e-3)
    assert chunked.y_events[0].shape == (expected.size, 6)


def test_chunk_shorter_than_output_step_keeps_only_grid_samples():
    y0, period, _ = _orbit()
    t_eval = np.linspace(0.0, period, 6)
    ref = _run(y0, t_eval, None)
    chunked = _run(y0, t_eval, 0.07 * period)
    np.testing.assert_array_equal(chunked.t, ref.t)
    np.testing.assert_allclose(chunked.y[:, :3], ref.y[:, :3], rtol=0.0, atol=1.0)
    assert chunked.t_events[0].size == ref.t_events[0].size == 1


@pytest.mark.parametrize("chunk_s", [None, 1000.0])
def test_stop_file_present_before_start_stops_at_t0(tmp_path, chunk_s):
    stop = tmp_path / "STOP"
    stop.write_text("stop", encoding="utf-8")
    y0, period, _ = _orbit()
    t_eval = np.linspace(0.0, period, 11)
    res = _run(y0, t_eval, chunk_s, stop_file=str(stop))
    assert res.stopped_early is True
    assert res.stop_reason == "stop file"
    np.testing.assert_array_equal(res.t, [0.0])
    np.testing.assert_array_equal(res.y[0], y0)
    assert res.t_stop_s == pytest.approx(0.0)
