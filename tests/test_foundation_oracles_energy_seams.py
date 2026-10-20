"""Energy and state across propagation seams, judged against runs without the seam.

Lunar point mass, inertial axes, SI units, solver time in seconds. For point-mass
dynamics the specific energy and angular momentum are conserved, so any error a
seam adds shows up against a control run of the same method and step. The rule
these tests encode: a seam's extra error must be truncation-sized and shrink with
the step or tolerance at the method's order; a seam defect (a step taken twice, a
state copied before an update, a time offset) does not shrink and grows with the
number of seams.

The seams covered: many zero-delta-v impulses (manoeuvre segmentation), many
SciPy chunks, stop-file checkpoint and resume, and the long-run energy of the
fixed-step symplectic methods.

Adapted from the Sidera fork's foundation-oracle seam tests (point-mass
dynamics object, inertial impulse frame); every "measured" figure below was
measured on Lunaris.
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path

import numpy as np
import pytest

from lunaris.common.constants import MU_MOON, R_MOON
from lunaris.common.type_defs import EventConfig, PropagatorConfig, TimeConfig
from lunaris.core.propagation.checkpoint import load_propagation_checkpoint
from lunaris.core.propagation.plans import ImpulsiveManeuver, ManeuverPlan
from lunaris.core.propagation.propagator import propagate

MU = float(MU_MOON)
R = float(R_MOON)


class PointMassDynamics:
    """Lunar point-mass dynamics in inertial axes (no gravity field, ephemeris or flags)."""

    grav = None
    ephem = None
    flags = None

    def build_rhs(self):
        def rhs(_t, y):
            st = np.asarray(y, dtype=np.float64)
            out = np.empty_like(st)
            out[:3] = st[3:6]
            r = st[:3]
            rn = np.linalg.norm(r)
            out[3:6] = -MU * r / rn**3
            if st.size > 6:
                out[6:] = 0.0
            return out

        return rhs


def _kepler_state(a: float, e: float, mean_anomaly: float) -> np.ndarray:
    """Inertial state on an inclined ellipse (own Kepler solver; no Lunaris math)."""
    E = mean_anomaly if e < 0.8 else math.pi
    for _ in range(100):
        d = (E - e * math.sin(E) - mean_anomaly) / (1.0 - e * math.cos(E))
        E -= d
        if abs(d) < 1e-15:
            break
    n = math.sqrt(MU / a**3)
    b = a * math.sqrt(1.0 - e * e)
    xp, yp = a * (math.cos(E) - e), b * math.sin(E)
    vxp = -a * n * math.sin(E) / (1.0 - e * math.cos(E))
    vyp = b * n * math.cos(E) / (1.0 - e * math.cos(E))
    inc, raan, argp = 0.5, 0.3, 0.2
    co, so, ci, si = math.cos(raan), math.sin(raan), math.cos(inc), math.sin(inc)
    cw, sw = math.cos(argp), math.sin(argp)
    p = np.array([co * cw - so * sw * ci, so * cw + co * sw * ci, sw * si])
    q = np.array([-co * sw - so * cw * ci, -so * sw + co * cw * ci, cw * si])
    return np.concatenate([xp * p + yp * q, vxp * p + vyp * q])


def _period(a: float) -> float:
    return 2.0 * math.pi * math.sqrt(a**3 / MU)


def _energy(y: np.ndarray) -> np.ndarray:
    y = np.atleast_2d(y)
    return 0.5 * np.sum(y[:, 3:6] ** 2, axis=1) - MU / np.linalg.norm(y[:, :3], axis=1)


def _cfg(method: str, max_step: float, **kw) -> PropagatorConfig:
    kw.setdefault("events", EventConfig(detect_impact=False, enable_peri_apo_events=False))
    return PropagatorConfig(method=method, verbose=False, compute_2body_baseline=False,
                            use_nyquist_max_step=False, user_max_step_s=max_step, **kw)


def _run(y0, cfg, duration, dt_out, t0=0.0, **kw):
    tc = TimeConfig(duration_s=duration, output_dt_s=dt_out, t0_s=t0,
                    samples_per_period=2, max_points_cap=10**7)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return propagate(PointMassDynamics(), y0, cfg, time_cfg=tc, **kw)


# -----------------------------------------------------------------------------
# Long-run energy of the symplectic fixed-step methods
# -----------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("method", "order"),
    # YOSHIDA4 takes ~5 s (three force calls per step); PEFRL keeps a 4th-order case in CI.
    [("PEFRL", 4), pytest.param("YOSHIDA4", 4, marks=pytest.mark.slow), ("VV", 2)],
)
def test_symplectic_energy_error_stays_bounded_and_falls_at_the_method_order(method, order):
    # e = 0.1 (a circular orbit cannot show a secular drift: its energy error is flat).
    a = R + 500e3
    T = _period(a)
    y0 = _kepler_state(a, 0.1, 0.1)
    e0 = _energy(y0)[0]
    n_periods, dt_out = 200, T / 20
    window = 10 * 20  # samples in 10 periods
    worst = {}
    for nsub in (5, 10):
        h = dt_out / nsub
        res = _run(y0, _cfg(method, h * 1.0000001, max_internal_steps=10**8), n_periods * T, dt_out)
        de = np.abs((_energy(res.y) - e0) / e0)
        first, last = de[:window].max(), de[-window:].max()
        # Bounded: a symplectic method's energy error oscillates; over 200 periods it must not
        # grow beyond 1 % of its first-window amplitude (measured last/first 0.9990-1.0003,
        # 2026-10-20).
        assert last <= 1.01 * first, f"{method} h={h:.1f}s: first {first:.3e} last {last:.3e}"
        worst[nsub] = first
    ratio = worst[5] / worst[10]
    # Halving h divides the amplitude by 2**order (measured 4.03 VV, 16.0 PEFRL, 15.9 YOSHIDA4);
    # +-15 % in the exponent keeps the test off the asymptotic edge.
    assert 2.0 ** (0.85 * order) < ratio < 2.0 ** (1.15 * order), f"{method}: ratio {ratio:.2f}"


# -----------------------------------------------------------------------------
# Manoeuvre segmentation: 100 zero-delta-v impulses
# -----------------------------------------------------------------------------

def _zero_dv_plan(n: int, duration: float) -> ManeuverPlan:
    # Epochs off the output grid and off apsides (0.37 of a spacing in).
    epochs = (np.arange(n) + 0.37) * duration / n
    return ManeuverPlan(tuple(ImpulsiveManeuver(float(t), (0.0, 0.0, 0.0), frame="inertial") for t in epochs))


@pytest.mark.parametrize(("method", "steps"), [("RK4", (60.0, 30.0, 15.0)), ("PEFRL", (60.0, 30.0, 15.0))])
def test_zero_dv_impulses_add_only_truncation_sized_error(method, steps):
    a = R + 2000e3
    T = _period(a)
    y0 = _kepler_state(a, 0.25, 0.1)
    duration = 3.0 * T
    plan = _zero_dv_plan(100, duration)
    diffs, truncs = [], []
    for h in steps:
        plain = _run(y0, _cfg(method, h), duration, T / 40)
        seamed = _run(y0, _cfg(method, h), duration, T / 40, maneuver_plan=plan)
        truth = _kepler_state(a, 0.25, 0.1 + 2.0 * math.pi * duration / T)
        diffs.append(float(np.linalg.norm(seamed.y[-1, :3] - plain.y[-1, :3])))
        truncs.append(float(np.linalg.norm(plain.y[-1, :3] - truth[:3])))
    for d, tr in zip(diffs, truncs, strict=False):
        # 100 restarts each lose at most one step's local error; the sum stays below the
        # global truncation error of the plain run (measured RK4: 0.15 vs 0.83 m at 60 s).
        assert d < tr, f"{method}: seam {diffs} truncation {truncs}"
    # The seam error falls with the step (a defect would stay put): two halvings of a
    # 4th-order method must shrink it at least 10x (measured RK4: 0.15 -> 9.0e-5 m).
    assert diffs[-1] < diffs[0] / 10.0, f"{method}: {diffs}"


def test_zero_dv_impulse_plan_leaves_dop853_on_its_tolerance():
    a = R + 2000e3
    T = _period(a)
    y0 = _kepler_state(a, 0.25, 0.1)
    duration = 3.0 * T
    plan = _zero_dv_plan(100, duration)
    diffs = []
    for rtol in (1e-10, 1e-12):
        cfg = _cfg("DOP853", 60.0, rtol=rtol, atol=1e-14)
        plain = _run(y0, cfg, duration, T / 40)
        seamed = _run(y0, cfg, duration, T / 40, maneuver_plan=plan)
        diffs.append(float(np.linalg.norm(seamed.y[-1, :3] - plain.y[-1, :3])))
    # With the step capped at 60 s DOP853 sits on its rounding floor, so the 100 restarts
    # may only add rounding (measured 9.8e-8 and 7.0e-7 m on a 3.7e6 m orbit, 2026-10-20);
    # 1e-5 m (3e-12 of the radius) leaves room for platform rounding and none for a leak.
    assert max(diffs) < 1e-5, diffs


# -----------------------------------------------------------------------------
# Chunked SciPy runs: 1000 chunks
# -----------------------------------------------------------------------------

def test_a_thousand_chunks_do_not_leak_energy():
    a = R + 2000e3
    T = _period(a)
    y0 = _kepler_state(a, 0.5, 0.1)
    e0 = _energy(y0)[0]
    duration = 10.0 * T
    for rtol in (1e-11, 1e-12):
        base = dict(rtol=rtol, atol=1e-14)
        plain = _run(y0, _cfg("DOP853", 1e6, **base), duration, T / 20)
        chunked = _run(y0, _cfg("DOP853", 1e6, chunk_s=duration / 1000.0, **base), duration, T / 20)
        assert np.array_equal(plain.t, chunked.t)
        de_plain = float(np.max(np.abs(_energy(plain.y) - e0)) / abs(e0))
        de_chunked = float(np.max(np.abs(_energy(chunked.y) - e0)) / abs(e0))
        # Each chunk restarts the step-size history (small first steps), which if anything
        # lowers the energy error (measured 4.5e-11 vs 1.6e-10 at rtol 1e-11, 7.9e-12 vs
        # 1.4e-11 at 1e-12); a leak of one state ulp per chunk would add ~1e-13 x 1000.
        assert de_chunked < 2.0 * de_plain, f"rtol {rtol}: chunked {de_chunked:.2e} plain {de_plain:.2e}"
    truth = _kepler_state(a, 0.5, 0.1 + 2.0 * math.pi * duration / T)
    # Position stays at the tolerance level (measured 1.7e-3 m after 10 periods at rtol 1e-12).
    assert np.linalg.norm(chunked.y[-1, :3] - truth[:3]) < 0.1


# -----------------------------------------------------------------------------
# Stop-file checkpoint and resume, ten interruptions
# -----------------------------------------------------------------------------

def _stop_trigger(stop: Path, t_trigger: float):
    def g(t, y):
        if t >= t_trigger and not stop.exists():
            stop.write_text("x")
        return 1.0
    g.terminal = False
    g.direction = 0.0
    return g


@pytest.mark.parametrize(("method", "max_step"), [("DOP853", 60.0), ("RK4", 30.0), ("PEFRL", 30.0)])
def test_ten_resumes_reach_the_uninterrupted_state(tmp_path, method, max_step):
    a = R + 2000e3
    T = _period(a)
    y0 = _kepler_state(a, 0.25, 0.1)
    tf, dt_out = 3.0 * T, T / 40
    extra = {"rtol": 1e-12, "atol": 1e-14} if method == "DOP853" else {}
    reference = _run(y0, _cfg(method, max_step, **extra), tf, dt_out)
    truth = _kepler_state(a, 0.25, 0.1 + 2.0 * math.pi * tf / T)
    truncation = float(np.linalg.norm(reference.y[-1, :3] - truth[:3]))

    stop, ckpt = tmp_path / "STOP", tmp_path / "ckpt.npz"
    triggers = list(np.linspace(0.0, tf, 12)[1:-1] + 0.37 * dt_out)
    t_cur, y_cur, legs = 0.0, y0.copy(), 0
    while t_cur < tf - 1e-6 and legs < 30:
        if stop.exists():
            stop.unlink()
        pending = [x for x in triggers if x > t_cur + 1e-9]
        events = [_stop_trigger(stop, pending[0])] if pending else []
        cfg = _cfg(method, max_step, stop_file=str(stop), checkpoint_path=str(ckpt), **extra)
        res = _run(y_cur, cfg, tf - t_cur, dt_out, t0=t_cur, extra_events=events)
        saved = load_propagation_checkpoint(ckpt)
        # The checkpoint is the run's last accepted state, bit for bit.
        assert saved.t_current == res.t[-1]
        assert np.array_equal(saved.y_current, res.y[-1])
        t_cur, y_cur = float(saved.t_current), np.asarray(saved.y_current)
        legs += 1
    assert legs >= 10 and abs(t_cur - tf) < 1e-6
    diff = float(np.linalg.norm(y_cur[:3] - reference.y[-1, :3]))
    # Each resume restarts the integrator (fixed-step: a shortened last step; DOP853: a fresh
    # step-size history), so the resumed run may differ by a fraction of the truncation error,
    # never by more (measured RK4 5.3e-4 m vs 4.3e-2 m truncation, PEFRL 2.9e-4 vs 2.6e-2 m;
    # DOP853 8.3e-7 m, under the 1e-5 m rounding floor).
    assert diff < max(0.1 * truncation, 1e-5), f"{method}: {diff:.3e} m vs truncation {truncation:.3e} m"


# -----------------------------------------------------------------------------
# A manoeuvre run's diagnostics cover the whole run, not its last segment
# -----------------------------------------------------------------------------

def test_a_zero_dv_impulse_leaves_the_run_diagnostics_of_the_plain_run():
    a = R + 2000e3
    T = _period(a)
    y0 = _kepler_state(a, 0.25, 0.1)
    duration = 3.0 * T
    cfg = _cfg("RK45", 60.0, rtol=1e-6, atol=1e-9)
    plain = _run(y0, cfg, duration, T / 50)
    plan = ManeuverPlan((ImpulsiveManeuver(2.98 * T, (0.0, 0.0, 0.0), frame="inertial"),))
    seamed = _run(y0, cfg, duration, T / 50, maneuver_plan=plan)
    assert seamed.diagnostics["n_points"] == float(len(seamed.t))
    drift_plain = plain.diagnostics["kepler_energy_rel_drift"]
    drift_seamed = seamed.diagnostics["kepler_energy_rel_drift"]
    # A run that reported only its last 0.02 T segment (a defect the Sidera fork had) would
    # show a far smaller drift than the plain run. Per-segment worst drift tracks the run's
    # own (measured 1.15e-10 vs 1.21e-10, ratio 0.955, 2026-10-20).
    assert 0.5 < drift_seamed / drift_plain < 2.0, (drift_seamed, drift_plain)
    assert seamed.diagnostics["nfev"] >= 0.9 * plain.diagnostics["nfev"]
