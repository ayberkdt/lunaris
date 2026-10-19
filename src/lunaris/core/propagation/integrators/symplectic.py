"""Symplectic and acceleration-form fixed-step kernels."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

# ---- Velocity-Verlet base step (symmetric, 2nd order) -----------------------

def _vv_step(accel: Callable[[float, np.ndarray], np.ndarray], t: float, y6: np.ndarray, h: float) -> np.ndarray:
    """Advance one velocity-Verlet kick–drift–kick step.

    ``y6=[r,v]`` is SI Cartesian state, ``t``/``h`` are seconds, and ``accel``
    returns [m/s²]. Symplectic/separable guarantees require acceleration to be
    position dependent and conservative; time-, velocity-, or dissipative
    forces can still be evaluated but void those guarantees.
    """
    r = y6[:3]
    v = y6[3:6]
    a0 = accel(t, y6)

    v_half = v + 0.5 * h * a0
    r1 = r + h * v_half

    y_half = np.empty(6, dtype=np.float64)
    y_half[0:3] = r1
    y_half[3:6] = v_half
    a1 = accel(t + h, y_half)

    v1 = v_half + 0.5 * h * a1

    y1 = np.empty(6, dtype=np.float64)
    y1[0:3] = r1
    y1[3:6] = v1
    return y1


# ---- Yoshida symmetric triple-jump composition ------------------------------

def _composition_weights(num_levels: int) -> tuple[float, ...]:
    """Flat velocity-Verlet sub-step fractions for a Yoshida composition.

    ``num_levels == 0`` returns ``(1.0,)`` (plain 2nd-order Verlet); each
    additional level raises the order by two (4th, 6th, 8th, ...). The returned
    fractions always sum to 1.0.

    Level ``k`` composes the whole order-``2k`` method three times,
    ``S(x1 h) S(x0 h) S(x1 h)`` with ``p = 2k + 1``, so the newest level is the
    outermost. Nesting it inside (each Verlet sub-step split by the new
    level's weights) keeps the same multiset of weights, and so the same
    order on a linear problem, but stays 4th order on a nonlinear one.
    """
    weights: list[float] = [1.0]
    for k in range(1, int(num_levels) + 1):
        p = 2 * k + 1
        root = 2.0 ** (1.0 / p)
        x1 = 1.0 / (2.0 - root)
        x0 = -root / (2.0 - root)
        weights = [x1 * w for w in weights] + [x0 * w for w in weights] + [x1 * w for w in weights]
    return tuple(weights)


# Order 4 / 6 / 8 weight tables. Order-4 reproduces the classic Yoshida4 step.
_Y4_WEIGHTS = _composition_weights(1)
_Y6_WEIGHTS = _composition_weights(2)
_Y8_WEIGHTS = _composition_weights(3)


def _composition_time_overhang(weights: tuple[float, ...]) -> float:
    """Largest excursion of a composition's force-evaluation times outside [t, t+h], in steps.

    Velocity-Verlet evaluates the force at both ends of each sub-step, so the
    evaluation times are the partial sums of the weights. The negative middle
    weight of a triple jump sends them before ``t`` and past ``t + h``.
    """
    ends = np.concatenate(([0.0], np.cumsum(np.asarray(weights, dtype=np.float64))))
    return float(max(0.0, -float(ends.min()), float(ends.max()) - 1.0))


_STAGE_TIME_OVERHANG: dict[str, float] = {
    "Y4": _composition_time_overhang(_Y4_WEIGHTS),
    "Y6": _composition_time_overhang(_Y6_WEIGHTS),
    "Y8": _composition_time_overhang(_Y8_WEIGHTS),
}


def stage_time_overhang(method: str) -> float:
    """Force-evaluation overhang beyond [t, t+h] of a canonical fixed-step method, in steps.

    0.351 (Y4), 0.649 (Y6), 0.916 (Y8); 0 for every method whose stage times
    stay inside the step (VV, PEFRL, RKN4, RK4, RK8). The fixed-step driver
    takes a run's first step with a one-shot DOP853 step, so a run never
    samples the force (or the ephemeris) before ``t0``; past ``tf`` it samples
    up to ``overhang * h``.
    """
    return _STAGE_TIME_OVERHANG.get(str(method).strip().upper(), 0.0)


def _composed_step(
    accel: Callable[[float, np.ndarray], np.ndarray],
    t: float,
    y6: np.ndarray,
    h: float,
    weights: tuple[float, ...],
) -> np.ndarray:
    """Apply a sequence of velocity-Verlet sub-steps (Yoshida composition)."""
    y = y6
    tau = t
    for w in weights:
        hw = w * h
        y = _vv_step(accel, tau, y, hw)
        tau += hw
    return y


def _y4_step(accel: Callable[[float, np.ndarray], np.ndarray], t: float, y6: np.ndarray, h: float) -> np.ndarray:
    """Apply the fourth-order Yoshida triple-jump composition of Verlet."""
    return _composed_step(accel, t, y6, h, _Y4_WEIGHTS)


def _y6_step(accel: Callable[[float, np.ndarray], np.ndarray], t: float, y6: np.ndarray, h: float) -> np.ndarray:
    """Apply the recursive sixth-order triple-jump composition of Verlet.

    The accepted ``YOSHIDA6`` token is retained for compatibility; these are
    recursively composed weights, not Yoshida's optimized minimal-stage
    sixth-order coefficient set.
    """
    return _composed_step(accel, t, y6, h, _Y6_WEIGHTS)


def _y8_step(accel: Callable[[float, np.ndarray], np.ndarray], t: float, y6: np.ndarray, h: float) -> np.ndarray:
    """Apply the recursive eighth-order triple-jump composition of Verlet.

    The accepted ``YOSHIDA8`` token is retained for compatibility; these are
    recursively composed weights, not Yoshida's optimized minimal-stage
    eighth-order coefficient set.
    """
    return _composed_step(accel, t, y6, h, _Y8_WEIGHTS)


# ---- PEFRL: Position-Extended Forest-Ruth-Like (4th order, optimized) -------
# Omelyan, Mryglod & Folk (2002), Computer Physics Communications 146, 188.
# Four force evaluations per step with a markedly smaller error constant than the
# Yoshida4 triple jump, which makes it a strong default 4th-order symplectic
# integrator for smooth gravitational fields.

_PEFRL_XI = 0.1786178958448091
_PEFRL_LAMBDA = -0.2123418310626054
_PEFRL_CHI = -0.06626458266981849


def _pack6(r: np.ndarray, v: np.ndarray) -> np.ndarray:
    y = np.empty(6, dtype=np.float64)
    y[0:3] = r
    y[3:6] = v
    return y


def _pefrl_step(accel: Callable[[float, np.ndarray], np.ndarray], t: float, y6: np.ndarray, h: float) -> np.ndarray:
    """Advance one fourth-order PEFRL split step for separable dynamics.

    The Omelyan–Mryglod–Folk coefficients operate on SI ``[r,v]`` state with
    second-based time/step and acceleration [m/s²]. Four force evaluations are
    used. The symplectic guarantee applies only to smooth conservative,
    position-dependent acceleration; non-conservative or velocity-dependent
    terms void it.
    """
    xi = _PEFRL_XI
    lam = _PEFRL_LAMBDA
    chi = _PEFRL_CHI

    r = np.array(y6[:3], dtype=np.float64, copy=True)
    v = np.array(y6[3:6], dtype=np.float64, copy=True)

    # Drift/kick fractions (positions telescope to 1; velocity kicks sum to 1).
    drift_mid = 1.0 - 2.0 * (chi + xi)
    kick_end = (1.0 - 2.0 * lam) * 0.5

    tau = t
    r = r + xi * h * v
    tau += xi * h
    v = v + kick_end * h * accel(tau, _pack6(r, v))

    r = r + chi * h * v
    tau += chi * h
    v = v + lam * h * accel(tau, _pack6(r, v))

    r = r + drift_mid * h * v
    tau += drift_mid * h
    v = v + lam * h * accel(tau, _pack6(r, v))

    r = r + chi * h * v
    tau += chi * h
    v = v + kick_end * h * accel(tau, _pack6(r, v))

    r = r + xi * h * v
    return _pack6(r, v)


__all__ = [
    "_vv_step",
    "_composition_weights",
    "_Y4_WEIGHTS",
    "_Y6_WEIGHTS",
    "_Y8_WEIGHTS",
    "_composed_step",
    "_y4_step",
    "_y6_step",
    "_y8_step",
    "_pack6",
    "_PEFRL_XI",
    "_PEFRL_LAMBDA",
    "_PEFRL_CHI",
    "_pefrl_step",
]
