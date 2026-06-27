"""Runge-Kutta and Runge-Kutta-Nystrom fixed-step kernels."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from lunaris.core.propagation.integrators.symplectic import _pack6


def _rkn4_step(accel: Callable[[float, np.ndarray], np.ndarray], t: float, y6: np.ndarray, h: float) -> np.ndarray:
    r = np.asarray(y6[:3], dtype=np.float64)
    v = np.asarray(y6[3:6], dtype=np.float64)
    h2 = h * h

    k1 = accel(t, y6)
    r_mid = r + 0.5 * h * v + 0.125 * h2 * k1
    # k2 and k3 share this argument in the classical RKN4 (both reuse k1), so a
    # single evaluation suffices.
    k2 = accel(t + 0.5 * h, _pack6(r_mid, v))
    r_end = r + h * v + 0.5 * h2 * k2
    k4 = accel(t + h, _pack6(r_end, v))

    r_next = r + h * v + (h2 / 6.0) * (k1 + 2.0 * k2)
    v_next = v + (h / 6.0) * (k1 + 4.0 * k2 + k4)
    return _pack6(r_next, v_next)

def _rk4_step_full(rhs: Callable[[float, np.ndarray], np.ndarray], t: float, y: np.ndarray, h: float) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    k1 = np.asarray(rhs(t, y), dtype=np.float64)
    k2 = np.asarray(rhs(t + 0.5 * h, y + 0.5 * h * k1), dtype=np.float64)
    k3 = np.asarray(rhs(t + 0.5 * h, y + 0.5 * h * k2), dtype=np.float64)
    k4 = np.asarray(rhs(t + h, y + h * k3), dtype=np.float64)
    return y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

_RK8_SEQUENCE: tuple[int, ...] = (2, 4, 6, 8)

def _modified_midpoint(
    rhs: Callable[[float, np.ndarray], np.ndarray], t: float, y: np.ndarray, H: float, n: int
) -> np.ndarray:
    """Gragg's symmetric modified-midpoint rule over ``n`` sub-steps of ``H``."""
    h = H / n
    z0 = np.asarray(y, dtype=np.float64)
    z1 = z0 + h * np.asarray(rhs(t, z0), dtype=np.float64)
    for i in range(1, n):
        z2 = z0 + 2.0 * h * np.asarray(rhs(t + i * h, z1), dtype=np.float64)
        z0, z1 = z1, z2
    return 0.5 * (z0 + z1 + h * np.asarray(rhs(t + H, z1), dtype=np.float64))

def _rk8_step_full(rhs: Callable[[float, np.ndarray], np.ndarray], t: float, y: np.ndarray, h: float) -> np.ndarray:
    seq = _RK8_SEQUENCE
    table = [_modified_midpoint(rhs, t, y, h, n) for n in seq]
    # Bulirsch-Stoer extrapolation to sub-step -> 0 (Deuflhard recurrence).
    for k in range(1, len(seq)):
        for i in range(len(seq) - 1, k - 1, -1):
            ratio = (seq[i] / seq[i - k]) ** 2
            table[i] = table[i] + (table[i] - table[i - 1]) / (ratio - 1.0)
    return table[-1]


__all__ = ["_rkn4_step", "_rk4_step_full", "_RK8_SEQUENCE", "_modified_midpoint", "_rk8_step_full"]
