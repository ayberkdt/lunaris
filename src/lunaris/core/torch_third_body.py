# lunaris.core.torch_third_body
"""
Vectorized third-body gravity + ephemeris interpolation for the torch batch
path (roadmap R03, ``gpu_st_lrps_third_body`` backend).

Two building blocks, each numerically consistent with the CPU reference:

* :func:`interp_vec3_catmull_torch` — Catmull-Rom interpolation of a
  constant-step (N, 3) ephemeris table, replicating
  :func:`lunaris.common.math_utils.interp_vec3_catmull` (same endpoint
  clamping, same control-point index clamping, same basis weights). In the
  fixed-step batch loop every sample shares one epoch, so the interpolation is
  scalar-in-time and only the table lives on the device.

* :func:`third_body_accel_batch` — batched differential (tidal) third-body
  acceleration in the cancellation-free Battin ``F(q)`` form, matching
  :func:`lunaris.physics.third_body_effects.accel_third_body_numba`
  term-for-term (including the singularity guard policy: zero vector when the
  spacecraft-body or central-body distance degenerates).

Frames/units: positions in metres, Moon-centred inertial frame — the same
convention the CPU dynamics engine feeds ``accel_third_body_numba``.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

# Same singularity-guard radius^2 as the CPU kernel (physics/third_body_effects).
_MIN_R2 = 1.0


def _table_endpoint_index(t: float, dt: float, n: int) -> int:
    """CPU-parity endpoint selection (see common.math_utils._table_endpoint_index)."""
    if n <= 0:
        return -1
    if n == 1 or dt <= 0.0 or t <= 0.0:
        return 0
    if t >= dt * (n - 1):
        return n - 1
    return -1


def _table_index_frac(t: float, dt: float, n: int) -> tuple[int, float]:
    """CPU-parity segment index + fraction (see common.math_utils._table_index_frac)."""
    if dt <= 0.0 or n < 2:
        return 0, 0.0
    u = t / dt
    i = int(math.floor(u))
    if i < 0:
        i = 0
    elif i > n - 2:
        i = n - 2
    f = u - i
    if f < 0.0:
        f = 0.0
    elif f > 1.0:
        f = 1.0
    return i, f


def interp_vec3_catmull_torch(t_s: float, dt_s: float, v_tab: Any) -> Any:
    """Catmull-Rom interpolation of a device-resident ``(N, 3)`` table at ``t_s``.

    Returns a ``(3,)`` tensor on the table's device/dtype. Index/fraction math
    runs on the host from python floats (no device sync); only the 4-row gather
    and the weighted sum execute on the device.
    """
    n = int(v_tab.shape[0])
    if n == 0:
        return v_tab.new_zeros(3)

    idx = _table_endpoint_index(float(t_s), float(dt_s), n)
    if idx >= 0:
        return v_tab[idx]

    i, f = _table_index_frac(float(t_s), float(dt_s), n)
    i0 = i - 1 if i > 0 else 0
    i1 = i
    i2 = i + 1
    i3 = i + 2 if i < n - 2 else n - 1

    f2 = f * f
    f3 = f2 * f
    w0 = -f + 2.0 * f2 - f3
    w1 = 2.0 - 5.0 * f2 + 3.0 * f3
    w2 = f + 4.0 * f2 - 3.0 * f3
    w3 = -f2 + f3

    return 0.5 * (w0 * v_tab[i0] + w1 * v_tab[i1] + w2 * v_tab[i2] + w3 * v_tab[i3])


def third_body_accel_batch(r_sc: Any, r_body: Any, mu: float) -> Any:
    """Batched Battin ``F(q)`` third-body differential acceleration.

    Parameters
    ----------
    r_sc : Tensor, shape (N, 3)
        Spacecraft positions wrt the central body [m], inertial frame.
    r_body : Tensor, shape (3,)
        Third-body position wrt the central body [m], same frame/epoch.
    mu : float
        Third-body gravitational parameter [m^3/s^2].

    Returns
    -------
    Tensor, shape (N, 3)
        Differential acceleration [m/s^2]; zero rows where the singularity
        guard fires (matches the CPU kernel's policy).
    """
    import torch

    b = r_body.reshape(1, 3)
    d = b - r_sc                                        # (N, 3) sc -> body
    d2 = (d * d).sum(dim=1, keepdim=True)               # |r_tb - r_sc|^2
    b2 = (b * b).sum(dim=1, keepdim=True)                # |r_tb|^2

    r2 = (r_sc * r_sc).sum(dim=1, keepdim=True)
    r_dot_b = (r_sc * b).sum(dim=1, keepdim=True)
    # Guard denominators before dividing; guarded rows are zeroed at the end.
    ok = (d2 > _MIN_R2) & (b2 > _MIN_R2)
    safe_b2 = torch.where(ok, b2, torch.ones_like(b2))
    safe_d2 = torch.where(ok, d2, torch.ones_like(d2))

    q = (r2 - 2.0 * r_dot_b) / safe_b2
    safe_q = torch.where(ok, q, torch.zeros_like(q))
    one_plus_q = 1.0 + safe_q                            # == d2/b2 > 0 on ok rows
    f_q = safe_q * (3.0 + 3.0 * safe_q + safe_q * safe_q) / (
        1.0 + one_plus_q * torch.sqrt(one_plus_q)
    )

    inv_d3 = 1.0 / (safe_d2 * torch.sqrt(safe_d2))
    accel = (-mu * inv_d3) * (r_sc + b * f_q)
    return torch.where(ok, accel, torch.zeros_like(accel))


class TorchEphemerisTables:
    """Device-resident Sun/Earth position tables with CPU-parity interpolation.

    Built from the strict ephemeris tables the dynamics engine uses
    (``extract_ephem_tables_strict``). Fails closed (R29b #2) when a body
    enabled for third-body gravity has an all-zero table: interpolating zeros
    would silently vanish the force instead of erroring.
    """

    def __init__(
        self,
        *,
        dt_s: float,
        r_sun_tab_m: np.ndarray,
        r_earth_tab_m: np.ndarray,
        device: Any,
        dtype: Any,
        need_sun: bool,
        need_earth: bool,
    ) -> None:
        import torch

        self.dt_s = float(dt_s)
        sun = np.ascontiguousarray(r_sun_tab_m, dtype=np.float64)
        earth = np.ascontiguousarray(r_earth_tab_m, dtype=np.float64)
        if need_sun and not np.any(sun):
            raise RuntimeError(
                "gpu_st_lrps_third_body: Sun third-body gravity is enabled but the "
                "ephemeris Sun table is all zeros. Rebuild the ephemeris with "
                "include_third_body=True, or disable the Sun perturbation."
            )
        if need_earth and not np.any(earth):
            raise RuntimeError(
                "gpu_st_lrps_third_body: Earth third-body gravity is enabled but the "
                "ephemeris Earth table is all zeros. Rebuild the ephemeris with "
                "include_third_body=True, or disable the Earth perturbation."
            )
        self.r_sun_tab = torch.as_tensor(sun, device=device, dtype=dtype)
        self.r_earth_tab = torch.as_tensor(earth, device=device, dtype=dtype)

    def sun_position(self, t_s: float) -> Any:
        return interp_vec3_catmull_torch(t_s, self.dt_s, self.r_sun_tab)

    def earth_position(self, t_s: float) -> Any:
        return interp_vec3_catmull_torch(t_s, self.dt_s, self.r_earth_tab)


__all__ = [
    "TorchEphemerisTables",
    "interp_vec3_catmull_torch",
    "third_body_accel_batch",
]
