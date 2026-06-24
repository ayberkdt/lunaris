"""Shared PyTorch Moon inertial/body-fixed frame helpers."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


class TorchFrameError(RuntimeError):
    """Raised when the Torch frame timeline cannot satisfy the runtime contract."""


def quat_rotate_torch(q: Any, v: Any) -> Any:
    """Rotate ``[N, 3]`` vectors by scalar-first quaternion ``q``."""
    import torch

    q0, q1, q2, q3 = q[0], q[1], q[2], q[3]
    vx, vy, vz = v[:, 0], v[:, 1], v[:, 2]
    tx = 2.0 * (q2 * vz - q3 * vy)
    ty = 2.0 * (q3 * vx - q1 * vz)
    tz = 2.0 * (q1 * vy - q2 * vx)
    rx = vx + q0 * tx + (q2 * tz - q3 * ty)
    ry = vy + q0 * ty + (q3 * tx - q1 * tz)
    rz = vz + q0 * tz + (q1 * ty - q2 * tx)
    return torch.stack((rx, ry, rz), dim=1)


def quat_conjugate_torch(q: Any) -> Any:
    """Return the conjugate of a scalar-first quaternion."""
    import torch

    return torch.cat((q[:1], -q[1:]))


def line_sphere_intersection(
    p_prev: Any,
    p_curr: Any,
    r_impact: Any,
) -> tuple[Any, Any]:
    """Return per-row segment-hit mask and entering fraction for a sphere.

    Solves ``||p_prev + alpha*(p_curr - p_prev)||^2 = r_impact^2`` and returns the
    entering (smaller) root. Unlike an endpoint-radius check, the hit mask also
    catches a segment whose two endpoints are outside while its interior crosses
    the sphere. Invalid or degenerate rows receive ``alpha=1``.
    """
    import torch

    dp = p_curr - p_prev
    a = (dp * dp).sum(dim=1)
    b = 2.0 * (p_prev * dp).sum(dim=1)
    c = (p_prev * p_prev).sum(dim=1) - r_impact * r_impact
    disc = b * b - 4.0 * a * c
    disc_scale = (b * b + (4.0 * a * c).abs()).clamp_min(1.0)
    disc_tol = 32.0 * torch.finfo(a.dtype).eps * disc_scale
    has_real_root = disc >= -disc_tol
    sqrt_disc = disc.clamp_min(0.0).sqrt()
    one = a.new_ones(())
    nondegenerate = a.abs() >= 1e-18
    a_safe = a.where(nondegenerate, one)
    alpha_raw = (-b - sqrt_disc) / (2.0 * a_safe)
    alpha_tol = 32.0 * torch.finfo(a.dtype).eps
    hit = (
        nondegenerate
        & has_real_root
        & (alpha_raw >= -alpha_tol)
        & (alpha_raw <= 1.0 + alpha_tol)
    )
    alpha = alpha_raw.clamp(0.0, 1.0).where(hit, one)
    return hit, alpha


def line_sphere_alpha(p_prev: Any, p_curr: Any, r_impact: Any) -> Any:
    """Backward-compatible entering fraction for known crossing segments."""
    _, alpha = line_sphere_intersection(p_prev, p_curr, r_impact)
    return alpha


def sample_topo_radius_torch(topo: dict[str, Any], lat_deg: Any, lon_deg: Any) -> Any:
    """Vectorised terrain-radius sampler for a ``loaders._grid_topo_payload``.

    Torch counterpart of ``loaders.io_surface.sample_topo_radius_m`` and the Numba
    ``_sample_topo_radius_cuda`` device function: longitude wrapped to ``[0, 360)``,
    latitude clamped to ``[lat_min_deg, lat_max_deg]``, scaled bilinear lookup
    ``scale_m * DN + bias_m``. ``lat_deg``/``lon_deg`` are ``[N]`` tensors; returns
    a ``[N]`` tensor of surface radii [m].

    When ``topo`` carries no grid (constant fallback) every sample resolves to the
    payload's ``radius_const_m`` so the caller stays terrain-clean over voids.
    """
    import torch

    dn = topo.get("dn", None)
    const_m = float(topo.get("radius_const_m", 0.0))
    if dn is None:
        return torch.full_like(lat_deg, const_m)

    n_lines = int(topo["n_lines"])
    n_samples = int(topo["n_samples"])
    res = float(topo["res_deg"])
    lon0 = float(topo["lon0_deg"])
    lat0 = float(topo["lat0_deg"])
    scale_m = float(topo["scale_m"])
    bias_m = float(topo["bias_m"])
    flip = int(topo.get("flip_lat", 0))
    lat_min = float(topo["lat_min_deg"])
    lat_max = float(topo["lat_max_deg"])

    lon_w = torch.remainder(lon_deg, 360.0)
    lat_c = lat_deg.clamp(lat_min, lat_max)
    lat_sign = 1.0 if flip == 0 else -1.0
    i_f = (lat_sign * (lat0 - lat_c)) / res
    j_f = torch.remainder((lon_w - lon0) / res, float(n_samples))

    i0 = torch.floor(i_f)
    j0 = torch.floor(j_f)
    di = i_f - i0
    dj = j_f - j0
    i0l = i0.long().clamp(0, n_lines - 1)
    i1l = (i0.long() + 1).clamp(0, n_lines - 1)
    j0l = j0.long().clamp(0, n_samples - 1)
    j1l = (j0.long() + 1).clamp(0, n_samples - 1)

    dn = dn if dn.dtype == lat_deg.dtype else dn.to(lat_deg.dtype)
    v00 = dn[i0l, j0l]
    v01 = dn[i0l, j1l]
    v10 = dn[i1l, j0l]
    v11 = dn[i1l, j1l]
    top = v00 * (1.0 - dj) + v01 * dj
    bot = v10 * (1.0 - dj) + v11 * dj
    dn_interp = top * (1.0 - di) + bot * di
    return scale_m * dn_interp + bias_m


def terrain_segment_intersection(
    p_prev: Any,
    p_curr: Any,
    *,
    t_prev_s: float,
    dt_s: float,
    frame: Any,
    topo: dict[str, Any],
    impact_alt_m: float,
    bisection_iters: int = 32,
) -> tuple[Any, Any]:
    """Terrain-aware analogue of :func:`line_sphere_intersection`.

    Two stages, fully vectorised:

    1. **Outer envelope.** A segment can only touch terrain if it reaches the
       bounding sphere ``r_terrain_max + impact_alt`` (max terrain height plus the
       altitude threshold). Rows that miss this sphere cannot impact.
    2. **Terrain bisection.** On ``[alpha_enter, 1]`` solve the downward crossing
       of ``f(alpha) = ||p(alpha)|| - terrain_radius(lat, lon) - impact_alt``. The
       sub-point latitude/longitude come from the body-fixed position; the
       inertial→fixed rotation is evaluated once per segment at its midpoint
       (Moon rotation over a single step is negligible relative to terrain relief),
       keeping the SLERP-frame provenance.

    Returns ``(hit, alpha)`` with the same contract as
    :func:`line_sphere_intersection`: non-hit rows get ``alpha = 1``. The caller
    forms the crossing state by linear interpolation at ``alpha`` exactly as for
    the sphere path, so position **and** velocity freeze on the terrain surface.
    """
    import torch

    dp = p_curr - p_prev
    impact_alt = float(impact_alt_m)
    r_outer = float(topo.get("r_terrain_max_m", topo.get("radius_const_m", 0.0))) + impact_alt
    hit_outer, alpha_enter = line_sphere_intersection(p_prev, p_curr, r_outer)

    t_rep = float(t_prev_s) + 0.5 * float(dt_s)
    use_rot = frame is not None and bool(getattr(frame, "uses_rotation", False))

    def _f(alpha: Any) -> Any:
        p = p_prev + alpha.unsqueeze(1) * dp
        r = torch.linalg.norm(p, dim=1)
        p_bf = frame.inertial_to_fixed(t_rep, p) if use_rot else p
        r_safe = r.clamp_min(1e-30)
        lat = torch.rad2deg(torch.asin((p_bf[:, 2] / r_safe).clamp(-1.0, 1.0)))
        lon = torch.rad2deg(torch.atan2(p_bf[:, 1], p_bf[:, 0]))
        terrain_r = sample_topo_radius_torch(topo, lat, lon)
        return r - terrain_r - impact_alt

    one = p_prev.new_ones(p_prev.shape[0])
    lo = alpha_enter
    hi = one
    f_lo = _f(lo)
    f_hi = _f(hi)

    # Impact iff the envelope is entered and the step endpoint is at/below terrain.
    # (A transient dip-and-recover within one step is below this scheme's
    # resolution and is intentionally treated as no-impact, mirroring the
    # entry-only sphere logic.)
    bracket = hit_outer & (f_hi <= 0.0)

    # Where the segment is already below terrain at envelope entry, freeze at entry.
    already = bracket & (f_lo <= 0.0)

    for _ in range(int(bisection_iters)):
        mid = 0.5 * (lo + hi)
        go_right = _f(mid) > 0.0
        lo = torch.where(go_right, mid, lo)
        hi = torch.where(go_right, hi, mid)

    alpha = torch.where(bracket, 0.5 * (lo + hi), one)
    alpha = torch.where(already, alpha_enter, alpha)
    return bracket, alpha


class TorchMoonFrame:
    """SLERP-interpolated Moon inertial-to-fixed quaternion timeline."""

    def __init__(
        self,
        ephem: Any,
        *,
        device: Any,
        dtype: Any,
        allow_identity: bool = False,
    ) -> None:
        import torch

        self.device = device
        self.dtype = dtype
        if ephem is None:
            if not allow_identity:
                raise TorchFrameError(
                    "Moon-fixed Torch gravity requires an ephemeris q_i2f timeline. "
                    "Identity rotation is allowed only in explicit validation tests."
                )
            self.uses_rotation = False
            self.dt_s = 1.0
            self.q_tab = torch.tensor(
                [[1.0, 0.0, 0.0, 0.0]], device=device, dtype=dtype
            )
            return

        provider = ephem.get_data_provider()
        q_raw = provider.get("q_i2f_tab", provider.get("rot_table"))
        dt_raw = provider.get("dt_s", provider.get("dt"))
        if q_raw is None or dt_raw is None:
            raise TorchFrameError(
                "Ephemeris provider must expose q_i2f_tab/rot_table and dt_s/dt."
            )
        q_np = np.asarray(q_raw, dtype=np.float64)
        if q_np.ndim != 2 or q_np.shape[1] != 4 or q_np.shape[0] < 1:
            raise TorchFrameError(
                f"ephemeris q_i2f timeline must have shape (N, 4); got {q_np.shape}."
            )
        self.dt_s = float(dt_raw)
        if self.dt_s <= 0.0:
            raise TorchFrameError(f"ephemeris frame dt must be > 0; got {self.dt_s}.")
        self.q_tab = torch.as_tensor(q_np, device=device, dtype=dtype)
        self.q_tab = self.q_tab / torch.linalg.norm(
            self.q_tab, dim=1, keepdim=True
        ).clamp_min(1e-30)
        self.uses_rotation = True

    def quat_i2f(self, t_s: float) -> Any:
        """Return ``q_i2f`` at ``t_s`` using shortest-path SLERP."""
        import torch

        if self.q_tab.shape[0] <= 1:
            return self.q_tab[0]
        u = max(0.0, float(t_s) / max(self.dt_s, 1e-12))
        i0 = int(math.floor(u))
        if i0 >= self.q_tab.shape[0] - 1:
            return self.q_tab[-1]
        frac = torch.tensor(u - i0, device=self.device, dtype=self.dtype)
        qa = self.q_tab[i0]
        qb = self.q_tab[i0 + 1]
        dot_raw = torch.dot(qa, qb)
        sign = torch.where(
            dot_raw < 0.0, -torch.ones_like(dot_raw), torch.ones_like(dot_raw)
        )
        qb = qb * sign
        dot = torch.clamp(dot_raw * sign, -1.0, 1.0)
        q_linear = (1.0 - frac) * qa + frac * qb
        theta_0 = torch.acos(dot)
        sin_theta_0 = torch.sin(theta_0).clamp_min(1e-30)
        theta = theta_0 * frac
        q_slerp = (
            torch.sin(theta_0 - theta) / sin_theta_0 * qa
            + torch.sin(theta) / sin_theta_0 * qb
        )
        q = torch.where(dot > 0.9995, q_linear, q_slerp)
        return q / torch.linalg.norm(q).clamp_min(1e-30)

    def inertial_to_fixed(self, t_s: float, vectors: Any) -> Any:
        if not self.uses_rotation:
            return vectors
        return quat_rotate_torch(self.quat_i2f(t_s), vectors)

    def fixed_to_inertial(self, t_s: float, vectors: Any) -> Any:
        if not self.uses_rotation:
            return vectors
        return quat_rotate_torch(
            quat_conjugate_torch(self.quat_i2f(t_s)),
            vectors,
        )


__all__ = [
    "TorchFrameError",
    "TorchMoonFrame",
    "line_sphere_alpha",
    "line_sphere_intersection",
    "quat_conjugate_torch",
    "quat_rotate_torch",
    "sample_topo_radius_torch",
    "terrain_segment_intersection",
]
