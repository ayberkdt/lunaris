"""Shared PyTorch Moon inertial/body-fixed frame helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from lunaris.common.constants import EPS_1E12, EPS_1E18, EPS_1E30, R_MOON


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
    nondegenerate = a.abs() >= EPS_1E18
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


#: Cubic-Hermite sub-samples per fixed step for batched impact detection. The
#: straight chord of a whole step sits radially below the true arc by up to
#: ``g * dt**2 / 8`` (``g = mu / r**2``): 183 m for a low lunar orbit at
#: ``dt = 30 s``, so a chord test flags orbits that pass that close above the
#: surface. Between two neighbouring sub-samples the residual sag is
#: ``g * (dt / K)**2 / 8``: 0.18 m at ``dt = 30 s`` and 0.71 m at ``dt = 60 s``
#: for ``K = 32``. The Hermite curve itself matches the arc to
#: ``dt**4 * |d4r/dt4| / 384`` (about 3 mm for the same orbit and step).
IMPACT_HERMITE_SUBSTEPS: int = 32

#: Terrain mode: the largest allowed along-track spacing of the Hermite samples,
#: as a fraction of the map's grid spacing ``radius_const * res``. The bilinear
#: map is piecewise linear between grid lines, so a one-pixel spike of height
#: ``h`` is a tent at least one pixel wide on each side of its peak along any
#: track. With spacing ``<= 0.5`` pixel a sample lies within a quarter pixel of
#: the peak and sees at least ``0.75 h``. More generally, an arc that dips
#: below the terrain by more than ``G * spacing / 2`` is caught, where ``G`` is
#: the along-track gradient of ``||p|| - terrain_r``. At ``K = 32`` a low lunar
#: orbit at ``dt = 30 s`` samples every 1.6 km, more than three LDEM 64 ppd
#: pixels (0.47 km).
IMPACT_TERRAIN_SPACING_FRACTION: float = 0.5

#: Terrain mode: cap on the per-step sample count. 512 keeps the 0.5-pixel
#: spacing on an LDEM 64 ppd map (0.237 km) for arcs up to 121 km per step,
#: i.e. ``dt <= 72 s`` at low-lunar-orbit speed (1.68 km/s) and ``dt <= 51 s``
#: at lunar escape speed (2.38 km/s, the fastest bound arc near the surface).
#: Longer steps or finer maps fall back to ``arc / 512`` spacing. The count
#: applies only to steps that pass the terrain envelope gate.
IMPACT_TERRAIN_SUBSTEPS_MAX: int = 512


def _hermite_coeffs(
    p0: Any, v0: Any, p1: Any, v1: Any, dt_s: float
) -> tuple[Any, Any, Any, Any]:
    """Power-basis coefficients of one step's cubic Hermite position curve.

    ``p(s) = c0 + c1 s + c2 s**2 + c3 s**3`` on ``s in [0, 1]`` with ``p(0) = p0``,
    ``p(1) = p1``, ``dp/ds(0) = dt v0`` and ``dp/ds(1) = dt v1`` -- the standard
    endpoint-value/endpoint-derivative Hermite basis regrouped by powers of
    ``s``. ``c0 = p0`` carries the large magnitude and the higher coefficients
    are step-sized, so Horner evaluation adds a small correction to ``p0``.
    """
    dt = float(dt_s)
    dp = p1 - p0
    a0 = dt * v0
    a1 = dt * v1
    return p0, a0, 3.0 * dp - 2.0 * a0 - a1, a0 + a1 - 2.0 * dp


def _hermite_points(coeffs: tuple[Any, Any, Any, Any], s: Any) -> Any:
    """Hermite positions ``[N, M, 3]`` at the step fractions ``s`` (``[M]`` or per-row ``[N, M]``)."""
    c0, c1, c2, c3 = (c.unsqueeze(1) for c in coeffs)
    s = s.view(1, -1, 1) if s.dim() == 1 else s.unsqueeze(2)
    return c0 + s * (c1 + s * (c2 + s * c3))


def terrain_substep_count(
    v_prev: Any,
    v_curr: Any,
    dt_s: float,
    topo: dict[str, Any],
    *,
    n_sub: int = IMPACT_HERMITE_SUBSTEPS,
    n_sub_max: int = IMPACT_TERRAIN_SUBSTEPS_MAX,
) -> Any:
    """Per-row Hermite sample count ``[N]`` (int64) for a terrain impact step.

    ``K = arc / (IMPACT_TERRAIN_SPACING_FRACTION * radius_const * res)`` rounded
    up to a multiple of ``n_sub`` and clamped to ``[n_sub, n_sub_max]`` (a
    multiple of ``n_sub``). The multiple keeps the ``n_sub`` grid ``j / n_sub``
    inside the finer one, so a step never finds its first at/below-terrain
    sample later than at ``K = n_sub``. ``arc = dt * max(|v_prev|, |v_curr|)`` [m]
    is the step's along-track length at the orbit radius. A step that passes
    the terrain envelope is within the map's relief (about 10 km) of the
    reference radius, so ``arc`` matches the ground-track length at that radius
    to better than 1%. ``res`` is the map's grid spacing [deg]
    (``res_deg``, both axes). Rows with a non-finite speed and maps without a
    grid spacing get ``n_sub``. The numba_cuda_sh kernel computes the same
    count in ``_rk4_batch_kernel``.
    """
    import torch

    k_min = float(n_sub)
    res_deg = float(topo.get("res_deg", 0.0))
    if res_deg <= 0.0:
        return torch.full((int(v_prev.shape[0]),), int(n_sub), dtype=torch.int64, device=v_prev.device)
    r_ref = float(topo.get("radius_const_m", R_MOON))
    spacing_m = IMPACT_TERRAIN_SPACING_FRACTION * r_ref * res_deg * (math.pi / 180.0)
    speed = torch.maximum((v_prev * v_prev).sum(dim=1).sqrt(), (v_curr * v_curr).sum(dim=1).sqrt())
    k_need = torch.ceil(float(dt_s) * speed / spacing_m / k_min) * k_min
    # NaN compares False, so non-finite rows fall back to n_sub.
    k = torch.where(k_need > k_min, k_need.clamp_max(float(n_sub_max)), k_need.new_full((), k_min))
    return k.to(torch.int64)


def _hermite_point_tangent(coeffs: tuple[Any, Any, Any, Any], s: Any) -> tuple[Any, Any]:
    """Hermite position ``[N, 3]`` and ``dp/ds`` ``[N, 3]`` at per-row fractions ``s`` ``[N]``."""
    c0, c1, c2, c3 = coeffs
    s = s.unsqueeze(1)
    p = c0 + s * (c1 + s * (c2 + s * c3))
    dp_ds = c1 + s * (2.0 * c2 + s * (3.0 * c3))
    return p, dp_ds


def hermite_state(prev_state: Any, curr_state: Any, dt_s: float, alpha: Any) -> Any:
    """``[N, 6]`` state on the cubic Hermite curve of one step at fraction ``alpha``.

    Built from the two step-end states (position and velocity) of the fixed-step
    integrator; the velocity is the curve's derivative ``dp/ds / dt``. At
    ``alpha = 0`` and ``alpha = 1`` it reproduces the endpoint states.
    """
    import torch

    coeffs = _hermite_coeffs(
        prev_state[:, :3], prev_state[:, 3:6], curr_state[:, :3], curr_state[:, 3:6], dt_s
    )
    p, dp_ds = _hermite_point_tangent(coeffs, alpha)
    return torch.cat((p, dp_ds / float(dt_s)), dim=1)


def hermite_sphere_intersection(
    prev_state: Any,
    curr_state: Any,
    dt_s: float,
    r_impact: Any,
    *,
    n_sub: int = IMPACT_HERMITE_SUBSTEPS,
    newton_iters: int = 3,
) -> tuple[Any, Any]:
    """Per-row hit mask and entering fraction of one step's arc into a sphere.

    Same ``(hit, alpha)`` contract as :func:`line_sphere_intersection`, but the
    step is the cubic Hermite curve through both step-end positions and
    velocities instead of their straight chord:

    1. **Gate.** The whole-step chord lies radially below a gravity-bent arc, so
       a row whose chord misses the sphere cannot hit; such rows keep
       ``hit = False`` exactly as before.
    2. **Sub-chords.** The curve is sampled at ``n_sub + 1`` fractions and the
       first sub-chord that enters the sphere is taken (residual sag
       ``g (dt / n_sub)**2 / 8``, see :data:`IMPACT_HERMITE_SUBSTEPS`). A
       sub-chord crossing still catches a step whose endpoints both lie
       outside while its middle dips inside.
    3. **Refine.** When the sub-interval ends inside the sphere, a few Newton
       steps on ``||p(s)|| - r`` (clamped to the sub-interval) move the
       sub-chord root onto the Hermite curve.
    """
    import torch

    p0 = prev_state[:, :3]
    p1 = curr_state[:, :3]
    n = int(p0.shape[0])
    k_sub = int(n_sub)

    gate, _ = line_sphere_intersection(p0, p1, r_impact)

    coeffs = _hermite_coeffs(p0, prev_state[:, 3:6], p1, curr_state[:, 3:6], dt_s)
    s_grid = torch.linspace(0.0, 1.0, k_sub + 1, dtype=p0.dtype, device=p0.device)
    q = _hermite_points(coeffs, s_grid)
    sub_hit, sub_alpha = line_sphere_intersection(
        q[:, :-1, :].reshape(-1, 3), q[:, 1:, :].reshape(-1, 3), r_impact
    )
    sub_hit = sub_hit.view(n, k_sub)
    sub_alpha = sub_alpha.view(n, k_sub)
    hit = gate & sub_hit.any(dim=1)

    k_first = sub_hit.to(torch.int32).argmax(dim=1)
    a_k = sub_alpha.gather(1, k_first.unsqueeze(1)).squeeze(1)
    k_f = k_first.to(p0.dtype)
    s_lo = k_f / k_sub
    s_hi = (k_f + 1.0) / k_sub
    s = s_lo + a_k / k_sub

    r_end = torch.linalg.norm(q, dim=2).gather(1, (k_first + 1).unsqueeze(1)).squeeze(1)
    bracketed = r_end <= r_impact
    for _ in range(int(newton_iters)):
        p, dp_ds = _hermite_point_tangent(coeffs, s)
        r = torch.linalg.norm(p, dim=1).clamp_min(EPS_1E30)
        f = r - r_impact
        fp = (p * dp_ds).sum(dim=1) / r
        descending = fp < 0.0
        step = f / fp.where(descending, fp.new_ones(()))
        s = torch.where(bracketed & descending, torch.minimum(torch.maximum(s - step, s_lo), s_hi), s)

    alpha = s.clamp(0.0, 1.0).where(hit, p0.new_ones(()))
    return hit, alpha


def sample_topo_radius_torch(topo: dict[str, Any], lat_deg: Any, lon_deg: Any) -> Any:
    """Vectorised terrain-radius sampler for a ``loaders.grid_topo_payload``.

    Torch counterpart of ``loaders.io_surface.sample_topo_radius_m`` and the Numba
    ``_sample_topo_radius_cuda`` device function: longitude wrapped to ``[0, 360)``,
    latitude clamped to ``[lat_min_deg, lat_max_deg]``, scaled bilinear lookup
    ``scale_m * DN + bias_m``. ``lat_deg``/``lon_deg`` are ``[N]`` tensors; returns
    a ``[N]`` tensor of surface radii [m].

    When ``topo`` carries no grid (constant fallback) every sample resolves to the
    payload's ``radius_const_m`` so the caller stays terrain-clean over voids.
    """
    import torch

    dn = topo.get("dn")
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


def topo_payload_to_torch(
    payload: dict[str, Any] | None,
    *,
    device: Any,
    dtype: Any,
) -> dict[str, Any] | None:
    """Move a ``loaders.grid_topo_payload`` onto a torch device for the kernels.

    Returns ``None`` when ``payload`` is missing or carries no grid (constant
    fallback), signalling the caller to keep the constant-sphere impact path.
    Otherwise the ``dn`` raster becomes a device tensor and the POD metadata is
    passed through unchanged (consumed as Python scalars by
    :func:`sample_topo_radius_torch`).
    """
    import torch

    if payload is None or payload.get("dn", None) is None:
        return None
    out = dict(payload)
    out["dn"] = torch.as_tensor(
        np.asarray(payload["dn"], dtype=np.float64), device=device, dtype=dtype
    )
    if payload.get("map_rotation") is not None:
        out["map_rotation"] = torch.as_tensor(
            np.asarray(payload["map_rotation"], dtype=np.float64).reshape(3, 3),
            device=device,
            dtype=dtype,
        )
    return out


def topo_latlon_deg(topo: dict[str, Any], p_bf: Any, r_safe: Any) -> tuple[Any, Any]:
    """Map latitude/longitude [deg] of ``[N, 3]`` body-fixed positions.

    Applies the payload's body-fixed -> map-frame rotation (``map_rotation``,
    present when the propagation frame is MOON_PA and the map is on mean-Earth
    axes, see :mod:`lunaris.common.lunar_map_frame`) before the angles are
    taken; the rotation preserves ``r_safe``.
    """
    import torch

    rotation = topo.get("map_rotation")
    p_map = p_bf if rotation is None else p_bf @ rotation.T
    lat = torch.rad2deg(torch.asin((p_map[:, 2] / r_safe).clamp(-1.0, 1.0)))
    lon = torch.rad2deg(torch.atan2(p_map[:, 1], p_map[:, 0]))
    return lat, lon


def terrain_segment_intersection(
    p_prev: Any,
    p_curr: Any,
    *,
    t_prev_s: float,
    dt_s: float,
    frame: Any,
    topo: dict[str, Any],
    impact_alt_m: float,
    bisection_iters: int = 27,
    v_prev: Any = None,
    v_curr: Any = None,
    n_sub: int = IMPACT_HERMITE_SUBSTEPS,
    n_sub_max: int = IMPACT_TERRAIN_SUBSTEPS_MAX,
) -> tuple[Any, Any]:
    """Terrain-aware analogue of :func:`hermite_sphere_intersection`.

    The step is the cubic Hermite curve through the step-end positions and
    velocities ``v_prev`` / ``v_curr`` (when they are omitted both tangents are
    the chord direction and the curve *is* the straight chord). Three stages:

    1. **Outer envelope.** A step can only touch terrain if it reaches the
       bounding sphere ``r_terrain_max + impact_alt`` (max terrain height plus the
       altitude threshold). The whole-step chord lies below the arc, so rows
       whose chord misses this sphere cannot impact. A row that already starts
       inside the envelope (a descent lasting several steps) is examined too.
       Only the rows through this gate are processed further (host syncs per
       call size the subset and its sample count); the others return
       ``hit = False``.
    2. **Hermite samples.** ``f(s) = ||p(s)|| - terrain_radius(lat, lon) -
       impact_alt`` is evaluated at ``K + 1`` fractions ``s = j / K``, with ``K``
       per row from :func:`terrain_substep_count` (along-track spacing at most
       :data:`IMPACT_TERRAIN_SPACING_FRACTION` of the map's grid spacing, ``K``
       in ``[n_sub, n_sub_max]``), in blocks of ``n_sub + 1`` samples. The first
       sample at or below the terrain marks the impact (a row starting at/below
       it enters at ``alpha = 0``). Sampling the curve rather than only the step
       end also catches a dip below the terrain between the step ends.
    3. **Terrain bisection** of ``f`` on the Hermite curve inside the bracketing
       sub-interval (width ``1 / K``; the default 27 halvings resolve at least
       ``2**-32`` of the step, as the former 32 halvings of ``[0, 1]`` did).

    The sub-point latitude/longitude come from the body-fixed position; the
    inertial→fixed rotation is evaluated once per step at its midpoint (Moon
    rotation over a single step is negligible relative to terrain relief),
    keeping the SLERP-frame provenance.

    Returns ``(hit, alpha)`` with the same contract as
    :func:`line_sphere_intersection`: non-hit rows get ``alpha = 1``. The caller
    forms the crossing state on the same Hermite curve (:func:`hermite_state`),
    so position **and** velocity freeze on the terrain surface.

    Limitations: an arc that dips below the terrain by less than ``G * spacing
    / 2`` between two samples (``G`` the along-track gradient of ``f``) can go
    undetected; for a one-pixel spike that is the top quarter of its height.
    Past the ``n_sub_max`` cap the spacing is ``arc / n_sub_max`` instead. East-
    west grid spacing shrinks by ``cos(lat)`` while ``K`` uses the meridional
    spacing, so near the poles a feature one longitude pixel wide is narrower
    than the sample spacing.
    """
    import torch

    dp = p_curr - p_prev
    if v_prev is None:
        v_prev = dp / float(dt_s)
    if v_curr is None:
        v_curr = dp / float(dt_s)
    impact_alt = float(impact_alt_m)
    r_outer = float(topo.get("r_terrain_max_m", topo.get("radius_const_m", 0.0))) + impact_alt
    hit_outer, _ = line_sphere_intersection(p_prev, p_curr, r_outer)
    # A step that starts inside the envelope has its entering root at
    # alpha <= 0, outside line_sphere_intersection's [0, 1] hit window, so a
    # descent lasting several steps never bracketed the ground.
    starts_inside = torch.linalg.norm(p_prev, dim=1) <= r_outer
    hit_outer = hit_outer | starts_inside

    n = int(p_prev.shape[0])
    k_sub = int(n_sub)
    one = p_prev.new_ones(n)
    k_rows = terrain_substep_count(v_prev, v_curr, dt_s, topo, n_sub=k_sub, n_sub_max=n_sub_max)
    # Host sync: the largest gated count sizes the sample blocks; 0 = no row gated.
    k_max = int(torch.where(hit_outer, k_rows, torch.zeros_like(k_rows)).max()) if n else 0
    if k_max == 0:
        return torch.zeros_like(hit_outer), one
    idx = torch.nonzero(hit_outer, as_tuple=False).squeeze(1)
    g = int(idx.numel())
    k_g = k_rows.index_select(0, idx)

    t_rep = float(t_prev_s) + 0.5 * float(dt_s)
    use_rot = frame is not None and bool(getattr(frame, "uses_rotation", False))

    def _f_at(p: Any) -> Any:
        r = torch.linalg.norm(p, dim=1)
        p_bf = frame.inertial_to_fixed(t_rep, p) if use_rot else p
        r_safe = r.clamp_min(EPS_1E30)
        lat, lon = topo_latlon_deg(topo, p_bf, r_safe)
        terrain_r = sample_topo_radius_torch(topo, lat, lon)
        return r - terrain_r - impact_alt

    coeffs = _hermite_coeffs(
        p_prev.index_select(0, idx),
        v_prev.index_select(0, idx),
        p_curr.index_select(0, idx),
        v_curr.index_select(0, idx),
        dt_s,
    )
    c0, c1, c2, c3 = coeffs

    def _f(alpha: Any) -> Any:
        s = alpha.unsqueeze(1)
        return _f_at(c0 + s * (c1 + s * (c2 + s * c3)))

    # Samples s = j / K, j = 0..K, in blocks of n_sub + 1 (a single block when
    # every gated row has K = n_sub); past a row's own K, s holds at 1. A true
    # division keeps s = m / n_sub exact where j = m K / n_sub.
    k_gf = k_g.to(p_prev.dtype)
    found = torch.zeros(g, dtype=torch.bool, device=p_prev.device)
    k_first = torch.zeros(g, dtype=torch.int64, device=p_prev.device)
    for j0 in range(0, k_max + 1, k_sub + 1):
        j = torch.arange(
            j0, min(j0 + k_sub + 1, k_max + 1), dtype=p_prev.dtype, device=p_prev.device
        )
        s_blk = (j.unsqueeze(0) / k_gf.unsqueeze(1)).clamp_max(1.0)
        q = _hermite_points(coeffs, s_blk)
        below = (_f_at(q.reshape(-1, 3)) <= 0.0).view(g, -1)
        # Impact iff some Hermite sample is at/below the terrain; the first
        # such sample closes the bracket.
        has = below.any(dim=1)
        first = below.to(torch.int32).argmax(dim=1).to(torch.int64) + j0
        k_first = torch.where(found | ~has, k_first, first)
        found = found | has
    # A step that already starts at/below the terrain freezes at its start.
    already = found & (k_first == 0)

    k_f = k_first.to(p_prev.dtype)
    lo = (k_f - 1.0).clamp_min(0.0) / k_gf
    hi = k_f / k_gf
    for _ in range(int(bisection_iters)):
        mid = 0.5 * (lo + hi)
        go_right = _f(mid) > 0.0
        lo = torch.where(go_right, mid, lo)
        hi = torch.where(go_right, hi, mid)

    alpha_g = torch.where(found, 0.5 * (lo + hi), p_prev.new_ones(g))
    alpha_g = torch.where(already, torch.zeros_like(alpha_g), alpha_g)
    bracket = torch.zeros_like(hit_outer).index_copy(0, idx, found)
    alpha = one.index_copy(0, idx, alpha_g)
    return bracket, alpha


@dataclass(frozen=True, slots=True)
class TorchFrameCache:
    """Precomputed per-step, per-RK-stage frame quaternions (i2f and its conjugate)."""

    q_i2f: Any
    q_f2i: Any


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
        ).clamp_min(EPS_1E30)
        self.uses_rotation = True

    def quat_i2f(self, t_s: float) -> Any:
        """Return ``q_i2f`` at ``t_s`` using shortest-path SLERP."""
        import torch

        if self.q_tab.shape[0] <= 1:
            return self.q_tab[0]
        u = max(0.0, float(t_s) / max(self.dt_s, EPS_1E12))
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
        sin_theta_0 = torch.sin(theta_0).clamp_min(EPS_1E30)
        theta = theta_0 * frac
        q_slerp = (
            torch.sin(theta_0 - theta) / sin_theta_0 * qa
            + torch.sin(theta) / sin_theta_0 * qb
        )
        q = torch.where(dot > 0.9995, q_linear, q_slerp)
        return q / torch.linalg.norm(q).clamp_min(EPS_1E30)

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

    def precompute_rk_stage_quaternions(
        self, total_steps: int, dt_eff: float, stage_time_offsets: list[float]
    ) -> TorchFrameCache:
        """Vectorized SLERP of ``q_i2f`` for every step x RK-stage epoch.

        ``stage_time_offsets`` are the per-step stage times relative to the step
        start (e.g. ``[0, dt/2, dt/2, dt]`` for classic RK4). Returns tensors of
        shape ``(total_steps, n_stages, 4)`` matching the per-epoch semantics of
        :meth:`quat_i2f`, including the held-endpoint behaviour past the table.
        """
        import torch

        t_base = torch.arange(total_steps, dtype=torch.float64, device=self.device) * dt_eff
        t_rel = torch.tensor(stage_time_offsets, dtype=torch.float64, device=self.device)
        t_all = t_base[:, None] + t_rel[None, :]

        if not self.uses_rotation or self.q_tab.shape[0] <= 1:
            q_i2f = self.q_tab[0].expand(*t_all.shape, 4).clone()
        else:
            u = (t_all / max(self.dt_s, EPS_1E12)).clamp_min(0.0)
            i0 = torch.floor(u).long()
            max_idx = self.q_tab.shape[0] - 2
            i0 = i0.clamp(max=max_idx)

            # Match the dynamic quat_i2f() out-of-range behaviour: beyond the
            # final ephemeris interval (u >= N-1) hold the LAST quaternion rather
            # than extrapolating. Clamping i0 alone would leave frac > 1 and
            # extrapolate; clamping frac to [0, 1] reproduces the held endpoint.
            frac = (u - i0).clamp(0.0, 1.0).to(dtype=self.dtype)
            qa = self.q_tab[i0]
            qb = self.q_tab[i0 + 1]

            dot = (qa * qb).sum(dim=-1)
            sign = torch.where(dot < 0.0, -torch.ones_like(dot), torch.ones_like(dot))
            qb = qb * sign.unsqueeze(-1)
            dot = (dot * sign).clamp(-1.0, 1.0)

            q_linear = (1.0 - frac.unsqueeze(-1)) * qa + frac.unsqueeze(-1) * qb

            theta_0 = torch.acos(dot)
            sin_theta_0 = torch.sin(theta_0).clamp_min(EPS_1E30)
            theta = theta_0 * frac

            s0 = (torch.sin(theta_0 - theta) / sin_theta_0).unsqueeze(-1)
            s1 = (torch.sin(theta) / sin_theta_0).unsqueeze(-1)

            q_slerp = s0 * qa + s1 * qb
            q_i2f = torch.where((dot > 0.9995).unsqueeze(-1), q_linear, q_slerp)
            q_i2f = q_i2f / torch.linalg.norm(q_i2f, dim=-1, keepdim=True).clamp_min(EPS_1E30)

        q_f2i = q_i2f.clone()
        q_f2i[..., 1:] = -q_f2i[..., 1:]

        return TorchFrameCache(q_i2f=q_i2f, q_f2i=q_f2i)


__all__ = [
    "IMPACT_HERMITE_SUBSTEPS",
    "IMPACT_TERRAIN_SPACING_FRACTION",
    "IMPACT_TERRAIN_SUBSTEPS_MAX",
    "TorchFrameCache",
    "TorchFrameError",
    "TorchMoonFrame",
    "hermite_sphere_intersection",
    "hermite_state",
    "line_sphere_alpha",
    "line_sphere_intersection",
    "quat_conjugate_torch",
    "quat_rotate_torch",
    "sample_topo_radius_torch",
    "terrain_segment_intersection",
    "terrain_substep_count",
    "topo_latlon_deg",
    "topo_payload_to_torch",
]
