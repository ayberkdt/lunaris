"""Numba CUDA batch-kernel fixes, checked under Numba's CPU simulator.

Each test runs the real device functions of ``lunaris.core.batch_propagator``
with ``NUMBA_ENABLE_CUDASIM=1`` and compares them with the CPU kernels they must
mirror:

* t = 0 terrain impact: the host mask uses the terrain radius, not the mean
  sphere (a sample under the datum but above a basin floor is not impacted).
* SRP shadow: ``_srp_accel_cuda`` equals ``physics.solar_effects.accel_srp``
  over a sweep of Moon-shadow, penumbra, day-side and lunar-eclipse (Earth
  shadow) geometries. The former cylinder measured its axis projection from
  the Sun, so it blanked the sunlit hemisphere behind the Moon's disk too.
* SH polar axis: ``_sh_accel_cuda`` matches the CPU serial kernel on, next to
  and away from the rotation axis (the CPU applies the analytic m = 1 limit).
* Quaternion table ends: ``_interp4_cuda`` holds the end rows like the CPU
  ``interp_quat_safe`` instead of extrapolating the SLERP parameter.

The simulator evaluates the kernels in Python float64, so these pin the
algorithm and branch structure; they do not exercise NVVM code generation.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path


def _run_cudasim(body: str) -> subprocess.CompletedProcess[str]:
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["NUMBA_ENABLE_CUDASIM"] = "1"
    env["PYTHONPATH"] = str(root / "src") + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(body)],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _assert_ok(result: subprocess.CompletedProcess[str]) -> None:
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout, result.stdout + result.stderr


def test_numba_t0_impact_uses_terrain_radius_not_mean_sphere() -> None:
    script = """
    import math
    import numpy as np

    from lunaris.core.batch_propagator import (
        _initial_impact_bookkeeping,
        _initial_terrain_at_surface,
        topo_meta_array,
    )

    DATUM = 1737400.0
    lat_c = np.arange(89.5, -90.0, -1.0)
    lon_c = np.arange(0.5, 360.0, 1.0)
    lat_g, lon_g = np.meshgrid(lat_c, lon_c, indexing="ij")
    dn = np.where(lon_g < 180.0, -3000.0, 4000.0)  # basin west, plateau east
    dn = np.ascontiguousarray(dn)
    payload = {
        "dn": dn, "n_lines": lat_c.size, "n_samples": lon_c.size, "res_deg": 1.0,
        "lon0_deg": 0.5, "lat0_deg": 89.5, "scale_m": 1.0, "bias_m": DATUM,
        "missing_dn": float("nan"), "flip_lat": 0,
        "lat_min_deg": -89.5, "lat_max_deg": 89.5,
        "r_terrain_min_m": DATUM - 3000.0, "r_terrain_max_m": DATUM + 4000.0,
        "radius_const_m": DATUM,
    }

    def at(lon_deg, radius):
        lon = math.radians(lon_deg)
        return [radius * math.cos(lon), radius * math.sin(lon), 0.0, 0.0, 0.0, 0.0]

    Y0 = np.array([
        at(90.0, DATUM - 1000.0),   # under the datum, 2 km above the basin floor
        at(90.0, DATUM - 3500.0),   # under the basin floor
        at(270.0, DATUM + 1000.0),  # above the datum, inside the plateau
        at(270.0, DATUM + 7000.0),  # 3 km above the plateau
    ])
    q_tab = np.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    mask = _initial_terrain_at_surface(Y0, q_tab, dn, topo_meta_array(payload), 0.0, 32)
    assert mask.tolist() == [False, True, True, False], mask
    flags, times, pos = _initial_impact_bookkeeping(
        Y0, DATUM, detect_impact=True, at_surface=mask
    )
    assert flags.tolist() == [0, 1, 1, 0], flags
    assert times[1] == 0.0 and times[2] == 0.0 and np.isnan(times[0])
    # The mean-sphere rule this replaces would have frozen samples 0 and 1 only.
    sphere_flags, _, _ = _initial_impact_bookkeeping(Y0, DATUM, detect_impact=True)
    assert sphere_flags.tolist() == [1, 1, 0, 0]
    # A 2.5 km impact altitude catches sample 0 (2 km above ground), not sample 3 (3 km).
    mask_alt = _initial_terrain_at_surface(Y0, q_tab, dn, topo_meta_array(payload), 2500.0, 32)
    assert mask_alt.tolist() == [True, True, True, False], mask_alt
    print("OK")
    """
    _assert_ok(_run_cudasim(script))


def test_numba_srp_shadow_matches_cpu_accel_srp_over_geometry_sweep() -> None:
    script = """
    import math
    import numpy as np
    from numba import cuda

    from lunaris.common.constants import AU, P_SUN_1AU, R_EARTH_MEAN, R_MOON_MEAN
    from lunaris.core.batch_propagator import _srp_accel_cuda
    from lunaris.physics.solar_effects import accel_srp

    @cuda.jit
    def launch(r, sun, earth, r_moon, r_earth, out):
        k = cuda.grid(1)
        if k < r.shape[0]:
            a = cuda.local.array(3, np.float64)
            _srp_accel_cuda(
                r[k, 0], r[k, 1], r[k, 2], sun[k, 0], sun[k, 1], sun[k, 2],
                earth[k, 0], earth[k, 1], earth[k, 2], r_moon, r_earth,
                AU, P_SUN_1AU, 1.3, 4.0, 500.0, a,
            )
            out[k, 0] = a[0]
            out[k, 1] = a[1]
            out[k, 2] = a[2]

    sun_dir = np.array([0.8, 0.6, 0.0])
    sun = AU * sun_dir
    perp = np.array([-0.6, 0.8, 0.0])
    pts = []
    # 1) 100 km circular orbit in the Sun plane, 2 deg steps: day side,
    #    terminator and the Moon's umbra.
    n_orbit = 180
    for k in range(n_orbit):
        th = math.radians(2.0 * k)
        pts.append((R_MOON_MEAN + 1.0e5) * (math.cos(th) * sun_dir + math.sin(th) * perp))
    # 2) Cut across the shadow edge 2000 km behind the Moon (penumbra ~19 km
    #    wide there; 4.4 km spacing).
    for k in range(80):
        rho = R_MOON_MEAN * (0.9 + 0.2 * k / 79.0)
        pts.append(-2.0e6 * sun_dir + rho * perp)
    r = np.ascontiguousarray(np.array(pts))
    n = r.shape[0]
    sun_arr = np.ascontiguousarray(np.tile(sun, (n, 1)))
    # Unshadowed SRP magnitude: the error scale for every comparison below.
    full_sun = float(np.linalg.norm(accel_srp(
        *r[0], *sun, 0.0, 0.0, 0.0, R_MOON_MEAN, R_EARTH_MEAN, AU, P_SUN_1AU,
        1.3, 4.0, 500.0, False, False)))

    def compare(earth_vec, moon_on, earth_on):
        earth_arr = np.ascontiguousarray(np.tile(earth_vec, (n, 1)))
        out = np.zeros((n, 3))
        launch[(n + 63) // 64, 64](
            r, sun_arr, earth_arr,
            R_MOON_MEAN if moon_on else 0.0, R_EARTH_MEAN if earth_on else 0.0, out,
        )
        earth_r2 = float(np.dot(earth_vec, earth_vec))
        ref = np.array([
            accel_srp(*r[k], *sun, *earth_vec, R_MOON_MEAN, R_EARTH_MEAN, AU, P_SUN_1AU,
                      1.3, 4.0, 500.0, moon_on, earth_on and earth_r2 > 1.0e12)
            for k in range(n)
        ])
        err = float(np.max(np.abs(out - ref))) / full_sun
        return out, ref, err

    out, ref, err = compare(np.zeros(3), True, False)
    assert err < 1e-12, err
    lit = np.linalg.norm(ref, axis=1) > 0.0
    assert 0 < np.count_nonzero(~lit) < n
    # Sunlit side of the orbit (|theta| < 90 deg) is never blanked.
    th = 2.0 * np.arange(n_orbit)
    day = (th < 90.0) | (th > 270.0)
    assert np.all(np.linalg.norm(out[:n_orbit][day], axis=1) > 0.0)
    # Penumbra exists: some partial factors strictly between 0 and full.
    full = np.linalg.norm(ref[0])
    partial = (np.linalg.norm(ref, axis=1) > 0.0) & (np.linalg.norm(ref, axis=1) < 0.999 * full)
    assert np.count_nonzero(partial) > 0

    # Earth between Sun and Moon (lunar eclipse): Earth shadow on and off.
    earth_vec = 3.844e8 * sun_dir
    out_e, ref_e, err_e = compare(earth_vec, True, True)
    assert err_e < 1e-12, err_e
    assert np.count_nonzero(np.linalg.norm(ref_e, axis=1) == 0.0) > np.count_nonzero(~lit)
    out_m, ref_m, err_m = compare(earth_vec, False, False)
    assert err_m < 1e-12 and np.all(np.linalg.norm(out_m, axis=1) > 0.0)
    print("OK", err, err_e, err_m)
    """
    _assert_ok(_run_cudasim(script))


def test_numba_sh_matches_cpu_on_and_near_polar_axis() -> None:
    script = """
    import numpy as np
    from numba import cuda

    from lunaris.common.constants import MU_MOON, R_MOON
    from lunaris.core.batch_propagator import _sh_accel_cuda
    from lunaris.physics.spherical_harmonics import build_legendre_coeffs, sh_accel_fixed

    n_sh = 8
    rng = np.random.default_rng(7)
    C = np.zeros((n_sh + 1, n_sh + 1))
    S = np.zeros((n_sh + 1, n_sh + 1))
    for n in range(2, n_sh + 1):
        for m in range(n + 1):
            C[n, m] = rng.normal() * 1e-5
            if m:
                S[n, m] = rng.normal() * 1e-5
    C[0, 0] = 1.0
    diag, sub, A, B, scl = build_legendre_coeffs(n_sh)

    @cuda.jit
    def launch(p, out):
        k = cuda.grid(1)
        if k < p.shape[0]:
            a = cuda.local.array(3, np.float64)
            _sh_accel_cuda(p[k, 0], p[k, 1], p[k, 2], n_sh, R_MOON, MU_MOON,
                           C, S, diag, sub, A, B, scl, a)
            out[k, 0] = a[0]
            out[k, 1] = a[1]
            out[k, 2] = a[2]

    z = R_MOON + 1.0e5
    pts = np.array([
        [0.0, 0.0, z], [0.0, 0.0, -z],          # on the axis
        [3e-13, -2e-13, z], [1e-13, 0.0, -z],  # inside the rho^2 < 1e-24 cutoff
        [1e-9, 2e-9, z], [-4e-7, 1e-7, -z],    # between the old 1e-6 m and new 1e-12 m cutoffs
        [1e-3, 0.0, z], [1.0e6, -3.0e5, 9.0e5], [-1.2e6, 1.1e6, -5.0e5],
    ])
    out = np.zeros_like(pts)
    launch[1, 32](np.ascontiguousarray(pts), out)
    P = np.zeros((n_sh + 1, n_sh + 1)); dP = np.zeros_like(P)
    cm = np.zeros(n_sh + 1); sm = np.zeros(n_sh + 1)
    worst = 0.0
    for k, p in enumerate(pts):
        ref = np.array(sh_accel_fixed(p[0], p[1], p[2], n_sh, R_MOON, MU_MOON, C, S,
                                      diag, sub, A, B, scl, P, dP, cm, sm))
        # Relative to the non-central part (the central term would hide errors).
        pert = ref - (-MU_MOON * p / np.linalg.norm(p) ** 3)
        err = np.max(np.abs(out[k] - ref)) / np.max(np.abs(pert))
        worst = max(worst, err)
        assert err < 1e-9, (k, p, out[k], ref, err)
    print("OK", worst)
    """
    _assert_ok(_run_cudasim(script))


def test_numba_interp4_holds_table_ends_like_cpu() -> None:
    script = """
    import math
    import numpy as np
    from numba import cuda

    from lunaris.core.batch_propagator import _interp4_cuda
    from lunaris.physics.ephemeris import interp_quat_safe

    dt = 600.0
    ang = 0.05 * np.arange(5)
    q = np.ascontiguousarray(np.column_stack(
        [np.cos(ang / 2), np.zeros(5), np.zeros(5), np.sin(ang / 2)]))

    @cuda.jit
    def launch(ts, out):
        k = cuda.grid(1)
        if k < ts.shape[0]:
            r = cuda.local.array(4, np.float64)
            _interp4_cuda(ts[k], dt, q, 5, r)
            for j in range(4):
                out[k, j] = r[j]

    ts = np.array([-900.0, 0.0, 1000.0, 2399.0, 2400.0, 2700.0, 6000.0, 1.0e5])
    out = np.zeros((ts.size, 4))
    launch[1, 32](ts, out)
    for k, t in enumerate(ts):
        ref = np.array(interp_quat_safe(float(t), dt, q))
        assert np.max(np.abs(out[k] - ref)) < 1e-14, (t, out[k], ref)
    # Past the end the rotation angle stays at the last row (no extrapolation).
    assert abs(2.0 * math.atan2(out[-1, 3], out[-1, 0]) - ang[-1]) < 1e-14
    print("OK")
    """
    _assert_ok(_run_cudasim(script))
