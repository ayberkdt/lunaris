"""Batch impact detection follows the step's cubic Hermite arc, not its chord.

The torch loop (``batched_fixed_step``) and the Numba CUDA kernel used to
intersect the straight chord between two RK4 step ends with the impact sphere
(or terrain). The chord sits radially below the true arc by up to
``g dt^2 / 8`` -- 183 m for a low lunar orbit at ``dt = 30 s`` (``g = MU_MOON /
R^2 = 1.62 m/s^2``) -- so circular orbits 5-150 m above the sphere were flagged
as impacts, and a real crossing was placed early by ``sag / v_radial`` (11.5 s
for a 2 km-deep shallow descent). The kernels now decide on the cubic Hermite
curve through both step-end positions and velocities: it matches the arc to
``dt^4 |d4r/dt4| / 384`` (about 3 mm here) and is sampled at 32 sub-steps
(residual sub-chord sag ``g (dt/32)^2 / 8 = 0.18 m``).

References (CPU, follow the true arc): SciPy ``solve_ivp`` DOP853 at
rtol 1e-12 with the CPU impact events of :mod:`lunaris.core.events`.

Geometry: point-mass Moon (``MU_MOON``), inertial frame with identity
inertial->body-fixed rotation, impact sphere ``R_MOON`` (1738 km), times in
seconds from the initial epoch.
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from lunaris.common.constants import MU_MOON, R_MOON

R_IMP = float(R_MOON)
DT_S = 30.0
CLOSE_ALTITUDES_M = (5.0, 50.0, 100.0, 150.0)
# The grazing pass: ellipse with periapsis 20 m under the sphere and apoapsis
# 2000 km above it, started 15 s before periapsis. Both ends of the first
# 30 s step are 46.7 m above the sphere; the arc is inside on [6.786, 23.214] s.
GRAZE_DEPTH_M = 20.0
GRAZE_APO_ALT_M = 2.0e6
GRAZE_T_PERI_S = 15.0
# Impact-time tolerance vs the SciPy event. The batch states carry the RK4
# truncation error of a 30 s step (sub-millimetre to centimetre over these
# arcs) and the Hermite interpolation error (~3 mm); at the grazing pass'
# 13 m/s radial speed a 1 cm radial error is 0.8 ms. 5 ms covers both with
# margin, while the chord placed this crossing 6.8 s early.
EVENT_TOL_S = 5.0e-3


def _rhs(_t: float, y: np.ndarray) -> np.ndarray:
    r = y[:3]
    return np.concatenate([y[3:], -MU_MOON * r / np.linalg.norm(r) ** 3])


def circular_state(alt_m: float, radius_m: float = R_IMP) -> np.ndarray:
    r = radius_m + alt_m
    return np.array([r, 0.0, 0.0, 0.0, math.sqrt(MU_MOON / r), 0.0])


def grazing_state(radius_m: float = R_IMP) -> np.ndarray:
    """State 15 s before the periapsis of a pass that dips 20 m inside ``radius_m``."""
    from scipy.integrate import solve_ivp

    rp = radius_m - GRAZE_DEPTH_M
    a = 0.5 * (rp + radius_m + GRAZE_APO_ALT_M)
    vp = math.sqrt(MU_MOON * (2.0 / rp - 1.0 / a))
    back = solve_ivp(
        _rhs, (0.0, -GRAZE_T_PERI_S), np.array([rp, 0.0, 0.0, 0.0, vp, 0.0]),
        method="DOP853", rtol=1e-13, atol=1e-7,
    )
    return back.y[:, -1].copy()


def sphere_event_time(y0: np.ndarray, duration_s: float, radius_m: float = R_IMP) -> float | None:
    from scipy.integrate import solve_ivp

    from lunaris.core.events import make_impact_event

    # SciPy only tests the event sign at its accepted steps, so a dip shorter
    # than a solver step goes unseen; max_step = 1 s resolves the 16 s dip.
    sol = solve_ivp(
        _rhs, (0.0, duration_s), y0, method="DOP853", rtol=1e-12, atol=1e-6,
        events=make_impact_event(radius_m, 0.0, terminal=True), max_step=1.0,
    )
    return float(sol.t_events[0][0]) if sol.t_events[0].size else None


def test_grazing_scenario_dips_inside_one_step_with_both_ends_outside() -> None:
    """Guard the scenario: step ends above the sphere, arc inside in between."""
    from scipy.integrate import solve_ivp

    sol = solve_ivp(
        _rhs, (0.0, DT_S), grazing_state(), method="DOP853", rtol=1e-12, atol=1e-6,
        dense_output=True,
    )
    r = np.linalg.norm(sol.sol(np.linspace(0.0, DT_S, 3001))[:3], axis=0)
    assert r[0] - R_IMP > 40.0 and r[-1] - R_IMP > 40.0
    assert r.min() - R_IMP < -15.0
    t_ev = sphere_event_time(grazing_state(), DT_S)
    assert t_ev is not None and 6.0 < t_ev < 8.0


# --------------------------------------------------------------------------
# torch helpers
# --------------------------------------------------------------------------


def test_hermite_state_reproduces_a_cubic_trajectory_exactly() -> None:
    """Independent check of the basis: a cubic p(t) is reproduced to round-off."""
    torch = pytest.importorskip("torch")
    from lunaris.core.torch_frame import hermite_state

    rng = np.random.default_rng(7)
    c = rng.normal(0.0, 1.0, size=(4, 3)) * np.array([[1.7e6], [1.6e3], [1.0], [1e-3]])
    dt = 30.0

    def p(t):
        return c[0] + c[1] * t + c[2] * t**2 + c[3] * t**3

    def v(t):
        return c[1] + 2.0 * c[2] * t + 3.0 * c[3] * t**2

    prev = torch.tensor(np.concatenate([p(0.0), v(0.0)])[None, :], dtype=torch.float64)
    curr = torch.tensor(np.concatenate([p(dt), v(dt)])[None, :], dtype=torch.float64)
    for a in (0.0, 0.137, 0.5, 0.91, 1.0):
        got = hermite_state(prev, curr, dt, torch.tensor([a], dtype=torch.float64))[0].numpy()
        np.testing.assert_allclose(got[:3], p(a * dt), rtol=0.0, atol=1e-6)
        np.testing.assert_allclose(got[3:], v(a * dt), rtol=0.0, atol=1e-9)


def test_kernel_substep_count_matches_torch() -> None:
    from lunaris.core.batch_propagator import _IMPACT_HERMITE_SUBSTEPS
    from lunaris.core.torch_frame import IMPACT_HERMITE_SUBSTEPS

    assert _IMPACT_HERMITE_SUBSTEPS == IMPACT_HERMITE_SUBSTEPS


def _uniform_terrain_payload(radius_m: float) -> dict:
    """Global 1-degree grid of constant radius (terrain == sphere)."""
    lat_c = np.arange(89.5, -90.0, -1.0)
    lon_c = np.arange(0.5, 360.0, 1.0)
    return {
        "dn": np.zeros((lat_c.size, lon_c.size)), "n_lines": int(lat_c.size),
        "n_samples": int(lon_c.size), "res_deg": 1.0, "lon0_deg": 0.5, "lat0_deg": 89.5,
        "scale_m": 1.0, "bias_m": radius_m, "missing_dn": float("nan"), "flip_lat": 0,
        "lat_min_deg": -89.5, "lat_max_deg": 89.5, "r_terrain_min_m": radius_m,
        "r_terrain_max_m": radius_m, "radius_const_m": radius_m,
    }


def _run_torch(torch, device, Y0: np.ndarray, duration_s: float, *, topo_payload=None,
               detect_impact: bool = True):
    from lunaris.core.batched_fixed_step import run_batched_fixed_step
    from lunaris.core.torch_frame import TorchMoonFrame, topo_payload_to_torch

    class _PointMass:
        def acceleration(self, t_s, s):
            r = s[:, :3]
            return -MU_MOON * r / torch.linalg.norm(r, dim=1, keepdim=True) ** 3

    topo = None
    if topo_payload is not None:
        topo = topo_payload_to_torch(topo_payload, device=device, dtype=torch.float64)
    return run_batched_fixed_step(
        torch_mod=torch, device=device, dtype=torch.float64, provider=_PointMass(),
        frame=TorchMoonFrame(None, device=device, dtype=torch.float64, allow_identity=True),
        Y0=Y0, duration_s=duration_s, output_dt_s=600.0 if duration_s > 600.0 else duration_s,
        dt_s=DT_S, impact_r_m=R_IMP, detect_impact=detect_impact, topo=topo, impact_alt_m=0.0,
    )


def _torch_device(device_name: str):
    torch = pytest.importorskip("torch")
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA device not available")
    return torch, torch.device(device_name)


@pytest.mark.requires_torch
@pytest.mark.parametrize("device_name", ["cpu", "cuda"])
@pytest.mark.parametrize("terrain", [False, True], ids=["sphere", "terrain"])
def test_torch_orbit_close_above_surface_is_not_an_impact(device_name: str, terrain: bool) -> None:
    """5-150 m above the surface for a full orbit: no impact, trajectories untouched."""
    torch, device = _torch_device(device_name)
    Y0 = np.stack([circular_state(h) for h in CLOSE_ALTITUDES_M])
    payload = _uniform_terrain_payload(R_IMP) if terrain else None
    res = _run_torch(torch, device, Y0, 7200.0, topo_payload=payload)
    np.testing.assert_array_equal(res.impact_flags, np.zeros(len(CLOSE_ALTITUDES_M)))
    assert np.isnan(res.t_impact).all()
    # Impact detection never alters the integration of a non-impacting sample.
    free = _run_torch(torch, device, Y0, 7200.0, detect_impact=False)
    np.testing.assert_array_equal(res.Y_out, free.Y_out)


@pytest.mark.requires_torch
@pytest.mark.parametrize("device_name", ["cpu", "cuda"])
@pytest.mark.parametrize("terrain", [False, True], ids=["sphere", "terrain"])
def test_torch_grazing_dip_inside_one_step_is_detected_at_the_true_time(
    device_name: str, terrain: bool
) -> None:
    """Both step ends outside, arc inside: hit at the SciPy event time.

    The terrain path used to require the step END below the terrain and missed
    this impact entirely; the sphere chord placed it 6.8 s early.
    """
    torch, device = _torch_device(device_name)
    y0 = grazing_state()
    t_ref = sphere_event_time(y0, 2.0 * DT_S)
    payload = _uniform_terrain_payload(R_IMP) if terrain else None
    res = _run_torch(torch, device, y0[None, :], 2.0 * DT_S, topo_payload=payload)
    assert res.impact_flags[0] == 1.0
    assert abs(float(res.t_impact[0]) - t_ref) < EVENT_TOL_S, (res.t_impact[0], t_ref)
    pos = res.impact_positions_inertial[0]
    assert abs(float(np.linalg.norm(pos)) - R_IMP) < 1e-3
    np.testing.assert_allclose(res.Y_out[-1, 0, :3], pos, rtol=0.0, atol=1e-9)


# --------------------------------------------------------------------------
# Numba CUDA kernel under the simulator
# --------------------------------------------------------------------------


def test_numba_kernel_hermite_impact_detection_cudasim() -> None:
    root = Path(__file__).resolve().parents[1]
    script = textwrap.dedent(
        """
        import sys
        import numpy as np
        sys.path.insert(0, r"{tests_dir}")
        import test_hermite_impact_detection as T
        from lunaris.common.constants import AU, MU_EARTH, MU_MOON, MU_SUN, P_SUN_1AU, R_EARTH_MEAN, R_MOON
        from lunaris.core.batch_propagator import _rk4_batch_kernel, topo_meta_array

        def run(Y, n_steps, payload):
            n = Y.shape[0]
            zeros3 = np.zeros((2, 3))
            q = np.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
            coeff = np.zeros((2, 2))
            vec = np.zeros(2)
            one = np.ones(n)
            flags = np.zeros(n, dtype=np.int32)
            times = np.full(n, np.nan)
            pos = np.full((n, 3), np.nan)
            if payload is None:
                dn, meta, use_t = np.zeros((1, 1)), topo_meta_array(None), 0
            else:
                dn, meta, use_t = np.ascontiguousarray(payload["dn"]), topo_meta_array(payload), 1
            for k in range(n_steps):
                _rk4_batch_kernel[1, 32](
                    Y, k * T.DT_S, T.DT_S,
                    1.0e6, zeros3, zeros3, zeros3, zeros3, q, 2, 0,
                    0, R_MOON, MU_MOON, coeff, coeff, vec, vec, coeff, coeff, vec,
                    0, 0, 0, 0, 0, 0,
                    one, one, one,
                    MU_SUN, MU_EARTH, R_EARTH_MEAN, 0.0, 0.0, 0.0, 1.0,
                    0.0, 0.0, AU, P_SUN_1AU,
                    flags, times, pos, 1, T.R_IMP,
                    dn, meta, 0.0, use_t,
                )
            return flags, times, pos

        y_graze = T.grazing_state()
        t_ref = T.sphere_event_time(y_graze, 2.0 * T.DT_S)
        for payload in (None, T._uniform_terrain_payload(T.R_IMP)):
            close = np.stack([T.circular_state(h) for h in T.CLOSE_ALTITUDES_M])
            Y = np.ascontiguousarray(np.vstack([close, y_graze[None, :]]))
            # 40 steps = 1200 s, a sixth of an orbit: every step of the close
            # orbits passes the chord gate and is decided on the Hermite arc.
            flags, times, pos = run(Y, 40, payload)
            assert flags.tolist() == [0, 0, 0, 0, 1], flags
            assert abs(times[4] - t_ref) < T.EVENT_TOL_S, (times[4], t_ref)
            assert abs(np.linalg.norm(pos[4]) - T.R_IMP) < 1e-3
            np.testing.assert_allclose(Y[4, :3], pos[4], rtol=0.0, atol=1e-9)
        print("OK")
        """
    ).format(tests_dir=str(Path(__file__).resolve().parent))
    env = os.environ.copy()
    env["NUMBA_ENABLE_CUDASIM"] = "1"
    env["PYTHONPATH"] = str(root / "src") + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=root, env=env,
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout, result.stdout + result.stderr
