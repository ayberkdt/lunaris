"""Terrain impact for a descent that spends several steps inside the envelope.

The batch terrain check first brackets each step against the outer envelope
sphere ``r_terrain_max + impact_alt``. A trajectory that crosses that sphere in
one step and reaches the ground several steps later starts every later step
*inside* the envelope; its entering root is then negative and the step used to
be skipped, so the impact was never recorded and the sample flew through the
terrain. These tests fly such a descent on the torch loop (CPU, and CUDA when
present) and on the Numba CUDA kernel (under the simulator) and compare the
impact time with a CPU reference: SciPy ``solve_ivp`` with the hybrid impact
event of :func:`lunaris.core.events.make_hybrid_impact_event` sampling the same
grid through :func:`lunaris.loaders.io_surface.sample_topo_radius_m`.

Geometry (Moon-centred, identity inertial->body-fixed rotation, point-mass
gravity ``MU_MOON``): start at lat 5 deg, lon 10 deg, 12 km above the 1737.4 km
datum, falling at 150 m/s with 200 m/s eastward drift. The envelope is 8 km
above the datum (a mountain far away), the local ground about 1.5 km below it,
so the fall spends ~6 steps of 10 s inside the envelope before the ground.

Tolerances: the batch kernels place the crossing on the straight chord of the
impact step, by design. The chord sags inside the arc by up to
``g dt^2 / 8 = 20.3 m`` (g = 1.62 m/s^2, dt = 10 s), which at the ~255 m/s
descent rate near the ground moves the crossing up to ~0.08 s earlier than the
true event. So the kernels are compared with the *chord* crossing of the SciPy
trajectory (its states at the bounding step epochs, root of the same terrain
residual on the chord) to 1e-3 s -- the RK4 state error over these few steps is
millimetres and both bisections resolve alpha far below that -- and that chord
crossing is checked against the true event time with the 0.1 s sag bound.
Before the fix both kernels reported no impact at all.
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

DATUM_M = 1_737_400.0
RES_DEG = 1.0
DT_S = 10.0
DURATION_S = 200.0
OUTPUT_DT_S = 50.0
CHORD_TOL_S = 1.0e-3
SAG_TOL_S = 0.1


def synthetic_payload() -> dict:
    """Global 1-degree grid: a gentle eastward slope plus one distant mountain."""
    lat_centers = np.arange(90.0 - 0.5 * RES_DEG, -90.0, -RES_DEG)
    lon_centers = np.arange(0.5 * RES_DEG, 360.0, RES_DEG)
    lat_g, lon_g = np.meshgrid(lat_centers, lon_centers, indexing="ij")
    height = -2000.0 + 50.0 * np.clip(lon_g, 0.0, 40.0)  # -2000 m .. 0 m over lon 0..40
    mountain = (np.abs(lat_g - 61.0) < 1.6) & (np.abs(lon_g - 201.0) < 1.6)
    height = np.where(mountain, 8000.0, height)
    dn = np.ascontiguousarray(height, dtype=np.float64)  # scale 1 m per DN
    r = dn + DATUM_M
    return {
        "dn": dn,
        "n_lines": int(lat_centers.size),
        "n_samples": int(lon_centers.size),
        "res_deg": RES_DEG,
        "lon0_deg": float(lon_centers[0]),
        "lat0_deg": float(lat_centers[0]),
        "scale_m": 1.0,
        "bias_m": DATUM_M,
        "missing_dn": float("nan"),
        "flip_lat": 0,
        "lat_min_deg": float(lat_centers.min()),
        "lat_max_deg": float(lat_centers.max()),
        "r_terrain_min_m": float(r.min()),
        "r_terrain_max_m": float(r.max()),
        "radius_const_m": DATUM_M,
    }


def initial_state() -> np.ndarray:
    lat = math.radians(5.0)
    lon = math.radians(10.0)
    up = np.array([math.cos(lat) * math.cos(lon), math.cos(lat) * math.sin(lon), math.sin(lat)])
    east = np.array([-math.sin(lon), math.cos(lon), 0.0])
    r0 = (DATUM_M + 12_000.0) * up
    v0 = -150.0 * up + 200.0 * east
    return np.concatenate([r0, v0]).reshape(1, 6)


def _point_mass_rhs(_t: float, y: np.ndarray) -> np.ndarray:
    r = y[:3]
    return np.concatenate([y[3:], -MU_MOON * r / np.linalg.norm(r) ** 3])


def _terrain_residual_m(r: np.ndarray) -> float:
    from lunaris.loaders.io_surface import sample_topo_radius_m

    rn = float(np.linalg.norm(r))
    lat = math.degrees(math.asin(r[2] / rn))
    lon = math.degrees(math.atan2(r[1], r[0]))
    return rn - sample_topo_radius_m(synthetic_payload(), lat, lon)


def chord_reference_impact_time(t_true: float) -> float:
    """Crossing of the terrain on the chord of the step that contains ``t_true``."""
    from scipy.integrate import solve_ivp
    from scipy.optimize import brentq

    k = math.floor(t_true / DT_S)
    sol = solve_ivp(
        _point_mass_rhs, (0.0, (k + 1) * DT_S), initial_state()[0], method="DOP853",
        rtol=1e-12, atol=1e-6, dense_output=True,
    )
    p0 = sol.sol(k * DT_S)[:3]
    p1 = sol.sol((k + 1) * DT_S)[:3]
    assert _terrain_residual_m(p0) > 0.0 > _terrain_residual_m(p1)
    alpha = brentq(lambda a: _terrain_residual_m(p0 + a * (p1 - p0)), 0.0, 1.0, xtol=1e-14)
    return (k + alpha) * DT_S


def cpu_reference_impact_time() -> float:
    """SciPy reference: DOP853 at rtol 1e-12 with the CPU hybrid impact event."""
    from scipy.integrate import solve_ivp

    from lunaris.core.events import make_hybrid_impact_event
    from lunaris.loaders.io_surface import sample_topo_radius_m

    payload = synthetic_payload()

    class _Topo:
        def radius_m_deg(self, lat_deg: float, lon_deg: float) -> float:
            return sample_topo_radius_m(payload, lat_deg, lon_deg)

    event = make_hybrid_impact_event(
        R_ref_m=DATUM_M,
        impact_alt_m=0.0,
        topo=_Topo(),
        r_i_to_bf=lambda _t, r: np.asarray(r, dtype=np.float64),
        switch_alt_m=1.0e9,
        terminal=True,
    )
    sol = solve_ivp(
        _point_mass_rhs, (0.0, DURATION_S), initial_state()[0], method="DOP853",
        rtol=1e-12, atol=1e-6, events=event,
    )
    assert sol.t_events[0].size == 1
    return float(sol.t_events[0][0])


def assert_batch_impact_time(t_batch: float) -> None:
    """Batch crossing == chord crossing (1e-3 s); chord within the sag bound of the event."""
    t_true = cpu_reference_impact_time()
    t_chord = chord_reference_impact_time(t_true)
    assert abs(t_chord - t_true) < SAG_TOL_S, (t_chord, t_true)
    assert abs(t_batch - t_chord) < CHORD_TOL_S, (t_batch, t_chord, t_true)


def test_descent_really_spends_several_steps_inside_the_envelope() -> None:
    """Guard the scenario itself: >= 3 steps start inside the envelope."""
    from scipy.integrate import solve_ivp

    t_ref = cpu_reference_impact_time()
    sol = solve_ivp(
        _point_mass_rhs, (0.0, t_ref), initial_state()[0], rtol=1e-12, atol=1e-6,
        dense_output=True,
    )
    ts = np.linspace(0.0, t_ref, 20001)
    radii = np.linalg.norm(sol.sol(ts)[:3], axis=0)
    t_env = float(ts[np.argmax(radii <= synthetic_payload()["r_terrain_max_m"])])
    steps_inside = math.floor(t_ref / DT_S) - math.ceil(t_env / DT_S) + 1
    assert steps_inside >= 3, (t_env, t_ref)


class _PointMass:
    def __init__(self, torch_mod):
        self._torch = torch_mod

    def acceleration(self, t_s, state):
        r = state[:, :3]
        rn = self._torch.linalg.norm(r, dim=1, keepdim=True)
        return -MU_MOON * r / rn**3


@pytest.mark.requires_torch
@pytest.mark.parametrize("device_name", ["cpu", "cuda"])
def test_torch_loop_detects_multistep_terrain_descent(device_name: str) -> None:
    torch = pytest.importorskip("torch")
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA device not available")
    from lunaris.core.batched_fixed_step import run_batched_fixed_step
    from lunaris.core.torch_frame import TorchMoonFrame, topo_payload_to_torch

    device = torch.device(device_name)
    dtype = torch.float64
    frame = TorchMoonFrame(None, device=device, dtype=dtype, allow_identity=True)
    topo = topo_payload_to_torch(synthetic_payload(), device=device, dtype=dtype)
    res = run_batched_fixed_step(
        torch_mod=torch, device=device, dtype=dtype, provider=_PointMass(torch),
        frame=frame, Y0=initial_state(), duration_s=DURATION_S,
        output_dt_s=OUTPUT_DT_S, dt_s=DT_S, impact_r_m=float(R_MOON),
        detect_impact=True, topo=topo, impact_alt_m=0.0,
    )
    assert res.impact_flags[0] == 1.0
    assert_batch_impact_time(float(res.t_impact[0]))
    # Frozen on the terrain surface, not left underground at a step end.
    p = res.impact_positions_inertial[0]
    assert abs(_terrain_residual_m(p)) < 1e-3
    np.testing.assert_allclose(res.Y_out[-1, 0, :3], p, rtol=0.0, atol=1e-9)


def test_numba_kernel_detects_multistep_terrain_descent_cudasim() -> None:
    root = Path(__file__).resolve().parents[1]
    script = textwrap.dedent(
        """
        import sys
        import numpy as np
        sys.path.insert(0, r"{tests_dir}")
        import test_terrain_multistep_descent as T
        from lunaris.common.constants import AU, MU_EARTH, MU_MOON, MU_SUN, P_SUN_1AU, R_EARTH_MEAN, R_MOON
        from lunaris.core.batch_propagator import _rk4_batch_kernel, topo_meta_array

        payload = T.synthetic_payload()
        Y = np.ascontiguousarray(T.initial_state())
        zeros3 = np.zeros((2, 3))
        q = np.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
        coeff = np.zeros((2, 2))
        vec = np.zeros(2)
        one = np.ones(1)
        flags = np.zeros(1, dtype=np.int32)
        times = np.full(1, np.nan)
        pos = np.full((1, 3), np.nan)
        meta = topo_meta_array(payload)
        t = 0.0
        while t < T.DURATION_S - 1e-9:
            _rk4_batch_kernel[1, 32](
                Y, t, T.DT_S,
                1.0e6, zeros3, zeros3, zeros3, zeros3, q, 2, 0,
                0, R_MOON, MU_MOON, coeff, coeff, vec, vec, coeff, coeff, vec,
                0, 0, 0, 0, 0, 0,
                one, one, one,
                MU_SUN, MU_EARTH, R_EARTH_MEAN, 0.0, 0.0, 0.0, 1.0,
                0.0, 0.0, AU, P_SUN_1AU,
                flags, times, pos, 1, float(R_MOON),
                np.ascontiguousarray(payload["dn"]), meta, 0.0, 1,
            )
            t += T.DT_S
        assert flags[0] == 1, ("no impact", flags, times)
        T.assert_batch_impact_time(float(times[0]))
        assert abs(T._terrain_residual_m(pos[0])) < 1e-3
        print("OK", times[0])
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
