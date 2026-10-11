"""Terrain impact detection resolves relief at the loaded map's grid spacing.

The batch terrain check samples each step's cubic Hermite arc and takes the
first sample at or below the terrain. At a fixed 32 samples per step a low
lunar orbit at ``dt = 30 s`` is sampled every ``v dt / 32 = 1.6 km`` along
track, more than three pixels of an LDEM 64 ppd map (0.474 km at the 1737.4 km
datum), so a ridge one pixel wide could lie between two samples and be flown
through. Terrain steps that pass the envelope gate now use ``K = arc / (0.5
pixel)`` samples, rounded up to a multiple of 32 and capped at 512; the sphere
path keeps 32.

Scenario (Moon-centred, identity inertial->body-fixed rotation, point-mass
``MU_MOON``, times in seconds from the initial epoch): an equatorial band of a
64 ppd map, flat at the datum except one longitude column 300 m high (a
bilinear tent 0.95 km wide at its base, 0.47 km wide at 150 m). A circular
equatorial orbit 150 m above the datum is phased so the ridge sits midway
between two of the 32 samples of step 10. Reference: SciPy DOP853 at rtol 1e-12
with the CPU hybrid impact event and ``max_step = 0.02 s`` (34 m along track),
which resolves the ridge; without ``max_step`` SciPy misses it too.

Tolerance: the crossing lies on the ridge flank (slope 300 m / 474 m). The RK4
state error over 10 steps of 30 s and the Hermite error are millimetres, i.e.
microseconds of along-track time at 1.68 km/s; measured 1.5e-6 s on torch CPU,
CUDA and the Numba simulator. 1e-4 s (0.17 m along track) keeps a wide margin.
"""

from __future__ import annotations

import functools
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
PPD = 64
RES_DEG = 1.0 / PPD
RIDGE_H_M = 300.0
RIDGE_COL = 640
DT_S = 30.0
LOW_ALT_M = 150.0
RIDGE_STEP = 10
RIDGE_SUB = 10
DURATION_S = 360.0
EVENT_TOL_S = 1.0e-4


def ridge_payload() -> dict:
    """Equatorial 64 ppd band (latitude clamped), one longitude column 300 m high."""
    n_lines = 5
    lat_c = (2 - np.arange(n_lines)) * RES_DEG
    n_samples = 360 * PPD
    dn = np.zeros((n_lines, n_samples))
    dn[:, RIDGE_COL] = RIDGE_H_M
    return {
        "dn": dn, "n_lines": n_lines, "n_samples": n_samples, "res_deg": RES_DEG,
        "lon0_deg": 0.5 * RES_DEG, "lat0_deg": float(lat_c[0]), "scale_m": 1.0,
        "bias_m": DATUM_M, "missing_dn": float("nan"), "flip_lat": 0,
        "lat_min_deg": float(lat_c.min()), "lat_max_deg": float(lat_c.max()),
        "r_terrain_min_m": DATUM_M, "r_terrain_max_m": DATUM_M + RIDGE_H_M,
        "radius_const_m": DATUM_M,
    }


def _circular(alt_m: float, lon_deg: float) -> np.ndarray:
    r = DATUM_M + alt_m
    v = math.sqrt(MU_MOON / r)
    lo = math.radians(lon_deg)
    return np.array([r * math.cos(lo), r * math.sin(lo), 0.0, -v * math.sin(lo), v * math.cos(lo), 0.0])


def initial_states() -> np.ndarray:
    """Rows: 150 m (crosses the ridge flank), 400 m (gated, clears it), 20 km (ungated)."""
    r = DATUM_M + LOW_ALT_M
    omega = math.sqrt(MU_MOON / r) / r
    t_mid = (RIDGE_STEP + (RIDGE_SUB + 0.5) / 32.0) * DT_S
    lon0 = (RIDGE_COL + 0.5) * RES_DEG - math.degrees(omega * t_mid)
    return np.stack([_circular(LOW_ALT_M, lon0), _circular(400.0, lon0), _circular(20_000.0, lon0)])


def _rhs(_t: float, y: np.ndarray) -> np.ndarray:
    return np.concatenate([y[3:], -MU_MOON * y[:3] / np.linalg.norm(y[:3]) ** 3])


@functools.cache
def cpu_event_time(row: int, max_step: float = 0.02) -> float | None:
    """SciPy terrain event of ``initial_states()[row]`` up to the end of step ``RIDGE_STEP + 1``.

    ``max_step`` applies on the window ``[RIDGE_STEP - 1, RIDGE_STEP + 2) * DT_S``
    around the ridge passage; the flat datum 150 m below is the only other
    terrain, and it needs no step limit.
    """
    from scipy.integrate import solve_ivp

    from lunaris.core.events import make_hybrid_impact_event
    from lunaris.loaders.io_surface import sample_topo_radius_m

    payload = ridge_payload()

    class _Topo:
        def radius_m_deg(self, lat_deg: float, lon_deg: float) -> float:
            return sample_topo_radius_m(payload, lat_deg, lon_deg)

    event = make_hybrid_impact_event(
        R_ref_m=DATUM_M, impact_alt_m=0.0, topo=_Topo(),
        r_i_to_bf=lambda _t, r: np.asarray(r, dtype=np.float64),
        switch_alt_m=1.0e9, terminal=True,
    )
    y = initial_states()[row]
    t_window = (RIDGE_STEP - 1) * DT_S
    for t_span, step in (((0.0, t_window), math.inf), ((t_window, (RIDGE_STEP + 2) * DT_S), max_step)):
        sol = solve_ivp(
            _rhs, t_span, y, method="DOP853", rtol=1e-12, atol=1e-6,
            events=event, max_step=step,
        )
        if sol.t_events[0].size:
            return float(sol.t_events[0][0])
        y = sol.y[:, -1]
    return None


def terrain_residual_m(p: np.ndarray) -> float:
    from lunaris.loaders.io_surface import sample_topo_radius_m

    rn = float(np.linalg.norm(p))
    lat = math.degrees(math.asin(p[2] / rn))
    lon = math.degrees(math.atan2(p[1], p[0]))
    return rn - sample_topo_radius_m(ridge_payload(), lat, lon)


def test_cpu_reference_hits_the_ridge_only_with_a_small_max_step() -> None:
    """Guard the scenario: the ridge is real for the CPU event, and narrow."""
    t_ref = cpu_event_time(0)
    assert t_ref is not None
    assert RIDGE_STEP * DT_S < t_ref < (RIDGE_STEP + 1) * DT_S
    assert cpu_event_time(0, max_step=math.inf) is None
    assert cpu_event_time(1) is None
    assert cpu_event_time(2) is None


def test_substep_count_follows_arc_and_grid_spacing() -> None:
    torch = pytest.importorskip("torch")
    from lunaris.core.batch_propagator import (
        _IMPACT_TERRAIN_SPACING_FRACTION,
        _IMPACT_TERRAIN_SUBSTEPS_MAX,
    )
    from lunaris.core.torch_frame import (
        IMPACT_HERMITE_SUBSTEPS,
        IMPACT_TERRAIN_SPACING_FRACTION,
        IMPACT_TERRAIN_SUBSTEPS_MAX,
        terrain_substep_count,
    )

    assert _IMPACT_TERRAIN_SPACING_FRACTION == IMPACT_TERRAIN_SPACING_FRACTION
    assert _IMPACT_TERRAIN_SUBSTEPS_MAX == IMPACT_TERRAIN_SUBSTEPS_MAX
    v = torch.tensor([[0.0, 1680.0, 0.0], [0.0, 0.0, float("nan")]], dtype=torch.float64)
    pixel_m = DATUM_M * math.radians(RES_DEG)  # 473.8 m
    k = terrain_substep_count(v, v, DT_S, ridge_payload())
    k_need = DT_S * 1680.0 / (0.5 * pixel_m)  # 212.8 samples at half-pixel spacing
    assert k.tolist() == [32 * math.ceil(k_need / 32), IMPACT_HERMITE_SUBSTEPS]
    assert k[0].item() == 224  # 225 m spacing <= half a 473.8 m pixel
    assert terrain_substep_count(v, v, 120.0, ridge_payload())[0].item() == 512
    coarse = dict(ridge_payload(), res_deg=1.0)  # 30.3 km pixels: K stays 32
    assert terrain_substep_count(v, v, DT_S, coarse)[0].item() == IMPACT_HERMITE_SUBSTEPS


def test_fixed_32_samples_straddle_the_ridge() -> None:
    """The step that crosses the ridge: missed at K = 32, caught at the map-based K."""
    torch = pytest.importorskip("torch")
    from scipy.integrate import solve_ivp

    from lunaris.core.torch_frame import terrain_segment_intersection, topo_payload_to_torch

    y0 = initial_states()[0]
    t0, t1 = RIDGE_STEP * DT_S, (RIDGE_STEP + 1) * DT_S
    sol = solve_ivp(_rhs, (0.0, t1), y0, method="DOP853", rtol=1e-12, atol=1e-6, dense_output=True)
    prev = torch.tensor(sol.sol(t0)[None, :], dtype=torch.float64)
    curr = torch.tensor(sol.sol(t1)[None, :], dtype=torch.float64)
    topo = topo_payload_to_torch(ridge_payload(), device=torch.device("cpu"), dtype=torch.float64)
    kw = dict(
        t_prev_s=t0, dt_s=DT_S, frame=None, topo=topo, impact_alt_m=0.0,
        v_prev=prev[:, 3:], v_curr=curr[:, 3:],
    )
    hit32, _ = terrain_segment_intersection(prev[:, :3], curr[:, :3], n_sub_max=32, **kw)
    hit, alpha = terrain_segment_intersection(prev[:, :3], curr[:, :3], **kw)
    assert not bool(hit32[0])
    assert bool(hit[0])
    assert abs(t0 + float(alpha[0]) * DT_S - cpu_event_time(0)) < EVENT_TOL_S


def _run_torch(torch, device, *, detect_impact: bool):
    from lunaris.core.batched_fixed_step import run_batched_fixed_step
    from lunaris.core.torch_frame import TorchMoonFrame, topo_payload_to_torch

    class _PointMass:
        def acceleration(self, t_s, s):
            r = s[:, :3]
            return -MU_MOON * r / torch.linalg.norm(r, dim=1, keepdim=True) ** 3

    return run_batched_fixed_step(
        torch_mod=torch, device=device, dtype=torch.float64, provider=_PointMass(),
        frame=TorchMoonFrame(None, device=device, dtype=torch.float64, allow_identity=True),
        Y0=initial_states(), duration_s=DURATION_S, output_dt_s=DT_S, dt_s=DT_S,
        impact_r_m=float(R_MOON), detect_impact=detect_impact,
        topo=topo_payload_to_torch(ridge_payload(), device=device, dtype=torch.float64),
        impact_alt_m=0.0,
    )


@pytest.mark.requires_torch
@pytest.mark.parametrize("device_name", ["cpu", "cuda"])
def test_torch_loop_detects_the_narrow_ridge(device_name: str) -> None:
    torch = pytest.importorskip("torch")
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA device not available")
    device = torch.device(device_name)
    t_ref = cpu_event_time(0)
    res = _run_torch(torch, device, detect_impact=True)
    np.testing.assert_array_equal(res.impact_flags, [1.0, 0.0, 0.0])
    assert abs(float(res.t_impact[0]) - t_ref) < EVENT_TOL_S, (res.t_impact[0], t_ref)
    p = res.impact_positions_inertial[0]
    assert abs(terrain_residual_m(p)) < 1e-3
    np.testing.assert_allclose(res.Y_out[-1, 0, :3], p, rtol=0.0, atol=1e-9)
    # The gated 400 m orbit (K = 224 every step) and the ungated 20 km orbit
    # are integrated exactly as without impact detection.
    free = _run_torch(torch, device, detect_impact=False)
    np.testing.assert_array_equal(res.Y_out[:, 1:, :], free.Y_out[:, 1:, :])


def test_numba_kernel_detects_the_narrow_ridge_cudasim() -> None:
    root = Path(__file__).resolve().parents[1]
    script = textwrap.dedent(
        """
        import sys
        import numpy as np
        sys.path.insert(0, r"{tests_dir}")
        import test_terrain_ridge_impact_resolution as T
        from lunaris.common.constants import AU, MU_EARTH, MU_MOON, MU_SUN, P_SUN_1AU, R_EARTH_MEAN, R_MOON
        from lunaris.core.batch_propagator import _rk4_batch_kernel, topo_meta_array

        payload = T.ridge_payload()

        def run(detect, n_steps):
            Y = np.ascontiguousarray(T.initial_states())
            n = Y.shape[0]
            zeros3 = np.zeros((2, 3))
            q = np.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
            coeff = np.zeros((2, 2))
            vec = np.zeros(2)
            one = np.ones(n)
            flags = np.zeros(n, dtype=np.int32)
            times = np.full(n, np.nan)
            pos = np.full((n, 3), np.nan)
            meta = topo_meta_array(payload)
            dn = np.ascontiguousarray(payload["dn"])
            hist = [Y.copy()]
            for k in range(n_steps):
                _rk4_batch_kernel[1, 32](
                    Y, k * T.DT_S, T.DT_S,
                    1.0e6, zeros3, zeros3, zeros3, zeros3, q, 2, 0,
                    0, R_MOON, MU_MOON, coeff, coeff, vec, vec, coeff, coeff, vec,
                    0, 0, 0, 0, 0, 0,
                    one, one, one,
                    MU_SUN, MU_EARTH, R_EARTH_MEAN, 0.0, 0.0, 0.0, 1.0,
                    0.0, 0.0, AU, P_SUN_1AU,
                    flags, times, pos, detect, float(R_MOON),
                    dn, meta, 0.0, 1,
                )
                hist.append(Y.copy())
            return np.stack(hist), flags, times, pos

        n_steps = T.RIDGE_STEP + 2
        t_ref = T.cpu_event_time(0)
        hist, flags, times, pos = run(1, n_steps)
        assert flags.tolist() == [1, 0, 0], flags
        assert abs(times[0] - t_ref) < T.EVENT_TOL_S, (times[0], t_ref)
        assert abs(T.terrain_residual_m(pos[0])) < 1e-3
        free, _, _, _ = run(0, n_steps)
        np.testing.assert_array_equal(hist[:, 1:, :], free[:, 1:, :])
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
