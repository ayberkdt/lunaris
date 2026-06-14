"""
Torch classic-SH Monte Carlo propagator: scientific validation (task §16).

CPU-only with torch (deterministic, no CUDA required); a single CUDA-vs-CPU
consistency test is gated on ``torch.cuda.is_available()``.

Coverage
--------
A. Acceleration validation — TorchSHGravityEvaluator vs the canonical CPU
   ``GravityModel.accel_fixed`` reference at equatorial / near-pole / generic
   points and 50/100/500/2000 km altitudes, for degrees 20/25/50/100/200,
   with strict float64 and looser float32 tolerances (measured, not invented).
B. Single-trajectory propagation — Torch RK4 vs an independent numpy fixed-step
   RK4 reference (same dt / span / degree / identity frame model).
C. Batch consistency — an N-sample batch equals the same trajectories run alone.
D. Chunk consistency — chunk_size in {N, 1, intermediate} produce identical output.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from lunaris.common.type_defs import PerturbationFlags  # noqa: E402
from lunaris.core.torch_sh_propagator import (  # noqa: E402
    TorchSHBatchPropagator,
    TorchSHPreflightError,
)
from lunaris.physics.spherical_harmonics import GravityModel  # noqa: E402
from lunaris.physics.torch_spherical_harmonics import TorchSHGravityEvaluator  # noqa: E402

R_REF = 1.738e6
GM = 4.9028e12
MAX_DEGREE = 200


def _make_gravity_model(degree: int, seed: int = 1) -> GravityModel:
    """Synthetic normalized lunar-scale SH model with non-zero terms n>=2."""
    rng = np.random.default_rng(seed)
    n = int(degree)
    C = np.zeros((n + 1, n + 1), dtype=np.float64)
    S = np.zeros((n + 1, n + 1), dtype=np.float64)
    for nn in range(2, n + 1):
        for mm in range(0, nn + 1):
            C[nn, mm] = 1e-4 * rng.standard_normal() / (nn * nn)
            if mm > 0:
                S[nn, mm] = 1e-4 * rng.standard_normal() / (nn * nn)
    return GravityModel.from_arrays(
        degree_max=n, r_ref=R_REF, mu=GM, c_coeffs_full=C, s_coeffs_full=S
    )


def _pos(lat_deg: float, lon_deg: float, alt_km: float) -> list[float]:
    r = R_REF + alt_km * 1_000.0
    la = math.radians(lat_deg)
    lo = math.radians(lon_deg)
    return [r * math.cos(la) * math.cos(lo), r * math.cos(la) * math.sin(lo), r * math.sin(la)]


_VALIDATION_POINTS = [
    _pos(0.0, 0.0, 50.0),       # equatorial, 50 km
    _pos(0.0, 45.0, 100.0),     # equatorial, 100 km
    _pos(89.5, 10.0, 100.0),    # near-pole, 100 km
    _pos(37.0, 120.0, 500.0),   # generic lat/lon, 500 km
    _pos(-60.0, 200.0, 2000.0), # generic, 2000 km
]


# ---------------------------------------------------------------------------
# A. Acceleration validation vs the canonical CPU spherical-harmonic evaluator
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def grav_full() -> GravityModel:
    return _make_gravity_model(MAX_DEGREE, seed=1)


@pytest.mark.parametrize("degree", [20, 25, 50, 100, 200])
def test_acceleration_matches_cpu_reference_float64(grav_full: GravityModel, degree: int) -> None:
    if degree > grav_full.degree_max:
        pytest.skip(f"coefficient model only supports degree {grav_full.degree_max}")
    pts = np.array(_VALIDATION_POINTS, dtype=np.float64)
    a_cpu = np.array([grav_full.accel_fixed(p, degree=degree) for p in _VALIDATION_POINTS])
    ev = TorchSHGravityEvaluator(grav_full, degree=degree, device=torch.device("cpu"), dtype=torch.float64)
    a_torch = ev.acceleration(torch.as_tensor(pts, dtype=torch.float64)).numpy().astype(np.float64)
    # Measured agreement is ~1e-15 relative; keep a strict bound with margin.
    np.testing.assert_allclose(a_torch, a_cpu, rtol=1e-9, atol=1e-11)


@pytest.mark.parametrize("degree", [20, 50, 100])
def test_acceleration_matches_cpu_reference_float32(grav_full: GravityModel, degree: int) -> None:
    pts = np.array(_VALIDATION_POINTS, dtype=np.float64)
    a_cpu = np.array([grav_full.accel_fixed(p, degree=degree) for p in _VALIDATION_POINTS])
    ev = TorchSHGravityEvaluator(grav_full, degree=degree, device=torch.device("cpu"), dtype=torch.float32)
    a_torch = ev.acceleration(torch.as_tensor(pts, dtype=torch.float32)).numpy().astype(np.float64)
    # Measured float32 agreement is ~1e-6 relative; a separate, looser bound.
    np.testing.assert_allclose(a_torch, a_cpu, rtol=1e-4, atol=1e-4)


# ---------------------------------------------------------------------------
# Shared propagator harness (identity frame so the reference is unambiguous)
# ---------------------------------------------------------------------------

def _mc_cfg(degree: int, *, chunk_size: int = 0, dtype: str = "float64"):
    from types import SimpleNamespace

    return SimpleNamespace(
        gpu_sh_degree=int(degree),
        dt_s=60.0,
        impact_alt_km=0.0,
        torch_dtype=dtype,
        torch_sh_chunk_size=int(chunk_size),
        gpu_device_id=0,
    )


def _make_propagator(grav: GravityModel, degree: int, *, device="cpu", dtype=torch.float64, chunk_size=0):
    from types import SimpleNamespace

    dyn = SimpleNamespace(grav=grav, ephem=None)
    flags = PerturbationFlags(enable_sh=True)
    return TorchSHBatchPropagator(
        dyn, _mc_cfg(degree, chunk_size=chunk_size),
        flags, device=device, dtype=dtype, chunk_size=chunk_size,
    )


def _circular_state(alt_km: float) -> np.ndarray:
    r = R_REF + alt_km * 1_000.0
    v = math.sqrt(GM / r)
    return np.array([r, 0.0, 0.0, 0.0, v, 0.0], dtype=np.float64)


def _reference_rk4(grav: GravityModel, y0: np.ndarray, degree: int, dt: float, n_steps: int) -> np.ndarray:
    """Independent numpy fixed-step RK4 (identity frame) using the CPU SH kernel."""
    def rhs(y: np.ndarray) -> np.ndarray:
        a = grav.accel_fixed(y[:3], degree=degree)
        return np.concatenate([y[3:], a])

    y = y0.astype(np.float64).copy()
    for _ in range(n_steps):
        k1 = rhs(y)
        k2 = rhs(y + 0.5 * dt * k1)
        k3 = rhs(y + 0.5 * dt * k2)
        k4 = rhs(y + dt * k3)
        y = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return y


# ---------------------------------------------------------------------------
# B. Single-trajectory propagation vs the numpy fixed-step RK4 reference
# ---------------------------------------------------------------------------

def test_single_trajectory_matches_reference_rk4() -> None:
    degree = 50
    grav = _make_gravity_model(degree, seed=3)
    prop = _make_propagator(grav, degree)

    y0 = _circular_state(100.0)
    Y0 = y0[None, :].copy()
    duration, out_dt, dt = 1_200.0, 300.0, 60.0
    t_out, Y_out, impact, t_imp = prop.propagate(
        Y0, np.ones(1), np.ones(1), np.ones(1), np.ones(1),
        duration_s=duration, output_dt_s=out_dt,
    )

    n_steps = int(round(duration / dt))
    y_ref = _reference_rk4(grav, y0, degree, dt, n_steps)
    y_torch_final = Y_out[-1, 0, :]

    # Same dt, span, degree and (identity) frame model: difference is pure
    # float rounding-order, not an SH or integrator-formulation discrepancy.
    np.testing.assert_allclose(y_torch_final, y_ref, rtol=1e-8, atol=1e-3)
    assert impact.shape == (1,) and t_imp.shape == (1,)
    assert impact[0] == 0.0  # 100 km circular orbit does not impact in 20 minutes


# ---------------------------------------------------------------------------
# C. Batch consistency — N-sample batch == per-sample runs
# ---------------------------------------------------------------------------

def test_batch_matches_individual_runs() -> None:
    degree = 40
    grav = _make_gravity_model(degree, seed=5)
    prop = _make_propagator(grav, degree)

    rng = np.random.default_rng(0)
    base = _circular_state(120.0)
    Y0 = np.repeat(base[None, :], 6, axis=0)
    Y0[:, :3] += rng.normal(0.0, 2_000.0, size=(6, 3))   # ~2 km position spread
    Y0[:, 3:] += rng.normal(0.0, 1.0, size=(6, 3))       # ~1 m/s velocity spread

    args = dict(duration_s=900.0, output_dt_s=300.0)
    _, Y_batch, _, _ = prop.propagate(Y0, *([np.ones(6)] * 4), **args)

    for i in range(6):
        _, Y_i, _, _ = prop.propagate(Y0[i:i + 1], *([np.ones(1)] * 4), **args)
        np.testing.assert_allclose(Y_batch[:, i, :], Y_i[:, 0, :], rtol=0.0, atol=0.0)


# ---------------------------------------------------------------------------
# D. Chunk consistency — chunk_size must not change the numbers
# ---------------------------------------------------------------------------

def test_chunk_size_is_result_invariant() -> None:
    degree = 40
    grav = _make_gravity_model(degree, seed=7)
    N = 7

    rng = np.random.default_rng(1)
    base = _circular_state(150.0)
    Y0 = np.repeat(base[None, :], N, axis=0)
    Y0[:, :3] += rng.normal(0.0, 3_000.0, size=(N, 3))
    Y0[:, 3:] += rng.normal(0.0, 1.0, size=(N, 3))

    args = dict(duration_s=900.0, output_dt_s=300.0)
    results = {}
    for chunk in (N, 1, 3):
        prop = _make_propagator(grav, degree, chunk_size=chunk)
        assert prop.diagnostics_snapshot()["chunk_size"] == chunk
        _, Y, _, _ = prop.propagate(Y0, *([np.ones(N)] * 4), **args)
        results[chunk] = Y

    np.testing.assert_allclose(results[1], results[N], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(results[3], results[N], rtol=0.0, atol=0.0)


# ---------------------------------------------------------------------------
# Preflight: degree above the loaded coefficient file must error, not clip
# ---------------------------------------------------------------------------

def test_degree_above_loaded_model_raises_not_clips() -> None:
    grav = _make_gravity_model(50, seed=2)
    with pytest.raises(TorchSHPreflightError) as exc:
        _make_propagator(grav, degree=100)  # model only supports degree 50
    assert "degree 100" in str(exc.value).lower()
    assert "50" in str(exc.value)


def test_unsupported_physics_raises_preflight_error() -> None:
    from types import SimpleNamespace

    grav = _make_gravity_model(20, seed=2)
    dyn = SimpleNamespace(grav=grav, ephem=None)
    flags = PerturbationFlags(enable_sh=True, enable_srp=True)  # gravity-only path can't do SRP
    with pytest.raises(TorchSHPreflightError) as exc:
        TorchSHBatchPropagator(dyn, _mc_cfg(20), flags, device="cpu", dtype=torch.float64)
    assert "srp" in str(exc.value).lower()


# ---------------------------------------------------------------------------
# Provenance never records a clipped degree
# ---------------------------------------------------------------------------

def test_provenance_records_requested_and_actual_degree() -> None:
    grav = _make_gravity_model(100, seed=4)
    prop = _make_propagator(grav, degree=100)
    diag = prop.diagnostics_snapshot()
    assert diag["requested_gpu_sh_degree"] == 100
    assert diag["actual_gpu_sh_degree"] == 100
    assert diag["actual_gpu_sh_degree"] != 24  # never clipped to the Numba limit
    assert diag["backend_implementation"] == "torch"
    assert diag["integrator"] == "fixed-step RK4"
    prov = prop.provenance()
    assert prov["actual_backend"] == "torch_cpu_sh"  # cpu device -> torch_cpu_sh
    assert prov["backend_family"] == "classic_sh"


# ---------------------------------------------------------------------------
# CUDA-vs-CPU consistency (gated; only runs with a real CUDA device)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device available")
def test_cuda_matches_cpu_trajectory_float64() -> None:
    degree = 50
    grav = _make_gravity_model(degree, seed=9)
    Y0 = np.repeat(_circular_state(100.0)[None, :], 4, axis=0)
    args = dict(duration_s=900.0, output_dt_s=300.0)

    prop_cpu = _make_propagator(grav, degree, device="cpu", dtype=torch.float64)
    _, Y_cpu, _, _ = prop_cpu.propagate(Y0, *([np.ones(4)] * 4), **args)

    prop_cuda = _make_propagator(grav, degree, device="cuda", dtype=torch.float64)
    _, Y_cuda, _, _ = prop_cuda.propagate(Y0, *([np.ones(4)] * 4), **args)

    np.testing.assert_allclose(Y_cuda, Y_cpu, rtol=1e-10, atol=1e-4)



class MockRotEphemeris:
    """Mock ephemeris that rotates 90 degrees around Z axis every 100 seconds."""
    def __init__(self):
        self.t_0 = 0.0

    def get_data_provider(self):
        c45 = math.cos(math.pi / 4.0)
        return {
            "q_i2f_tab": np.array([[1.0, 0.0, 0.0, 0.0], [c45, 0.0, 0.0, c45], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
            "dt_s": 100.0
        }

    def get_moon_fixed_to_icrf_quaternion(self, t_s: float) -> np.ndarray:
        # 90 degrees every 100s
        theta = (t_s / 100.0) * (math.pi / 2.0)
        # Quaternion for rotation around Z: [cos(theta/2), 0, 0, sin(theta/2)]
        return np.array([math.cos(theta / 2.0), 0.0, 0.0, math.sin(theta / 2.0)], dtype=np.float64)

    # Required interface methods
    def get_moon_state(self, t_s: float) -> np.ndarray:
        return np.zeros(6)

    def get_sun_state(self, t_s: float) -> np.ndarray:
        return np.zeros(6)

    def get_earth_state(self, t_s: float) -> np.ndarray:
        return np.zeros(6)



# ---------------------------------------------------------------------------
# E. Frame Rotation (SLERP) testing (Task §16 MISSING EVIDENCE)
# ---------------------------------------------------------------------------

def test_frame_rotation_slerp() -> None:
    """Test frame interpolation and transformation mathematically."""
    from types import SimpleNamespace

    from lunaris.core.torch_sh_propagator import _TorchMoonFrame

    dyn = SimpleNamespace(ephem=MockRotEphemeris())
    PerturbationFlags(enable_sh=True)
    frame = _TorchMoonFrame(dyn.ephem, device=torch.device("cpu"), dtype=torch.float64)

    # 1. Test SLERP endpoints
    # At t=0, rotation should be 0 degrees (quaternion [1, 0, 0, 0])
    q_0 = frame.quat_i2f(0.0)
    np.testing.assert_allclose(q_0.numpy(), [1.0, 0.0, 0.0, 0.0], atol=1e-10)

    # At t=100, rotation should be 90 degrees around Z: [cos(45), 0, 0, sin(45)]
    q_100 = frame.quat_i2f(100.0)
    c45 = math.cos(math.pi / 4.0)
    np.testing.assert_allclose(q_100.numpy(), [c45, 0.0, 0.0, c45], atol=1e-10)

    # 2. Test SLERP midpoint
    # At t=50, rotation should be 45 degrees around Z: [cos(22.5), 0, 0, sin(22.5)]
    q_50 = frame.quat_i2f(50.0)
    c225 = math.cos(math.pi / 8.0)
    s225 = math.sin(math.pi / 8.0)
    np.testing.assert_allclose(q_50.numpy(), [c225, 0.0, 0.0, s225], atol=1e-10)

    # 3. Test vector rotation (90 degrees around Z)
    # Rotating vector [1, 0, 0] by 90 deg around Z should yield [0, 1, 0]
    from lunaris.core.torch_sh_propagator import _quat_rotate
    v_in = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
    v_out = _quat_rotate(q_100, v_in)
    np.testing.assert_allclose(v_out.numpy()[0], [0.0, 1.0, 0.0], atol=1e-10)

    # 4. Test inverse rotation (using conjugate)
    q_conj = q_100.clone()
    q_conj[1:] = -q_conj[1:]
    v_back = _quat_rotate(q_conj, v_out)
    np.testing.assert_allclose(v_back.numpy()[0], [1.0, 0.0, 0.0], atol=1e-10)

    # 5. Test norm preservation
    v_rand = torch.randn(10, 3, dtype=torch.float64)
    norm_in = torch.linalg.norm(v_rand, dim=1)
    v_rot = _quat_rotate(q_50, v_rand)
    norm_out = torch.linalg.norm(v_rot, dim=1)
    np.testing.assert_allclose(norm_out.numpy(), norm_in.numpy(), atol=1e-10)


# ---------------------------------------------------------------------------
# F. Throughput Metrics and Impact Stability
# ---------------------------------------------------------------------------

def test_throughput_metrics_and_impact_stability() -> None:
    """Test the newly added throughput metrics and impact masking logic."""
    degree = 20
    grav = _make_gravity_model(degree, seed=8)

    # Create N=5 trajectories, 3 of which are definitely impacting (R < R_REF)
    y_safe = _circular_state(100.0)
    y_impact = _circular_state(-10.0)  # Inside the moon

    Y0 = np.stack([y_safe, y_impact, y_safe, y_impact, y_impact])
    prop = _make_propagator(grav, degree, chunk_size=2)  # Use small chunk to test accumulation

    duration = 300.0
    dt = 60.0
    steps_per_snap = int(duration / dt)
    t_out, Y_out, impact, t_imp = prop.propagate(
        Y0, np.zeros(5), np.zeros(5), np.zeros(5), np.zeros(5),
        duration_s=duration, output_dt_s=duration,
    )

    diag = prop.diagnostics_snapshot()

    # 1. Verify throughput metrics exist
    assert "active_state_steps_per_second" in diag
    assert "raw_batch_state_steps_per_second" in diag
    assert diag["total_raw_state_steps"] == 5 * steps_per_snap

    # Active steps should be strictly less than raw steps because 3 samples impact immediately
    assert diag["total_active_state_steps"] < diag["total_raw_state_steps"]

    # 2. Verify impact stability
    assert np.sum(impact) == 3.0
    assert impact[1] == 1.0
    assert impact[3] == 1.0
    assert impact[4] == 1.0
    assert impact[0] == 0.0
    assert impact[2] == 0.0

    # 3. Verify no NaNs in output (impacted trajectories freeze or stay finite)
    assert np.all(np.isfinite(Y_out))


def test_throughput_no_impact_raw_equals_active() -> None:
    """No-impact case: every sample is alive at every step, so the raw and active
    state-step counts (and therefore the two throughput rates) must coincide."""
    degree = 20
    grav = _make_gravity_model(degree, seed=14)
    Y0 = np.repeat(_circular_state(120.0)[None, :], 4, axis=0)  # circular -> no impact
    prop = _make_propagator(grav, degree, chunk_size=2)  # >1 chunk: exercise aggregation
    prop.propagate(
        Y0, *([np.ones(4)] * 4), duration_s=600.0, output_dt_s=300.0,
    )
    diag = prop.diagnostics_snapshot()
    assert diag["impacted_sample_count"] == 0
    assert diag["active_sample_count"] == 4
    assert diag["total_active_state_steps"] == diag["total_raw_state_steps"]
    assert diag["active_state_steps_per_second"] == pytest.approx(
        diag["raw_batch_state_steps_per_second"], rel=1e-9
    )


# ---------------------------------------------------------------------------
# G. Autograd inference safety (task §1 MAJOR)
# ---------------------------------------------------------------------------

def test_rk4_step_builds_no_autograd_graph() -> None:
    """Inside the propagator's no-grad RK4 scope, stepping an input that *requires*
    grad must not build an autograd graph: the output has no grad_fn and does not
    require grad. Proves the loop never depends on autograd (acceleration is analytic)."""
    degree = 20
    grav = _make_gravity_model(degree, seed=11)
    prop = _make_propagator(grav, degree)

    s = torch.as_tensor(_circular_state(100.0)[None, :], dtype=torch.float64)
    s.requires_grad_(True)
    with torch.no_grad():
        rhs = prop._rhs(0.0, s)
        out = prop._rk4_step(s, 0.0, 60.0)
    assert rhs.grad_fn is None and rhs.requires_grad is False
    assert out.grad_fn is None and out.requires_grad is False


def test_propagation_is_nograd_under_outer_enable_grad() -> None:
    """An outer torch.enable_grad() must NOT reactivate graph construction inside
    the RK4 loop. Results must be bitwise-identical to the default call and finite."""
    degree = 30
    grav = _make_gravity_model(degree, seed=12)
    prop = _make_propagator(grav, degree)

    Y0 = np.repeat(_circular_state(120.0)[None, :], 3, axis=0)
    args = dict(duration_s=600.0, output_dt_s=300.0)

    _, Y_default, _, _ = prop.propagate(Y0, *([np.ones(3)] * 4), **args)
    with torch.enable_grad():
        _, Y_enabled, _, _ = prop.propagate(Y0, *([np.ones(3)] * 4), **args)

    np.testing.assert_allclose(Y_enabled, Y_default, rtol=0.0, atol=0.0)
    assert np.all(np.isfinite(Y_enabled))


# ---------------------------------------------------------------------------
# H. Quaternion sign equivalence + convention (task §16 MISSING EVIDENCE)
# ---------------------------------------------------------------------------

class MockSignFlipEphemeris:
    """Ephemeris whose second quaternion is the SIGN-FLIPPED 90 deg z-rotation.

    ``-q`` and ``q`` describe the same rotation; the SLERP must take the shortest
    path across the sign flip rather than the long way around.
    """

    def get_data_provider(self):
        c45 = math.cos(math.pi / 4.0)
        return {
            "q_i2f_tab": np.array(
                [[1.0, 0.0, 0.0, 0.0], [-c45, 0.0, 0.0, -c45]], dtype=np.float64
            ),
            "dt_s": 100.0,
        }


def test_quaternion_sign_equivalence_and_convention() -> None:
    """q and -q rotate a vector identically (same rotation); the active i2f
    convention maps +x -> +y for a 90 deg +z rotation."""
    from lunaris.core.torch_sh_propagator import _quat_rotate

    c45 = math.cos(math.pi / 4.0)
    q = torch.tensor([c45, 0.0, 0.0, c45], dtype=torch.float64)
    v = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
    v_q = _quat_rotate(q, v)
    v_negq = _quat_rotate(-q, v)
    np.testing.assert_allclose(v_q.numpy(), v_negq.numpy(), atol=1e-12)
    # Convention: active 90 deg z-rotation maps +x -> +y.
    np.testing.assert_allclose(v_q.numpy()[0], [0.0, 1.0, 0.0], atol=1e-12)


def test_slerp_takes_shortest_path_across_sign_flip() -> None:
    """SLERP midpoint between identity and the sign-flipped 90 deg rotation must be
    the 45 deg rotation (compared via |dot| ≈ 1 to absorb the q/-q ambiguity)."""
    from lunaris.core.torch_sh_propagator import _TorchMoonFrame

    frame = _TorchMoonFrame(
        MockSignFlipEphemeris(), device=torch.device("cpu"), dtype=torch.float64
    )
    q_50 = frame.quat_i2f(50.0).numpy()
    c225 = math.cos(math.pi / 8.0)
    s225 = math.sin(math.pi / 8.0)
    expected = np.array([c225, 0.0, 0.0, s225])
    np.testing.assert_allclose(abs(float(np.dot(q_50, expected))), 1.0, atol=1e-9)
    np.testing.assert_allclose(float(np.linalg.norm(q_50)), 1.0, atol=1e-10)
