"""R03: gpu_st_lrps_third_body hybrid backend.

Coverage
--------
A. Numerical parity with the CPU reference kernels:
   - Catmull-Rom vec3 ephemeris interpolation vs common.math_utils
   - Battin F(q) third-body acceleration vs physics.third_body_effects
B. Policy routing: third-body flags upgrade auto/mission ST-LRPS to the hybrid
   backend with NO fallback; other perturbations still fall back; explicit
   requests honored; provenance fields populated.
C. Propagator: hybrid provider end-to-end vs an independent numpy RK4
   reference (point-mass + third-body, zero-weight surrogate); fail-closed
   preflight on missing/zeroed ephemeris.
D. 10k-batch GPU acceptance (CUDA-gated).
"""

from __future__ import annotations

import pytest

try:
    import torch
    import torch.nn
    _ = torch.cuda
    _ = getattr(torch, "compile", None)
except (ImportError, AttributeError, ModuleNotFoundError):
    pytest.skip("PyTorch not installed or missing required attributes like cuda/compile", allow_module_level=True)


from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.requires_torch

from lunaris.common.constants import MU_EARTH, MU_SUN  # noqa: E402
from lunaris.common.math_utils import interp_vec3_catmull  # noqa: E402
from lunaris.common.type_defs import PerturbationFlags  # noqa: E402
from lunaris.core.torch_batch_propagator import (  # noqa: E402
    TorchSTLRPSPreflightError,
    _resolve_third_body_tables,
)
from lunaris.core.torch_third_body import (  # noqa: E402
    TorchEphemerisTables,
    interp_vec3_catmull_torch,
    third_body_accel_batch,
)
from lunaris.physics.third_body_effects import accel_third_body_numba  # noqa: E402

R_MOON = 1.7374e6


# ---------------------------------------------------------------------------
# A. CPU parity — interpolation + Battin F(q)
# ---------------------------------------------------------------------------


def test_catmull_interp_matches_cpu_reference():
    rng = np.random.default_rng(5)
    tab = rng.normal(0.0, 4.0e8, size=(12, 3))
    dt = 600.0
    t_tab = torch.as_tensor(tab, dtype=torch.float64)
    # Interior points, endpoints, out-of-range clamps, exact grid nodes.
    times = [
        -10.0, 0.0, 1.0, 299.9, 600.0, 725.3, 3000.0, 6000.0, 6600.0 - 1e-9, 6600.0, 9999.0,
    ]
    for t in times:
        expected = np.array(interp_vec3_catmull(float(t), dt, tab))
        got = interp_vec3_catmull_torch(float(t), dt, t_tab).numpy()
        np.testing.assert_allclose(got, expected, rtol=1e-14, atol=1e-6)


def test_catmull_interp_degenerate_tables():
    one_row = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64)
    np.testing.assert_allclose(
        interp_vec3_catmull_torch(123.0, 60.0, one_row).numpy(), [1.0, 2.0, 3.0]
    )
    empty = torch.empty((0, 3), dtype=torch.float64)
    np.testing.assert_allclose(interp_vec3_catmull_torch(5.0, 60.0, empty).numpy(), [0.0, 0.0, 0.0])


def test_battin_third_body_matches_cpu_reference():
    rng = np.random.default_rng(7)
    n = 64
    r_sc = rng.normal(0.0, 1.0, size=(n, 3))
    r_sc *= (R_MOON + rng.uniform(30e3, 500e3, size=(n, 1))) / np.linalg.norm(
        r_sc, axis=1, keepdims=True
    )
    for r_body, mu in (
        (np.array([3.5e8, 1.2e8, -0.4e8]), float(MU_EARTH)),   # Earth-like range
        (np.array([1.3e11, -0.6e11, 0.2e11]), float(MU_SUN)),  # Sun-like range
    ):
        expected = np.array(
            [accel_third_body_numba(*row, *r_body, mu) for row in r_sc], dtype=np.float64
        )
        got = third_body_accel_batch(
            torch.as_tensor(r_sc, dtype=torch.float64),
            torch.as_tensor(r_body, dtype=torch.float64),
            mu,
        ).numpy()
        np.testing.assert_allclose(got, expected, rtol=1e-13, atol=1e-20)


def test_battin_singularity_guard_matches_cpu_zero_policy():
    # Spacecraft coincident with the third body -> guarded zero row.
    r_body = np.array([1.0e8, 0.0, 0.0])
    r_sc = np.array([[1.0e8, 0.0, 0.0], [2.0e6, 0.0, 0.0]])
    got = third_body_accel_batch(
        torch.as_tensor(r_sc, dtype=torch.float64),
        torch.as_tensor(r_body, dtype=torch.float64),
        float(MU_EARTH),
    ).numpy()
    assert np.all(got[0] == 0.0)
    assert np.all(np.isfinite(got[1])) and np.any(got[1] != 0.0)


def test_ephemeris_tables_fail_closed_on_zero_tables():
    zeros = np.zeros((4, 3))
    sun = np.tile([1.3e11, 0.0, 0.0], (4, 1))
    with pytest.raises(RuntimeError, match="Earth table is all zeros"):
        TorchEphemerisTables(
            dt_s=600.0, r_sun_tab_m=sun, r_earth_tab_m=zeros,
            device=torch.device("cpu"), dtype=torch.float64,
            need_sun=True, need_earth=True,
        )
    # Not needed -> zero table tolerated.
    tables = TorchEphemerisTables(
        dt_s=600.0, r_sun_tab_m=sun, r_earth_tab_m=zeros,
        device=torch.device("cpu"), dtype=torch.float64,
        need_sun=True, need_earth=False,
    )
    np.testing.assert_allclose(tables.sun_position(0.0).numpy(), sun[0])


def test_resolve_third_body_tables_preserves_ephemeris_gm_values():
    sun_tab, earth_tab = _make_tables(4, 600.0)
    custom_mu_sun = float(MU_SUN) * 1.000001
    custom_mu_earth = float(MU_EARTH) * 0.999999

    class _Ephem:
        def get_data_provider(self):
            return {
                "dt_s": 600.0,
                "r_sun_tab_m": sun_tab,
                "r_earth_tab_m": earth_tab,
                "q_i2f_tab": np.tile(
                    np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64),
                    (sun_tab.shape[0], 1),
                ),
                "mu_sun_m3s2": custom_mu_sun,
                "mu_earth_m3s2": custom_mu_earth,
            }

    tables = _resolve_third_body_tables(
        ("third_body_sun", "third_body_earth"),
        _Ephem(),
        device=torch.device("cpu"),
        dtype=torch.float64,
    )

    assert tables.mu_source == "ephemeris_provider"
    assert tables.mu_sun_m3s2 == pytest.approx(custom_mu_sun)
    assert tables.mu_earth_m3s2 == pytest.approx(custom_mu_earth)


# ---------------------------------------------------------------------------
# B. Policy routing (R03 acceptance #2: no fallback with third-body enabled)
# ---------------------------------------------------------------------------


def _st_lrps_sim_cfg(**flag_kwargs):
    return SimpleNamespace(
        flags=PerturbationFlags(enable_sh=True, **flag_kwargs),
        gravity=SimpleNamespace(uses_st_lrps=True),
    )


def test_policy_third_body_flags_select_hybrid_backend_no_fallback(monkeypatch):
    import lunaris.batch.backend_policy as policy_mod
    from lunaris.batch.backend_policy import BatchBackend, resolve_batch_backend_policy

    monkeypatch.setattr(policy_mod, "_torch_cuda_available", lambda: True)
    monkeypatch.setattr(policy_mod, "_numba_cuda_available", lambda: False)

    batch_cfg = SimpleNamespace(use_gpu=True, gravity_mode_override="st_lrps")
    sim_cfg = _st_lrps_sim_cfg(enable_3rd_body_sun=True, enable_3rd_body_earth=True)
    plan = resolve_batch_backend_policy(batch_cfg, sim_cfg)

    assert plan.final_backend == BatchBackend.GPU_ST_LRPS
    assert plan.actual_backend == "gpu_st_lrps_third_body"
    assert plan.fallback_applied is False
    assert plan.use_gpu is True
    # R03 provenance fields.
    assert plan.gravity_backend == "st_lrps"
    assert plan.third_body_backend == "analytic_vectorized"
    assert "srp" in plan.unsupported_forces
    assert "third_body_sun" not in plan.unsupported_forces
    assert plan.effective_dtype in ("float32", "float64")


def test_policy_gravity_only_still_selects_potential_backend(monkeypatch):
    import lunaris.batch.backend_policy as policy_mod
    from lunaris.batch.backend_policy import resolve_batch_backend_policy

    monkeypatch.setattr(policy_mod, "_torch_cuda_available", lambda: True)
    monkeypatch.setattr(policy_mod, "_numba_cuda_available", lambda: False)

    plan = resolve_batch_backend_policy(
        SimpleNamespace(use_gpu=True, gravity_mode_override="st_lrps"),
        _st_lrps_sim_cfg(),
    )
    assert plan.actual_backend == "gpu_st_lrps_potential"
    assert plan.third_body_backend == ""


def test_policy_srp_still_falls_back_even_with_third_body(monkeypatch):
    import lunaris.batch.backend_policy as policy_mod
    from lunaris.batch.backend_policy import BatchBackend, resolve_batch_backend_policy

    monkeypatch.setattr(policy_mod, "_torch_cuda_available", lambda: True)
    monkeypatch.setattr(policy_mod, "_numba_cuda_available", lambda: False)

    plan = resolve_batch_backend_policy(
        SimpleNamespace(use_gpu=True, gravity_mode_override="st_lrps"),
        _st_lrps_sim_cfg(enable_3rd_body_sun=True, enable_srp=True),
    )
    assert plan.final_backend == BatchBackend.CPU
    assert plan.fallback_applied is True
    assert "srp" in plan.fallback_reason.lower()


def test_policy_explicit_hybrid_request_honored(monkeypatch):
    import lunaris.batch.backend_policy as policy_mod
    from lunaris.batch.backend_policy import resolve_batch_backend_policy

    monkeypatch.setattr(policy_mod, "_torch_cuda_available", lambda: True)
    monkeypatch.setattr(policy_mod, "_numba_cuda_available", lambda: False)

    plan = resolve_batch_backend_policy(
        SimpleNamespace(
            use_gpu=True, gravity_mode_override="st_lrps",
            batch_backend="gpu_st_lrps_third_body",
        ),
        _st_lrps_sim_cfg(enable_3rd_body_earth=True),
    )
    assert plan.actual_backend == "gpu_st_lrps_third_body"
    assert plan.fallback_applied is False


def test_policy_explicit_potential_request_with_third_body_falls_back(monkeypatch):
    # An explicit gravity-only request is never silently upgraded.
    import lunaris.batch.backend_policy as policy_mod
    from lunaris.batch.backend_policy import BatchBackend, resolve_batch_backend_policy

    monkeypatch.setattr(policy_mod, "_torch_cuda_available", lambda: True)
    monkeypatch.setattr(policy_mod, "_numba_cuda_available", lambda: False)

    plan = resolve_batch_backend_policy(
        SimpleNamespace(
            use_gpu=True, gravity_mode_override="st_lrps",
            batch_backend="gpu_st_lrps_potential",
        ),
        _st_lrps_sim_cfg(enable_3rd_body_sun=True),
    )
    assert plan.final_backend == BatchBackend.CPU
    assert plan.fallback_applied is True


def test_registry_entry_matches_r03_scope():
    from lunaris.core.backend_capabilities import get_capabilities

    caps = get_capabilities("gpu_st_lrps_third_body")
    assert caps.family == "st_lrps"
    assert caps.supports_third_body is True
    for unsupported in ("earth_j2", "srp", "albedo", "thermal_ir", "solid_tides_k2", "relativity_1pn"):
        assert caps.supports_force_model(unsupported) is False, unsupported
    assert caps.integrator == "fixed-step RK4"
    assert set(caps.dtype_support) == {"float32", "float64"}


# ---------------------------------------------------------------------------
# C. Hybrid provider end-to-end vs independent numpy reference (CPU device)
# ---------------------------------------------------------------------------


class _ZeroLunarModel:
    """Stand-in surrogate: pure point-mass Moon gravity (zero neural residual)."""

    def __init__(self, mu: float) -> None:
        self.mu = float(mu)

    def predict_total_accel_torch(self, r_f):
        rn = torch.linalg.norm(r_f, dim=1, keepdim=True).clamp_min(1.0)
        return -self.mu * r_f / (rn**3)


def _make_tables(n_rows: int, dt: float) -> tuple[np.ndarray, np.ndarray]:
    # Slowly rotating Earth/Sun positions (realistic magnitudes).
    ang = np.linspace(0.0, 0.05, n_rows)
    earth = np.stack(
        [3.844e8 * np.cos(ang), 3.844e8 * np.sin(ang), 0.02 * 3.844e8 * np.sin(ang)], axis=1
    )
    sun = np.stack(
        [1.496e11 * np.cos(ang / 13.0), 1.496e11 * np.sin(ang / 13.0), np.zeros(n_rows)], axis=1
    )
    return sun, earth


def test_hybrid_provider_matches_numpy_reference_rk4():
    """Acceptance #4: small batch vs CPU reference (same force set, same tables)."""
    from lunaris.core.batched_fixed_step import run_batched_fixed_step
    from lunaris.core.torch_batch_propagator import _STLRPSThirdBodyAccelerationProvider
    from lunaris.core.torch_frame import TorchMoonFrame

    mu_moon = 4.9028e12
    dt_tab = 600.0
    sun_tab, earth_tab = _make_tables(8, dt_tab)
    tables = TorchEphemerisTables(
        dt_s=dt_tab, r_sun_tab_m=sun_tab, r_earth_tab_m=earth_tab,
        device=torch.device("cpu"), dtype=torch.float64,
        need_sun=True, need_earth=True,
    )
    frame = TorchMoonFrame(None, device=torch.device("cpu"), dtype=torch.float64, allow_identity=True)
    provider = _STLRPSThirdBodyAccelerationProvider(
        _ZeroLunarModel(mu_moon), frame,
        ephem_tables=tables, use_sun=True, use_earth=True,
        mu_sun=float(MU_SUN), mu_earth=float(MU_EARTH),
    )

    r0 = R_MOON + 100e3
    v0 = float(np.sqrt(mu_moon / r0))
    Y0 = np.array(
        [
            [r0, 0.0, 0.0, 0.0, v0, 0.0],
            [0.0, r0 + 50e3, 0.0, -v0 * 0.98, 0.0, 120.0],
        ],
        dtype=np.float64,
    )
    duration, out_dt, dt = 1_200.0, 300.0, 60.0
    res = run_batched_fixed_step(
        torch_mod=torch, device=torch.device("cpu"), dtype=torch.float64,
        provider=provider, frame=frame, Y0=Y0,
        duration_s=duration, output_dt_s=out_dt, dt_s=dt,
        impact_r_m=R_MOON, detect_impact=True,
    )

    # Independent numpy fixed-step RK4 with the CPU kernels + CPU interpolation.
    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        r = y[:3]
        rn = max(float(np.linalg.norm(r)), 1.0)
        a = -mu_moon * r / rn**3
        sun = np.array(interp_vec3_catmull(t, dt_tab, sun_tab))
        earth = np.array(interp_vec3_catmull(t, dt_tab, earth_tab))
        a = a + np.array(accel_third_body_numba(*r, *sun, float(MU_SUN)))
        a = a + np.array(accel_third_body_numba(*r, *earth, float(MU_EARTH)))
        return np.concatenate([y[3:], a])

    for j in range(Y0.shape[0]):
        y = Y0[j].copy()
        t = 0.0
        n_steps = int(round(duration / dt))
        for _ in range(n_steps):
            k1 = rhs(t, y)
            k2 = rhs(t + 0.5 * dt, y + 0.5 * dt * k1)
            k3 = rhs(t + 0.5 * dt, y + 0.5 * dt * k2)
            k4 = rhs(t + dt, y + dt * k3)
            y = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            t += dt
        np.testing.assert_allclose(res.Y_out[-1, j, :3], y[:3], rtol=1e-9, atol=1e-3)
        np.testing.assert_allclose(res.Y_out[-1, j, 3:], y[3:], rtol=1e-9, atol=1e-6)


def test_third_body_term_changes_trajectory():
    """The hybrid provider's third-body terms are actually live (not a no-op)."""
    from lunaris.core.torch_batch_propagator import (
        _STLRPSAccelerationProvider,
        _STLRPSThirdBodyAccelerationProvider,
    )
    from lunaris.core.torch_frame import TorchMoonFrame

    mu_moon = 4.9028e12
    sun_tab, earth_tab = _make_tables(4, 600.0)
    tables = TorchEphemerisTables(
        dt_s=600.0, r_sun_tab_m=sun_tab, r_earth_tab_m=earth_tab,
        device=torch.device("cpu"), dtype=torch.float64,
        need_sun=False, need_earth=True,
    )
    frame = TorchMoonFrame(None, device=torch.device("cpu"), dtype=torch.float64, allow_identity=True)
    model = _ZeroLunarModel(mu_moon)
    base = _STLRPSAccelerationProvider(model, frame)
    hybrid = _STLRPSThirdBodyAccelerationProvider(
        model, frame, ephem_tables=tables, use_sun=False, use_earth=True,
        mu_sun=float(MU_SUN), mu_earth=float(MU_EARTH),
    )
    s = torch.tensor([[R_MOON + 100e3, 0.0, 0.0, 0.0, 1.6e3, 0.0]], dtype=torch.float64)
    a0 = base.acceleration(0.0, s)
    a1 = hybrid.acceleration(0.0, s)
    diff = float(torch.linalg.norm(a1 - a0))
    assert diff > 0.0
    # Earth tidal term at 100 km LLO is ~1e-6..1e-5 m/s^2 — small but nonzero.
    assert 1e-8 < diff < 1e-3


def test_propagator_preflight_missing_ephemeris_fails_closed():
    # Re-run only the third-body preflight logic; the full constructor is
    # CUDA-gated and heavy for this failure-mode unit test.
    with pytest.raises(TorchSTLRPSPreflightError, match="requires an ephemeris"):
        _resolve_third_body_tables(
            ("third_body_earth",),
            None,
            device=torch.device("cpu"),
            dtype=torch.float64,
        )


# ---------------------------------------------------------------------------
# D. 10k batch on a real GPU (acceptance #1; CUDA-gated)
# ---------------------------------------------------------------------------


_cuda_available = bool(hasattr(torch, "cuda") and torch.cuda.is_available())


@pytest.mark.skipif(not _cuda_available, reason="no CUDA device available")
def test_hybrid_10k_batch_propagates_on_gpu_without_fallback():
    from lunaris.core.batched_fixed_step import run_batched_fixed_step
    from lunaris.core.torch_batch_propagator import _STLRPSThirdBodyAccelerationProvider
    from lunaris.core.torch_frame import TorchMoonFrame

    device = torch.device("cuda:0")
    mu_moon = 4.9028e12
    sun_tab, earth_tab = _make_tables(8, 600.0)
    tables = TorchEphemerisTables(
        dt_s=600.0, r_sun_tab_m=sun_tab, r_earth_tab_m=earth_tab,
        device=device, dtype=torch.float64, need_sun=True, need_earth=True,
    )
    frame = TorchMoonFrame(None, device=device, dtype=torch.float64, allow_identity=True)
    provider = _STLRPSThirdBodyAccelerationProvider(
        _ZeroLunarModel(mu_moon), frame,
        ephem_tables=tables, use_sun=True, use_earth=True,
        mu_sun=float(MU_SUN), mu_earth=float(MU_EARTH),
    )
    n = 10_000
    rng = np.random.default_rng(11)
    r0 = R_MOON + 100e3
    v0 = float(np.sqrt(mu_moon / r0))
    Y0 = np.tile(np.array([r0, 0.0, 0.0, 0.0, v0, 0.0]), (n, 1))
    Y0[:, :3] += rng.normal(0.0, 5_000.0, size=(n, 3))
    res = run_batched_fixed_step(
        torch_mod=torch, device=device, dtype=torch.float64,
        provider=provider, frame=frame, Y0=Y0,
        duration_s=600.0, output_dt_s=300.0, dt_s=60.0,
        impact_r_m=R_MOON, chunk_size=8192,
    )
    assert res.Y_out.shape == (3, n, 6)
    assert np.all(np.isfinite(res.Y_out))
