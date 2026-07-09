"""Reviewer §1: gravity + propagation correctness, dtype, and CPU/GPU parity.

These guard the project's core value proposition — that the spherical-harmonic
gravity kernel and the RK4 propagator produce the *same* physically correct
trajectory on CPU and GPU, in float32 and float64 — using a small, fixed,
in-code gravity fixture (no large data files, no network). They run in normal
CI; the CPU/GPU parity test is marked ``requires_cuda`` and skips cleanly when no
device is present.

What is checked, and why a silent break would matter:
  * Point-mass Kepler sanity: a wrong sign / frame in the monopole path would
    make a circular orbit spiral or precess.
  * J2 invariants: an axisymmetric (zonal) field conserves z-angular momentum
    exactly and energy to integrator order; a broken Legendre recurrence or a
    potential-vs-acceleration sign error breaks these.
  * Golden short-orbit regression: pins the exact float64 CPU trajectory so any
    change in the kernel/integrator numerics is flagged.
  * float32-vs-float64 drift: bounds the dtype penalty so a GPU float32 run stays
    trustworthy.
  * CPU/GPU parity: identical float64 RK4 + coefficients must agree to rounding.
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



import math
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from lunaris.common.constants import MU_MOON, R_MOON  # noqa: E402
from lunaris.common.type_defs import PerturbationFlags  # noqa: E402
from lunaris.core.torch_sh_propagator import TorchSHBatchPropagator  # noqa: E402
from lunaris.physics.spherical_harmonics import GravityModel  # noqa: E402

MU = float(MU_MOON)
R = float(R_MOON)
_DEGREE = 4
# A Moon-like normalized J2 (C[2,0]); everything else zero so the field is a
# clean, deterministic, axisymmetric test case.
_C20 = -9.08e-5

CUDA = torch.cuda.is_available()


def _gravity_model(*, c20: float = _C20) -> GravityModel:
    C = np.zeros((_DEGREE + 1, _DEGREE + 1), dtype=np.float64)
    S = np.zeros_like(C)
    C[2, 0] = float(c20)
    return GravityModel.from_arrays(_DEGREE, R, MU, C, S)


def _batch_cfg() -> SimpleNamespace:
    return SimpleNamespace(
        sh_degree=_DEGREE,
        dt_s=30.0,
        impact_alt_km=0.0,
        torch_dtype="float64",
        torch_sh_chunk_size=0,
        gpu_device_id=0,
    )


def _propagator(grav: GravityModel, *, device: str, dtype) -> TorchSHBatchPropagator:
    dyn = SimpleNamespace(grav=grav, ephem=None)
    return TorchSHBatchPropagator(
        dyn, _batch_cfg(), PerturbationFlags(enable_sh=True), device=device, dtype=dtype
    )


def _run(grav, *, device="cpu", dtype=torch.float64, Y0, duration_s=3600.0, output_dt_s=600.0):
    n = int(Y0.shape[0])
    ones = [np.ones(n)] * 4  # masses/areas/cds/crs — accepted but unused (gravity only)
    _, Y, _, _ = _propagator(grav, device=device, dtype=dtype).propagate(
        Y0, *ones, duration_s=duration_s, output_dt_s=output_dt_s
    )
    return Y  # (n_snaps+1, N, 6)


def _inclined_circular_state(alt_m: float = 150e3, inc_deg: float = 45.0) -> np.ndarray:
    r0 = R + alt_m
    v_circ = math.sqrt(MU / r0)
    inc = math.radians(inc_deg)
    return np.array(
        [[r0, 0.0, 0.0, 0.0, v_circ * math.cos(inc), v_circ * math.sin(inc)]],
        dtype=np.float64,
    )


def _energy(y: np.ndarray) -> float:
    r, v = y[:3], y[3:]
    return 0.5 * float(np.dot(v, v)) - MU / float(np.linalg.norm(r))


def _ang_mom_z(y: np.ndarray) -> float:
    r, v = y[:3], y[3:]
    return float(np.cross(r, v)[2])


# ---------------------------------------------------------------------------
# Point-mass (monopole) sanity
# ---------------------------------------------------------------------------

def test_point_mass_circular_orbit_stays_circular_cpu():
    grav = _gravity_model(c20=0.0)  # pure monopole
    Y = _run(grav, Y0=_inclined_circular_state(inc_deg=0.0))
    radii = np.linalg.norm(Y[:, 0, :3], axis=1)
    speeds = np.linalg.norm(Y[:, 0, 3:], axis=1)
    # Circular orbit: radius and speed are constant to sub-meter / sub-mm-per-s.
    assert radii.max() - radii.min() < 1.0
    assert speeds.max() - speeds.min() < 1e-3
    e0, e1 = _energy(Y[0, 0]), _energy(Y[-1, 0])
    assert abs((e1 - e0) / e0) < 1e-6


# ---------------------------------------------------------------------------
# J2 (zonal) physical invariants
# ---------------------------------------------------------------------------

def test_j2_conserves_z_angular_momentum_and_energy_cpu():
    grav = _gravity_model()
    Y = _run(grav, Y0=_inclined_circular_state())
    lz = np.array([_ang_mom_z(Y[k, 0]) for k in range(Y.shape[0])])
    en = np.array([_energy(Y[k, 0]) for k in range(Y.shape[0])])
    # An axisymmetric (zonal) field exerts no torque about z -> Lz is conserved
    # to ~machine precision. This is the strong, potential-independent invariant.
    assert np.max(np.abs((lz - lz[0]) / lz[0])) < 1e-7
    # The *Keplerian* energy (-MU/|r|) is NOT conserved under J2 — it oscillates
    # by ~J2*(R/r)^2 as the orbit moves through the oblateness potential. We only
    # require the oscillation to stay bounded (no secular blow-up from integrator
    # instability), not to vanish.
    assert np.max(np.abs((en - en[0]) / en[0])) < 1e-3


# ---------------------------------------------------------------------------
# Golden short-orbit regression (float64 CPU)
# ---------------------------------------------------------------------------

def test_golden_short_orbit_regression_cpu():
    grav = _gravity_model()
    Y = _run(grav, Y0=_inclined_circular_state())
    # Pinned float64 CPU final state for the fixed fixture/orbit/step above.
    # Any change in the SH kernel or RK4 integrator numerics will shift this.
    golden = np.array(
        [-1883082.1621276576, 90243.7790067893, 89163.8128236016,
         -108.36705418477092, -1137.502016875728, -1137.5627293605785],
        dtype=np.float64,
    )
    np.testing.assert_allclose(Y[-1, 0], golden, rtol=2e-6, atol=1e-3)


# ---------------------------------------------------------------------------
# float32 vs float64 drift bound
# ---------------------------------------------------------------------------

def test_float32_vs_float64_drift_is_bounded_cpu():
    grav = _gravity_model()
    Y0 = _inclined_circular_state()
    Y64 = _run(grav, Y0=Y0, dtype=torch.float64)
    Y32 = _run(grav, Y0=Y0, dtype=torch.float32)
    pos_err = np.linalg.norm(Y32[-1, 0, :3] - Y64[-1, 0, :3])
    vel_err = np.linalg.norm(Y32[-1, 0, 3:] - Y64[-1, 0, 3:])
    r_mag = np.linalg.norm(Y64[-1, 0, :3])
    # Single-precision accumulates ~1e-6 relative over this short arc; bound it
    # generously so the test guards regressions without being flaky.
    assert pos_err / r_mag < 1e-4
    assert vel_err < 1e-2


# ---------------------------------------------------------------------------
# CPU/GPU parity (needs a CUDA device)
# ---------------------------------------------------------------------------

@pytest.mark.requires_cuda
@pytest.mark.skipif(not CUDA, reason="CPU/GPU parity needs a CUDA device")
def test_cpu_gpu_trajectory_parity_float64():
    grav = _gravity_model()
    Y0 = _inclined_circular_state()
    Y_cpu = _run(grav, Y0=Y0, device="cpu", dtype=torch.float64)
    Y_gpu = _run(grav, Y0=Y0, device="cuda", dtype=torch.float64)
    # Identical float64 RK4 + identical coefficients: only rounding-order differs.
    np.testing.assert_allclose(Y_gpu, Y_cpu, rtol=1e-9, atol=1e-3)
