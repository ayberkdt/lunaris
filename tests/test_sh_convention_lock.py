"""
Spherical-harmonic convention lock (scientific-hardening Phase 4).

Locks normalization, phase, potential sign, and acceleration sign across every
SH evaluation path with SINGLE-coefficient artificial fields. Combined-field
parity tests (test_spherical_harmonics.py, test_st_lrps_sh_baseline.py,
test_torch_sh_evaluator.py) catch that *something* broke; these per-term tests
localize *which* (n, m) term / phase convention broke, and pin

  - ``GravityModel.accel_fixed``            (recurrence acceleration kernel)
  - ``sh_potential_accel_fixed``            (potential + gradient kernel)
  - ``TorchSHGravityEvaluator.acceleration``(torch tensor kernel)
  - central finite differences of the potential (a = +grad(U) sign lock)

against each other as full VECTORS (not norms) at equatorial, mid-latitude,
near-pole, multi-longitude, and multi-altitude positions.
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

import numpy as np
import pytest

from lunaris.physics.spherical_harmonics import (
    GravityModel,
    sh_potential_accel_fixed,
)

R_REF = 1_737_400.0        # [m]  Moon-like reference radius
GM = 4.904_869_5e12        # [m^3/s^2]

# One term at a time: every low-degree shape class that a phase/normalization
# regression could affect differently (zonal, tesseral odd-m, sectoral, S vs C).
SINGLE_TERMS = (
    ("C", 2, 0, -2.0e-4),   # zonal (J2-like)
    ("C", 2, 1, 1.5e-5),    # tesseral, odd m — Condon-Shortley sensitive
    ("S", 2, 1, -1.2e-5),   # tesseral, odd m, sine
    ("C", 2, 2, 1.3e-5),    # sectoral, even m
    ("S", 2, 2, -1.1e-5),   # sectoral, even m, sine
    ("C", 3, 1, 2.2e-6),    # odd degree, odd m
    ("S", 3, 3, 6.0e-7),    # odd sectoral, sine
    ("C", 4, 3, -8.0e-7),   # odd m tesseral, higher degree
)

# (lat_deg, lon_deg, alt_km): equator, mid-lat both hemispheres, near-pole
# both hemispheres, several longitudes, two altitudes.
SAMPLE_POINTS = (
    (0.0, 0.0, 100.0),
    (0.0, 135.0, 100.0),
    (45.0, 60.0, 100.0),
    (-30.0, 250.0, 100.0),
    (89.95, 15.0, 100.0),
    (-89.95, 200.0, 100.0),
    (45.0, 60.0, 1500.0),
    (-60.0, 310.0, 1500.0),
)

DEGREE = 4


def _position(lat_deg: float, lon_deg: float, alt_km: float) -> np.ndarray:
    r = R_REF + alt_km * 1000.0
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    return np.array(
        [
            r * math.cos(lat) * math.cos(lon),
            r * math.cos(lat) * math.sin(lon),
            r * math.sin(lat),
        ],
        dtype=np.float64,
    )


def _single_term_model(kind: str, n: int, m: int, value: float) -> GravityModel:
    C = np.zeros((DEGREE + 1, DEGREE + 1), dtype=np.float64)
    S = np.zeros_like(C)
    (C if kind == "C" else S)[n, m] = value
    return GravityModel.from_arrays(DEGREE, R_REF, GM, C, S)


def _rel_vec_err(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


# ---------------------------------------------------------------------------
# Potential / acceleration sign convention
# ---------------------------------------------------------------------------

def test_potential_positive_and_acceleration_inward_point_mass() -> None:
    """Lock the geodesy sign convention: U = +mu/r > 0 and a = +grad(U) points
    toward the body (a . r_hat < 0). A sign flip in either the potential or
    the gradient convention fails here before any parity test can."""
    model = _single_term_model("C", 2, 0, 0.0)  # all zeros -> pure monopole
    for lat, lon, alt in SAMPLE_POINTS:
        pos = _position(lat, lon, alt)
        r = float(np.linalg.norm(pos))
        V, a = model.potential_accel_fixed(pos, degree=DEGREE)
        assert V > 0.0, f"potential must be positive (geodesy +mu/r): V={V}"
        assert abs(V - GM / r) / (GM / r) < 1e-12
        r_hat = pos / r
        assert float(a @ r_hat) < 0.0, "acceleration must point inward"
        assert np.allclose(a, -GM / r**2 * r_hat, rtol=1e-12)


# ---------------------------------------------------------------------------
# Cross-path vector agreement, one coefficient at a time
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kind,n,m,value", SINGLE_TERMS,
                         ids=[f"{k}{n}{m}" for k, n, m, _ in SINGLE_TERMS])
def test_accel_and_potential_paths_agree_per_term(kind: str, n: int, m: int, value: float) -> None:
    """GravityModel.accel_fixed (recurrence kernel) vs sh_potential_accel_fixed
    (potential-path kernel) as full vectors, per single coefficient."""
    model = _single_term_model(kind, n, m, value)
    pts = np.stack([_position(*p) for p in SAMPLE_POINTS])

    a_recurrence = np.stack([model.accel_fixed(p, degree=DEGREE) for p in pts])
    _V, a_potential = sh_potential_accel_fixed(pts, model.Cnm, model.Snm, GM, R_REF, DEGREE)

    assert np.all(np.isfinite(a_recurrence))
    assert np.all(np.isfinite(a_potential))
    perturbation_seen = False
    for i, p in enumerate(SAMPLE_POINTS):
        err = _rel_vec_err(a_potential[i], a_recurrence[i])
        assert err < 1e-10, (
            f"term {kind}{n}{m}: potential-path vs recurrence-path acceleration "
            f"vector mismatch at {p}: rel={err:.3e}"
        )
        pm = -GM / np.linalg.norm(pts[i]) ** 2 * (pts[i] / np.linalg.norm(pts[i]))
        if np.linalg.norm(a_recurrence[i] - pm) > 0.0:
            perturbation_seen = True
    # The term must perturb the field at at least one sample point (guards
    # against a term silently evaluating to zero and the test passing
    # vacuously). Individual points may legitimately sit on nodal lines.
    assert perturbation_seen, f"term {kind}{n}{m} never perturbed the point-mass field"


@pytest.mark.parametrize("kind,n,m,value", SINGLE_TERMS,
                         ids=[f"{k}{n}{m}" for k, n, m, _ in SINGLE_TERMS])
def test_potential_finite_difference_gradient_per_term(kind: str, n: int, m: int, value: float) -> None:
    """a = +grad(U) numerically: central finite differences of the potential
    kernel's own U must reproduce its acceleration vector for each term."""
    model = _single_term_model(kind, n, m, value)
    # Off-pole subset: FD across the pole guard region is not meaningful.
    pts = [p for p in SAMPLE_POINTS if abs(p[0]) < 80.0]

    for lat, lon, alt in pts:
        pos = _position(lat, lon, alt)
        h = 3.0e-4 * float(np.linalg.norm(pos))
        grad = np.zeros(3)
        for k in range(3):
            dp = np.zeros(3)
            dp[k] = h
            V_plus, _ = sh_potential_accel_fixed(
                (pos + dp).reshape(1, 3), model.Cnm, model.Snm, GM, R_REF, DEGREE)
            V_minus, _ = sh_potential_accel_fixed(
                (pos - dp).reshape(1, 3), model.Cnm, model.Snm, GM, R_REF, DEGREE)
            grad[k] = (float(V_plus[0]) - float(V_minus[0])) / (2.0 * h)
        _V, a = sh_potential_accel_fixed(
            pos.reshape(1, 3), model.Cnm, model.Snm, GM, R_REF, DEGREE)
        err = _rel_vec_err(grad, a[0])
        assert err < 1e-6, (
            f"term {kind}{n}{m}: FD gradient of U disagrees with the analytic "
            f"acceleration at ({lat},{lon},{alt}): rel={err:.3e} "
            "(sign convention a=+grad(U) may have flipped)"
        )


@pytest.mark.parametrize("kind,n,m,value", SINGLE_TERMS,
                         ids=[f"{k}{n}{m}" for k, n, m, _ in SINGLE_TERMS])
def test_torch_sh_matches_cpu_per_term(kind: str, n: int, m: int, value: float) -> None:
    """TorchSHGravityEvaluator (float64, CPU) vs GravityModel.accel_fixed as
    full vectors, per single coefficient. Catches any torch-side phase or
    normalization drift independently of combined-field parity tests."""
    torch = pytest.importorskip("torch")
    from lunaris.physics.torch_spherical_harmonics import TorchSHGravityEvaluator

    model = _single_term_model(kind, n, m, value)
    evaluator = TorchSHGravityEvaluator(
        model, degree=DEGREE, device=torch.device("cpu"), dtype=torch.float64)

    pts = np.stack([_position(*p) for p in SAMPLE_POINTS])
    a_torch = evaluator.acceleration(torch.from_numpy(pts)).cpu().numpy()
    a_cpu = np.stack([model.accel_fixed(p, degree=DEGREE) for p in pts])

    assert np.all(np.isfinite(a_torch)), f"term {kind}{n}{m}: non-finite torch SH output"
    for i, p in enumerate(SAMPLE_POINTS):
        err = _rel_vec_err(a_torch[i], a_cpu[i])
        assert err < 1e-12, (
            f"term {kind}{n}{m}: torch SH vs CPU SH acceleration vector "
            f"mismatch at {p}: rel={err:.3e}"
        )


def test_near_pole_outputs_finite_all_paths() -> None:
    """Exactly-polar and near-polar inputs must stay finite on the potential
    path and the torch path (the accel kernel is covered in
    test_spherical_harmonics.py::test_pole_robustness)."""
    model = _single_term_model("C", 2, 1, 1.5e-5)  # tesseral: worst case at pole
    polar_pts = np.array(
        [
            [0.0, 0.0, R_REF + 200e3],
            [0.0, 0.0, -(R_REF + 200e3)],
            [1e-6, -1e-6, R_REF + 200e3],
            [1e-6, 1e-6, -(R_REF + 200e3)],
        ],
        dtype=np.float64,
    )
    V, a = sh_potential_accel_fixed(polar_pts, model.Cnm, model.Snm, GM, R_REF, DEGREE)
    assert np.all(np.isfinite(V)) and np.all(np.isfinite(a))

    torch = pytest.importorskip("torch")
    from lunaris.physics.torch_spherical_harmonics import TorchSHGravityEvaluator

    evaluator = TorchSHGravityEvaluator(
        model, degree=DEGREE, device=torch.device("cpu"), dtype=torch.float64)
    a_t = evaluator.acceleration(torch.from_numpy(polar_pts)).cpu().numpy()
    assert np.all(np.isfinite(a_t))
