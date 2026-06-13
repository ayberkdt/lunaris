# -*- coding: utf-8 -*-
"""
Torch spherical-harmonic evaluator: high-degree construction + canonicalization.

Validates (task §17 / §5), CPU-only with torch:

* the canonical TorchSHGravityEvaluator constructs at SH25 / SH100 / SH200 with
  no hard-coded degree-24 error;
* a small CPU batch yields finite accelerations;
* a zero-coefficient (point-mass) model matches the analytic Keplerian field
  (the canonical low-fidelity reference) within tolerance;
* the benchmark copy of the evaluator is numerically identical to the canonical
  physics evaluator (justifies the §5 canonicalization).
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from lunaris.physics.spherical_harmonics import build_legendre_coeffs  # noqa: E402
from lunaris.physics.torch_spherical_harmonics import (  # noqa: E402
    TorchSHGravityEvaluator as CanonicalEvaluator,
)

R_REF = 1.738e6
GM = 4.9028e12


class _SyntheticGravityModel:
    """Minimal object satisfying the normalized-gravity contract the evaluator reads."""

    def __init__(self, degree: int, *, seed: int = 0, zero_coeffs: bool = False) -> None:
        n = int(degree)
        self.R_ref_m = R_REF
        self.GM_m3s2 = GM
        rng = np.random.default_rng(seed)
        C = np.zeros((n + 1, n + 1), dtype=np.float64)
        S = np.zeros((n + 1, n + 1), dtype=np.float64)
        if not zero_coeffs:
            # Small normalized coefficients for n >= 2 (m <= n).
            for nn in range(2, n + 1):
                for mm in range(0, nn + 1):
                    C[nn, mm] = 1e-4 * rng.standard_normal() / (nn * nn)
                    if mm > 0:
                        S[nn, mm] = 1e-4 * rng.standard_normal() / (nn * nn)
        self.Cnm = C
        self.Snm = S
        diag, subdiag, A, B, scale_m = build_legendre_coeffs(n)
        self.diag = diag
        self.subdiag = subdiag
        self.A = A
        self.B = B
        self.scale_m = scale_m


def _sample_positions() -> "torch.Tensor":
    pts = np.array(
        [
            [R_REF + 1.0e5, 0.0, 0.0],
            [0.0, R_REF + 1.2e5, 0.0],
            [0.0, 0.0, R_REF + 1.5e5],
            [1.0e6, 1.0e6, 8.0e5],
        ],
        dtype=np.float64,
    )
    return torch.as_tensor(pts, dtype=torch.float64)


@pytest.mark.parametrize("degree", [25, 100, 200])
def test_evaluator_constructs_high_degree_and_is_finite(degree: int) -> None:
    model = _SyntheticGravityModel(degree, seed=degree)
    ev = CanonicalEvaluator(model, degree=degree, device=torch.device("cpu"), dtype=torch.float64)
    accel = ev.acceleration(_sample_positions())
    assert accel.shape == (4, 3)
    assert torch.isfinite(accel).all()


def test_point_mass_matches_keplerian() -> None:
    # Zero coefficients -> pure central gravity -mu * r / |r|^3 (canonical low-fidelity ref).
    degree = 50
    model = _SyntheticGravityModel(degree, zero_coeffs=True)
    ev = CanonicalEvaluator(model, degree=degree, device=torch.device("cpu"), dtype=torch.float64)
    pos = _sample_positions()
    accel = ev.acceleration(pos).numpy()
    r = pos.numpy()
    norm = np.linalg.norm(r, axis=1, keepdims=True)
    expected = -GM * r / norm**3
    np.testing.assert_allclose(accel, expected, rtol=1e-6, atol=1e-9)


def test_benchmark_evaluator_matches_canonical() -> None:
    # §5: the benchmark copy must be numerically identical to the canonical one,
    # which justifies importing the canonical evaluator in the benchmark.
    from lunaris.surrogate.st_lrps.evaluation._gravity_benchmark.compute import (
        TorchSHGravityEvaluator as BenchmarkEvaluator,
    )

    degree = 60
    model = _SyntheticGravityModel(degree, seed=7)
    pos = _sample_positions()
    canon = CanonicalEvaluator(model, degree=degree, device=torch.device("cpu"), dtype=torch.float64)
    bench = BenchmarkEvaluator(model, degree=degree, device=torch.device("cpu"), dtype=torch.float64)
    a_canon = canon.acceleration(pos)
    a_bench = bench.acceleration(pos)
    assert torch.allclose(a_canon, a_bench, rtol=0.0, atol=0.0), "benchmark evaluator diverged from canonical"
