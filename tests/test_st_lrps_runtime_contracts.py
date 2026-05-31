# tests/test_st_lrps_runtime_contracts.py
# -*- coding: utf-8 -*-
"""
CPU-only runtime-contract tests for ``lunaris.surrogate.st_lrps.runtime.force_model``.

No trained checkpoint is required: we wrap an *analytically known* potential
``U(x_scaled) = sum(x_scaled^2)`` in a tiny ``torch.nn.Module`` and an identity-ish
``ScalerPack``. With unit scaling, ``Delta_a = a_sign * grad(U) = 2*x``, which lets
us assert the autograd acceleration path exactly.

These tests guard the surrogate-runtime contracts that, if broken, would feed a
propagator physically wrong (but error-free) accelerations:

- output shapes/types for single point and batch,
- finite-input rejection (NaN / Inf),
- domain reporting + ``strict_domain`` hard-fail outside the trained shell,
- the ``force_direct`` runtime placeholder failing loudly,
- the residual SH-baseline requiring an explicit ``base_accel_fn``,
- the point-mass fallback only firing when the target contract allows it.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

from lunaris.surrogate.st_lrps.runtime.force_model import (  # noqa: E402
    DirectForceRuntime,
    SurrogateForceModel,
)
from lunaris.surrogate.st_lrps.shared.scaling import (  # noqa: E402
    IsometricScaleParams,
    ScalerPack,
)
from lunaris.surrogate.st_lrps.data.dataset_parameters import (  # noqa: E402
    MU_MOON_SI,
    R_MOON_SI,
)

pytestmark = pytest.mark.requires_torch


class _ToyPotential(nn.Module):
    """U(x) = sum_i x_i^2 -> grad = 2x (analytically known)."""

    def forward(self, x):  # x: (N, 3)
        return (x * x).sum(dim=1, keepdim=True)


def _scaler(x_scale: float = 1.0) -> ScalerPack:
    return ScalerPack(
        x=IsometricScaleParams(mean=[0.0, 0.0, 0.0], scale=float(x_scale)),
        u=IsometricScaleParams(mean=[0.0], scale=1.0),
        a=IsometricScaleParams(mean=[0.0, 0.0, 0.0], scale=1.0),
    )


def _model(cfg=None, *, x_scale=1.0, strict_domain=False) -> SurrogateForceModel:
    return SurrogateForceModel(
        model=_ToyPotential(),
        scaler=_scaler(x_scale),
        cfg=dict(cfg or {}),
        device=torch.device("cpu"),
        strict_domain=strict_domain,
    )


# =============================================================================
# Shapes & types
# =============================================================================

def test_residual_potential_shape_and_type():
    fm = _model()
    # Single point -> python float.
    u = fm.predict_residual_potential(np.array([1.0, 2.0, 3.0]))
    assert isinstance(u, float)
    assert u == pytest.approx(14.0)  # 1 + 4 + 9

    # Batch -> (N,) array.
    ub = fm.predict_residual_potential(np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]))
    assert isinstance(ub, np.ndarray) and ub.shape == (2,)
    np.testing.assert_allclose(ub, [1.0, 4.0])


def test_residual_accel_shapes_and_known_gradient():
    fm = _model()
    # grad(sum x^2) = 2x; unit scaling -> Delta_a = 2x.
    da = fm.predict_residual_accel(np.array([1.0, 2.0, 3.0]))
    assert da.shape == (3,)
    np.testing.assert_allclose(da, [2.0, 4.0, 6.0], rtol=1e-5)

    dab = fm.predict_residual_accel(np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]))
    assert dab.shape == (2, 3)
    np.testing.assert_allclose(dab, [[2.0, 0.0, 0.0], [0.0, 4.0, 0.0]], rtol=1e-5)


# =============================================================================
# Finite-input validation
# =============================================================================

@pytest.mark.parametrize("bad", [
    np.array([np.nan, 0.0, 0.0]),
    np.array([np.inf, 0.0, 0.0]),
    np.array([0.0, -np.inf, 0.0]),
])
def test_nonfinite_inputs_are_rejected(bad):
    fm = _model()
    with pytest.raises(ValueError, match="finite|NaN|Inf"):
        fm.predict_residual_potential(bad)
    with pytest.raises(ValueError, match="finite|NaN|Inf"):
        fm.predict_residual_accel(bad)
    with pytest.raises(ValueError, match="finite|NaN|Inf"):
        fm.predict_total_accel(bad)


# =============================================================================
# Domain status & strict-domain enforcement
# =============================================================================

def test_domain_status_reports_altitude_and_training_range():
    # Scaler radius large enough that real lunar positions don't trip the scaler
    # check, and explicit training altitude bounds [100, 500] km.
    fm = _model(cfg={"altitude_min_km": 100.0, "altitude_max_km": 500.0},
                x_scale=R_MOON_SI + 600e3)

    in_band = fm.domain_status(np.array([R_MOON_SI + 300e3, 0.0, 0.0]))
    assert in_band["finite_input"] is True
    assert in_band["altitude_km_min"] == pytest.approx(300.0, abs=1.0)
    assert in_band["altitude_km_max"] == pytest.approx(300.0, abs=1.0)
    assert in_band["in_training_altitude_range"] is True
    assert in_band["recommended_fallback"] is False

    out_band = fm.domain_status(np.array([R_MOON_SI + 1_000e3, 0.0, 0.0]))
    assert out_band["in_training_altitude_range"] is False
    assert out_band["recommended_fallback"] is True
    assert "training range" in out_band["reason"]


def test_domain_status_flags_scaler_radius_extrapolation():
    # Tiny scaler radius (1 m) -> any real position is far outside the trained shell.
    fm = _model(x_scale=1.0)
    status = fm.domain_status(np.array([R_MOON_SI + 100e3, 0.0, 0.0]))
    assert status["exceeds_scaler_radius"] is True
    assert status["recommended_fallback"] is True


def test_strict_domain_raises_outside_trained_domain():
    fm = _model(x_scale=1.0, strict_domain=True)
    with pytest.raises(RuntimeError, match="strict_domain"):
        fm.predict_residual_accel(np.array([R_MOON_SI + 100e3, 0.0, 0.0]))


def test_non_strict_domain_still_returns_prediction():
    # Default (strict_domain=False) must extrapolate (with a one-time warning),
    # not raise. The numeric value is irrelevant here; the contract is "no raise".
    fm = _model(x_scale=1.0, strict_domain=False)
    out = fm.predict_residual_accel(np.array([R_MOON_SI + 100e3, 0.0, 0.0]))
    assert out.shape == (3,)
    assert np.all(np.isfinite(out))


# =============================================================================
# Runtime-kind & baseline contracts
# =============================================================================

def test_force_direct_runtime_is_not_implemented():
    with pytest.raises(NotImplementedError, match="force_direct"):
        DirectForceRuntime()


def test_residual_sh_baseline_requires_base_accel_fn():
    # residual / SH baseline through degree 20 -> point-mass fallback is NOT valid.
    fm = _model(cfg={"target_mode": "residual", "degree_min": 20, "degree_max": 100},
                x_scale=R_MOON_SI + 600e3)
    assert fm.target_contract.target_mode == "residual"
    assert fm.target_contract.baseline_kind == "spherical_harmonics"

    x = np.array([R_MOON_SI + 300e3, 0.0, 0.0])  # inside scaler domain
    with pytest.raises(ValueError, match="base_accel_fn"):
        fm.predict_total_accel(x)  # no baseline provided

    # Providing the baseline makes it work and returns the right shape.
    base_fn = lambda arr: np.zeros((np.asarray(arr).reshape(-1, 3).shape[0], 3))
    out = fm.predict_total_accel(x, base_fn)
    assert np.asarray(out).shape == (3,)


def test_point_mass_fallback_only_when_contract_allows():
    # Full-field / point-mass contract (empty cfg) -> point-mass baseline is allowed.
    fm = _model(x_scale=R_MOON_SI + 600e3)
    assert fm.target_contract.baseline_kind == "point_mass"

    x = np.array([R_MOON_SI + 300e3, 0.0, 0.0])
    total = fm.predict_total_accel(x)            # no base_accel_fn -> point-mass fallback
    residual = fm.predict_residual_accel(x)
    r = np.linalg.norm(x)
    expected_base = -fm.mu_si * x / r ** 3
    np.testing.assert_allclose(total, expected_base + residual, rtol=1e-5)
