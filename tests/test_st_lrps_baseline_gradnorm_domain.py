"""Review fixes: single-pass SH baseline (Risk 3), representative GradNorm
parameter set (Risk 4), and orbit-drift domain tracking (Risk 5)."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from lunaris.common.lunar_data import MU_MOON_SI, R_MOON_SI
from lunaris.surrogate.st_lrps.shared import scaling
from lunaris.surrogate.st_lrps.shared.contracts import TargetContract
from lunaris.surrogate.st_lrps.shared.scaling import (
    compute_base_accel_from_contract,
    compute_base_potential_accel_from_contract,
    compute_base_potential_from_contract,
)

# --- Risk 3: SH baseline computed once -------------------------------------

def _full_sh_contract() -> TargetContract:
    return TargetContract(
        central_body="moon",
        target_mode="full",
        base_degree=4,
        target_degree=8,
        baseline_kind="spherical_harmonics",
        unit_system="si",
        frame="moon_fixed_cartesian",
        derivative_convention_version="dP_dphi_corrected_v1",
        a_sign=1.0,
        mu_si=MU_MOON_SI,
        r_ref_m=R_MOON_SI,
    )


def test_combined_baseline_matches_point_mass_singles():
    contract = TargetContract(
        central_body="moon", target_mode="full", base_degree=0, target_degree=0,
        baseline_kind="point_mass", unit_system="si", frame="moon_fixed_cartesian",
        derivative_convention_version="dP_dphi_corrected_v1", a_sign=1.0,
        mu_si=MU_MOON_SI, r_ref_m=R_MOON_SI,
    )
    x = torch.tensor([[R_MOON_SI + 1e5, 0.0, 0.0], [0.0, R_MOON_SI + 2e5, 0.0]], dtype=torch.float64)
    u, a = compute_base_potential_accel_from_contract(x, contract)
    assert torch.allclose(u, compute_base_potential_from_contract(x, contract))
    assert torch.allclose(a, compute_base_accel_from_contract(x, contract))


def test_combined_baseline_residual_is_zero():
    contract = TargetContract(
        central_body="moon", target_mode="residual", base_degree=20, target_degree=60,
        baseline_kind="spherical_harmonics", unit_system="si", frame="moon_fixed_cartesian",
        derivative_convention_version="dP_dphi_corrected_v1", a_sign=1.0,
        mu_si=MU_MOON_SI, r_ref_m=R_MOON_SI,
    )
    x = torch.randn(5, 3, dtype=torch.float64) * 1e5 + R_MOON_SI
    u, a = compute_base_potential_accel_from_contract(x, contract)
    assert torch.count_nonzero(u) == 0 and torch.count_nonzero(a) == 0


def test_sh_baseline_evaluated_once_combined_vs_twice_separate(monkeypatch):
    contract = _full_sh_contract()
    x = torch.zeros(3, 3, dtype=torch.float64)
    x[:, 0] = R_MOON_SI + 1e5

    calls = {"n": 0}

    def _counting_sh(x_phys, _contract, _gm):
        calls["n"] += 1
        n = x_phys.shape[0]
        return (
            torch.ones((n, 1), dtype=x_phys.dtype),
            torch.ones((n, 3), dtype=x_phys.dtype),
        )

    monkeypatch.setattr(scaling, "_sh_baseline_field", _counting_sh)

    calls["n"] = 0
    compute_base_potential_accel_from_contract(x, contract, gravity_model=object())
    assert calls["n"] == 1  # the combined helper evaluates the SH field once

    calls["n"] = 0
    compute_base_potential_from_contract(x, contract, gravity_model=object())
    compute_base_accel_from_contract(x, contract, gravity_model=object())
    assert calls["n"] == 2  # separate calls cost two passes — what the hot paths avoid


# --- Risk 4: representative GradNorm parameter set --------------------------

def _multiscale_model():
    from lunaris.surrogate.st_lrps.networks.models import build_model_from_config

    cfg = {
        "activation": "sine",
        "hidden": 48,
        "depth": 4,
        "n_bands": 3,
        "degree_min": 10,
        "degree_max": 60,
        "use_residual_blocks": True,
        "runtime_model_kind": "potential_autograd",
        "output_dim": 1,
    }
    return build_model_from_config(cfg, device=torch.device("cpu"), dtype=torch.float32)


def test_backbone_gradnorm_set_excludes_head_and_covers_more_than_one_layer():
    from lunaris.surrogate.st_lrps.training.losses import (
        _get_backbone_shared_params,
        _get_last_hidden_params,
        _gradnorm_shared_params,
    )

    model = _multiscale_model()
    linears = [m for m in model.modules() if isinstance(m, torch.nn.Linear)]
    head_param_ids = {id(p) for p in linears[-1].parameters()}

    backbone = _get_backbone_shared_params(model)
    backbone_ids = {id(p) for p in backbone}
    # Output head is excluded; the set is far larger than the single-layer proxy.
    assert head_param_ids.isdisjoint(backbone_ids)
    assert len(backbone) > len(_get_last_hidden_params(model))

    # ntk_init -> broad backbone set; dynamic -> cheap single-layer proxy.
    assert len(_gradnorm_shared_params(model, "ntk_init")) == len(backbone)
    assert len(_gradnorm_shared_params(model, "dynamic")) == len(_get_last_hidden_params(model))


def test_backbone_gradnorm_params_receive_gradient_from_both_losses():
    from lunaris.surrogate.st_lrps.training.losses import _get_backbone_shared_params

    model = _multiscale_model()
    params = _get_backbone_shared_params(model)
    x = (torch.randn(16, 3) * 0.3).requires_grad_(True)
    u = model(x)
    grads = torch.autograd.grad(u.sum(), params, allow_unused=True, retain_graph=True)
    assert any(g is not None for g in grads)


# --- Risk 5: orbit-drift domain extrapolation tracking ---------------------

class _StubRuntime:
    """Minimal runtime: extrapolates beyond a radius threshold."""

    def __init__(self, threshold_m: float):
        self.threshold_m = float(threshold_m)

    def domain_status(self, r):
        r = np.asarray(r, dtype=np.float64).reshape(-1, 3)
        rn = np.linalg.norm(r, axis=1)
        return {"recommended_fallback": bool(np.any(rn > self.threshold_m)), "reason": "test"}

    def predict_residual_accel_fixed(self, r):
        r = np.asarray(r, dtype=np.float64).reshape(-1, 3)
        return np.zeros_like(r)


def test_domain_tracking_counts_extrapolation():
    from lunaris.surrogate.st_lrps.evaluation.orbit_drift import _DomainTrackingTotalAccel

    rt = _StubRuntime(threshold_m=R_MOON_SI + 150e3)
    acc = _DomainTrackingTotalAccel(rt, MU_MOON_SI, strict_domain=False)

    acc(np.array([R_MOON_SI + 100e3, 0.0, 0.0]))  # inside
    acc(np.array([R_MOON_SI + 300e3, 0.0, 0.0]))  # outside

    rep = acc.domain_report()
    assert rep["evaluations"] == 2
    assert rep["extrapolating_evaluations"] == 1
    assert rep["extrapolation_fraction"] == pytest.approx(0.5)


def test_domain_tracking_strict_raises_on_extrapolation():
    from lunaris.surrogate.st_lrps.evaluation.orbit_drift import _DomainTrackingTotalAccel

    rt = _StubRuntime(threshold_m=R_MOON_SI + 150e3)
    acc = _DomainTrackingTotalAccel(rt, MU_MOON_SI, strict_domain=True)
    # Inside the domain integrates fine.
    out = acc(np.array([R_MOON_SI + 100e3, 0.0, 0.0]))
    assert np.all(np.isfinite(out))
    # Outside the domain hard-fails.
    with pytest.raises(RuntimeError):
        acc(np.array([R_MOON_SI + 300e3, 0.0, 0.0]))
