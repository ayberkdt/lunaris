"""Direction-accuracy fixes: relative acceleration loss, worse-than-zero metric,
radial output envelope, and their contract/selection plumbing."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from lunaris.surrogate.st_lrps.networks.models import (
    ARCH_SIGNATURE_FIELDS,
    PhysicsNet,
    architecture_fingerprint,
    compute_architecture_signature,
)
from lunaris.surrogate.st_lrps.training.losses import relative_accel_error_terms
from lunaris.surrogate.st_lrps.training.metrics import (
    HISTORY_FIELDNAMES,
    SUPPORTED_BEST_METRICS,
    checkpoint_formula,
    compute_checkpoint_score,
    flatten_epoch_metrics,
)

# --------------------------------------------------------------------------
# relative_accel_error_terms
# --------------------------------------------------------------------------

def test_relative_terms_reference_values():
    a = torch.tensor([[1.0, 0.0, 0.0]] * 4, dtype=torch.float64)
    pred = torch.stack([
        a[0],                    # exact
        torch.zeros(3, dtype=torch.float64),  # predicts nothing
        2.0 * a[0],              # right direction, double magnitude
        -a[0],                   # anti-parallel
    ])
    rel_sq, rel_err, wtz = relative_accel_error_terms(pred - a, a, floor_abs=0.0)
    assert rel_sq.tolist() == pytest.approx([0.0, 1.0, 1.0, 4.0])
    assert rel_err.tolist() == pytest.approx([0.0, 1.0, 1.0, 2.0])
    # Only the anti-parallel guess is strictly worse than predicting zero;
    # zero and double-magnitude sit exactly on the boundary.
    assert wtz.tolist() == [False, False, False, True]


def test_relative_terms_floor_bounds_tiny_targets():
    a = torch.tensor([[1e-12, 0.0, 0.0]], dtype=torch.float64)
    pred = torch.tensor([[1e-9, 0.0, 0.0]], dtype=torch.float64)
    rel_sq, rel_err, wtz = relative_accel_error_terms(pred - a, a, floor_abs=1e-8)
    # ||err||~1e-9 against floor 1e-8 -> ratio ~0.1, not 1e3.
    assert rel_sq.item() == pytest.approx(1e-2, rel=1e-2)
    assert rel_err.item() == pytest.approx(0.1, rel=1e-2)
    assert bool(wtz.item()) is True


def test_relative_terms_are_differentiable():
    a = torch.randn(16, 3, dtype=torch.float64)
    pred = (a + 0.1 * torch.randn(16, 3, dtype=torch.float64)).requires_grad_(True)
    rel_sq, _, _ = relative_accel_error_terms(pred - a, a, floor_abs=1e-3)
    rel_sq.mean().backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()


# --------------------------------------------------------------------------
# PhysicsNet output envelope
# --------------------------------------------------------------------------

class _Const(torch.nn.Module):
    def forward(self, x):  # noqa: D401 - trivial backbone
        return torch.ones(x.shape[0], 1, dtype=x.dtype, device=x.device)


def test_envelope_off_is_passthrough():
    net = PhysicsNet(_Const())
    x = torch.randn(5, 3, dtype=torch.float64)
    assert torch.equal(net(x), torch.ones(5, 1, dtype=torch.float64))
    assert "output_envelope_power" not in architecture_fingerprint({"hidden": 8})


def test_envelope_matches_rho_power_and_decays():
    r_ref, x_scale, p = 1738000.0, 2738000.0, 26.0
    net = PhysicsNet(_Const(), envelope_power=p, x_scale_m=x_scale, r_ref_m=r_ref)
    radii_m = torch.tensor([1838000.0, 2238000.0, 2738000.0], dtype=torch.float64)
    x = torch.zeros(3, 3, dtype=torch.float64)
    x[:, 0] = radii_m / x_scale
    out = net(x).squeeze(-1)
    expected = (r_ref / radii_m) ** p
    assert out.tolist() == pytest.approx(expected.tolist(), rel=1e-12)
    assert out[0] > out[1] > out[2]
    assert out[2].item() == pytest.approx((r_ref / x_scale) ** p, rel=1e-12)


def test_envelope_gradient_matches_finite_difference():
    r_ref, x_scale, p = 1738000.0, 2738000.0, 26.0
    net = PhysicsNet(_Const(), envelope_power=p, x_scale_m=x_scale, r_ref_m=r_ref)
    x = torch.tensor([[0.5, 0.4, 0.3]], dtype=torch.float64, requires_grad=True)
    out = net(x)
    grad = torch.autograd.grad(out.sum(), x)[0]
    h = 1e-6
    fd = torch.zeros_like(x)
    for k in range(3):
        e = torch.zeros_like(x)
        e[0, k] = h
        fd[0, k] = (net(x + e).sum() - net(x - e).sum()).item() / (2 * h)
    assert grad.detach()[0].tolist() == pytest.approx(fd[0].tolist(), rel=1e-6)


def test_envelope_rejects_bad_scales():
    with pytest.raises(ValueError):
        PhysicsNet(_Const(), envelope_power=26.0, x_scale_m=0.0, r_ref_m=1.0)
    with pytest.raises(ValueError):
        PhysicsNet(_Const(), envelope_power=-1.0)


# --------------------------------------------------------------------------
# Architecture signature: legacy hashes unchanged, active envelope changes it
# --------------------------------------------------------------------------

def test_signature_stable_for_legacy_configs_and_sensitive_to_envelope():
    base = {"activation": "sine", "hidden": 512, "depth": 5, "n_bands": 2, "degree_min": 25}
    legacy = compute_architecture_signature(base)
    assert compute_architecture_signature({**base, "output_envelope_power": 0.0}) == legacy
    assert compute_architecture_signature({**base, "output_envelope_power": None}) == legacy
    assert compute_architecture_signature({**base, "output_envelope_power": 26.0}) != legacy
    assert "output_envelope_power" in ARCH_SIGNATURE_FIELDS


# --------------------------------------------------------------------------
# Evaluation angle must be scale-free (tiny residuals used to read as ~60-90 deg)
# --------------------------------------------------------------------------

def test_vector_angle_is_scale_free_for_tiny_residuals():
    import numpy as np

    from lunaris.surrogate.st_lrps.evaluation.cli import vector_cosine_and_angle_deg

    a_true = np.array([[5e-10, 0.0, 0.0]], dtype=np.float32)
    a_pred = np.array([[5e-10, 2e-11, 0.0]], dtype=np.float32)   # 4% error, ~2.3 deg
    cos, ang = vector_cosine_and_angle_deg(a_pred, a_true)
    assert ang[0] == pytest.approx(2.29, abs=0.05)
    assert cos[0] == pytest.approx(0.9992, abs=1e-3)
    # Same geometry, 1e6 times larger: identical answer.
    cos2, ang2 = vector_cosine_and_angle_deg(a_pred * 1e6, a_true * 1e6)
    assert ang2[0] == pytest.approx(ang[0], rel=1e-6)
    # Degenerate cases stay finite.
    z = np.zeros((1, 3))
    _, ang0 = vector_cosine_and_angle_deg(z, z)
    assert np.isfinite(ang0).all()


# --------------------------------------------------------------------------
# Checkpoint selection + history plumbing
# --------------------------------------------------------------------------

def _val_stats():
    return {
        "val_total_loss": 1.20, "val_base_loss": 1.00, "val_physics_loss": 0.20,
        "mse_u": 0.40, "mse_a": 0.60, "loss_dir": 0.50, "loss_rel": 2.0,
        "rel_a_mean": 0.7, "wtz_frac": 0.25, "mask_frac": 0.3,
    }


def test_hybrid_rel_score_and_formula():
    cfg = SimpleNamespace(best_metric="hybrid_rel", hybrid_direction_alpha=0.30, hybrid_relative_beta=0.10)
    score, report = compute_checkpoint_score(_val_stats(), cfg)
    assert "hybrid_rel" in SUPPORTED_BEST_METRICS
    assert score == pytest.approx(1.00 + 0.30 * 0.50 + 0.10 * 2.0)
    assert report["val_loss_rel"] == pytest.approx(2.0)
    assert checkpoint_formula("hybrid_rel", cfg) == "val_base_loss + 0.30 * val_loss_dir + 0.10 * val_loss_rel"
    # Plain hybrid must be unaffected by the new term.
    score_h, _ = compute_checkpoint_score(_val_stats(), SimpleNamespace(best_metric="hybrid", hybrid_direction_alpha=0.30))
    assert score_h == pytest.approx(1.15)


def test_history_row_carries_direction_diagnostics():
    cfg = SimpleNamespace(best_metric="hybrid", hybrid_direction_alpha=0.30)
    row = flatten_epoch_metrics(3, _val_stats(), _val_stats(), {"score": 1.0}, cfg)
    for key in ("val_loss_rel", "val_rel_a_mean", "val_wtz_frac", "val_mask_frac",
                "train_loss_rel", "train_wtz_frac"):
        assert key in HISTORY_FIELDNAMES
        assert row[key] is not None
    assert row["val_wtz_frac"] == pytest.approx(0.25)
    assert row["val_mask_frac"] == pytest.approx(0.3)
