"""Unit tests for ST-LRPS hardware-independent compute accounting.

The FLOP convention is pinned against a hand-computed analytic count on a
Linear-only MLP, so the measurement cannot silently drift. The potential_autograd
acceleration eval includes the ∇ΔU gradient pass, asserted as a strict
inequality against the bare forward rather than a magic number.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

from lunaris.surrogate.st_lrps.training.compute_accounting import (  # noqa: E402
    CLASSICAL_SH_BASE_FLOPS,
    CLASSICAL_SH_FLOPS_PER_TERM,
    PFLOP_S_DAY_IN_FLOPS,
    ComputeAccounting,
    build_compute_accounting,
    build_compute_speed_section,
    classical_sh_flops_per_eval,
    classical_sh_terms,
    compare_eval_cost,
    flops_to_human,
    lookup_device_peak_flops,
    measure_forward_flops_per_sample,
    measure_inference_flops_per_eval,
    measure_train_step_flops_per_sample,
    pflops_days,
    render_compute_report,
)


def _linear_only_mlp(in_dim: int, hidden: int, out_dim: int, n_hidden: int) -> nn.Module:
    """An MLP whose only FLOP-bearing ops are Linear layers (ReLU is elementwise)."""
    layers: list[nn.Module] = [nn.Linear(in_dim, hidden), nn.ReLU()]
    for _ in range(n_hidden):
        layers += [nn.Linear(hidden, hidden), nn.ReLU()]
    layers.append(nn.Linear(hidden, out_dim))
    return nn.Sequential(*layers)


def _analytic_forward_flops_per_sample(in_dim, hidden, out_dim, n_hidden) -> int:
    # Each Linear(a,b) costs 2*a*b FLOP per sample (multiply-add = 2).
    total = 2 * in_dim * hidden
    total += n_hidden * (2 * hidden * hidden)
    total += 2 * hidden * out_dim
    return total


def test_forward_flops_match_analytic_linear_count():
    in_dim, hidden, out_dim, n_hidden, batch = 3, 32, 1, 2, 8
    model = _linear_only_mlp(in_dim, hidden, out_dim, n_hidden)
    x = torch.randn(batch, in_dim)

    measured = measure_forward_flops_per_sample(model, x)
    analytic = _analytic_forward_flops_per_sample(in_dim, hidden, out_dim, n_hidden)
    assert measured == pytest.approx(analytic, rel=0, abs=0)


def test_pflops_days_conversion_is_openai_unit():
    # 1 PF-day == 1e15 FLOP/s * 86400 s == 8.64e19 FLOP.
    assert PFLOP_S_DAY_IN_FLOPS == pytest.approx(8.64e19)
    assert pflops_days(PFLOP_S_DAY_IN_FLOPS) == pytest.approx(1.0)
    assert pflops_days(0.0) == 0.0


def test_total_training_flops_scale_with_samples():
    model = _linear_only_mlp(3, 16, 3, 1)
    x = torch.randn(4, 3)
    acc = build_compute_accounting(
        model, x, model_kind="potential_autograd", total_samples_processed=1_000_000
    )
    assert acc.total_training_flops == pytest.approx(
        acc.train_step_flops_per_sample * 1_000_000
    )
    assert acc.total_training_pflops_days == pytest.approx(
        acc.total_training_flops / PFLOP_S_DAY_IN_FLOPS
    )
    # Zero samples -> zero portable compute, no division blow-ups.
    acc0 = build_compute_accounting(
        model, x, model_kind="potential_autograd", total_samples_processed=0
    )
    assert acc0.total_training_flops == 0.0
    assert acc0.total_training_pflops_days == 0.0


def test_train_step_costs_more_than_inference():
    model = _linear_only_mlp(3, 32, 3, 2)
    x = torch.randn(8, 3)
    infer = measure_inference_flops_per_eval(model, x, model_kind="potential_autograd")
    step = measure_train_step_flops_per_sample(model, x, model_kind="potential_autograd")
    # A backward pass adds matmul work on top of the forward.
    assert step > infer > 0.0


def test_potential_autograd_inference_includes_gradient_pass():
    # A scalar-output net evaluated as potential vs as a direct vector output:
    # the potential path must do strictly more work (it adds the ∇U autograd pass).
    model = _linear_only_mlp(3, 32, 1, 2)
    x = torch.randn(8, 3)
    pot_infer = measure_inference_flops_per_eval(model, x, model_kind="potential_autograd")
    fwd = measure_forward_flops_per_sample(model, x)
    assert pot_infer > fwd > 0.0


def test_build_accounting_records_hardware_context_separately():
    model = _linear_only_mlp(3, 16, 1, 1)
    x = torch.randn(4, 3)
    acc = build_compute_accounting(
        model,
        x,
        model_kind="potential_autograd",
        total_samples_processed=500_000,
        wall_clock_seconds=120.0,
        device="cpu-test",
    )
    d = acc.to_manifest_dict()
    assert d["schema_version"] == "st_lrps_compute_accounting_v1"
    assert d["model_kind"] == "potential_autograd"
    # Portable numbers at the top level; machine-dependent context quarantined.
    assert "total_training_pflops_days" in d
    assert d["hardware"]["device"] == "cpu-test"
    assert d["hardware"]["wall_clock_seconds"] == pytest.approx(120.0)
    expected_achieved = acc.total_training_flops / 120.0
    assert d["hardware"]["achieved_flops_per_s"] == pytest.approx(expected_achieved)


def test_n_params_counted():
    model = _linear_only_mlp(3, 16, 1, 1)
    x = torch.randn(2, 3)
    acc = build_compute_accounting(
        model, x, model_kind="potential_autograd", total_samples_processed=1
    )
    expected = sum(p.numel() for p in model.parameters())
    assert acc.n_params == expected


def test_eval_mode_restored_and_weights_unchanged():
    model = _linear_only_mlp(3, 16, 3, 1)
    model.train()
    before = [p.detach().clone() for p in model.parameters()]
    x = torch.randn(4, 3)
    measure_train_step_flops_per_sample(model, x, model_kind="potential_autograd")
    # Measurement must not flip the module out of train mode...
    assert model.training is True
    # ...nor mutate any weights (no optimiser step).
    for p, p0 in zip(model.parameters(), before, strict=True):
        assert torch.equal(p, p0)
    # ...and grads are left zeroed.
    assert all(p.grad is None for p in model.parameters())


def test_invalid_inputs_rejected():
    model = _linear_only_mlp(3, 8, 1, 1)
    with pytest.raises(ValueError):
        measure_forward_flops_per_sample(model, torch.randn(3))  # not 2-D
    with pytest.raises(ValueError):
        measure_inference_flops_per_eval(model, torch.randn(4, 3), model_kind="bogus")
    with pytest.raises(ValueError):
        build_compute_accounting(
            model, torch.randn(4, 3), model_kind="potential_autograd", total_samples_processed=-1
        )


def test_flops_to_human_readable():
    assert flops_to_human(8.64e19).endswith("EFLOP") or "PFLOP" in flops_to_human(8.64e19)
    assert "TFLOP" in flops_to_human(3.4e12)
    assert "GFLOP" in flops_to_human(2.0e9)


def test_classical_sh_terms_match_kernel_double_loop():
    # The kernel sums n=2..N, m=0..n. Count terms directly and compare the
    # closed form (N+4)(N-1)/2.
    def brute(n_max: int) -> int:
        return sum((n + 1) for n in range(2, n_max + 1))

    for n in range(0, 130):
        assert classical_sh_terms(n) == brute(n)
    # Spot anchors.
    assert classical_sh_terms(0) == 0
    assert classical_sh_terms(1) == 0
    assert classical_sh_terms(2) == 3
    assert classical_sh_terms(3) == 7
    assert classical_sh_terms(4) == 12


def test_classical_sh_flops_scale_quadratically():
    # With the base overhead removed, the harmonic work scales as O(N^2):
    # flops(2N)/flops(N) -> 4 for large N.
    n = 200
    ratio = (
        classical_sh_flops_per_eval(2 * n, include_base=False)
        / classical_sh_flops_per_eval(n, include_base=False)
    )
    assert ratio == pytest.approx(4.0, rel=0.02)


def test_classical_sh_flops_monotonic_and_base():
    prev = -1.0
    for n in range(0, 50):
        f = classical_sh_flops_per_eval(n)
        assert f >= prev
        prev = f
    # Degree < 2 has no harmonic terms -> base overhead only.
    assert classical_sh_flops_per_eval(0) == pytest.approx(CLASSICAL_SH_BASE_FLOPS)
    assert classical_sh_flops_per_eval(1) == pytest.approx(CLASSICAL_SH_BASE_FLOPS)
    # include_base toggle removes exactly the base overhead.
    assert classical_sh_flops_per_eval(8) - classical_sh_flops_per_eval(
        8, include_base=False
    ) == pytest.approx(CLASSICAL_SH_BASE_FLOPS)


def test_classical_sh_flops_value_and_overridable_constant():
    # Exact composition: terms * per_term + base.
    expected = classical_sh_terms(120) * CLASSICAL_SH_FLOPS_PER_TERM + CLASSICAL_SH_BASE_FLOPS
    assert classical_sh_flops_per_eval(120) == pytest.approx(expected)
    # The per-term constant is a documented modelling assumption -> overridable.
    doubled = classical_sh_flops_per_eval(120, flops_per_term=2 * CLASSICAL_SH_FLOPS_PER_TERM,
                                          include_base=False)
    assert doubled == pytest.approx(2 * classical_sh_terms(120) * CLASSICAL_SH_FLOPS_PER_TERM)


def test_classical_sh_rejects_negative_degree():
    with pytest.raises(ValueError):
        classical_sh_flops_per_eval(-1)


def test_compare_eval_cost_composition_and_reciprocity():
    sh_t = classical_sh_flops_per_eval(120)
    sh_b = classical_sh_flops_per_eval(8)
    surrogate = 5.0e5
    cmp = compare_eval_cost(surrogate, target_sh_degree=120, baseline_sh_degree=8)
    assert cmp["sh_target_flops_per_eval"] == pytest.approx(sh_t)
    assert cmp["sh_baseline_flops_per_eval"] == pytest.approx(sh_b)
    # Deployed path = baseline SH + surrogate.
    assert cmp["surrogate_path_flops_per_eval"] == pytest.approx(sh_b + surrogate)
    # ratio and speedup are reciprocals.
    assert cmp["ratio_vs_target"] == pytest.approx(1.0 / cmp["speedup_vs_target"])
    # Residual band = target terms minus baseline terms.
    assert cmp["sh_residual_terms_modelled"] == classical_sh_terms(120) - classical_sh_terms(8)
    assert "measured" in cmp["surrogate_flops_source"].lower()
    assert cmp["sh_flops_source"] == "analytic_estimate"


def test_compare_eval_cost_cheap_surrogate_beats_huge_degree():
    # A tiny surrogate replacing a very high-degree field does less arithmetic.
    cmp = compare_eval_cost(2.0e4, target_sh_degree=300, baseline_sh_degree=0)
    assert cmp["speedup_vs_target"] > 1.0
    assert cmp["ratio_vs_target"] < 1.0


def test_compare_eval_cost_realistic_surrogate_is_costlier_than_mid_degree():
    # The ~330k-param SIREN (~2 MFLOP/eval) does MORE arithmetic than SH degree-120.
    cmp = compare_eval_cost(1.97e6, target_sh_degree=120, baseline_sh_degree=0)
    assert cmp["speedup_vs_target"] < 1.0
    assert cmp["comparison_basis"] == "arithmetic_work_only"


def test_compare_eval_cost_rejects_baseline_above_target():
    with pytest.raises(ValueError):
        compare_eval_cost(1.0e6, target_sh_degree=50, baseline_sh_degree=80)


def test_compare_eval_cost_rejects_nonpositive_surrogate():
    with pytest.raises(ValueError):
        compare_eval_cost(0.0, target_sh_degree=120)


def test_compute_speed_section_sh_only():
    sec = build_compute_speed_section(sh_degrees={"sh32": 32, "sh120": 120})
    assert sec["unit"].startswith("FLOP per")
    assert sec["comparison_basis"] == "arithmetic_work_only"
    assert sec["per_model"]["sh32"]["source"] == "analytic_estimate"
    assert sec["per_model"]["sh32"]["sh_degree"] == 32
    assert sec["per_model"]["sh120"]["flops_per_eval"] == pytest.approx(
        classical_sh_flops_per_eval(120)
    )
    # No surrogate -> no deployed-path comparison.
    assert sec["surrogate_vs_target"] is None


def test_compute_speed_section_with_surrogate_and_target():
    sec = build_compute_speed_section(
        sh_degrees={"sh8": 8, "sh120": 120},
        surrogate_model_name="st_lrps",
        surrogate_inference_flops_per_eval=1.97e6,
        surrogate_target_sh_degree=120,
        surrogate_baseline_sh_degree=8,
    )
    assert sec["per_model"]["st_lrps"]["source"].startswith("measured")
    assert sec["per_model"]["st_lrps"]["flops_per_eval"] == pytest.approx(1.97e6)
    cmp = sec["surrogate_vs_target"]
    assert cmp is not None
    assert cmp["sh_target_degree"] == 120
    assert cmp["speedup_vs_target"] < 1.0  # realistic surrogate is costlier per eval


def test_compute_speed_section_surrogate_without_target_degree():
    # Surrogate cost listed, but no deployed-path comparison without a target degree.
    sec = build_compute_speed_section(
        sh_degrees={"sh32": 32},
        surrogate_model_name="st_lrps",
        surrogate_inference_flops_per_eval=5.0e5,
    )
    assert "st_lrps" in sec["per_model"]
    assert sec["surrogate_vs_target"] is None


# --- Phase 5: device peak / MFU -------------------------------------------

def test_lookup_device_peak_matches_known_gpus():
    assert lookup_device_peak_flops("NVIDIA GeForce GTX 1660 Ti") == pytest.approx(5.4e12)
    assert lookup_device_peak_flops("NVIDIA A100-SXM4-40GB") == pytest.approx(1.95e13)
    assert lookup_device_peak_flops("Tesla V100-SXM2-16GB") == pytest.approx(1.57e13)
    # Unknown / empty -> None (no guessing).
    assert lookup_device_peak_flops("Some Unknown Accelerator") is None
    assert lookup_device_peak_flops(None) is None
    assert lookup_device_peak_flops("") is None


def test_mfu_computed_for_known_device():
    model = _linear_only_mlp(3, 32, 3, 2)
    x = torch.randn(8, 3)
    acc = build_compute_accounting(
        model,
        x,
        model_kind="potential_autograd",
        total_samples_processed=10_000_000,
        wall_clock_seconds=60.0,
        device="NVIDIA GeForce GTX 1660 Ti",
    )
    assert acc.device_peak_flops_per_s == pytest.approx(5.4e12)
    assert acc.model_flops_utilization == pytest.approx(
        acc.achieved_flops_per_s / 5.4e12
    )
    d = acc.to_manifest_dict()
    assert d["hardware"]["model_flops_utilization"] == pytest.approx(acc.model_flops_utilization)
    assert d["hardware"]["device_peak_fp32_flops_per_s"] == pytest.approx(5.4e12)


def test_mfu_none_for_unknown_device_or_no_wallclock():
    model = _linear_only_mlp(3, 16, 3, 1)
    x = torch.randn(4, 3)
    # Unknown device -> no peak, no MFU even with wall-clock.
    a1 = build_compute_accounting(
        model, x, model_kind="potential_autograd", total_samples_processed=1000,
        wall_clock_seconds=10.0, device="Mystery GPU",
    )
    assert a1.device_peak_flops_per_s is None
    assert a1.model_flops_utilization is None
    # Known device but no wall-clock -> peak known, MFU still None.
    a2 = build_compute_accounting(
        model, x, model_kind="potential_autograd", total_samples_processed=1000,
        device="NVIDIA A100",
    )
    assert a2.device_peak_flops_per_s == pytest.approx(1.95e13)
    assert a2.model_flops_utilization is None


# --- Phase 4: markdown report renderer ------------------------------------

def test_render_compute_report_full():
    model = _linear_only_mlp(3, 32, 1, 2)
    x = torch.randn(8, 3)
    acc = build_compute_accounting(
        model, x, model_kind="potential_autograd", total_samples_processed=5_000_000,
        wall_clock_seconds=120.0, device="NVIDIA GeForce GTX 1660 Ti",
    )
    speed = build_compute_speed_section(
        sh_degrees={"sh32": 32, "sh120": 120},
        surrogate_model_name="st_lrps",
        surrogate_inference_flops_per_eval=acc.inference_flops_per_eval,
        surrogate_target_sh_degree=120,
        surrogate_baseline_sh_degree=8,
    )
    md = render_compute_report(acc.to_manifest_dict(), speed)
    assert "## Compute & Speed" in md
    assert "PF-days" in md
    assert "Inference" in md
    assert "MFU" in md  # known device -> MFU line present
    assert "| Model | FLOP/eval | Source |" in md
    assert "sh120" in md and "st_lrps" in md
    assert "speedup" in md.lower()


def test_render_compute_report_empty_is_graceful():
    md = render_compute_report(None, None)
    assert "## Compute & Speed" in md
    assert "No compute accounting" in md


def test_real_siren_model_accounting_runs():
    # Smoke test against the actual model factory / PhysicsNet path.
    from lunaris.surrogate.st_lrps.networks.models import build_model_from_config

    cfg = {
        "activation": "sine",
        "hidden": 64,
        "depth": 3,
        "runtime_model_kind": "potential_autograd",
        "output_dim": 1,
    }
    model = build_model_from_config(cfg)
    x = torch.randn(16, 3)
    acc = build_compute_accounting(
        model, x, model_kind="potential_autograd", total_samples_processed=2_000_000
    )
    assert acc.inference_flops_per_eval > 0.0
    assert acc.train_step_flops_per_sample > acc.forward_flops_per_sample
    assert math.isfinite(acc.total_training_pflops_days)
    assert isinstance(acc, ComputeAccounting)
