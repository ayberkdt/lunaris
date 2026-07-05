"""R07: shared batched fixed-step RK4 + impact loop.

The impact behavior suite is parametrized over chunking so the SAME assertions
cover the single-batch (ST-LRPS style) and chunked (classic-SH style)
configurations of the one shared loop — impact fixes can no longer diverge
between backends.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.requires_torch

from lunaris.core.batched_fixed_step import (  # noqa: E402
    build_output_grid,
    rhs_batch,
    rk4_step,
    run_batched_fixed_step,
)
from lunaris.core.torch_frame import TorchMoonFrame  # noqa: E402

R_REF = 1.738e6
GM = 4.9028e12


class _PointMassProvider:
    """Analytic monopole provider: a = -mu r / |r|^3 (frame-independent)."""

    def __init__(self, mu: float = GM) -> None:
        self.mu = float(mu)

    def acceleration(self, t_s: float, s):
        r = s[:, :3]
        rn = torch.linalg.norm(r, dim=1, keepdim=True).clamp_min(1.0)
        return -self.mu * r / (rn**3)


def _frame() -> TorchMoonFrame:
    return TorchMoonFrame(None, device=torch.device("cpu"), dtype=torch.float64, allow_identity=True)


def _run(Y0: np.ndarray, *, duration_s: float, output_dt_s: float, dt_s: float = 60.0,
         chunk_size=None, detect_impact: bool = True, callback=None,
         callback_granularity: str = "chunk"):
    return run_batched_fixed_step(
        torch_mod=torch,
        device=torch.device("cpu"),
        dtype=torch.float64,
        provider=_PointMassProvider(),
        frame=_frame(),
        Y0=Y0,
        duration_s=duration_s,
        output_dt_s=output_dt_s,
        dt_s=dt_s,
        impact_r_m=R_REF,
        detect_impact=detect_impact,
        chunk_size=chunk_size,
        callback=callback,
        callback_granularity=callback_granularity,
    )


def _circular_state(alt_km: float) -> np.ndarray:
    r = R_REF + alt_km * 1_000.0
    v = math.sqrt(GM / r)
    return np.array([r, 0.0, 0.0, 0.0, v, 0.0], dtype=np.float64)


CHUNKS = [None, 1, 2]


# ---------------------------------------------------------------------------
# Output-grid + basic contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("chunk", CHUNKS)
def test_output_grid_endpoints_and_shapes(chunk) -> None:
    Y0 = np.stack([_circular_state(100.0), _circular_state(150.0), _circular_state(200.0)])
    res = _run(Y0, duration_s=1000.0, output_dt_s=600.0, chunk_size=chunk)
    assert res.t_out[0] == 0.0
    assert res.t_out[-1] == pytest.approx(1000.0)
    assert res.Y_out.shape == (len(res.t_out), 3, 6)
    assert res.impact_flags.shape == (3,)
    assert res.t_impact.shape == (3,)
    assert res.impact_positions_inertial.shape == (3, 3)
    assert np.all(np.isfinite(res.Y_out))


def test_build_output_grid_delegates_to_canonical_contract() -> None:
    t, n_snaps, snap_interval = build_output_grid(1000.0, 600.0)
    assert t[0] == 0.0
    assert t[-1] == pytest.approx(1000.0)
    assert n_snaps == len(t) - 1
    assert snap_interval == pytest.approx(500.0)


def test_chunking_is_result_invariant() -> None:
    rng = np.random.default_rng(3)
    Y0 = np.repeat(_circular_state(150.0)[None, :], 5, axis=0)
    Y0[:, :3] += rng.normal(0.0, 3_000.0, size=(5, 3))
    Y0[:, 3:] += rng.normal(0.0, 1.0, size=(5, 3))
    results = {
        chunk: _run(Y0, duration_s=900.0, output_dt_s=300.0, chunk_size=chunk)
        for chunk in (None, 1, 2, 5)
    }
    ref = results[None]
    for _chunk, res in results.items():
        np.testing.assert_allclose(res.Y_out, ref.Y_out, rtol=0.0, atol=0.0)
        np.testing.assert_array_equal(res.impact_flags, ref.impact_flags)


# ---------------------------------------------------------------------------
# Impact behavior — the single parametrized suite (R07 acceptance)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("chunk", CHUNKS)
def test_below_surface_sample_impacts_at_t0_and_freezes(chunk) -> None:
    y_safe = _circular_state(100.0)
    y_below = _circular_state(-10.0)
    Y0 = np.stack([y_safe, y_below])
    res = _run(Y0, duration_s=600.0, output_dt_s=120.0, chunk_size=chunk)
    assert res.impact_flags[1] == 1.0
    assert res.t_impact[1] == 0.0
    np.testing.assert_array_equal(
        res.Y_out[:, 1, :], np.broadcast_to(y_below, res.Y_out[:, 1, :].shape)
    )
    assert res.impact_flags[0] == 0.0
    assert not np.allclose(res.Y_out[0, 0, :], res.Y_out[-1, 0, :])


@pytest.mark.parametrize("chunk", CHUNKS)
def test_impact_crossing_is_interpolated_and_freezes_on_surface(chunk) -> None:
    r0 = R_REF + 60_000.0
    Y0 = np.array(
        [
            [r0, 0.0, 0.0, -700.0, 0.0, 0.0],      # radial inward fall
            [r0, 0.0, 0.0, -500.0, 300.0, 0.0],    # oblique
        ],
        dtype=np.float64,
    )
    duration = 600.0
    res = _run(Y0, duration_s=duration, output_dt_s=60.0, chunk_size=chunk)
    assert res.impact_flags[0] == 1.0 and res.impact_flags[1] == 1.0
    for i in (0, 1):
        # True line-sphere crossing sits on the impact sphere; time is sub-step.
        assert abs(float(np.linalg.norm(res.impact_positions_inertial[i])) - R_REF) < 1.0
        assert 0.0 < res.t_impact[i] < duration
        grid_dist = min(
            abs(float(res.t_impact[i]) - k * 60.0) for k in range(int(duration / 60.0) + 1)
        )
        assert grid_dist > 1e-3
        final_r = float(np.linalg.norm(res.Y_out[-1, i, :3]))
        assert abs(final_r - R_REF) < 1.0


@pytest.mark.parametrize("chunk", [None, 1])
def test_impact_detection_catches_outside_to_outside_step(chunk) -> None:
    y_offset = 0.98 * R_REF
    x_offset = 0.30 * R_REF
    duration = 60.0
    Y0 = np.array(
        [[x_offset, y_offset, 0.0, -2.0 * x_offset / duration, 0.0, 0.0]],
        dtype=np.float64,
    )
    assert np.linalg.norm(Y0[0, :3]) > R_REF
    # The raw RK4 endpoint stays outside the sphere: only the segment test hits.
    provider = _PointMassProvider()
    endpoint = rk4_step(torch, provider, torch.as_tensor(Y0, dtype=torch.float64), 0.0, duration)
    assert float(torch.linalg.norm(endpoint[0, :3])) > R_REF

    res = _run(Y0, duration_s=duration, output_dt_s=duration, dt_s=duration, chunk_size=chunk)
    assert res.impact_flags[0] == 1.0
    assert 0.0 < res.t_impact[0] < duration
    pos = res.impact_positions_inertial[0]
    assert abs(float(np.linalg.norm(pos)) - R_REF) < 1.0
    np.testing.assert_allclose(res.Y_out[-1, 0, :3], pos, atol=1e-9, rtol=0.0)


@pytest.mark.parametrize("chunk", CHUNKS)
def test_detect_impact_false_propagates_through(chunk) -> None:
    y_below = _circular_state(-10.0)
    res = _run(y_below[None, :], duration_s=300.0, output_dt_s=300.0,
               chunk_size=chunk, detect_impact=False)
    assert res.impact_flags[0] == 0.0
    assert np.isnan(res.t_impact[0])


# ---------------------------------------------------------------------------
# Throughput accounting + callbacks
# ---------------------------------------------------------------------------


def test_metrics_no_impact_raw_equals_active() -> None:
    Y0 = np.repeat(_circular_state(120.0)[None, :], 4, axis=0)
    res = _run(Y0, duration_s=600.0, output_dt_s=300.0, chunk_size=2)
    m = res.metrics
    assert m["impacted_sample_count"] == 0
    assert m["active_sample_count"] == 4
    assert m["total_active_state_steps"] == m["total_raw_state_steps"]


def test_metrics_impacted_samples_reduce_active_steps() -> None:
    Y0 = np.stack([_circular_state(100.0), _circular_state(-10.0), _circular_state(-10.0)])
    res = _run(Y0, duration_s=300.0, output_dt_s=300.0, chunk_size=2)
    m = res.metrics
    assert m["impacted_sample_count"] == 2
    assert m["total_active_state_steps"] < m["total_raw_state_steps"]


@pytest.mark.parametrize(
    ("granularity", "chunk"),
    [("chunk", 2), ("snapshot", None)],
)
def test_callback_fractions_monotone_and_complete(granularity, chunk) -> None:
    fractions: list[float] = []
    Y0 = np.repeat(_circular_state(120.0)[None, :], 4, axis=0)
    _run(
        Y0,
        duration_s=600.0,
        output_dt_s=200.0,
        chunk_size=chunk,
        callback=fractions.append,
        callback_granularity=granularity,
    )
    assert fractions, "callback was never invoked"
    assert all(b >= a for a, b in zip(fractions, fractions[1:], strict=False))
    assert fractions[-1] == pytest.approx(1.0)


def test_invalid_callback_granularity_raises() -> None:
    Y0 = _circular_state(120.0)[None, :]
    with pytest.raises(ValueError, match="callback_granularity"):
        _run(
            Y0,
            duration_s=60.0,
            output_dt_s=60.0,
            callback_granularity="sample",
        )


# ---------------------------------------------------------------------------
# Autograd safety of the shared stepper
# ---------------------------------------------------------------------------


def test_shared_rk4_step_builds_no_autograd_graph() -> None:
    provider = _PointMassProvider()
    s = torch.as_tensor(_circular_state(100.0)[None, :], dtype=torch.float64)
    s.requires_grad_(True)
    with torch.no_grad():
        rhs = rhs_batch(torch, provider, 0.0, s)
        out = rk4_step(torch, provider, s, 0.0, 60.0)
    assert rhs.grad_fn is None and rhs.requires_grad is False
    assert out.grad_fn is None and out.requires_grad is False
