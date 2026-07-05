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
    resolve_vram_aware_chunk_size,
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
# R08 — alive-sample compaction
# ---------------------------------------------------------------------------


class _CountingProvider(_PointMassProvider):
    """Counts the state rows it evaluates: a deterministic step-cost proxy."""

    def __init__(self) -> None:
        super().__init__()
        self.rows_evaluated = 0

    def acceleration(self, t_s: float, s):
        self.rows_evaluated += int(s.shape[0])
        return super().acceleration(t_s, s)


def _run_counting(Y0: np.ndarray, *, duration_s: float, output_dt_s: float,
                  dt_s: float = 60.0, chunk_size=None):
    provider = _CountingProvider()
    res = run_batched_fixed_step(
        torch_mod=torch,
        device=torch.device("cpu"),
        dtype=torch.float64,
        provider=provider,
        frame=_frame(),
        Y0=Y0,
        duration_s=duration_s,
        output_dt_s=output_dt_s,
        dt_s=dt_s,
        impact_r_m=R_REF,
        detect_impact=True,
        chunk_size=chunk_size,
    )
    return res, provider


def test_compaction_reduces_step_cost_for_impacted_batch() -> None:
    """R08 acceptance: with 70% of the batch impacted, the per-step evaluation
    cost drops accordingly (tolerant assert: >= 50% fewer provider rows than
    the raw N x steps x 4-stage count)."""
    n_total, n_dead = 10, 7
    states = [_circular_state(120.0) for _ in range(n_total - n_dead)]
    states += [_circular_state(-10.0) for _ in range(n_dead)]  # impact at t=0
    Y0 = np.stack(states)

    res, provider = _run_counting(Y0, duration_s=600.0, output_dt_s=120.0)

    assert res.metrics["impacted_sample_count"] == n_dead
    total_steps = int(res.metrics["total_raw_state_steps"]) // n_total
    raw_rows = n_total * total_steps * 4  # 4 RK4 stages per step
    assert provider.rows_evaluated <= 0.5 * raw_rows
    # Exactly the alive samples are evaluated once compacted at t=0.
    assert provider.rows_evaluated == (n_total - n_dead) * total_steps * 4
    # Alive trajectories are unaffected by their dead neighbours.
    assert np.all(np.isfinite(res.Y_out))


def test_compaction_all_impacted_skips_provider_entirely() -> None:
    Y0 = np.stack([_circular_state(-10.0), _circular_state(-20.0)])
    res, provider = _run_counting(Y0, duration_s=600.0, output_dt_s=120.0)
    assert provider.rows_evaluated == 0
    assert np.all(res.impact_flags == 1.0)
    assert np.all(res.t_impact == 0.0)
    # Frozen/terminal output: every snapshot repeats the initial states.
    for snap in range(res.Y_out.shape[0]):
        np.testing.assert_array_equal(res.Y_out[snap], Y0)


@pytest.mark.parametrize("chunk", [None, 2])
def test_compaction_preserves_mixed_impact_results(chunk) -> None:
    """Compacted runs are result-invariant across chunk sizes with mid-run
    impacts present (compaction is a memory/dispatch optimization, never a
    physics change)."""
    r0 = R_REF + 60_000.0
    Y0 = np.stack(
        [
            _circular_state(120.0),
            np.array([r0, 0.0, 0.0, -700.0, 0.0, 0.0], dtype=np.float64),  # impacts mid-run
            _circular_state(200.0),
            _circular_state(-10.0),  # impacts at t=0
        ]
    )
    ref = _run(Y0, duration_s=900.0, output_dt_s=300.0, chunk_size=None)
    res = _run(Y0, duration_s=900.0, output_dt_s=300.0, chunk_size=chunk)
    np.testing.assert_allclose(res.Y_out, ref.Y_out, rtol=0.0, atol=0.0)
    np.testing.assert_array_equal(res.impact_flags, ref.impact_flags)
    np.testing.assert_allclose(res.t_impact, ref.t_impact, rtol=0.0, atol=0.0)


# ---------------------------------------------------------------------------
# R06 — VRAM-aware chunk sizing + OOM recovery
# ---------------------------------------------------------------------------

GiB = 2**30


def test_chunk_resolver_cpu_uses_requested_or_default() -> None:
    chunk, prov = resolve_vram_aware_chunk_size(
        bytes_per_sample=1_000, free_bytes=0, total_bytes=0, requested=0
    )
    assert chunk == 1024 and prov["chunk_size_source"] == "cpu_default"
    chunk, prov = resolve_vram_aware_chunk_size(
        bytes_per_sample=1_000, free_bytes=0, total_bytes=0, requested=7
    )
    assert chunk == 7 and prov["chunk_size_source"] == "requested"


def test_chunk_resolver_band_by_total_vram() -> None:
    # Small GPU (4 GB): band cap 8192.
    chunk, prov = resolve_vram_aware_chunk_size(
        bytes_per_sample=1_000, free_bytes=3 * GiB, total_bytes=4 * GiB
    )
    assert chunk == 8192 and prov["vram_band"] == [2048, 8192]
    # Mid GPU (12 GB): band cap 32768.
    chunk, prov = resolve_vram_aware_chunk_size(
        bytes_per_sample=1_000, free_bytes=10 * GiB, total_bytes=12 * GiB
    )
    assert chunk == 32768 and prov["vram_band"] == [8192, 32768]
    # Large GPU (48 GB): band cap 262144.
    chunk, prov = resolve_vram_aware_chunk_size(
        bytes_per_sample=1_000, free_bytes=40 * GiB, total_bytes=48 * GiB
    )
    assert chunk == 262144 and prov["vram_band"] == [32768, 262144]


def test_chunk_resolver_memory_cap_dominates_band_and_request() -> None:
    # 1 MB/sample with 100 MB free -> cap = 80 samples, far below any band.
    chunk, prov = resolve_vram_aware_chunk_size(
        bytes_per_sample=2**20, free_bytes=100 * 2**20, total_bytes=4 * GiB
    )
    assert chunk == 80 and prov["chunk_size_memory_cap"] == 80
    # An explicit request above the cap is capped, recorded as such.
    chunk, prov = resolve_vram_aware_chunk_size(
        bytes_per_sample=2**20, free_bytes=100 * 2**20, total_bytes=4 * GiB, requested=512
    )
    assert chunk == 80 and prov["chunk_size_source"] == "requested_capped"


def test_chunk_resolver_single_sample_over_budget_raises() -> None:
    with pytest.raises(RuntimeError, match="VRAM"):
        resolve_vram_aware_chunk_size(
            bytes_per_sample=10 * GiB, free_bytes=1 * GiB, total_bytes=4 * GiB
        )


class _OOMOnLargeBatchProvider(_PointMassProvider):
    """Simulates CUDA OOM whenever a chunk larger than ``max_rows`` arrives."""

    def __init__(self, max_rows: int) -> None:
        super().__init__()
        self.max_rows = int(max_rows)

    def acceleration(self, t_s: float, s):
        if int(s.shape[0]) > self.max_rows:
            raise RuntimeError("CUDA out of memory. Tried to allocate everything.")
        return super().acceleration(t_s, s)


def test_oom_recovery_halves_chunk_and_matches_reference() -> None:
    rng = np.random.default_rng(9)
    Y0 = np.repeat(_circular_state(150.0)[None, :], 6, axis=0)
    Y0[:, :3] += rng.normal(0.0, 3_000.0, size=(6, 3))

    ref = _run(Y0, duration_s=600.0, output_dt_s=300.0, chunk_size=1)

    res = run_batched_fixed_step(
        torch_mod=torch,
        device=torch.device("cpu"),
        dtype=torch.float64,
        provider=_OOMOnLargeBatchProvider(max_rows=2),
        frame=_frame(),
        Y0=Y0,
        duration_s=600.0,
        output_dt_s=300.0,
        dt_s=60.0,
        impact_r_m=R_REF,
        chunk_size=6,
    )
    # Halvings 6 -> 3 -> 1 recorded; the run completes with the smaller chunk.
    assert [r["retry_chunk_size"] for r in res.metrics["oom_recoveries"]] == [3, 1]
    assert res.metrics["chunk_size_requested"] == 6
    assert res.metrics["chunk_size_effective"] == 1
    np.testing.assert_allclose(res.Y_out, ref.Y_out, rtol=0.0, atol=0.0)


def test_oom_at_chunk_one_reraises() -> None:
    Y0 = _circular_state(150.0)[None, :]
    with pytest.raises(RuntimeError, match="out of memory"):
        run_batched_fixed_step(
            torch_mod=torch,
            device=torch.device("cpu"),
            dtype=torch.float64,
            provider=_OOMOnLargeBatchProvider(max_rows=0),
            frame=_frame(),
            Y0=Y0,
            duration_s=120.0,
            output_dt_s=120.0,
            dt_s=60.0,
            impact_r_m=R_REF,
            chunk_size=1,
        )


# ---------------------------------------------------------------------------
# G3 — 100k screening smoke: chunking + compaction + summary-only together
# ---------------------------------------------------------------------------


def test_g3_100k_screening_smoke_chunked_compacted_summary() -> None:
    """G3 gate: a 100k-sample screening pass runs through the shared loop with
    VRAM-style chunking and alive compaction, then reduces to the versioned
    summary + top-K without ever holding more than one output grid of states.

    CPU-sized: one RK4 step per sample (the memory/plumbing is what's under
    test; per-step physics cost is covered elsewhere). On a real GPU the same
    path runs with the full step count.
    """
    from lunaris.batch.summary import TopKTrajectoryBuffer, summarize_ensemble

    n_total = 100_000
    rng = np.random.default_rng(42)
    Y0 = np.repeat(_circular_state(120.0)[None, :], n_total, axis=0)
    Y0[:, :3] += rng.normal(0.0, 5_000.0, size=(n_total, 3))
    Y0[:, 3:] += rng.normal(0.0, 2.0, size=(n_total, 3))
    # ~30% start below the surface: compaction drops them after the t=0 check.
    dead = rng.choice(n_total, size=n_total // 3, replace=False)
    Y0[dead, :3] *= 0.5

    res = _run(
        Y0,
        duration_s=60.0,
        output_dt_s=60.0,
        dt_s=60.0,
        chunk_size=8192,  # VRAM-band-sized chunks (small-GPU band)
    )

    assert res.Y_out.shape == (2, n_total, 6)
    assert res.metrics["impacted_sample_count"] >= dead.size
    # Compaction: dead samples never reached the provider.
    assert res.metrics["total_active_state_steps"] < res.metrics["total_raw_state_steps"]
    assert res.metrics["chunk_size_effective"] == 8192

    summary = summarize_ensemble(
        res.t_out, res.Y_out, res.impact_flags, res.t_impact,
        mu_m3s2=GM, r_ref_m=R_REF,
    )
    assert summary["n_samples"] == n_total
    scores = np.asarray(summary["fields"]["score"], dtype=np.float64)
    topk = TopKTrajectoryBuffer(16)
    topk.offer_batch(
        global_start=0, scores=scores, Y_batch=res.Y_out,
        impact_flags=res.impact_flags, t_impact=res.t_impact,
    )
    kept = topk.stacked_trajectories(res.Y_out.shape[0])
    assert kept.shape[1] <= 16
    assert all(np.isfinite(s) for s in topk.scores)


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
