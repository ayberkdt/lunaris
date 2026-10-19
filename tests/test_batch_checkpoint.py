"""Resumable checkpoints for the shared batched fixed-step loop.

The contract under test: a propagation interrupted at any snapshot boundary
and resumed from ``lunaris.core.batch_checkpoint`` files produces bitwise the
same trajectories, impact bookkeeping and output grid as an uninterrupted run.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.requires_torch

from lunaris.core.batch_checkpoint import (  # noqa: E402
    BatchCheckpointSpec,
    BatchStepCheckpointer,
)
from lunaris.core.batched_fixed_step import run_batched_fixed_step  # noqa: E402
from lunaris.core.torch_frame import TorchMoonFrame  # noqa: E402

R_REF = 1.738e6
GM = 4.9028e12


class _KillableProvider:
    """Point-mass provider that raises after ``fail_after`` acceleration calls."""

    def __init__(self, fail_after: int | None = None) -> None:
        self.calls = 0
        self.fail_after = fail_after

    def acceleration(self, t_s: float, s):
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            raise RuntimeError("simulated kill")
        r = s[:, :3]
        rn = torch.linalg.norm(r, dim=1, keepdim=True).clamp_min(1.0)
        return -GM * r / (rn**3)


def _y0(n: int, *, low_perilune: bool = False) -> np.ndarray:
    rng = np.random.default_rng(11)
    Y0 = np.zeros((n, 6))
    for j in range(n):
        alt = 100e3 + 400e3 * rng.random()
        r = R_REF + alt
        v = np.sqrt(GM / r)
        if low_perilune and j % 3 == 0:
            v *= 0.90  # eccentric enough to impact within the arc
        Y0[j] = [r, 0.0, 0.0, 0.0, v * np.cos(0.3 * j), v * np.sin(0.3 * j)]
    return Y0


def _run(Y0, provider, *, spec=None, chunk_size=None, dt_s=60.0):
    frame = TorchMoonFrame(None, device=torch.device("cpu"), dtype=torch.float64, allow_identity=True)
    return run_batched_fixed_step(
        torch_mod=torch,
        device=torch.device("cpu"),
        dtype=torch.float64,
        provider=provider,
        frame=frame,
        Y0=Y0,
        duration_s=6.0 * 3600.0,
        output_dt_s=600.0,
        dt_s=dt_s,
        impact_r_m=R_REF,
        chunk_size=chunk_size,
        checkpoint=spec,
    )


def _assert_identical(a, b) -> None:
    assert np.array_equal(a.t_out, b.t_out)
    assert np.array_equal(a.Y_out, b.Y_out)  # bitwise, NaNs excluded by construction
    assert np.array_equal(a.impact_flags, b.impact_flags)
    assert np.array_equal(np.nan_to_num(a.t_impact, nan=-1.0), np.nan_to_num(b.t_impact, nan=-1.0))
    assert np.array_equal(
        np.nan_to_num(a.impact_positions_inertial, nan=-1.0),
        np.nan_to_num(b.impact_positions_inertial, nan=-1.0),
    )


@pytest.mark.parametrize("chunk_size", [None, 3])
def test_interrupted_run_resumes_bitwise_identical(tmp_path, chunk_size, caplog) -> None:
    Y0 = _y0(7, low_perilune=True)
    ref_provider = _KillableProvider()
    reference = _run(Y0, ref_provider, chunk_size=chunk_size)
    assert reference.impact_flags.sum() >= 1  # impacts are part of the contract

    spec = BatchCheckpointSpec(directory=tmp_path / "ckpt", interval_s=0.0)  # save every snapshot
    # ~40 % of the way through the first chunk's RHS budget.
    steps_total = int(reference.metrics["total_raw_state_steps"] / Y0.shape[0])
    killer = _KillableProvider(fail_after=4 * int(0.4 * steps_total))
    with pytest.raises(RuntimeError, match="simulated kill"):
        _run(Y0, killer, spec=spec, chunk_size=chunk_size)
    assert (tmp_path / "ckpt" / "meta.json").exists()
    assert any(p.name.startswith("chunk_") for p in (tmp_path / "ckpt").iterdir())

    resumed_provider = _KillableProvider()
    with caplog.at_level("INFO", logger="lunaris.core.batch_checkpoint"):
        resumed = _run(Y0, resumed_provider, spec=spec, chunk_size=chunk_size)
    _assert_identical(resumed, reference)
    # A resume must announce itself: silence would look like a fresh restart.
    assert any("RESTORED at snapshot" in r.message for r in caplog.records)
    assert resumed.metrics["checkpoint_enabled"] is True
    assert resumed.metrics["checkpoint_restores"] >= 1
    assert resumed.metrics["checkpoint_restored_snapshots"] >= 1
    # The resumed run did strictly less work than a full run.
    assert 0 < resumed_provider.calls < ref_provider.calls


def test_completed_checkpoint_skips_propagation_entirely(tmp_path) -> None:
    Y0 = _y0(5)
    spec = BatchCheckpointSpec(directory=tmp_path / "ckpt", interval_s=3600.0)
    first = _run(Y0, _KillableProvider(), spec=spec)
    assert first.metrics["checkpoint_saves"] >= 1  # completion save
    replay_provider = _KillableProvider(fail_after=0)  # any RHS call would raise
    replay = _run(Y0, replay_provider, spec=spec)
    _assert_identical(replay, first)
    assert replay_provider.calls == 0
    assert replay.metrics["checkpoint_skipped_complete_chunks"] == 1


def test_contract_mismatch_moves_stale_checkpoints_aside(tmp_path) -> None:
    Y0 = _y0(4)
    spec = BatchCheckpointSpec(directory=tmp_path / "ckpt", interval_s=3600.0)
    _run(Y0, _KillableProvider(), spec=spec, dt_s=60.0)
    meta_before = json.loads((tmp_path / "ckpt" / "meta.json").read_text(encoding="utf-8"))
    provider = _KillableProvider()
    changed = _run(Y0, provider, spec=spec, dt_s=30.0)  # different effective dt
    assert provider.calls > 0  # not served from the stale checkpoint
    assert changed.metrics["checkpoint_restores"] == 0
    assert changed.metrics["checkpoint_stale_dir"] is not None
    stale_dirs = [p for p in tmp_path.iterdir() if "_stale_" in p.name]
    assert len(stale_dirs) == 1
    meta_after = json.loads((tmp_path / "ckpt" / "meta.json").read_text(encoding="utf-8"))
    assert meta_before["effective_dt_s"] == 60.0
    assert meta_after["effective_dt_s"] == 30.0


def test_orphan_snapshot_part_after_last_state_is_ignored(tmp_path) -> None:
    Y0 = _y0(3)
    spec = BatchCheckpointSpec(directory=tmp_path / "ckpt", interval_s=0.0)
    reference = _run(Y0, _KillableProvider())
    killer = _KillableProvider(fail_after=4 * 120)
    with pytest.raises(RuntimeError):
        _run(Y0, killer, spec=spec)
    chunk_dir = next(p for p in (tmp_path / "ckpt").iterdir() if p.name.startswith("chunk_"))
    parts = sorted(p for p in chunk_dir.iterdir() if p.name.startswith("snaps_"))
    lo = max(int(p.name.split("_")[2].split(".")[0]) for p in parts)
    # Simulate a crash between writing a new part and updating state.npz.
    np.save(chunk_dir / f"snaps_{lo:06d}_{lo + 2:06d}.npy", np.full((2, 3, 6), 123.0))
    resumed = _run(Y0, _KillableProvider(), spec=spec)
    _assert_identical(resumed, reference)


def test_spec_rejects_negative_interval(tmp_path) -> None:
    with pytest.raises(ValueError, match="interval_s"):
        BatchCheckpointSpec(directory=tmp_path, interval_s=-1.0)


def test_checkpointer_metrics_start_clean(tmp_path) -> None:
    ck = BatchStepCheckpointer(
        BatchCheckpointSpec(directory=tmp_path / "c", interval_s=5.0),
        contract={"n_samples": 1},
        torch_mod=torch,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    assert ck.metrics["checkpoint_saves"] == 0
    assert ck.load_chunk(0, 1) is None
    assert ck.due() is False


class _ClockedProvider(_KillableProvider):
    """Point mass plus a small time-dependent term, so the clock reaches the state."""

    def acceleration(self, t_s: float, s):
        return super().acceleration(t_s, s) + 1.0e-12 * float(t_s) * s[:, :3]


def test_resume_recomputes_the_clock_from_the_step_index(tmp_path) -> None:
    # dt = 600/86 s is not representable: an older checkpoint saved t_curr as a
    # running sum, which differs from k * dt. The resume must use global_step.
    Y0 = _y0(4)
    reference = _run(Y0, _ClockedProvider(), dt_s=7.0)
    spec = BatchCheckpointSpec(directory=tmp_path / "ckpt", interval_s=0.0)
    steps_total = int(reference.metrics["total_raw_state_steps"] / Y0.shape[0])
    with pytest.raises(RuntimeError, match="simulated kill"):
        _run(Y0, _ClockedProvider(fail_after=4 * int(0.4 * steps_total)), spec=spec, dt_s=7.0)
    states = list((tmp_path / "ckpt").glob("chunk_*/state.npz"))
    assert states
    for path in states:
        with np.load(path, allow_pickle=False) as data:
            payload = {key: data[key] for key in data.files}
        # A stale clock, as a pre-fix running sum (or any other float) would leave.
        payload["t_curr"] = np.asarray(float(payload["t_curr"]) + 1.0e-3, dtype=np.float64)
        np.savez(path, **payload)
    resumed = _run(Y0, _ClockedProvider(), spec=spec, dt_s=7.0)
    _assert_identical(resumed, reference)
