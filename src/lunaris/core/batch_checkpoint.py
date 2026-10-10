# lunaris.core.batch_checkpoint
"""
Resumable checkpoints for the shared batched fixed-step loop.

A 4096-orbit, 60-day screening propagation is one ``run_batched_fixed_step``
call that takes tens of hours; before this module existed a killed process
lost all of it (see ``tasks/lessons.md``, 2026-09-25). The checkpointer
persists, per device chunk ``[a, b)``:

* ``state.npz`` — the integrator state at a snapshot boundary: the state
  tensor in its *exact* device dtype, the alive mask, the impact bookkeeping
  tensors, ``t_curr``, ``global_step``, the number of finished snapshots and
  a ``complete`` flag;
* ``snaps_<lo>_<hi>.npy`` — the host snapshot rows ``Y_out[lo:hi, a:b]``
  written since the previous checkpoint (append-only parts, so each save costs
  only the new snapshots, never a rewrite of the whole ``(T, N, 6)`` block).

Disk cost over a full run is therefore one uncompressed copy of ``Y_out``
(e.g. ~1.7 GB for 4096 samples x 8641 snapshots), the same order as the host
buffer the loop already holds. The caller deletes the directory when the run's
own artifact is safely written.

Resuming restores the tensors bit-for-bit and continues the same RK4 arithmetic
from the same ``t_curr``; the trajectories are identical to an uninterrupted
run (the loop is deterministic and every provider is stateless in time). That
identity is *measured* bitwise on CPU float64 in
``tests/test_batch_checkpoint.py``. On CUDA it follows from the same argument —
the restored state is exact and the resumed chunk evaluates the same shapes, so
the same kernels run — but it has not been measured, so treat a CUDA resume as
bit-identical by construction, not by evidence.

A ``meta.json`` records the grid contract (N, snapshots, steps per snapshot,
effective dt, dtype, a hash of ``Y0``, impact settings). Checkpoints whose
contract differs from the current call are never reused: they are moved to a
``stale_<timestamp>`` sibling directory and the run starts fresh.

Chunk boundaries are part of the record's identity (the directory is named
``chunk_<a>_<b>``), so a resumed run that picks a different chunk size — after
an OOM halving, or on a machine with different VRAM — simply finds no matching
record and recomputes those samples. That is a lost speed-up, never a wrong
answer.

Writes are atomic (temp file + ``os.replace``) and ordered so that a crash
between the snapshot part and the state file leaves a consistent older state
plus an orphan part that the loader truncates.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from torch import Tensor

logger = logging.getLogger(__name__)

CHECKPOINT_SCHEMA_VERSION = 1
_META = "meta.json"
_STATE = "state.npz"
_SNAP_RE = re.compile(r"^snaps_(\d+)_(\d+)\.npy$")


@dataclass(frozen=True)
class BatchCheckpointSpec:
    """Where and how often the shared loop checkpoints.

    Parameters
    ----------
    directory :
        Checkpoint directory for ONE ``run_batched_fixed_step`` call (one
        ensemble block). Reusing it for a different block is caught by the
        contract check.
    interval_s :
        Minimum wall-clock seconds between periodic checkpoints. ``0`` saves
        after every snapshot (tests); a completed chunk is always saved.
    """

    directory: Path
    interval_s: float = 1200.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "directory", Path(self.directory))
        if float(self.interval_s) < 0.0:
            raise ValueError("interval_s must be >= 0")


@dataclass
class ChunkCheckpoint:
    """One restored chunk record (tensors as host arrays in device dtype)."""

    state: np.ndarray
    alive: np.ndarray
    impact_step: np.ndarray
    impact_time: np.ndarray
    impact_pos: np.ndarray
    t_curr: float
    global_step: int
    snap_done: int
    active_steps: int
    complete: bool
    snapshots: np.ndarray  # (snap_done + 1, n, 6) float64 host rows


@dataclass
class _ChunkProgress:
    snap_written: int = 0  # Y_out rows [0, snap_written) already on disk


def y0_digest(Y0: np.ndarray) -> str:
    """SHA-256 of the initial-state block (float64 bytes) for the contract."""
    return hashlib.sha256(
        np.ascontiguousarray(Y0, dtype=np.float64).tobytes()
    ).hexdigest()


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        fh.write(payload)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _atomic_save_npy(path: Path, array: np.ndarray) -> None:
    tmp = path.with_name(path.name + ".tmp.npy")
    np.save(tmp, np.ascontiguousarray(array), allow_pickle=False)
    os.replace(tmp, path)


def _atomic_savez(path: Path, **arrays: Any) -> None:
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


class BatchStepCheckpointer:
    """Per-chunk checkpoint reader/writer bound to one propagation contract."""

    def __init__(
        self,
        spec: BatchCheckpointSpec,
        *,
        contract: dict[str, Any],
        torch_mod: Any,
        device: Any,
        dtype: Any,
    ) -> None:
        self.spec = spec
        self.directory = Path(spec.directory)
        self.contract = {"schema_version": CHECKPOINT_SCHEMA_VERSION, **contract}
        self._torch = torch_mod
        self._device = device
        self._dtype = dtype
        self._last_save = time.perf_counter()
        self._progress: dict[tuple[int, int], _ChunkProgress] = {}
        self.metrics: dict[str, Any] = {
            "checkpoint_dir": str(self.directory),
            "checkpoint_interval_s": float(spec.interval_s),
            "checkpoint_saves": 0,
            "checkpoint_restores": 0,
            "checkpoint_restored_snapshots": 0,
            "checkpoint_skipped_complete_chunks": 0,
            "checkpoint_stale_dir": None,
        }
        self._prepare_directory()

    # -- directory / contract -------------------------------------------------

    def _prepare_directory(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        meta_path = self.directory / _META
        if meta_path.exists():
            try:
                existing = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                existing = None
            if existing == self.contract:
                return
            stale = self.directory.with_name(
                f"{self.directory.name}_stale_{time.strftime('%Y%m%d_%H%M%S')}"
            )
            shutil.move(str(self.directory), str(stale))
            self.metrics["checkpoint_stale_dir"] = str(stale)
            logger.warning(
                "[CHECKPOINT] %s holds checkpoints for a different propagation "
                "contract; moved aside to %s and starting fresh.",
                self.directory,
                stale,
            )
            self.directory.mkdir(parents=True, exist_ok=True)
        _atomic_write_bytes(
            meta_path, json.dumps(self.contract, indent=2, sort_keys=True).encode("utf-8")
        )

    def _chunk_dir(self, a: int, b: int) -> Path:
        return self.directory / f"chunk_{int(a):07d}_{int(b):07d}"

    # -- timing -----------------------------------------------------------------

    def due(self) -> bool:
        """True when the periodic save interval has elapsed."""
        return (time.perf_counter() - self._last_save) >= float(self.spec.interval_s)

    # -- load -------------------------------------------------------------------

    def load_chunk(self, a: int, b: int) -> ChunkCheckpoint | None:
        """Return the newest consistent record for chunk ``[a, b)`` or ``None``."""
        cdir = self._chunk_dir(a, b)
        state_path = cdir / _STATE
        if not state_path.exists():
            return None
        with np.load(state_path, allow_pickle=False) as data:
            rec = {key: np.asarray(data[key]) for key in data.files}
        snap_done = int(rec["snap_done"])
        n = int(b - a)
        parts: list[tuple[int, int, Path]] = []
        for path in cdir.iterdir():
            match = _SNAP_RE.match(path.name)
            if match:
                parts.append((int(match.group(1)), int(match.group(2)), path))
        parts.sort()
        snapshots = np.empty((snap_done + 1, n, 6), dtype=np.float64)
        covered = 0
        for lo, hi, path in parts:
            if lo != covered:
                logger.warning(
                    "[CHECKPOINT] snapshot parts for chunk %s are not contiguous at %d; "
                    "ignoring the checkpoint.",
                    cdir.name,
                    lo,
                )
                return None
            if lo > snap_done:
                break  # orphan part written after the last consistent state
            hi_eff = min(hi, snap_done + 1)
            block = np.load(path, allow_pickle=False)
            if block.shape != (hi - lo, n, 6):
                logger.warning(
                    "[CHECKPOINT] snapshot part %s has shape %s; ignoring the checkpoint.",
                    path.name,
                    block.shape,
                )
                return None
            snapshots[lo:hi_eff] = block[: hi_eff - lo]
            covered = hi_eff
            if covered >= snap_done + 1:
                break
        if covered < snap_done + 1:
            logger.warning(
                "[CHECKPOINT] chunk %s: only %d of %d snapshot rows on disk; ignoring.",
                cdir.name,
                covered,
                snap_done + 1,
            )
            return None
        self._progress[(int(a), int(b))] = _ChunkProgress(snap_written=snap_done + 1)
        record = ChunkCheckpoint(
            state=rec["state"],
            alive=rec["alive"].astype(bool),
            impact_step=rec["impact_step"].astype(np.int64),
            impact_time=rec["impact_time"],
            impact_pos=rec["impact_pos"],
            t_curr=float(rec["t_curr"]),
            global_step=int(rec["global_step"]),
            snap_done=snap_done,
            active_steps=int(rec["active_steps"]),
            complete=bool(rec["complete"]),
            snapshots=snapshots,
        )
        self.metrics["checkpoint_restores"] += 1
        self.metrics["checkpoint_restored_snapshots"] += snap_done
        if record.complete:
            self.metrics["checkpoint_skipped_complete_chunks"] += 1
        # A restore must be visible in the log: a resumed run that silently
        # restarted from zero looks exactly like a resumed run that worked.
        logger.info(
            "[CHECKPOINT] chunk %s: RESTORED at snapshot %d/%d (t=%.1f s)%s",
            cdir.name,
            snap_done,
            self.contract.get("n_snapshots", -1),
            record.t_curr,
            " - already complete, skipping propagation" if record.complete else "",
        )
        return record

    def to_device(self, array: np.ndarray, *, dtype: Any = None) -> Tensor:
        """Host array -> device tensor in the requested (or checkpointed) dtype."""
        torch = self._torch
        kwargs: dict[str, Any] = {"device": self._device}
        if dtype is not None:
            kwargs["dtype"] = dtype
        return torch.as_tensor(np.ascontiguousarray(array), **kwargs)

    # -- save -------------------------------------------------------------------

    def save_chunk(
        self,
        a: int,
        b: int,
        *,
        state: Tensor,
        alive: Tensor,
        impact_step: Tensor,
        impact_time: Tensor,
        impact_pos: Tensor,
        t_curr: float,
        global_step: int,
        snap_done: int,
        active_steps: int,
        Y_out: np.ndarray,
        complete: bool,
    ) -> None:
        """Persist chunk ``[a, b)`` at the boundary after snapshot ``snap_done``."""
        key = (int(a), int(b))
        progress = self._progress.setdefault(key, _ChunkProgress())
        cdir = self._chunk_dir(a, b)
        cdir.mkdir(parents=True, exist_ok=True)
        lo, hi = progress.snap_written, snap_done + 1
        if hi > lo:
            _atomic_save_npy(cdir / f"snaps_{lo:06d}_{hi:06d}.npy", Y_out[lo:hi, a:b, :])
        _atomic_savez(
            cdir / _STATE,
            state=state.detach().cpu().numpy(),
            alive=alive.detach().cpu().numpy(),
            impact_step=impact_step.detach().cpu().numpy(),
            impact_time=impact_time.detach().cpu().numpy(),
            impact_pos=impact_pos.detach().cpu().numpy(),
            t_curr=np.asarray(float(t_curr), dtype=np.float64),
            global_step=np.asarray(int(global_step), dtype=np.int64),
            snap_done=np.asarray(int(snap_done), dtype=np.int64),
            active_steps=np.asarray(int(active_steps), dtype=np.int64),
            complete=np.asarray(bool(complete)),
            saved_at=np.asarray(time.strftime("%Y-%m-%dT%H:%M:%S")),
        )
        progress.snap_written = hi
        self._last_save = time.perf_counter()
        self.metrics["checkpoint_saves"] += 1
        logger.info(
            "[CHECKPOINT] chunk %s: snapshot %d saved%s (%s)",
            cdir.name,
            snap_done,
            " [complete]" if complete else "",
            self.directory,
        )


__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "BatchCheckpointSpec",
    "BatchStepCheckpointer",
    "ChunkCheckpoint",
    "y0_digest",
]
