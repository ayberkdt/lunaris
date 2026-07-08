"""Checkpoint and cooperative-stop helpers for propagation."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

from lunaris.common.contracts.checkpoint import CHECKPOINT_SCHEMA_VERSION
from lunaris.common.hashing import canonical_json_sha256


@dataclass(frozen=True, slots=True)
class PropagationCheckpoint:
    """Loaded propagation checkpoint metadata.

    Propagation checkpoints are write-only diagnostic artifacts today. This
    loader describes what was written and exposes the latest state, but it does
    not promise full solver resume semantics.
    """

    path: Path
    checkpoint_schema_version: int
    t_current: float
    y_current: np.ndarray
    method: str | None
    config_hash: str | None
    t: np.ndarray | None
    y_row: np.ndarray | None


def _stop_requested(stop_file: str | None) -> bool:
    """Stop if a sentinel file exists (safe on all platforms)."""
    if not stop_file:
        return False
    try:
        return Path(os.fspath(stop_file)).exists()
    except (OSError, TypeError, ValueError):
        return False


def _config_payload(config: Any) -> Any:
    if config is None:
        return None
    try:
        if is_dataclass(config) and not isinstance(config, type):
            return asdict(config)
    except (TypeError, ValueError):
        return repr(config)
    return config


def _checkpoint_config_hash(config: Any) -> str:
    """Best-effort stable hash for the config object attached to a checkpoint."""

    try:
        return canonical_json_sha256(_config_payload(config))
    except (TypeError, ValueError):
        return ""


def _checkpoint_metadata(*, method: str, config: Any) -> dict[str, Any]:
    """Metadata fields stamped into every propagation checkpoint."""

    return {
        "method": str(method),
        "config_hash": _checkpoint_config_hash(config),
    }


def _latest_t(t: Any) -> float:
    try:
        arr = np.asarray(t, dtype=np.float64).reshape(-1)
        if arr.size:
            return float(arr[-1])
    except (TypeError, ValueError):
        pass
    return float("nan")


def _latest_y(y_row: Any) -> np.ndarray:
    try:
        arr = np.asarray(y_row, dtype=np.float64)
        if arr.ndim == 2 and arr.shape[0] > 0:
            return np.asarray(arr[-1], dtype=np.float64)
        if arr.ndim == 1 and arr.size > 0:
            return np.asarray(arr, dtype=np.float64)
    except (TypeError, ValueError):
        pass
    return np.asarray([], dtype=np.float64)


def _with_checkpoint_contract(arrays: dict[str, Any]) -> dict[str, Any]:
    payload = dict(arrays)
    payload.setdefault(
        "checkpoint_schema_version",
        np.asarray(CHECKPOINT_SCHEMA_VERSION, dtype=np.int64),
    )
    payload.setdefault("method", "")
    payload.setdefault("config_hash", "")
    if "t_current" not in payload:
        payload["t_current"] = np.asarray(_latest_t(payload.get("t")), dtype=np.float64)
    if "y_current" not in payload:
        payload["y_current"] = _latest_y(payload.get("y_row"))
    return payload


def _scalar_item(value: Any) -> Any:
    arr = np.asarray(value)
    if arr.size == 0:
        return None
    item = arr.reshape(-1)[0].item()
    if isinstance(item, bytes):
        return item.decode("utf-8", errors="replace")
    return item


def _read_int(data: Any, key: str, default: int) -> int:
    if key not in data:
        return int(default)
    try:
        return int(_scalar_item(data[key]))
    except (TypeError, ValueError):
        return int(default)


def _read_float(data: Any, key: str, default: float) -> float:
    if key not in data:
        return float(default)
    try:
        return float(_scalar_item(data[key]))
    except (TypeError, ValueError):
        return float(default)


def _read_str(data: Any, key: str) -> str | None:
    if key not in data:
        return None
    try:
        text = str(_scalar_item(data[key]) or "").strip()
    except (TypeError, ValueError):
        text = ""
    return text or None


def _atomic_save_npz(path: str | os.PathLike[str], **arrays: Any) -> None:
    """
    Atomically write an NPZ file:
      - writes to <path>.tmp.npz
      - then os.replace() onto final path
    """
    if not path:
        return

    dst = Path(os.fspath(path))
    if dst.suffix.lower() != ".npz":
        dst = dst.with_suffix(".npz")

    tmp = dst.with_name(dst.name + ".tmp")  # e.g., out.npz.tmp
    tmp_npz = tmp.with_suffix(".npz")       # ensure numpy does not auto-append

    dst.parent.mkdir(parents=True, exist_ok=True)

    try:
        np.savez(str(tmp_npz), **_with_checkpoint_contract(arrays))
        os.replace(str(tmp_npz), str(dst))
    finally:
        for p in (tmp, tmp_npz):
            try:
                if p.exists():
                    p.unlink()
            except OSError:
                # R29b-justified: temp-file cleanup on the (possibly failing)
                # save path; masking the original np.savez error would be worse.
                pass


def load_propagation_checkpoint(path: str | os.PathLike[str]) -> PropagationCheckpoint:
    """Load propagation checkpoint metadata and latest state from an NPZ file."""

    src = Path(os.fspath(path))
    if src.suffix.lower() != ".npz":
        src = src.with_suffix(".npz")

    with np.load(src, allow_pickle=False) as data:
        t = np.asarray(data["t"], dtype=np.float64).copy() if "t" in data else None
        y_row = (
            np.asarray(data["y_row"], dtype=np.float64).copy()
            if "y_row" in data
            else None
        )
        t_default = _latest_t(t)
        y_default = _latest_y(y_row)
        y_current = (
            np.asarray(data["y_current"], dtype=np.float64).reshape(-1).copy()
            if "y_current" in data
            else y_default
        )
        return PropagationCheckpoint(
            path=src,
            checkpoint_schema_version=_read_int(
                data,
                "checkpoint_schema_version",
                0,
            ),
            t_current=_read_float(data, "t_current", t_default),
            y_current=y_current,
            method=_read_str(data, "method"),
            config_hash=_read_str(data, "config_hash"),
            t=t,
            y_row=y_row,
        )


__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "PropagationCheckpoint",
    "load_propagation_checkpoint",
    "_stop_requested",
    "_atomic_save_npz",
]
