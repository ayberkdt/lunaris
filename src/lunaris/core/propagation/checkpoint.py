"""Checkpoint and cooperative-stop helpers for propagation."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np


def _stop_requested(stop_file: str | None) -> bool:
    """Stop if a sentinel file exists (safe on all platforms)."""
    if not stop_file:
        return False
    try:
        return Path(os.fspath(stop_file)).exists()
    except Exception:
        return False

def _atomic_save_npz(path: str, **arrays) -> None:
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
        np.savez(str(tmp_npz), **arrays)
        os.replace(str(tmp_npz), str(dst))
    finally:
        for p in (tmp, tmp_npz):
            try:
                if p.exists():
                    p.unlink()
            except Exception:
                # R29b-justified: temp-file cleanup on the (possibly failing)
                # save path; masking the original np.savez error would be worse.
                pass


__all__ = ["_stop_requested", "_atomic_save_npz"]
