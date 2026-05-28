"""Lightweight tests for ST-LRPS training diagnostics."""

from __future__ import annotations

from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lunaris.surrogate.st_lrps.training.engine import _cuda_memory_string, _format_cuda_memory_mib


def test_cuda_memory_string_is_empty_for_cpu() -> None:
    assert _cuda_memory_string(torch.device("cpu")) == ""


def test_format_cuda_memory_mib_includes_current_peak_and_total() -> None:
    text = _format_cuda_memory_mib(
        allocated_mib=44,
        reserved_mib=982,
        peak_allocated_mib=812,
        peak_reserved_mib=982,
        total_vram_mib=6144,
    )

    assert text == " cuda_mem=44/982MiB peak=812/982MiB total=6144MiB"
