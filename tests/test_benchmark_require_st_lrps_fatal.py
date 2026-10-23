"""A failed ST-LRPS series aborts a ``--require-st-lrps`` GPU batch benchmark.

Before 2026-10-23 the GPU batch loop printed the error and went on, so a run
whose surrogate could not be built (here: the SH25 baseline file not found under
the canonical name a fresh ``lunaris-data download`` installs) exited 0 with no
ST-LRPS result. The HPC benchmark template always passes ``--require-st-lrps``.
"""

from __future__ import annotations

import argparse

import pytest

pytest.importorskip("torch")

from lunaris.surrogate.st_lrps.evaluation._gravity_benchmark.modes import (  # noqa: E402
    _gpu_task_failure_is_fatal,
)
from lunaris.surrogate.st_lrps.evaluation._gravity_benchmark.types import GpuBatchTask  # noqa: E402


def _task(model: str) -> GpuBatchTask:
    return GpuBatchTask(model_name=model, cache_name=model, display_name=model.upper(), rk4_dt_s=10.0)


@pytest.mark.parametrize(
    ("model", "fail_fast", "require", "fatal"),
    [
        ("st_lrps", False, True, True),
        ("st_lrps", False, False, False),
        ("sh200", False, True, False),  # only the required series is fatal
        ("sh200", True, False, True),
        ("st_lrps", True, False, True),
    ],
)
def test_which_gpu_series_failures_abort_the_run(model, fail_fast, require, fatal) -> None:
    args = argparse.Namespace(fail_fast=fail_fast, require_st_lrps=require)
    assert _gpu_task_failure_is_fatal(_task(model), args) is fatal
