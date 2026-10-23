"""A failed ST-LRPS series fails a ``--require-st-lrps`` GPU batch benchmark.

Before 2026-10-23 such a run exited 0 with no ST-LRPS result: the GPU batch
loop only printed a model's exception, and the integration itself returns
``status="failed"`` (its own errors, non-finite states) without raising. The
HPC benchmark template always passes ``--require-st-lrps``. The check runs after
the reports, so the finished SH series keep their metrics.
"""

from __future__ import annotations

import argparse

import numpy as np
import pytest

pytest.importorskip("torch")

from lunaris.surrogate.st_lrps.evaluation._gravity_benchmark.modes import (  # noqa: E402
    _raise_if_required_st_lrps_failed,
)
from lunaris.surrogate.st_lrps.evaluation._gravity_benchmark.types import (
    BatchModelResult,  # noqa: E402
)


def _result(model: str, status: str) -> BatchModelResult:
    return BatchModelResult(
        model_name=model, display_name=model.upper(), backend="failed" if status == "failed" else "torch",
        device="cpu", dtype="float64", t=np.array([], dtype=np.float64), y=np.empty((0, 1, 6)),
        runtime_s=float("nan"), n_steps=0, n_scenarios=1, rk4_dt_s=10.0, output_dt_s=60.0,
        status=status, failure_reason="non-finite state" if status == "failed" else "",
    )


def test_a_failed_required_st_lrps_series_fails_the_run() -> None:
    args = argparse.Namespace(require_st_lrps=True)
    with pytest.raises(RuntimeError, match="ST-LRPS series failed.*non-finite state"):
        _raise_if_required_st_lrps_failed(args, [_result("sh25", "ok"), _result("st_lrps", "failed")])
    with pytest.raises(RuntimeError):  # dt variants carry a suffix
        _raise_if_required_st_lrps_failed(args, [_result("st_lrps_dt10", "failed")])


@pytest.mark.parametrize(
    ("require", "results"),
    [
        (False, [("st_lrps", "failed")]),  # not required: reported, not fatal
        (True, [("st_lrps", "ok"), ("sh200", "failed")]),  # only the ST-LRPS series is required
        (True, []),
    ],
)
def test_other_outcomes_do_not_fail_the_run(require, results) -> None:
    args = argparse.Namespace(require_st_lrps=require)
    _raise_if_required_st_lrps_failed(args, [_result(m, s) for m, s in results])
