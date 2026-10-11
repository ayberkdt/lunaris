"""``lunaris-batch`` exits non-zero when the run produced no usable result.

A run whose every sample failed (``valid_mask`` all False) or whose metrics
could not be built used to print a ``[BATCH][WARN]`` line and exit 0.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

import lunaris.batch as batch_pkg
from lunaris.cli import batch_runner
from lunaris.common.batch_defs import BatchPropagationResult
from lunaris.common.constants import MU_MOON
from lunaris.common.type_defs import GravityConfig, InitialState, PerturbationFlags, TimeConfig
from lunaris.core.config import SimConfig

R0_M = 1_838_000.0
V0_M_S = math.sqrt(MU_MOON / R0_M)


def _cfg() -> SimConfig:
    return SimConfig(
        gravity=GravityConfig(file_path="gravity.tab"),
        spice=SimpleNamespace(include_third_body=True, kernels=()),
        initial_state=InitialState(x=R0_M, y=0.0, z=0.0, vx=0.0, vy=V0_M_S, vz=0.0),
        time=TimeConfig(duration_s=120.0, output_dt_s=60.0),
        flags=PerturbationFlags(enable_sh=False),
    )


def _result(valid: list[bool]) -> BatchPropagationResult:
    n = len(valid)
    Y = np.tile(np.array([R0_M, 0.0, 0.0, 0.0, V0_M_S, 0.0]), (3, n, 1))
    return BatchPropagationResult(
        t=np.array([0.0, 60.0, 120.0]),
        Y=Y,
        sc_samples=np.ones((n, 4)),
        impact_mask=np.zeros(n),
        t_impact=np.full(n, np.nan),
        valid_mask=np.array(valid),
    )


def _run(monkeypatch, tmp_path, result: BatchPropagationResult) -> int:
    class _Engine:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def run(self) -> BatchPropagationResult:
            return result

    monkeypatch.setattr(batch_runner, "load_default_config", lambda **kwargs: _cfg())
    monkeypatch.setattr(batch_pkg, "BatchPropagationEngine", _Engine)
    return batch_runner.main(
        [
            "--n-samples", str(int(result.Y.shape[1])),
            "--use-gpu", "off",
            "--batch-output-format", "npz",
            "--batch-output-path", str(tmp_path / "out.npz"),
        ]
    )


def test_every_sample_failed_exits_with_runtime_error(monkeypatch, tmp_path, capsys) -> None:
    code = _run(monkeypatch, tmp_path, _result([False, False]))

    out = capsys.readouterr().out
    assert code == 2
    assert "[BATCH_METRICS]" not in out
    assert "[BATCH][FATAL]" in out


def test_metrics_failure_exits_with_runtime_error(monkeypatch, tmp_path) -> None:
    def _boom(*args, **kwargs):
        raise RuntimeError("metrics exploded")

    monkeypatch.setattr(batch_runner, "_build_metrics", _boom)
    assert _run(monkeypatch, tmp_path, _result([True, True])) == 2


def test_usable_run_still_exits_zero(monkeypatch, tmp_path, capsys) -> None:
    code = _run(monkeypatch, tmp_path, _result([True, False]))

    assert code == 0
    assert "[BATCH_METRICS]" in capsys.readouterr().out


@pytest.mark.parametrize("valid", [[True, True]])
def test_metrics_report_only_valid_samples(valid) -> None:
    from lunaris.common.batch_defs import BatchPropagationConfig

    metrics = batch_runner._build_metrics(
        _result(valid), 1.0, BatchPropagationConfig(n_samples=2, use_gpu=False)
    )
    assert metrics["n_valid_samples"] == 2
