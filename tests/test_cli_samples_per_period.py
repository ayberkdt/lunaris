"""``--samples-per-period`` sets the output grid of single and batch runs.

The default config fixes ``output_dt_s = 60 s``, and the time-grid plan only
derives the spacing from ``samples_per_period`` when ``output_dt_s`` is None,
so the flag used to be recorded and then ignored.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from lunaris.cli import batch_runner
from lunaris.cli.common_args import apply_args_to_config
from lunaris.cli.options import parse_args
from lunaris.common.constants import MU_MOON
from lunaris.common.type_defs import GravityConfig, InitialState, PerturbationFlags, TimeConfig
from lunaris.core.config import SimConfig
from lunaris.core.propagation.plans import resolve_time_grid_plan

R0_M = 1_838_000.0
V0_M_S = math.sqrt(MU_MOON / R0_M)  # circular, so a = r0
PERIOD_S = 2.0 * math.pi * math.sqrt(R0_M**3 / MU_MOON)


def _cfg() -> SimConfig:
    return SimConfig(
        gravity=GravityConfig(file_path="gravity.tab"),
        spice=SimpleNamespace(include_third_body=True, kernels=()),
        initial_state=InitialState(x=R0_M, y=0.0, z=0.0, vx=0.0, vy=V0_M_S, vz=0.0),
        time=TimeConfig(duration_s=4.0 * PERIOD_S, output_dt_s=60.0),
        flags=PerturbationFlags(enable_sh=False),
    )


def test_single_run_samples_per_period_replaces_the_default_spacing() -> None:
    cfg = apply_args_to_config(_cfg(), parse_args(["--samples-per-period", "100"]))

    assert cfg.time.output_dt_s is None
    assert cfg.time.samples_per_period == 100
    plan = resolve_time_grid_plan(
        dynamics=SimpleNamespace(grav=None),  # point mass: mu = MU_MOON
        y0=np.array([R0_M, 0.0, 0.0, 0.0, V0_M_S, 0.0]),
        cfg=cfg.propagator,
        time_cfg=cfg.time,
        verbose=False,
    )
    assert plan.requested_output_dt_s == pytest.approx(PERIOD_S / 100.0, rel=1e-12)
    assert plan.t_eval.size == 401


def test_without_samples_per_period_the_config_spacing_is_kept() -> None:
    cfg = apply_args_to_config(_cfg(), parse_args([]))
    assert cfg.time.output_dt_s == 60.0


def test_single_run_rejects_both_output_grid_flags() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--samples-per-period", "100", "--output-dt-s", "30"])


def test_apply_args_rejects_both_output_grid_settings() -> None:
    args = parse_args(["--samples-per-period", "100"])
    args.output_dt_s = 30.0
    with pytest.raises(ValueError, match="not both"):
        apply_args_to_config(_cfg(), args)


def test_batch_runner_rejects_both_output_grid_flags() -> None:
    with pytest.raises(SystemExit):
        batch_runner._parse_args(["--samples-per-period", "100", "--output-dt-s", "30"])


def test_batch_runner_resolves_samples_per_period_from_the_nominal_period() -> None:
    args = batch_runner._parse_args(["--samples-per-period", "50"])
    cfg = apply_args_to_config(_cfg(), args)
    assert cfg.time.output_dt_s is None

    resolved = batch_runner._resolve_batch_output_dt(cfg)

    # Same rule as the single run's time-grid plan: one osculating period of the
    # nominal state over samples_per_period, mu from the gravity source (point
    # mass here, enable_sh off).
    assert resolved.time.output_dt_s == pytest.approx(PERIOD_S / 50.0, rel=1e-12)


def test_batch_runner_keeps_an_explicit_output_spacing() -> None:
    cfg = _cfg()
    assert batch_runner._resolve_batch_output_dt(cfg) is cfg
