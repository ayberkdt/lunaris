"""Scientific telemetry regression tests for solver-returned trajectory states."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from lunaris.common.constants import MU_MOON, R_MOON
from lunaris.common.telemetry_contract import decode_sample_line
from lunaris.common.type_defs import EventConfig, PropagatorConfig, TimeConfig
from lunaris.core.propagation.propagator import propagate


class _PointMassDynamics:
    grav = None
    ephem = None

    def build_rhs(self):
        def rhs(_t: float, y: np.ndarray) -> np.ndarray:
            state = np.asarray(y, dtype=np.float64)
            r = state[:3]
            dy = np.empty_like(state)
            dy[:3] = state[3:6]
            dy[3:6] = -float(MU_MOON) * r / float(np.linalg.norm(r)) ** 3
            if state.size > 6:
                dy[6:] = 0.0
            return dy

        return rhs


def _state(altitude_m: float = 100_000.0) -> np.ndarray:
    radius = float(R_MOON) + altitude_m
    speed = math.sqrt(float(MU_MOON) / radius)
    return np.asarray([radius, 0.0, 0.0, 0.0, speed, 0.0], dtype=np.float64)


def _config(sink: Path, **changes) -> PropagatorConfig:
    values = {
        "method": "DOP853",
        "rtol": 1e-11,
        "atol": 1e-13,
        "verbose": False,
        "compute_2body_baseline": False,
        "use_nyquist_max_step": False,
        "events": EventConfig(detect_impact=False, enable_peri_apo_events=False),
        "enable_telemetry": True,
        "telem_cadence_s": 0.01,
        "telemetry_run_id": "accepted-state-test",
        "telemetry_sink_path": str(sink),
    }
    values.update(changes)
    return PropagatorConfig(**values)


def _samples(path: Path):
    return [
        decode_sample_line(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.startswith("[TELEMETRY]")
    ]


def _assert_replay_matches_result(path: Path, result) -> None:
    samples = _samples(path)
    assert len(samples) == result.t.size
    assert [sample.sample_kind for sample in samples] == ["output_state"] * len(samples)
    assert [sample.sequence_id for sample in samples] == list(range(len(samples)))
    times = np.asarray([sample.simulation_time_s for sample in samples])
    assert np.all(np.diff(times) > 0.0)
    np.testing.assert_allclose(times, result.t - result.t[0], rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(
        np.asarray([sample.state_inertial for sample in samples]),
        result.y[:, :6],
        rtol=0.0,
        atol=0.0,
    )


def test_adaptive_replay_uses_output_states_not_rhs_evaluations(tmp_path: Path) -> None:
    sink = tmp_path / "telemetry.ndjson"
    result = propagate(
        _PointMassDynamics(),
        _state(),
        _config(sink),
        time_cfg=TimeConfig(duration_s=120.0, output_dt_s=6.0),
    )

    assert result.ode.nfev > result.t.size
    _assert_replay_matches_result(sink, result)


@pytest.mark.parametrize("method", ["DOP853", "RK4"])
def test_telemetry_observation_does_not_change_the_numerical_trajectory(
    tmp_path: Path, method: str
) -> None:
    time_cfg = TimeConfig(duration_s=30.0, output_dt_s=2.0)
    enabled_cfg = _config(
        tmp_path / f"{method}.ndjson",
        method=method,
        user_max_step_s=1.0,
    )
    disabled_cfg = PropagatorConfig(
        method=method,
        rtol=enabled_cfg.rtol,
        atol=enabled_cfg.atol,
        verbose=False,
        compute_2body_baseline=False,
        use_nyquist_max_step=False,
        user_max_step_s=1.0,
        events=enabled_cfg.events,
        enable_telemetry=False,
    )
    observed = propagate(_PointMassDynamics(), _state(), enabled_cfg, time_cfg=time_cfg)
    reference = propagate(_PointMassDynamics(), _state(), disabled_cfg, time_cfg=time_cfg)
    np.testing.assert_array_equal(observed.t, reference.t)
    np.testing.assert_array_equal(observed.y, reference.y)


def test_event_terminal_state_is_present_once_and_no_sample_follows_it(tmp_path: Path) -> None:
    sink = tmp_path / "telemetry.ndjson"

    def terminal_event(t_s: float, _y: np.ndarray) -> float:
        return t_s - 23.5

    terminal_event.terminal = True  # type: ignore[attr-defined]
    terminal_event.direction = 1.0  # type: ignore[attr-defined]
    result = propagate(
        _PointMassDynamics(),
        _state(),
        _config(sink),
        time_cfg=TimeConfig(duration_s=60.0, output_dt_s=10.0),
        extra_events=[terminal_event],
    )

    assert result.stopped_early is True
    assert result.t[-1] == pytest.approx(23.5)
    assert np.count_nonzero(np.isclose(result.t, 23.5, rtol=0.0, atol=1e-10)) == 1
    _assert_replay_matches_result(sink, result)
    assert max(sample.simulation_time_s for sample in _samples(sink)) == pytest.approx(23.5)


def test_chunked_adaptive_replay_has_one_coherent_sequence(tmp_path: Path) -> None:
    sink = tmp_path / "telemetry.ndjson"
    result = propagate(
        _PointMassDynamics(),
        _state(),
        _config(sink, chunk_s=20.0),
        time_cfg=TimeConfig(duration_s=60.0, output_dt_s=5.0),
    )

    assert np.all(np.diff(result.t) > 0.0)
    _assert_replay_matches_result(sink, result)


def test_fixed_step_replay_matches_completed_output_states(tmp_path: Path) -> None:
    sink = tmp_path / "telemetry.ndjson"
    result = propagate(
        _PointMassDynamics(),
        _state(),
        _config(sink, method="RK4", telem_cadence_s=0.1, user_max_step_s=1.0),
        time_cfg=TimeConfig(duration_s=20.0, output_dt_s=2.0),
    )

    _assert_replay_matches_result(sink, result)
