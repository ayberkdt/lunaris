"""Replay timeline single-step from a cursor between two samples.

Regression test: ``TimelineController.step(-1)`` from a paused playback
cursor strictly between samples ``t_k`` and ``t_{k+1}`` landed on
``t_{k-1}``, skipping the sample just behind the cursor.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6")

from lunaris.ui.monitor.replay import TimelineController  # noqa: E402


class _Store:
    mode = "replay"

    def __init__(self, times: np.ndarray) -> None:
        self._times = times

    def times(self) -> np.ndarray:
        return self._times

    def time_bounds(self) -> tuple[float, float]:
        return float(self._times[0]), float(self._times[-1])


class _Monitor:
    def __init__(self, times: np.ndarray) -> None:
        self.store = _Store(times)
        self.cursor_time_s: float | None = None

    def jump_to_time(self, t_s: float) -> None:
        self.cursor_time_s = float(t_s)


_TIMES = np.array([0.0, 10.0, 20.0, 30.0, 40.0])


@pytest.mark.parametrize(
    ("cursor", "direction", "expected"),
    [
        (25.0, -1, 20.0),  # between samples: back to the one just behind
        (25.0, +1, 30.0),  # between samples: forward to the one just ahead
        (20.0, -1, 10.0),  # on a sample
        (20.0, +1, 30.0),
        (0.0, -1, 0.0),  # clamped at the ends
        (40.0, +1, 40.0),
        (5.0, -1, 0.0),
        (35.0, +1, 40.0),
    ],
)
def test_step_lands_on_the_adjacent_sample(cursor: float, direction: int, expected: float) -> None:
    monitor = _Monitor(_TIMES)
    monitor.cursor_time_s = cursor
    controller = TimelineController(monitor)
    controller.step(direction)
    assert monitor.cursor_time_s == expected
