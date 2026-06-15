from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np

from lunaris.core.mc_propagator import _initial_impact_bookkeeping


def test_numba_host_preflight_marks_t0_impacts_at_and_below_boundary() -> None:
    radius = 10.0
    Y0 = np.zeros((3, 6), dtype=np.float64)
    Y0[:, 0] = [9.0, 10.0, 11.0]
    flags, times, positions = _initial_impact_bookkeeping(
        Y0,
        radius,
        detect_impact=True,
    )
    np.testing.assert_array_equal(flags, [1, 1, 0])
    np.testing.assert_allclose(times[:2], [0.0, 0.0])
    assert np.isnan(times[2])
    # t=0 impacts record their initial position; survivors stay NaN.
    np.testing.assert_allclose(positions[0], [9.0, 0.0, 0.0])
    np.testing.assert_allclose(positions[1], [10.0, 0.0, 0.0])
    assert np.isnan(positions[2]).all()


def test_numba_host_preflight_honors_disabled_detection() -> None:
    Y0 = np.zeros((2, 6), dtype=np.float64)
    flags, times, positions = _initial_impact_bookkeeping(
        Y0,
        10.0,
        detect_impact=False,
    )
    np.testing.assert_array_equal(flags, [0, 0])
    assert np.isnan(times).all()
    assert np.isnan(positions).all()


def test_numba_cuda_swept_segment_detects_tunneling_without_false_positive() -> None:
    """1B regression runs the CUDA kernel under Numba's CPU simulator."""
    root = Path(__file__).resolve().parents[1]
    script = textwrap.dedent(
        """
        import numpy as np

        from lunaris.common.constants import (
            AU,
            MU_EARTH,
            MU_SUN,
            P_SUN_1AU,
            R_EARTH_MEAN,
            R_MOON,
        )
        from lunaris.core.mc_propagator import _rk4_batch_kernel

        radius = float(R_MOON)
        dt = 60.0
        x = 0.30 * radius
        y = np.array([0.98 * radius, 1.25 * radius], dtype=np.float64)
        states = np.array(
            [
                [x, y[0], 0.0, -2.0 * x / dt, 0.0, 0.0],
                [x, y[1], 0.0, -2.0 * x / dt, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        raw_end = states[:, :3] + dt * states[:, 3:]
        assert np.all(np.linalg.norm(states[:, :3], axis=1) > radius)
        assert np.all(np.linalg.norm(raw_end, axis=1) > radius)

        zeros3 = np.zeros((1, 3), dtype=np.float64)
        q = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64)
        coeff = np.zeros((1, 1), dtype=np.float64)
        vec = np.zeros(1, dtype=np.float64)
        ones = np.ones(2, dtype=np.float64)
        flags = np.zeros(2, dtype=np.int32)
        times = np.full(2, np.nan, dtype=np.float64)
        positions = np.full((2, 3), np.nan, dtype=np.float64)

        _rk4_batch_kernel[1, 32](
            states, 0.0, dt,
            1.0, zeros3, zeros3, q, 1,
            0, radius, 0.0,
            coeff, coeff, vec, vec, vec, vec, vec,
            0, 0, 0, 0, 0, 0,
            ones, ones, ones,
            MU_SUN, MU_EARTH, R_EARTH_MEAN, 0.0, 0.0, 0.0, 1.0,
            radius, R_EARTH_MEAN, AU, P_SUN_1AU,
            flags, times, positions, 1, radius,
        )

        np.testing.assert_array_equal(flags, [1, 0])
        assert 0.0 < times[0] < dt
        assert np.isnan(times[1])
        assert abs(np.linalg.norm(positions[0]) - radius) < 1e-6
        assert np.isnan(positions[1]).all()
        np.testing.assert_allclose(states[0, :3], positions[0], atol=1e-9, rtol=0.0)
        np.testing.assert_allclose(states[1, :3], raw_end[1], atol=1e-9, rtol=0.0)
        """
    )
    env = os.environ.copy()
    env["NUMBA_ENABLE_CUDASIM"] = "1"
    src_path = str(root / "src")
    env["PYTHONPATH"] = src_path + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
