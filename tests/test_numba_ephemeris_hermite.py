"""CUDA-simulator parity for schema-v2 Hermite ephemeris kernels."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_numba_cuda_hermite_position_and_derivative_match_cpu() -> None:
    root = Path(__file__).resolve().parents[1]
    script = r"""
import numpy as np
from numba import cuda

from lunaris.core.batch_propagator import _interp3_cuda, _interp3_derivative_cuda
from lunaris.physics.ephemeris import interp_vec3_hermite, interp_vec3_hermite_derivative

@cuda.jit
def launch(t, dt, p, v, pos_out, vel_out):
    if cuda.grid(1) == 0:
        _interp3_cuda(t, dt, p, v, p.shape[0], 1, pos_out)
        _interp3_derivative_cuda(t, dt, p, v, p.shape[0], 1, vel_out)

dt = 30.0
times = np.arange(8, dtype=np.float64) * dt
p = np.ascontiguousarray(np.column_stack((times**3, 2.0 * times**2, -4.0 * times + 7.0)))
v = np.ascontiguousarray(np.column_stack((3.0 * times**2, 4.0 * times, np.full_like(times, -4.0))))
for t in np.linspace(0.0, times[-1], 31):
    pos = np.zeros(3, dtype=np.float64)
    vel = np.zeros(3, dtype=np.float64)
    launch[1, 1](float(t), dt, p, v, pos, vel)
    np.testing.assert_allclose(pos, interp_vec3_hermite(float(t), dt, p, v), rtol=2e-15, atol=2e-9)
    np.testing.assert_allclose(vel, interp_vec3_hermite_derivative(float(t), dt, p, v), rtol=2e-15, atol=2e-10)
"""
    env = os.environ.copy()
    env["NUMBA_ENABLE_CUDASIM"] = "1"
    env["PYTHONPATH"] = str(root / "src") + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr

