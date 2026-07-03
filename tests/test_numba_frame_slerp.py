"""3B regression: the Numba CUDA quaternion interpolator is shortest-path SLERP.

Historically ``_interp4_cuda`` was an nlerp (component linear blend + renormalise)
while the Torch backend used SLERP, so the two backends rotated the body-fixed
frame along different paths between ephemeris samples. These tests run the real
device function under Numba's CPU simulator and assert it agrees with
:meth:`TorchMoonFrame.quat_i2f` — including the long-path (dot < 0) hemisphere
flip and the near-parallel branch — so both backends share one frame contract.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def _run_cudasim(body: str) -> subprocess.CompletedProcess[str]:
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["NUMBA_ENABLE_CUDASIM"] = "1"
    src_path = str(root / "src")
    env["PYTHONPATH"] = src_path + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-c", body],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_numba_interp4_matches_torch_slerp_short_long_and_parallel() -> None:
    script = """
import math

import numpy as np
import torch
from numba import cuda
from types import SimpleNamespace

from lunaris.core.batch_propagator import _interp4_cuda
from lunaris.core.torch_frame import TorchMoonFrame


@cuda.jit
def _launch(t, dt, tab, n, out):
    i = cuda.grid(1)
    if i == 0:
        _interp4_cuda(t, dt, tab, n, out)


def _norm(q):
    q = np.asarray(q, dtype=np.float64)
    return q / np.linalg.norm(q)


def _torch_ref(qa, qb, dt, t):
    ephem = SimpleNamespace(
        get_data_provider=lambda: {
            "dt_s": float(dt),
            "q_i2f_tab": np.asarray([qa, qb], dtype=np.float64),
        }
    )
    frame = TorchMoonFrame(ephem, device=torch.device("cpu"), dtype=torch.float64)
    return frame.quat_i2f(float(t)).numpy()


dt = 10.0
# (qa, qb) pairs exercising: short-path arc, long-path (dot<0) hemisphere flip,
# and a near-parallel pair that must take the nlerp branch.
half = math.sqrt(0.5)
cases = [
    (_norm([1.0, 0.0, 0.0, 0.0]), _norm([half, 0.0, 0.0, half])),          # 90 deg
    (_norm([1.0, 0.0, 0.0, 0.0]), _norm([-half, 0.0, 0.0, -half])),        # dot<0
    (_norm([0.2, 0.6, 0.1, 0.3]), _norm([-0.25, -0.55, -0.05, -0.35])),    # dot<0, oblique
    (_norm([1.0, 0.0, 0.0, 0.0]), _norm([1.0, 1e-4, 0.0, 0.0])),           # near-parallel
]
fracs = [0.0, 0.25, 0.5, 0.6, 1.0]

max_err = 0.0
for qa, qb in cases:
    for f in fracs:
        t = f * dt
        out = np.zeros(4, dtype=np.float64)
        tab = np.asarray([qa, qb], dtype=np.float64)
        _launch[1, 1](t, dt, tab, 2, out)
        ref = _torch_ref(qa, qb, dt, t)
        # Quaternions q and -q represent the same rotation; compare up to sign.
        if float(np.dot(out, ref)) < 0.0:
            ref = -ref
        err = float(np.max(np.abs(out - ref)))
        max_err = max(max_err, err)
        # Output stays a unit quaternion.
        assert abs(np.linalg.norm(out) - 1.0) < 1e-12, (qa, qb, f, out)

assert max_err < 1e-9, max_err
print("OK", max_err)
"""
    result = _run_cudasim(script)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout, result.stdout + result.stderr
