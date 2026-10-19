"""GPUBatchPropagator host code, checked under Numba's CUDA simulator.

* ``diagnostics_snapshot`` after a successful propagation: the simulator
  reports its device memory as ``inf``, which the snapshot cast to ``int``
  (OverflowError). A non-finite reading now counts as unknown (0), as a
  failed memory query already did.

The simulator runs the kernels in Python float64; these pin host-side
bookkeeping, not NVVM code generation.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

_PRELUDE = """
import numpy as np
from types import SimpleNamespace

from lunaris.common.batch_defs import BatchPropagationConfig
from lunaris.common.constants import MU_MOON, R_MOON
from lunaris.common.type_defs import PerturbationFlags
import lunaris.core.batch_propagator as bp
from numba import cuda

# Numba 0.64's simulator has no module-level get_current_device (the real
# CUDA target does); hand the constructor the simulator's fake device.
if not hasattr(cuda, "get_current_device"):
    cuda.get_current_device = lambda: cuda.current_context().device

def point_mass_propagator(dt_s=60.0):
    dyn = SimpleNamespace(grav=None, ephem=None, earth_j2=None, srp=None)
    cfg = BatchPropagationConfig(n_samples=2, sh_degree=0, dt_s=dt_s, impact_alt_km=0.0)
    return bp.GPUBatchPropagator(dyn, cfg, PerturbationFlags(enable_sh=False))

def circular_states(n):
    r = float(R_MOON) + 100.0e3
    v = float(np.sqrt(MU_MOON / r))
    return np.repeat(np.array([[r, 0.0, 0.0, 0.0, v, 0.0]]), n, axis=0)
"""


def _run_cudasim(body: str) -> subprocess.CompletedProcess[str]:
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["NUMBA_ENABLE_CUDASIM"] = "1"
    env["PYTHONPATH"] = str(root / "src") + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(_PRELUDE) + textwrap.dedent(body)],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _assert_ok(result: subprocess.CompletedProcess[str]) -> None:
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout, result.stdout + result.stderr


def test_diagnostics_snapshot_reads_the_simulators_unbounded_memory_as_unknown() -> None:
    script = """
    import math
    from numba import cuda

    with cuda.gpus[0]:
        free, total = cuda.current_context().get_memory_info()
    assert not (math.isfinite(free) and math.isfinite(total)), (free, total)  # the simulator's inf

    prop = point_mass_propagator()
    n = 2
    one = np.ones(n)
    t, Y, flags, t_imp = prop.propagate(circular_states(n), one, one, one, one, 120.0, 60.0)
    assert Y.shape == (3, n, 6) and np.all(np.isfinite(Y))
    diag = prop.diagnostics_snapshot()
    assert diag["gpu_free_mem_bytes"] == 0 and diag["gpu_total_mem_bytes"] == 0, diag
    assert prop.recommended_max_batch() >= 1
    print("OK")
    """
    _assert_ok(_run_cudasim(script))
