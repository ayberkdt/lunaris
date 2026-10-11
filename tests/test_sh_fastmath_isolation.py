"""Strict SH helpers must not inherit ``fastmath`` from a fastmath caller.

Numba compiles a callee with the flags of the caller that triggers its first
compilation unless the callee sets the flag itself, and it caches that build on
disk. The helpers below are shared by the strict serial SH kernel (the golden
RHS path) and by the fastmath kernels ``_compute_sh_acceleration_parallel`` and
``sh_potential_accel_batch_serial``. Without an explicit ``fastmath=False`` the
serial acceleration moved by up to 11 ulp when a fastmath kernel compiled first
(measured on a degree-200 field), and fastmath may legally cancel the Kahan
compensation term ``(s_new - s) - y`` to zero.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest

from lunaris.common import math_utils
from lunaris.physics import spherical_harmonics as sh

_SHARED_HELPERS = (
    "_imin",
    "_imax",
    "_compute_stable_m_limit",
    "_fill_longitude_trig_tables",
    "_apply_legendre_normalization",
    "_compute_legendre_polynomials_inplace",
    "_determine_effective_degree",
    "_transform_cartesian_to_spherical_basis",
    "_compute_pole_safe_inv_rho_sq",
    "_convert_spherical_gradients_to_cartesian",
    "_kahan_sum_step",
    "_axis_transverse_m1",
    "_prepare_evaluation_tables",
    "_prepare_simulation_preamble",
    "_get_optimal_block_size",
)


@pytest.mark.parametrize("name", _SHARED_HELPERS)
def test_shared_sh_helper_pins_fastmath_off(name: str) -> None:
    dispatcher = getattr(sh, name)
    assert dispatcher.targetoptions.get("fastmath") is False


def test_clamp_pins_fastmath_off() -> None:
    assert math_utils.clamp.targetoptions.get("fastmath") is False


_PROBE = textwrap.dedent(
    """
    import sys
    import numpy as np
    from lunaris.physics import spherical_harmonics as sh

    N = 60
    rng = np.random.default_rng(7)
    C = np.zeros((N + 1, N + 1)); S = np.zeros((N + 1, N + 1))
    for n in range(2, N + 1):
        C[n, : n + 1] = rng.normal(scale=1e-4 / n**2, size=n + 1)
        S[n, 1 : n + 1] = rng.normal(scale=1e-4 / n**2, size=n)
    if sys.argv[1] == "fastmath_first":
        sh.sh_potential_accel_fixed(np.array([[0.0, 0.0, 1.9e6]]), C, S, 4.9e12, 1.738e6, N)
    tabs = sh.build_legendre_coeffs(N)
    ws = sh.make_sh_workspace(N)
    pts = rng.normal(size=(16, 3)); pts *= 1.9e6 / np.linalg.norm(pts, axis=1)[:, None]
    out = [sh.sh_accel_fixed(*p, N, 1.738e6, 4.9e12, C, S, *tabs, ws.P, ws.dP, ws.cos_m, ws.sin_m) for p in pts]
    print(np.asarray(out).tobytes().hex())
    """
)


@pytest.mark.slow
def test_serial_rhs_independent_of_compile_order(tmp_path) -> None:
    """Fresh Numba caches: compiling a fastmath kernel first must not change the
    serial kernel's result by a single bit."""
    results = {}
    for order in ("serial_first", "fastmath_first"):
        env = dict(os.environ, NUMBA_CACHE_DIR=str(tmp_path / order))
        proc = subprocess.run(
            [sys.executable, "-c", _PROBE, order],
            capture_output=True, text=True, env=env, timeout=900, check=True,
        )
        results[order] = proc.stdout.strip().splitlines()[-1]
    assert results["serial_first"] == results["fastmath_first"]
