"""The numba CUDA 1PN device functions agree with the CPU kernels.

``_relativity_1pn_cuda`` (the central Schwarzschild term), ``_de_sitter_cuda``
and ``_external_1pn_cuda`` (the external-body term the batch kernel applies
for the Sun and the Earth) run on Numba's CPU simulator (no CUDA needed) and
are compared with ``schwarzschild_components``, ``de_sitter_components`` and
``external_1pn_components`` for lunar-orbit states and Sun and Earth
geometries (Moon-centred inertial axes, SI). The external term is the de
Sitter term alone on both paths; it used to add the differential
Schwarzschild term as well, which counted the geodetic precession twice.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_cuda_relativity_terms_match_the_cpu_kernels() -> None:
    root = Path(__file__).resolve().parents[1]
    script = """
import numpy as np
from numba import cuda

from lunaris.core.batch_propagator import _de_sitter_cuda, _external_1pn_cuda, _relativity_1pn_cuda
from lunaris.common.constants import MU_EARTH, MU_MOON, MU_SUN
from lunaris.physics.relativity_effects import (
    de_sitter_components,
    external_1pn_components,
    schwarzschild_components,
)


@cuda.jit
def launch(states, bodies, mus, central_mu, central, dsitter, external):
    k = cuda.grid(1)
    if k < states.shape[0]:
        s = states[k]
        b = bodies[k]
        _relativity_1pn_cuda(s[0], s[1], s[2], s[3], s[4], s[5], central_mu[0], central[k])
        _de_sitter_cuda(s[3], s[4], s[5], b[0], b[1], b[2], b[3], b[4], b[5], mus[k], dsitter[k])
        _external_1pn_cuda(s[0], s[1], s[2], s[3], s[4], s[5], b[0], b[1], b[2], b[3], b[4], b[5], mus[k], external[k])


rng = np.random.default_rng(7)
n = 32
states = np.zeros((n, 6))
bodies = np.zeros((n, 6))
mus = np.zeros(n)
for k in range(n):
    r = rng.normal(size=3)
    r *= (1.75e6 + rng.uniform(2e4, 3e5)) / np.linalg.norm(r)
    v = np.cross(r, rng.normal(size=3))
    v *= rng.uniform(1.5e3, 1.7e3) / np.linalg.norm(v)
    states[k] = np.concatenate((r, v))
    sun = k % 2 == 0
    direction = rng.normal(size=3)
    distance = 1.496e11 if sun else 3.84e8
    position = direction / np.linalg.norm(direction) * distance
    velocity = np.cross(position, rng.normal(size=3))
    velocity *= (2.98e4 if sun else 1.02e3) / np.linalg.norm(velocity)
    bodies[k] = np.concatenate((position, velocity))
    mus[k] = MU_SUN if sun else MU_EARTH
central_mu = np.array([MU_MOON])
central = np.zeros((n, 3))
dsitter = np.zeros((n, 3))
external = np.zeros((n, 3))
launch[1, n](states, bodies, mus, central_mu, central, dsitter, external)

worst = {"central": 0.0, "de_sitter": 0.0, "external": 0.0}
for k in range(n):
    s, b, mu = states[k], bodies[k], float(mus[k])
    ref_central = np.array(schwarzschild_components(*s, float(MU_MOON)))
    ref_dsitter = np.array(de_sitter_components(*s[3:], *b, mu))
    ref_external = np.array(external_1pn_components(*s, *b, mu))
    for name, got, ref in (("central", central[k], ref_central), ("de_sitter", dsitter[k], ref_dsitter),
                           ("external", external[k], ref_external)):
        worst[name] = max(worst[name], float(np.linalg.norm(got - ref) / np.linalg.norm(ref)))
    # The external term the batch kernel applies is the de Sitter term alone.
    assert np.array_equal(external[k], dsitter[k]), k
# Same float64 formulas, a few roundings of 1.1e-16 apart at most; 1e-13
# relative to each term's magnitude leaves margin.
assert max(worst.values()) <= 1e-13, worst
print("OK", worst)
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
        timeout=600,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout, result.stdout + result.stderr
