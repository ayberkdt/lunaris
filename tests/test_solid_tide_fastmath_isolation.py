"""Strict helpers must not inherit ``fastmath`` from a fastmath caller (solid tides).

Numba compiles a callee that does not set ``fastmath`` itself with the flags of
whichever caller compiles it first, and caches that build on disk for every
later caller. ``third_body_effects.accel_solid_tide`` is ``fastmath=True`` and
calls ``solid_tides.accel_solid_tides_numba``, which the strict dynamics RHS
also calls. Before the helpers pinned ``fastmath=False``, compiling the fastmath
wrapper first changed the strict kernel's output in 8913 of 12000 components
(up to 2.7e-15 relative to the vector norm), and the change persisted through
the on-disk cache into a fresh process. ``tests/test_sh_fastmath_isolation.py``
covers the SH helpers; this file covers the solid tides and guards the whole
package against new unpinned shared helpers.
"""

from __future__ import annotations

import importlib
import os
import pkgutil
import subprocess
import sys
import textwrap
from typing import Any

import pytest

import lunaris.common
import lunaris.core.dynamics
import lunaris.physics
from lunaris.physics import solid_tides


@pytest.mark.parametrize("name", ["accel_solid_tides_numba", "solid_tide_accel_degree_numba"])
def test_shared_solid_tide_kernel_pins_fastmath_off(name: str) -> None:
    assert getattr(solid_tides, name).targetoptions.get("fastmath") is False


def _is_dispatcher(obj: Any) -> bool:
    return hasattr(obj, "py_func") and hasattr(obj, "targetoptions")


def _code_names(code: Any) -> Any:
    yield from code.co_names
    for const in code.co_consts:
        if hasattr(const, "co_names"):
            yield from _code_names(const)


def _callees(dispatcher: Any) -> list[Any]:
    g = dispatcher.py_func.__globals__
    return [g[n] for n in _code_names(dispatcher.py_func.__code__) if n in g and _is_dispatcher(g[n])]


def test_no_unpinned_helper_compiles_under_a_fastmath_caller() -> None:
    """Walk every fastmath=True dispatcher's callees; flags propagate through
    callees that do not set ``fastmath`` and stop at callees that pin it.
    A non-inline callee reached that way is shared state across callers."""
    dispatchers: dict[int, Any] = {}
    for pkg in (lunaris.common, lunaris.physics, lunaris.core.dynamics):
        for info in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + "."):
            module = importlib.import_module(info.name)
            for obj in vars(module).values():
                if _is_dispatcher(obj):
                    dispatchers[id(obj)] = obj
    fastmath_callers = [d for d in dispatchers.values() if d.targetoptions.get("fastmath") is True]
    assert any(d.__name__ == "accel_solid_tide" for d in fastmath_callers)  # the walk is not vacuous

    contaminated: set[str] = set()
    for caller in fastmath_callers:
        stack, seen = [caller], {id(caller)}
        while stack:
            for callee in _callees(stack.pop()):
                if id(callee) in seen or "fastmath" in callee.targetoptions:
                    continue
                seen.add(id(callee))
                if callee.targetoptions.get("inline", "never") != "always":
                    contaminated.add(f"{callee.py_func.__module__}.{callee.__name__} (via {caller.__name__})")
                stack.append(callee)
    assert not contaminated, sorted(contaminated)


_PROBE = textwrap.dedent(
    """
    import sys
    import numpy as np
    from lunaris.physics import solid_tides, third_body_effects

    if sys.argv[1] == "fastmath_first":
        third_body_effects.accel_solid_tide(1.9e6, 0.0, 0.0, 3.8e8, 1.0e7, 2.0e6, 3.986e14, 1.738e6, 0.024, 0.008)
    rng = np.random.default_rng(3)
    pts = rng.normal(size=(500, 3))
    pts *= (1.738e6 + rng.uniform(1e4, 2e6, size=(500, 1))) / np.linalg.norm(pts, axis=1)[:, None]
    bodies = rng.normal(size=(500, 3))
    bodies *= rng.uniform(3.5e8, 1.5e11, size=(500, 1)) / np.linalg.norm(bodies, axis=1)[:, None]
    out = [solid_tides.accel_solid_tides_numba(*p, *b, 3.986e14, 1.738e6, 0.024, 0.008, True, True)
           for p, b in zip(pts, bodies)]
    print(np.asarray(out).tobytes().hex())
    """
)


@pytest.mark.slow
def test_strict_solid_tide_independent_of_compile_order(tmp_path) -> None:
    """Fresh Numba caches, both compile orders, plus a warm-cache second process:
    the strict solid-tide kernel's output must be bit-identical in every case."""
    results = {}
    for order in ("strict_first", "fastmath_first"):
        env = dict(os.environ, NUMBA_CACHE_DIR=str(tmp_path / order))
        for run in ("cold", "warm"):
            arg = order if run == "cold" else "strict_first"
            proc = subprocess.run(
                [sys.executable, "-c", _PROBE, arg],
                capture_output=True, text=True, env=env, timeout=900, check=True,
            )
            results[(order, run)] = proc.stdout.strip().splitlines()[-1]
    assert len(set(results.values())) == 1, results
