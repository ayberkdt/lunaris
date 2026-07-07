from __future__ import annotations

import importlib


def test_propagator_canonical_module_is_public_core_implementation() -> None:
    canonical = importlib.import_module("lunaris.core.propagation.propagator")
    core = importlib.import_module("lunaris.core")

    assert core.propagate is canonical.propagate


def test_propagation_responsibility_modules_import() -> None:
    events = importlib.import_module("lunaris.core.propagation.events")
    checkpoint = importlib.import_module("lunaris.core.propagation.checkpoint")
    time_grid = importlib.import_module("lunaris.core.propagation.time_grid")
    fixed_step = importlib.import_module("lunaris.core.propagation.integrators.fixed_step")
    rk = importlib.import_module("lunaris.core.propagation.integrators.rk")
    symplectic = importlib.import_module("lunaris.core.propagation.integrators.symplectic")

    assert callable(events.build_events)
    assert callable(checkpoint._atomic_save_npz)
    assert callable(checkpoint.load_propagation_checkpoint)
    assert callable(time_grid.make_time_grid)
    assert callable(fixed_step._build_fixed_stepper)
    assert callable(rk._rk4_step_full)
    assert callable(symplectic._vv_step)


def test_canonical_propagator_monkeypatch_updates_canonical_module() -> None:
    canonical = importlib.import_module("lunaris.core.propagation.propagator")
    original = canonical.solve_ivp
    sentinel = object()
    try:
        canonical.solve_ivp = sentinel
        assert canonical.solve_ivp is sentinel
    finally:
        canonical.solve_ivp = original
