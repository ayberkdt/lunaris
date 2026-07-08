from __future__ import annotations

import importlib

_PROPAGATION_PUBLIC_SURFACE = {
    "propagate",
    "PropagationResult",
    "EventOutcome",
    "EventSpec",
    "TimeGridPlan",
    "StepSizePlan",
    "IntegrationPlan",
    "build_events",
    "make_time_grid",
    "resolve_time_grid_plan",
    "resolve_step_size_policy",
    "resolve_integration_plan",
    "event_outcome_from_solver_events",
}


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


def test_propagation_facade_exposes_only_public_contracts() -> None:
    propagation = importlib.import_module("lunaris.core.propagation")
    core = importlib.import_module("lunaris.core")

    assert set(propagation.__all__) == _PROPAGATION_PUBLIC_SURFACE
    assert not any(name.startswith("_") for name in propagation.__all__)
    assert propagation.propagate is core.propagate

    private_helpers = {
        "_ACCEL_METHODS",
        "_RHS_METHODS",
        "_rk4_step_full",
        "_rk8_step_full",
        "_clamp_output_dt",
        "_composition_weights",
        "_norm_method",
        "_accel_stepper",
        "_is_fixed_step_method",
        "_Y4_WEIGHTS",
        "_Y6_WEIGHTS",
        "_Y8_WEIGHTS",
    }
    assert private_helpers.isdisjoint(set(dir(propagation)))


def test_propagation_public_and_internal_import_paths_resolve() -> None:
    from lunaris.core import propagate as core_propagate
    from lunaris.core.propagation import EventOutcome, EventSpec, TimeGridPlan, propagate
    from lunaris.core.propagation.integrators.rk import _rk4_step_full

    assert propagate is core_propagate
    assert TimeGridPlan.__name__ == "TimeGridPlan"
    assert EventOutcome.__name__ == "EventOutcome"
    assert EventSpec.__name__ == "EventSpec"
    assert callable(_rk4_step_full)


def test_canonical_propagator_monkeypatch_updates_canonical_module() -> None:
    canonical = importlib.import_module("lunaris.core.propagation.propagator")
    original = canonical.solve_ivp
    sentinel = object()
    try:
        canonical.solve_ivp = sentinel
        assert canonical.solve_ivp is sentinel
    finally:
        canonical.solve_ivp = original
