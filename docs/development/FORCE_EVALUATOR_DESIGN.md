# ForceEvaluator Design Note

Status: P3 design note only. This document records the intended seam; it does
not authorize a runtime migration in the current cleanup pass.

## Goal

The current force stack is centered on `DynamicsEngine`: it prepares the gravity
model, perturbation flags, ephemeris objects, and hot-path RHS packs used by the
propagator. That remains the production contract today.

A future `ForceEvaluator` can make the force boundary more explicit without
changing the numerical kernels:

```text
DynamicsEngine  -> prepares model packs, flags, ephemeris, frame helpers
ForceEvaluator  -> acceleration_total(t, y)
ForceEvaluator  -> acceleration_breakdown(t, y)
Propagator      -> consumes acceleration_total through RHS/integrator adapters
Diagnostics/UI  -> consume acceleration_breakdown for labeled force budgets
```

## Non-Goals For This Cleanup

- Do not implement `ForceEvaluator` in this phase.
- Do not move Numba kernels or alter the default propagation hot path.
- Do not change `DynamicsEngine.build_rhs()` behavior.
- Do not replace the existing `get_acceleration_breakdown()` API.

## Constraints

The default hot path must stay specialized. Numba-compiled RHS and acceleration
kernels should keep using typed packs and direct function calls; a generic
Python-object loop over force providers is not acceptable in the default
propagation path.

Labels and enable/disable policy need one source of truth. The future evaluator
should not invent separate names for the same perturbations already exposed by
the capability registry, UI labels, backend policy, and diagnostics.

`DynamicsEngine` remains the owner of preparation. A `ForceEvaluator` may wrap
prepared packs, but it should not duplicate ephemeris loading, gravity-model
selection, or backend capability decisions.

`get_acceleration_breakdown()` remains valid until a migration lands. Any future
replacement must preserve the existing ability to inspect total acceleration and
per-force contributions for diagnostics, UI force probes, and regression tests.

## Migration Shape

1. Introduce an internal evaluator adapter backed by the existing
   `DynamicsEngine` preparation outputs.
2. Route diagnostic force-breakdown calls through the adapter first, keeping the
   existing public function as a compatibility facade.
3. Only after benchmark parity, consider passing the evaluator into propagation
   runners. Fixed-step and SciPy paths must continue to see the same acceleration
   values and event behavior.

Acceptance for a real implementation is physical equivalence against current
CPU references, unchanged hot-path allocation behavior, and explicit tests for
total acceleration, labeled breakdown, and disabled-force policy.
