# Architecture Seam Cleanup Plan

**Status:** implementation complete for items 1-8 (SciPy runner extraction remains P3 TODO per item 5 guardrail)
**Author basis:** external senior-architect review, ingested 2026-07-07
**Branch context:** `refactor/frame-handling-and-physics` (post #96/#97)
**Scope guardrails:** preserve public behavior and numerical results; no new physics;
do not touch spacecraft attitude/geometry; keep `common` dependency-light; keep
`physics` free of `core`/`analysis`/`ui`; keep `core` free of CLI/batch orchestration.

This plan sharpens the remaining structural seams of an already-sound layered
architecture (`common â†’ physics â†’ core â†’ analysis/ui/cli`). It is a cleanup pass,
**not** a redesign. Each item below was verified against the current tree; the
"Current state" lines cite what actually exists today so we do not re-do work.

---

## Repository verification snapshot (2026-07-07)

| # | Review item | Verified current state | Done? |
|---|-------------|------------------------|-------|
| 1 | `core.propagation` public/internal boundary | Done: explicit `core.propagation` facade, trimmed `propagator.__all__`, private-helper tests repointed to canonical internal modules | done |
| 2 | Shared output-grid contract | Done: `common/time_grid_contract.py` owns `build_output_time_grid`; `batch_defs.build_batch_output_grid` remains as the compat wrapper; single-run grid uses the neutral contract | done |
| 3 | `DynamicsRequirements` in a contract module | Done: dataclass moved to `core/dynamics/contracts.py`; `preparation.py` keeps the compat re-export; `engine.py` imports from `contracts` | done |
| 3b | External-relativity derived-field recompute | Done: regression test locks recompute through the Sun-only + external-relativity downgrade path | done |
| 4 | Centralize schema/provenance constants | Done: `common/contracts/` owns batch archive, checkpoint, and diagnostics schema constants; storage/checkpoint/propagation/audit call sites are wired to the registry with legacy re-exports preserved | done |
| 5 | Split `propagate()` runners/diagnostics | Done: diagnostics assembly moved to `core/propagation/diagnostics.py`; fixed-step orchestration moved to `fixed_step_runner.py`; SciPy/chunked extraction is left as explicit P3 TODO per guardrail | done |
| 6 | ForceEvaluator design note (doc only) | Done: `docs/development/FORCE_EVALUATOR_DESIGN.md` records the future seam while preserving `DynamicsEngine` and `get_acceleration_breakdown()` as current contracts | done |
| 7 | Expand mypy coverage to batch/cli | Done: `[tool.mypy].files` includes the first clean batch/CLI subset (`batch/{backend_policy,requirements,storage,provenance}.py`, `cli/{options,common_args,run}.py`) | done |
| 8 | ARCHITECTURE verified date | Done: `docs/ARCHITECTURE.md` verified date and layer notes refreshed for the seam cleanup | done |

**Net:** items 1-8 are complete. The remaining runner extraction debt is the P3 `scipy_runner.py` follow-up explicitly scoped out by item 5 guardrail.

---

## Priority order & commit slicing

Implement in the review's order â€” it is dependency-safe and each step is a small,
reviewable commit. Suggested commit boundaries:

1. **C1** â€” `core.propagation` facade + `__all__` cleanup (+ test moves)
2. **C2** â€” `common/time_grid_contract.py` neutral module + rewires
3. **C3** â€” `DynamicsRequirements` â†’ `core/dynamics/contracts.py` (+ derived-field test)
4. **C4** â€” `common/contracts/` schema registry + rewires
5. **C5** â€” propagation runner/diagnostics split (`diagnostics.py` first, then runners)
6. **C6** â€” ForceEvaluator design note (doc only)
7. **C7** â€” mypy coverage expansion (batch subset first)
8. **C8** â€” ARCHITECTURE date + text refresh

Each commit must leave the suite green and `ruff`/`mypy` clean before the next.
**No Claude co-author trailer on any commit** (author = user only).

---

## Item 1 â€” `core.propagation` public/internal API boundary

**Problem.** `core/propagation/__init__.py` re-exports every non-dunder name of
`propagator` by iterating `dir(_impl)`, and `propagator.__all__` lists private
helpers. Both promote implementation details into package-level API.

**Target.** `core/propagation/__init__.py` explicitly exposes only:

```
propagate, PropagationResult, EventOutcome,
TimeGridPlan, StepSizePlan, IntegrationPlan,
build_events, make_time_grid,
resolve_time_grid_plan, resolve_step_size_policy,
resolve_integration_plan, event_outcome_from_solver_events
```

Replace the `for _name in dir(_impl)` loop with explicit imports + `__all__`.

**Also trim `propagator.__all__`** to the public surface. Private helpers stay
*importable from their canonical internal modules* (they already live there):
- `_rk4_step_full`, `_rk8_step_full` â†’ `integrators/rk.py`
- `_is_fixed_step_method`, `_ACCEL_METHODS`, `_RHS_METHODS`, `_accel_stepper` â†’ `integrators/fixed_step.py`
- `_composition_weights`, `_Y{4,6,8}_WEIGHTS` â†’ `integrators/symplectic.py`
- `_clamp_output_dt`, `_norm_method` â†’ `time_grid.py`

`propagator.py` may keep importing these internally; it just should not
re-advertise them in `__all__` (leaving them as module globals is fine, but the
package facade must not surface them).

**Test churn.** Any test importing a private helper *from the package facade*
(`from lunaris.core.propagation import _rk4_step_full`) must be repointed to the
canonical internal module. Search first:
```
grep -rn "from lunaris.core.propagation import" tests | grep "_"
```

**Acceptance tests** (extend `tests/test_propagation_import_compat.py`):
- `from lunaris.core import propagate` works.
- `from lunaris.core.propagation import propagate, TimeGridPlan, EventOutcome` works.
- no name in `dir(lunaris.core.propagation)` that is a public re-export starts with `_`
  (assert the facade exposes no `_`-prefixed helpers).
- canonical internal imports still resolve
  (`from lunaris.core.propagation.integrators.rk import _rk4_step_full`).

---

## Item 2 â€” Neutral output time-grid contract module

**Problem.** A single-run module (`core/propagation/time_grid.py`) imports the
grid builder from `common.batch_defs` â€” legal by layering but semantically odd.

**Target.** New `src/lunaris/common/time_grid_contract.py`:

```python
def build_output_time_grid(duration_s: float, output_dt_s: float) -> tuple[np.ndarray, int, float]:
    ...
```

Move the **exact** current implementation from `batch_defs.build_batch_output_grid`
(`batch_defs.py:49-90`) verbatim â€” same `n_snaps = max(1, ceil(duration/out_dt))`,
same `np.linspace(0, duration, n_snaps+1)`, same `snap_interval = duration/n_snaps`,
same non-positive raises. **Do not alter numerics.**

Rewire:
- `common/batch_defs.py`: `build_batch_output_grid = build_output_time_grid` (alias)
  OR a thin wrapper `def build_batch_output_grid(...): return build_output_time_grid(...)`.
  Keep the name exported (still in `batch_defs.__all__`) for backward compat.
- `core/propagation/time_grid.py:11`: import `build_output_time_grid` from the new
  module and call it inside `make_time_grid`.

**Contract to preserve** (already holds; assert it): `t[0]==0.0`, `t[-1]==duration_s`,
strictly increasing, realized interval `duration_s/ceil(duration_s/output_dt_s)`,
no overshoot, raises on non-positive `duration`/`output_dt`.

**Acceptance tests** (new `tests/test_time_grid_contract.py`, plus keep existing
batch grid tests green):
- `build_batch_output_grid` and `build_output_time_grid` return identical arrays.
- single-run `make_time_grid(0, D, dt)` matches `build_output_time_grid(D, dt)[0]`.
- exact final epoch and realized spacing unchanged vs a captured golden.

---

## Item 3 â€” `DynamicsRequirements` â†’ `core/dynamics/contracts.py`

**Problem.** The typed contract dataclass lives inside the module that also owns
providerâ†’pack preparation logic, mixing contract with implementation.

**Target.** New `src/lunaris/core/dynamics/contracts.py` holding the
`DynamicsRequirements` dataclass (lift `preparation.py:69-219` verbatim, including
`to_dict()`, all typed properties, and `without_external_relativity()`).

Rewire imports:
- `preparation.py` â†’ `from lunaris.core.dynamics.contracts import DynamicsRequirements`
- `engine.py:87` â†’ import `DynamicsRequirements` from `contracts` (not `preparation`)
- keep a compat re-export in `preparation.py` (`from .contracts import DynamicsRequirements`)
  so existing `from ...preparation import DynamicsRequirements` callers/tests don't break.

**Circular-import guard.** `contracts.py` must import only from
`common.force_requirements` (for `ForceRequirements`) and stdlib â€” no imports from
`preparation`, `engine`, or pack modules. `preparation` and `engine` import *from*
`contracts`, so the dependency is one-directional.

**Facade policy.** Keep `DynamicsRequirements` internal (importable from
`core.dynamics.contracts`); do **not** add it to a public `core` facade unless a
downstream consumer needs it as stable API.

**#3b derived-field recompute â€” already satisfied.** `ForceRequirements.without_external_relativity()`
(`force_requirements.py:53-77`) already recomputes `need_sun/need_earth/need_body_vectors/need_ephem`
and `DynamicsRequirements` delegates to it. No behavior change needed â€” just lock it
with a regression test so it can't silently rot.

**Acceptance tests** (extend `tests/test_dynamics_contracts.py`):
- `compute_requirements(...)` returns a `DynamicsRequirements`.
- `resolve_effective_requirements()` returns a **new** object (raw not mutated).
- external-relativity downgrade path: build a req with `use_rel_external=True` and a
  Sun-only force set, call `.without_external_relativity()`, assert `need_earth`
  recomputes correctly (no stale `True`) and `use_rel_external is False`.
- `engine._prep["req"]` is a `DynamicsRequirements` (already asserted at `engine.py:1124`).
- both import paths resolve (`contracts` and the `preparation` compat re-export).

---

## Item 4 â€” Centralize schema/provenance constants

**Problem.** Schema/version contracts are scattered across three layers and one is
an inline literal in a diagnostics dict.

**Target.** New package `src/lunaris/common/contracts/`:

```
common/contracts/
    __init__.py          # re-export the three constants
    batch_archive.py     # BATCH_ARCHIVE_SCHEMA_VERSION = 2
                         # REQUIRED_ARCHIVE_V2_FIELDS, REQUIRED_ARCHIVE_V2_ARRAYS
    checkpoint.py        # CHECKPOINT_SCHEMA_VERSION = 1
    diagnostics.py       # PROPAGATION_DIAGNOSTICS_SCHEMA_VERSION = 1
```

Move the canonical definitions here and rewire the current owners to import them,
**keeping backward-compatible re-exports** at the old sites (they are imported by
tests / provenance today):
- `batch/storage.py` â€” move `REQUIRED_ARCHIVE_V2_FIELDS/ARRAYS` and the literal `2`
  (`storage.py:24,38,310`) to `contracts/batch_archive.py`; re-export from `storage`
  and use `BATCH_ARCHIVE_SCHEMA_VERSION` at the write site (`storage.py:310`) and the
  validation sites (`storage.py:437-461`).
- `core/propagation/checkpoint.py:14` â€” move `CHECKPOINT_SCHEMA_VERSION` to
  `contracts/checkpoint.py`; re-export from `checkpoint.py` (it's in its `__all__`);
  writer at `checkpoint.py:99` uses the centralized constant.
- `propagator.py:672` â€” replace the inline `"diagnostics_schema_version": 1` with
  `PROPAGATION_DIAGNOSTICS_SCHEMA_VERSION` from `contracts/diagnostics.py`.

**Note (do not touch):** ST-LRPS already has its own richer contract subsystem
(`surrogate/st_lrps/shared/contracts.py`, `data/dataset_contract.py`). Leave it; add a
one-line pointer in `common/contracts/__init__.py` docstring noting ST-LRPS owns its own.

**Layering check.** `common.contracts` must stay dependency-light (constants + tuples
only, no numpy-heavy logic, no upward imports). `batch` and `core` importing *down*
into `common.contracts` is legal.

**Acceptance tests** (new `tests/test_contract_registry.py`):
- all three constants importable from `lunaris.common.contracts`.
- old locations still resolve the same value (`checkpoint.CHECKPOINT_SCHEMA_VERSION`,
  `storage.REQUIRED_ARCHIVE_V2_FIELDS`, etc.) and are identical objects/values.
- a propagation run's `diagnostics["diagnostics_schema_version"]` equals the registry value.
- checkpoint writer stamps the registry checkpoint version (round-trip via `load_propagation_checkpoint`).
- batch storage validation uses the registry field/array tuples (identity check).

---

## Item 5 â€” Split `propagate()` integration/diagnostics helpers

**Problem.** Planning is extracted, but `propagate()` (`propagator.py:153-749`) still
owns telemetry wrapping, the symplectic guard, both scipy runners, checkpoint writes,
diagnostics assembly, result assembly, and the 2-body baseline.

**Target structure** under `core/propagation/`:
```
diagnostics.py        # build_propagation_diagnostics(...) -> dict   [do FIRST]
scipy_runner.py       # normal + chunked solve_ivp orchestration
fixed_step_runner.py  # fixed-step orchestration wrapper
result_assembly.py    # EventOutcome/arrays -> PropagationResult (optional)
```

**Surgical, not a rewrite.** Minimum acceptable slice, in order:
1. **`diagnostics.py`** â€” extract the diagnostics dict build (`propagator.py:671-738`,
   including energy-drift stats and SH-degree adequacy warning) into
   `build_propagation_diagnostics(...)`. Keep **every key and value byte-identical**,
   including `diagnostics_schema_version` (wired to Item 4), `symplectic_violation`,
   `single_run_stlrps_cpu_warning`, `recommended_degree`, etc.
2. **`scipy_runner.py`** â€” move the normal (`propagator.py:469-483`) and chunked
   (`484-626`) `solve_ivp` paths. Preserve exactly: chunk masking, checkpoint-every-chunk
   modes (`latest/state/last`, `chunks/chunk`, default), terminal-endpoint preservation
   (`_terminal_event_endpoint` splice), event accumulation, integration-failure status,
   stop-file behavior, and the `SimpleNamespace` sol shape.
3. **`fixed_step_runner.py`** â€” wrap the fixed-step branch (`propagator.py:365-422`):
   6D guard, `_integrate_fixed_step` call, `EventOutcome` + `PropagationResult` assembly.
4. **`result_assembly.py`** *(optional)* â€” centralize `EventOutcome`â†’`PropagationResult`.

**Hard constraints:**
- Public `propagate()` signature unchanged.
- **No numerical change**; do not touch Numba hot loops or `_integrate_fixed_step` internals.
- All existing diagnostics keys present and equal.
- The symplectic guard and telemetry-wrap can stay in `propagate()` for slice 1;
  move only if it stays clean.

**If it grows too large:** land `diagnostics.py` + `fixed_step_runner.py` first, leave
a `# TODO(seam-cleanup #5): extract scipy_runner` marker and note it in the deferred
section here rather than forcing a risky big-bang.

**Acceptance tests** (extend `tests/test_propagator_physics.py` / `test_propagation_plans.py`):
- fixed-step run works and diagnostics keys unchanged.
- scipy run works; chunked run preserves terminal-event endpoint (existing test).
- checkpoint written for both `checkpoint_every_chunk` and final-write paths.
- `compute_2body_baseline=True` still returns a baseline `PropagationResult`.
- golden diagnostics-dict key set equals pre-refactor snapshot.

---

## Item 6 â€” ForceEvaluator design note (documentation only)

**Do not refactor** `get_acceleration_breakdown()` â€” it touches force-model hot paths.

**Target.** New `docs/development/FORCE_EVALUATOR_DESIGN.md` describing the future
structure only:
```
DynamicsEngine  -> prepares packs
ForceEvaluator  -> acceleration_total(t, y)
                -> acceleration_breakdown(t, y)   # shares one label/enable SSOT
```
Constraints to record: Numba hot path stays specialized/fast; no Python object loops
in the default hot path; breakdown labels + enable/disable policy share one source of
truth; existing `get_acceleration_breakdown()` remains valid until then. Mark **P3 /
future**, not current work.

**Acceptance:** if a docs-freshness/existence test covers `docs/development/`, add the
new file to it; otherwise no test needed.

---

## Item 7 â€” Expand mypy coverage incrementally

**Problem.** `batch` (central to backend policy, provenance, storage, sampling) and key
CLI modules are outside mypy's `files` list (`pyproject.toml:221-249`).

**Target (first expansion).** Add to `[tool.mypy].files`:
```
"src/lunaris/batch/backend_policy.py",
"src/lunaris/batch/requirements.py",
"src/lunaris/batch/storage.py",
"src/lunaris/batch/provenance.py",
"src/lunaris/cli/options.py",
"src/lunaris/cli/common_args.py",
"src/lunaris/cli/run.py",
```
If the whole `src/lunaris/batch` package types clean, prefer adding `"src/lunaris/batch"`
wholesale instead of the four files.

**Guardrail.** Do **not** introduce noisy typing churn. Run `python -m mypy src/lunaris`;
if a file explodes with errors that need real code changes, drop it from this pass, keep
the clean subset, and leave a `TODO` in this plan + a note in `docs/`. Verify actual CLI
module names before editing (`cli/run.py` etc. â€” confirm they exist).

**Acceptance:** `python -m mypy src/lunaris` clean with the expanded list.

---

## Item 8 â€” ARCHITECTURE doc verification date

**Target.** `docs/ARCHITECTURE.md:17`:
```
Last verified: 2026-07-07, covering #96/#97 frame handling, dynamics preparation,
propagation plans, event outcome, and state-vector contract refactors.
```
Also sweep the ARCHITECTURE text for statements made inaccurate by this cleanup:
- if it names the `core.propagation` facade behavior, update it to the explicit surface (Item 1);
- if it lists import-linter contracts, ensure any new module (`common/time_grid_contract.py`,
  `common/contracts/`, `core/dynamics/contracts.py`) is named **verbatim** (the
  architecture-documentation test asserts contract names appear in the doc).

**Acceptance:** `tests/test_architecture_documentation.py` green after edits.

---

## Import-linter / architecture-test watchpoints

New modules must respect existing contracts and appear in ARCHITECTURE.md if the
freshness test enumerates them:
- `common/time_grid_contract.py`, `common/contracts/*` â€” pure `common`, no upward imports.
- `core/dynamics/contracts.py` â€” imports only `common.force_requirements` + stdlib.
- `core/propagation/{diagnostics,scipy_runner,fixed_step_runner,result_assembly}.py` â€”
  stay within `core`; no CLI/batch imports.
- Compat shims (re-exports at old sites) must be **dynamic** where `ruff --fix` would
  otherwise strip a static re-export â€” see the known shim/ruff trap; verify `ruff check`
  does not remove them.

## Required test + gate commands

```
python -m pytest tests/test_propagation_import_compat.py
python -m pytest tests/test_state_vector_contract.py
python -m pytest tests/test_dynamics_contracts.py
python -m pytest tests/test_event_outcome_contract.py
python -m pytest tests/test_propagation_plans.py
python -m pytest tests/test_batch_requirements.py
python -m pytest tests/test_force_budget_sanity.py
python -m pytest tests/test_propagator_physics.py
python -m pytest tests/test_architecture_documentation.py
python -m ruff check src tests
python -m mypy src/lunaris
# full run if time permits:
python -m pytest
```
CUDA/data-dependent tests: skip per existing markers and state clearly if not run locally.

## Intentionally deferred

- **ForceEvaluator implementation** (Item 6 is a design note only â€” P3).
- **ST-LRPS artifact contracts** â€” not moved; ST-LRPS owns its own contract subsystem.
- **Full `result_assembly.py` extraction** if Item 5 is landed as the minimal slice.
- **Whole-`batch` mypy** if the four-file subset is the clean stopping point.

## Final report checklist (when implementing)

Report: (1) structural changes summary, (2) files changed, (3) public API changes,
(4) backward-compat notes, (5) tests added/updated, (6) exact commands + results,
(7) deferred work (esp. ForceEvaluator).
