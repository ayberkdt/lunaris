# Modular Refactor Notes

Date: 2026-06-27
Scope: P0 baseline harness plus P1-P6 structural package splits.

## P0 Baseline

The root `AGENTS.md` referenced by the architecture skill is not present in this
checkout; only `src/lunaris/ui/web/AGENTS.md` exists and is outside this change.

Tooling environment:

| Command | Result |
|---|---|
| `python -m pytest -q` | Not run: `python` is not on PATH. |
| `.venv\Scripts\python.exe -m pytest -q` | Blocked: `No module named pytest`. |
| `.venv\Scripts\python.exe -m pytest -q -m "not slow and not requires_data and not requires_cuda"` | Blocked: `No module named pytest`. The repo markers are `requires_data` / `requires_cuda`, not `gpu`. |
| `.venv\Scripts\python.exe -m ruff check .` | Blocked: `No module named ruff`. |
| `.venv\Scripts\python.exe -m mypy src\lunaris` | Blocked: `No module named mypy`. |

Tiny numerical golden captured with the local `.venv` and no external data:

| Check | Result |
|---|---|
| 300 s point-mass propagation, `enable_sh=False`, DOP853, final `[r, v]` | `[1773050.703713613, 484288.345988895, 0.0, -430.426980182, 1575.856339368, 0.0]` |
| Batch standard-normal design hashes, `n=7`, `dim=4`, `seed=42` | `random=3827c2dcf7e0fc2d`, `lhs=6e2d0524cf73c496`, `sobol=4f0de7624ff5ef66`, `sobol_scrambled=327f8f9a8cf36577` |
| Surrogate runtime metadata load | Not captured: a run manifest exists under `outputs/training/st_lrps_train_20260619_005520`, but no checkpoint file was present in that run directory. |

P1 smoke checks run after the split:

| Command | Result |
|---|---|
| `.venv\Scripts\python.exe -m compileall -q src\lunaris\batch src\lunaris\core\batch.engine.py tests\test_batch_import_compat.py tests\test_batch_sampling.py tests\test_batch_memory_policy.py` | Pass |
| Import smoke for `lunaris.batch.*` and `lunaris.batch.engine` | Pass |
| Legacy monkeypatch smoke for `_available_host_memory_bytes` through `lunaris.batch.engine` | Pass |

## P1 Batch Split

New canonical package:

- `lunaris.batch.engine`: `BatchPropagationEngine`, `batch_entry`, `batch_entry`.
- `lunaris.batch.sampling`: random, LHS, and Sobol standard-normal designs plus state/spacecraft sampling.
- `lunaris.batch.storage`: HDF5/NPZ writers, lazy HDF5 view, result loading, archive manifest validation, result storage policy.
- `lunaris.batch.memory_policy`: host-memory probe and safety factor.
- `lunaris.batch.provenance`: file hashing and metadata JSON encode/decode helpers.
- `lunaris.batch.requirements`: ephemeris/body-vector/topography and impact-frame helpers.
- `lunaris.batch.backend_policy`: thin adapter over the existing `lunaris.batch.backend_policy`.
- `lunaris.batch.types`: re-exports the canonical dataclasses from `lunaris.common.batch_defs`.
- `lunaris.batch.progress`: progress callback type surface for the later progress extraction.

Canonical naming:

- `lunaris.batch` is the only batch orchestration surface; `batch_entry` in
  `lunaris.batch.engine` backs the `lunaris-batch` console script via
  `lunaris.cli.batch`.
- The canonical public types are `BatchPropagationConfig` /
  `BatchPropagationEngine` / `BatchPropagationResult`; there are no alternate
  aliases.
- Storage helpers (`_resolve_result_storage`, `_allocate_result_buffer`,
  writers, `load_batch_result`) live in `lunaris.batch.storage`.

Deferred:

- `batch/backends/` is still deferred; backend selection remains in
  `lunaris.batch.backend_policy` and concrete propagators remain in
  `lunaris.core.batch_propagator` / `lunaris.core.torch_sh_propagator`.
- The progress loop is not behaviorally extracted yet; `progress.py` only owns
  the callback type surface for now.
- Full test/lint baselines need a Python environment with `pytest`, `ruff`, and
  `mypy` installed.

## P2 Surrogate Runtime Split

New canonical package:

- `lunaris.surrogate.runtime.adapter`: compatibility facade only (62 LOC).
- `lunaris.surrogate.runtime.artifact`: artifact/run discovery helpers.
- `lunaris.surrogate.runtime.metadata`: degree/config/path metadata helpers and
  `SurrogateGravityMetadata`.
- `lunaris.surrogate.runtime.scalers`: scaler vector/bundle normalization.
- `lunaris.surrogate.runtime.networks`: network/checkpoint construction helpers.
- `lunaris.surrogate.runtime.device`: lazy torch import guard and torch module
  handles.
- `lunaris.surrogate.runtime.gravity_provider`: `SurrogateGravityModel` runtime
  provider (582 LOC).
- `lunaris.surrogate.runtime.force_runtime`: lazy bridge to the canonical
  `st_lrps.runtime.force_model` loader.

Compatibility cleanup:

- The temporary `lunaris.surrogate.runtime_adapter` shim was removed after the
  runtime split. Use `lunaris.surrogate.runtime` imports.
- No production code was moved into `physics` or `core`; the adapter remains in
  the surrogate subsystem. Existing lazy/failing optional torch behavior is
  preserved.
- The previous 1153-LOC adapter implementation is now split across the modules
  above; `adapter.py` is no longer the god-module.

## P3 Propagation Split

New canonical package:

- `lunaris.core.propagation.propagator`: propagation orchestration and
  solve_ivp-facing public API (727 LOC).
- `lunaris.core.propagation.{events,checkpoint,time_grid,telemetry,result}`:
  event detection/refinement, checkpoint IO, time-grid policy, telemetry, and
  result helpers.
- `lunaris.core.propagation.integrators.{scipy,fixed_step,rk,symplectic}`:
  scipy method normalization, fixed-step driver, RK steppers, and symplectic
  steppers.

Compatibility cleanup:

- The temporary `lunaris.core.propagator` shim was removed after the propagation
  split. Use `lunaris.core.propagation.propagator` imports.
- Tests and downstream monkeypatches should target the canonical propagation
  module.
- The previous 1981-LOC propagator implementation is now split; event,
  checkpoint, time-grid, telemetry, and fixed-step integrator code are no longer
  stub re-exports.

## P4 Dynamics Split

New canonical package:

- `lunaris.core.dynamics.engine`: `DynamicsEngine`, RHS assembly, and the
  jitted RHS closures (1961 LOC).
- `lunaris.core.dynamics.requirements`: strict gravity/ephemeris/surface
  provider extraction and spacecraft/ephemeris requirement helpers.
- `lunaris.core.dynamics.{gravity_pack,ephemeris_pack,perturbation_packs}`:
  validated data packs used by RHS construction.
- `lunaris.core.dynamics.adaptive_degree`: adaptive SH degree and albedo DN
  sampling kernels.
- `lunaris.core.dynamics.surrogate_bridge`: surrogate gravity provider
  detection.
- `lunaris.core.dynamics.{rhs,context}`: compatibility/facade surfaces for the
  current engine-owned RHS assembly.
- `lunaris.core.dynamics.rhs_numba`: placeholder surface documenting that the
  jitted closures intentionally remain inside `engine.py` for now to avoid
  changing hot-loop object boundaries.

Compatibility shim:

- `lunaris.core.dynamics` is now a package. Its `__init__.py` re-exports the
  moved implementation, so `from lunaris.core.dynamics import DynamicsEngine`
  and private helper imports used by tests still work.
- The old `src/lunaris/core/dynamics.py` module was removed to avoid
  module/package shadowing.
- The previous 2739-LOC implementation has a real reduction: strict contracts,
  packs, adaptive kernels, and surrogate detection moved out of `engine.py`.
  Deeper RHS extraction is intentionally deferred because it touches the Numba
  hot path and needs stronger performance/physics regression coverage.

P2-P4 smoke checks run after the split:

| Command | Result |
|---|---|
| `.venv\Scripts\python.exe -m compileall -q src\lunaris\surrogate\runtime src\lunaris\core\propagation src\lunaris\core\dynamics tests\test_surrogate_runtime_import_compat.py tests\test_propagation_import_compat.py tests\test_dynamics_import_compat.py` | Pass |
| Manual import calls for P2/P3/P4 canonical paths | Pass |
| Manual execution of new import-compat tests | Pass |
| P0 point-mass propagation golden after P3/P4 | Unchanged: `[1773050.703713613, 484288.345988895, 0.0, -430.426980182, 1575.856339368, 0.0]` |
| P0 batch sampling hashes after P2-P4 | Unchanged: `random=3827c2dcf7e0fc2d`, `lhs=6e2d0524cf73c496`, `sobol=4f0de7624ff5ef66`, `sobol_scrambled=327f8f9a8cf36577` |
| Surrogate metadata helper smoke | Pass: `_extract_degree_metadata({"dataset_meta": {"degree_min": 10, "requested_degree": 50}}) == (10, 50)` |
| `.venv\Scripts\python.exe -m pytest ...` | Blocked: `No module named pytest`. |
| `.venv\Scripts\python.exe -m ruff check ...` | Blocked: `No module named ruff`. |
| `.venv\Scripts\python.exe -m mypy ...` | Blocked: `No module named mypy`. |

Deferred:

- P4 `rhs.py`, `rhs_numba.py`, and parts of `context.py` remain facade or
  placeholder surfaces. Moving the jitted RHS closure out of `engine.py` is the
  next logical decomposition step, but it should be done with a full
  pytest/ruff/mypy environment plus physics/performance checks.
- No finite-difference or CPU/GPU physics validation was added because this pass
  intentionally moved code without changing kernels, frames, signs, or
  integrator algorithms. The point-mass RHS and short propagation golden are the
  numerical checks run in this environment.

## P5 CLI Split

New canonical modules:

- `lunaris.cli.options`: parser construction and argument validation for the
  main `lunaris` command.
- `lunaris.cli.summary`: run-summary and time-step summary helpers.
- `lunaris.cli.run`: runtime wiring for the main propagation command.
- `lunaris.cli.batch`: CLI-package surface for `batch_entry` and `batch_entry`
  without changing the historical console-script targets.

Compatibility:

- `lunaris.cli.main` remains the public facade and `pyproject.toml` still points
  `lunaris = "lunaris.cli.main:main_entry"`.
- Existing imports such as `lunaris.cli.main.parse_args`,
  `lunaris.cli.main.init_ephemeris`, and `lunaris.cli.main.print_summary`
  re-export the split-module implementations.
- The canonical ST-LRPS model-directory validation remains delegated to
  `validate_st_lrps_model_dir` inside `cli.options`.

## P6 Docs And Architecture Guards

Updated or added:

- `docs/ARCHITECTURE.md`: new `batch/`, `core/propagation/`,
  `core/dynamics/`, `surrogate/runtime/`, and split-CLI layout.
- `docs/backend_matrix.md`: backend capability matrix and provenance field
  reference, cross-linked to the executable registry and policy modules.
- `tests/test_architecture_boundaries.py`: AST-based dependency guard for
  `common`, `physics`, and `core`, plus public import smoke for the new
  refactor surfaces.
- `tests/test_cli_import_compat.py`: old `lunaris.cli.main` facade imports stay
  identity-compatible with the split CLI modules.

P5-P6 smoke checks run after the split:

| Command | Result |
|---|---|
| `.venv\Scripts\python.exe -m compileall -q src\lunaris\cli src\lunaris\batch src\lunaris\core\propagation src\lunaris\core\dynamics src\lunaris\surrogate\runtime tests\test_cli_import_compat.py tests\test_architecture_boundaries.py` | Pass |
| Manual execution of `tests/test_cli_import_compat.py` and `tests/test_architecture_boundaries.py` test functions | Pass |
| Manual execution of P1-P4 import-compat tests | Pass |
| `lunaris.cli.main` source-contract smoke for `validate_st_lrps_model_dir` delegation | Pass |
| P0 point-mass propagation golden after P5-P6 | Unchanged: `[1773050.703713613, 484288.345988895, 0.0, -430.426980182, 1575.856339368, 0.0]` |
| P0 batch sampling hashes after P5-P6 | Unchanged: `random=3827c2dcf7e0fc2d`, `lhs=6e2d0524cf73c496`, `sobol=4f0de7624ff5ef66`, `sobol_scrambled=327f8f9a8cf36577` |
| `.venv\Scripts\python.exe -m pytest ...` | Blocked: `No module named pytest`. |
| `.venv\Scripts\python.exe -m ruff check ...` | Blocked: `No module named ruff`. |
| `.venv\Scripts\python.exe -m mypy ...` | Blocked: `No module named mypy`. |
