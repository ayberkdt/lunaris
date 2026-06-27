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
| `.venv\Scripts\python.exe -m compileall -q src\lunaris\batch src\lunaris\core\monte_carlo_engine.py tests\test_batch_import_compat.py tests\test_batch_sampling.py tests\test_batch_memory_policy.py` | Pass |
| Import smoke for `lunaris.batch.*` and `lunaris.core.monte_carlo_engine` | Pass |
| Legacy monkeypatch smoke for `_available_host_memory_bytes` through `lunaris.core.monte_carlo_engine` | Pass |

## P1 Batch Split

New canonical package:

- `lunaris.batch.engine`: `MonteCarloEngine`, `mc_entry`, `batch_entry`.
- `lunaris.batch.sampling`: random, LHS, and Sobol standard-normal designs plus state/spacecraft sampling.
- `lunaris.batch.storage`: HDF5/NPZ writers, lazy HDF5 view, result loading, archive manifest validation, result storage policy.
- `lunaris.batch.memory_policy`: host-memory probe and safety factor.
- `lunaris.batch.provenance`: file hashing and metadata JSON encode/decode helpers.
- `lunaris.batch.requirements`: ephemeris/body-vector/topography and impact-frame helpers.
- `lunaris.batch.backend_policy`: thin adapter over the existing `lunaris.core.mc_backend_policy`.
- `lunaris.batch.types`: re-exports the canonical dataclasses from `lunaris.common.montecarlo_defs`.
- `lunaris.batch.progress`: progress callback type surface for the later progress extraction.

Compatibility shim:

- `lunaris.core.monte_carlo_engine` remains importable and keeps `mc_entry` /
  `batch_entry` for the existing `pyproject.toml [project.scripts]` entry
  points.
- Public names still import from both `lunaris.batch` and
  `lunaris.core.monte_carlo_engine`.
- Private helpers currently used by tests or downstream code remain importable
  from the legacy module. `_resolve_result_storage` and `_allocate_result_buffer`
  are proxied so legacy monkeypatches against `lunaris.core.monte_carlo_engine`
  still affect the helper behavior.

Deferred:

- `batch/backends/` is still deferred; backend selection remains in
  `lunaris.core.mc_backend_policy` and concrete propagators remain in
  `lunaris.core.mc_propagator` / `lunaris.core.torch_sh_propagator`.
- The progress loop is not behaviorally extracted yet; `progress.py` only owns
  the callback type surface for now.
- Full test/lint baselines need a Python environment with `pytest`, `ruff`, and
  `mypy` installed.

## P2 Surrogate Runtime Split

New canonical package:

- `lunaris.surrogate.runtime.adapter`: moved implementation of the production
  ST-LRPS gravity runtime adapter.
- `lunaris.surrogate.runtime.artifact`: artifact/run discovery import surface.
- `lunaris.surrogate.runtime.metadata`: degree/config/path metadata helpers.
- `lunaris.surrogate.runtime.scalers`: scaler bundle helpers.
- `lunaris.surrogate.runtime.networks`: network/checkpoint construction helpers.
- `lunaris.surrogate.runtime.device`: `_require_torch` import guard surface.
- `lunaris.surrogate.runtime.gravity_provider`: `SurrogateGravityModel` surface.
- `lunaris.surrogate.runtime.force_runtime`: force-runtime facade surface.

Compatibility shim:

- `lunaris.surrogate.runtime_adapter` now aliases the canonical
  `lunaris.surrogate.runtime.adapter` module object. Old imports and module-level
  inspection continue to resolve through the historical path.
- No production code was moved into `physics` or `core`; the adapter remains in
  the surrogate subsystem. Existing lazy/failing optional torch behavior is
  preserved.

## P3 Propagation Split

New canonical package:

- `lunaris.core.propagation.propagator`: moved implementation of the propagation
  orchestration, solve_ivp path, fixed-step path, events, checkpointing, and
  telemetry helpers.
- `lunaris.core.propagation.{events,checkpoint,time_grid,telemetry,result}`:
  responsibility import surfaces over the canonical implementation.
- `lunaris.core.propagation.integrators.{scipy,fixed_step,rk,symplectic}`:
  integrator helper import surfaces.

Compatibility shim:

- `lunaris.core.propagator` aliases the canonical
  `lunaris.core.propagation.propagator` module object, not a copied facade. This
  preserves existing tests and downstream monkeypatches such as
  `lunaris.core.propagator.solve_ivp`.
- Public and private helper imports used by tests remain available from the old
  path.

## P4 Dynamics Split

New canonical package:

- `lunaris.core.dynamics.engine`: moved implementation of `DynamicsEngine`, RHS
  assembly, strict provider extraction, packs, adaptive-degree helpers, and
  surrogate bridge.
- `lunaris.core.dynamics.{requirements,gravity_pack,ephemeris_pack,perturbation_packs,adaptive_degree,surrogate_bridge,rhs,context}`:
  responsibility import surfaces over the canonical implementation.
- `lunaris.core.dynamics.rhs_numba`: placeholder surface documenting that the
  first structural split keeps jitted closures inside `engine.py` to avoid
  changing hot-loop object boundaries.

Compatibility shim:

- `lunaris.core.dynamics` is now a package. Its `__init__.py` re-exports the
  moved implementation, so `from lunaris.core.dynamics import DynamicsEngine`
  and private helper imports used by tests still work.
- The old `src/lunaris/core/dynamics.py` module was removed to avoid
  module/package shadowing.

P2-P4 smoke checks run after the split:

| Command | Result |
|---|---|
| `.venv\Scripts\python.exe -m compileall -q src\lunaris\surrogate\runtime src\lunaris\surrogate\runtime_adapter.py src\lunaris\core\propagation src\lunaris\core\propagator.py src\lunaris\core\dynamics tests\test_surrogate_runtime_import_compat.py tests\test_propagation_import_compat.py tests\test_dynamics_import_compat.py` | Pass |
| Manual import-compat calls for P2/P3/P4 new and old paths | Pass |
| Manual execution of new import-compat tests | Pass |
| P0 point-mass propagation golden after P3/P4 | Unchanged: `[1773050.703713613, 484288.345988895, 0.0, -430.426980182, 1575.856339368, 0.0]` |
| P0 batch sampling hashes after P2-P4 | Unchanged: `random=3827c2dcf7e0fc2d`, `lhs=6e2d0524cf73c496`, `sobol=4f0de7624ff5ef66`, `sobol_scrambled=327f8f9a8cf36577` |
| Surrogate metadata helper smoke | Pass: `_extract_degree_metadata({"dataset_meta": {"degree_min": 10, "requested_degree": 50}}) == (10, 50)` |
| `.venv\Scripts\python.exe -m pytest ...` | Blocked: `No module named pytest`. |
| `.venv\Scripts\python.exe -m ruff check ...` | Blocked: `No module named ruff`. |
| `.venv\Scripts\python.exe -m mypy ...` | Blocked: `No module named mypy`. |

Deferred:

- P2/P3/P4 submodules are import surfaces over moved canonical implementation
  modules in this first structural pass. Deeper physical extraction can follow
  once a full pytest/ruff/mypy environment is available.
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
- `lunaris.cli.batch`: CLI-package surface for `mc_entry` and `batch_entry`
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
| `.venv\Scripts\python.exe -m compileall -q src\lunaris\cli src\lunaris\batch src\lunaris\core\propagation src\lunaris\core\propagator.py src\lunaris\core\dynamics src\lunaris\surrogate\runtime src\lunaris\surrogate\runtime_adapter.py tests\test_cli_import_compat.py tests\test_architecture_boundaries.py` | Pass |
| Manual execution of `tests/test_cli_import_compat.py` and `tests/test_architecture_boundaries.py` test functions | Pass |
| Manual execution of P1-P4 import-compat tests | Pass |
| `lunaris.cli.main` source-contract smoke for `validate_st_lrps_model_dir` delegation | Pass |
| P0 point-mass propagation golden after P5-P6 | Unchanged: `[1773050.703713613, 484288.345988895, 0.0, -430.426980182, 1575.856339368, 0.0]` |
| P0 batch sampling hashes after P5-P6 | Unchanged: `random=3827c2dcf7e0fc2d`, `lhs=6e2d0524cf73c496`, `sobol=4f0de7624ff5ef66`, `sobol_scrambled=327f8f9a8cf36577` |
| `.venv\Scripts\python.exe -m pytest ...` | Blocked: `No module named pytest`. |
| `.venv\Scripts\python.exe -m ruff check ...` | Blocked: `No module named ruff`. |
| `.venv\Scripts\python.exe -m mypy ...` | Blocked: `No module named mypy`. |
