# Modular Architecture Refactor Plan

Status: **APPLIED** (P0-P6 applied 2026-06-27; local smoke/golden checks passed; full pytest/ruff/mypy unavailable in this venv)
Owner: Ayberk
Date: 2026-06-27
Scope: structural refactor only — **no numerical, CLI, or workflow behavior changes**

---

## 0. Guardrails (apply to every phase)

These are non-negotiable acceptance constraints. Re-check them at the end of each phase.

1. Preserve numerical behavior (no algorithm edits, no default-tolerance edits).
2. Preserve public CLI commands and `[project.scripts]` entry points.
3. Preserve existing import paths via compatibility shims.
4. Do not rename user-facing commands.
5. Do not change default `SimConfig` / simulation settings.
6. Do not silently change backend selection / fallback behavior.
7. Do not add new **required** dependencies (optional stays optional: torch, PySide6, h5py, numba, spiceypy).
8. Do not break packaging (`[tool.setuptools.packages.find] include = ["lunaris*"]` must still capture all new subpackages — it does, since they all live under `lunaris/`).
9. No large algorithmic rewrites.
10. Incremental, independently-testable commits — one target per phase, shim + tests land in the same commit.

**Definition of done for "no behavior change":** the existing test suite passes identically before and after each phase, plus the new import-compatibility tests pass. We do **not** claim numerical equivalence beyond what the tests actually exercise.

---

## 1. Current state (measured 2026-06-27)

Oversized modules before the refactor (line counts):

| File | LOC | Target |
|---|---|---|
| `src/lunaris/core/dynamics.py` | 2739 | Target 2 |
| `src/lunaris/core/monte_carlo_engine.py` | 2106 | Target 1 |
| `src/lunaris/core/propagator.py` | 1981 | Target 3 |
| `src/lunaris/core/mc_propagator.py` | 1958 | (out of scope this pass) |
| `src/lunaris/core/events.py` | 1211 | (consumed by Target 3) |
| `src/lunaris/surrogate/runtime_adapter.py` | 1153 | Target 4 |
| `src/lunaris/cli/main.py` | 853 | Target 5 (only if needed) |
| `src/lunaris/core/config.py` | 540 | Target 6 (leave mostly alone) |

Post-refactor measured state (same date):

| Target | Main compatibility/canonical file after split | LOC | Real code moved to |
|---|---:|---:|---|
| P1 batch | `src/lunaris/core/monte_carlo_engine.py` shim | 121 | `lunaris/batch/{engine,storage,sampling,provenance,requirements,...}.py` |
| P1 batch engine | `src/lunaris/batch/engine.py` | 1119 | Storage/sampling/provenance/requirements/memory policy are separate modules |
| P2 surrogate runtime facade | `src/lunaris/surrogate/runtime/adapter.py` | 62 | `runtime/{artifact,metadata,scalers,networks,gravity_provider,force_runtime,device}.py` |
| P3 propagation orchestration | `src/lunaris/core/propagation/propagator.py` | 727 | `propagation/{events,checkpoint,time_grid,telemetry,result,integrators/*}.py` |
| P4 dynamics engine | `src/lunaris/core/dynamics/engine.py` | 1961 | `dynamics/{requirements,gravity_pack,ephemeris_pack,perturbation_packs,adaptive_degree,surrogate_bridge}.py` |
| P5 CLI facade | `src/lunaris/cli/main.py` | 53 | `cli/{options,run,summary,batch}.py` |

Note: P4 intentionally keeps the RHS assembly and jitted closure inside
`dynamics/engine.py`; only validation helpers, data packs, adaptive kernels, and
the surrogate-provider bridge were moved. This is a real reduction, but the
remaining engine is still a large hot-path module and should be split further
only with stronger physics/performance regression coverage.

Relevant existing structure (do **not** duplicate these):

- `src/lunaris/physics/` **already holds the force math**: `third_body_effects.py`, `solar_effects.py`, `solid_tides.py`, `lunar_albedo.py`, `thermal_ir.py`, `relativity_effects.py`, `spherical_harmonics.py`, `torch_spherical_harmonics.py`, `ephemeris.py`, `surface_effects.py`, `surrogate_gravity.py`.
- `src/lunaris/analysis/monte_carlo/` already exists (`plotting.py`, `result_audit.py`, `statistics.py`) — **post-processing**, distinct from the engine. The new `batch/` package must not absorb or shadow it.
- `src/lunaris/common/` is already lightweight (`constants`, `hashing`, `lunar_data`, `math_utils`, `batch_defs`, `montecarlo_defs`, `paths`, `time_utils`, `type_defs`).

Entry points that pin internal modules (`pyproject.toml [project.scripts]`):

```
lunaris-mc    = "lunaris.cli.batch:mc_entry"
lunaris-batch = "lunaris.cli.batch:batch_entry"
lunaris       = "lunaris.cli.main:main_entry"
... (lunaris-ui/-train/-eval/-benchmark/-data via lunaris.cli.entrypoints)
```

**Consequence:** `mc_entry` and `batch_entry` must remain importable from `lunaris.core.monte_carlo_engine` (re-export from the shim), while packaging resolves through the canonical CLI wrapper.

---

## 2. Phase ordering & rationale

Ordered lowest-risk-highest-value first; each phase is shippable on its own.

| Phase | Target | Risk | Why this order |
|---|---|---|---|
| **P0** | Baseline harness | none | Lock a behavior baseline before touching code |
| **P1** | Target 1 — `batch/` (monte_carlo_engine) | medium | Highest value; clean responsibility seams; entry points need care |
| **P2** | Target 4 — `surrogate/runtime/` (runtime_adapter) | low–med | Self-contained, optional-torch boundary, good test coverage payoff |
| **P3** | Target 3 — `core/propagation/` (propagator) | medium | Many private helpers, integrators isolate cleanly |
| **P4** | Target 2 — `core/dynamics/` (dynamics) | **high** | Hot Numba path, frame conventions, surrogate bridge — do last |
| **P5** | Target 5 — `cli/` split | low | Only if `main.py` still feels unwieldy after P1–P4 |
| **P6** | Docs + architecture tests | low | Lock in the new boundaries |

Target 6 (config.py): **explicitly skipped** this pass except for trivially-safe extractions, per the review.

---

## P0 — Baseline harness (do first)

Goal: be able to prove "no behavior change" cheaply after each phase.

Steps:
1. Run and record the current suite:
   - `python -m pytest -q` (full)
   - `python -m pytest -q -m "not slow and not gpu"` (fast subset, confirm markers exist first via `pyproject.toml [tool.pytest.ini_options]`)
   - `ruff check .`
   - `mypy` only if already wired and fast.
2. Capture a tiny numerical golden baseline (scratchpad, not committed unless useful as a test):
   - one short classical-SH propagation → save final `[r, v]`,
   - one small batch run (random + LHS + Sobol, fixed seed) → save sample-design hashes / array shapes,
   - one surrogate runtime metadata load (if an artifact is available).
3. Re-run these exact checks at the end of every phase. Any diff in the golden values is a regression to fix before merging the phase.

Deliverable: `docs/refactor_notes.md` started, with the baseline command list + results table.

---

## P1 — Target 1: split `monte_carlo_engine.py` → `lunaris/batch/`

### Responsibility map (from current file)

| Concern | Current symbols | New home |
|---|---|---|
| Sampling (normal design, init states, sc props, Sobol/LHS) | `generate_standard_normal_design`, `sample_initial_states`, `sample_spacecraft_props`, `_sobol_size_note` | `batch/sampling.py` |
| Storage / archive IO (HDF5, NPZ, views, writers, loader) | `HDF5TrajectoryView`, `_HDF5Writer`, `_NPZWriter`, `_make_writer`, `_resolve_result_storage`, `_allocate_result_buffer`, `load_mc_result`, `_validate_archive_v2_manifest`, `_infer_valid_mask_from_dataset` | `batch/storage.py` |
| Memory / RAM budgeting | `_available_host_memory_bytes` (+ chunk-size logic inside engine) | `batch/memory_policy.py` |
| Provenance / metadata encode-decode | `_sha256_file`, `_metadata_value_to_jsonable`, `_decode_archive_metadata`, `_decode_metadata_value`, `_active_physics_capabilities` | `batch/provenance.py` |
| Ephemeris/requirement prep helpers | `_need_ephemeris`, `_need_body_vectors`, `_build_ephemeris_manager`, `_surface_topography_requested`, `_impact_positions_fixed`, `_state_to_array` | `batch/requirements.py` (or fold into engine if small) |
| Backend selection / CPU fallback | (currently delegates to `core/mc_backend_policy.py` + `core/backend_capabilities.py`) | `batch/backend_policy.py` = thin re-export/adapter; **do not move** the existing policy yet |
| Orchestration | `MonteCarloEngine`, `mc_entry`, `batch_entry` | `batch/engine.py` |
| Shared dataclasses/types | result/run types (`BatchPropagationResult` / `MCRunResult`, etc., sourced from `common/batch_defs` with legacy aliases) | `batch/types.py` (re-export, don't fork) |

### Target layout
```
src/lunaris/batch/
  __init__.py          # public surface re-exports
  engine.py            # MonteCarloEngine, mc_entry, batch_entry
  sampling.py
  storage.py
  memory_policy.py
  backend_policy.py     # adapter over existing core.mc_backend_policy (no logic move yet)
  provenance.py
  progress.py          # progress reporting extracted from engine loop
  requirements.py
  types.py
```
(Backend subpackage `batch/backends/{cpu,numba_cuda_sh,torch_cuda_sh}.py` is **deferred** — backend kernels already live in `core/torch_*`/`mc_backend_policy`. Only create it if P1 reveals a clean seam; otherwise leave a TODO in `refactor_notes.md`.)

### Compatibility shim
Rewrite `src/lunaris/core/monte_carlo_engine.py` to:
```python
from lunaris.batch.engine import MonteCarloEngine, mc_entry, batch_entry
from lunaris.batch.storage import HDF5TrajectoryView, load_mc_result
from lunaris.batch.sampling import (
    generate_standard_normal_design, sample_initial_states, sample_spacecraft_props,
)
# ... every name external code / tests / entry points currently import
__all__ = [...]
```
Entry points in `pyproject.toml` point at `lunaris.cli.batch:mc_entry` / `:batch_entry`; the historical `lunaris.core.monte_carlo_engine` and `lunaris.core.mc_runner` paths remain import-compatible shims.

### Tests (add)
- `tests/test_batch_import_compat.py`: every public symbol importable from **both** `lunaris.core.monte_carlo_engine` and `lunaris.batch`.
- `tests/test_batch_sampling.py`: random design shape + determinism with fixed seed; LHS/Sobol paths guarded by dependency availability (`pytest.importorskip`).
- `tests/test_batch_memory_policy.py`: chunk/budget math on small synthetic sizes.
- Reuse existing MC tests unchanged as the behavior oracle.

### Acceptance
- Archive format byte-compatible (P0 golden hashes unchanged).
- CPU fallback + provenance strings unchanged.
- `lunaris-mc --help` / `lunaris-batch --help` unchanged.

---

## P2 — Target 4: split `runtime_adapter.py` → `surrogate/runtime/`

### Responsibility map
| Concern | Current symbols | New home |
|---|---|---|
| Artifact discovery | `_is_valid_surrogate_run`, `_find_checkpoint_for_run`, `find_checkpoint_for_st_lrps_run`, `_looks_like_lunar_run`, `discover_st_lrps_model_dirs`, `find_latest_st_lrps_model_dir` | `runtime/artifact.py` |
| Metadata extraction | `_extract_degree_metadata`, `_config_path_value`, `_resolve_baseline_gravity_path`, `SurrogateGravityMetadata` | `runtime/metadata.py` |
| Scalers | `_ScaleVector`, `_ScalerBundle`, `_normalize_scale_mapping`, `_load_scaler_bundle` | `runtime/scalers.py` |
| Network construction | `_build_model_from_config`, `_extract_state_dict`, `_load_checkpoint` | `runtime/networks.py` |
| Device selection | (device logic inside model class) | `runtime/device.py` |
| Torch import guard | `_require_torch` | `runtime/device.py` or `runtime/__init__.py` |
| Gravity provider runtime + force_direct/potential_autograd | `SurrogateGravityModel` (predict total/residual accel) | `runtime/gravity_provider.py` + `runtime/force_runtime.py` |
| Public adapter facade | (module-level API) | `runtime/adapter.py` |

### Target layout
```
src/lunaris/surrogate/runtime/
  __init__.py
  adapter.py
  artifact.py
  metadata.py
  scalers.py
  networks.py
  gravity_provider.py
  force_runtime.py
  device.py
```

### Compatibility shim
`src/lunaris/surrogate/runtime_adapter.py` re-exports all current public names from `lunaris.surrogate.runtime.*`.

### Critical behavior to preserve (verify with tests)
- residual-potential, absolute-potential, force_direct, and potential_autograd artifact paths all behave as before.
- **force_direct must NOT silently fall back to scalar-potential legacy behavior** (currently forbidden — keep the guard/raise).
- device selection order unchanged.
- body / fixed-frame compatibility checks unchanged.
- Training/eval modules must **not** become import-time runtime deps (keep `_require_torch` lazy; no `import torch` at module top of the runtime facade).

### Tests (add)
- `tests/test_surrogate_runtime_import_compat.py`: old + new import paths.
- `tests/test_surrogate_runtime_metadata.py`: metadata parsing from a synthetic config dict.
- `tests/test_surrogate_runtime_artifact_validation.py`: invalid run dir → expected failure mode; force_direct-no-fallback assertion.

---

## P3 — Target 3: split `propagator.py` → `core/propagation/`

### Responsibility map
| Concern | Current symbols | New home |
|---|---|---|
| Public propagate orchestration | `propagate`, `_compute_2body_baseline` | `propagation/propagator.py` |
| Result packaging | telemetry/result dict builders (`_make_telem_dict`) + `PropagationResult` | `propagation/result.py` + `propagation/telemetry.py` |
| Events / impact | `build_events`, `_wrap_event_first6`, `_terminal_event_endpoint`, `_find_event_index`, `_event_crossed`, `_refine_event_time_bisect`, impact/lat-lon helpers, surface radius sampler | `propagation/events.py` (coordinate with existing `core/events.py`) |
| Checkpointing | `_atomic_save_npz`, `_stop_requested`, checkpoint/resume logic | `propagation/checkpoint.py` |
| Time grid / max-step | `make_time_grid`, `_clamp_output_dt`, Nyquist max-step (`_get_ref_radius_and_mu`, `_get_sh_degree`) | `propagation/time_grid.py` |
| scipy integrator | `_resolve_scipy_method`, solve_ivp call | `propagation/integrators/scipy.py` |
| fixed-step driver | `_integrate_fixed_step`, `_build_fixed_stepper`, `_accel_stepper`, `_norm_method`, `_is_*_method`, `_fixed_step_requires_6d` | `propagation/integrators/fixed_step.py` |
| RK steppers | `_rk4_step_full`, `_rk8_step_full`, `_rkn4_step`, `_modified_midpoint` | `propagation/integrators/rk.py` |
| Symplectic steppers | `_vv_step`, `_composition_weights`, `_composed_step`, `_y4/_y6/_y8_step`, `_pefrl_step`, `_pack6` | `propagation/integrators/symplectic.py` |

### Target layout
```
src/lunaris/core/propagation/
  __init__.py
  propagator.py
  result.py
  events.py
  checkpoint.py
  telemetry.py
  time_grid.py
  integrators/
    __init__.py
    scipy.py
    fixed_step.py
    rk.py
    symplectic.py
```

### Compatibility shim
`src/lunaris/core/propagator.py` re-exports `Propagator`/`propagate`/`PropagationResult`/`make_time_grid`/`build_events` etc.
> Note: confirm whether the public symbol is `Propagator` (class) or `propagate` (fn) — the current file exposes `propagate(...)`. Shim must re-export **whatever the tests/CLI currently import**; grep before writing the shim.

### Behavior to preserve
- solve_ivp default tolerances + max-step logic unchanged.
- fixed-step / symplectic **6D-only** limitation preserved (`_fixed_step_requires_6d`).
- event + impact detection + bisection refinement unchanged.
- checkpoint NPZ format + atomic-save unchanged.

### Tests (add)
- import-compat for `Propagator`/`propagate` + `events`/`checkpoint` submodules.
- minimal fixed-step propagation if fixtures allow (else import smoke).

---

## P4 — Target 2: split `dynamics.py` → `core/dynamics/` (do LAST, highest risk)

### Key insight
Force *math* already lives in `lunaris/physics/`. `dynamics.py` holds the **assembly/prep packs** that adapt those into the RHS. So this split is mostly about separating *prep* (object-heavy, runs once) from the *hot RHS* (Numba-friendly, runs every step). **Do not push Python objects into the Numba inner loop.**

### Responsibility map
| Concern | Current symbols | New home |
|---|---|---|
| Engine facade | `DynamicsEngine` | `dynamics/engine.py` |
| Strict requirement extraction/validation | `extract_gravity_strict`, `extract_ephem_tables_strict`, `extract_surface_provider_strict`, `need_ephemeris`, `require_srp_props`, `_require_attr`, `_as_f64_c` | `dynamics/requirements.py` |
| Gravity pack | `_GravPack` | `dynamics/gravity_pack.py` |
| Ephemeris pack | `_EphemPack` | `dynamics/ephemeris_pack.py` |
| Perturbation packs | `_AlbedoPack`, `_EarthJ2Pack`, `_TidePack`, `_ThermalPack` | `dynamics/perturbation_packs.py` (or one file each under `dynamics/packs/`) |
| Adaptive SH degree policy | `_select_adaptive_sh_degree`, `_sample_albedo_dn_scaled` | `dynamics/adaptive_degree.py` |
| Surrogate gravity bridge | `_is_surrogate_gravity_provider` + surrogate RHS path | `dynamics/surrogate_bridge.py` |
| RHS assembly (Python) | RHS builder | `dynamics/rhs.py` |
| RHS Numba kernel | numba-jitted inner loop | `dynamics/rhs_numba.py` |
| Shared context/dtos | `PerturbationFlags`, packs container | `dynamics/context.py` |

### Target layout (package replacing the module)
```
src/lunaris/core/dynamics/
  __init__.py          # exports DynamicsEngine (+ all currently-public names)
  engine.py
  rhs.py
  rhs_numba.py
  context.py
  requirements.py
  gravity_pack.py
  ephemeris_pack.py
  perturbation_packs.py   # albedo/earth_j2/tides/thermal
  surrogate_bridge.py
  adaptive_degree.py
```
> Packaging note: replacing `dynamics.py` with `dynamics/` directory is safe under `packages.find` (still `lunaris*`). `from lunaris.core.dynamics import DynamicsEngine` keeps working because `__init__.py` re-exports it. **Delete the old `dynamics.py` in the same commit** to avoid module/package shadowing.

### Behavior to preserve (test each)
- Frame conventions: integration frame (inertial/MCI/J2000-like), gravity fixed frame (Moon-fixed), state `[r, v]` (+ optional mass where supported).
- Numba inner-loop performance characteristics (no object calls inside jitted loop) — keep `rhs_numba.py` free of Python-object args.
- Surrogate path behavior identical.
- Strict validation of required ephemeris/surface/spacecraft props — keep fail-fast raises.
- Point-mass fallback where currently intended.
- Adaptive SH degree selection identical.

### Tests (add)
- `from lunaris.core.dynamics import DynamicsEngine` import-compat.
- one RHS-evaluation golden: same `[r,v,t]` input → same accel before/after (numerical equality check — this is the one place we *can* assert numerical equivalence).

---

## P5 — Target 5: `cli/main.py` split (only if still needed)

Re-evaluate after P1–P4. If `main.py` is still large/mixed, extract per-command modules:
```
src/lunaris/cli/{run,batch,train,eval,benchmark,options,summary}.py
```
Rules: preserve every `[project.scripts]` entry point and all argparse args/help; CLI stays a thin adapter (no physics). Keep `lunaris.cli.entrypoints` (the wiring for `lunaris-ui/-train/-eval/...`) intact.

---

## P6 — Docs + architecture tests

Update/create:
- `docs/ARCHITECTURE.md` — new `batch/`, `core/propagation/`, `core/dynamics/`, `surrogate/runtime/` layout + layer dependency rules.
- `docs/backend_matrix.md` — backend capability matrix (create; cross-link `mc_backend_policy`/`backend_capabilities`).
- `docs/refactor_notes.md` — compat shims list, deferred items (batch/backends subpackage), terminology note (Monte Carlo name kept for compat; internal package uses `batch`).

Architecture guard tests (`tests/test_architecture_boundaries.py`, AST/import-based):
- `physics/*` must not import `core`, `ui`, `cli`, `analysis`, surrogate-training.
- `common/*` must not import torch/PySide6/matplotlib/h5py/scipy.
- `core/*` must not import `ui` or plotting/reporting.
- no new circular imports (smoke: import every public module).

---

## 3. Per-phase checklist (repeat each phase)

```
[ ] grep current public imports of the target (tests + src + entrypoints) BEFORE moving
[ ] create new package/modules, move code verbatim (no logic edits)
[ ] write compat shim re-exporting EVERY previously-public name
[ ] add import-compat test (old path + new path)
[ ] add responsibility-specific tests
[ ] run P0 baseline checks → diff against recorded baseline (must match)
[ ] ruff check . ; mypy (if wired)
[ ] update docs/refactor_notes.md
[ ] commit (single target, author = Ayberk, NO Claude co-author trailer)
```

---

## 4. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Entry points `lunaris-mc`/`lunaris-batch` break | Keep `mc_entry`/`batch_entry` re-exported from `core/monte_carlo_engine.py`; don't touch `[project.scripts]` |
| Module/package shadowing when `dynamics.py` → `dynamics/` | Delete old `.py` in same commit; verify `python -c "import lunaris.core.dynamics"` resolves to package |
| Circular imports (engine ↔ packs ↔ physics) | Keep prep one-directional: `core` → `physics`, never reverse; `batch` → `core`, never reverse |
| Numba perf regression in dynamics | Keep `rhs_numba.py` object-free; benchmark hot path against P0 baseline |
| Hidden private symbol imported by tests | grep-before-move; re-export privates too if referenced |
| Optional deps become required | Keep `_require_torch`/lazy imports; common/physics import guards |

## 5. Out of scope (explicit)

- `config.py` aggressive split (Target 6) — leave SSOT intact.
- `mc_propagator.py` / `events.py` standalone splits (only touched as P3 consumers).
- `batch/backends/` subpackage — deferred unless a clean seam appears in P1.
- Full plugin registry — not built; simple module-level prepare/evaluate functions only.
- Any numerical/algorithmic change.

## 6. Deliverables on completion

1. Refactor summary + files moved/created list (in `refactor_notes.md`).
2. Compatibility guarantees (shim inventory).
3. Tests run + results table.
4. Known limitations / follow-ups (deferred backends subpackage, config split, mc_propagator).
5. **No numerical-equivalence claim beyond what tests verify** (only the dynamics RHS golden + batch sampling determinism are asserted numerically).
