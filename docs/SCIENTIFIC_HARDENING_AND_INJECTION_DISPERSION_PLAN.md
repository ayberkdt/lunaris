# Scientific Hardening and Injection Dispersion Plan

Status legend: todo / in_progress / done / blocked

Source: review attachment `8adb3aa7-6485-4da5-9d15-280663435850/pasted-text.txt`
plus the terminology review note:

> Use `Enjeksiyon Tolerans Analizi (Injection Dispersion Analysis)` instead of
> the generic `Belirsizlik Analizi` label where the workflow is about injected
> initial-state / spacecraft-parameter dispersions.

## Objective

Harden Lunaris for scientific and reviewer scrutiny without broad rewrites or
cosmetic documentation churn. The work should make correctness, frame handling,
dtype behavior, backend fallback, dataset contracts, and benchmark evidence
explicit, tested, and difficult to misrepresent.

## Global Rules

- Do not fake benchmark results or add synthetic numbers as scientific evidence.
- Do not silently change physics behavior, force-model fidelity, backend,
  dtype, frame mode, or spherical-harmonic degree.
- Preserve public APIs unless there is a strong, tested reason to change them.
- Every behavior change needs a regression test.
- Prefer small auditable patches over sweeping refactors.
- Keep comments/docstrings only when they clarify physics convention, frame,
  dtype, backend fallback, dataset contract, or benchmark evidence status.
- Record commands actually run; do not claim validation that was not executed.

## Scope Inventory

Primary review targets:

- `src/lunaris/core/dynamics/engine.py`
- `src/lunaris/physics/spherical_harmonics.py`
- `src/lunaris/physics/torch_spherical_harmonics.py`
- `src/lunaris/core/propagation/propagator.py`
- `src/lunaris/core/propagation/time_grid.py`
- `src/lunaris/batch/backend_policy.py`
- `src/lunaris/batch/engine.py`
- `src/lunaris/core/torch_batch_propagator.py`
- `src/lunaris/core/torch_sh_propagator.py`
- `src/lunaris/core/batched_fixed_step.py`
- `src/lunaris/surrogate/runtime/gravity_provider.py`
- `src/lunaris/surrogate/runtime/scalers.py`
- `src/lunaris/surrogate/st_lrps/training/config.py`
- `src/lunaris/surrogate/st_lrps/training/engine.py`
- `src/lunaris/surrogate/st_lrps/training/losses.py`
- `src/lunaris/surrogate/st_lrps/shared/scaling.py`
- `src/lunaris/surrogate/st_lrps/data/datasets.py`
- `src/lunaris/surrogate/st_lrps/data/splits.py`
- `tests/`

Terminology targets:

- `src/lunaris/ui/pages/batch_propagation_page.py`
- `src/lunaris/common/batch_defs.py`
- `src/lunaris/cli/batch_runner.py`
- `src/lunaris/analysis/ensemble/uq_report.py`
- `docs/UQ_COVARIANCE.md`
- `docs/ARCHITECTURE.md`
- `docs/PUBLIC_API.md`
- `docs/HPC.md`
- `README.md`
- relevant UI / CLI / docs tests

## Phase 0 - Baseline and Branch Hygiene

Status: todo

1. Create a clean branch, for example
   `codex/scientific-hardening-injection-dispersion`.
2. Install editable dev dependencies using the repository's actual
   `pyproject.toml` configuration.
3. Run baseline gates before edits and save failures in the working notes:

```bash
python -m pip install -e ".[dev]"
pytest -q
ruff check .
mypy src/lunaris/common src/lunaris/core src/lunaris/physics src/lunaris/surrogate/st_lrps
```

Acceptance:

- Baseline failures are recorded before code changes.
- No test is weakened to hide an existing or new failure.

## Phase 1 - Terminology: Injection Dispersion Analysis

Status: done

Implementation note, 2026-07-06:

- User-facing batch/ensemble workflow language was changed from generic
  uncertainty propagation to injection-dispersion terminology in the UI, CLI
  help, architecture/HPC/README docs, and batch config prose.
- The dispersion-model dataclass names `StateUncertainty` and
  `SpacecraftUncertainty` are preserved. (After merging main's batch-subsystem
  rework, the top-level config is `BatchPropagationConfig`; the older
  random-sampling config name was retired by that rework, not by this branch.)
- The UQ covariance report terminology was left intact where it specifically
  describes the provenance-stamped covariance artifact.

Problem:

The generic label `Belirsizlik Analizi` / `Uncertainty Analysis` is too broad
for the batch workflow when it is specifically perturbing injection conditions
and spacecraft parameters. The technical label should be
`Enjeksiyon Tolerans Analizi (Injection Dispersion Analysis)`.

Plan:

1. Separate terminology by meaning:
   - Use `Injection Dispersion Analysis` for initial state and spacecraft
     parameter dispersion propagated through deterministic dynamics.
   - Keep `Uncertainty Quantification (UQ)` only for the covariance/report
     artifact where it is explicitly defined and provenance-stamped.
   - Keep `force-model uncertainty` in the perturbation-budget subsystem.
   - Keep `tolerance` for numerical solver tolerances and validation thresholds.
2. Update user-facing UI labels and help text first:
   - page title / workspace copy in `batch_propagation_page.py`,
   - cards currently labelled `Initial State Uncertainty` and
     `Spacecraft Property Uncertainty`,
   - CLI help in `batch_runner.py`.
3. Preserve public Python dataclass names initially:
   - `StateUncertainty`,
   - `SpacecraftUncertainty`,
   - the retired top-level config compatibility aliases.
   These are public API terms; rename only through a deprecation plan if a later
   session decides the API should change.
4. Update docs to state the relationship:
   - `Injection Dispersion Analysis` is the workflow name.
   - `UQ report` is the covariance evidence product generated from an ensemble
     archive.
   - It is not orbit determination, navigation covariance, or process-noise
     estimation.
5. Add or update tests that pin exact user-facing labels where tests already
   cover the UI/CLI.

Acceptance:

- The UI and CLI no longer present the workflow as a vague uncertainty
  analysis.
- `docs/UQ_COVARIANCE.md` still honestly defines the covariance report and does
  not overclaim navigation performance.
- No public API break is introduced just for terminology.

## Phase 2 - ST-LRPS Laplacian Default Ambiguity

Status: done

Implementation note, 2026-07-06:

- `laplacian_mode="off"` now hard-disables all Laplacian work even when stale
  Laplacian weights or compatibility flags are nonzero.
- Regression assertions were added for this hard-disable behavior.
- Local PyTorch-backed tests were not executable in the Codex environment
  (incomplete `torch` namespace); the affected test modules skip cleanly when
  `torch.nn` is unavailable. Follow-up session 2026-07-06: verified on a full
  torch 2.5.1+cu121 environment — the touched test modules pass
  (57 passed).

Problem:

`TrainConfig` appears to enable sparse Laplacian-related defaults while
`_laplacian_requested()` describes disabled/off/zero behavior. This may silently
add expensive second-derivative or Hutchinson work and may affect the loss.

Plan:

1. Audit:
   - `TrainConfig` defaults,
   - `apply_run_preset()`,
   - `apply_model_preset()`,
   - `STLRPSTrainer.__init__()`,
   - `STLRPSTrainer.run_epoch()`,
   - `STLRPSTrainer._compute_loss()`,
   - `SobolevLoss`,
   - `collocation_laplacian_loss`.
2. Choose and encode the safest scientific default:
   - normal training should not compute Laplacian work unless explicitly
     requested or selected by a clearly named preset.
3. Make behavior consistent across config, training loop, logging, provenance,
   and tests.

Required tests:

- default `TrainConfig` has exactly the intended Laplacian behavior.
- `laplacian_mode="off"` performs zero Laplacian work.
- `laplacian_mode="diagnostic"` does not affect optimizer loss.
- `laplacian_mode="train"` with positive weight affects optimizer loss or fails
  loudly if computation fails.
- `use_laplacian_regularization=False` does not trigger in-batch Laplacian loss.

Acceptance:

- No hidden expensive physics regularization in default training unless it is
  explicitly intended and tested.

## Phase 3 - ST-LRPS Runtime Dtype Policy

Status: done

Implementation note, 2026-07-06:

- `SurrogateGravityModel.to_device(device, dtype=...)` now converts model
  weights, cached scaler tensors, constants, and the delegated canonical runtime
  to the requested dtype.
- The canonical `SurrogateForceModel` runtime now tracks dtype and converts its
  model/scaler/input tensors consistently.
- GPU ST-LRPS propagation and orbit-benchmark GPU helpers now pass the requested
  torch dtype into the surrogate runtime instead of only casting output tensors.
- Batch diagnostics now report requested dtype, effective dtype, downgrade flag,
  model dtype, and scaler dtype.
- Local PyTorch-backed tests were not executable in the Codex environment
  (incomplete `torch` namespace); the affected test modules skip cleanly when
  `torch.nn` is unavailable. Follow-up session 2026-07-06: verified on a full
  torch 2.5.1+cu121 environment — the touched test modules pass
  (57 passed).

Problem:

Surrogate runtime paths may pin scaler tensors and constants to `torch.float32`.
Throughput defaults can remain float32, but paper-safe or strict float64 runs
must either run as requested or fail/record a downgrade honestly.

Plan:

1. Audit `SurrogateGravityModel` load, device transfer, scalar tensors, model
   weights, input tensors, baseline SH delegation, and force-runtime paths.
2. Implement an explicit dtype policy:
   - throughput defaults may use float32 only when provenance records it,
   - requested float64 must convert model/scalers/constants end to end, or fail
     loudly in strict/paper-safe paths,
   - diagnostics must report requested dtype, effective dtype, downgrade flag,
     model parameter dtype, and scaler dtype.
3. Ensure GPU batch paths call a dtype-aware `to_device(...)` or equivalent.

Required tests:

- mocked or tiny surrogate moves model, scalers, and constants to float64.
- strict/paper-safe float64 request cannot silently run float32.
- diagnostics report effective dtype honestly.
- CPU fallback after GPU failure updates dtype provenance.

Acceptance:

- No paper-safe benchmark can report float64 while executing surrogate inference
  in float32.

## Phase 4 - Spherical-Harmonic Convention Lock

Status: done

Implementation note, 2026-07-06:

- Audit found existing coverage already locks combined-field parity
  (`test_spherical_harmonics.py` FD-oracle tesseral test,
  `test_torch_sh_evaluator.py` torch-vs-numba, `test_st_lrps_sh_baseline.py`
  potential-path parity, `test_st_lrps_generator_phase_parity.py` odd-m phase,
  `test_torch_sh_batch_propagator.py` degree-never-clipped + provenance).
- Added `tests/test_sh_convention_lock.py` (26 tests) for the remaining gap:
  SINGLE-coefficient artificial fields (C20/C21/S21/C22/S22/C31/S33/C43)
  compared as full vectors across `GravityModel.accel_fixed`,
  `sh_potential_accel_fixed`, and `TorchSHGravityEvaluator` at equatorial /
  mid-latitude / near-pole / multi-longitude / multi-altitude points, plus a
  central-FD `a = +grad(U)` sign lock per term, a geodesy potential-sign test,
  and near-pole finiteness for the potential and torch paths.

Problem:

Multiple SH implementations must agree on normalization, phase, potential sign,
acceleration sign, degree truncation, residual baseline handling, and pole
behavior.

Plan:

1. Compare:
   - `GravityModel.accel_fixed()`,
   - `sh_accel_fixed_numba()`,
   - `sh_potential_accel_fixed()`,
   - `sh_potential_accel_batch_serial()`,
   - `TorchSHGravityEvaluator.acceleration()`.
2. Add low-degree artificial coefficient tests using `C20`, `C21`, `S21`,
   `C22`, `S22`, and odd-order tesseral/sectoral terms.
3. Test equatorial, mid-latitude, near-pole, multi-longitude, and multi-altitude
   positions.
4. Compare acceleration vectors, not only norms.
5. Add finite-difference potential-gradient checks for selected low-degree
   cases.

Required tests:

- CPU acceleration paths agree on artificial fields.
- Torch SH agrees with CPU SH for same coefficients and positions.
- odd-m no-Condon-Shortley phase regressions are caught.
- near-pole outputs remain finite.
- requested SH degree is never silently clipped in torch CUDA preflight.

Acceptance:

- Future SH phase/sign/convention regressions fail CI.

## Phase 5 - Benchmark Evidence Taxonomy

Status: done

Implementation note, 2026-07-06:

- Added `benchmark_evidence_taxonomy.py` codifying the five categories
  (`model_error_field`, `integrator_error`, `trajectory_error`,
  `phase_corrected_error`, `runtime_metrics`) with a metric-name classifier
  (field/phase/integrator markers matched before generic trajectory markers)
  and `summarize_evidence_taxonomy()` that buckets a run's metric columns,
  flags `has_field_level_evidence` / `trajectory_error_only`, and stamps
  synthetic output `scientific_evidence=False`.
- `validate_benchmark_outputs` now emits an `evidence_taxonomy` block in
  `validation_report.json` and warns when a run reports orbit-level trajectory
  error only (no `model_error_field`), so a trajectory artifact can never be
  read as ST-LRPS gravity-field accuracy.
- Paper-safe gate: synthetic/quick evidence under paper_safe is a hard error;
  trajectory-only is deliberately a labeled WARNING, not an error, because an
  honest orbit benchmark is legitimate *trajectory* evidence (the config-driven
  `lunaris-benchmark` reports trajectory error by design). The legacy
  `--batch-rk4` path already computes the total/model/integrator split; this
  phase makes the taxonomy an explicit, testable schema over any run's columns.
- Tests: `test_benchmark_evidence_taxonomy.py` (classification, bucketing,
  paper-safe gate) + `test_benchmark_validation.py` (report block, warning,
  paper-safe trajectory-only not failed).

Problem:

Orbit error mixes surrogate field error, integrator error, dtype error, frame
interpolation error, output interpolation error, and domain extrapolation. A
benchmark must not collapse these into one generic `surrogate error`.

Plan:

1. Separate benchmark output schema into:
   - `model_error_field`,
   - `integrator_error`,
   - `trajectory_error`,
   - `phase_corrected_error`,
   - `runtime_metrics`.
2. Report field-level gravity error at fixed Moon-fixed points:
   RMS/max/relative acceleration error and radial/cross-track decomposition
   where available.
3. Report orbit-level trajectory error separately:
   final/RMS/P95/max position error, velocity error, and phase drift metrics.
4. Report runtime separately:
   cold time, warm time, propagation time, acceleration evaluations/s,
   propagated seconds per wall second, and batch state steps/s.
5. Add quick synthetic smoke tests for schema only; stamp synthetic output as
   non-scientific.

Acceptance:

- A benchmark artifact cannot imply ST-LRPS field accuracy from trajectory
  error alone.
- Paper-safe mode requires real, non-synthetic, validated evidence fields.

## Phase 6 - Frame Safety and Identity Rotation

Status: done

Implementation note, 2026-07-06:

- Audit found the enforcement path already largely fail-closed:
  `DynamicsEngine._validate_dependencies` raises when body-fixed gravity needs
  `q_i2f` but no ephemeris is present and `allow_identity_rotation=False`
  (default); `_validate_legacy_batch_frame_inputs` rejects a non-legacy
  `--batch-rk4` frame mode without an ephemeris; the config-driven
  `lunaris-benchmark` always emits `match_dynamics_engine` (rotating) via
  `config_to_legacy_argv` and can never select identity; TorchBatchPropagator
  diagnostics already record `frame_mode`/`uses_frame_rotation`.
- Added provenance: `build_benchmark_manifest` now records
  `numerics.frame_mode` + `numerics.uses_frame_rotation`.
- Added a belt-and-suspenders fail-closed check
  `benchmark_validation._check_paper_safe_frame_mode`: a `paper_safe` run whose
  manifest records an identity/inertial frame mode is a hard validation error.
- Tests added: `test_dynamics.py` fail-closed + explicit-opt-in for body-fixed
  gravity without ephemeris; `test_benchmark_validation.py` paper-safe identity
  frame error / rotating-frame pass / non-paper-safe identity allowed;
  `test_benchmark_provenance.py` manifest frame-mode default + override.
  Existing coverage (`test_st_lrps_legacy_batch_frame.py`,
  `test_dynamics.py::test_cpu_sh_rhs_uses_i2f_then_conjugate_frame_bridge`,
  `test_torch_sh_batch_propagator.py` slerp/quaternion tests) retained.

Problem:

SH gravity and ST-LRPS are Moon-fixed/body-fixed. Identity rotation may be
acceptable for explicit smoke tests, but not for scientific benchmark claims.

Plan:

1. Audit:
   - `DynamicsEngine` `allow_identity_rotation`,
   - `TorchMoonFrame`,
   - CPU propagation path,
   - `torch_cuda_sh`,
   - `gpu_st_lrps_potential`,
   - `gpu_st_lrps_third_body`,
   - benchmark runner strict/paper-safe modes.
2. Enforce:
   - identity rotation requires explicit opt-in for smoke runs,
   - paper-safe/strict frame modes fail without real frame/ephemeris when
     body-fixed gravity is used,
   - provenance records whether identity rotation was used.

Required tests:

- smoke run can use identity only when explicitly allowed.
- paper-safe benchmark with missing ephemeris fails.
- GPU SH/ST-LRPS diagnostics record frame mode.
- CPU and GPU frame paths agree for a simple fixed quaternion table.

Acceptance:

- No paper-safe benchmark can accidentally evaluate Moon-fixed gravity in
  inertial coordinates with identity rotation.

## Phase 7 - Dataset Contract Enforcement

Status: done

Implementation note, 2026-07-06:

- Audit found `validate_training_dataset_convention` already fail-closes on
  wrong/missing derivative convention, non-lunar body, missing/invalid
  target_mode, missing/inverted degree bounds, missing altitude bounds, and
  unsupported unit_system.
- Gap fixed: a `unit_system='canonical'` dataset missing the DU/TU/VU scaling
  constants previously passed preflight and only failed deep inside data
  loading (`convert_xyz_U_a_to_si`). Added a preflight guard that rejects it via
  `DatasetMeta.can_convert_to_si()` before training begins.
- Tests added to `test_dataset_contract.py`: canonical-without-constants fails,
  canonical-with-constants passes, non-lunar body fails, missing derivative
  convention (absent attr) fails, residual inverted degree bounds fails.
  Existing coverage (missing degree, missing target_mode, derivative mismatch,
  altitude bounds, unit_system) retained.

Problem:

Dataset metadata warnings are not enough for training or paper-safe mode. Unsafe
or stale datasets should fail closed.

Plan:

1. Audit:
   - `DatasetMeta.from_h5()`,
   - `validate_training_dataset_convention()`,
   - `_resolve_lunar_dataset_contract()`,
   - `validate_dataset_file()`,
   - training preflight.
2. Training should refuse datasets when derivative convention is missing/wrong,
   body/frame/units are ambiguous, degree ranges are invalid, residual baseline
   metadata is missing, or residual `degree_max <= degree_min`.

Required tests:

- valid tiny HDF5 dataset passes.
- missing derivative convention fails preflight.
- wrong derivative convention fails.
- non-lunar body fails.
- residual with invalid degree bounds fails.
- canonical units without required DU/TU/VU metadata fail when conversion is
  ambiguous.

Acceptance:

- Unsafe legacy datasets cannot enter training silently.

## Phase 8 - Backend Fallback and Provenance

Status: done

Implementation note, 2026-07-06:

- Audit found Phase 8's requirements almost entirely covered already:
  `test_classic_sh_policy.py` (numba high-degree `fallback_policy="error"`
  requires_error, `compatible_gpu`→torch, `cpu`→cpu, degree never clipped),
  `test_mc_gpu_policy.py` (st_lrps torch true/false, unsupported physics +
  third-body fallback provenance, high-degree fallback without clipping),
  `test_torch_sh_batch_propagator.py` (degree above loaded raises not clips,
  provenance records requested==actual degree),
  `test_backend_fallback_and_scaler_consolidation.py` (`_fallback_forbidden`
  on policy=error and paper_safe/strict_backend/benchmark_mode flags,
  init-failure raises when forbidden).
- Gap fixed: strengthened the GPU→CPU downgrade test to assert the FULL
  provenance rewrite (`actual_backend`, `actual_sh_degree=None`,
  `actual_device=cpu`, `cuda_device_name=None`, `dtype=float64`, DOP853
  integrator, `fallback_reason`) for both ST-LRPS and classic-SH plans, so a
  CPU run can never be labelled GPU.

Problem:

Backend routing appears strong, but strict and paper-safe fallback behavior must
be regression-tested.

Plan:

1. Add backend-policy tests for:
   - `numba_cuda_sh` requested above supported degree with
     `fallback_policy="error"`,
   - compatible GPU route to `torch_cuda_sh`,
   - CUDA unavailable with fallback disallowed,
   - unsupported force models on gravity-only GPU paths,
   - ST-LRPS third-body support limits.
2. Add batch-engine tests for:
   - strict/paper-safe forbids GPU-to-CPU fallback,
   - fallback updates actual backend/device/dtype and fallback reason,
   - no run is labelled GPU when CPU was used.
3. Add torch CUDA SH tests:
   - requested degree above loaded degree fails,
   - degree is never clipped,
   - `actual_sh_degree == requested_sh_degree`.

Acceptance:

- Backend routing is auditable and fail-closed in scientific modes.

## Phase 9 - Propagator Risk Reduction

Status: done

Implementation note, 2026-07-06:

- Followed the plan's explicit guidance to prefer behaviour tests over a risky
  rewrite. Audit found most required behaviour already pinned in
  `test_propagator_physics.py`: terminal adaptive-chunk checkpoint uses the
  event endpoint (`checkpoint_mode="latest"` records chunk END not start),
  failed adaptive chunk is not checkpointed and reports `stopped_early=True`,
  chunked impact-event stop reason, stop-file halt.
- Gap filled with two behaviour-lock tests (no source change, so no numerical
  behaviour drift): adaptive AND fixed-step output time arrays are strictly
  monotonic and start at 0; chunked vs unchunked adaptive solve_ivp agree for a
  two-body orbit (same output grid to 1e-9, same trajectory to integrator tol).
- `propagate()` left intact: the existing `_resolve_atol`, `make_time_grid`,
  and `_clamp_output_dt` helpers are already extracted and unit-tested; no
  further extraction was worth the numerical risk.

Problem:

`propagate()` is broad and brittle. Avoid a risky rewrite; extract pure helpers
only where tests make behavior easier to pin.

Plan:

1. Identify safe helper candidates:
   - output time-grid resolution,
   - absolute tolerance vector resolution,
   - solve_ivp chunk concatenation/finalization,
   - event endpoint handling,
   - diagnostics assembly.
2. Add tests before or during extraction.
3. Preserve public API and result structure.

Required tests:

- fixed-step and solve_ivp paths return monotonic time arrays.
- chunked and unchunked solve_ivp agree for a simple two-body problem.
- terminal event endpoint is included.
- checkpoint latest/state mode records chunk end, not chunk start.
- integration failure produces `stopped_early=True` and honest diagnostics.

Acceptance:

- `propagate()` is either less brittle or better tested, with no casual
  numerical behavior changes.

## Phase 10 - Fast Scientific Smoke Suite

Status: done

Implementation note, 2026-07-06:

- The smoke coverage is satisfied by the focused CPU-only modules added across
  the phases, all fast and CUDA-free:
  - SH artificial-field convention + torch-vs-CPU: `test_sh_convention_lock.py`.
  - Dataset contract fail-closed: `test_dataset_contract.py`.
  - Backend policy strict fallback: `test_backend_fallback_and_scaler_consolidation.py`,
    `test_classic_sh_policy.py`.
  - ST-LRPS dtype propagation (tiny model): `test_backend_fallback_and_scaler_consolidation.py`.
  - Propagator two-body energy/monotonic/chunk sanity: `test_propagator_physics.py`.
  - Field-level vs orbit-level benchmark schema: `test_benchmark_evidence_taxonomy.py`.
  - Injection Dispersion terminology (UI/CLI/docs): `test_injection_dispersion_terminology.py`.
- No new pytest marker system was introduced (the repo's GPU tests already
  `importorskip`/skip cleanly when CUDA is unavailable), avoiding churn to
  `pyproject.toml`.

Plan:

Create a small CI-friendly test group using artificial in-memory gravity models
and tiny synthetic datasets:

- SH artificial field convention test.
- Torch SH vs CPU SH small-degree test.
- Dataset contract fail-closed test.
- Backend policy strict fallback test.
- ST-LRPS dtype propagation test using a tiny mocked model.
- Propagator two-body energy drift sanity test.
- Field-level vs orbit-level benchmark schema test.
- Injection Dispersion Analysis terminology smoke test for UI/CLI/docs labels.

Use markers if needed:

- `pytest.mark.fast`
- `pytest.mark.gpu`
- `pytest.mark.slow`

Acceptance:

- CPU smoke tests always run.
- GPU smoke tests skip cleanly when CUDA is unavailable.

## Phase 11 - Final Validation and Report

Status: done

### Commands run and results (2026-07-06 follow-up session)

- `ruff check .` — **All checks passed** (whole repo).
- Focused pytest suites, all green:
  `test_sh_convention_lock.py` (26), `test_dataset_contract.py` (16),
  `test_benchmark_validation.py` + `test_benchmark_provenance.py` (29),
  `test_benchmark_evidence_taxonomy.py` (9), `test_dynamics.py` +
  `test_st_lrps_legacy_batch_frame.py` (16 + 1 skipped),
  `test_propagator_physics.py` (17), `test_injection_dispersion_terminology.py`
  (8), `test_backend_fallback_and_scaler_consolidation.py` (12),
  `test_classic_sh_policy.py`, `test_mc_gpu_policy.py`,
  `test_surrogate_architecture_upgrades.py`, `test_surrogate_upgrades.py`,
  `test_batch_propagation_page.py`. Combined non-torch batch: 145 passed / 1 skipped;
  torch/UI batch: 59 passed.
- `mypy` — the touched files introduced **zero new errors**. Pre-existing
  baseline errors remain in `runtime/force_model.py` + `training/engine.py`
  (29, identical count before/after) and `evaluation/benchmark_validation.py`
  (6, all in pre-existing `_check_scenario_ids` / `_check_model_name_consistency`).
  New modules `benchmark_evidence_taxonomy.py`, `provenance.py`, and the
  `datasets.py` edit are mypy-clean.

### Behavior changes that could affect old experiments

- Phase 3: ST-LRPS GPU batch now honors requested `torch_dtype` end-to-end
  (model + scalers + constants), not just the output cast. A `float64` request
  that previously ran surrogate inference in float32 now runs float64 — results
  may shift at the ~1e-7 level and throughput drops; recorded in diagnostics.
- Phase 2: `laplacian_mode="off"` is now an unconditional hard-disable (was
  promoted to "diagnostic" when `use_laplacian_regularization=True`). A config
  relying on that promotion loses Laplacian diagnostics until it sets
  `laplacian_mode="diagnostic"` explicitly.
- Phase 7: canonical datasets missing DU/TU/VU now fail at preflight rather than
  deep in data loading; such datasets could not train correctly anyway.

### Paper-safe defensibility

Paper-safe benchmark generation is materially more defensible: dtype is honestly
reported (no float64 label over float32 inference), identity/inertial frame under
paper_safe fails validation, canonical/unsafe datasets fail closed, a GPU→CPU
downgrade rewrites all provenance so a CPU run is never labeled GPU, and every
benchmark artifact carries an `evidence_taxonomy` block so trajectory error
cannot masquerade as ST-LRPS gravity-field accuracy.

### Remaining risks

- CUDA paths were exercised only on CPU in this environment; a real-GPU
  confirmation of the dtype/frame diagnostics is still advisable (roadmap G6).
- Pre-existing mypy debt in `force_model.py` / `engine.py` is untouched baseline,
  not introduced here.

## Suggested Work Order Across Sessions

1. Phase 0 and Phase 1 together: cheap, clarifies terminology and baseline.
2. Phase 2 and Phase 3: ST-LRPS training/runtime correctness.
3. Phase 4: SH convention lock.
4. Phase 6, Phase 7, Phase 8: fail-closed scientific mode behavior.
5. Phase 5: benchmark evidence schema after dtype/frame/backend semantics are
   pinned.
6. Phase 9: propagator cleanup only after behavior tests exist.
7. Phase 10 and Phase 11: final smoke suite and full validation report.
