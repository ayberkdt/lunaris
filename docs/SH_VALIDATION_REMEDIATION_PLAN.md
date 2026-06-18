# Spherical-Harmonic Validation Remediation Plan

Status tracker for the gravity / SH-validation findings raised in independent
review and the Condon–Shortley (CS) phase bug they uncovered. Each item lists its
verification verdict, the fix, and where it lives. Statuses: **DONE** (landed +
tested), **PARTIAL** (started; remainder noted), **BLOCKED** (needs a dependency
or data not present in this environment — scaffolded, never faked).

> Convention decision (anchors everything below): lunar geodesy/GRAIL gravity
> coefficients are **4π fully-normalized with NO Condon–Shortley phase**. The
> runtime engine, the ST-LRPS label generator, and all independent harnesses must
> agree on this. Confirmed against pyshtools (`MakeGravGridPoint`, no-phase) and
> the GRAIL JGGRX_1800F contract.

## Root cause

A commit had added the `(-1)^m` Condon–Shortley phase to the engine's `scale_m`
table, flipping every odd-order (tesseral/sectoral) term. It was invisible to the
J2/zonal tests (m=0 is unaffected) and produced ~1e-3 m/s² acceleration error on
the real GRAIL degree-120 field. The ST-LRPS data generator carried a matching
phase, so training labels and the corrected runtime represented *different*
fields.

## Findings (review) — all verified TRUE

| # | Finding | Verdict |
|---|---|---|
| 1 | Generator still on the CS-phase convention while the engine was fixed | TRUE |
| 2 | Old accuracy benchmarks predate the fix; not physical evidence | TRUE (relative/internal comparisons may survive) |
| 3 | Dataset/artifact contract did not capture the SH phase convention | TRUE |
| 4 | Propagator: chunk-checkpoint order, single-chunk terminal-event bookkeeping, forced telemetry | TRUE (3/3) |
| 5 | "Independent integrator" wording wrong (both sides DOP853) | TRUE |
| 6 | `MOON_PA` frame name misleading for a frozen-field test | TRUE |
| 7 | Manifest/CSV grid + frame contracts not enforced; "cubic" comment but linear interp | TRUE |
| 8 | `reference_class="published_field_vectors"` wrong; only 8 points | TRUE |

## P0 — scientific correctness (must precede any regeneration)

- **P0.1 Generator phase fix** — **DONE.** `spatial_cloud_generator`: sectoral
  seed/recurrence flipped to positive (no phase). Matches `GravityModel.accel_fixed`
  to ~4e-16 on odd-`m` fields.
- **P0.2 Generator↔engine parity test** — **DONE.**
  `tests/test_st_lrps_generator_phase_parity.py` (odd-`m` field + negative control
  proving it detects a CS field; zonal-only checks forbidden).
- **P0.3 SH-convention in the contract, reject stale artifacts** — **DONE.**
  `spherical_harmonic_convention="4pi_geodesy_no_condon_shortley_v1"` (+
  `gravity_label_engine_version="lunaris_sh_v2"`) stamped into dataset attrs,
  `DatasetContract` (validated; default `None` → pre-fix datasets fail closed
  unless explicit legacy override), and the artifact dataset-meta block.
- **P0.4 Invalidate stale benchmarks; regenerate + retrain** — **PARTIAL.**
  Done: `README.md` + `docs/BENCHMARK_RESULTS.md` carry an "invalidated / pending
  regeneration" warning; accuracy numbers tagged `(invalidated)`.
  **Remaining (heavy, state-producing):** regenerate ST-LRPS datasets with the
  fixed generator, retrain the model, and re-run the accuracy benchmarks. Only
  relative runtime/throughput figures stay indicative; old checkpoints are
  incompatible with the corrected runtime.

## P1 — correctness / honesty hardening — DONE

- **P1.1 Propagator** — chunk checkpoints now record the *end* of the completed
  chunk (state advanced before the write); single-chunk path sets `stopped_early`
  from `sol.status==1` so `stop_reason` is never set while `stopped_early` is
  False.
- **P1.2 Telemetry opt-in** — JSON telemetry gated on `cfg.enable_telemetry` (or a
  positive `telem_cadence_s`); library/validation/batch callers no longer pollute
  stdout.
- **P1.3 Validation contract + wording** — trajectory runner is **fail-closed**:
  reference epoch grid must be uniform and match the manifest
  `output_step_s`/`duration_s`, and `comparison_frame` must equal `state_frame`,
  else `INCOMPLETE_CONTRACT` (no silent interpolation). Frames renamed
  `NONROTATING_FROZEN_BODY_FIXED`; "independent integrator" corrected to
  "independent force model with a separately-configured DOP853 integration".

## P2 — coverage — DONE

- **P2.1 Statistical field validation** — `grail_degree120_pyshtools_sobol`: ~2000
  deterministic Sobol points over the sphere × 50–2000 km altitude strata + polar/
  equatorial sets. Metrics add max/P95/P99 and **latitude/altitude-binned** error
  tables (`field_metrics.latitude_altitude_error_tables`). Lunaris matches
  pyshtools to ~machine precision in every band (worst ~8e-11 m/s² at low-altitude
  poles — the expected `1/cos φ` regime). Both GRAIL field benchmarks relabeled
  `independent_high_precision_field_oracle`.

## Extra gap closed (beyond the review)

- **Torch/GPU SH path parity** — **DONE.** The torch evaluator (GPU + benchmark)
  had the same zonal/finite-only blind spot. `test_torch_evaluator_matches_numba_engine_on_tesseral_field`
  (degrees 8/60/120/200) pins it to the validated numba engine on odd-`m` fields
  (~3e-16). Also added kernel-free body-frame invariants
  (`tests/test_sh_frame_invariants.py`: zonal z-rotation equivariance, zero
  longitudinal accel, pure-radial monopole).

## Independent cross-validation tiers

| Reference | Independence | Status |
|---|---|---|
| in-repo numpy oracle | different algorithm, same language | VERIFIED (~1e-12, low degree) |
| pyshtools | separate SH library (C/Fortran) | VERIFIED (~1.9e-13 at degree 120; statistical to ~2000 pts) |
| Energy/momentum invariants | model-free physics | VERIFIED (~1e-15) |
| tudatpy | separate toolkit (C++) | **BLOCKED** — not installed; gated `requires_tudatpy`, scaffold in `validation/independent/tudatpy_reference.py` |
| SPICE rotating-frame (physical `MOON_PA`) | real lunar orientation | **BLOCKED** — needs NAIF `.tls` (leapseconds) + `.tf` (lunar frame) kernels not present; rotating-frame manifest already fails closed |

## Remaining work (priority order)

1. **Regenerate datasets + retrain + re-run accuracy benchmarks** (P0.4 tail) —
   prerequisite for any accuracy claim; do only after P0.1/P0.3 (both DONE).
2. **Install tudatpy** (`conda install -c tudat-team tudatpy`) and confirm the
   gated cross-check's point-gradient API on first run.
3. **Add NAIF kernels** (`naif00xx.tls`, `moon_de440_*.tf`; or `lunaris-data
   download`) to enable the physical rotating-frame gravity-only reference.
4. *(Optional)* extend external pyshtools field validation to degree 200 (the max
   degree used by the benchmarks; internal torch↔numba consistency already covers
   200).
