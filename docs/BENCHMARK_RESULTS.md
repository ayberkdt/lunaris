# ST-LRPS Orbit-Level Benchmark Reporting

This file is the active reporting contract for ST-LRPS orbit-level gravity
benchmarks. It intentionally contains no accepted accuracy table until the
dataset, model artifact, and benchmark run have been regenerated under the
current validation contract.

## Current Status

No regenerated, paper-safe ST-LRPS orbit-accuracy result is published here yet.

A benchmark result is accepted for documentation only when:

- `validation_report.json` passes.
- `benchmark_manifest.json` records the resolved config hash, git commit, dirty
  state, scenario seed/count, numerical reference, integrator, dtype, output
  cadence, runtime backend, hardware, model/checkpoint hashes when available,
  dataset/gravity-file hashes when available, and ST-LRPS artifact-contract
  compatibility.
- The run uses a real ST-LRPS artifact whose dataset contract matches the
  benchmark config.
- The output is not a quick, synthetic, legacy, or validation-failed run.

Until those conditions are met, do not cite ST-LRPS orbit accuracy, speedup, or
speed-accuracy trade-off numbers as paper-safe evidence. The
[research-preview results](#research-preview-results) below may be cited only
with that label, their configuration and their limits.

## Research-preview results

The README section "Results so far" quotes these runs. None of them meets the
acceptance conditions above: there is no passing `validation_report.json`, the
split is interpolation (no spatial-block split), and the runs come from one
trained model on one laptop. The artifacts are local run folders under the
git-ignored `outputs/`; the table names each one so a reader can ask for it or
regenerate it.

Model: run `st_lrps_train_20260920_141322_planD_dirfix`, `potential_autograd`,
512×5 residual SIREN (1,848,321 parameters), `target_mode=residual`, SH25 →
SH200 of GL1800F (`jggrx_1800f_sha.tab`), convention
`4pi_geodesy_no_condon_shortley_v1`, best checkpoint epoch 792 of 800. Dataset:
suite `moon_deg25to200_alt100to1000km_20260704_103234` (train content SHA-256
`9801507d…cba7d2c1`), 100–1000 km, Moon-fixed frame. Hardware: one laptop with
an NVIDIA GTX 1660 Ti.

| Result | Configuration | Artifact (under `outputs/`) |
|---|---|---|
| Field accuracy: test, 60–100 km, 1000–1040 km | 2 M / 0.5 M / 0.5 M held-out points, residual acceleration against SH200 − SH25 | `training/<run>/evals/publication_eval/` (`evaluate_metrics.json`, `ood_low/`, `ood_high/`) |
| Five-day orbit error | 256 orbits (first 256 of a 1024-orbit scrambled Sobol set, seed 42, bounded Keplerian, 100–1000 km), RK4 Δt = 15 s, float64, gravity only, reference SH200 RK4 on the CPU | `gravity_benchmark/comparison_rk4_dt15/summary.json` (pair A: ST-LRPS vs SH200; pair B: SH25 vs SH200) |
| Per-orbit runtime | Same runs: GPU SH25 and GPU ST-LRPS over 1024 orbits, CPU SH200 over 256 | `gravity_benchmark/planD_1024_5day_rk4dt15_sh25truth/metrics/gpu_batch_runtime_metrics.csv`, `gravity_benchmark/planD_1024_5day_rk4dt15_cpu_sh200_first256/metrics/gpu_batch_runtime_metrics.csv` (commit `27e9ab86`) |
| 60-day frozen-family check | 8 candidates, perilune 100–200 km, e = 0.03–0.06, i = 27–90°, ω₀ = 90°, RK4 Δt = 15 s, reference SH200 | `gravity_benchmark/comparison_rk4_dt60/frozen_60day_validation.csv` |
| Frozen-search campaign wall time | 4096 orbits, 60 days, RK4 Δt = 30 s; time from stage 0 to the end of stage 1 (screening) in `manifest.json` | `frozen_search/v2_S_stlrps_n4096`, `frozen_search/v2_C_sh25_n4096`, `frozen_search/v3_S_stlrps_fullforce_n4096` (commit `27e9ab86`, uncommitted frozen-search options) |

The campaign estimate for SH200 (about 74 days on the CPU) is not a run. It
multiplies the measured CPU SH200 cost per five-day, Δt = 15 s orbit by the
campaign's orbit count, arc length (×12) and step (÷2), then scales it by the ratio of the measured SH25
campaign to the same estimate made for SH25. It compares a CPU estimate with a
GPU run; an SH200 timing on the same GPU has not been measured.

## Benchmark Purpose

An orbit-level gravity benchmark answers a specific question:

> For this scenario distribution, duration, integrator, dtype, output cadence,
> reference model, hardware, and ST-LRPS artifact, how close is each propagated
> trajectory to the declared numerical reference, and how long did it take?

The benchmark does not prove universal lunar-orbit performance. It measures the
stated configuration.

## Metric Definitions

Use these definitions whenever publishing a table or figure.

| Metric | Unit | Meaning |
|---|---:|---|
| Median RMS position error | km | Median over scenarios of time-RMS position error against the declared numerical reference. |
| P95 RMS position error | km | 95th percentile over scenario RMS errors; a tail-risk indicator. |
| Max RMS position error | km | Worst scenario RMS error in the reported scenario set. |
| Radial RMS error | km | RIC radial component; closest to altitude error. |
| Along-track RMS error | km | RIC along-track component; mainly phase/timing drift. |
| Cross-track RMS error | km | RIC cross-track component; mainly orbital-plane error. |
| Phase lag | s | Estimated along-track time shift; positive means the model leads the reference. |
| Phase-corrected RMS | km | Residual RMS after fitting and removing secular phase lag; diagnostic, not a replacement accuracy number. |
| Energy drift | relative | Relative change in the model trajectory's diagnostic specific energy. |
| Acceleration max error | m/s² | Maximum acceleration-vector error when acceleration samples are available. |
| Potential error | m²/s² | Potential error when potential samples are available. |
| Runtime | s | Wall-clock time for the benchmark stage being reported. |
| Cold time | s | Load/build/warm-up plus propagation timing. Reported separately from warm timing. |
| Warm time | s | Timing used for paper tables after load/JIT warm-up. |
| JIT compile time | s | Cold minus warm timing attributed to JIT/build warm-up when measured. |
| Steps/s | steps/s | Propagation throughput for the reported backend and hardware. |
| Acceleration eval/s | eval/s | RHS/acceleration evaluations per wall second, scaled by the active integrator stage count. |
| Propagated seconds per wall second | s/s | Scenario-duration throughput; useful across different output cadences. |
| Speedup | dimensionless | Runtime ratio against the named baseline/reference on the same scenario set. |

Always state the comparison frame for vector errors. A high-degree SH DOP853 run
is a numerical reference, not physical truth.

When the reference degree exceeds the ST-LRPS training target degree, state the
representation ceiling next to the table: the band between the training target
and the reference (e.g. SH61–100 for an SH60-trained surrogate against SH100)
is structurally unrepresentable for the surrogate and is its irreducible error
floor — the surrogate is not expected to approach the reference-degree
baseline. Measured band shares of the non-spherical acceleration RMS are in
`docs/ST_LRPS_VALIDATION_HYGIENE.md` ("What ST-LRPS is (and is not)").

## Required Context

Every published benchmark table must state:

- ST-LRPS model kind: `potential_autograd` (the only supported kind).
- Target contract: `target_mode`, `baseline_kind`, baseline degree, target
  degree, and spherical-harmonic convention.
- Scenario domain: count, seed, altitude/eccentricity/inclination envelope, and
  whether the split is interpolation, spatial generalization, or altitude OOD.
- Propagation settings: duration, output cadence, integrator, step size, dtype,
  enabled force models, and backend.
- Reference model: SH degree/order, integrator, tolerances, and frame.
- Hardware: CPU, GPU, CUDA/torch versions when runtime or GPU throughput is
  discussed.
- Provenance: config path/hash, manifest path, validation-report path, commit,
  and dirty-state summary.

## Table Template

When a new run is accepted, add a table in this shape and link the manifest:

| Benchmark | Model | Scenario count | Duration | Reference | Median RMS [km] | P95 RMS [km] | Runtime [s] | Backend | Manifest |
|---|---|---:|---:|---|---:|---:|---:|---|---|
| `<config name>` | `<model name>` | `<n>` | `<days>` | `<reference>` | `<value>` | `<value>` | `<value>` | `<backend>` | `<path>` |

Keep interpretation separate from the measured result. For example:

- Result: "Median RMS position error was `<value>` km for `<n>` scenarios."
- Interpretation: "For this configuration, tail error was dominated by
  along-track drift."
- Limitation: "This does not establish performance outside the stated altitude
  envelope or with other force-model settings."

## Regeneration Command Pattern

Paper-safe benchmarks should be produced from fixed config files:

```bash
lunaris-benchmark \
    --config configs/benchmarks/st_lrps_1day_high_degree.json \
    --model-dir outputs/training/st_lrps_train_YYYYMMDD_HHMMSS \
    --out outputs/gravity_benchmark/st_lrps_1day_high_degree
```

For CPU-only CI or local wiring checks, quick mode is allowed only as a smoke
test:

```bash
lunaris-benchmark --config configs/benchmarks/st_lrps_1day_high_degree.json --quick
```

Quick-mode output is never scientific evidence and must not be copied into
accuracy tables.

## Historical Note

Earlier active tables were removed from this document because they were produced
from pre-alignment datasets/artifacts and are not accepted evidence under the
current benchmark contract. The current engine, label generator, and contracts
use `4pi_geodesy_no_condon_shortley_v1`. The remediation audit trail is kept as
a local planning scratchpad under `PLANS/` when needed; public benchmark docs
should not depend on ignored plan files.
