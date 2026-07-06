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
speed-accuracy trade-off numbers from this document.

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
use `4pi_geodesy_no_condon_shortley_v1`; see
[SH_VALIDATION_REMEDIATION_PLAN.md](SH_VALIDATION_REMEDIATION_PLAN.md) for the
audit trail.
