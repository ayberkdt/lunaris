# Reproducible Benchmarks

Lunaris benchmark claims should be regenerated from a fixed config file and
accepted only after provenance capture and validation checks complete. The
preferred entry point is:

```bash
lunaris-benchmark --config configs/benchmarks/st_lrps_1day_high_degree.json
```

JSON config files are used in this repository to avoid a required YAML
dependency. The loader also accepts YAML when PyYAML is available.

## Benchmark Configs

Benchmark configs live under `configs/benchmarks/`. Each config records:

- scenario seed, count, sampling family, and altitude envelope
- propagation duration, output cadence, integrator, step size, and dtype
- truth gravity model, degree, integrator, and tolerances
- compared spherical-harmonic baselines
- optional ST-LRPS model directory and baseline degree
- output behavior

Do not hardcode private local checkpoint paths into committed configs. Use
`null` in the config and pass the path at runtime:

```bash
lunaris-benchmark \
  --config configs/benchmarks/st_lrps_1day_high_degree.json \
  --model-dir outputs/training/st_lrps_train_YYYYMMDD_HHMMSS \
  --out outputs/gravity_benchmark/st_lrps_1day_high_degree
```

## Artifact Contract Compatibility

When `surrogate.model_dir` is set, the benchmark builds a requested
`ArtifactContract` from the config and compares it to the selected ST-LRPS run.
The default is strict:

- ST-LRPS baseline degree/kind must match the config
- truth degree must match the artifact target degree
- runtime kind must be `potential_autograd`
- `mu_si`, `r_ref_m`, and `a_sign` must agree
- the scenario altitude envelope must stay inside the artifact training
  envelope

The compatibility report is written to both `resolved_config.json` and
`benchmark_manifest.json` as `contract_compatibility`, then folded into
`validation_report.json`.

Exploratory overrides are explicit:

```bash
lunaris-benchmark --config configs/benchmarks/st_lrps_1day_high_degree.json --allow-contract-mismatch
lunaris-benchmark --config configs/benchmarks/st_lrps_1day_high_degree.json --allow-domain-extrapolation
lunaris-benchmark --config configs/benchmarks/st_lrps_1day_high_degree.json --allow-legacy-artifact
```

Use these only when the resulting report is clearly labeled exploratory. A
strictly reproducible benchmark claim should pass without these flags.

## Output Layout

Unless `--out` or `outputs.out_dir` is set, runs write to:

```text
outputs/gravity_benchmark/<benchmark_name>_<timestamp>/
```

The standardized files are:

```text
benchmark_manifest.json
resolved_config.json
metrics_summary.csv
metrics_summary.json
scenario_results.csv
runtime_summary.csv
validation_report.json
report.md
figures/
```

Generated benchmark outputs, checkpoints, datasets, and reports should not be
committed.

## Provenance

`benchmark_manifest.json` links the result to:

- Git commit, branch, dirty working tree state, and git failure reasons when git
  is unavailable
- original config path/hash and resolved config hash
- scenario seed/count/type and altitude envelope
- truth model degree and local gravity-file hash when configured
- baseline model names/degrees
- ST-LRPS model directory, checkpoint hash, config hash, runtime model kind, and
  baseline degree when available
- ST-LRPS artifact-contract compatibility findings when a model directory is
  configured
- dataset hash when a dataset path is configured
- integrator settings, step sizes, duration, output cadence, and dtype
- Python, platform, NumPy, SciPy, PyTorch, CUDA availability, and device name

If an optional file does not exist locally, the manifest records `null` plus a
clear `missing_reason` instead of failing the run.

## Dataset Inputs

When a benchmark or training run depends on an ST-LRPS HDF5 cloud, generate or
inspect the dataset with the dataset pipeline first:

```bash
lunaris-data inspect --data outputs/datasets/cloud.h5
lunaris-data validate --data outputs/datasets/cloud.h5 --out outputs/dataset_reports/cloud
lunaris-data report --data outputs/datasets/cloud.h5 --out outputs/dataset_reports/cloud
```

The dataset validation report and quality report should be kept with any
benchmark paper trail that depends on that cloud. Training run manifests also
record the embedded dataset contract, the validation report path, and the split
manifest path.

## Validation

`validation_report.json` is machine-readable:

```json
{
  "passed": true,
  "errors": [],
  "warnings": [],
  "checked_files": [],
  "checked_metrics": []
}
```

Validation fails on impossible or incomplete evidence, including:

- missing or empty required output files
- NaN or Inf in numeric metrics
- scenario IDs that are non-integer or not the exact contiguous range `0..N-1`
  (non-contiguous external IDs require `scenario_id_policy="external_noncontiguous"`
  and a `scenario_id_mapping.json`)
- per-model scenario row counts that differ from the expected count
- `runtime_summary.csv` missing `n_scenarios`, or `n_scenarios` ≠ expected count
- `total_runtime_s / n_scenarios` inconsistent with `runtime_per_scenario_s`
- model names that differ across `metrics_summary`, `scenario_results`, and
  `runtime_summary`
- a `report.md` scenario count that disagrees with the validated count
- `max < p95` or `p95 < median`
- non-positive runtime or step counts
- duplicate model names
- accidental truth-model duplication as a baseline unless explicitly allowed
- artifact-contract incompatibility between a configured ST-LRPS model and the
  benchmark request
- missing RIC columns when RIC metrics are requested
- missing distance/time units in metrics JSON

Domain-envelope issues are recorded as warnings. Warnings do not fail the run.

By default, validation failure returns a non-zero CLI exit code. For exploratory
runs only:

```bash
lunaris-benchmark --config configs/benchmarks/st_lrps_1day_high_degree.json --allow-validation-fail
```

## Quick Mode

`--quick` reduces the scenario count and duration, avoids expensive high-degree
work, and writes synthetic benchmark artifacts. It is intended for CI and for
checking that the config, manifest, output layout, report, and validator all
work without large SPICE/gravity/checkpoint files.

```bash
lunaris-benchmark \
  --config configs/benchmarks/st_lrps_1day_high_degree.json \
  --out outputs/gravity_benchmark/quick_smoke \
  --quick
```

Quick-mode numbers are pipeline smoke-test evidence, not scientific benchmark
claims. Synthetic artifacts are stamped
`SYNTHETIC SMOKE TEST - NOT A SCIENTIFIC BENCHMARK` in `report.md` and
`metrics_summary.json`.

## Paper-Safe Mode

`--paper-safe` (or `paper_safe: true` in the config) makes a benchmark
defensible. It **hard-fails before producing any output** on unsafe settings and
forces the strict flags so they cannot be re-enabled by an `allow_*` flag:

```bash
lunaris-benchmark --config configs/benchmarks/st_lrps_1day_high_degree.json --paper-safe
```

Paper-safe enforces: no synthetic/quick output; `allow_contract_mismatch`,
`allow_domain_extrapolation`, `allow_legacy_artifact`, `allow_validation_fail`
all false; `strict_domain` true; no truth-as-baseline unless
`validation.truth_baseline_justification` is given; a real surrogate whose
artifact contract matches the config and whose altitude domain covers all
scenarios. It writes `resolved_config.json`, `benchmark_manifest.json` (with a
`paper_safe` block), `validation_report.json`, and `run_command.txt`. See
[ST_LRPS_VALIDATION_HYGIENE.md](ST_LRPS_VALIDATION_HYGIENE.md) for the full
recommended paper-evidence workflow.

## Paper Trail

When citing or recording a benchmark result in a paper or technical note, keep
the benchmark config, `benchmark_manifest.json`, `resolved_config.json`,
`metrics_summary.json`, `scenario_results.csv`, and `validation_report.json`
together. The manifest hash and resolved config hash are the shortest anchors
for tracing the run.
