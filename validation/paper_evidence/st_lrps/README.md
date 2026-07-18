# ST-LRPS Paper Evidence Pipeline

This is the **canonical location for ST-LRPS paper evidence**: reproducible
training runs, field/orbit validation, worst-case analysis, multi-seed
comparison, provenance manifests, and the tables/figures that back any
scientific claim about ST-LRPS.

Legacy-claim restrictions, including the five-day float32 result, are recorded
in [`EVIDENCE_STATUS.md`](EVIDENCE_STATUS.md). That scope gate applies even when
a legacy number appears in an external slide deck or report.

> This pipeline **prepares reproducible ST-LRPS evidence generation**. It does
> not by itself prove ST-LRPS performance or "complete validation". Random
> validation measures **interpolation**; spatial/OOD validation are
> **generalization stress tests**. The SH200 DOP853 model is a **numerical
> reference, not physical truth**. Results are valid only for the specific
> dataset, split, artifact, and benchmark configuration used.

## Final-candidate eligibility

An ST-LRPS checkpoint is a final paper candidate only if it was produced with
train-only scalers, spatial/OOD split metadata, strict dataset contracts,
explicit runtime-frame metadata, and paper-safe benchmark outputs. A checkpoint
that lacks those requirements is **preliminary** and must not be used for final
paper claims. Mark non-final runs explicitly:

```bash
python -m lunaris.surrogate.st_lrps.paper_evidence.runner --mark-pre-hygiene outputs/run_to_mark
# writes outputs/run_to_mark/PRE_HYGIENE.json  {status: pre_hygiene, not_for_final_paper_claims: true}
```

Final candidates are trained from the paper configs here, which enforce
train-only scalers and strict dataset contracts.

## Folder layout

```
validation/paper_evidence/st_lrps/
  README.md              <- you are here
  configs/               <- pointer to canonical configs (configs/st_lrps/paper/)
  scripts/               <- run_all_st_lrps_paper_evidence.py (thin wrapper)
  manifests/             <- evidence_manifest.json (hashes + environment + provenance)
  training/              <- per-candidate training evidence bundles
  field_validation/      <- per-candidate field-validation CSVs (Task 4)
  orbit_benchmarks/      <- per-config orbit benchmark outputs (Task 5)
  worst_case_analysis/   <- worst-case scenario analysis (Task 6)
  ablation/              <- ablation suite outputs (Task 8, optional)
  figures/               <- figures regenerated from CSVs (Task 9)
  tables/                <- tables + multi-seed summary regenerated from CSVs (Tasks 7, 9)
```

`--evidence-root` controls where every stage writes (default: this folder). No
large checkpoints are committed: the evidence references each checkpoint by
**path + SHA-256**.

## Canonical configs (`configs/st_lrps/paper/`)

| Config | Purpose |
|---|---|
| `train_full_seed{42,123,2026}.json` | Final-candidate training (train-only scaler, strict contracts) |
| `field_validation.json` | Field validation across interpolation/spatial/OOD splits |
| `benchmark_1day_high_degree.json` | Paper-safe 1-day high-degree orbit benchmark |
| `benchmark_5day_general.json` | Paper-safe 5-day general low-lunar-orbit benchmark |
| `worst_case_analysis.json` | Worst-case scenario analysis over a benchmark output |
| `ablation_suite.json` | A0..A6 ablation suite (secondary/optional) |

Fill the `<FILL: ...>` placeholders (dataset and `model_dir` paths) before a real
run. Do not edit the safety fields.

### Current repository state

The checked-in configs are paper-safe templates, not completed evidence. A
dry-run is allowed to record unresolved `<FILL: ...>` values in
`manifests/evidence_manifest.json`; a real run fails closed until those values
point at the actual dataset, final-candidate model directory, and benchmark
output directory. Do not cite this folder as final ST-LRPS evidence until the
manifest contains non-dry training, field-validation, orbit-benchmark,
worst-case, multi-seed, and table entries for the same artifact chain.

## Running the pipeline

The runner is `lunaris-st-lrps-paper-evidence` (or `python -m
lunaris.surrogate.st_lrps.paper_evidence.runner`). Every stage supports
`--dry-run` (validate + plan + write a dry-run manifest, no heavy work).

### Full pipeline

```bash
lunaris-st-lrps-paper-evidence --stage all \
    --config configs/st_lrps/paper/train_full_seed42.json --model-dir outputs/st_lrps_paper/seed42
```

`--stage all` runs train -> field-validation -> orbit-benchmark -> worst-case ->
multi-seed -> tables. Ablation is **secondary** and is run explicitly with
`--stage ablation` so it never blocks final-candidate evidence.

### Train a final candidate

```bash
lunaris-st-lrps-paper-evidence --stage train --config configs/st_lrps/paper/train_full_seed42.json
# repeat for seed123 / seed2026, or reuse one config with --seed
```

### Field validation only

```bash
lunaris-st-lrps-paper-evidence --stage field-validation \
    --config configs/st_lrps/paper/field_validation.json --model-dir outputs/st_lrps_paper/seed42
```

Produces `field_validation_metrics.csv`, `field_validation_by_altitude.csv`,
`field_validation_by_lat_lon.csv`, and `field_validation_summary.md`, separated
by split kind (interpolation vs spatial generalization vs altitude extrapolation).

### Orbit benchmarks only (paper-safe)

```bash
lunaris-st-lrps-paper-evidence --stage orbit-benchmark --model-dir outputs/st_lrps_paper/seed42
# defaults to the 1-day + 5-day paper configs; override with --benchmark-config
```

Paper-safe mode forbids synthetic data, contract-free artifacts, and target-mode
mismatches. It requires a strict domain and a contract-checked surrogate. Outputs include
`orbit_benchmark_metrics.csv`, `orbit_benchmark_scenario_results.csv`,
`orbit_benchmark_runtime.csv`, `orbit_benchmark_summary.md`,
`benchmark_manifest.json`, and `validation_report.json`.

### Worst-case, multi-seed, tables

```bash
lunaris-st-lrps-paper-evidence --stage worst-case --config configs/st_lrps/paper/worst_case_analysis.json
lunaris-st-lrps-paper-evidence --stage multi-seed
lunaris-st-lrps-paper-evidence --stage tables   # regenerate all tables/figures from CSVs
```

### Reproduce paper tables/figures

`--stage tables` regenerates `tables/table_*.md` and `figures/*.png` **from the
CSV outputs**; numbers are never hardcoded into the documents, so the set is
fully regenerable. (Figures require matplotlib; without it, tables are still
produced and figures are skipped with a note.)

## Expected files after a successful run

- Training (`training/<run>/`): `training_config_resolved.json`,
  `artifact_contract.json`, `scaler.json`, `split_manifest.json`, `history.csv`,
  `training_summary.md`, `environment.json`, `train_command.txt`,
  `paper_evidence.json` (checkpoint referenced by path + SHA-256).
- Field validation: the four field CSV/MD files above.
- Orbit benchmark: the metrics/scenario/runtime CSVs + manifest + validation report.
- Worst-case: `worst_case_scenarios.csv`, `worst_case_summary.md`.
- Multi-seed: `multi_seed_summary.csv`, `multi_seed_summary.md`.
- Provenance: `manifests/evidence_manifest.json` records git commit + dirty
  state, Python/package/torch/CUDA versions, and config/dataset/split/scaler/
  checkpoint/benchmark hashes for every stage.

## Final vs preliminary

- **Final**: artifacts produced by this pipeline from the paper configs, with a
  train-only scaler and strict contracts, carrying full provenance.
- **Preliminary**: any artifact that lacks the final-candidate requirements
  above, or any synthetic/quick benchmark output. Synthetic output is a smoke
  test, never a scientific result.

## Limitations (read before citing)

- **Benchmark-config dependence.** Results hold for the specific scenario seed,
  count, altitude envelope, duration, integrator, and dtype in the config. One
  benchmark does not prove all lunar orbit regimes.
- **Dataset-envelope dependence.** The surrogate is trained on a residual cloud
  with a fixed altitude/coverage envelope; behavior outside it is extrapolation
  and is reported as OOD, not generalization.
- **Orbit-regime dependence.** Circular vs elliptic, low vs medium altitude, and
  inclination all affect error growth (phase drift vs radial). The worst-case
  analysis labels these explicitly.
- **Non-gravitational perturbations.** These benchmarks isolate the gravity
  field; SRP, drag (negligible at the Moon), third-body, and thermal effects are
  not included unless a benchmark explicitly enables them.
- **CPU/GPU force-stack differences.** Runtime and last-bit numerics can differ
  between CPU and GPU propagation backends; compare like-for-like.
- **Reference, not truth.** The high-degree SH DOP853 model is a numerical
  reference. "ST-LRPS beats SHxxx" is a statement about this reference and this
  configuration, not a universal accuracy claim.

## Correct phrasing

- "Results are valid for this dataset, split, artifact, and benchmark configuration."
- "Random validation measures interpolation."
- "Spatial/OOD validation measures generalization stress tests."
- "SH200 DOP853 is a numerical reference, not physical truth."
