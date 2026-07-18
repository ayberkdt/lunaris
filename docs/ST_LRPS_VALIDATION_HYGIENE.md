# ST-LRPS Validation Hygiene & Paper-Safe Benchmarks

This document is the canonical reference for how ST-LRPS results are made
**reproducible and defensible**: how scalers are fit, how datasets are split,
what runtime frame the model lives in, and how the paper-safe benchmark mode and
strengthened benchmark validation prevent over-optimistic or mislabelled
numbers. It doubles as the developer summary for the validation-hygiene work.

## What ST-LRPS is (and is not)

* The **default, validation-safe ST-LRPS path is a scalar residual potential
  surrogate** (`runtime_model_kind="potential_autograd"`). The network learns a
  residual potential `ΔU(r)` above a lower-degree spherical-harmonic baseline.
* In that default path **acceleration is the autograd gradient** of the learned
  residual potential: `Δa = a_sign · ∇ΔU`. The `potential_autograd` model has no
  separately trained force head; acceleration is always derived from `ΔU`.
* The earlier **`force_direct` student runtime** (a direct residual-acceleration
  model, `runtime_model_kind="force_direct"`) has been **archived** in the
  `experimental/force-direct-archive` branch and is rejected fail-closed on main.
  Only the conservative `potential_autograd` surrogate is supported here.
* Dataset labels are **residual or full-field** according to an explicit
  `TargetContract` (`target_mode`, `baseline_kind`, `base_degree`,
  `target_degree`). Residual datasets carry `ΔU`/`Δa`; full-field datasets carry
  the total field minus the declared baseline.
* The runtime frame is **Moon-fixed / body-fixed Cartesian**
  (`moon_fixed_cartesian`). It is **not** an inertial / MCMF-inertial / PA model.
* **Representation ceiling: the training target degree (SH60).** ST-LRPS learns
  the residual field up to its training target; the SH61–100 band is
  structurally unrepresentable for the surrogate and forms its irreducible
  error floor against an SH100 numerical reference. When a benchmark uses SH100
  truth, ST-LRPS is therefore *not expected* to approach the SH100 baseline —
  that gap measures the ceiling, not a training failure. Quantitatively (from
  the real JGGRX 1800F coefficients, `validation/gravity/band_share_analysis.py`,
  1000 points, seed 20260718): at 80 km altitude the SH61–100 band carries
  **3.60%** of the non-spherical acceleration RMS (2.7e-5 m/s²) and the >100
  tail 0.45%; at 100 km, 1.81% and 0.15%. For scale, the SH21–60 residual band
  the network actually learns carries 29.1% (80 km) — the structural floor is
  roughly an eighth of the learned signal. (That 29.1% band share and the
  historical "29.1x median improvement" figure are a numerical coincidence;
  they measure unrelated quantities.)

## 1. Scalers are fit on TRAIN ONLY (no leakage)

The isometric scalers include **target** scalers for residual potential and
acceleration. Fitting them on validation rows leaks the validation target
distribution into training.

* For a single HDF5 file, the train/val/test/OOD split indices are built first,
  and `fit_scaler_streaming(..., indices=train_indices)` streams **only the
  train rows** (sampled and read in sorted chunks, memory-bounded).
* For independent train/val files, the scaler is fit only on the train file.
* `scaler.json` records provenance:
  `fit_scope="train_only"`, `split_policy`, `split_seed`,
  `train_count`/`val_count`/`test_count`, `train_index_hash`/`val_index_hash`/
  `test_index_hash`, `dataset_content_sha256`, and `dataset_contract_hash`.

Tests (`tests/test_st_lrps_scaler_leakage.py`) build a dataset whose validation
rows have extreme targets and prove the train-only scaler is unaffected while the
whole-file fit would be contaminated. The guarded invariant is simple: validation
and test targets must not influence any scaler used during training.

## 2. Split policies — interpolation vs generalization vs extrapolation

`split_policy` (CLI `--split-policy`, recorded in `split_manifest.json`):

| policy | what it measures |
|---|---|
| `seeded_random` | interpolation inside the cloud (quick/debug) |
| `altitude_stratified` | interpolation, altitude-balanced |
| `spatial_block` | **spatial generalization** — holds out whole Moon-fixed lon/lat blocks so train/val never share a local patch |
| `ood_low_altitude` | **altitude extrapolation** — holds out the lowest altitude band |
| `ood_high_altitude` | **altitude extrapolation** — holds out the highest altitude band |
| `spatial_plus_altitude_stratified` | spatial holdout with altitude kept balanced |

Knobs: `--spatial-lon-bins`, `--spatial-lat-bins`, `--spatial-val-block-fraction`,
`--spatial-test-block-fraction`, `--ood-low-altitude-max-km`,
`--ood-high-altitude-min-km`, `--ood-holdout-fraction`, `--split-seed`.

`split_manifest.json` records the policy, seed, counts, per-split index hashes,
and per-split altitude/latitude/longitude ranges plus spatial-bin / OOD-threshold
definitions, so train/val/test separation is fully auditable.

> **Random validation is interpolation validation, not generalization.** Cite
> spatial-block and OOD-altitude results for generalization/extrapolation claims.

## 3. Runtime frame safety

`SurrogateForceModel` exposes frame-explicit methods:

* `predict_residual_potential_fixed(r_fixed_m)`,
  `predict_residual_accel_fixed(r_fixed_m)`,
  `predict_total_accel_fixed(r_fixed_m, base_accel_fixed_fn)` — inputs are
  **Moon-fixed Cartesian**.
* `predict_residual_accel_inertial(r_inertial_m, q_i2f)` (and `_potential_`,
  `_total_` variants) rotate inertial → fixed, evaluate, and rotate the
  acceleration back to inertial.
* The unsuffixed methods (`predict_residual_potential`, ...) are documented
  **fixed-frame** wrappers.

The constructor reads the artifact's declared frame and **hard-fails** unless it
is `moon_fixed_cartesian`. Dynamics integration
(`surrogate/runtime/adapter.py`) already rotates inertial→fixed around
`acceleration_fixed` and back.

## 4. Paper-safe benchmark mode

`--paper-safe` (or `paper_safe: true` in the benchmark config) makes a benchmark
defensible. It **hard-fails before producing any output** if the configuration is
unsafe, and forces the strict flags so they cannot be bypassed by `allow_*`:

* `run_options.synthetic` must be false (synthetic is a smoke test, never a
  scientific benchmark); `quick` mode is forbidden.
* `allow_contract_mismatch`, `allow_domain_extrapolation`, and
  `allow_validation_fail` are all forced false; `strict_domain` is true.
* `allow_truth_baseline` is forbidden unless
  `validation.truth_baseline_justification` is provided.
* A real surrogate (`surrogate.enabled` + `surrogate.model_dir`) is required; its
  artifact contract must match the benchmark config and its altitude domain must
  cover all scenario altitudes.
* The run writes `resolved_config.json`, `benchmark_manifest.json` (with a
  `paper_safe` block), `validation_report.json`, and `run_command.txt`.

Synthetic output, when used outside paper-safe, is stamped
`SYNTHETIC SMOKE TEST - NOT A SCIENTIFIC BENCHMARK` in `report.md` and
`metrics_summary.json`.

## 5. Benchmark scenario / metadata validation

`validate_benchmark_outputs` enforces:

* `scenario_id` values are integers and (for standard benchmarks) the exact
  contiguous range `0..N-1`. Non-contiguous external IDs require
  `scenario_id_policy="external_noncontiguous"` **and** a `scenario_id_mapping.json`.
* `runtime_summary.csv` carries `n_scenarios` per model, equal to the expected
  count; `scenario_results.csv` has exactly that many rows per model.
* Model names are consistent across `metrics_summary`, `scenario_results`, and
  `runtime_summary`.
* `report.md`'s stated scenario count matches the validated count (no stale text).
* `total_runtime_s / n_scenarios` equals `runtime_per_scenario_s` within
  tolerance.

These checks make scenario-count drift explicit: a report cannot claim one
scenario count while the result tables, runtime table, or per-scenario runtime
were produced for a different count.

`validation_report.json` additionally carries a self-describing `evidence`
block (`benchmark_name`, `synthetic`, `quick`, `paper_safe`,
`scientific_evidence`, banner, resolved-config hash, timestamp): quick/synthetic
output is stamped `scientific_evidence: false`, a missing resolved config fails
closed, and a resolved config claiming `paper_safe` together with
synthetic/quick run options is a validation **error**. After validation,
`benchmark_manifest.json` records the final `validation` status
(`pending` → `passed`/`failed`), so the provenance artifact of a crashed or
failed run can never be mistaken for a validated one.

## 6. Validation suite beyond random

`evaluation/validation_suite.py` reports field-level metrics on **each** split
kind, clearly separated:

* **interpolation** — random / altitude,
* **spatial generalization** — spatial-block,
* **altitude extrapolation** — OOD low / high,
* **trajectory** — orbit-level propagation (hook; see the orbit benchmark).

Field metrics per split: residual potential RMSE [m²/s²], residual acceleration
RMSE [m/s²], relative acceleration error [%], angular error [deg], radial /
cross-radial RMS [m/s²], altitude/latitude/longitude-binned error, and
P95/P99/worst-1%/worst-5% tails.

```
python -m lunaris.surrogate.st_lrps.evaluation.validation_suite \
    --model-dir runs/st_lrps_train_... --data clouds/residual_cloud.h5 --out validation/suite_out
```

## 7. Ablation suite

`evaluation/ablation.py` provides the cumulative A0..A6 progression (default
matrix) plus optional A7/A8/A9 encoding variants:

* **A0** raw SIREN + Sobolev U/a only → **A1** +residual blocks → **A2**
  +multi-scale → **A3** +altitude-balanced loss → **A4** +direction loss →
  **A5** +radial/cross loss → **A6** full recommended (control).
* **A7** physical radial-decay encoding, **A8** real-SH basis, **A9** additive
  multi-band (non-default; run with `--matrix all` or `--only NAME`).

Outputs include `st_lrps_ablation_summary.csv` / `.md` with parameter count,
training time, best epoch, and validation/test/OOD metrics per ablation.

> **Monitor-only contract:** checkpoint selection uses the validation metric
> only; periodic evaluation runs *after* selection and never influences it.

```
python -m lunaris.surrogate.st_lrps.evaluation.ablation \
    --train-data train.h5 --val-data val.h5 --out-root ablation_runs --execute
```

## Recommended paper-evidence workflow

1. Generate a residual cloud with explicit `target_mode` and altitude bounds.
2. Train with a **generalization** split (`--split-policy spatial_block` or an
   OOD split) so the headline numbers are not interpolation; the scaler is
   train-only automatically.
3. Run the **validation suite** and report interpolation, spatial, and OOD field
   metrics separately.
4. Run the **ablation suite** (A0..A6) to show each component's contribution.
5. Run the orbit benchmark in **`--paper-safe`** mode; archive
   `benchmark_manifest.json`, `resolved_config.json`, `validation_report.json`,
   and `run_command.txt`.

## Limitations

* Spatial blocks are rectangular in lon/lat; near the poles cells cover less area.
* OOD validation measures altitude extrapolation only; it does not certify
  extrapolation in latitude/longitude or to other bodies.
* The orbit-level validation hook depends on the propagation/SPICE stack and is
  not exercised by the field-only suite.
* Paper-safe mode validates the contract and altitude envelope at the config
  level; it does not re-derive the truth model's own numerical accuracy.
* `runtime_model_kind="potential_autograd"` is the scalar residual-potential
  path: acceleration comes from autograd and is conservative by construction up
  to network smoothness/numerics. It is the only supported runtime kind; the
  archived `force_direct` kind is rejected fail-closed
  (`experimental/force-direct-archive`).

