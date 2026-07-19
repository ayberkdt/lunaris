# ST-LRPS Config And Artifact Contracts

ST-LRPS artifacts now carry an explicit, versioned scientific contract. The
goal is to make target semantics, dataset provenance, scaler assumptions, and
runtime compatibility visible to training, evaluation, propagation, and
benchmark code instead of relying on scattered config conventions.

## Existing Surfaces

Before the refactor, the same scientific assumptions were spread across several
objects and files:

- Common/core config dataclasses describe mission propagation and force-model
  switches.
- `TrainConfig` describes ST-LRPS data paths, architecture, degree range,
  altitude envelope, scaling policy, loss settings, and runtime kind.
- `TargetContract` records target mode, baseline kind/degree, target degree,
  frame, derivative convention, `mu_si`, `r_ref_m`, and `a_sign`.
- HDF5 dataset attrs carry unit system, body constants, degree metadata, target
  mode, altitude bounds, column labels, and derivative convention.
- `ScalerPack` and `scaler.json` carry x/u/a scaling plus provenance.
- Checkpoints contain model weights, config, scaler, architecture, dataset,
  scoring, and training state.
- Run manifests point at config, scaler, checkpoints, architecture signature,
  evaluations, and provenance files.
- Benchmark configs describe scenario sampling, propagation, truth degree,
  baseline models, ST-LRPS model path, output layout, and validation policy.

The failure mode was that a run could be syntactically loadable while still
being scientifically incompatible with a downstream request, for example a
residual trained over SH20 being used as if it corrected SH30, or a benchmark
running outside the trained altitude envelope.

## Artifact Contract

`ArtifactContract` lives in
`lunaris.surrogate.st_lrps.shared.contracts` and uses
`schema_version = 1`. New resolved configs and checkpoints include it under
`artifact_contract`; checkpoints also repeat critical contract blocks at the
top level.

Core fields:

- `target_mode`: `residual` or `full`
- `baseline_kind`: `none`, `point_mass`, or `spherical_harmonics`
- `base_degree` and `target_degree`
- `runtime_model_kind`: `potential_autograd` (the archived `force_direct` kind is rejected)
- `prediction_kind`: potential/residual label for the model output
- `output_dim`: `1` for scalar-potential artifacts
- `mu_si`, `r_ref_m`, and `a_sign`
- `altitude_min_km` and `altitude_max_km`
- `input_encoding`
- `scaler_contract`
- `dataset_contract`
- `architecture_signature`

The contract validates lunar body constants, residual degree ordering, runtime
kind, output dimension, scaler x/u/a blocks, dataset target/degree metadata, and
altitude range ordering. `potential_autograd` artifacts must use scalar
potential output (`output_dim=1`). An artifact declaring the archived
`force_direct` kind is rejected fail-closed (see the
`experimental/force-direct-archive` branch).

## Dataset Contract

Generated HDF5 clouds now include a normalized `DatasetContract` in both root
attrs (`dataset_contract_json`) and `/metadata/contract_json`.

Core fields:

- `schema_version`, `dataset_id`, `dataset_kind`, `created_at_utc`
- `generator_name`, `generator_version`, `repo_commit_sha`, `random_seed`
- `n_samples`, `dataset_layout`, and column names
- `target_mode`, `baseline_kind`, `degree_min`, `degree_max`
- `mu_si`, `r_ref_m`, `a_sign`, coordinate frame, and SI unit block
- `altitude_min_km`, `altitude_max_km`, sampling policy, and split policy
- source gravity model/path/SHA-256 and dataset content SHA-256 when known
- `derivative_convention`, which must be `dP_dphi_corrected_v1`

Training reads this contract before any optimizer work, runs dataset validation,
writes `provenance/dataset_validation_report.json`, creates
`provenance/split_manifest.json`, and copies the dataset contract into
`provenance/dataset_meta.json`, `config.json`, and `run_manifest.json`.

Validation rejects missing contracts, missing `target_mode`, missing degree
metadata, invalid unit system, invalid altitude bounds, non-lunar body
metadata, unsafe derivative conventions, non-finite values, invalid HDF5 shape,
and altitude-envelope violations. Contract-free or pre-fix datasets must be
regenerated with the current generator before training.

Dataset inspection and report commands:

```bash
lunaris-data inspect --data outputs/datasets/cloud.h5
lunaris-data validate --data outputs/datasets/cloud.h5 --out outputs/dataset_reports/cloud
lunaris-data report --data outputs/datasets/cloud.h5 --out outputs/dataset_reports/cloud
```

## Training Outputs

`build_resolved_config(...)` now writes:

- `dataset_contract`
- `artifact_contract`
- `training_config_hash`

Checkpoint payloads repeat:

- `artifact_contract`
- `dataset_contract`
- `resolved_config`
- `training_config_hash`
- `dataset_hash`
- `model_builder_version`

Run manifests also include the artifact contract, dataset contract, training
config hash, dataset hash, and a compact resolved-config summary. This makes a
single run directory self-describing without re-reading the original HDF5 file.

## Runtime Checks

Classical gravity files attach an immutable `GravityModelMetadata` contract to
the loaded field: model ID, coefficient-file SHA-256, normalization, coefficient
frame, tide system, and the source GM/reference radius. When an ephemeris frame
is available, a known coefficient-frame mismatch fails before propagation. A
known non-tide-free static field cannot be combined with additive solid tides;
strict runs also reject unknown/incomplete gravity metadata.

Canonical ephemeris archives use schema version 2 and store matched SPICE
position and velocity tables in SI units. Loading a position-only legacy NPZ
fails closed so a resumed run cannot silently switch from cubic Hermite to a
different interpolant. In-memory custom providers may omit both velocity tables
only through the explicitly labelled Catmull-Rom compatibility path.

`load_surrogate_force_model(...)` validates the checkpoint contract before
returning a runtime object. The returned `SurrogateForceModel` exposes:

- `artifact_contract`
- `target_contract`
- `run_manifest`

Strict loading requires an embedded versioned contract. Contract-free
checkpoints are rejected and must be regenerated.

`load_surrogate_force_model` loads `potential_autograd` artifacts as
`SurrogateForceModel`. An artifact declaring the archived `force_direct` kind is
rejected with a clear error pointing at the `experimental/force-direct-archive`
branch.
Domain checks use the artifact altitude envelope, and `strict_domain=True`
raises instead of extrapolating when inputs leave the trained shell or scaler
radius.

## Benchmark Compatibility

Config-driven benchmark runs build a requested artifact contract from the
benchmark config and compare it to the selected ST-LRPS model directory.

Default behavior:

- baseline kind and degree must match
- target degree must match truth degree
- runtime kind must match
- lunar constants and acceleration sign must match
- altitude extrapolation is a hard error unless allowed

Compatibility results are written to both `resolved_config.json` and
`benchmark_manifest.json` under `contract_compatibility`, then included in
`validation_report.json`.

Explicit overrides:

```bash
lunaris-benchmark --allow-contract-mismatch
lunaris-benchmark --allow-domain-extrapolation
```

`--allow-contract-mismatch` downgrades contract mismatches to warnings. Use it
for exploratory comparisons only; it is not appropriate for benchmark claims.

## Contract-Free Artifacts

Artifacts without `artifact_contract` are not runtime-compatible. Regenerate the
ST-LRPS run so checkpoints, `config.json`, and the run manifest all embed the
current dataset and artifact contracts.

## Limitations

- The only supported runtime model kind is `potential_autograd`; the archived
  `force_direct` kind (`experimental/force-direct-archive`) is rejected.
- Dataset and source gravity hashes are recorded when known. Old local datasets
  may lack these hashes, which is reported as a warning.
- The contract does not prove model accuracy; it proves that the artifact and
  downstream request describe the same scientific problem.
