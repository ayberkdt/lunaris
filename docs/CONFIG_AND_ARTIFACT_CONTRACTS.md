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
- `runtime_model_kind`: currently only `potential_autograd` is implemented
- `prediction_kind`: potential/residual/force label for the model output
- `mu_si`, `r_ref_m`, and `a_sign`
- `altitude_min_km` and `altitude_max_km`
- `input_encoding`
- `scaler_contract`
- `dataset_contract`
- `architecture_signature`

The contract validates lunar body constants, residual degree ordering, runtime
kind, scaler x/u/a blocks, dataset target/degree metadata, and altitude range
ordering.

## Dataset Contract

Generated HDF5 clouds now include a normalized dataset contract with:

- schema and dataset kind
- target mode, baseline kind, degree range
- `mu_si`, `r_ref_m`, `a_sign`
- altitude envelope
- coordinate frame and units
- generator/source gravity metadata
- dataset/source hashes when available
- derivative convention

Training rejects missing `target_mode`, missing degree metadata, invalid unit
system, invalid altitude bounds, non-lunar body metadata, and old derivative
conventions by default. Compatibility escape hatches are explicit:

```bash
lunaris-train --allow-legacy-target-mode-inference
lunaris-train --allow-missing-dataset-contract
lunaris-train --allow-legacy-derivative-convention
```

These flags are intended for inspecting old data, not for producing new
research artifacts.

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

`load_surrogate_force_model(...)` validates the checkpoint contract before
returning a runtime object. The returned `SurrogateForceModel` exposes:

- `artifact_contract`
- `target_contract`
- `legacy_contract`
- `run_manifest`

Strict loading requires an embedded versioned contract. Older checkpoints can
be inspected only with:

```python
load_surrogate_force_model(path, allow_legacy_contract=True)
```

The runtime rejects `force_direct` artifacts until that runtime is implemented.
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
lunaris-benchmark --allow-legacy-artifact
```

`--allow-contract-mismatch` downgrades contract mismatches to warnings. Use it
for exploratory comparisons only; it is not appropriate for benchmark claims.

## Legacy Behavior

Legacy artifacts without `artifact_contract` can still be normalized when the
caller opts in. The normalized contract is marked as inferred and the runtime
sets `legacy_contract=True`. New canonical checkpoints must embed the full
contract.

Evaluation reload paths still infer enough metadata to inspect old runs, but
runtime and benchmark entry points require explicit legacy allowances before
they will trust a contract-free artifact.

## Limitations

- The only implemented runtime model kind is `potential_autograd`.
- `force_direct` is reserved for a future distilled direct-force artifact and
  intentionally fails today.
- Dataset and source gravity hashes are recorded when known. Old local datasets
  may lack these hashes, which is reported as a warning.
- The contract does not prove model accuracy; it proves that the artifact and
  downstream request describe the same scientific problem.
