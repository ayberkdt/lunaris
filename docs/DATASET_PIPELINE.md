# ST-LRPS Dataset Pipeline

This note describes the strict dataset-generation and validation path used by
the ST-LRPS surrogate. The goal is to make every HDF5 cloud self-describing
before training can consume it.

## Dataset Contract

Generated HDF5 files embed `dataset_contract_json` in root attrs and
`/metadata/contract_json`. The contract is versioned with
`schema_version = 1` and records:

- dataset identity, kind, generator name/version, creation time, repo commit,
  random seed, and sample count
- target semantics: `target_mode`, `baseline_kind`, `degree_min`,
  `degree_max`, and derivative convention
- physics constants and frame: `mu_si`, `r_ref_m`, `a_sign`,
  `coordinate_frame`, and SI units
- altitude envelope, sampling policy, split policy, HDF5 layout, and columns
- source gravity model path/hash and dataset content hash when available

Residual datasets are strict by default: `degree_max > degree_min`, a non-none
baseline is required, and the derivative convention must be
`dP_dphi_corrected_v1`.

## Generation Rules

`spatial_cloud_generator` writes new datasets under `outputs/datasets/` unless
an explicit output path is supplied. It refuses to write into the source package
tree and refuses to overwrite existing outputs unless `--overwrite` is passed.

Every generated dataset writes both compatibility attrs and the normalized
contract block. New generation should also carry source-gravity provenance
(`source_gravity_model`, `source_gravity_file_path`,
`source_gravity_file_sha256`) so downstream runs can tie labels to a specific
gravity model.

## Validation

Use `lunaris-data validate` before training or publishing a cloud:

```bash
lunaris-data validate --data outputs/datasets/cloud.h5 --out outputs/dataset_reports/cloud
```

The validator checks:

- readable `DatasetContract`
- HDF5 shape `(N, 7)`
- finite values and NaN/Inf counts
- position norms above the lunar reference radius
- altitude envelope consistency with a small float32 tolerance
- optional residual recomputation when truth/baseline callables are supplied
- duplicate point estimate and residual-acceleration outlier warnings

The JSON output is `dataset_validation_report.json`.

## Quality Reports

Use `lunaris-data report` for descriptive statistics:

```bash
lunaris-data report --data outputs/datasets/cloud.h5 --out outputs/dataset_reports/cloud --bins 30
```

The report includes altitude histograms, latitude/longitude statistics,
position norm stats, residual potential stats, residual acceleration magnitude
stats, finite fraction, duplicate fraction, split counts when supplied, source
gravity metadata, and the embedded contract. It writes:

- `dataset_quality_report.json`
- `dataset_quality_summary.md`

## Split Policy

Training now writes `provenance/split_manifest.json`. Supported automatic split
policies are:

- `seeded_random`: deterministic random train/validation split
- `random`: alias for seeded random using the configured seed
- `altitude_stratified`: keeps validation coverage across altitude bins

Reserved policies such as `spatial_block`, `ood_low_altitude`, and
`ood_high_altitude` are recognized as metadata concepts but intentionally raise
until implemented.

## Training Integration

`lunaris-train` reads the contract before any training step, runs dataset
validation, writes the validation report, creates a split manifest, and copies
the dataset contract into the run provenance and run manifest.

Strict defaults reject missing or inferred contracts. Old datasets can be
inspected or migrated only with explicit flags:

```bash
lunaris-train --allow-legacy-dataset-contract
lunaris-train --allow-missing-dataset-contract
lunaris-train --allow-legacy-derivative-convention
lunaris-train --allow-dataset-validation-fail
```

These flags should not be used for new research artifacts.
