# External Data Directory

Lunaris depends on large external scientific data files: SPICE/ephemeris
kernels, lunar gravity coefficients, LOLA/LDEM topography, LOLA LDAM albedo,
thermal rasters, UI assets, and locally generated ST-LRPS datasets. These files
are not bundled in the Python package. Keep them in this editable checkout's
`data/` directory, or on shared storage referenced by `LUNARIS_DATA_DIR` on HPC.

## Canonical Layout

```text
data/
  gravity_models/
  ephemeris_models/
  topography_models/
  albedo_models/
  thermal_models/
  assets/
  datasets/
```

The same layout is expected under `$LUNARIS_DATA_DIR` on cluster systems.
Resolution order is `--data-dir`, then `LUNARIS_DATA_DIR`, then the repository
`data/` directory.

## Acquiring Data

The asset catalogue is [`data_sources.json`](data_sources.json). Use the
headless `lunaris-data` tool to list, download, verify, and locate files:

```bash
lunaris-data list
lunaris-data download --group ephemeris
lunaris-data download --group gravity
lunaris-data verify
lunaris-data verify --strict --runtime
lunaris-data path
```

Entries with an official provider URL download directly from that provider
(NAIF/JPL or NASA PDS). Entries with a recorded SHA-256 are verified by
`lunaris-data verify`; supported local wrapper aliases such as `.tls.txt`,
`.tpc.txt`, `.tf.txt`, and `.tab.txt` are accepted. Companion labels/XML files
are also checked, and hash-checked when their manifest entry records a SHA-256.

`lunaris-data verify --strict` promotes strict-required assets, such as
`gm_de440.tpc`, to required status. `lunaris-data verify --runtime` additionally
builds a small SPICE ephemeris table from the resolved kernels, so the check
covers both file presence and runtime readability.

## Generated Datasets

`data/datasets/st_lrps_cloud_suite.h5` is not a download target. It is an
optional generated ST-LRPS artifact. When a local HDF5 dataset exists, inspect
and validate it with:

```bash
lunaris-data inspect --data data/datasets/st_lrps_cloud_suite.h5
lunaris-data validate --data data/datasets/st_lrps_cloud_suite.h5 --out outputs/dataset_reports/st_lrps_cloud_suite
```

## Notes

- Download large files once to shared storage and reuse them through
  `LUNARIS_DATA_DIR`; do not copy them into each run directory.
- Downloaded data and generated datasets are git-ignored and must not be
  committed. Only lightweight catalogues and documentation belong in Git.
