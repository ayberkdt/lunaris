# External Data Directory

Lunaris depends on large external scientific data files (lunar gravity
coefficients, SPICE/ephemeris kernels, LOLA/LDEM topography, optional albedo
grids) and locally generated datasets. These are **not** committed to Git or
bundled in the `lunaris` package — place them here in an editable checkout, or on
shared storage referenced by `LUNARIS_DATA_DIR` on HPC.

## Canonical layout

```text
data/
  gravity_models/
  ephemeris_models/
  topography_models/
  albedo_models/
  datasets/
```

The same layout is expected under `$LUNARIS_DATA_DIR` on HPC/cluster systems.
`LUNARIS_DATA_DIR` (read by the framework) overrides this repository `data/`
folder; point it at a shared scratch/project path that contains these
subdirectories. Additional category subdirectories (e.g. `thermal_models/`)
follow the same convention.

## Acquiring data

The asset catalogue is [`data_sources.json`](data_sources.json). Use the headless
`lunaris-data` tool to list, download, verify, and place files:

```bash
lunaris-data list
lunaris-data download --group ephemeris
lunaris-data download --group gravity
lunaris-data verify
lunaris-data path          # show the resolved data root
```

Data-root resolution order: `--data-dir` → `LUNARIS_DATA_DIR` → this repository
`data/` folder.

Entries in the manifest that carry an official URL (currently the NAIF/JPL SPICE
kernels) download directly. Entries without a pinned URL (e.g. GRAIL gravity,
LOLA topography/albedo) print the official provider and the directory to place
the file in manually — `lunaris-data` never downloads from unofficial mirrors.

## Notes

- Large files (400 MB+ gravity coefficients, SPICE kernels, topography grids)
  should be downloaded **once** to shared storage and reused via
  `LUNARIS_DATA_DIR`; do not copy them into each run directory.
- Downloaded data and generated datasets are git-ignored and must not be
  committed. Only this `README.md` and `data_sources.json` are tracked under
  `data/`.
