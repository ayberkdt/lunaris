# Lunaris Data Asset Audit

Generated: 2026-06-24

This audit freezes the current local `data/` state before the data-manifest and
loader cleanup. The goal is to make later changes measurable: we should know
which files already exist, which files are runtime-valid, which entries only
fail because of naming aliases, and which assets are genuinely missing.

The machine-readable snapshot is in `data/data_inventory.generated.json`.

## Implementation Status

Phase 2/3 cleanup is now represented in the runtime/data contract:

- `lunaris-data verify` accepts manifest aliases such as `.tls.txt`,
  `.tpc.txt`, `.tf.txt`, and `.tab.txt`.
- Manifest entries can declare required/optional companion files; labels and
  XML metadata are checked next to their primary rasters/kernels.
- `thermal` and `assets` are first-class manifest groups.
- `lunaris-data verify --strict` treats `strict_required` entries as required,
  which surfaces the missing `gm_de440.tpc` without breaking normal local runs.
- `lunaris-data verify --runtime` builds a small real SPICE ephemeris table from
  manifest-resolved files and reports GM fallback warnings.
- `load_default_config()` opportunistically includes `gm_de440.tpc` when it is
  present, but still works with existing local data bundles that do not have it.
- The albedo raster resolver now accepts only LOLA LDAM products; Diviner
  `DGDR_*` rasters are rejected for albedo even when they are parseable
  cylindrical grids.
- SHA-256 values from the frozen local inventory have been promoted into
  `data/data_sources.json` for the currently archived primary files and
  companion labels/XML files. `lunaris-data verify` now hash-checks companion
  files when their manifest entries provide `sha256`.
- `gm_de440.tpc` has been downloaded from NAIF, recorded with SHA-256, and now
  passes `lunaris-data verify --strict --runtime` without Earth/Sun GM fallback
  warnings.
- `st_lrps_cloud_suite.h5` is explicitly classified as an optional generated
  artifact, not an external download; when present it should be inspected and
  validated through the ST-LRPS dataset contract.

The local data root still lacks `datasets/st_lrps_cloud_suite.h5`, but that is
now an expected optional/generated state rather than an external-data gap. The
inventory and gap sections below remain a pre-cleanup baseline snapshot, so
they intentionally preserve the earlier missing-GM observation.

## Executive Summary

Current ephemeris data works at runtime:

- `MOON_PA` rotation tables build successfully.
- Sun/Earth position tables build successfully when third-body sampling is
  requested.
- SPICE kernel path resolution accepts local `.txt` wrappers such as
  `naif0012.tls.txt`.

The data-management layer is not yet trustworthy:

- `lunaris-data verify` reports required ephemeris files missing because it
  checks exact manifest filenames and does not know the runtime alias policy.
- `gm_de440.tpc` is absent, so Earth/Sun GM lookups fall back to constants.
- `thermal_models/` and `assets/` exist locally but are not represented as
  first-class manifest groups.
- PDS label/XML companion files are present but not represented as required
  companions in the manifest.
- Albedo auto-selection can choose Diviner `DGDR_RA` rock-abundance data over
  LOLA `LDAM_*` albedo data.

## Local Inventory

| Area | Current state | Runtime status | Manifest status |
| --- | --- | --- | --- |
| Ephemeris | DE440 SPK, LSK, PCK, lunar BPC, lunar FK are present | Works; GM falls back without `gm_de440.tpc` | False negatives for `.txt` aliases |
| Gravity | `jggrx_1800f_sha.tab.txt` and label are present | Used by default gravity helpers | False negative for `.tab` exact filename |
| Topography | `ldem_64_float` IMG/LBL/XML are present; `ldem_16_float.img` lacks canonical label pair | `ldem_64_float` samples successfully | Only IMG is catalogued; companions are not |
| Albedo | `ldam_8_float`, `ldam_10_float`, and Diviner `dgdr_ra` are present | LDAM samples are valid; Diviner RA is a mis-selection risk | Only `ldam_8_float.img` is catalogued |
| Thermal | Diviner `dgdr_st` IMG/LBL/XML are present | Label parses as a cylindrical raster; no dedicated thermal-grid resolver yet | Group missing |
| Assets | `lroc_color_2k.jpg`, `lunar_map.jpg` are present | Discoverable by UI/visual helpers | Group missing |
| Datasets | `data/datasets/` is absent | No ST-LRPS cloud-suite dataset available | Optional dataset entry missing |

## Exact Gaps

These are not necessarily runtime failures, but they are data-management gaps:

- `data/ephemeris_models/naif0012.tls` is missing by exact name; local
  `naif0012.tls.txt` exists and runtime accepts it.
- `data/ephemeris_models/pck00011.tpc` is missing by exact name; local
  `pck00011.tpc.txt` exists and runtime accepts it.
- `data/ephemeris_models/moon_de440_220930.tf` is missing by exact name; local
  `moon_de440_220930.tf.txt` exists and runtime auto-includes it.
- `data/gravity_models/jggrx_1800f_sha.tab` is missing by exact name; local
  `jggrx_1800f_sha.tab.txt` exists and runtime helpers use it.
- `data/ephemeris_models/gm_de440.tpc` is truly missing.
- `data/datasets/st_lrps_cloud_suite.h5` is truly missing.

## Source URLs To Preserve

Ephemeris:

- `naif0012.tls`:
  https://naif.jpl.nasa.gov/pub/naif/generic_kernels/lsk/naif0012.tls
- `de440.bsp`:
  https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de440.bsp
- `de440s.bsp`:
  https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de440s.bsp
- `pck00011.tpc`:
  https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/pck00011.tpc
- `gm_de440.tpc`:
  https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/gm_de440.tpc
- `moon_pa_de440_200625.bpc`:
  https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/moon_pa_de440_200625.bpc
- `moon_de440_220930.tf`:
  https://naif.jpl.nasa.gov/pub/naif/generic_kernels/fk/satellites/a_old_versions/moon_de440_220930.tf
- Current main-directory lunar frame alternative:
  https://naif.jpl.nasa.gov/pub/naif/generic_kernels/fk/satellites/moon_de440_250416.tf

Gravity:

- `jggrx_1800f_sha.tab`:
  https://pds-geosciences.wustl.edu/grail/grail-l-lgrs-5-rdr-v1/grail_1001/shadr/jggrx_1800f_sha.tab
- `jggrx_1800f_sha.lbl`:
  https://pds-geosciences.wustl.edu/grail/grail-l-lgrs-5-rdr-v1/grail_1001/shadr/jggrx_1800f_sha.lbl

Topography:

- `ldem_64_float.img`:
  https://pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/lrolol_1xxx/data/lola_gdr/cylindrical/float_img/ldem_64_float.img
- `ldem_64_float.lbl`:
  https://pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/lrolol_1xxx/data/lola_gdr/cylindrical/float_img/ldem_64_float.lbl
- `ldem_64_float.xml`:
  https://pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/lrolol_1xxx/data/lola_gdr/cylindrical/float_img/ldem_64_float.xml

Albedo:

- `ldam_10_float.img`:
  https://pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/lrolol_1xxx/data/lola_gdr/cylindrical/float_img/ldam_10_float.img
- `ldam_10_float.lbl`:
  https://pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/lrolol_1xxx/data/lola_gdr/cylindrical/float_img/ldam_10_float.lbl
- `ldam_8_float.img`:
  https://pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/lrolol_1xxx/data/lola_gdr/cylindrical/float_img/ldam_8_float.img
- `ldam_8_float.lbl`:
  https://pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/lrolol_1xxx/data/lola_gdr/cylindrical/float_img/ldam_8_float.lbl

Thermal and Diviner products:

- `dgdr_st_avg_cyl_032_img.img`:
  https://pds-geosciences.wustl.edu/lro/urn-nasa-pds-lro_diviner_derived1/data_derived_gdr_l3/cylindrical/img/dgdr_st_avg_cyl_032_img.img
- `dgdr_st_avg_cyl_032_img.lbl`:
  https://pds-geosciences.wustl.edu/lro/urn-nasa-pds-lro_diviner_derived1/data_derived_gdr_l3/cylindrical/img/dgdr_st_avg_cyl_032_img.lbl
- `dgdr_ra_avg_cyl_032_img.img`:
  https://pds-geosciences.wustl.edu/lro/urn-nasa-pds-lro_diviner_derived1/data_derived_gdr_l3/cylindrical/img/dgdr_ra_avg_cyl_032_img.img
- `dgdr_ra_avg_cyl_032_img.lbl`:
  https://pds-geosciences.wustl.edu/lro/urn-nasa-pds-lro_diviner_derived1/data_derived_gdr_l3/cylindrical/img/dgdr_ra_avg_cyl_032_img.lbl

## Runtime Checks Observed

- `tests/test_ephemeris.py` and `tests/test_loader_helpers.py`: passed.
- Default ephemeris rotation smoke: passed.
- Direct third-body SPICE smoke: passed with Earth/Sun GM fallback warnings.
- Surface smoke:
  - `ldem_64_float` sampled radius near `1736681.8196862936 m`.
  - `ldam_10_float` sampled albedo near `0.1839846670627594`.
  - `ldam_8_float` sampled albedo near `0.1839342676103115`.
  - Diviner `dgdr_ra` would sample near `0.004` if misused as albedo.
  - Diviner `dgdr_st` label parses with `UNIT = K`, scale `0.02`, missing constant `-32768`.

## Next Cleanup Targets

Done:

1. Add alias and companion-file support to `lunaris-data verify`.
2. Add `gm_de440.tpc` to the ephemeris manifest/default strict kernel set.
3. Fix `moon_de440_220930.tf` source URL.
4. Add `thermal` and `assets` manifest groups.
5. Make albedo product selection LDAM-specific so Diviner `DGDR_RA` cannot be
   selected as an albedo grid.
6. Promote source hash information from this audit into `data/data_sources.json`.
7. Classify `datasets/st_lrps_cloud_suite.h5` as an optional generated artifact
   governed by the ST-LRPS dataset contract, not as a downloadable external
   data dependency.

Open:

- None for the external-data manifest/verification contract. Future ST-LRPS
  work may generate and validate a cloud-suite artifact for a specific
  experiment, but it is outside the baseline data bundle.

