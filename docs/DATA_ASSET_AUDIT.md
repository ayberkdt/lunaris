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
- `lunaris-data verify --strict` treats `strict_required` entries as required;
  `gm_de440.tpc` is now present and hash-checked.
- `lunaris-data verify --runtime` builds a small real SPICE ephemeris table from
  manifest-resolved files and reports any GM fallback warnings.
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
- The current NAIF lunar DE440 frame kernel has been promoted to
  `moon_de440_250416.tf` from the official NAIF main FK directory. The older
  `moon_de440_220930.tf[.txt]` remains accepted only as a legacy alias.

The local data root still lacks `datasets/st_lrps_cloud_suite.h5`, but that is
now an expected optional/generated state rather than an external-data gap. The
machine-readable inventory has been refreshed for the current NAIF kernel set.

## Executive Summary (pre-cleanup baseline)

> **Historical snapshot.** Every "not yet trustworthy" item below has since been
> resolved; see **Implementation Status** above for the current contract and
> **Next Cleanup Targets → Done** for the per-item closure. This section is kept
> verbatim as the pre-cleanup baseline so later changes remain measurable. For
> the live state, run `lunaris-data verify [--strict --runtime]`.

Current ephemeris data works at runtime:

- `MOON_PA` rotation tables build successfully.
- Sun/Earth position tables build successfully when third-body sampling is
  requested.
- SPICE kernel path resolution accepts local `.txt` wrappers such as
  `naif0012.tls.txt`.

The data-management layer was not yet trustworthy at the time of this snapshot
(all items now **resolved** — resolution noted inline):

- `lunaris-data verify` reported required ephemeris files missing because it
  checked exact manifest filenames and did not know the runtime alias policy.
  *Resolved: the verifier resolves manifest `aliases` and prints `(via alias: …)`.*
- `gm_de440.tpc` was absent, so Earth/Sun GM lookups fell back to constants.
  *Resolved: `gm_de440.tpc` downloaded, hash-recorded, and present; passes
  `verify --strict --runtime` with no GM fallback warnings.*
- `thermal_models/` and `assets/` existed locally but were not represented as
  first-class manifest groups. *Resolved: both are now manifest groups.*
- PDS label/XML companion files were present but not represented as required
  companions in the manifest. *Resolved: entries declare `companion_files`,
  which `verify` checks (and hash-checks when `sha256` is provided).*
- Albedo auto-selection could choose Diviner `DGDR_RA` rock-abundance data over
  LOLA `LDAM_*` albedo data. *Resolved: the albedo resolver accepts only LOLA
  LDAM products and rejects Diviner `DGDR_*` rasters even when they parse
  (`io_surface._looks_like_lola_ldam_albedo_label`).*

## Local Inventory

| Area | Current state | Runtime status | Manifest status |
| --- | --- | --- | --- |
| Ephemeris | DE440 SPK, LSK, PCK, GM PCK, lunar BPC, lunar FK are present | Works; strict runtime check uses `gm_de440.tpc` when present | Canonical NAIF files plus accepted legacy aliases |
| Gravity | `jggrx_1800f_sha.tab.txt` and label are present | Used by default gravity helpers | False negative for `.tab` exact filename |
| Topography | `ldem_64_float` IMG/LBL/XML are present; `ldem_16_float.img` lacks canonical label pair | `ldem_64_float` samples successfully | Only IMG is catalogued; companions are not |
| Albedo | `ldam_8_float`, `ldam_10_float`, and Diviner `dgdr_ra` are present | LDAM samples are valid; Diviner RA is a mis-selection risk | Only `ldam_8_float.img` is catalogued |
| Thermal | Diviner `dgdr_st` IMG/LBL/XML are present | Label parses as a cylindrical raster; no dedicated thermal-grid resolver yet | Group missing |
| Assets | `lroc_color_2k.jpg`, `lunar_map.jpg` are present | Discoverable by UI/visual helpers | Group missing |
| Datasets | `data/datasets/` is absent | No ST-LRPS cloud-suite dataset available | Optional dataset entry missing |

## Exact Gaps

These are the remaining current data-management gaps:

- `data/ephemeris_models/naif0012.tls`, `pck00011.tpc`,
  `gm_de440.tpc`, and `moon_de440_250416.tf` are present by exact name.
- Legacy wrapper files such as `naif0012.tls.txt`, `pck00011.tpc.txt`, and
  `moon_de440_220930.tf.txt` remain accepted for compatibility only.
- `data/gravity_models/jggrx_1800f_sha.tab` is missing by exact name; local
  `jggrx_1800f_sha.tab.txt` exists and runtime helpers use it.
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
- `moon_de440_250416.tf`:
  https://naif.jpl.nasa.gov/pub/naif/generic_kernels/fk/satellites/moon_de440_250416.tf
- Legacy `moon_de440_220930.tf` alias:
  https://naif.jpl.nasa.gov/pub/naif/generic_kernels/fk/satellites/a_old_versions/moon_de440_220930.tf

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
- Direct third-body SPICE smoke: passed without Earth/Sun GM fallback warnings
  after `gm_de440.tpc` was added.
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
4. Promote the current main-directory NAIF lunar FK to `moon_de440_250416.tf`,
   keeping `moon_de440_220930.tf[.txt]` as a legacy alias.
5. Add `thermal` and `assets` manifest groups.
6. Make albedo product selection LDAM-specific so Diviner `DGDR_RA` cannot be
   selected as an albedo grid.
7. Promote source hash information from this audit into `data/data_sources.json`.
8. Classify `datasets/st_lrps_cloud_suite.h5` as an optional generated artifact
   governed by the ST-LRPS dataset contract, not as a downloadable external
   data dependency.

Open:

- None for the external-data manifest/verification contract. Future ST-LRPS
  work may generate and validate a cloud-suite artifact for a specific
  experiment, but it is outside the baseline data bundle.

