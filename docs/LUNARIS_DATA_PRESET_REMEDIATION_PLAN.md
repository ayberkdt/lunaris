# Lunaris Data Preset Remediation Plan

## Summary

- Fix the mismatch where docs advertised `lunaris-data verify --preset ...` but
  the CLI only accepted `--group`.
- Make preset definitions fail fast when they reference names missing from
  `data/data_sources.json`.
- Keep ST-LRPS as an optional advanced subsystem; classical propagation must not
  require a trained ST-LRPS artifact.

## Implemented Changes

- Align `DATA_PRESETS` with the committed manifest:
  - `minimal`: SPICE runtime kernels plus GRAIL gravity.
  - `full-gravity`: `minimal` plus `naif_pck_gm_de440`.
  - `surface`: `full-gravity` plus topography, albedo, and thermal rasters.
  - `st-lrps-dev`: `full-gravity` plus the generated `st_lrps_cloud_suite`
    placeholder.
- Add fail-fast preset selection for `list`, `download`, and `verify` paths.
- Add `verify --preset`; keep overlapping list/download/verify selectors
  mutually exclusive so no command silently ignores a user-selected filter.
- Add the lightweight `lunaris.api` public facade for stable script imports.
- Replace legacy user-facing ST_LRPS branding in the main framework with
  Lunaris naming while preserving ST-LRPS subsystem-specific names.

## Validation

- Preset entries are checked against the committed manifest.
- CLI parser coverage includes `verify --preset minimal --runtime`, selector
  conflicts, and duplicate manifest-name fail-fast behavior.
- Docs exact preset lists are compared against `DATA_PRESETS`.
- Public API smoke tests cover both lazy `import lunaris.api` and normal facade
  object resolution.
