# Data Presets

`lunaris-data` is manifest-driven: every file still comes from
`data/data_sources.json`, with the same URL, alias, companion-file, and SHA-256
verification rules. Presets are only named selections of manifest entries for
common workflows.

## Commands

```bash
lunaris-data presets
lunaris-data list --preset minimal
lunaris-data download --preset minimal
lunaris-data verify --preset minimal --runtime
```

Use `--data-dir <path>` or `LUNARIS_DATA_DIR` when storing data outside the
repository checkout.

## Presets

| Preset | Purpose |
| --- | --- |
| `minimal` | Default single-orbit propagation: SPICE kernels plus GRAIL gravity. |
| `full-gravity` | `minimal` plus SPICE GM constants for strict/runtime provenance. |
| `surface` | `full-gravity` plus topography, albedo, and thermal surface rasters. |
| `st-lrps-dev` | `full-gravity` plus the generated ST-LRPS cloud-suite placeholder entry. |

## Exact Entry Lists

`minimal`:

- `naif_lsk_naif0012`
- `naif_spk_de440`
- `naif_pck_pck00011`
- `naif_moon_pa_de440`
- `naif_moon_fk_de440`
- `grail_gravity_jggrx`

`full-gravity`:

- all `minimal` entries
- `naif_pck_gm_de440`

`surface`:

- all `full-gravity` entries
- `lola_ldem_topography`
- `lola_albedo`
- `diviner_thermal_dgdr_st`

`st-lrps-dev`:

- all `full-gravity` entries
- `st_lrps_cloud_suite`

`st_lrps_cloud_suite` is generated locally, not downloaded from an external
provider. The data CLI reports it as a manual/generated entry so users do not
mistake it for a bundled requirement.

## Verification Posture

For normal onboarding:

```bash
lunaris-data verify --preset minimal --runtime
```

For stricter data provenance:

```bash
lunaris-data verify --preset full-gravity --strict --runtime
```

Surface and ST-LRPS workflows should verify their preset before starting a long
run. Missing optional/generated ST-LRPS data should not block classical
propagation.
