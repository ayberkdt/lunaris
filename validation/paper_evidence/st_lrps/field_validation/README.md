# field_validation/

Field-level (residual) validation evidence, written per candidate under a
`<model-name>/` subfolder by `--stage field-validation`:
`field_validation_metrics.csv`, `field_validation_by_altitude.csv`,
`field_validation_by_lat_lon.csv`, `field_validation_summary.md`.

Random/altitude splits are **interpolation**; spatial-block is **spatial
generalization**; OOD low/high are **altitude extrapolation** — never conflate
them.
