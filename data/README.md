# External Data Directory

Place local SPICE kernels, gravity coefficient files, lunar surface products,
trained checkpoints, and other large scientific inputs under this directory in
an editable checkout.

Large binary data are intentionally not packaged with `lunaris`. On HPC systems,
prefer setting `LUNARIS_DATA_DIR` to a shared filesystem path that contains the
same subdirectory layout, for example:

- `ephemeris_models/`
- `gravity_models/`
- `topography_models/`
- `albedo_models/`
- `thermal_models/`
- `assets/`
