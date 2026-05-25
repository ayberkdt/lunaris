# Visualization Tools

This directory contains reusable tools for visualizing the lunar environment and simulation results.

## `orbit_animation.py`

Purpose: 
Animates a propagated trajectory (from HDF5/NPZ output) in 3D around the Moon. 

Expected Inputs:
- A Monte Carlo or ST-LRPS orbit output file (`.h5` or `.npz`).
- Topography grid inputs (optional, for surface mapping).

Usage:
```bash
python -m visualization.orbit_animation ...
```

## `surface_explorer.py`

Purpose:
Visualizes topography (LOLA LDEM_*_FLOAT) and albedo (LOLA LDAM_*) products. Also includes utilities to query maximum topography inside coordinate windows.

Expected Inputs:
- Topography `.lbl` and `.img` paths.
- Albedo `.lbl` and `.img` paths (optional).

Example Usage:
```bash
python -m visualization.surface_explorer \
    --topo-label data/topography_models/ldem_64_float.lbl \
    --topo-img data/topography_models/ldem_64_float.img \
    --out-dir outputs/surface_explorer \
    --plot-2d --plot-3d
```
