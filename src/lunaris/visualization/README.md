# Visualization Tools

This directory contains standalone visualization helpers for propagated orbit results and lunar surface products.

## `lunaris.visualization.orbit_animation`

Purpose: orbit animation and trajectory visualization around the Moon.

Expected inputs:
- Propagated output/history data from a single run or Monte Carlo workflow.
- Optional trajectory metadata, surface context, and animation output path.

Primary API:

```python
from lunaris.visualization.orbit_animation import render_orbit_animation
```

The module currently exposes a Python rendering API rather than a stable command-line interface. Use it from analysis or UI code with propagated history data.

## `lunaris.visualization.surface_explorer`

Purpose: topography and albedo visualization for LOLA surface products.

Expected inputs:
- LOLA topography `.lbl` and `.img` files.
- Optional LOLA albedo `.lbl` and `.img` files.

Command example:

```bash
python -m lunaris.visualization.surface_explorer \
    --topo-label data/topography_models/ldem_64_float.lbl \
    --topo-img data/topography_models/ldem_64_float.img \
    --out-dir outputs/surface_explorer \
    --plot-2d --plot-3d
```

Large LOLA grids can be memory-heavy. Use `--stride-2d`, `--stride-3d`, and `--stride-albedo` for quick previews.

Generated plots and animations should be written under `outputs/`, `results/`, or another ignored output directory and should not be committed.
