# Analysis Extras

This folder contains optional, manual analysis utilities that are useful during
development but are not part of the core mission propagation runtime.

## `surface_topography_explorer.py`

Moved from the old root-level `zextra_analizler_taslak/max_irtifa.py` draft
folder.

Purpose:
- Load LOLA topography products (`LDEM_*_FLOAT` label + IMG pairs).
- Optionally load LOLA albedo products.
- Generate quick 2D and 3D surface preview images.
- Report the maximum topography height inside a user-defined latitude/longitude
  window.

How to use:
1. Open `surface_topography_explorer.py`.
2. Edit the `USER INPUT` section for local `TOPO_LBL`, `TOPO_IMG`, and optional
   albedo paths.
3. Run:

```bash
python analysis/extras/surface_topography_explorer.py
```

Outputs are written to `analysis/extras/outputs_surface/`.

Notes:
- This script is intentionally not imported by the main application.
- It is a sandbox-style diagnostic helper for inspecting surface assets before
  using them in collision/radiation analyses.
