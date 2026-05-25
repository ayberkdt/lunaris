# Validation Layer

The validation layer contains independent physics, orbit, and cross-model validation harnesses. These tools compare model behavior against trusted references and write run-specific evidence for review.

## Package Boundaries

- `analysis/`:
  post-processing and plotting of already generated simulation outputs.

- `st_lrps/evaluation/`:
  dataset-level and artifact-level evaluation of trained ST-LRPS models.

- `validation/`:
  physics-level, orbit-level, and cross-model validation against trusted references.

## Current Submodule

- `validation/gravity/`
  lunar gravity model validation, including high-degree spherical-harmonic references, lower-degree spherical-harmonic baselines, and optional ST-LRPS comparison.

Current gravity validation command:

```bash
python -m validation.gravity.compare_gravity_models --help
```

## Future Expected Submodules

These submodules are expected to be implemented in the future:
- `validation/orbits/`
- `validation/monte_carlo/`
- `validation/reports/`

## Do not put here

Do not place generated outputs, run artifacts, checkpoints, or trained models under `validation/`. Write validation products under ignored output locations such as `results/`, `outputs/`, `artifacts/`, or external scratch storage.

## Current status

The current gravity validation harness is still monolithic in `validation/gravity/compare_gravity_models.py`; future refactors may split it into scenarios, metrics, runners, reports, and schemas.
