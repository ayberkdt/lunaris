# Validation Layer

The validation layer contains independent physics, orbit, and cross-model validation harnesses. These tools compare model behavior against trusted references and write run-specific evidence for review.

## Package Boundaries

- `src/lunaris/analysis/`:
  post-processing and plotting of already generated simulation outputs.

- `src/lunaris/surrogate/st_lrps/evaluation/`:
  dataset-level and artifact-level evaluation of trained ST-LRPS models.

- `validation/`:
  physics-level, orbit-level, and cross-model validation against trusted references.

## Current Submodule

- `validation/gravity/`
  documentation and output schema for lunar gravity model validation. The
  executable harness itself now lives in the ST-LRPS package at
  `src/lunaris/surrogate/st_lrps/evaluation/compare_gravity_models.py` and is wired into the ST-LRPS
  Studio under **Analysis → Orbit-Level Benchmark**.

Current gravity validation command:

```bash
python -m lunaris.surrogate.st_lrps.evaluation.compare_gravity_models --help
```

## Future Expected Submodules

These submodules are expected to be implemented in the future:
- `validation/orbits/`
- `validation/monte_carlo/`
- `validation/reports/`

## Do not put here

Do not place generated outputs, run artifacts, checkpoints, or trained models under `validation/`. Write validation products under the repository-level `outputs/` directory (for example `outputs/gravity_benchmark/`) or an external scratch path. The `outputs/` tree is git-ignored.

## Current status

The gravity validation harness has been split: `src/lunaris/surrogate/st_lrps/evaluation/compare_gravity_models.py` remains the stable CLI/facade (and backs the `lunaris-benchmark` entry point), while the implementation lives in the internal subpackage `src/lunaris/surrogate/st_lrps/evaluation/_gravity_benchmark/` (`types`, `compute`, `metrics`, `modes`, `plotting`, `results_io`). The module path, CLI flags, and outputs are unchanged.
