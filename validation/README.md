# Validation Layer

The validation layer contains independent physics, orbit, and cross-model
validation harnesses. These tools compare Lunaris behavior against declared
references and write run-specific evidence for review.

Generated outputs, checkpoints, plots, cached trajectories, and reports belong
under `outputs/` or an external scratch path. The `validation/` tree is for
source, contracts, harness documentation, and small reference files.

## Package Boundaries

- `src/lunaris/analysis/`:
  post-processing and plotting of already generated simulation outputs.

- `src/lunaris/surrogate/st_lrps/evaluation/`:
  dataset-level, artifact-level, field-level, and orbit-level evaluation of
  trained ST-LRPS models.

- `validation/`:
  independent-reference validation material, gravity benchmark documentation,
  and reference contracts.

## Available Harnesses

- `validation/gravity/`
  documents the lunar gravity validation schema and the orbit-level benchmark
  command. The command entry point is:

  ```bash
  python -m lunaris.surrogate.st_lrps.evaluation.compare_gravity_models --help
  ```

  The same benchmark is available through `lunaris-benchmark` and the ST-LRPS
  Studio under **Analysis -> Orbit-Level Benchmark**.

- `validation/independent/`
  contains external-reference checks that use separate numerical paths from the
  production force models. This reduces correlated-error risk between the model
  under test and its validation reference.

  Reference paths include:

  - `independent_sh.py`: geopotential via `scipy.special.lpmn` plus explicit
    `4*pi` normalization, with acceleration taken as the numerical gradient of
    that potential.
  - `naif_ephemeris.py`: direct `spiceypy.spkpos` queries checked against the
    Lunaris `EphemerisManager`, including frame, target, observer, unit, and
    interpolation behavior.
  - `pyshtools_reference.py`: optional spherical-harmonic reference through the
    external `pyshtools` library, gated behind the `requires_pyshtools` pytest
    marker and skipped cleanly when unavailable.

  Tests live in `tests/test_independent_sh_validation.py`; the spherical-
  harmonic cross-check is pinned against closed-form point-mass and J2 anchors
  plus the real lunar gravity model.

- `validation/gravity_reference/`
  contains immutable field and trajectory reference contracts for the lunar
  spherical-harmonic gravity engine. The normal committed state-history
  benchmark is a non-rotating pyshtools/DOP853 regression. A separate TudatPy
  1.0.0 harness and checksummed evidence validate the physical DE440-rotated
  `MOON_PA` gravity-only path with an independent fixed-step RK4 integrator.
  Evidence includes degree/order 120 one-, five-, and thirty-day arcs, four
  additional five-day geometries, and a low-altitude degree/order-360 arc.
  Three-level RK4 convergence, hard predeclared caps, within-tool numerical
  bands, and measured altitude/latitude/longitude coverage are enforced. Large
  generated histories remain external; source, portable contracts, compact
  evidence, and whole-directory hashes are kept in the repository.

## Gravity Benchmark Layout

The gravity benchmark command is exposed by:

```text
src/lunaris/surrogate/st_lrps/evaluation/compare_gravity_models.py
```

Its implementation modules live under:

```text
src/lunaris/surrogate/st_lrps/evaluation/_gravity_benchmark/
```

The internal modules cover shared types, propagation/error computation, metric
aggregation, run modes, plotting, and result I/O. Treat
`compare_gravity_models.py` and `lunaris-benchmark` as the public command
surface.

## Reserved Validation Areas

The following names are reserved for validation material as the project grows:

- `validation/orbits/`
- `validation/ensemble/`
- `validation/reports/`

Keep these directories documentation/source oriented. Store produced artifacts
under `outputs/`.
