# Public API

Lunaris is under active development, so this page defines the supported public
surface and its stability expectations, separating it from internal
implementation modules. Public means "reasonable to use in examples, scripts, and
downstream workflows"; it does not mean the whole package is frozen forever. Pin
a version when you need a stable surface.

## Stable Entry Points

Console scripts are the most stable user-facing API:

| Command | Purpose |
| --- | --- |
| `lunaris` | Single-run orbit propagation |
| `lunaris-data` | External data download, verification, path discovery, ST-LRPS dataset inspection |
| `lunaris-batch` | Batch/ensemble propagation |
| `lunaris-perturbation-budget` | Acceleration and perturbation budget analysis |
| `lunaris-ui` / `lunaris-launcher` | Mission desktop UI and launcher |
| `lunaris-studio` | Optional ST-LRPS Studio UI |
| `lunaris-train`, `lunaris-eval`, `lunaris-benchmark` | Optional ST-LRPS workflows (`potential_autograd`) |
| `lunaris-validate` | Validation runners: gravity-reference field/trajectory checks and the ST-LRPS validation suite |
| `lunaris-ablation` | ST-LRPS ablation suite runner |
| `lunaris-st-lrps-paper-evidence` | End-to-end paper-evidence pipeline (multi-seed training, benchmarks, ablation, evidence manifest) |
| `lunaris-frozen-search` | Surrogate-assisted frozen-orbit candidate search (staged screening -> classical SH validation -> family report; experimental) |

This table is the canonical console-script inventory: every entry in
`pyproject.toml [project.scripts]` must appear here (enforced by
`tests/test_repo_hygiene.py::test_console_scripts_documented_in_public_api`).

## Python Surface Tiers

Python imports have three explicit stability levels. The tier tables below are
the canonical inventory. `docs/public_api_manifest.json` machine-encodes the
subset of these modules whose literal `__all__` is snapshotted in
`docs/api_snapshot.json`; documented modules without a literal `__all__`
(for example `lunaris.common`, whose exports are assembled dynamically, and
the convenience modules `lunaris.common.hashing`, `lunaris.common.paths`, and
`lunaris.analysis.postprocess`) appear in the tables only.

### User-stable public API (`user-stable`)

These are the preferred downstream imports. Within a released MINOR line they
follow the deprecation process in [VERSIONING.md](VERSIONING.md). Pre-release
builds can still change before the final release, with the change called out in
the changelog.

| Module | Public objects |
| --- | --- |
| `lunaris` | `__version__` |
| `lunaris.api` | `load_default_config`, `replace_sim_config`, `DynamicsEngine`, `propagate`, `BatchPropagationConfig`, `BatchPropagationEngine`, `BatchPropagationResult` |
| `lunaris.batch` | `BatchPropagationEngine`, `BatchPropagationConfig`, `BatchPropagationResult`, `generate_standard_normal_design`, `sample_initial_states`, `sample_spacecraft_props`, `HDF5TrajectoryView`, `load_batch_result`, `batch_entry` |
| `lunaris.core.propagation` | `propagate`, `make_time_grid`, `build_events`, `PropagationResult`, `EventOutcome`, `EventSpec`, `event_outcome_from_solver_events`, `TimeGridPlan`, `StepSizePlan`, `IntegrationPlan`, `resolve_time_grid_plan`, `resolve_step_size_policy`, `resolve_integration_plan` |
| `lunaris.surrogate.runtime` | `SurrogateGravityModel`, `SurrogateGravityMetadata`, `DEFAULT_ST_LRPS_RUNS_DIR`, `discover_st_lrps_model_dirs`, `find_checkpoint_for_st_lrps_run`, `find_latest_st_lrps_model_dir` |

### Documented provisional Python API (`documented-provisional`)

These lower-level modules are intended for advanced direct use, but signatures
may change at a MINOR release while Lunaris is 0.x. Changes still require a
changelog entry; removals of previously released names retain a compatibility
alias for at least one MINOR line.

| Module | Public objects |
| --- | --- |
| `lunaris.common` | Flat re-exports from `lunaris.common.constants` and `lunaris.common.type_defs` |
| `lunaris.common.constants` | Physical constants and unit conversions such as `DAY_S`, `C_LIGHT`, `MU_MOON`, `R_MOON`, `AU`, `DEG2RAD`, `RAD2DEG` |
| `lunaris.common.type_defs` | `GravityConfig`, `AdaptiveDegreeConfig`, `PerturbationFlags`, `SolidTideConfig`, `TimeConfig`, `InitialState`, `SpacecraftProps`, `PropagatorConfig`, `PropagationResult`, `SimulationHistory` |
| `lunaris.common.hashing` | `canonical_json_text`, `canonical_json_sha256` |
| `lunaris.common.paths` | `find_project_root`, `project_root_from_file`, `data_dir_from_root` |
| `lunaris.common.batch_defs` | `BatchPropagationConfig`, `BatchPropagationResult`, `StateUncertainty`, `SpacecraftUncertainty`, `build_batch_output_grid`, `validate_st_lrps_model_dir` |
| `lunaris.common.math_utils` | Shared numerical helpers, including `batch_y_to_elements`, the explicitly screening-only `screening_orbital_elements_vec`, `sample_grid_bilinear_kernel`, and `sample_2d_scaled_bilinear_kernel` |
| `lunaris.core.config` | `SimConfig`, `load_default_config`, `get_default_config`, `replace_sim_config`, `VisualConfig`, `OutputConfig` |
| `lunaris.core.dynamics` | `DynamicsEngine` and dependency-extraction helpers |
| `lunaris.physics.spherical_harmonics` | `GravityModel` and documented SH evaluators |
| `lunaris.physics.ephemeris` | `SpiceBuildConfig`, `EphemerisManager`, `build_spice_tables`, `build_tables` |
| `lunaris.analysis.postprocess` | `process_simulation_results`, orbital/invariant extraction helpers |
| `lunaris.analysis.reporting.manager` | `plot_all` and report assembly helpers |
| `lunaris.analysis.ensemble.statistics` | Propagated-ensemble statistics, covariance, RIC uncertainty, impact statistics, OE dispersion |
| `lunaris.analysis.ensemble.plotting` | Ensemble plots and `plot_ensemble_report` |
| `lunaris.analysis.ensemble.uq_report` | Provenance-stamped ensemble UQ report builder |

### Cross-subsystem internal contracts (`cross-subsystem-internal`)

These public-looking names exist so one Lunaris subsystem can depend on another
without importing a private underscore symbol. They are not a supported
downstream API. Their current `__all__` is snapshotted because internal callers
still need reviewable change control.

| Module | Contract objects |
| --- | --- |
| `lunaris.physics.relativity_effects` | `schwarzschild_components`, `external_schwarzschild_diff_components`, `de_sitter_components`, `external_1pn_components`; pre-rc2 underscore aliases remain through at least 0.2.x |
| `lunaris.surrogate.st_lrps.networks.models` | `compute_harmonic_w0_bands`, `get_output_head_params`; pre-rc2 underscore aliases remain through at least 0.2.x |

Terminology note: the canonical concept is batch/ensemble propagation —
`BatchPropagation*` names for execution and `lunaris.analysis.ensemble` for
statistics and reporting. "Monte Carlo" appears only as the statistical
description of the `random` sampling design, alongside Latin Hypercube and
Sobol variants.

For new scripts, prefer the user-stable `lunaris.api` facade unless you
intentionally need a provisional lower-level module. The examples in
`examples/` are the executable reference for this Python API.

`lunaris.common.lunar_data` is a dependency-light bridge for lunar gravity-path
resolution and lunar body-signature checks. It is stable for internal framework
subsystems, but downstream scripts should prefer `lunaris.api` or the documented
console workflows unless they are deliberately extending data/artifact plumbing.

## Naming And Boundary Policy

These rules make the public/internal boundary mechanical instead of
folklore:

- A single leading underscore means **private to its defining module and its
  own subsystem**. Code in another subsystem must not import it. The unit
  boundaries match the import-linter contracts: top-level packages
  (`lunaris.core`, `lunaris.batch`, ...), with each
  `lunaris.surrogate.st_lrps.*` subpackage (`data`, `networks`, `training`,
  `evaluation`, `runtime`, `ui`, ...) counted separately.
- A helper that is legitimately consumed across such a boundary must carry a
  public (non-underscored) name in its defining module, or be re-exported
  through an explicit facade. `tests/test_api_boundaries.py` enforces this
  for `src/lunaris`; `docs/api_snapshot.json` is the reviewable inventory
  (regenerate with `python tools/api_inventory.py --write`).
- An underscore-named **package** (for example
  `lunaris.surrogate.st_lrps.evaluation._gravity_benchmark`) is the
  sanctioned pattern for an internal implementation tree consumed through a
  single public facade module (`compare_gravity_models`).
- **White-box tests are legitimate**: tests may import private helpers to
  validate numerical internals (integrator steppers, relativity components,
  storage writers) — that is deliberate coverage, not API. The
  `tests/test_*_import_compat.py` files pin the *compatibility* surfaces: an
  underscore import path asserted there is a contract and must survive until
  its documented removal release.
- Compatibility shims that re-export retired underscore names (for example
  the dynamic fold in `lunaris.surrogate.st_lrps.training.cli`) survive at
  least one MINOR release after the rename lands, then go through the
  deprecation process in [VERSIONING.md](VERSIONING.md).

## Configuration Contract

Full simulation configuration flows through `SimConfig`:

```python
from dataclasses import replace

from lunaris.core.config import load_default_config, replace_sim_config

cfg = load_default_config()
# Sub-configs are plain frozen dataclasses (dataclasses.replace is fine there);
# the top-level SimConfig copy must go through the validating helper.
cfg = replace_sim_config(cfg, time=replace(cfg.time, duration_s=2 * 3600.0))
```

Do not construct ad-hoc dictionaries of physics flags or pass loose keyword
arguments through the engine. Use the frozen dataclasses in
`lunaris.common.type_defs` and `replace_sim_config` when a workflow needs a
validated copy.

## Internal Or Provisional Modules

These modules are implementation details unless a specific public object is
listed above:

- `lunaris.physics.*` low-level Numba kernels other than documented loaders or
  model facades.
- `lunaris.batch.backend_policy`, `batch_propagator`, `torch_*` backend internals.
- `lunaris.cli.*` parser and wiring helpers.
- `lunaris.ui.*` page/widget internals and web preview implementation details.
- `lunaris.surrogate.st_lrps.*` training, data, evaluation, UI, and artifact
  internals, except where exposed through console scripts, documented artifact
  contracts, or `lunaris.surrogate.runtime`.

## ST-LRPS Boundary

ST-LRPS is an optional advanced subsystem. Classical propagation must work
without a trained ST-LRPS artifact. Runtime integration should go through:

```python
from lunaris.surrogate.runtime import SurrogateGravityModel
```

The historical `lunaris.surrogate.runtime_adapter` and
`lunaris.physics.surrogate_gravity` paths have been removed. This keeps the
physics layer independent of the optional surrogate subsystem and avoids stale
import aliases in new experiments.

Training/evaluation package internals may change as artifact contracts evolve.
Downstream code should prefer the `lunaris-train`, `lunaris-eval`, and
`lunaris-benchmark` entry points unless it is deliberately extending the ST-LRPS
pipeline.
