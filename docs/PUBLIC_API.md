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
| `lunaris-batch` / `lunaris-mc` | Batch and Monte Carlo style propagation |
| `lunaris-perturbation-budget` | Acceleration and perturbation budget analysis |
| `lunaris-ui` / `lunaris-launcher` | Mission desktop UI and launcher |
| `lunaris-studio` | Optional ST-LRPS Studio UI |
| `lunaris-train`, `lunaris-eval`, `lunaris-benchmark` | Optional ST-LRPS workflows |

## Stable Python Surface

These modules are intended for direct import:

| Module | Public objects |
| --- | --- |
| `lunaris` | `__version__` |
| `lunaris.api` | `load_default_config`, `replace_sim_config`, `DynamicsEngine`, `propagate`, `MonteCarloConfig`, `MonteCarloEngine` |
| `lunaris.common` | Flat re-exports from `lunaris.common.constants` and `lunaris.common.type_defs` |
| `lunaris.common.constants` | Physical constants and unit conversions such as `DAY_S`, `C_LIGHT`, `MU_MOON`, `R_MOON`, `AU`, `DEG2RAD`, `RAD2DEG` |
| `lunaris.common.type_defs` | `GravityConfig`, `AdaptiveDegreeConfig`, `PerturbationFlags`, `SolidTideConfig`, `TimeConfig`, `InitialState`, `SpacecraftProps`, `PropagatorConfig`, `PropagationResult`, `SimulationHistory` |
| `lunaris.common.hashing` | `canonical_json_text`, `canonical_json_sha256` |
| `lunaris.common.paths` | `find_project_root`, `project_root_from_file`, `data_dir_from_root` |
| `lunaris.batch` | `MonteCarloEngine`, `generate_standard_normal_design`, `sample_initial_states`, `sample_spacecraft_props`, `HDF5TrajectoryView`, `load_mc_result` |
| `lunaris.core.config` | `SimConfig`, `load_default_config`, `get_default_config`, `replace_sim_config`, `VisualConfig`, `OutputConfig` |
| `lunaris.core.dynamics` | `DynamicsEngine` |
| `lunaris.core.propagator` | `propagate`, `make_time_grid`, `build_events` |
| `lunaris.core.propagation` | Canonical propagation package; `propagate`, `make_time_grid`, `build_events` |
| `lunaris.physics.spherical_harmonics` | `GravityModel` |
| `lunaris.physics.ephemeris` | `SpiceBuildConfig`, `EphemerisManager`, `build_spice_tables`, `build_tables` |
| `lunaris.surrogate.runtime` | `SurrogateGravityModel` for production-facing ST-LRPS inference |
| `lunaris.surrogate.runtime_adapter` | Historical compatibility alias for production-facing ST-LRPS inference |
| `lunaris.analysis.postprocess` | `process_simulation_results`, orbital/invariant extraction helpers |
| `lunaris.analysis.reporting.manager` | `plot_all` |

For new scripts, prefer `lunaris.api` unless you intentionally need a lower-level
module listed below. The examples in `examples/` are the executable reference for
this Python API.

`lunaris.common.lunar_data` is a dependency-light bridge for lunar gravity-path
resolution and lunar body-signature checks. It is stable for internal framework
subsystems, but downstream scripts should prefer `lunaris.api` or the documented
console workflows unless they are deliberately extending data/artifact plumbing.

## Configuration Contract

Full simulation configuration flows through `SimConfig`:

```python
from dataclasses import replace

from lunaris.core.config import load_default_config

cfg = load_default_config()
cfg = replace(cfg, time=replace(cfg.time, duration_s=2 * 3600.0))
cfg.validate()
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
- `lunaris.core.mc_backend_policy`, `mc_propagator`, `torch_*` backend internals.
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

The `lunaris.surrogate.runtime_adapter` path is retained for historical
compatibility.

The old `lunaris.physics.surrogate_gravity` compatibility path is retired and
does not re-export ST-LRPS runtime objects. This keeps the physics layer
independent of the optional surrogate subsystem.

Training/evaluation package internals may change as artifact contracts evolve.
Downstream code should prefer the `lunaris-train`, `lunaris-eval`, and
`lunaris-benchmark` entry points unless it is deliberately extending the ST-LRPS
pipeline.
