# Public API

Lunaris is still alpha software, so this page separates the supported public
surface from internal implementation modules. Public means "reasonable to use in
examples, scripts, and downstream workflows"; it does not mean the whole package
is frozen forever.

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
| `lunaris.common.type_defs` | `GravityConfig`, `AdaptiveDegreeConfig`, `PerturbationFlags`, `SolidTideConfig`, `TimeConfig`, `InitialState`, `SpacecraftProps`, `PropagatorConfig`, `PropagationResult`, `SimulationHistory` |
| `lunaris.core.config` | `SimConfig`, `load_default_config`, `get_default_config`, `replace_sim_config`, `VisualConfig`, `OutputConfig` |
| `lunaris.core.dynamics` | `DynamicsEngine` |
| `lunaris.core.propagator` | `propagate`, `make_time_grid`, `build_events` |
| `lunaris.physics.spherical_harmonics` | `GravityModel` |
| `lunaris.physics.ephemeris` | `SpiceBuildConfig`, `EphemerisManager`, `build_spice_tables`, `build_tables` |
| `lunaris.surrogate.runtime_adapter` | `SurrogateGravityModel` for production-facing ST-LRPS inference |
| `lunaris.analysis.postprocess` | `process_simulation_results`, orbital/invariant extraction helpers |
| `lunaris.analysis.reporting.manager` | `plot_all` |

The examples in `examples/` are the executable reference for this Python API.

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
  contracts, or `lunaris.surrogate.runtime_adapter`.

## ST-LRPS Boundary

ST-LRPS is an optional advanced subsystem. Classical propagation must work
without a trained ST-LRPS artifact. Runtime integration should go through:

```python
from lunaris.surrogate.runtime_adapter import SurrogateGravityModel
```

Training/evaluation package internals may change as artifact contracts evolve.
Downstream code should prefer the `lunaris-train`, `lunaris-eval`, and
`lunaris-benchmark` entry points unless it is deliberately extending the ST-LRPS
pipeline.
