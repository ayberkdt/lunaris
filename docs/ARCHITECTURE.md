# Architecture

This document is the canonical reference for the internal architecture of the
`lunaris` framework. The [README](../README.md) gives the high-level overview and
usage; this file explains how the layers fit together and where to make changes.

ST-LRPS (Sobolev-Trained Lunar Residual Potential Surrogate) is the named
surrogate-gravity family that ships inside the framework as
`lunaris.surrogate.st_lrps`.

Lunaris is a modular monolith. Its propagation engine has four internal layers,
while surrogate modeling, desktop applications, analysis/validation, and
CLI/loaders are declared sibling subsystems with enforced dependency directions.
The import contracts in `pyproject.toml` are authoritative when prose and code
disagree.

Last verified: 2026-06-14, against base commit `1fd95880` plus the Phase
2.2-2.6 change set.

## Engine core layers

The propagation engine is organized into four strict layers. An engine layer
never imports from a layer above it.

### Layer 1 — `lunaris.common`
Dependency-light shared layer.
- `constants.py` — single source of truth for physical constants (SI units).
- `type_defs.py` — configuration dataclasses (`PerturbationFlags`, `TimeConfig`,
  `SpacecraftProps`, `InitialState`, …).
- `math_utils.py`, `time_utils.py` — pure helpers.
- `montecarlo_defs.py` — Monte Carlo configuration/result dataclasses.

### Layer 2 — `lunaris.physics`
Numba-JIT-compiled force-model kernels. Each file is one force model:
- `spherical_harmonics.py` — gravity field evaluation (reusable `SHWorkspace`).
- `third_body_effects.py` — Sun/Earth third-body perturbations.
- `solar_effects.py` — solar radiation pressure.
- `surface_effects.py` — `AlbedoConfig`/`ThermalConfig`, legacy cannonball
  albedo kernels, and the standalone albedo/thermal wrappers.
- `lunar_albedo.py` — Lambertian lunar albedo (reflected-solar) facets.
- `thermal_ir.py` — Lambertian lunar thermal IR radiation-pressure facets.
- `solid_tides.py` — elastic lunar solid-body tide potential and acceleration.
- `relativity_effects.py` — first-order post-Newtonian.
- `ephemeris.py` — SPICE kernel wrapper; ephemerides are pre-tabulated at startup.
- `gravity_adapter.py` — engine-facing gravity-provider normalization.

Physics models never import from `core/` or `surrogate/`.

### Layer 3 — `lunaris.core`
Numerical engine and configuration.
- `config.py` — `SimConfig` SSOT (`load_default_config()` returns a frozen
  config; `validate()` does cross-field checks).
- `dynamics.py` — assembles a Numba-compiled RHS closure by wiring the active
  physics models together.
- `propagator.py` — calls `scipy.integrate.solve_ivp()` with event detection;
  returns `PropagationResult(t, y, events, status)`.
- `events.py` — impact / periapsis-apoapsis / eclipse / occultation events.
- `monte_carlo_engine.py`, `mc_propagator.py`, `mc_backend_policy.py`,
  `mc_runner.py` — Monte Carlo orchestration and CPU/GPU backends.

### Layer 4 — `lunaris.analysis`, `lunaris.visualization`, `lunaris.ui`
Post-processing and presentation.
- `analysis/postprocess.py` — orbital elements, invariants, metrics.
- `analysis/reporting/` — report `manager`, `plotting`, `styling`.
- `analysis/monte_carlo/` — Monte Carlo `statistics` and `plotting`.
- `analysis/perturbation_budget/` — mission-analysis acceleration budgets,
  spherical-harmonic degree sensitivity, force-model uncertainty comparisons,
  and per-configuration gravity-degree recommendations. It calls existing
  physics kernels but does not alter propagation RHS behavior.
- `visualization/` — standalone orbit-animation and surface-explorer tools.
- `ui/app.py` + `ui/pages/`, `ui/core/`, `ui/theme/` — PySide6 desktop UI
  (mission simulator). The calm *Lunar Graphite* theme — typed tokens, the
  `THEME`/`LOG_COLORS`/`ORBIT_THEME` palettes, and the global stylesheet — is
  defined once in the binding-neutral `ui_foundation` package, the UI design
  single source of truth. `ui/core/ui_commons.py` and `ui/theme/` only
  re-export it for backward compatibility. See [docs/UI_THEME.md](UI_THEME.md).

### Support packages
Alongside the four layers:
- `lunaris.loaders` — dependency-light data loading (gravity coefficient files,
  SPICE kernels, topography/albedo grids) consumed by layers 2–3.
- `lunaris.cli` — console entry points (`lunaris`, `lunaris-mc`, …) and shared
  CLI argument helpers; wires user input into the `core` configuration. Optional
  subsystem commands use import-safe wrappers in `cli/entrypoints.py`.
- `lunaris.surrogate.st_lrps` — the ST-LRPS surrogate-gravity family
  (see [ST-LRPS surrogate](#st-lrps-surrogate)).
- `lunaris.surrogate.runtime_adapter` — the only production-facing adapter from
  engine gravity-provider semantics to ST-LRPS runtime artifacts.
- `lunaris.ui_foundation` — Qt-binding-neutral tokens, palettes, color helpers,
  and stylesheet generation shared by the mission UI and ST-LRPS Studio.

## Enforced subsystem boundaries

| Rule | Enforcement |
|---|---|
| `common` has no upward imports | import-linter: `common stays dependency-light (no upward imports)` |
| `physics` does not depend on core, presentation, or surrogate code | import-linter: `physics never imports core/analysis/visualization/ui` and `physics does not depend on the ST-LRPS subsystem` |
| `core` does not import desktop UI or the ST-LRPS data/training/evaluation/UI pipelines | import-linter: `core does not import the desktop UI or the ST-LRPS ML pipeline` |
| ST-LRPS inference stays independent of training/evaluation/UI | import-linter: `ST-LRPS runtime (inference path) stays light` |
| Analysis and visualization do not import desktop UI | import-linter: `analysis and visualization do not import the desktop UI` |
| ST-LRPS Studio consumes only the published UI foundation | import-linter: `ST-LRPS studio does not import mission-UI internals` |
| The shared UI foundation imports neither desktop application | import-linter: `UI foundation stays independent of both desktop applications` |
| All full simulation config construction/replacement is validated | `tests/test_sim_config_ssot.py` |
| Bare package import and every console `--help` survive missing optional dependencies | `tests/test_optional_dependency_boundaries.py` |
| Removed Studio-to-`data_pages` wildcard edges stay removed | `tests/test_st_lrps_ui_modularity.py` |

## Configuration (SSOT)

`lunaris.core.config` is the single source of truth. All configuration flows
through the frozen `SimConfig` dataclass; never pass ad-hoc kwargs.

```python
from lunaris.core.config import load_default_config

cfg = load_default_config()
# CLI overrides are applied in lunaris.cli.main via apply_args_to_config(cfg, args)
cfg.validate()  # cross-field consistency checks
```

Specialized non-CLI workflows must use
`lunaris.core.config.replace_sim_config(cfg, **changes)`, which validates the
copy before returning it. Direct `SimConfig(...)` construction outside
`core/config.py` is forbidden by repository contract tests.

Key sub-configs: `GravityConfig`, `SpiceBuildConfig`, `InitialState`,
`PerturbationFlags`, `SpacecraftProps`, `PropagatorConfig`, `TimeConfig`.

## Data flow

```
CLI (lunaris.cli.main) / UI (lunaris.ui.app)
  → lunaris.core.config (SimConfig)
  → lunaris.loaders (gravity model, SPICE kernels, surface grids)
  → lunaris.core.dynamics (build Numba RHS closure)
  → lunaris.core.propagator (solve_ivp → PropagationResult)
  → lunaris.analysis.postprocess (orbital elements, metrics)
  → lunaris.analysis.reporting.{plotting,manager} (PNG/PDF output)
```

Perturbation Budget Analysis is a sibling analysis flow:

```text
lunaris-perturbation-budget
  -> lunaris.analysis.perturbation_budget.config
  -> sampling (representative states and RIC frames)
  -> existing physics kernels / gravity loader
  -> acceleration, SH-increment, uncertainty, and recommendation tables
  -> CSV + Markdown outputs
```

## Perturbation flags

`PerturbationFlags` (in `lunaris.common.type_defs`) all default to `False` except
`enable_sh=True`. Enabling a flag requires the corresponding config section to be
non-`None` (e.g. `enable_srp=True` requires `cfg.srp`).

| Flag | Model | Status |
|------|-------|--------|
| `enable_sh` | Spherical-harmonics gravity (default degree 100, up to 1800) | Implemented |
| `enable_3rd_body_sun` / `enable_3rd_body_earth` | Third-body perturbations | Implemented |
| `enable_earth_j2` | Earth oblateness (differential) | Implemented |
| `enable_srp` | Solar radiation pressure | Implemented |
| `enable_albedo` | Reflected-solar radiation pressure (facet Lambertian) | Implemented |
| `enable_relativity_1pn` | First-order post-Newtonian | Implemented |
| `enable_thermal` | Lunar thermal IR radiation pressure | Implemented on CPU RHS |
| `enable_tides_k2` / `enable_tides_k3` | Elastic lunar solid-body tides | Implemented on CPU RHS |

Lunar albedo is configured through `AlbedoConfig` and evaluated in
`lunaris.physics.lunar_albedo`. It is a non-gravitational **reflected-solar**
radiation-pressure perturbation (sunlight reflected from the lunar surface and
received by the spacecraft); it belongs with SRP and thermal IR, not with
gravity. The default `lambert_facets` backend discretizes the Moon into
Moon-fixed latitude-longitude facets (the same discretization as thermal IR),
treats each facet as a Lambertian reflector with reflected exitance
`M_i = A_i * S_i * mu_sun_i`, and sums the contributions of facets that are
simultaneously sunlit (`mu_sun > 0`) and visible to the spacecraft
(`mu_view > 0`) before rotating the result back to the inertial frame. Per-facet
albedo `A_i` is precomputed at setup time from one of three `albedo_mode`
sources: `constant_albedo` (provider-free), `albedo_grid` (provider-supplied
[0,1] grid), or `scaled_dn_grid` (provider digital-number grid via
`A = scale*DN + offset`, with nodata falling back to `albedo_const`). The model
uses a dedicated coefficient `albedo_pressure_coefficient` (C_R_albedo), **not**
the SRP `cr`. An optional lunar-eclipse (Earth-umbra) dimming reuses the SRP
conical-shadow geometry. The legacy `simple` cannonball backend remains
available for backward compatibility via `albedo_model='simple'`. The facet
model is Lambertian only: it
does not model non-Lambertian BRDFs, wavelength dependence, surface roughness,
terrain self-shadowing beyond the incidence/visibility cutoffs, photometric
phase functions, multiple scattering, or local topography.

Thermal IR is configured through `ThermalConfig` and evaluated in
`lunaris.physics.thermal_ir`. The current model discretizes the Moon into
Moon-fixed latitude-longitude facets, treats each facet as a Lambertian emitter,
and rotates the resulting acceleration back to the inertial integration frame.
Supported modes are `constant_temperature`, `equilibrium_temperature`
(instantaneous solar incidence with no thermal inertia), and `temperature_grid`
(provider-supplied facet temperatures). The model is a radiation-pressure
perturbation only; it does not alter lunar gravity.

Solid tides are configured through `SolidTideConfig` and evaluated in
`lunaris.physics.solid_tides`. The model is an instantaneous elastic response
only. For each enabled tide-raising body (`earth`, `sun`, or both), the
Moon-fixed disturbing potential

```text
dU_l = k_l * mu_j / |R_j| * (R / |r|)^(l+1) * (R / |R_j|)^l * P_l(c)
```

is differentiated analytically and the resulting acceleration is rotated back
to the inertial integration frame. Degree 2 uses the documented default
`k2=0.02416` from the GRAIL/LRO monthly lunar Love-number solution reported by
Williams & Boggs (2015; [NASA PGDA product 96](https://pgda.gsfc.nasa.gov/products/96)).
Degree 3 has no project default:
`enable_tides_k3=True` requires an explicit `SolidTideConfig.k3` or CLI
`--tide-k3` value. The current model does not include dissipation/time lag,
ocean tides, or thermal tides.

## External data (`data/`)

Mandatory at runtime:
- `data/ephemeris_models/` — SPICE kernels (`.tls`, `.bsp`, `.tpc`, `.bpc`).
- `data/gravity_models/` — spherical-harmonic coefficients (e.g. `jggrx_1800f_sha.tab`).

Optional (only when the corresponding flag is enabled):
- `data/topography_models/` — lunar DEM rasters.
- `data/albedo_models/` — surface albedo grids.
- `data/thermal_models/` — thermal property grids.

Data-root discovery is folder-name independent: the repository root is located by
walking up to the first directory containing `pyproject.toml`, `.git`, or
`data/assets`. Loader overrides may be supplied via the `LUNARIS_LDEM_ROOT`,
`LUNARIS_ALBEDO_ROOT`, `LUNARIS_KERNEL_DIR`, `LUNARIS_LUNAR_MAP`, and
`LUNARIS_ASSETS_DIR` environment variables (generic `LDEM_ROOT`, `ALBEDO_ROOT`,
and `SPICE_KERNELS` fallbacks are also honored).

## Monte Carlo infrastructure

| Module | Purpose |
|--------|---------|
| `common/montecarlo_defs.py` | `MonteCarloConfig`, `StateUncertainty`, `SpacecraftUncertainty`, `MCRunResult` |
| `core/mc_propagator.py` | `GPUBatchPropagator` (CUDA RK4), `CPUBatchPropagator` (process pool) |
| `core/monte_carlo_engine.py` | `MonteCarloEngine.run()` — sampling, backend dispatch, HDF5/NPZ output |
| `analysis/monte_carlo/statistics.py` | `compute_mc_statistics()` → covariance, ellipsoids, impact probability, OE dispersion |
| `analysis/monte_carlo/plotting.py` | altitude envelopes, 3-D covariance tubes, impact map, OE dispersion |

```python
from lunaris.core.config import load_default_config
from lunaris.common.montecarlo_defs import MonteCarloConfig, StateUncertainty
from lunaris.core.monte_carlo_engine import MonteCarloEngine
from lunaris.analysis.monte_carlo.statistics import compute_mc_statistics
from lunaris.analysis.monte_carlo.plotting import plot_mc_report

sim_cfg = load_default_config()
mc_cfg = MonteCarloConfig(
    n_samples=500,
    state=StateUncertainty(sigma_r_m=500.0, sigma_v_m_s=0.5),
    use_gpu=True,
    mc_backend="auto",   # auto, cpu_sh, numba_cuda_sh (alias gpu_sh), torch_cuda_sh, gpu_st_lrps_potential, gpu_st_lrps_direct
    gpu_sh_degree=10,    # requested SH degree; numba_cuda_sh supports <=24, torch_cuda_sh is high-degree
    output_format="hdf5",
    output_path="outputs/monte_carlo/run.h5",
)
result = MonteCarloEngine(sim_cfg, mc_cfg).run()      # MCRunResult
stats = compute_mc_statistics(result)
figs = plot_mc_report(result, stats, output_path="outputs/monte_carlo/report.pdf")
```

Reload a saved run with `from lunaris.core.monte_carlo_engine import load_mc_result`.

### GPU classic-SH backends (Numba vs. Torch)
- Two distinct classic-SH GPU runtimes exist and are kept separate everywhere:
  - **`numba_cuda_sh`** (alias `gpu_sh`) — the Numba CUDA RK4 kernel. Its workspace
    uses compile-time fixed `(26 x 26)` per-thread arrays, so its degree ceiling is
    **24**. That ceiling is a kernel-workspace limit, **not** a physical one. Best
    for low-degree, high-throughput screening.
  - **`torch_cuda_sh`** — the PyTorch CUDA RK4 path
    (`lunaris.core.torch_sh_propagator.TorchSHBatchPropagator`) using the canonical
    `TorchSHGravityEvaluator`. Arbitrary degree, bounded only by the loaded
    coefficient file, VRAM, batch size, dtype, and step. This first runtime form is
    **gravity-only** (lunar SH + per-RK-stage Moon inertial↔fixed frame transform).
- **`degree > 24` with PyTorch CUDA available now uses `torch_cuda_sh`** (when the
  requested physics is supported), instead of falling back to CPU. The requested
  degree is never clipped — `requested_sh_degree` and `actual_sh_degree` are
  recorded separately, and a successful SH100 run reports `actual_sh_degree=100`.
- Backend selection is a single source of truth: `select_classic_sh_backend()`
  (in `mc_backend_policy`) decides between `numba_cuda_sh` / `torch_cuda_sh` /
  `cpu_sh`; `resolve_mc_backend_policy()` consumes that decision directly.
- An explicit `numba_cuda_sh` request above degree 24 obeys `gpu_sh_fallback_policy`:
  `compatible_gpu` (try `torch_cuda_sh`, else CPU), `cpu`, or `error`.
- The GPU paths do not support albedo, thermal IR, or solid tides; `torch_cuda_sh`
  additionally does not yet model third-body, Earth J2, SRP, or relativity (those
  force an explicit, recorded fallback). Use the CPU path for those models.
- `numba_cuda_sh` requires `numba` plus a CUDA GPU; `torch_cuda_sh` and the ST-LRPS
  GPU paths require PyTorch CUDA. The engine falls back to CPU with a warning and
  metadata when the requested GPU path is unavailable — and a CPU run is never
  labeled with a GPU backend/device in provenance.

## Performance notes

- All inner-loop physics use `@njit(cache=True)` / `@njit(parallel=True)`; avoid
  Python-level loops inside physics kernels.
- Ephemeris data is pre-tabulated at startup, not queried per integration step.
- Spherical-harmonic evaluation reuses an `SHWorkspace` to avoid heap allocation
  in the hot path.
- Default integrator is DOP853 (8th-order Runge-Kutta); step size is bounded via
  a Nyquist criterion on the gravity-field degree.

## ST-LRPS surrogate

`lunaris.surrogate.st_lrps` is a self-contained pipeline for training neural
networks that approximate the lunar gravity field as a **residual** above a
lower-degree spherical-harmonic baseline:

```
total acceleration = SH(degree_min) baseline + neural residual correction
```

| Subpackage | Purpose |
|------------|---------|
| `data/` | dataset definitions, spatial-cloud generation, dataset loading |
| `training/` | training config, CLI, engine, losses, metrics |
| `networks/` | neural-network architecture definitions |
| `artifacts/` | run layout, checkpoints, manifests, artifact validation |
| `evaluation/` | trained-model evaluation, ablation, orbit-level benchmark CLIs |
| `runtime/` | internal artifact loading and neural force-model inference |
| `shared/` | scaling utilities and target/derivative contracts |
| `ui/` | ST-LRPS Studio desktop UI |

The large UI and evaluation areas were measured in Phase 2.5. The only
low-risk split was Qt-independent HDF5 metadata inspection, now in
`ui/studio_parts/dataset_introspection.py`; see
[ST_LRPS_MODULARITY_AUDIT.md](ST_LRPS_MODULARITY_AUDIT.md).

Data generation pulls strictly from the lunar GFC model in
`data/dataset_parameters.py` and natively computes residual fields. Shared
lunar path/default/signature helpers live in `common/lunar_data.py`, so the
engine does not import the ML data pipeline:
- `degree_min` — maximum degree of the analytical base model; the network learns
  the difference from `degree_min + 1`. `-1` evaluates the full field.
- `degree_max` — target high-fidelity resolution.

```bash
python -m lunaris.surrogate.st_lrps.data.spatial_cloud_generator \
    --degree-max 50 --degree-min 10 --n-samples 250000 \
    --alt-range 30 120 --format h5 --workers 8
```

Model target semantics are recorded explicitly via a `target_contract` in new
configs/checkpoints, distinguishing residual labels from full-field labels and
keeping the runtime path aligned with the scaler and loss.

**Runtime.** The engine-facing API is
`lunaris.surrogate.runtime_adapter.SurrogateGravityModel`. It delegates artifact
loading and neural inference to the internal ST-LRPS API in
`runtime/force_model.py`.
`potential_autograd` (`SurrogateForceModel`) evaluates the learned scalar
potential and differentiates it with autograd to obtain residual acceleration,
which is added to the SH(`degree_min`) baseline. `force_direct`
(`DirectForceRuntime`) loads a 3-output residual-acceleration artifact and uses
`torch.no_grad()` inference with the acceleration scaler; it never predicts
`DeltaU`. `load_surrogate_force_model` dispatches by `runtime_model_kind` and
strictly validates artifact contracts, output dimension, and frame. Direct-force
is a faster inference target but is not conservative by construction and needs
separate curl / orbit-level validation.

**Frame.** ST-LRPS is a **Moon-fixed / body-fixed Cartesian** surrogate
(`moon_fixed_cartesian`). The runtime exposes explicit `predict_*_fixed`
(body-fixed inputs) and `predict_*_inertial(q_i2f)` (rotate in → evaluate fixed →
rotate out) methods, and the loader hard-fails on a non-fixed artifact frame.

**Validation hygiene.** Scalers — including the residual target scalers — are fit
on **training rows only** (recorded as `fit_scope="train_only"` in `scaler.json`).
Splits go beyond random interpolation: `spatial_block` (spatial generalization)
and `ood_low_altitude`/`ood_high_altitude` (altitude extrapolation), with
geometry recorded in `split_manifest.json`. The benchmark pipeline adds a
`--paper-safe` mode that forbids synthetic/legacy/mismatch/extrapolation settings,
and a strengthened scenario-metadata validator. See
[ST_LRPS_VALIDATION_HYGIENE.md](ST_LRPS_VALIDATION_HYGIENE.md) for the full
reference, the validation/ablation suites, and the recommended paper workflow.
