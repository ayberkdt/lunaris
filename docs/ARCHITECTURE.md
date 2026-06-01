# Architecture

This document is the canonical reference for the internal architecture of the
`lunaris` framework. The [README](../README.md) gives the high-level overview and
usage; this file explains how the layers fit together and where to make changes.

ST-LRPS (Sobolev-Trained Lunar Residual Potential Surrogate) is the named
surrogate-gravity family that ships inside the framework as
`lunaris.surrogate.st_lrps`.

## Layered design

The propagation framework is organized into four strict layers. A layer never
imports from a layer above it.

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
- `surrogate_gravity.py`, `gravity_adapter.py` — surrogate force-model adapters.

Physics models never import from `core/`.

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
- `ui/app.py` + `ui/widgets/` — PySide6 desktop UI (mission simulator).

### Support packages
Alongside the four layers:
- `lunaris.loaders` — dependency-light data loading (gravity coefficient files,
  SPICE kernels, topography/albedo grids) consumed by layers 2–3.
- `lunaris.cli` — console entry points (`lunaris`, `lunaris-mc`, …) and shared
  CLI argument helpers; wires user input into the `core` configuration.
- `lunaris.surrogate.st_lrps` — the ST-LRPS surrogate-gravity family
  (see [ST-LRPS surrogate](#st-lrps-surrogate)).

## Configuration (SSOT)

`lunaris.core.config` is the single source of truth. All configuration flows
through the frozen `SimConfig` dataclass; never pass ad-hoc kwargs.

```python
from lunaris.core.config import load_default_config

cfg = load_default_config()
# CLI overrides are applied in lunaris.cli.main via apply_args_to_config(cfg, args)
cfg.validate()  # cross-field consistency checks
```

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
    use_gpu=True,        # requires CUDA + numba.cuda; falls back to CPU with a warning
    gpu_sh_degree=10,    # SH degree evaluated per GPU thread (0 = point mass)
    output_format="hdf5",
    output_path="outputs/monte_carlo/run.h5",
)
result = MonteCarloEngine(sim_cfg, mc_cfg).run()      # MCRunResult
stats = compute_mc_statistics(result)
figs = plot_mc_report(result, stats, output_path="outputs/monte_carlo/report.pdf")
```

Reload a saved run with `from lunaris.core.monte_carlo_engine import load_mc_result`.

### GPU kernel constraints
- The CUDA kernel workspace uses compile-time fixed `(26×26)` arrays, supporting
  SH degree ≤ 24. `gpu_sh_degree > 24` raises `ValueError`; use the CPU path for
  higher-degree fields.
- The GPU path does not support albedo, thermal IR, or solid tides (use the CPU
  path for those models).
- CUDA requires `numba` plus a CUDA-capable GPU; the engine falls back to CPU
  (emitting a `RuntimeWarning` and recording a `fallback_reason`) when CUDA is
  unavailable.

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
| `runtime/` | propagator-facing ST-LRPS force-model API |
| `shared/` | scaling utilities and target/derivative contracts |
| `ui/` | ST-LRPS Studio desktop UI |

Data generation pulls strictly from the lunar GFC model in
`data/dataset_parameters.py` and natively computes residual fields:
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

**Runtime.** The propagator-facing API is `runtime/force_model.py`. The only
implemented runtime is `potential_autograd` (`SurrogateForceModel`): it evaluates
the learned scalar potential and differentiates it with autograd to obtain the
residual acceleration, which is added to the SH(`degree_min`) baseline. The
distilled direct-force runtime (`force_direct` / `DirectForceRuntime`) is a
reserved placeholder and raises `NotImplementedError`; `load_surrogate_force_model`
rejects any artifact whose `runtime_model_kind` is not `potential_autograd`.
Because each evaluation is a network forward pass plus an autograd pass, the
runtime is most efficient in batched / GPU configurations.
