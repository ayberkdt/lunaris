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
- `surface_effects.py` — albedo / thermal.
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
- `visualization/` — standalone orbit-animation and surface-explorer tools.
- `ui/app.py` + `ui/widgets/` — PySide6 desktop UI (mission simulator).

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

## Perturbation flags

`PerturbationFlags` (in `lunaris.common.type_defs`) all default to `False` except
`enable_sh=True`. Enabling a flag requires the corresponding config section to be
non-`None` (e.g. `enable_srp=True` requires `cfg.srp`).

| Flag | Model |
|------|-------|
| `enable_sh` | Spherical-harmonics gravity (default degree 100, up to 1800) |
| `enable_3rd_body_sun` / `enable_3rd_body_earth` | Third-body perturbations |
| `enable_earth_j2` | Earth oblateness (differential) |
| `enable_srp` | Solar radiation pressure |
| `enable_albedo` | Reflected solar from the lunar surface |
| `enable_thermal` | Lunar thermal emission |
| `enable_tides_k2` / `enable_tides_k3` | Tidal dissipation |
| `enable_relativity_1pn` | First-order post-Newtonian |

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
- The GPU path does not support albedo / thermal / tides — these are CPU-only.
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
