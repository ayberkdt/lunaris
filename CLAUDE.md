# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run CLI simulation
python main.py --start-date 2025-01-01T00:00:00 --days 1 --alt-km 100

# Common CLI flags
--hp-km 80 --ha-km 200          # Periselene/aposelene altitudes
--inc-deg 90                    # Inclination (polar orbit)
--enable-sh --enable-srp        # Enable physics perturbations
--out-dir mission_results/run1  # Output directory

# Run GUI
python ui.py

# Run tests
pytest tests/
pytest tests/test_dynamics.py -v
```

## Architecture

The codebase is a lunar orbit propagation framework organized in four strict layers:

**Layer 1 — `common/`**: Dependency-light shared layer. `constants.py` is the single source of truth for all physical constants (SI units). `type_defs.py` holds all configuration dataclasses. No layer imports from above it.

**Layer 2 — `models/`**: Numba-JIT-compiled physics kernels. Each file is one force model: `spherical_harmonics.py` (gravity), `third_body_effects.py` (Sun/Earth), `solar_effects.py` (SRP), `surface_effects.py` (albedo/thermal), `relativity_effects.py` (1PN). `ephemeris.py` wraps SPICE kernels. Models never import from `core/`.

**Layer 3 — `core/`**: Numerical engine. `dynamics.py` assembles a Numba-compiled RHS closure by wiring together the active physics models. `propagator.py` calls `scipy.integrate.solve_ivp()` with event detection from `events.py`. The propagator returns a `PropagationResult(t, y, events, status)`.

**Layer 4 — `analysis/` + desktop UI**: Post-processing (`analysis.postprocess`), report plotting/styling/management (`analysis.reporting.*`), Monte Carlo statistics/plotting (`analysis.monte_carlo.*`), and the PySide6 GUI (`ui.py` + `ui_parts/`).

### Configuration (SSOT)

`config.py` is the single source of truth. `load_default_config()` returns a frozen `SimConfig` dataclass. CLI overrides are applied via `apply_args_to_config()` in `main.py`. All config paths go through `SimConfig`, never ad-hoc kwargs.

```python
cfg = load_default_config()
cfg = apply_args_to_config(cfg, args)
cfg.validate()  # cross-field consistency checks
```

Key sub-configs: `GravityConfig`, `SpiceBuildConfig`, `InitialState`, `PerturbationFlags`, `SpacecraftProps`, `PropagatorConfig`, `TimeConfig`.

### Data Flow

```
CLI/GUI → config.py (SimConfig)
        → loaders/ (gravity model, SPICE kernels, surface grids)
        → core/dynamics.py (build Numba RHS closure)
        → core/propagator.py (solve_ivp → PropagationResult)
        → analysis/postprocess.py (orbital elements, metrics)
        → analysis.reporting.plotting + analysis.reporting.manager (PNG/PDF output)
```

### Physics Perturbation Flags (`PerturbationFlags`)

All flags default to `False` except `enable_sh=True`. Enabling a flag requires the corresponding config section to be non-None (e.g., `enable_srp=True` requires `cfg.srp` to be set).

| Flag | Model |
|------|-------|
| `enable_sh` | Spherical harmonics gravity (default degree 100, up to 1800) |
| `enable_3rd_body_sun/earth` | Third-body perturbations |
| `enable_srp` | Solar radiation pressure |
| `enable_albedo` | Reflected solar from lunar surface |
| `enable_thermal` | Lunar thermal emission |
| `enable_tides_k2/k3` | Tidal dissipation |
| `enable_relativity_1pn` | 1st-order post-Newtonian |

### External Data (`data/` directory)

Mandatory at runtime:
- `data/ephemeris_models/` — SPICE kernels (leap seconds `.tls`, planetary ephemerides `.bsp`, constants `.tpc`, lunar orientation `.bpc`)
- `data/gravity_models/` — Spherical harmonic coefficients (e.g., `jggrx_1800f_sha.tab`)

Optional (needed only when corresponding flags are enabled):
- `data/topography_models/` — Lunar DEM rasters
- `data/albedo_models/` — Surface albedo grids
- `data/thermal_models/` — Thermal property grids

## Monte Carlo Infrastructure

### Quick start

```python
from config import load_default_config
from common.montecarlo_defs import MonteCarloConfig, StateUncertainty
from core.monte_carlo_engine import MonteCarloEngine
from analysis.monte_carlo.statistics import compute_mc_statistics
from analysis.monte_carlo.plotting import plot_mc_report

sim_cfg = load_default_config()
mc_cfg  = MonteCarloConfig(
    n_samples=500,
    state=StateUncertainty(sigma_r_m=500.0, sigma_v_m_s=0.5),
    use_gpu=True,          # requires CUDA + numba.cuda
    gpu_sh_degree=10,      # SH degree evaluated per-thread on GPU (0 = PM only)
    output_format="hdf5",
    output_path="mc_results/run.h5",
)
result   = MonteCarloEngine(sim_cfg, mc_cfg).run()   # MCRunResult
mc_stats = compute_mc_statistics(result)
figs     = plot_mc_report(result, mc_stats, output_path="mc_results/report.pdf")
```

### New modules

| Module | Purpose |
|--------|---------|
| `common/montecarlo_defs.py` | `MonteCarloConfig`, `StateUncertainty`, `SpacecraftUncertainty`, `MCRunResult` |
| `core/mc_propagator.py` | `GPUBatchPropagator` (CUDA RK4, per-thread SH workspace), `CPUBatchPropagator` (ProcessPoolExecutor) |
| `core/monte_carlo_engine.py` | `MonteCarloEngine.run()` — sample generation, backend dispatch, HDF5/NPZ streaming output |
| `analysis/mc_analysis.py` | `compute_mc_statistics()` → `MCStatistics` (covariance, ellipsoids, impact probability, OE dispersion) |
| `analysis/mc_plotting.py` | Matplotlib figures: altitude envelopes, 3-D covariance tubes, impact map (Mollweide), OE dispersion |

### GPU kernel constraints
- CUDA kernel workspace uses compile-time fixed arrays `(26×26)` supporting SH degree ≤ 24.
- `gpu_sh_degree > 24` → `ValueError`; use CPU path for higher-fidelity SH.
- GPU path does **not** support albedo / thermal / tides — these are CPU-only.
- CUDA requires `numba` + a CUDA-capable GPU; graceful fallback to CPU if unavailable.

### Output format
- HDF5 (default): streaming writes via `h5py`; extendable datasets for `t` and `Y`.
- NPZ: accumulates in RAM, writes on completion; suitable for small N.
- Saved arrays: `t (T,)`, `Y (T, N, 6)`, `sc_samples (N, 4)`, `impact_flags (N,)`, `t_impact (N,)`.
- Reload with `from core.monte_carlo_engine import load_mc_result`.

### Performance Notes

- All inner-loop physics use `@njit(cache=True)` or `@njit(parallel=True)` — avoid Python-level loops inside physics kernels.
- Ephemeris data is pre-tabulated at startup (not queried per integration step).
- Spherical harmonic evaluation uses a reusable `SHWorkspace` to avoid heap allocation in the hot path.
- Default integrator is DOP853 (8th-order Runge-Kutta). Step-size is bounded via Nyquist criterion on the gravity field degree.

## ST-LRPS Gravity Surrogate

The `st_lrps` directory contains a standalone pipeline for training Neural Networks (e.g., Sobolev-Trained Lunar Residual Potential Surrogates) to approximate the lunar gravity field. 

### Data Generation (Residual Network Setup)

We have removed legacy/Earth assumptions. The generator now strictly pulls from the lunar GFC model defined in `dataset_parameters.py` and natively calculates **residual** gravity fields.

Because the surrogate operates as a residual network over a base low-degree gravity model (usually 10x10), the data generator directly supports configurable `degree_min` and `degree_max` parameters:

- `degree_min`: The maximum degree of the base analytical model (e.g., 10). The network learns the *difference* starting from `degree_min + 1`. If set to `-1`, the full field (including point mass) is evaluated.
- `degree_max`: The target high-fidelity resolution (e.g., 50). 

**Example generation (50x50 target, residual above 10x10):**

```bash
python -m st_lrps.data.spatial_cloud_generator \
    --degree-max 50 \
    --degree-min 10 \
    --n-samples 250000 \
    --alt-range 30 120 \
    --format h5 \
    --workers 8
```

You can also use predefined configurations in `spatial_cloud_parameters.py` via the `--preset` flag (e.g., `--preset moon_llo_30_120km_deg10to50`).

## Workflow Orchestration

### 1. Plan Node Default
- Enter plan mode for ANY non-trivial task (3+ steps or architectural decisions)
- If something goes sideways, STOP and re-plan immediately - don't keep pushing
- Use plan mode for verification steps, not just building
- Write detailed specs upfront to reduce ambiguity

### 2. Subagent Strategy
- Use subagents liberally to keep main context window clean
- Offload research, exploration, and parallel analysis to subagents
- For complex problems, throw more compute at it via subagents
- One tack per subagent for focused execution

### 3. Self-Improvement Loop
- After ANY correction from the user: update `tasks/lessons.md` with the pattern
- Write rules for yourself that prevent the same mistake
- Ruthlessly iterate on these lessons until mistake rate drops
- Review lessons at session start for relevant project

### 4. Verification Before Done
- Never mark a task complete without proving it works
- Diff behavior between main and your changes when relevant
- Ask yourself: "Would a staff engineer approve this?"
- Run tests, check logs, demonstrate correctness

### 5. Demand Elegance (Balanced)
- For non-trivial changes: pause and ask "is there a more elegant way?"
- If a fix feels hacky: "Knowing everything I know now, implement the elegant solution"
- Skip this for simple, obvious fixes - don't over-engineer
- Challenge your own work before presenting it

### 6. Autonomous Bug Fixing
- When given a bug report: just fix it. Don't ask for hand-holding
- Point at logs, errors, failing tests - then resolve them
- Zero context switching required from the user
- Go fix failing CI tests without being told how

## Task Management

1. **Plan First**: Write plan to `tasks/todo.md` with checkable items
2. **Verify Plan**: Check in before starting implementation
3. **Track Progress**: Mark items complete as you go
4. **Explain Changes**: High-level summary at each step
5. **Document Results**: Add review section to `tasks/todo.md`
6. **Capture Lessons**: Update `tasks/lessons.md` after corrections

## Core Principles

- **Simplicity First**: Make every change as simple as possible. Impact minimal code.
- **No Laziness**: Find root causes. No temporary fixes. Senior developer standards.
- **Minimat Impact**: Changes should only touch what's necessary. Avoid introducing bugs.
