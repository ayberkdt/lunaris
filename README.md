# ST-LRPS

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](#testing)

A lunar residual-potential surrogate gravity framework with propagation, validation, Monte Carlo, and UI tooling.

---

## Overview

ST-LRPS propagates spacecraft trajectories in lunar orbit with scientific accuracy. The physics engine assembles a Numba-JIT-compiled right-hand-side closure from independent force models, then drives a variable-step DOP853 integrator with event detection. A separate GPU-accelerated path runs large Monte Carlo ensembles in parallel using fixed-step CUDA RK4, enabling statistical impact probability and covariance analysis within practical wall-clock times.

The **ST-LRPS** (Sobolev-Trained Lunar Residual Potential Surrogate) component replaces classical spherical-harmonic evaluation with a neural network that learns the residual scalar potential $\Delta U$ above a low-degree SH baseline. Gradients are obtained by automatic differentiation, so the acceleration is physically consistent without storing massive coefficient arrays.

---

## Features

**Dynamics**
- Spherical harmonics gravity up to degree/order 1800 (GRAIL/GRGM coefficients)
- ST-LRPS neural gravity surrogate — learned $\Delta U$ via autograd, CPU and GPU inference
- Third-body perturbations: Sun and Earth (differential formulation)
- Solar radiation pressure with penumbra/umbra shadow detection
- Lunar albedo radiation pressure (surface-grid backed)
- First-order post-Newtonian (Schwarzschild 1PN) relativistic correction
- Adaptive SH degree control: finer harmonics near periselene, coarser at apoapsis

**Integration**
- Primary integrator: DOP853 (8th-order Runge-Kutta), adaptive step
- Event detection: terrain-aware impact detection, apsis crossing, SOI transition
- Step-size bounded by Nyquist criterion on the active gravity-field degree

**Monte Carlo**
- GPU path: CUDA RK4 with per-thread SH workspace (degree ≤ 24), Sun/Earth third-body, SRP, 1PN
- CPU path: full-fidelity per-sample propagation reusing all active force models
- Gaussian initial-state and spacecraft-property perturbations
- HDF5/NPZ streaming output; impact probability with Wilson confidence intervals
- Automatic GPU→CPU fallback for unsupported physics (ST-LRPS, albedo)

**Ephemeris**
- SPICE kernel integration via SpiceyPy
- Pre-tabulated Sun/Earth position vectors and Moon-fixed attitude quaternions

**Analysis & UI**
- Post-processing: osculating elements, RAAN/argument-of-periapsis drift, energy invariants
- Matplotlib output: altitude, ground track, 3-D orbit, phase space, acceleration budget
- PDF mission report via ReportLab
- PySide6 desktop application with live telemetry, page-based configuration, and embedded analysis panels

---

## Architecture

The codebase follows a strict multi-layer dependency rule: each layer imports only from layers below it.

### Layer 1: Core Definitions
- `common/`: Physical constants (SI Single Source of Truth), pure math utilities, and configuration dataclasses. `config.py` acts as the global state and configuration SSOT.

### Layer 2: Force Models
- `models/`: High-performance Numba-JIT force kernels. Each perturbation (spherical harmonics, solar radiation pressure, third body, relativity, surface albedo) is encapsulated in its own module and exposes an allocation-free function.
- `st_lrps/`: ST-LRPS training, evaluation, and runtime package. Neural network surrogate models (ST-LRPS), dataset spatial point cloud generators, Sobolev loss formulations, scaling, and training pipelines.

### Layer 3: Dynamics & Propagation
- `core/dynamics.py`: The equations of motion (RHS) builder. It inspects requested flags, extracts Numba-friendly arrays from high-level models, and constructs a compiled closure for the numerical solver.
- `core/propagator.py`: The single authoritative entry point for time integration, wrapping SciPy `solve_ivp` and fixed-step solvers, implementing event geometric bounds, step-size limiters (Nyquist), and chunking.
- `core/monte_carlo_engine.py` & `core/mc_backend_policy.py`: Orchestrates multi-sample runs. It evaluates requested perturbation flags and automatically dispatches to the GPU (CUDA) or falls back to CPU multiprocessors depending on compatibility.

### Layer 4: Application & Analysis
- `analysis/`: Reusable post-processing, generic metrics extraction, Monte Carlo statistics, reporting, and formatting utilities.
- `validation/` & `visualization/`: Offline benchmark scripts and orbital animation.
- `ui_parts/` & `cli/`: PySide6 frontend logic and CLI parsing.

All configuration flows through the frozen `SimConfig` dataclass.

---

## Installation

**Runtime dependencies**

```
python >= 3.9
numpy
scipy
numba
torch          # ST-LRPS inference
spiceypy       # SPICE ephemeris
PySide6        # desktop UI
h5py           # HDF5 output
matplotlib
reportlab
```

**Optional**

```
numba[cuda]    # GPU Monte Carlo (requires CUDA toolkit)
rasterio       # LDEM terrain-based impact detection
```

**Steps**

```bash
git clone https://github.com/ayberkdt/ST_LRPS.git
cd ST_LRPS
pip install -r requirements.txt
```

SPICE kernels and gravity coefficient files are **not** bundled. Acquire them separately and place them under `data/` as described in the next section.

---

## Data Requirements

| Directory | Contents | Source |
|-----------|----------|--------|
| `data/ephemeris_models/` | Leap-second `.tls`, planetary `.bsp`, constants `.tpc`, lunar orientation `.bpc` | NAIF/SPICE |
| `data/gravity_models/` | SH coefficient file, e.g. `jggrx_1800f_sha.tab` | NASA PDS / GRAIL |
| `data/topography_models/` | Lunar DEM rasters (optional, for terrain-aware impact) | LOLA/LDEM |
| `data/albedo_models/` | Surface albedo grids (optional) | — |

---

## Quick Start

**Single propagation (CLI)**

```bash
python main.py \
    --start-date 2025-06-01T00:00:00 \
    --days 7 \
    --hp-km 100 --ha-km 500 \
    --inc-deg 90 \
    --enable-sh true --enable-srp true \
    --out-dir output/polar_run
```

**Monte Carlo (CLI)**

```bash
python mc_runner.py \
    --start-date 2025-06-01T00:00:00 \
    --days 30 \
    --alt-km 100 --inc-deg 90 \
    --n-samples 500 \
    --sigma-r-m 500 --sigma-v-m-s 0.5 \
    --use-gpu on \
    --mc-output-path results/mc_run.h5
```

**Desktop UI**

```bash
python ui.py
```

**Python API**

```python
from config import load_default_config
from main import apply_args_to_config
from core.propagator import propagate

cfg = load_default_config()
# adjust cfg fields or call apply_args_to_config(cfg, args)
result = propagate(dynamics=..., y0=..., cfg=cfg.propagator, time_cfg=cfg.time)
```

---

## ST-LRPS: Neural Gravity Surrogate

ST-LRPS learns the residual scalar gravitational potential above a baseline SH model:

```
a_total = a_SH(degree_min) + ∇ΔU_STLRPS(x)
```

The network is trained on spatial point clouds generated by `surrogate_gravity_model/spatial_cloud_generator.py` using the full GFC model. Training uses a Sobolev loss that penalises both potential and gradient error, which makes the learned acceleration physically consistent by construction.

**Generate a training dataset**

```bash
python -m st_lrps.spatial_cloud_generator \
    --degree-min 10 --degree-max 50 \
    --n-samples 250000 \
    --alt-range 30 120 \
    --format h5
```

**Train**

```bash
python -m st_lrps.st_lrps_train \
    --data <dataset.h5> \
    --out runs/st_lrps_train_<timestamp>
```

**Use in propagation**

```bash
python main.py \
    --gravity-backend st_lrps \
    --surrogate-gravity-model-dir runs/st_lrps_train_<timestamp>/ \
    ...
```

Trained run directories must contain `config.json` (including `degree_min`/`degree_max`), `scaler.json`, and `checkpoints/ckpt_best.pt`. If only `ckpt_last.pt` is present the model loads with a warning.

---

## Testing

Focused ST-LRPS contract checks:

```bash
python -m pytest tests/test_surrogate_training_contracts.py -q
python -m pytest tests/test_surrogate_architecture_upgrades.py -q
```

```bash
pytest tests/
```

Key test modules:

| Module | Coverage |
|--------|----------|
| `test_surrogate_gravity.py` | ST-LRPS metadata contract, ckpt fallback, propagator compatibility |
| `test_mc_gpu_policy.py` | Backend selection, GPU→CPU fallback rules |
| `test_dynamics.py` | Force model assembly, RHS evaluation |
| `test_spherical_harmonics.py` | SH acceleration kernels |
| `test_surrogate_training_contracts.py` | Training pipeline contracts |
| `test_surrogate_architecture_upgrades.py` | ST-LRPS architecture cleanup contracts |

---

## Project Structure

```text
common/                  Constants, type definitions, unified configuration dataclasses
models/                  Force kernels: gravity, SRP, third-body, albedo, relativity
core/                    Dynamics engine, propagator, Monte Carlo engine and propagators
analysis/                Post-processing, MC statistics, formatting, and report generation
  ├── reporting/         Report management, styling, and Matplotlib plotting
  └── monte_carlo/       Monte Carlo statistics and ensemble plotting
loaders/                 Asset loaders: gravity files, SPICE, surface grids
validation/              Validation and benchmark scripts
  └── gravity/           Gravity model comparison and benchmarks
visualization/           Interactive visualization and animation tools
st_lrps/                 ST-LRPS data generation, training, evaluation, inference API
ui_parts/                PySide6 page widgets and helpers
cli/                     Command-line interface helpers
tests/                   Unit and integration tests for all packages
data/                    Local storage for SPICE kernels, topography, and SH coefficients
results/                 Default output directory for single-run reports and telemetry
mc_results/              Default output directory for Monte Carlo run outputs
config.py                SimConfig Single Source of Truth
main.py                  Single-run CLI entry point
mc_runner.py             Monte Carlo CLI entry point
ui.py                    Desktop application entry point
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
