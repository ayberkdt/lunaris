# Lunaris

**Lunar orbit propagation and gravity-modeling framework.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Status: Alpha](https://img.shields.io/badge/status-alpha-orange.svg)](https://pypi.org/classifiers/)

Lunaris is a Python framework for lunar-orbit propagation and gravity modeling. It bundles spherical-harmonic lunar gravity, configurable physical force models, orbit propagation, Monte Carlo analysis, validation harnesses, visualization tools, and a PySide6 desktop UI.

It also ships **ST-LRPS** (Sobolev-Trained Lunar Residual Potential Surrogate) — a neural surrogate-gravity model under `lunaris.surrogate.st_lrps` that learns a residual scalar potential above a lower-degree spherical-harmonic baseline, with its own training, evaluation, and Studio UI.

> **Project status.** Lunaris is an **alpha-stage research prototype**
> (`Development Status :: 3 - Alpha`, version 0.1.0). It is intended for research
> and experimentation: APIs, trained artifacts, and reported benchmark numbers
> may change between versions, and some experimental runtime paths are still
> reserved for future work.

## Overview

Lunaris supports lunar-orbit propagation with configurable physical force models (spherical-harmonic gravity, third-body, Earth J2, solar radiation pressure, lunar albedo, thermal IR, solid tides, relativity — see [Force models](#force-models)), Monte Carlo workflows, validation harnesses, report generation, and PySide6-based desktop workflows. When the ST-LRPS surrogate is enabled, runtime acceleration is obtained from the learned potential gradient and combined with the lower-degree spherical-harmonic baseline.

Accuracy, runtime, and stability depend on the selected data, force-model configuration, trained artifacts, and validation scenario. Treat validation outputs as run-specific evidence rather than a blanket guarantee.

### Force models

Implemented and wired into the propagator (`lunaris.core.dynamics`):

- Spherical-harmonic lunar gravity (and the ST-LRPS surrogate-gravity model)
- Third-body perturbations (Sun, Earth)
- Earth oblateness (differential J2)
- Solar radiation pressure (with eclipse handling)
- Lunar albedo (reflected-solar) surface radiation
- Lunar thermal IR radiation pressure
- Elastic lunar solid-body tides (`k2`, optional explicit `k3`; Earth and/or Sun raised)
- First-order post-Newtonian relativity

Solid tides use `lunaris.physics.solid_tides`: the Moon-fixed disturbing
potential is evaluated with configurable Love numbers and differentiated
analytically, then the acceleration is rotated back to the inertial integration
frame. The default `SolidTideConfig.k2=0.02416` follows the GRAIL/LRO monthly
lunar `k2` value reported by Williams & Boggs (2015) in
[NASA PGDA product 96](https://pgda.gsfc.nasa.gov/products/96);
`k3` has no default and must be set explicitly for `--tides-kind k3`.
Supported tide-raising bodies are `earth`, `sun`, or both. This is an elastic
instantaneous solid-body model only: no time lag/dissipation, ocean tide, or
thermal tide is included.

CLI example:

```bash
lunaris --enable-tides on --tides-kind k2 --tide-bodies earth,sun
lunaris --enable-tides on --tides-kind k3 --tide-k3 0.01 --tide-bodies earth
```

Thermal IR uses `lunaris.physics.thermal_ir`: the lunar surface is discretized
into Lambertian facets and each visible facet contributes radiation pressure in
the spacecraft direction. Supported modes are `constant_temperature`,
`equilibrium_temperature` (instantaneous solar incidence, no thermal inertia),
and `temperature_grid` (caller/provider-supplied facet temperatures).

```bash
lunaris --enable-thermal on --thermal-mode constant_temperature --thermal-temperature-k 250
lunaris --enable-thermal on --thermal-mode equilibrium_temperature --thermal-night-temperature-k 100
```

Lunar albedo uses `lunaris.physics.lunar_albedo`: a non-gravitational
*reflected-solar* radiation pressure (sunlight reflected off the lunar surface),
not a gravity term. The Moon is discretized into Lambertian facets, and each
facet that is both sunlit and visible to the spacecraft reflects sunlight toward
it; the summed acceleration is rotated back to the inertial frame. Per-facet
albedo is set by `--albedo-mode`: `constant_albedo` (provider-free, uses
`--albedo-const`), `albedo_grid`, or `scaled_dn_grid` (the latter two sample a
LOLA-style grid supplied via `--albedo-root`). It uses a dedicated
`--albedo-pressure-coefficient` (C_R_albedo), not the SRP `cr`. The legacy
cannonball model stays available via `--albedo-model simple`. The facet model is
Lambertian only: no non-Lambertian BRDF, wavelength dependence, surface
roughness, terrain self-shadowing beyond the incidence/visibility cutoffs,
photometric phase functions, multiple scattering, or local topography.

```bash
lunaris --enable-albedo on --albedo-const 0.12
lunaris --enable-albedo on --albedo-mode scaled_dn_grid --albedo-root data/albedo_models
```

## Documentation

| Document | Contents |
|----------|----------|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Layered design, data flow, configuration model, perturbation flags, Monte Carlo internals, ST-LRPS surrogate |
| [docs/BENCHMARK_RESULTS.md](docs/BENCHMARK_RESULTS.md) | Full gravity-model benchmark tables and reproduction steps |
| [docs/REPRODUCIBLE_BENCHMARKS.md](docs/REPRODUCIBLE_BENCHMARKS.md) | Config-driven benchmark runs, provenance manifests, validation reports, and CI smoke mode |
| [docs/DATASET_PIPELINE.md](docs/DATASET_PIPELINE.md) | ST-LRPS dataset contract, validation, quality reports, split manifests, and strict training ingestion |
| [docs/CONFIG_AND_ARTIFACT_CONTRACTS.md](docs/CONFIG_AND_ARTIFACT_CONTRACTS.md) | ST-LRPS dataset, training, checkpoint, runtime, and benchmark contract rules |
| [docs/HPC.md](docs/HPC.md) | Cluster/headless install, Conda environment, Slurm templates |
| [docs/profiling.md](docs/profiling.md) | ST-LRPS runtime profiling and timing interpretation |
| [validation/README.md](validation/README.md) | Independent physics/orbit/gravity validation harness |

## Repository Architecture

The repository uses a `src/` package layout. The propagation framework is
organized into **four strict layers** — a layer never imports from a layer above
it — with a few dependency-light support packages alongside them. ST-LRPS remains
a named surrogate family under `lunaris.surrogate.st_lrps`. The map below is a
quick orientation; see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the
canonical layered design, data flow, configuration model, perturbation flags, and
Monte Carlo internals.

```text
src/lunaris/
  common/          [layer 1] shared constants, config dataclasses, math/time helpers
  physics/         [layer 2] Numba force-model kernels, ephemeris (SPICE), gravity adapters
  core/            [layer 3] config (SimConfig SSOT), dynamics RHS, propagator, events, Monte Carlo
  analysis/        [layer 4] post-processing, reports, Monte Carlo analysis
  visualization/   [layer 4] standalone visualization tools
  ui/              [layer 4] Lunar Orbit Simulator desktop UI (PySide6)
    widgets/                 desktop UI page components
  loaders/         (support) gravity / topography / ephemeris / data loading for layers 2–3
  cli/             (support) console entry points and shared CLI argument helpers
  surrogate/
    st_lrps/       Sobolev-Trained Lunar Residual Potential Surrogate family
      data/        dataset definitions, spatial cloud generation, dataset loading
      training/    ST-LRPS training config, CLI, engine, losses, metrics
      networks/    neural network architecture definitions
      artifacts/   run layout, checkpoints, manifests, artifact validation
      evaluation/  trained-model evaluation, ablation, and orbit-level benchmark CLIs
      runtime/     propagator-facing ST-LRPS force model API
      shared/      shared scaling utilities
      ui/          ST-LRPS Studio desktop UI
validation/        independent physics/orbit/gravity validation docs and schemas
tests/             unit and regression tests
data/              local input data such as SPICE kernels, gravity, topography
hpc/               example Slurm templates for cluster use
```

Console entry points (installed via `pip install -e .`):

```text
lunaris           single-run propagation CLI
lunaris-mc        Monte Carlo runner
lunaris-launcher  welcome hub (picks a workspace; optional 3D Moon preview)
lunaris-ui        mission desktop UI
lunaris-studio    ST-LRPS Studio UI
lunaris-train     ST-LRPS training CLI
lunaris-eval      ST-LRPS evaluation CLI
lunaris-benchmark ST-LRPS orbit-level gravity benchmark / validation CLI
lunaris-data      external-data download / verify CLI
```

### Desktop launcher & 3D web preview

`lunaris-launcher` is the top-level welcome hub. It presents two workspaces —
**Lunar Propagation** (`lunaris-ui`) and **ST-LRPS Studio** (`lunaris-studio`) —
as glassmorphic cards, and opens the chosen one in its own window. Each workspace
is imported lazily, so launching one never loads the other's dependencies; both
`lunaris-ui` and `lunaris-studio` also remain usable directly.

Behind the cards, the launcher can show an optional, **offline** interactive 3D
Moon (a Sobolev/visual and gravity-anomaly texture toggle, plus a small demo
orbit). The visual is a Next.js / Three.js scene under
`src/lunaris/ui/web`, statically exported and served from a local
loopback HTTP server — no internet and no Node runtime are required at app run
time. Build it once:

```bash
cd src/lunaris/ui/web
npm install
npm run build      # writes ./out (embed route at out/embed/index.html)
```

The preview is entirely optional. If the build is missing, QtWebEngine is not
installed, or the GPU lacks WebGL, the launcher falls back to a dark background
and still opens normally — the 3D scene never blocks the app. Point the launcher
at a custom build with the `LUNARIS_WEB_EMBED_DIR` environment variable. (The
satellite path in the preview is a *demo orbit*, not solver output.)

## Installation

Install the package in editable mode from the repository root. This wires up the
console entry points (`lunaris`, `lunaris-ui`, …) and lets code changes take
effect without reinstalling:

```bash
python -m pip install -e .            # core dependencies only
python -m pip install -e ".[all]"     # core + ML + UI + reports + dev extras
```

Optional dependency groups are declared in `pyproject.toml`: `core`, `ml`, `hpc`,
`ui`, `reports`, `dev`, and `all`. A pinned `requirements.txt` is also provided
for environments that prefer a flat dependency list.

**For HPC and cluster deployments**, use the headless `.[hpc]` extra and the Slurm templates under `hpc/`; keep GUI dependencies off compute nodes. The primary cluster workflows are ST-LRPS dataset generation, training, evaluation, and large orbit-level validation runs. See the [HPC and Cluster Deployment Guide](docs/HPC.md) for details.

Large mission data files are not bundled. Place local SPICE kernels, gravity coefficient files, topography grids, and albedo grids under `data/` or another local path configured at runtime.

Common data locations:

| Directory | Contents |
|-----------|----------|
| `data/ephemeris_models/` | SPICE leap-second, planetary, constants, and lunar orientation kernels |
| `data/gravity_models/` | Lunar spherical-harmonic coefficient files |
| `data/topography_models/` | LOLA/LDEM topography rasters |
| `data/albedo_models/` | Optional lunar albedo grids |
| `data/thermal_models/` | Optional thermal temperature/property grids |

### Acquiring external data

Large files (gravity coefficients, SPICE kernels, topography, albedo) are not bundled. Use the headless `lunaris-data` tool to fetch and verify them into `LUNARIS_DATA_DIR` (or the repository `data/` folder):

```bash
lunaris-data list
lunaris-data download --group ephemeris
lunaris-data verify
lunaris-data inspect --data outputs/datasets/cloud.h5
lunaris-data validate --data outputs/datasets/cloud.h5 --out outputs/dataset_reports/cloud
lunaris-data report --data outputs/datasets/cloud.h5 --out outputs/dataset_reports/cloud
```

The catalogue is `data/data_sources.json`; entries without an official pinned URL print manual-placement instructions. See the [HPC and Cluster Deployment Guide](docs/HPC.md) for the cluster data workflow.

## Quickstart

These checks do not require private local datasets.

```bash
python -m pip install -e ".[hpc]"
python -c "import lunaris; print(lunaris.__version__)"
lunaris-train --help
lunaris-eval --help
lunaris-benchmark --help
lunaris-mc --help
python -m lunaris.surrogate.st_lrps.training.cli --help
python -m lunaris.surrogate.st_lrps.evaluation.cli --help
python -m lunaris.surrogate.st_lrps.evaluation.compare_gravity_models --help
python -m lunaris.visualization.surface_explorer --help
```

Data-dependent examples such as full propagation, ST-LRPS training, gravity validation runs, and topography plots require local gravity, SPICE, or LOLA files.

## ST-LRPS Commands

Spatial cloud generation:

```bash
python -m lunaris.surrogate.st_lrps.data.spatial_cloud_generator --help
```

Training:

```bash
python -m lunaris.surrogate.st_lrps.training.cli --help
```

Evaluation:

```bash
python -m lunaris.surrogate.st_lrps.evaluation.cli --help
```

Ablation:

```bash
python -m lunaris.surrogate.st_lrps.evaluation.ablation --help
```

Runtime import example:

```python
from lunaris.surrogate.st_lrps.runtime.force_model import load_surrogate_force_model
```

At runtime the surrogate evaluates the learned **scalar residual potential** and
obtains the residual acceleration by autograd differentiation of that potential
(`runtime_model_kind="potential_autograd"`), then adds it to the lower-degree
spherical-harmonic baseline. This is currently the only implemented runtime; the
distilled direct-force path (`runtime_model_kind="force_direct"` /
`DirectForceRuntime`) is reserved for future work and raises `NotImplementedError`.
Because each acceleration evaluation is a network forward pass plus an autograd
pass, the runtime speedups reported below are obtained in batched / GPU
configurations — see [docs/BENCHMARK_RESULTS.md](docs/BENCHMARK_RESULTS.md) for
the exact settings.

Model target semantics are recorded explicitly through versioned
`artifact_contract` and `dataset_contract` blocks in new configs/checkpoints.
The contract distinguishes residual labels from full-field labels, records the
baseline degree/kind, target degree, altitude envelope, scaler contract, input
encoding, architecture signature, and runtime model kind. See
[docs/CONFIG_AND_ARTIFACT_CONTRACTS.md](docs/CONFIG_AND_ARTIFACT_CONTRACTS.md)
for the strict runtime and benchmark compatibility rules.

Model presets:

```text
baseline_raw                         raw xyz control representation
recommended_physical_radial_decay    physically informed R_ref/r radial decay encoding
ablation_radial_separation           radial/direction feature ablation
ablation_radial_decay_scaled         legacy scaled inverse-radius ablation
ablation_real_sh_low_degree          real spherical-harmonic basis ablation
custom                               manual encoding flags
```

Raw xyz remains the baseline. The physical radial-decay preset is intended for
benchmarking as a recommended representation, not as an automatic performance
claim.

Runtime profiling:

```bash
python -m lunaris.surrogate.st_lrps.runtime.profiling \
    --model-dir outputs/training/st_lrps_train_xxx \
    --batch-sizes 1,16,128,1024,8192 \
    --n-warmup 10 \
    --n-repeat 50 \
    --out-dir outputs/runtime/st_lrps_runtime_xxx
```

See `docs/profiling.md` for synthetic and dataset-backed profiling, CPU/CUDA timing, chunk-size sensitivity, and output interpretation.

Lightweight benchmark scaffolds:

```bash
python -m lunaris.surrogate.st_lrps.evaluation.runtime_benchmark --help
python -m lunaris.surrogate.st_lrps.evaluation.orbit_benchmark --help
```

Reproducible config-driven benchmark runs:

```bash
lunaris-benchmark --config configs/benchmarks/st_lrps_1day_high_degree.json
lunaris-benchmark \
  --config configs/benchmarks/st_lrps_1day_high_degree.json \
  --model-dir outputs/training/st_lrps_train_YYYYMMDD_HHMMSS \
  --out outputs/gravity_benchmark/st_lrps_1day_high_degree
lunaris-benchmark --config configs/benchmarks/st_lrps_1day_high_degree.json --quick
```

See [docs/REPRODUCIBLE_BENCHMARKS.md](docs/REPRODUCIBLE_BENCHMARKS.md) for
the manifest, resolved config, validation report, and standardized output
layout.

Generated outputs use the repository-level `outputs/` convention by default. Do not place generated runs inside source package directories.

## Resuming ST-LRPS Training

Training is checkpointed every epoch. If a run stops (Ctrl+C, machine shutdown), continue it from the last completed epoch instead of restarting:

```bash
python -m lunaris.surrogate.st_lrps.training.cli \
    --resume-from outputs/training/st_lrps_train_YYYYMMDD_HHMMSS \
    --epochs 300
```

`--resume-from` accepts a run directory, its `checkpoints/` directory, or a specific `.pt` checkpoint.

Key points:

- **`--epochs` is the TOTAL target epoch count, not additional epochs.** If a run completed epoch 100 and you resume with `--epochs 300`, training continues at epoch 101 and stops after epoch 300. To run 200 more epochs after epoch 100, pass `--epochs 300`.
- Resume defaults to **`ckpt_last.pt`** (it carries the optimizer, GradNorm, and RNG state needed to continue). `ckpt_best.pt` is for evaluation/selection; use `--resume-checkpoint best` only for fine-tuning.
- Resume restores model weights **and** optimizer state, the LR schedule position, loss-weighting (GradNorm) state, best-checkpoint tracking, `global_step`, and RNG state — not just the model weights.
- `--data` and `--out` are inferred from the previous run when omitted.
- History is **appended** by default (use `--resume-overwrite-history` to start fresh). The run manifest records the resume event (`resumed: true`, `resume_start_epoch`, `previous_latest_epoch`, `target_epochs`).
- Architecture, encoding, scaler, and dataset-identity settings are locked to the previous run; a critical mismatch fails in strict mode. Use `--resume-nonstrict` to allow non-critical differences.
- Resume is **epoch-level**: if interrupted mid-epoch, it resumes from the last fully completed (saved) epoch. RNG state is restored, but exact DataLoader worker ordering is not bitwise-guaranteed. There is no mid-batch resume.

Resuming from a specific checkpoint file:

```bash
python -m lunaris.surrogate.st_lrps.training.cli \
    --resume-from outputs/training/st_lrps_train_YYYYMMDD_HHMMSS/checkpoints/ckpt_last.pt \
    --epochs 300
```

## Propagation And Analysis

Single-run propagation is driven by the `lunaris` command; Monte Carlo workflows are driven by `lunaris-mc`. These commands are data-dependent and should be configured with local input files and output paths:

```bash
lunaris --help
lunaris-mc --help
```

Canonical analysis modules:

| Purpose | Module |
|---------|--------|
| Post-processing | `lunaris.analysis.postprocess` |
| Report management | `lunaris.analysis.reporting.manager` |
| Report plotting | `lunaris.analysis.reporting.plotting` |
| Report styling | `lunaris.analysis.reporting.styling` |
| Monte Carlo statistics | `lunaris.analysis.monte_carlo.statistics` |
| Monte Carlo plotting | `lunaris.analysis.monte_carlo.plotting` |

## Validation

The validation layer is for independent physics, orbit, and cross-model checks. The current gravity validation harness is:

```bash
python -m lunaris.surrogate.st_lrps.evaluation.compare_gravity_models --help
```

Gravity validation commonly uses a high-degree spherical-harmonic model such as SH200 as the truth/reference, lower-degree spherical-harmonic models as baselines, and optional ST-LRPS comparison when a trained artifact directory is supplied. See [validation/README.md](validation/README.md) and [validation/gravity/README.md](validation/gravity/README.md) for details.

### Orbit-Level Gravity Benchmark (128 scenarios, 5-day propagation)

Validation of the ST-LRPS surrogate against spherical-harmonic (SH) baselines over 128 randomized orbits, run on a consumer workstation (Intel CPU + GTX 1660 Ti). In this benchmark configuration:

* Median RMS position error of **1.106 km** over ~**704,160 km** of total traveled distance (relative error ~**0.00015%**), below the listed lower-degree SH baselines for this scenario set.
* Radial (altitude) error of **41 m** and cross-track error of **6 m** after 5 days of unguided propagation.
* Runtime, for this scenario set:
  * ≈**2x** faster wall-clock than the `SH50` model (3,377 s vs. 6,620 s) at higher accuracy in this configuration.
  * ≈**6%** wall-clock overhead relative to the lightweight `SH20` model.
  * **9.55x** wall-clock speedup versus the sequential CPU truth reference.

### High-Degree SH Comparison (100 scenarios, 1-day propagation)

ST-LRPS compared against high-degree spherical harmonics (`SH100`, `SH200`) under general elliptic orbits ($100\text{ km}$ to $1000\text{ km}$ altitude). In this benchmark configuration:

* Median RMS position error of **0.626 km** — below `SH30` (**1.450 km**) and `SH20` (**18.217 km**), and close to `SH100` (**0.461 km**) and `SH200` (**0.461 km**) for this scenario set.
* On the lightweight `SH20` baseline, the Sobolev-trained residual reduced median RMS position error by a factor of **29.1x** (18.217 km → 0.626 km).
* On GPU, **8.32x** faster wall-clock than `SH200` (665 s vs. 5,540 s) and **3.64x** faster than `SH100` (665 s vs. 2,423 s).

### Near-Circular Double-Precision Benchmark (100 scenarios, 1-day propagation)

Dense low-lunar mapping envelopes ($200\text{ km}$ to $400\text{ km}$ altitude) with double-precision GPU propagation. In this benchmark configuration:

* Median RMS position error of **15.83 cm** over a full day of unguided propagation.
* Radial (altitude) error within **4.58 cm** and cross-track within **2.00 cm** over the 1-day period.
* In `float64` GPU mode, **2.25x** wall-clock speedup versus sequential CPU truth generation.

### Side-by-side comparison

Results under different numerical precision (`float64`), integration step size ($\Delta t$), and orbit envelopes:

| Criterion / Metric | 5-Day General Stability Test | 1-Day High-Degree Comparison | 1-Day Near-Circular Benchmark |
| :--- | :---: | :---: | :---: |
| **Orbit Type** | Bounded Keplerian (Circular/Elliptic) | Bounded Keplerian (Circular/Elliptic) | Near-Circular (Low Circular Orbit) |
| **Numerical Precision (Dtype)** | Single-Precision `float32` | Double-Precision `float64` | Double-Precision `float64` |
| **Integration Step ($\Delta t$)** | $30.0\text{ s}$ | $30.0\text{ s}$ | $10.0\text{ s}$ |
| **ST-LRPS Median RMS Position Error** | **1.106 km** | **0.626 km** *(626.4 m)* | **15.83 cm** |
| **SH20 Baseline Median RMS Error** | **1.570 km** | **18.217 km** (Physical decay) | **1.821 km** (Destabilized orbit) |
| **Radial (Altitude) Median RMS** | **41 meters** | **7.20 cm** | **4.58 cm** |
| **Cross-Track (Inclination) Median RMS**| **6 meters** | **4.87 cm** | **2.00 cm** |
| **Along-Track (Phase) Median RMS** | **1.102 km** | **62.12 cm** | **15.03 cm** |
| **GPU Speedup (vs. Truth)** | **9.55x** speedup (vs. CPU) | **5.59x** speedup (**8.32x** vs. SH200) | **2.25x** speedup (vs. CPU) |

For the full benchmark breakdown, tables, analysis, and reproduction steps (CLI or desktop UI), see [ST-LRPS Gravity Model Benchmark Results](docs/BENCHMARK_RESULTS.md).


## Visualization

Standalone visualization tools live under `src/lunaris/visualization/`.

| Purpose | Module |
|---------|--------|
| Orbit animation and trajectory visualization | `lunaris.visualization.orbit_animation.render_orbit_animation` |
| Topography and albedo exploration | `lunaris.visualization.surface_explorer` |

Surface explorer help:

```bash
python -m lunaris.visualization.surface_explorer --help
```

Example topography render:

```bash
python -m lunaris.visualization.surface_explorer \
    --topo-label data/topography_models/ldem_64_float.lbl \
    --topo-img data/topography_models/ldem_64_float.img \
    --out-dir outputs/surface_explorer \
    --plot-2d --plot-3d
```

Large LOLA grids can be memory-heavy. Use `--stride-2d`, `--stride-3d`, or `--stride-albedo` for quick previews.

## Generated Output Policy

Generated outputs should not be committed. New tools should write generated products under `outputs/` unless the user explicitly chooses an external scratch directory.

Canonical generated-output layout:

```text
outputs/
  simulations/          core orbit propagation runs (CLI / GUI)
  monte_carlo/          Monte Carlo batch runs and reports
  missions/             GUI "mission" propagation outputs
  gravity_benchmark/    orbit-level gravity-model benchmark (CLI + Studio)
  training/             ST-LRPS training run directories
    st_lrps_train_<timestamp>/
      checkpoints/      model checkpoints for that run
      plots/            training curves and diagnostics
      evals/            evaluations attached to that trained run
      provenance/       run metadata and dataset snapshots
  evaluations/          standalone evaluation reports not attached to a run
  runtime/              ST-LRPS runtime profiling and benchmark reports
  dataset_reports/      generated cloud/dataset analysis reports
  datasets/
    cloud_suites/       generated train/val/test/OOD dataset suites
  validation/           validation harness outputs
  visualization/        standalone visualization outputs
```

The `evals/` directory is intentionally run-local: if an evaluation is launched for a selected training run and no output directory is provided, it is written below that run so the model artifact and quality report travel together.

Examples of generated paths and files:

```text
outputs/
checkpoints/
evals/
history.jsonl
run_manifest.json
metrics_summary.csv
topk_worst.csv
ood_metrics.csv
```

Source packages such as `src/lunaris/surrogate/st_lrps/`, `src/lunaris/analysis/`, `validation/`, and `src/lunaris/visualization/` should contain source code and documentation, not generated run artifacts, checkpoints, plots, or evaluation tables.

## Testing

Run lightweight documentation and visualization checks:

```bash
pytest tests/test_repo_hygiene.py
pytest tests/test_validation_docs.py
pytest tests/test_surface_explorer_visualization.py
```

Run the full test suite when making code changes:

```bash
pytest tests/
```

## License

MIT License. See [LICENSE](LICENSE) for details.
