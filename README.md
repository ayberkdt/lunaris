# ST-LRPS: Sobolev-Trained Lunar Residual Potential Surrogate

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

ST-LRPS is a lunar gravity modeling and propagation framework centered on a Sobolev-trained residual-potential surrogate model. The repository includes spherical-harmonic gravity modeling, ST-LRPS training/evaluation/runtime inference, orbit propagation, Monte Carlo analysis, validation tools, visualization tools, and a desktop UI.

## Overview

The framework supports lunar-orbit propagation with configurable physical force models, spherical-harmonic lunar gravity, optional ST-LRPS residual-potential inference, Monte Carlo workflows, validation harnesses, report generation, and PySide6-based desktop workflows. ST-LRPS is designed to learn a residual scalar potential above a lower-degree spherical-harmonic baseline; runtime acceleration can then be obtained from the learned potential gradient and combined with the baseline gravity model.

Accuracy, runtime, and stability depend on the selected data, force-model configuration, trained artifacts, and validation scenario. Treat validation outputs as run-specific evidence rather than a blanket guarantee.

## Repository Architecture

The current repository uses a `src/` package layout. The installable package is
`lunaris`; ST-LRPS remains a named surrogate family under
`lunaris.surrogate.st_lrps`.

```text
src/lunaris/
  common/          shared constants, dataclasses, validation helpers
  loaders/         gravity/topography/ephemeris/data loading
  physics/         physical/environment models and adapters
  core/            dynamics, propagation, Monte Carlo backend, configuration
  analysis/        post-processing, reports, Monte Carlo analysis
  visualization/   standalone visualization tools
  cli/             shared CLI argument helpers and main launcher
  ui/              general Lunar Orbit Simulator desktop UI
    widgets/       desktop UI page components
  surrogate/
    st_lrps/       Sobolev-Trained Lunar Residual Potential Surrogate package
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

Top-level entry points:

```text
main.py           compatibility launcher for `lunaris`
mc_runner.py      compatibility launcher for `lunaris-mc`
ui.py             compatibility launcher for `lunaris-ui`
studio.py         compatibility launcher for `lunaris-studio`
```

## Installation

Install the standard Python dependencies from the repository root:

```bash
python -m pip install -r requirements.txt
```

**For HPC and Cluster Deployments**, you should exclude GUI dependencies. See the [HPC and Cluster Deployment Guide](docs/HPC.md) for Conda (`environment.yml`), Headless CLI `requirements_hpc.txt`, and Slurm templates (`slurm_examples/`).

Large mission data files are not bundled. Place local SPICE kernels, gravity coefficient files, topography grids, and albedo grids under `data/` or another local path configured at runtime.

Common data locations:

| Directory | Contents |
|-----------|----------|
| `data/ephemeris_models/` | SPICE leap-second, planetary, constants, and lunar orientation kernels |
| `data/gravity_models/` | Lunar spherical-harmonic coefficient files |
| `data/topography_models/` | LOLA/LDEM topography rasters |
| `data/albedo_models/` | Optional lunar albedo grids |

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

Model target semantics are recorded explicitly through a `target_contract` in
new configs/checkpoints. The contract distinguishes residual labels from
full-field labels, records the baseline degree/kind, and keeps the runtime path
aligned with the scaler and loss.

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

Single-run propagation is driven by `main.py`; Monte Carlo workflows are driven by `mc_runner.py`. These commands are data-dependent and should be configured with local input files and output paths:

```bash
python main.py --help
python mc_runner.py --help
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

### 📊 Orbit-Level Gravity Benchmark Results (128 Scenarios, 5-Day Propagation)

A comprehensive physical validation benchmark of the **ST-LRPS neural surrogate** against classical Spherical Harmonic (SH) baselines over **128 randomized orbits** has been completed on a consumer laptop workstation (Intel CPU + GTX 1660 Ti):

* **The Accuracy Victory:** ST-LRPS outperformed all classical baselines in median trajectory accuracy, achieving a median RMS position error of **1.106 km** over a total traveled distance of **704,160 km** (a relative error of only **0.00015%**!).
* **Meter-Level Orbit Control:** Radial (Altitude) error of only **41 meters** and Cross-Track (Plane tilt) error of only **6 meters** after 5 days of unguided propagation!
* **The Computational Speedup:** 
  * ST-LRPS is **nearly 2x faster than SH50** (3,377 seconds vs. 6,620 seconds), while delivering far superior physical accuracy.
  * ST-LRPS adds only **6% computational overhead** compared to the extremely lightweight `SH20` baseline model.
  * It achieves a **9.55x wall-clock speedup** compared to the high-fidelity sequential CPU truth reference.

### ⚡ 1-Day High-Degree Spherical Harmonic Benchmark (100 Scenarios, 1-Day Propagation)

A validation comparing ST-LRPS directly against classical high-degree Spherical Harmonics (`SH100` and `SH200`) under general elliptic orbits ($100\text{ km}$ to $1000\text{ km}$ altitude):

* **High-Degree Accuracy Parity:** ST-LRPS achieved a median RMS position error of only **0.626 km**, outperforming `SH30` (**1.450 km**) and `SH20` (**18.217 km**), while closely matching the accuracy of `SH100` (**0.461 km**) and `SH200` (**0.461 km**).
* **29x Baseline Correction:** Sitting on a lightweight `SH20` baseline, ST-LRPS corrected the error by a factor of **29.1x** (from 18.217 km down to 0.626 km) using Sobolev neural residuals.
* **Massive GPU Speedup:** ST-LRPS executed **8.32x faster than SH200** (665s vs 5,540s) and **3.64x faster than SH100** (665s vs 2,423s), proving high-degree potential surrogate efficiency.

### 🎯 Ultra-Precision 1-Day Near-Circular Gravity Benchmark (100 Scenarios, 1-Day Propagation)

A specialized benchmark focusing on dense low-lunar mapping envelopes ($200\text{ km}$ to $400\text{ km}$ altitude) with double-precision GPU propagation showcases the extreme fidelity limits of ST-LRPS:

* **Sub-Meter Trajectory Accuracy:** ST-LRPS achieved a median RMS position error of only **15.83 cm** over a full day of unguided propagation.
* **Centimeter-Level RIC Control:** Radial (altitude) error was maintained within **4.58 cm** and plane inclination tilt within **2.00 cm** over the entire 1-day period.
* **GPU Double-Precision Performance:** Even in `float64` double-precision mode on the GPU, ST-LRPS delivered a **2.25x wall-clock speedup** compared to sequential CPU truth generation.

### 🔄 Side-by-Side Performance Comparison

The table below demonstrates how adjusting numerical precision (`float64`), integration step size ($\Delta t$), and focusing on specific orbit envelopes showcases different performance and accuracy regimes of the ST-LRPS surrogate:

| Criterion / Metric | 5-Day General Stability Test | 1-Day High-Degree Comparison | 1-Day Ultra-Precision Benchmark |
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

For the complete benchmark breakdown, tables, physical analyses, and step-by-step instructions on how to reproduce the results via the CLI or the Desktop UI, see the official **[ST-LRPS Gravity Model Benchmark Results](docs/BENCHMARK_RESULTS.md)**.


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
