# ST-LRPS: Sobolev-Trained Lunar Residual Potential Surrogate

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

ST-LRPS is a lunar gravity modeling and propagation framework centered on a Sobolev-trained residual-potential surrogate model. The repository includes spherical-harmonic gravity modeling, ST-LRPS training/evaluation/runtime inference, orbit propagation, Monte Carlo analysis, validation tools, visualization tools, and a desktop UI.

## Overview

The framework supports lunar-orbit propagation with configurable physical force models, spherical-harmonic lunar gravity, optional ST-LRPS residual-potential inference, Monte Carlo workflows, validation harnesses, report generation, and PySide6-based desktop workflows. ST-LRPS is designed to learn a residual scalar potential above a lower-degree spherical-harmonic baseline; runtime acceleration can then be obtained from the learned potential gradient and combined with the baseline gravity model.

Accuracy, runtime, and stability depend on the selected data, force-model configuration, trained artifacts, and validation scenario. Treat validation outputs as run-specific evidence rather than a blanket guarantee.

## Repository Architecture

The current repository uses the Part 2 ST-LRPS package layout.

```text
st_lrps/
  data/          dataset definitions, spatial cloud generation, dataset loading
  training/      ST-LRPS training config, CLI, engine, losses, metrics
  networks/      neural network architecture definitions
  artifacts/     run layout, checkpoints, manifests, artifact validation
  evaluation/    trained-model evaluation and ablation CLI
  runtime/       propagator-facing ST-LRPS force model API
  shared/        shared scaling utilities
  ui/            ST-LRPS-specific UI components

core/             dynamics, state conversion, propagation, Monte Carlo backend
models/           physical/environment models and adapters
loaders/          gravity/topography/ephemeris/data loading
common/           shared constants, dataclasses, validation helpers
analysis/         post-processing, reports, Monte Carlo analysis
  postprocess.py
  formatting.py
  reporting/
    manager.py
    plotting.py
    styling.py
  monte_carlo/
    statistics.py
    plotting.py
validation/       independent physics/orbit/gravity validation harnesses
  gravity/
    compare_gravity_models.py
visualization/    standalone visualization tools
  orbit_animation.py
  surface_explorer.py
ui_parts/         desktop UI page components
cli/              shared CLI argument helpers
tests/            unit and regression tests
data/             local input data such as SPICE kernels, gravity, topography
```

Top-level entry points:

```text
main.py           single-run propagation CLI entry point
mc_runner.py      Monte Carlo CLI entry point
ui.py             desktop application entry point
config.py         application configuration dataclasses and defaults
```

## Installation

Install the Python dependencies from the repository root:

```bash
python -m pip install -r requirements.txt
```

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
python -m pip install -r requirements.txt
python -c "import st_lrps; print(st_lrps.__version__)"
python -m st_lrps.training.cli --help
python -m st_lrps.evaluation.cli --help
python -m validation.gravity.compare_gravity_models --help
python -m visualization.surface_explorer --help
```

Data-dependent examples such as full propagation, ST-LRPS training, gravity validation runs, and topography plots require local gravity, SPICE, or LOLA files.

## ST-LRPS Commands

Spatial cloud generation:

```bash
python -m st_lrps.data.spatial_cloud_generator --help
```

Training:

```bash
python -m st_lrps.training.cli --help
```

Evaluation:

```bash
python -m st_lrps.evaluation.cli --help
```

Ablation:

```bash
python -m st_lrps.evaluation.ablation --help
```

Runtime import example:

```python
from st_lrps.runtime.force_model import load_surrogate_force_model
```

Training and evaluation outputs should be written to user-selected output directories such as top-level `runs/`, `artifacts/`, `outputs/`, or a scratch location outside the repository. Do not place generated runs inside source package directories.

## Resuming ST-LRPS Training

Training is checkpointed every epoch. If a run stops (Ctrl+C, machine shutdown), continue it from the last completed epoch instead of restarting:

```bash
python -m st_lrps.training.cli \
    --resume-from runs/st_lrps_train_YYYYMMDD_HHMMSS \
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
python -m st_lrps.training.cli \
    --resume-from runs/st_lrps_train_YYYYMMDD_HHMMSS/checkpoints/ckpt_last.pt \
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
| Post-processing | `analysis.postprocess` |
| Report management | `analysis.reporting.manager` |
| Report plotting | `analysis.reporting.plotting` |
| Report styling | `analysis.reporting.styling` |
| Monte Carlo statistics | `analysis.monte_carlo.statistics` |
| Monte Carlo plotting | `analysis.monte_carlo.plotting` |

## Validation

The validation layer is for independent physics, orbit, and cross-model checks. The current gravity validation harness is:

```bash
python -m validation.gravity.compare_gravity_models --help
```

Gravity validation commonly uses a high-degree spherical-harmonic model such as SH200 as the truth/reference, lower-degree spherical-harmonic models as baselines, and optional ST-LRPS comparison when a trained artifact directory is supplied. See `validation/README.md` and `validation/gravity/README.md` for details.

## Visualization

Standalone visualization tools live under `visualization/`.

| Purpose | Module |
|---------|--------|
| Orbit animation and trajectory visualization | `visualization.orbit_animation.render_orbit_animation` |
| Topography and albedo exploration | `visualization.surface_explorer` |

Surface explorer help:

```bash
python -m visualization.surface_explorer --help
```

Example topography render:

```bash
python -m visualization.surface_explorer \
    --topo-label data/topography_models/ldem_64_float.lbl \
    --topo-img data/topography_models/ldem_64_float.img \
    --out-dir outputs/surface_explorer \
    --plot-2d --plot-3d
```

Large LOLA grids can be memory-heavy. Use `--stride-2d`, `--stride-3d`, or `--stride-albedo` for quick previews.

## Generated Output Policy

Generated outputs should not be committed. Keep run products in ignored top-level output directories or external scratch storage.

Examples of generated paths and files:

```text
runs/
results/
artifacts/
outputs/
checkpoints/
evals/
history.jsonl
run_manifest.json
metrics_summary.csv
topk_worst.csv
ood_metrics.csv
```

Source packages such as `st_lrps/`, `analysis/`, `validation/`, and `visualization/` should contain source code and documentation, not generated run artifacts, checkpoints, plots, or evaluation tables.

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
