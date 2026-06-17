# Lunaris

**Lunar orbit propagation and gravity-modeling framework.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Status: Alpha](https://img.shields.io/badge/status-alpha-orange.svg)](https://pypi.org/classifiers/)

Lunaris is a Python framework for lunar-orbit propagation and gravity modeling. It
bundles spherical-harmonic lunar gravity, configurable physical force models, orbit
propagation, Monte Carlo analysis, validation harnesses, visualization tools, and a
PySide6 desktop UI.

It also ships **ST-LRPS** (Sobolev-Trained Lunar Residual Potential Surrogate) — a
neural surrogate-gravity model under `lunaris.surrogate.st_lrps` that learns a
residual scalar potential above a lower-degree spherical-harmonic baseline, with its
own training, evaluation, and Studio UI.

> **Project status.** Lunaris is an **alpha-stage research prototype**
> (`Development Status :: 3 - Alpha`). APIs, trained artifacts, and reported
> benchmark numbers may change between versions. Treat validation outputs as
> run-specific evidence, not a blanket guarantee.

## Documentation

This README is a landing page; the canonical detail lives in `docs/`.

| Document | Contents |
|----------|----------|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Layered design, data flow, configuration model, **force-model / perturbation flags**, Monte Carlo internals, ST-LRPS surrogate |
| [docs/ST_LRPS_VALIDATION_HYGIENE.md](docs/ST_LRPS_VALIDATION_HYGIENE.md) | Train-only scalers, spatial/OOD split policies, runtime frame safety, paper-safe benchmarks, validation + ablation suites |
| [docs/BENCHMARK_RESULTS.md](docs/BENCHMARK_RESULTS.md) | Full gravity-model benchmark tables and reproduction steps |
| [docs/REPRODUCIBLE_BENCHMARKS.md](docs/REPRODUCIBLE_BENCHMARKS.md) | Config-driven benchmark runs, provenance manifests, validation reports, and CI smoke mode |
| [docs/DATASET_PIPELINE.md](docs/DATASET_PIPELINE.md) | ST-LRPS dataset contract, validation, quality reports, split manifests, and strict training ingestion |
| [docs/CONFIG_AND_ARTIFACT_CONTRACTS.md](docs/CONFIG_AND_ARTIFACT_CONTRACTS.md) | ST-LRPS dataset, training, checkpoint, runtime, and benchmark contract rules |
| [docs/PERTURBATION_BUDGET.md](docs/PERTURBATION_BUDGET.md) | Perturbation-budget assumptions, outputs, and interpretation |
| [docs/HPC.md](docs/HPC.md) | Cluster/headless install, Conda environment, Slurm templates, scenario arrays |
| [docs/profiling.md](docs/profiling.md) | ST-LRPS runtime profiling and timing interpretation |
| [validation/README.md](validation/README.md) | Independent physics/orbit/gravity validation harnesses |
| [CONTRIBUTING.md](CONTRIBUTING.md) · [SECURITY.md](SECURITY.md) | Dev setup + quality gates · vulnerability reporting |

## Force models

Implemented and wired into the propagator (`lunaris.core.dynamics`):

- Spherical-harmonic lunar gravity (and the ST-LRPS surrogate-gravity model)
- Third-body perturbations (Sun, Earth)
- Earth oblateness (differential J2)
- Solar radiation pressure (with eclipse handling)
- Lunar albedo (reflected-solar) surface radiation
- Lunar thermal IR radiation pressure
- Elastic lunar solid-body tides (`k2`, optional explicit `k3`; Earth and/or Sun raised)
- First-order post-Newtonian relativity

Each model is configurable from the `lunaris` CLI; see
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full flag reference,
assumptions, and limitations. Example:

```bash
lunaris --enable-tides on --tides-kind k2 --tide-bodies earth,sun
lunaris --enable-thermal on --thermal-mode equilibrium_temperature
lunaris --enable-albedo on --albedo-mode scaled_dn_grid --albedo-root data/albedo_models
```

## Installation

Install in editable mode from the repository root to wire up the console entry
points and pick up code changes without reinstalling:

```bash
python -m pip install -e .            # core dependencies only
python -m pip install -e ".[all]"     # core + ML + UI + reports + dev extras
```

Optional dependency groups (`pyproject.toml`): `ml`, `hpc`, `ui`, `reports`,
`dev`, `all`. Every dependency carries lower and upper version bounds so a build
cannot silently pull in an incompatible major release. For reproducible
environments, exact pins live in `locks/*.lock.txt` (generated with `uv`). For
clusters, use the headless `.[hpc]` extra and the Slurm templates in `hpc/`; see
the [HPC guide](docs/HPC.md).

Large mission data (SPICE kernels, gravity coefficients, topography, albedo) is
**not bundled**. Fetch and verify it with the headless `lunaris-data` tool into
`LUNARIS_DATA_DIR` (or the repo `data/` folder):

```bash
lunaris-data list
lunaris-data download --group ephemeris
lunaris-data verify
```

The catalogue is `data/data_sources.json`; entries without a pinned URL print
manual-placement instructions. Common locations: `data/ephemeris_models/`,
`data/gravity_models/`, `data/topography_models/`, `data/albedo_models/`,
`data/thermal_models/`.

## Quickstart

These checks do not require private local datasets:

```bash
python -m pip install -e ".[hpc]"
python -c "import lunaris; print(lunaris.__version__)"
lunaris-train --help
lunaris-eval --help
lunaris-benchmark --help
lunaris-mc --help
lunaris-perturbation-budget --help
python -m lunaris.surrogate.st_lrps.training.cli --help
python -m lunaris.surrogate.st_lrps.evaluation.cli --help
python -m lunaris.surrogate.st_lrps.evaluation.compare_gravity_models --help
python -m lunaris.visualization.surface_explorer --help
```

Data-dependent workflows (full propagation, ST-LRPS training, gravity validation,
topography plots) require local gravity, SPICE, or LOLA files.

## Repository architecture

A `src/` package layout organized into **four strict layers** (a layer never
imports from a layer above it) plus dependency-light support packages. See
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the canonical design.

```text
src/lunaris/
  common/          [layer 1] shared constants, config dataclasses, math/time helpers
  physics/         [layer 2] Numba force-model kernels, ephemeris (SPICE), gravity adapters
  core/            [layer 3] config (SimConfig SSOT), dynamics RHS, propagator, events, Monte Carlo
  analysis/        [layer 4] post-processing, reports, Monte Carlo analysis
  visualization/   [layer 4] standalone visualization tools
  ui/              [layer 4] Lunar Orbit Simulator desktop UI (PySide6)
  loaders/         (support) gravity / topography / ephemeris / data loading
  cli/             (support) console entry points and shared CLI helpers
  surrogate/st_lrps/   Sobolev-Trained Lunar Residual Potential Surrogate family
      data/ training/ networks/ artifacts/ evaluation/ runtime/ shared/ ui/
validation/        independent physics/orbit/gravity validation harnesses + docs
tests/             unit and regression tests
data/              local input data (SPICE kernels, gravity, topography)
hpc/               example Slurm templates for cluster use
```

Console entry points (installed via `pip install -e .`):

```text
lunaris           single-run propagation CLI
lunaris-mc        Monte Carlo runner
lunaris-launcher  welcome hub (picks a workspace; optional offline 3D Moon preview)
lunaris-ui        mission desktop UI (Lunaris Mission Studio)
lunaris-studio    ST-LRPS Studio UI
lunaris-train / lunaris-train-force-direct      ST-LRPS training CLIs
lunaris-eval  / lunaris-eval-force-direct       ST-LRPS evaluation CLIs
lunaris-benchmark ST-LRPS orbit-level gravity benchmark / validation CLI
lunaris-data      external-data download / verify CLI
lunaris-perturbation-budget   acceleration / force-model uncertainty budget
```

The desktop UI uses a unified dark theme, **Lunar Graphite**, whose tokens flow
from `lunaris.ui_foundation` (see [docs/UI_THEME.md](docs/UI_THEME.md)). The
launcher can show an optional, **offline** Three.js Moon preview (built from
`src/lunaris/ui/web`); it never blocks the app and falls back gracefully when
absent.

## ST-LRPS at a glance

```bash
# dataset generation, training, evaluation, ablation
python -m lunaris.surrogate.st_lrps.data.spatial_cloud_generator --help
python -m lunaris.surrogate.st_lrps.training.cli --help
python -m lunaris.surrogate.st_lrps.evaluation.cli --help
python -m lunaris.surrogate.st_lrps.evaluation.ablation --help
```

At runtime ST-LRPS supports two artifact contracts: the default
`potential_autograd` (learned scalar residual potential, acceleration via
autograd; validation-safe) and the experimental `force_direct` (3-output
direct residual acceleration, not conservative by construction — requires curl
and orbit-level validation before scientific claims). Versioned
`artifact_contract` / `dataset_contract` blocks record target semantics, baseline
degree, altitude envelope, scaler contract, encoding, and runtime kind.

Run-level posture is selected with `--run-preset {development,quick,paper}`;
`paper` enforces a generalization split, deterministic execution, and the
preflight gate. Reproducible config-driven benchmarks:

```bash
lunaris-benchmark --config configs/benchmarks/st_lrps_1day_high_degree.json
lunaris-benchmark --config configs/benchmarks/st_lrps_1day_high_degree.json --quick
```

See [docs/CONFIG_AND_ARTIFACT_CONTRACTS.md](docs/CONFIG_AND_ARTIFACT_CONTRACTS.md),
[docs/ST_LRPS_VALIDATION_HYGIENE.md](docs/ST_LRPS_VALIDATION_HYGIENE.md), and
[docs/HPC.md](docs/HPC.md#1b-st-lrps-scenario-arrays-reproducible-sweeps) for
contracts, validation hygiene, and reproducible HPC sweeps.

**Resuming training.** Runs are checkpointed every epoch; continue with
`--resume-from <run-dir | checkpoints/ | ckpt.pt>`. Note that `--epochs` is the
**TOTAL** target epoch count, not additional epochs, and resume restores
optimizer / LR-schedule / GradNorm / RNG state (not just weights). Full semantics
are in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Propagation, Monte Carlo, and analysis

Single-run propagation is driven by `lunaris`; Monte Carlo workflows by
`lunaris-mc` (both data-dependent). Monte Carlo backends are explicit (`cpu_sh`
truth reference, `numba_cuda_sh`, `torch_cuda_sh`, `torch_cpu_sh`,
`gpu_st_lrps_potential`, `gpu_st_lrps_direct`); selection is resolved centrally by
`lunaris.core.mc_backend_policy`, and the requested vs. effective backend, device,
integrator, and any fallback reason are recorded in `MCRunResult.diagnostics`
rather than applied silently. The perturbation budget tool quantifies
acceleration contributions and force-model uncertainty:

```bash
lunaris-perturbation-budget --altitudes-km 50,100,300,1000 --sh-degrees 20,60,200 \
  --gravity-model path/to/lunar_gravity_model.tab --out-dir outputs/perturbation_budget/default
```

Post-processing, reporting, and Monte Carlo statistics/plotting live under
`lunaris.analysis.*`. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Validation and benchmarks

The validation layer provides independent physics, orbit, and cross-model checks,
including external-reference harnesses (`scipy`/`pyshtools` SH cross-checks and
direct NAIF/SPICE ephemeris checks) under `validation/independent/`. The gravity
benchmark CLI compares ST-LRPS against spherical-harmonic baselines:

```bash
python -m lunaris.surrogate.st_lrps.evaluation.compare_gravity_models --help
```

Selected results (consumer workstation, Intel CPU + GTX 1660 Ti; see
[docs/BENCHMARK_RESULTS.md](docs/BENCHMARK_RESULTS.md) for full tables, scenario
counts, and reproduction):

| Benchmark | ST-LRPS median RMS position error | Note |
|-----------|-----------------------------------|------|
| 5-day general stability (128 scenarios, `float32`) | **1.106 km** | ≈2× faster wall-clock than `SH50` at higher accuracy |
| 1-day high-degree comparison (100 scenarios, `float64`) | **0.626 km** | 29.1× lower error than `SH20`; 8.32× faster than `SH200` |
| 1-day near-circular mapping (100 scenarios, `float64`) | **15.83 cm** | 2.25× speedup vs. sequential CPU truth |

Numbers are run-specific evidence for the stated configuration, not a blanket
performance guarantee.

## Visualization

Standalone tools live under `src/lunaris/visualization/` (orbit animation via
`lunaris.visualization.orbit_animation.render_orbit_animation`; topography/albedo
via `lunaris.visualization.surface_explorer`):

```bash
python -m lunaris.visualization.surface_explorer \
    --topo-label data/topography_models/ldem_64_float.lbl \
    --topo-img data/topography_models/ldem_64_float.img \
    --out-dir outputs/surface_explorer --plot-2d --plot-3d
```

Large LOLA grids are memory-heavy; use `--stride-2d`/`--stride-3d`/`--stride-albedo`
for quick previews.

## Generated output policy

Generated products are not committed; tools write under `outputs/` (git-ignored)
unless an external scratch directory is chosen. Source packages
(`src/lunaris/...`, `validation/`) hold source and documentation, never run
artifacts, checkpoints, plots, or evaluation tables. The standard layout
(`outputs/{simulations,monte_carlo,missions,gravity_benchmark,training,evaluations,runtime,dataset_reports,datasets,validation,visualization}/`)
keeps a trained run's checkpoints, plots, evals, and provenance together.

## Testing

```bash
pytest tests/                       # full suite
pytest tests/test_repo_hygiene.py   # lightweight docs/hygiene checks
```

CI enforces a coverage baseline and uploads HTML coverage reports as artifacts.
See [CONTRIBUTING.md](CONTRIBUTING.md) for the full quality gates (`ruff`,
`mypy`, `lint-imports`, test markers).

## License

MIT License. See [LICENSE](LICENSE) for details.
