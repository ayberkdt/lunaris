# Lunaris

**Lunar orbit propagation and gravity-modeling framework.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Status: Active Development](https://img.shields.io/badge/status-active%20development-blue.svg)](https://pypi.org/classifiers/)

Lunaris is a Python framework for lunar-orbit propagation and gravity modeling. It
bundles spherical-harmonic lunar gravity, configurable physical force models, orbit
propagation, batch/ensemble uncertainty analysis, validation harnesses,
visualization tools, and a PySide6 desktop UI.

It also ships **ST-LRPS** (Sobolev-Trained Lunar Residual Potential Surrogate), a
neural surrogate-gravity model under `lunaris.surrogate.st_lrps` that learns a
residual scalar potential above a lower-degree spherical-harmonic baseline, with its
own training, evaluation, and Studio UI.

> **ST-LRPS is a high-throughput *batch* gravity backend, not a low-latency
> single-trajectory CPU replacement.** A single trajectory run through
> `propagate()` evaluates the surrogate as an interpreted PyTorch + autograd
> closure (not a Numba kernel), so it pays per-call Python/autograd overhead on
> every RHS evaluation and will be *slower* than the `@njit` spherical-harmonic
> kernel. The surrogate's advantage is amortized only across a large GPU batch
> (Monte Carlo / ensemble). Do **not** benchmark "ST-LRPS vs SH" by timing one
> CPU trajectory — that measures the wrong path. Compare like-for-like on the GPU
> batch backend. See [docs/profiling.md](docs/profiling.md).

> **Project status.** Lunaris is **actively developed research software** with
> versioned on-disk contracts (datasets, checkpoints, runtime, and benchmark
> artifacts) and a documented validation pipeline. Public APIs may still evolve
> between releases, so pin a version for reproducible work. Trained artifacts and
> reported benchmark numbers are tied to a specific run and configuration: treat
> validation outputs as run-specific evidence produced through the evidence
> pipeline, not a blanket guarantee.

## Documentation

This README is a landing page; the canonical detail lives in `docs/`.

| Document | Contents |
|----------|----------|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Layered design, data flow, configuration model, **force-model / perturbation flags**, batch/ensemble propagation internals, ST-LRPS surrogate |
| [docs/ST_LRPS_VALIDATION_HYGIENE.md](docs/ST_LRPS_VALIDATION_HYGIENE.md) | Train-only scalers, spatial/OOD split policies, runtime frame safety, paper-safe benchmarks, validation + ablation suites |
| [docs/BENCHMARK_RESULTS.md](docs/BENCHMARK_RESULTS.md) | Full gravity-model benchmark tables and reproduction steps |
| [docs/REPRODUCIBLE_BENCHMARKS.md](docs/REPRODUCIBLE_BENCHMARKS.md) | Config-driven benchmark runs, provenance manifests, validation reports, and CI smoke mode |
| [docs/DATASET_PIPELINE.md](docs/DATASET_PIPELINE.md) | ST-LRPS dataset contract, validation, quality reports, split manifests, and strict training ingestion |
| [docs/CONFIG_AND_ARTIFACT_CONTRACTS.md](docs/CONFIG_AND_ARTIFACT_CONTRACTS.md) | ST-LRPS dataset, training, checkpoint, runtime, and benchmark contract rules |
| [docs/PERTURBATION_BUDGET.md](docs/PERTURBATION_BUDGET.md) | Perturbation-budget assumptions, outputs, and interpretation |
| [docs/HPC.md](docs/HPC.md) | Cluster/headless install, Conda environment, Slurm templates, scenario arrays |
| [docs/profiling.md](docs/profiling.md) | ST-LRPS runtime profiling and timing interpretation |
| [validation/README.md](validation/README.md) | Independent physics/orbit/gravity validation harnesses |
| [CONTRIBUTING.md](CONTRIBUTING.md) / [SECURITY.md](SECURITY.md) | Dev setup + quality gates / vulnerability reporting |

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
lunaris-data download --group gravity
lunaris-data verify
lunaris-data verify --strict --runtime
```

The catalogue is `data/data_sources.json`. Official-provider entries download
from NAIF/JPL or NASA PDS, recorded hashes are checked by `lunaris-data verify`,
and `--strict --runtime` also proves the resolved SPICE kernels can build a small
ephemeris table. Common locations: `data/ephemeris_models/`,
`data/gravity_models/`, `data/topography_models/`, `data/albedo_models/`,
`data/thermal_models/`, and `data/assets/`. ST-LRPS cloud suites are generated
artifacts under `data/datasets/`, not external downloads.

## Quickstart

These checks do not require private local datasets:

```bash
python -m pip install -e ".[hpc]"
python -c "import lunaris; print(lunaris.__version__)"
lunaris-train --help
lunaris-eval --help
lunaris-benchmark --help
lunaris-batch --help
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
  core/            [layer 3] config (SimConfig SSOT), dynamics RHS, propagator, events, batch ensembles
  analysis/        [layer 4] post-processing, reports, ensemble uncertainty analysis
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
lunaris-batch     batch/ensemble propagation runner
lunaris-mc        Monte Carlo-oriented entry point for the batch runner
lunaris-launcher  welcome hub (picks a workspace; optional offline 3D Moon preview)
lunaris-ui        mission desktop UI (Lunaris Mission Studio)
lunaris-studio    ST-LRPS Studio UI
lunaris-train / lunaris-train-force-direct      ST-LRPS training CLIs
lunaris-eval  / lunaris-eval-force-direct       ST-LRPS evaluation CLIs
lunaris-benchmark ST-LRPS orbit-level gravity benchmark / validation CLI
lunaris-validate  gravity-reference checks + ST-LRPS validation suite
lunaris-ablation  ST-LRPS ablation suite runner
lunaris-st-lrps-paper-evidence  end-to-end paper-evidence pipeline
lunaris-data      external-data download / verify CLI
lunaris-perturbation-budget   acceleration / force-model uncertainty budget
```

The canonical inventory with stability classifications is
[docs/PUBLIC_API.md](docs/PUBLIC_API.md).

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
direct residual acceleration, not conservative by construction; requires curl
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

## Propagation, Batch Ensembles, And Analysis

Single-run propagation is driven by `lunaris`; ensemble propagation by
`lunaris-batch` and the Monte Carlo-oriented `lunaris-mc` entry point. The ensemble
sampling design is explicit: `random` is the classical Monte Carlo option, while
`lhs`, `sobol`, and `sobol_scrambled` are space-filling designs better suited to
validation and benchmark coverage. Batch backends are explicit (`cpu_sh` truth
reference, `numba_cuda_sh`, `torch_cuda_sh`, `torch_cpu_sh`,
`gpu_st_lrps_potential`, `gpu_st_lrps_direct`); selection is resolved centrally
by `lunaris.core.mc_backend_policy`, and the requested vs. effective backend,
device, integrator, sampling method, and any fallback reason are recorded in
`MCRunResult.diagnostics` rather than applied silently. The perturbation budget
tool quantifies acceleration contributions and force-model uncertainty:

```bash
lunaris-perturbation-budget --altitudes-km 50,100,300,1000 --sh-degrees 20,60,200 \
  --gravity-model path/to/lunar_gravity_model.tab --out-dir outputs/perturbation_budget/default
```

Post-processing, reporting, and ensemble statistics/plotting live under
`lunaris.analysis.*`. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Validation and benchmarks

The validation layer provides independent physics, orbit, and cross-model checks,
including external-reference harnesses (`scipy`/`pyshtools` SH cross-checks and
direct NAIF/SPICE ephemeris checks) under `validation/independent/`. The gravity
benchmark CLI evaluates ST-LRPS and classical spherical-harmonic runs against a
declared numerical reference:

```bash
python -m lunaris.surrogate.st_lrps.evaluation.compare_gravity_models --help
```

Current publication status: no regenerated, paper-safe ST-LRPS orbit-accuracy
table is published in this README. A benchmark becomes citeable only when its
`validation_report.json` passes and its `benchmark_manifest.json` records the
resolved config, commit, scenario seed/count, numerical reference, hardware,
backend, model artifact, dataset/gravity hashes when available, and ST-LRPS
artifact-contract compatibility.

How to read a gravity benchmark:

- **Median / P95 / max RMS position error [km]**: orbit-position error against
  the declared numerical reference for the stated scenario set and duration.
- **RIC errors [km]**: radial, along-track, and cross-track components; radial
  maps most directly to altitude, along-track to phase/timing drift, and
  cross-track to plane error.
- **Runtime [s], steps/s, and speedup**: hardware- and backend-specific
  throughput; compare only runs with the same scenario count, duration,
  integrator, dtype, and output cadence.
- **Reference model**: a high-degree SH DOP853 run is a numerical reference, not
  physical truth; claims must name the reference and configuration.

See [docs/BENCHMARK_RESULTS.md](docs/BENCHMARK_RESULTS.md) for the reporting
contract and regeneration command pattern.

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
artifacts, checkpoints, plots, or evaluation tables. The one deliberate
exception is the offline web preview's static demo assets under
`src/lunaris/ui/web/public/` (Moon textures and a precomputed demo
`orbit-data.json`) — these are display-only inputs for the optional Three.js
preview, not scientific outputs, and the allowlist is enforced by
`tests/test_repo_hygiene.py`. The standard layout
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
