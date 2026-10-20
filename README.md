<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/lunaris-banner-dark.svg">
    <img src="docs/assets/lunaris-banner-light.svg" alt="Lunaris: lunar orbit propagation and gravity modeling with the ST-LRPS neural surrogate" width="100%">
  </picture>
</p>

# Lunaris

**Lunar orbit propagation and gravity-modeling framework.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**[Quickstart](#quickstart)** · [Documentation](docs/README.md) ·
[HPC guide](docs/HPC.md) · [Architecture](docs/ARCHITECTURE.md) ·
[Changelog](CHANGELOG.md)

## What Lunaris does

Lunaris is a Python framework for lunar orbits. It has these parts:

- Spherical-harmonic lunar gravity.
- Configurable physical force models.
- Orbit propagation with adaptive and fixed-step integrators.
- Batch/ensemble propagation and injection dispersion analysis.
- Validation harnesses and visualization tools.
- A PySide6 desktop UI (Lunaris Mission Studio).

Lunaris also contains **ST-LRPS** (Sobolev-Trained Lunar Residual Potential
Surrogate). ST-LRPS is a neural surrogate-gravity model in
`lunaris.surrogate.st_lrps`. It learns a residual scalar potential above a
lower-degree spherical-harmonic baseline. It has its own training, evaluation
and Studio UI.

## Status and limits

Read these notes before you use the results.

> **ST-LRPS is a research preview.** This release makes no accuracy, speed,
> spatial-generalization or orbit-stability claim for surrogate results. The
> full paper-safe evidence chain is not complete yet: spatial-block split,
> low/high-altitude OOD, A0–A6 ablations, curl/energy diagnostics and complete
> benchmark artifacts. The current out-of-distribution report is a *negative*
> extrapolation finding. Use the classical spherical-harmonic engine for
> production numbers. See
> [docs/ST_LRPS_VALIDATION_HYGIENE.md](docs/ST_LRPS_VALIDATION_HYGIENE.md) and
> [docs/PAPER_SAFE_POLICY.md](docs/PAPER_SAFE_POLICY.md).

> **CUDA backends are experimental.** There is no recurring GPU CI: the nightly
> CUDA workflow needs a self-hosted GPU runner. We validate the GPU paths
> manually for each release. CPU results are the reference. GPU runs record the
> requested and the actual backend. If a GPU path cannot run, the run falls
> back to the CPU and records the reason.

> **ST-LRPS is a batch gravity backend.** It is not a low-latency replacement
> for one CPU trajectory. One trajectory through `propagate()` evaluates the
> surrogate as an interpreted PyTorch + autograd closure. This path is *slower*
> than the `@njit` spherical-harmonic kernel. The surrogate is faster only
> across a large GPU batch. Do **not** compare "ST-LRPS vs SH" with one CPU
> trajectory. Compare the two on the GPU batch backend. See
> [docs/profiling.md](docs/profiling.md).
>
> GPU ST-LRPS has two backend variants:
>
> - `gpu_st_lrps_potential`: surrogate lunar gravity only.
> - `gpu_st_lrps_third_body`: surrogate lunar gravity **plus analytic vectorized
>   Sun/Earth third-body** (Battin F(q) formulation).
>
> GPU ST-LRPS does not support SRP, albedo, thermal IR, solid tides, relativity
> or Earth J2. A run that needs them falls back to the CPU and records the
> reason, or stops with an error under the strict backend policy. The run never
> simplifies the force model silently. See
> [docs/backend_matrix.md](docs/backend_matrix.md). A benchmark table never
> mixes gravity-only results with full-dynamics results.

> **Project status.** Lunaris is research software in active development. It
> is in a **0.1.x pre-release (release-candidate) line**. There is no stable
> 1.0 API yet. [docs/VERSIONING.md](docs/VERSIONING.md) sets what each release
> can claim. The classical spherical-harmonic engine has the validation
> evidence in `docs/`. ST-LRPS is a research preview. Datasets, checkpoints,
> runtime and benchmark artifacts have versioned on-disk contracts. Public APIs
> can change between releases, so pin a version for reproducible work. A
> trained artifact or a benchmark number applies only to its run and
> configuration.

## Documentation

This README is a landing page. The full detail is in `docs/`.

| Document | Contents |
|----------|----------|
| [docs/README.md](docs/README.md) | Documentation index, development notes and subsystem guides |
| [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) | Offline deployment: air-gapped install, data mirrors, proxies, logging, privacy ("no telemetry"), security boundaries |
| [docs/VERSIONING.md](docs/VERSIONING.md) | Version scheme, stable surfaces, artifact-schema compatibility, deprecation and support policy |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Layers, data flow, configuration model, **force-model / perturbation flags**, batch/ensemble internals, ST-LRPS surrogate |
| [docs/MISSION_MONITOR.md](docs/MISSION_MONITOR.md) | Live probes, replay of accepted/output states, compatibility policy, telemetry diagnostics, widgets |
| [docs/ST_LRPS_VALIDATION_HYGIENE.md](docs/ST_LRPS_VALIDATION_HYGIENE.md) | Train-only scalers, spatial/OOD split policies, runtime frame safety, paper-safe benchmarks, validation and ablation suites |
| [docs/BENCHMARK_RESULTS.md](docs/BENCHMARK_RESULTS.md) | Gravity-model benchmark tables and reproduction steps |
| [docs/REPRODUCIBLE_BENCHMARKS.md](docs/REPRODUCIBLE_BENCHMARKS.md) | Config-driven benchmark runs, provenance manifests, validation reports, CI smoke mode |
| [docs/PAPER_SAFE_POLICY.md](docs/PAPER_SAFE_POLICY.md) | Paper-safe and research-mode failure policy: what stops a run, what gives a recorded warning |
| [docs/DATASET_PIPELINE.md](docs/DATASET_PIPELINE.md) | ST-LRPS dataset contract, validation, quality reports, split manifests, strict training input |
| [docs/CONFIG_AND_ARTIFACT_CONTRACTS.md](docs/CONFIG_AND_ARTIFACT_CONTRACTS.md) | ST-LRPS dataset, training, checkpoint, runtime and benchmark contract rules |
| [docs/PERTURBATION_BUDGET.md](docs/PERTURBATION_BUDGET.md) | Perturbation-budget assumptions, outputs and interpretation |
| [docs/UQ_COVARIANCE.md](docs/UQ_COVARIANCE.md) | Ensemble uncertainty: covariance definition, RIC uncertainty, error ellipsoids, provenance-stamped UQ reports, linear (STM) cross-check |
| [docs/HPC.md](docs/HPC.md) | Cluster/headless install, environment setup, Slurm templates, scenario arrays, exit codes and resume |
| [docs/profiling.md](docs/profiling.md) | ST-LRPS runtime profiling and timing interpretation |
| [validation/README.md](validation/README.md) | Independent physics/orbit/gravity validation harnesses |
| [CONTRIBUTING.md](CONTRIBUTING.md) / [SECURITY.md](SECURITY.md) | Development setup and quality gates / vulnerability reports |

## Force models

The propagator (`lunaris.core.dynamics`) applies these force models:

- Spherical-harmonic lunar gravity (and the ST-LRPS surrogate-gravity model)
- Third-body perturbations (Sun, Earth)
- Earth oblateness (differential J2)
- Solar radiation pressure, with eclipses. The adaptive integrator restarts at
  each shadow boundary.
- Lunar albedo (reflected sunlight) surface radiation
- Lunar thermal IR radiation pressure
- Elastic lunar solid-body tides (`k2`, optional explicit `k3`; raised by the
  Earth and/or the Sun)
- First-order post-Newtonian relativity

You can set each model from the `lunaris` CLI. For the full flag reference,
assumptions and limits, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
Examples:

```bash
lunaris --enable-tides on --tides-kind k2 --tide-bodies earth,sun
lunaris --enable-thermal on --thermal-mode equilibrium_temperature
lunaris --enable-albedo on --albedo-mode scaled_dn_grid --albedo-root data/albedo_models
```

## Installation

**Supported platforms.** The table shows where we develop and test Lunaris:

| Platform | Status |
|---|---|
| Windows 10/11 (x86-64) | Supported. Primary development platform; Python 3.11 core/wheel smoke CI |
| Ubuntu LTS (x86-64) | Supported. Full CI (tests, lint, type, architecture gates) runs here headless |
| Other Linux distributions | Expected to work (pure Python and wheels); not tested routinely |
| macOS | **Untested.** No CI and no manual validation |

Lunaris supports Python 3.10–3.12 (CI matrix). PyTorch installs as the **CPU
wheel by default**. For CUDA, install the matching GPU build *before* the
extras. Example:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Install Lunaris in editable mode from the repository root. This step installs
the console entry points. Code changes then apply without a new install.

```bash
python -m pip install -e .            # core dependencies only
python -m pip install -e ".[all]"     # core + ML + UI + reports + dev extras
```

`pyproject.toml` has these optional dependency groups: `ml`, `hpc`, `ui`,
`reports`, `dev`, `all`. Each dependency has a lower and an upper version
bound. Thus a build cannot pull in an incompatible major release. For
reproducible environments, use the exact pins in `locks/*.lock.txt`
(generated with `uv`). For clusters, see
[Train ST-LRPS on a Slurm cluster](#train-st-lrps-on-a-slurm-cluster).

Large mission data (SPICE kernels, gravity coefficients, topography, albedo)
is **not bundled**. Use the headless `lunaris-data` tool to download and verify
it into `LUNARIS_DATA_DIR` (or the repository `data/` folder):

```bash
lunaris-data list
lunaris-data download --preset full-gravity
lunaris-data verify --strict --runtime
```

The `full-gravity` preset contains all ephemeris and gravity assets that strict
runtime verification needs, including `gm_de440.tpc`. A plain
`download --group ephemeris` does not include `gm_de440.tpc`, but `--strict`
needs it. To download by group, include the optional entries:

```bash
lunaris-data download --group ephemeris --include-optional
lunaris-data download --group gravity
lunaris-data verify --strict --runtime
```

The catalogue is `data/data_sources.json`. Official entries download from
NAIF/JPL or NASA PDS. `lunaris-data verify` checks the recorded hashes.
`--strict --runtime` also builds a small ephemeris table from the resolved
SPICE kernels. The usual locations are `data/ephemeris_models/`,
`data/gravity_models/`, `data/topography_models/`, `data/albedo_models/`,
`data/thermal_models/` and `data/assets/`. ST-LRPS cloud suites are generated
artifacts in `data/datasets/`; they are not downloads.

> **Installed / wheel use:** The repository `data/` fallback works only in an
> editable checkout (the project-root marker finds it). If you install Lunaris
> as a wheel, set `LUNARIS_DATA_DIR` to your data root. You can also give an
> explicit `--kernel-dir` / `--gravity-file-path` to the CLI.

## Quickstart

These checks need no local datasets:

```bash
python -m pip install -e ".[hpc]"
python -c "import lunaris; print(lunaris.__version__)"
lunaris-train --help
lunaris-eval --help
lunaris-benchmark --help
lunaris-batch --help
lunaris-perturbation-budget --help
python -m lunaris.surrogate.st_lrps.training.cli --help
python -m lunaris.surrogate.st_lrps.evaluation.cli --help
python -m lunaris.surrogate.st_lrps.evaluation.compare_gravity_models --help
python -m lunaris.visualization.surface_explorer --help
```

Full propagation, ST-LRPS training, gravity validation and topography plots
need local gravity, SPICE or LOLA files.

## Repository architecture

The package uses a `src/` layout with **four strict layers**. A layer never
imports from a layer above it. Dependency-light support packages sit beside the
layers. For the full design, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

```text
src/lunaris/
  common/          [layer 1] shared constants, config dataclasses, math/time helpers
  physics/         [layer 2] Numba force-model kernels, ephemeris (SPICE), gravity adapters
  core/            [layer 3] config (SimConfig SSOT), dynamics RHS, propagator, events, batch ensembles
  analysis/        [layer 4] post-processing, reports, ensemble injection dispersion analysis
  visualization/   [layer 4] standalone visualization tools
  ui/              [layer 4] Lunar Orbit Simulator desktop UI (PySide6)
  loaders/         (support) gravity / topography / ephemeris / data loading
  cli/             (support) console entry points and shared CLI helpers
  surrogate/st_lrps/   Sobolev-Trained Lunar Residual Potential Surrogate family
      data/ training/ networks/ artifacts/ evaluation/ runtime/ shared/ ui/
validation/        independent physics/orbit/gravity validation harnesses + docs
tests/             unit and regression tests
data/              local input data (SPICE kernels, gravity, topography)
hpc/               Slurm templates and helper scripts for cluster use
```

Console entry points (installed by `pip install -e .`):

```text
lunaris           single-run propagation CLI
lunaris-batch     batch/ensemble propagation runner
lunaris-launcher  welcome hub (selects a workspace; optional offline 3D Moon preview)
lunaris-ui        mission desktop UI (Lunaris Mission Studio)
lunaris-studio    ST-LRPS Studio UI
lunaris-train     ST-LRPS training CLI (potential_autograd surrogate)
lunaris-eval      ST-LRPS evaluation CLI
lunaris-benchmark ST-LRPS orbit-level gravity benchmark / validation CLI
lunaris-validate  gravity-reference checks + ST-LRPS validation suite
lunaris-ablation  ST-LRPS ablation suite runner
lunaris-st-lrps-paper-evidence  end-to-end paper-evidence pipeline
lunaris-data      external-data download / verify CLI
lunaris-perturbation-budget   acceleration / force-model uncertainty budget
```

[docs/PUBLIC_API.md](docs/PUBLIC_API.md) lists each entry point and its
stability class.

The desktop UI uses one dark theme, **Lunar Graphite**. Its tokens come from
`lunaris.ui_foundation` (see [docs/UI_THEME.md](docs/UI_THEME.md)). The
launcher can show an optional, **offline** Three.js Moon preview (built from
`src/lunaris/ui/web`). The preview never blocks the app. If the preview is not
available, the launcher works without it.

## ST-LRPS at a glance

```bash
# dataset generation, training, evaluation, ablation
python -m lunaris.surrogate.st_lrps.data.spatial_cloud_generator --help
python -m lunaris.surrogate.st_lrps.training.cli --help
python -m lunaris.surrogate.st_lrps.evaluation.cli --help
python -m lunaris.surrogate.st_lrps.evaluation.ablation --help
```

At runtime, ST-LRPS has one artifact contract: `potential_autograd`. The network
learns a scalar residual potential. Autograd gives the acceleration, so the
force is conservative by construction. The earlier experimental `force_direct`
model (3-output direct residual acceleration, not conservative) is archived in
the `experimental/force-direct-archive` branch. On main it stops with an error.
Versioned `artifact_contract` / `dataset_contract` blocks record the target
semantics, baseline degree, altitude envelope, scaler contract, encoding and
runtime kind.

Select the run posture with `--run-preset {development,quick,paper}`. The
`paper` preset enforces a generalization split, deterministic execution and the
preflight gate. Reproducible config-driven benchmarks:

```bash
lunaris-benchmark --config configs/benchmarks/st_lrps_1day_high_degree.json
lunaris-benchmark --config configs/benchmarks/st_lrps_1day_high_degree.json --quick
```

For contracts, validation hygiene and reproducible HPC sweeps, see
[docs/CONFIG_AND_ARTIFACT_CONTRACTS.md](docs/CONFIG_AND_ARTIFACT_CONTRACTS.md),
[docs/ST_LRPS_VALIDATION_HYGIENE.md](docs/ST_LRPS_VALIDATION_HYGIENE.md) and
[docs/HPC.md](docs/HPC.md#1b-st-lrps-scenario-arrays-reproducible-sweeps).

### Exit codes

`lunaris-train` gives these exit codes:

| Exit | Meaning |
|------|---------|
| `0` | The run completed. |
| `75` | A stop signal stopped the run after it saved `ckpt_last.pt`. You can resume the run. |
| `1` | The run failed (NaN/Inf loss, stopped periodic evaluation) or an error occurred. |
| `130` | A second stop signal forced the stop. The current epoch was not saved. |

A second stop signal less than 5 s after the first has no effect. Slurm and the
job script can send the same stop two times.

### Resume training

The trainer saves a checkpoint after each epoch. Resume restores the optimizer,
LR schedule, GradNorm and RNG state, not only the weights.

1. Use the resume command that the trainer prints when it stops. The run
   manifest also records this command (`resume_hint`).
2. Keep all original flags in the command. Add `--resume-from <run-dir>`.
3. Set `--epochs` to the **total** epoch count, not to the additional epochs.

Do not use `--resume-from` alone. Resume locks only the network structure and
the scaler. A bare `--resume-from` sets the batch size, the learning rate and
the loss weights to their default values. The trainer then writes a warning
that lists each setting that changed. A bare resume restores the data files of
the run, including separate `--train-data` / `--val-data` files.

For the full semantics, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Train ST-LRPS on a Slurm cluster

The `hpc/` folder has Slurm templates and helper scripts. A profile for the
UHeM Altay cluster is in `hpc/clusters/`. Do these steps one time for each
cluster, then for each run.

1. Copy the site configuration and fill it in. Keep LF line endings.

   ```bash
   cp hpc/clusters/uhem_altay.env.example hpc/cluster.env
   ```

2. Build the headless environment on a login node. The script needs CPython
   3.11. It installs the hash-pinned HPC lock and then the package.

   ```bash
   bash hpc/setup_env.sh
   ```

3. Start a short interactive GPU job. Run the preflight check in it.

   ```bash
   srun -p a100q --gres=gpu:1 --cpus-per-task=4 --time=00:10:00 --pty bash
   bash hpc/preflight.sh
   ```

4. Make sure that the output shows `PREFLIGHT VERDICT: PASS`. The locked
   PyTorch build needs NVIDIA driver 580 or higher on the GPU nodes.

5. Submit the training job. Give an explicit `--out-dir`.

   ```bash
   hpc/submit.sh train -- \
       --data "$LUNARIS_DATA_DIR/datasets/st_lrps_cloud_suite.h5" \
       --epochs 300 --batch-size 8192 \
       --out-dir "$LUNARIS_OUTPUT_DIR/training/run1"
   ```

6. If the job stops with exit code 75, use the resume command in the job log.
   To requeue the job automatically, set `LUNARIS_AUTO_REQUEUE=1` in
   `hpc/cluster.env`.

The job sends a stop signal to the trainer 30 minutes before the time limit.
This time must be longer than one epoch plus its validation. For longer epochs,
set `LUNARIS_SIGNAL="B:TERM@3600"` in `hpc/cluster.env`. For scenario arrays,
dataset generation and batch jobs, see [docs/HPC.md](docs/HPC.md#running-on-slurm).
The frozen recipe of the successful May 2026 `resume_denemesi` ST-LRPS run is
[`hpc/scenarios/st_lrps_resume_denemesi_reproduction.jsonl`](hpc/scenarios/st_lrps_resume_denemesi_reproduction.jsonl).

## Propagation, Batch Ensembles, And Analysis

`lunaris` runs one propagation. The primary `lunaris-batch` command runs
propagated ensembles. Each ensemble run declares its sampling design. `random`
is the classical Monte Carlo design. `lhs`, `sobol` and `sobol_scrambled` are
space-filling designs for validation and benchmark coverage.

Mission Monitor replay uses the exact output grid that the solver returns. It
does not use adaptive Runge–Kutta stage evaluations. Live RHS observations are
explicit `rhs_probe` samples. `telemetry.ndjson` and the scientific trajectory
widgets do not include them. Old telemetry without sample semantics stays
readable, but it has an "uncertain" label. See
[docs/MISSION_MONITOR.md](docs/MISSION_MONITOR.md).

The batch backends are explicit: `cpu_sh` (truth reference), `numba_cuda_sh`,
`torch_cuda_sh`, `torch_cpu_sh`, `gpu_st_lrps_potential` and
`gpu_st_lrps_third_body`. `lunaris.batch.backend_policy` selects the backend.
The batch result records the requested and the effective backend, device,
integrator, sampling method and each fallback reason. No fallback occurs
silently. The perturbation budget tool shows acceleration contributions and
force-model uncertainty:

```bash
lunaris-perturbation-budget --altitudes-km 50,100,300,1000 --sh-degrees 20,60,200 \
  --gravity-model path/to/lunar_gravity_model.tab --out-dir outputs/perturbation_budget/default
```

`lunaris-frozen-search` runs a frozen-orbit campaign in stages:

1. Element samples.
2. A batched screening pass (ST-LRPS or torch SH, optionally with Sun/Earth
   third-body terms).
3. A scored shortlist.
4. Classical-SH validation of the shortlist.
5. A family report.

Each stage writes its own artifact (`stage0_samples.npz` …
`stage4_families.json`). Screening saves a checkpoint every
`--screening-checkpoint-minutes` (default 20). `--resume` continues an
interrupted run from the last checkpoint with bitwise-identical trajectories.
`FrozenClassificationConfig.for_mission_duration` sets the classification
thresholds. Each rate limit, including the inclination rate, is its envelope
limit divided by the arc length. The labels describe one simulated arc. They
are not a universal definition of a frozen orbit.

Post-processing, reports and ensemble statistics/plots are in
`lunaris.analysis.*`. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Validation and benchmarks

The validation layer has independent physics, orbit and cross-model checks.
External-reference harnesses are in `validation/independent/`: `scipy` and
`pyshtools` SH cross-checks and direct NAIF/SPICE ephemeris checks. The gravity
benchmark CLI compares ST-LRPS and classical spherical-harmonic runs with a
declared numerical reference:

```bash
python -m lunaris.surrogate.st_lrps.evaluation.compare_gravity_models --help
```

This README does not publish a paper-safe ST-LRPS orbit-accuracy table yet. You
can cite a benchmark only when its `validation_report.json` passes. Its
`benchmark_manifest.json` must record the resolved config, commit, scenario
seed/count, numerical reference, hardware, backend, model artifact,
dataset/gravity hashes (when available) and ST-LRPS artifact-contract
compatibility.

How to read a gravity benchmark:

- **Median / P95 / max RMS position error [km]**: orbit-position error against
  the declared numerical reference, for the stated scenario set and duration.
- **RIC errors [km]**: radial, along-track and cross-track components. Radial
  error relates to altitude. Along-track error relates to phase/timing drift.
  Cross-track error relates to plane error.
- **Runtime [s], steps/s and speedup**: these depend on the hardware and the
  backend. Compare only runs with the same scenario count, duration,
  integrator, dtype and output cadence.
- **Reference model**: a high-degree SH DOP853 run is a numerical reference,
  not physical truth. Each claim must name the reference and the configuration.

For the reporting contract and the regeneration commands, see
[docs/BENCHMARK_RESULTS.md](docs/BENCHMARK_RESULTS.md).

## Visualization

The standalone tools are in `src/lunaris/visualization/`. For orbit
animation, use `lunaris.visualization.orbit_animation.render_orbit_animation`.
For topography and albedo, use `lunaris.visualization.surface_explorer`:

```bash
python -m lunaris.visualization.surface_explorer \
    --topo-label data/topography_models/ldem_64_float.lbl \
    --topo-img data/topography_models/ldem_64_float.img \
    --out-dir outputs/surface_explorer --plot-2d --plot-3d
```

Large LOLA grids use much memory. For quick previews, use
`--stride-2d`/`--stride-3d`/`--stride-albedo`.

## Generated output policy

Do not commit generated products. The tools write to `outputs/` (git-ignored)
unless you select an external scratch folder. The source packages
(`src/lunaris/...`, `validation/`) contain only source and documentation. They
do not contain run artifacts, checkpoints, plots or evaluation tables.

There is one exception: the static demo assets of the offline web preview in
`src/lunaris/ui/web/public/` (Moon textures and a precomputed demo
`orbit-data.json`). These are display-only inputs for the optional Three.js
preview. They are not scientific outputs. `tests/test_repo_hygiene.py` enforces
the allowlist.

The standard layout keeps the checkpoints, plots, evaluations and provenance of
a trained run together:
`outputs/{simulations,ensemble,missions,gravity_benchmark,training,evaluations,runtime,dataset_reports,datasets,validation,visualization}/`.
Batch/ensemble outputs are in `outputs/ensemble/`.

## Testing

```bash
pytest tests/                       # full suite
pytest tests/test_repo_hygiene.py   # lightweight docs/hygiene checks
```

CI enforces a coverage baseline and uploads HTML coverage reports as
artifacts. For the full quality gates (`ruff`, `mypy`, `lint-imports`, test
markers), see [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT License. See [LICENSE](LICENSE). Third-party dependency and data-asset
license obligations are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
This includes the PySide6 LGPL notice for UI deployments.
