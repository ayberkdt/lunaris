# Lunaris

**Lunar orbit propagation and gravity-modeling framework.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Lunaris is a Python framework for lunar-orbit propagation and gravity modeling. It
bundles spherical-harmonic lunar gravity, configurable physical force models, orbit
propagation, batch/ensemble injection dispersion analysis, validation harnesses,
visualization tools, and a PySide6 desktop UI.

It also ships **ST-LRPS** (Sobolev-Trained Lunar Residual Potential Surrogate), a
neural surrogate-gravity model under `lunaris.surrogate.st_lrps` that learns a
residual scalar potential above a lower-degree spherical-harmonic baseline, with its
own training, evaluation, and Studio UI.

> **ST-LRPS is a research preview.** No accuracy, speed, spatial-generalization,
> or orbit-stability claim is made for surrogate results in this release: the
> full paper-safe evidence chain (spatial-block split, low/high-altitude OOD,
> A0–A6 ablations, curl/energy diagnostics, complete benchmark artifacts) has
> not yet been produced, and the existing out-of-distribution report is a
> *negative* extrapolation finding. Use the classical spherical-harmonic engine
> for production numbers; treat surrogate output as experimental until a
> validated evidence package accompanies it. See
> [docs/ST_LRPS_VALIDATION_HYGIENE.md](docs/ST_LRPS_VALIDATION_HYGIENE.md) and
> [docs/PAPER_SAFE_POLICY.md](docs/PAPER_SAFE_POLICY.md).
>
> **CUDA backends are experimental** in this release: there is currently no
> recurring GPU CI (the nightly CUDA workflow requires a self-hosted GPU runner),
> so GPU paths are re-validated manually per release rather than continuously.
> CPU results are the reference; GPU runs record requested-vs-actual backend
> provenance and fall back to CPU with a recorded reason.

> **ST-LRPS is a high-throughput *batch* gravity backend, not a low-latency
> single-trajectory CPU replacement.** A single trajectory run through
> `propagate()` evaluates the surrogate as an interpreted PyTorch + autograd
> closure (not a Numba kernel), so it pays per-call Python/autograd overhead on
> every RHS evaluation and will be *slower* than the `@njit` spherical-harmonic
> kernel. The surrogate's advantage is amortized only across a large GPU
> batch/ensemble. Do **not** benchmark "ST-LRPS vs SH" by timing one
> CPU trajectory — that measures the wrong path. Compare like-for-like on the GPU
> batch backend. See [docs/profiling.md](docs/profiling.md).
>
> **GPU ST-LRPS supports two backend variants:**
>
> - `gpu_st_lrps_potential`: surrogate lunar gravity only.
> - `gpu_st_lrps_third_body`: surrogate lunar gravity **plus analytic vectorized
>   Sun/Earth third-body** (Battin F(q) formulation).
>
> SRP, albedo, thermal IR, solid tides, relativity, and Earth J2 remain
> unsupported on GPU ST-LRPS and trigger a recorded CPU fallback or a hard
> error depending on the strict/backend policy — never a silent simplification;
> see [docs/backend_matrix.md](docs/backend_matrix.md). Non-gravity
> perturbations are handled separately in validation or future hybrid backends,
> and gravity-only results are never mixed with full-dynamics results in a
> single benchmark table.

> **Project status.** Lunaris is **actively developed research software** in a
> **0.1.x pre-release (release-candidate) line** — no stable 1.0 API promise is
> made yet; [docs/VERSIONING.md](docs/VERSIONING.md) governs what each release
> may claim. Maturity differs by subsystem: the classical spherical-harmonic
> engine carries the validation evidence documented in `docs/`, while ST-LRPS
> is a research preview (see above). Lunaris ships with
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
| [docs/README.md](docs/README.md) | Documentation index, including development notes and subsystem guides |
| [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) | Enterprise/offline deployment: air-gapped install, data mirroring, proxies, logging, privacy ("no telemetry"), security boundaries |
| [docs/VERSIONING.md](docs/VERSIONING.md) | Version scheme, stable surfaces, artifact-schema compatibility, deprecation and support policy |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Layered design, data flow, configuration model, **force-model / perturbation flags**, batch/ensemble propagation internals, ST-LRPS surrogate |
| [docs/MISSION_MONITOR.md](docs/MISSION_MONITOR.md) | Live probes, accepted/output-state replay semantics, compatibility policy, telemetry diagnostics, and widgets |
| [docs/ST_LRPS_VALIDATION_HYGIENE.md](docs/ST_LRPS_VALIDATION_HYGIENE.md) | Train-only scalers, spatial/OOD split policies, runtime frame safety, paper-safe benchmarks, validation + ablation suites |
| [docs/BENCHMARK_RESULTS.md](docs/BENCHMARK_RESULTS.md) | Full gravity-model benchmark tables and reproduction steps |
| [docs/REPRODUCIBLE_BENCHMARKS.md](docs/REPRODUCIBLE_BENCHMARKS.md) | Config-driven benchmark runs, provenance manifests, validation reports, and CI smoke mode |
| [docs/PAPER_SAFE_POLICY.md](docs/PAPER_SAFE_POLICY.md) | Paper-safe vs research-mode failure policy: what hard-fails, what warns-and-records |
| [docs/DATASET_PIPELINE.md](docs/DATASET_PIPELINE.md) | ST-LRPS dataset contract, validation, quality reports, split manifests, and strict training ingestion |
| [docs/CONFIG_AND_ARTIFACT_CONTRACTS.md](docs/CONFIG_AND_ARTIFACT_CONTRACTS.md) | ST-LRPS dataset, training, checkpoint, runtime, and benchmark contract rules |
| [docs/PERTURBATION_BUDGET.md](docs/PERTURBATION_BUDGET.md) | Perturbation-budget assumptions, outputs, and interpretation |
| [docs/UQ_COVARIANCE.md](docs/UQ_COVARIANCE.md) | Ensemble uncertainty quantification: covariance definition, RIC uncertainty, error ellipsoids, provenance-stamped UQ reports, linear (STM) cross-check |
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

**Supported platforms.** The matrix below reflects where Lunaris is actually
developed and tested, not aspiration:

| Platform | Status |
|---|---|
| Windows 10/11 (x86-64) | Supported — primary development platform; Python 3.11 core/wheel smoke CI |
| Ubuntu LTS (x86-64) | Supported — full CI (tests, lint, type, architecture gates) runs here headless |
| Other Linux distros | Expected to work (pure-Python + wheels); not routinely tested |
| macOS | **Untested** — no CI, no manual validation; try at your own risk |

Python 3.10–3.12 (CI matrix). PyTorch installs as the **CPU wheel by default**;
for CUDA use install the matching GPU build *before* the extras, e.g.:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

(CUDA backends are experimental in this release — see the note above.)

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
the [HPC guide](docs/HPC.md). The frozen recipe for reproducing the successful
May 2026 `resume_denemesi` ST-LRPS run is
[`hpc/scenarios/st_lrps_resume_denemesi_reproduction.jsonl`](hpc/scenarios/st_lrps_resume_denemesi_reproduction.jsonl).

Large mission data (SPICE kernels, gravity coefficients, topography, albedo) is
**not bundled**. Fetch and verify it with the headless `lunaris-data` tool into
`LUNARIS_DATA_DIR` (or the repo `data/` folder):

```bash
lunaris-data list
lunaris-data download --preset full-gravity
lunaris-data verify --strict --runtime
```

The `full-gravity` preset bundles the ephemeris and gravity assets that strict
runtime verification requires — including `gm_de440.tpc`, which is optional for a
plain `download --group ephemeris` but mandatory under `--strict`. If you prefer
to fetch by group, include the optional entries so strict verify passes:

```bash
lunaris-data download --group ephemeris --include-optional
lunaris-data download --group gravity
lunaris-data verify --strict --runtime
```

The catalogue is `data/data_sources.json`. Official-provider entries download
from NAIF/JPL or NASA PDS, recorded hashes are checked by `lunaris-data verify`,
and `--strict --runtime` also proves the resolved SPICE kernels can build a small
ephemeris table. Common locations: `data/ephemeris_models/`,
`data/gravity_models/`, `data/topography_models/`, `data/albedo_models/`,
`data/thermal_models/`, and `data/assets/`. ST-LRPS cloud suites are generated
artifacts under `data/datasets/`, not external downloads.

> **Installed / wheel use:** the repo `data/` fallback only resolves inside an
> editable checkout (it is found via the project-root marker). If you install
> Lunaris as a wheel/package, set `LUNARIS_DATA_DIR` to your data root — otherwise
> the CLI can also take an explicit `--kernel-dir` / `--gravity-file-path`, which
> are now honored even when no default `data/` layout is present.

## Quickstart

These checks do not require private local datasets:

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
hpc/               example Slurm templates for cluster use
```

Console entry points (installed via `pip install -e .`):

```text
lunaris           single-run propagation CLI
lunaris-batch     batch/ensemble propagation runner
lunaris-launcher  welcome hub (picks a workspace; optional offline 3D Moon preview)
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

At runtime ST-LRPS is a single artifact contract: `potential_autograd` (learned
scalar residual potential, acceleration via autograd; conservative by
construction and validation-safe). The earlier experimental `force_direct`
(3-output direct residual acceleration, not conservative by construction) has
been archived in the `experimental/force-direct-archive` branch and is
fail-closed on main. Versioned `artifact_contract` / `dataset_contract` blocks
record target semantics, baseline degree, altitude envelope, scaler contract,
encoding, and runtime kind.

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

Single-run propagation is driven by `lunaris`; propagated ensembles are driven
by the primary `lunaris-batch` command. Each ensemble run declares its sampling
design: `random` is the classical Monte Carlo design, while `lhs`, `sobol`, and
`sobol_scrambled` provide space-filling designs for validation and benchmark
coverage.

Mission Monitor replay is based on the exact solver-returned output grid, not
adaptive Runge–Kutta stage evaluations. Live cadence-gated RHS observations are
explicit `rhs_probe` samples and are excluded from `telemetry.ndjson` and all
scientific trajectory widgets. Old telemetry without sample semantics remains
decodable but is labelled uncertain; see [docs/MISSION_MONITOR.md](docs/MISSION_MONITOR.md).

Batch backends are explicit (`cpu_sh` truth reference, `numba_cuda_sh`,
`torch_cuda_sh`, `torch_cpu_sh`, `gpu_st_lrps_potential`,
`gpu_st_lrps_third_body`). Selection is resolved centrally by
`lunaris.batch.backend_policy`, and the requested vs. effective backend,
device, integrator, sampling method, and any fallback reason are recorded in the
batch result diagnostics rather than applied silently. The perturbation budget
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
(`outputs/{simulations,ensemble,missions,gravity_benchmark,training,evaluations,runtime,dataset_reports,datasets,validation,visualization}/`)
keeps a trained run's checkpoints, plots, evals, and provenance together.
Batch/ensemble outputs live under `outputs/ensemble/`.

## Testing

```bash
pytest tests/                       # full suite
pytest tests/test_repo_hygiene.py   # lightweight docs/hygiene checks
```

CI enforces a coverage baseline and uploads HTML coverage reports as artifacts.
See [CONTRIBUTING.md](CONTRIBUTING.md) for the full quality gates (`ruff`,
`mypy`, `lint-imports`, test markers).

## License

MIT License. See [LICENSE](LICENSE) for details. Third-party dependency and
data-asset license obligations (including the PySide6 LGPL notice for UI
deployments) are inventoried in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
