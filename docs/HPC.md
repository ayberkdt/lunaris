# HPC and Cluster Deployment

This guide outlines how to deploy and run Lunaris and its ST-LRPS workflows on
HPC clusters using Slurm.

[Lunaris](../README.md) is a lunar orbit propagation and gravity-modeling
framework. ST-LRPS (Sobolev-Trained Lunar Residual Potential Surrogate) is the
surrogate-gravity model/workflow inside Lunaris, under
`lunaris.surrogate.st_lrps`. ST-LRPS is the main HPC-heavy workflow in this
repository — not the desktop UI or single-orbit propagation. A typical cluster
session works through, in order:

1. **ST-LRPS dataset / spatial cloud generation**
2. **ST-LRPS training**
3. **ST-LRPS evaluation**
4. **Orbit-level gravity benchmark / validation**
5. **Batch propagation / injection dispersion ensembles**

> **Keep GUIs off compute nodes.** The desktop UI (`lunaris-ui`) and the ST-LRPS
> Studio (`lunaris-studio`) are interactive tools. Do not install or launch them
> on compute nodes. Install the GUI extras (`.[ui]`/`.[all]`: `PySide6`,
> `PyQt6`, `pyqtgraph`) only on a login or visualization node, and only if you
> actually need them there.

## Quick start

If you just want to get an ST-LRPS training job running, fill in **one** site
config file and use the helper scripts in `hpc/` (see
[`hpc/README.md`](../hpc/README.md)). The rest of this guide is the reference for
what those scripts do and the individual workflows.

```bash
# 1. Configure your cluster once: account, partition, modules, venv, storage.
cp hpc/cluster.env.example hpc/cluster.env
$EDITOR hpc/cluster.env

# 2. Build the headless environment once (creates a venv, installs .[hpc]).
bash hpc/setup_env.sh
#    then paste the printed LUNARIS_VENV=... line into hpc/cluster.env

# 3. Sanity-check, then submit. submit.sh injects partition/account/qos/gres
#    from cluster.env, so you never edit those into the .sbatch files.
bash hpc/preflight.sh
hpc/submit.sh train -- \
    --data "$LUNARIS_DATA_DIR/datasets/st_lrps_cloud_suite.h5" \
    --epochs 300 --batch-size 8192 \
    --out-dir "$LUNARIS_OUTPUT_DIR/training/run1"
```

`hpc/cluster.env` is git-ignored, so site-specific values never get committed.
Every `hpc/*.sbatch` job sources `hpc/env_template.sh`, which loads `cluster.env`,
runs the requested `module load`s, and activates your venv/conda env before the
headless entry point runs. The manual installation and per-workflow details
below still apply when you need finer control.

## Installation

The recommended setup registers the package and its console commands
(`lunaris-train`, `lunaris-eval`, `lunaris-benchmark`, `lunaris-batch`,
`lunaris-batch`, …) in an
isolated, GUI-free environment.

### Option A: pip / virtual environment (recommended)

```bash
git clone https://github.com/ayberkdt/lunaris.git
cd lunaris

python -m venv lunaris_env
source lunaris_env/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[hpc]"
```

### Option B: Conda

```bash
git clone https://github.com/ayberkdt/lunaris.git
cd lunaris

conda env create -f environment.yml
conda activate lunaris
python -m pip install -e ".[hpc]"
```

Edit `environment.yml` first to select the `pytorch-cuda` version that matches
the CUDA module on your cluster (e.g. `pytorch-cuda=12.1`).

The `.[hpc]` extra installs PyTorch + h5py on top of the core dependencies and
omits all GUI packages. `requirements_hpc.txt` is a convenience shortcut that
resolves to the same `.[hpc]` extra (version *ranges* from `pyproject.toml`), so
the preferred install remains the editable `python -m pip install -e ".[hpc]"` so
the console entry points are registered.

### Option C: Fully pinned, reproducible install (Paper-evidence runs)

For runs that must be byte-for-byte reproducible (paper benchmarks, audited HPC
jobs), install from the hash-verified lock instead of the version ranges:

```bash
pip install --require-hashes -r locks/requirements-hpc-linux-py311.lock.txt
pip install --no-deps -e .   # register console entry points
```

The lock pins every transitive dependency for **Linux + CPython 3.11** (the CI /
paper-evidence target). See [`locks/README.md`](../locks/README.md) for the CUDA
note and how to regenerate the locks after a dependency change.

Verify the headless entry points are available:

```bash
lunaris-train --help
lunaris-eval --help
lunaris-benchmark --help
lunaris-batch --help
```

## Data and Output Layout

Large mission/science data files are **not** tracked in Git or shipped in the
Python package. On a cluster, keep them on scratch/project storage and point the
framework at that location with `LUNARIS_DATA_DIR`. Generated outputs (training
runs, evaluations, benchmarks) should also live on scratch, not in the source
tree.

`hpc/env_template.sh` sets scratch defaults that the Slurm jobs source:

```bash
export LUNARIS_DATA_DIR="${LUNARIS_DATA_DIR:-/scratch/$USER/lunaris_data}"
export LUNARIS_OUTPUT_DIR="${LUNARIS_OUTPUT_DIR:-/scratch/$USER/lunaris_outputs}"
```

Recommended scratch layout:

```text
/scratch/$USER/lunaris_data/
  gravity_models/
  ephemeris_models/
  topography_models/
  datasets/

/scratch/$USER/lunaris_outputs/
  training/
  evaluations/
  gravity_benchmark/
  runtime/
  ensemble/
```

`LUNARIS_DATA_DIR` is read by the framework when locating external data;
`LUNARIS_OUTPUT_DIR` is a convenience the example jobs pass through to
`--out-dir`/`--output-dir`. Large spherical-harmonic / gravity coefficient files
(400 MB+), SPICE kernels, and topography grids should be stored **once** under
`LUNARIS_DATA_DIR`, not copied into each job folder and never committed to Git.

Inside the repository, external data uses these canonical directory names (the
same categories as the scratch layout):

```text
data/gravity_models/
data/ephemeris_models/
data/topography_models/
data/albedo_models/
data/thermal_models/
data/assets/
data/datasets/
```

## Acquiring External Data

Lunaris depends on large external files — lunar gravity coefficients, SPICE/
ephemeris kernels, LOLA/LDEM topography, and optional albedo grids — that are not
committed to Git or bundled in the package. Use the headless `lunaris-data` tool
to list, download, verify, and place them under `LUNARIS_DATA_DIR`:

```bash
export LUNARIS_DATA_DIR=/scratch/$USER/lunaris_data
lunaris-data list
lunaris-data download --group ephemeris
lunaris-data download --group gravity
lunaris-data verify
lunaris-data verify --strict --runtime
```

The catalogue lives in `data/data_sources.json`. Entries with an official URL
download directly from NAIF/JPL or NASA PDS, and entries with recorded SHA-256
values are hash-checked by `lunaris-data verify`. `--strict` promotes
strict-required assets such as `gm_de440.tpc`; `--runtime` additionally builds a
small SPICE ephemeris table to prove the resolved kernels are readable by the
runtime. `lunaris-data` resolves its data root as `--data-dir` ->
`LUNARIS_DATA_DIR` -> the repository `data/` folder, and writes into
`gravity_models/`, `ephemeris_models/`, `topography_models/`, `albedo_models/`,
`thermal_models/`, `assets/`, and `datasets/`.

ST-LRPS cloud-suite HDF5 files are generated artifacts, not external downloads.
When a job creates or consumes one, validate it with `lunaris-data validate`
before treating it as benchmark or training evidence.

Download large files **once** to shared scratch/project storage and let every job
reuse `LUNARIS_DATA_DIR`. Do not copy 400 MB+ gravity files into each run
directory, and do not commit downloaded data.

## Running on Slurm

Template batch scripts live in `hpc/`. Each one sources `hpc/env_template.sh`
and then calls a headless entry point, forwarding any extra arguments you pass
to `sbatch`. These are *templates*: open `hpc/env_template.sh` and the `.sbatch`
files and adapt the placeholders — partition/account names, module loads, the
environment activation, and the `#SBATCH` resource directives — to your cluster
before submitting.

| Workload | Script | Entry point |
|----------|--------|-------------|
| Shared environment setup | `hpc/env_template.sh` | sourced by each job |
| ST-LRPS training | `hpc/slurm_train_stlrps.sbatch` | `lunaris-train` |
| ST-LRPS scenario arrays (sweeps) | `hpc/slurm_train_scenario_array.sbatch` | `tools/hpc/run_training_scenario.py` |
| Orbit-level gravity benchmark / validation | `hpc/slurm_benchmark_gpu.sbatch` | `lunaris-benchmark` |
| Batch propagation / injection dispersion ensembles | `hpc/slurm_batch_array.sbatch` | `lunaris-batch` |

### 1. ST-LRPS training (primary workload)

Training is the main HPC job. Submit it with the training template; extra
arguments are forwarded to `lunaris-train`:

```bash
export LUNARIS_OUTPUT_DIR=/scratch/$USER/lunaris_outputs
RUN_NAME="st_lrps_train_$(date +%Y%m%d_%H%M%S)"

sbatch hpc/slurm_train_stlrps.sbatch \
  --out-dir "$LUNARIS_OUTPUT_DIR/training/$RUN_NAME" \
  --epochs 300 \
  --batch-size 8192
```

(`lunaris-train` is the `lunaris.surrogate.st_lrps.training.cli` entry point; run
`lunaris-train --help` for the full flag list.)

> **Submit-time vs. job-time variables.** `$SLURM_JOB_ID` exists only *inside* the
> running job, not reliably in the submit shell — putting it in the `sbatch`
> command above would expand to an empty string and create a malformed output
> path. Likewise, anything you expand before `sbatch` (such as
> `$LUNARIS_OUTPUT_DIR`) must already be set in the submit environment, so export
> it first. For reproducible run names use a timestamp (as above), a manual name,
> or a wrapper script. `hpc/env_template.sh` still sets defaults *inside* the job,
> but those defaults do not affect values you expand at submit time.

### 1b. ST-LRPS scenario arrays (reproducible sweeps)

For paper ablations and capacity / encoding / loss sweeps you usually want to
launch **many** clearly-named experiments in parallel rather than hand-writing a
long `lunaris-train` command for each one. The scenario launcher reads a JSONL
file where **each line is one self-describing experiment** and submits the whole
file as a Slurm array.

**Why JSONL scenario files exist.** Each scenario carries its own `name`,
`entrypoint`, `description`, `runtime_model_kind`, `tags`, and `flags`, so the
sweep is self-documenting and version-controlled. The `name` is intentionally
long and explicit (e.g.
`PotentialAutograd_A6FullRecommended_3Band_RawXYZ_DirectionW020_Seed42`) and is
reused verbatim as the run-directory name. JSONL is preferred over YAML to avoid
an extra parser dependency.

The committed sweep files live under `hpc/scenarios/`:

| File | Array range | Purpose |
|------|-------------|---------|
| `st_lrps_potential_autograd_paper_ablation_A0_to_A6.jsonl` | `0-6` | Cumulative scalar-potential ablation A0→A6 (mirrors `lunaris-ablation`) |
| `st_lrps_potential_autograd_capacity_sweep_A6_full.jsonl` | `0-4` | A6-full architecture-size sweep (4×256 … 5×768) |
| `st_lrps_potential_autograd_encoding_and_loss_sweep.jsonl` | `0-6` | 5×512 direction-weight + input-encoding matrix |
| `st_lrps_runtime_benchmark_smoke.jsonl` | `0-1` | CPU/CUDA timing smoke checks |

**How submission works.** The first positional argument to the `.sbatch` file is
the scenario JSONL path; **everything after it is forwarded to every array task**
as common flags (data paths, epochs, batch size, split policy …). The launcher
selects the line matching `$SLURM_ARRAY_TASK_ID`, validates it, and injects
`--out "$LUNARIS_OUTPUT_DIR/training/<scenario_name>"` itself — never set an
output flag in the scenario or in the common flags.

```bash
export TRAIN_DATA=/scratch/$USER/lunaris_data/datasets/train.h5
export VAL_DATA=/scratch/$USER/lunaris_data/datasets/val.h5
export TEST_DATA=/scratch/$USER/lunaris_data/datasets/test.h5
export OOD_DATA=/scratch/$USER/lunaris_data/datasets/ood_high.h5
export LUNARIS_OUTPUT_DIR=/scratch/$USER/lunaris_outputs

# A0–A6 paper ablation (7 tasks)
sbatch --array=0-6 hpc/slurm_train_scenario_array.sbatch \
  hpc/scenarios/st_lrps_potential_autograd_paper_ablation_A0_to_A6.jsonl \
  --train-data "$TRAIN_DATA" \
  --val-data "$VAL_DATA" \
  --test-data "$TEST_DATA" \
  --ood-data "$OOD_DATA" \
  --epochs 300 \
  --batch-size 8192 \
  --split-policy spatial_block
```

Capacity sweep (5 tasks):

```bash
sbatch --array=0-4 hpc/slurm_train_scenario_array.sbatch \
  hpc/scenarios/st_lrps_potential_autograd_capacity_sweep_A6_full.jsonl \
  --train-data "$TRAIN_DATA" \
  --val-data "$VAL_DATA" \
  --epochs 300 \
  --batch-size 8192 \
  --split-policy spatial_block
```

Encoding / loss sweep (7 tasks):

```bash
sbatch --array=0-6 hpc/slurm_train_scenario_array.sbatch \
  hpc/scenarios/st_lrps_potential_autograd_encoding_and_loss_sweep.jsonl \
  --train-data "$TRAIN_DATA" \
  --val-data "$VAL_DATA" \
  --test-data "$TEST_DATA" \
  --ood-data "$OOD_DATA" \
  --epochs 300 \
  --batch-size 8192 \
  --split-policy spatial_block
```

The former `force_direct` student sweep has been removed: the direct
residual-acceleration runtime is archived in the
`experimental/force-direct-archive` branch and is no longer trainable on main.

Runtime-benchmark smoke (2 tasks; pure SH timing, no artifact required):

```bash
sbatch --array=0-1 hpc/slurm_train_scenario_array.sbatch \
  hpc/scenarios/st_lrps_runtime_benchmark_smoke.jsonl
```

**Output-directory naming.** Each task writes to
`$LUNARIS_OUTPUT_DIR/training/<scenario_name>/` (falling back to
`outputs/training/<scenario_name>/` when `LUNARIS_OUTPUT_DIR` is unset). Before
launching, the launcher writes `scenario.json`, `scenario_command.txt`, and
`scenario_environment.json` into that directory for provenance. Outputs are never
written under `src/`.

**Log naming.** The array job writes
`lunaris_scenario_%A_%a.out` / `lunaris_scenario_%A_%a.err` to the submit
directory, where `%A` is the array job ID and `%a` the task ID.

**Resume / overwrite.** A task refuses to start if its output directory already
exists and is non-empty, so an accidental resubmission cannot clobber a finished
run. Pass `--force` (alias `--overwrite`) as a common flag to reuse the
directory. To preview without launching, run the launcher directly with
`--dry-run`:

```bash
python tools/hpc/run_training_scenario.py \
  hpc/scenarios/st_lrps_potential_autograd_paper_ablation_A0_to_A6.jsonl \
  --index 0 --dry-run --train-data "$TRAIN_DATA" --val-data "$VAL_DATA"
```

To adjust GPU/CPU/memory/time, edit the `#SBATCH` directives at the top of
`hpc/slurm_train_scenario_array.sbatch` or override them at submit time
(e.g. `sbatch --gres=gpu:2 --time=48:00:00 ...`).

### 2. Gravity benchmark / validation (after training)

The orbit-level gravity benchmark compares a **trained** ST-LRPS artifact against
spherical-harmonic baselines/references, so it is normally run after a training
run exists. Point it at the trained run directory:

```bash
lunaris-benchmark \
  --gpu-batch-compare \
  --st-lrps-model-dir "$LUNARIS_OUTPUT_DIR/training/<run_dir>" \
  --output-dir "$LUNARIS_OUTPUT_DIR/gravity_benchmark/<run_name>"
```

The same command runs under Slurm via `hpc/slurm_benchmark_gpu.sbatch` (which
calls `lunaris-benchmark "$@"`):

```bash
sbatch hpc/slurm_benchmark_gpu.sbatch \
  --gpu-batch-compare \
  --st-lrps-model-dir "$LUNARIS_OUTPUT_DIR/training/<run_dir>" \
  --output-dir "$LUNARIS_OUTPUT_DIR/gravity_benchmark/<run_name>"
```

### 3. Batch propagation / injection dispersion ensembles

```bash
sbatch hpc/slurm_batch_array.sbatch \
  --sampling-method sobol_scrambled \
  --batch-backend auto \
  --sh-degree 24
```

As an array job (`#SBATCH --array=0-9`) the script derives a distinct RNG seed
and an `--batch-output-path` per task, so the tasks run different ensembles and
write to separate files under `$LUNARIS_OUTPUT_DIR/ensemble/batch_<task>.h5`
instead of colliding on one output. To write elsewhere, override
`LUNARIS_OUTPUT_DIR` (the batch CLI's own output flag is `--batch-output-path`,
not `--out-dir`, which the batch path accepts but ignores).

Sampling and backend selection are explicit and recorded in ensemble outputs:

- `--sampling-method random` is the classical Monte Carlo option.
- `--sampling-method lhs`, `sobol`, or `sobol_scrambled` uses a space-filling
  design, which is usually preferable for validation and coverage studies.

- `--batch-backend auto` prefers the safe GPU path when available and records any
  fallback.
- `--batch-backend cpu_sh` uses the full CPU spherical-harmonic path and is the
  recommended high-fidelity truth/reference backend.
- `--batch-backend numba_cuda_sh` selects the true Numba CUDA classic-SH path. The current
  supported GPU SH tier is degree 24; higher `--sh-degree` requests follow
  `--sh-fallback-policy` (`torch_cuda_sh` when compatible, CPU, or error)
  without silently clipping the degree.
- `--batch-backend gpu_st_lrps_potential` uses the scalar-potential ST-LRPS artifact
  and autograd residual acceleration on PyTorch CUDA. (The former
  `gpu_st_lrps_direct` backend is archived in `experimental/force-direct-archive`.)
- `--batch-backend gpu_st_lrps_third_body` uses the same scalar-potential
  artifact and keeps analytic Sun/Earth third-body gravity on the PyTorch CUDA
  batch path. Other perturbations still fall back explicitly.

For 512-orbit GPU batch propagation, use `auto` or an explicit ST-LRPS GPU
backend for throughput runs, and keep `cpu_sh` high-degree runs as
validation/truth jobs.
Do not describe high-degree SH as a true GPU baseline unless the output metadata
shows `actual_batch_backend=torch_cuda_sh` and an `actual_sh_degree` at the requested tier.

### Dataset generation and evaluation

Dataset/spatial-cloud generation (step 1) and ST-LRPS evaluation (step 3) do not
ship dedicated templates. Run them through the headless CLI, or copy one of the
`.sbatch` files and swap in the relevant command:

```bash
# Dataset / spatial cloud generation
python -m lunaris.surrogate.st_lrps.data.spatial_cloud_generator --help

# Evaluation of a trained model
lunaris-eval \
  --model-dir "$LUNARIS_OUTPUT_DIR/training/<run_dir>" \
  --output-dir "$LUNARIS_OUTPUT_DIR/evaluations/<run_name>"
```

### Output policy

The provided Slurm scripts write their logs to the **submit directory** using
plain filenames (e.g. `lunaris_train_%j.out` / `lunaris_train_%j.err`), so they
do not depend on a pre-existing nested log directory. This is deliberate:
`#SBATCH --output`/`--error` are resolved by Slurm *before* the job script body
runs, so a `mkdir -p outputs/slurm` inside the script would be too late to create
a missing log directory and the job could fail at submission. To collect logs
elsewhere, either pass `sbatch -o <path> -e <path>` at submit time (after creating
the directory yourself), or edit the `#SBATCH` lines in the template. Run products
(checkpoints, metrics, plots) should go to `LUNARIS_OUTPUT_DIR` on scratch via the
`--out-dir`/`--output-dir` flags shown above. Do not modify scripts to write
outputs inside source directories such as `src/lunaris/surrogate/st_lrps/` or
`src/lunaris/core/`.
