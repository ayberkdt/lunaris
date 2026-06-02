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
5. **Monte Carlo / batch propagation**

> **Keep GUIs off compute nodes.** The desktop UI (`lunaris-ui`) and the ST-LRPS
> Studio (`lunaris-studio`) are interactive tools. Do not install or launch them
> on compute nodes. Install the GUI extras (`.[ui]`/`.[all]`: `PySide6`,
> `PyQt6`, `pyqtgraph`) only on a login or visualization node, and only if you
> actually need them there.

## Installation

The recommended setup registers the package and its console commands
(`lunaris-train`, `lunaris-eval`, `lunaris-benchmark`, `lunaris-mc`, …) in an
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
omits all GUI packages. A flat, pinned `requirements_hpc.txt` is also available
as an alternative dependency list (`pip install -r requirements_hpc.txt`), but
the preferred install is the editable `python -m pip install -e ".[hpc]"` so the
console entry points are registered.

Verify the headless entry points are available:

```bash
lunaris-train --help
lunaris-eval --help
lunaris-benchmark --help
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
  monte_carlo/
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
```

The catalogue lives in `data/data_sources.json`. Entries with an official URL
(currently the NAIF/JPL SPICE kernels) download directly; entries without a
pinned URL (e.g. GRAIL gravity, LOLA topography/albedo) print the official
provider and the directory to place the file in manually. `lunaris-data` resolves
its data root as `--data-dir` → `LUNARIS_DATA_DIR` → the repository `data/`
folder, and writes into `gravity_models/`, `ephemeris_models/`,
`topography_models/`, `albedo_models/`, and `datasets/`.

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
| Monte Carlo / batch propagation | `hpc/slurm_mc_array.sbatch` | `lunaris-mc` |

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
| `st_lrps_force_direct_student_sweep.jsonl` | `0-3` | Direct residual-acceleration student sweep |
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

Force-direct student sweep (4 tasks). **`force_direct` is a deployment/student
runtime**: it predicts residual acceleration directly, does **not** predict the
scalar potential, and needs field, curl, and orbit-level validation before any
scientific claim. It uses the single-file `--data` flag (no train/val split
flags), and it must never be mixed into the scalar-potential ablation matrix:

```bash
sbatch --array=0-3 hpc/slurm_train_scenario_array.sbatch \
  hpc/scenarios/st_lrps_force_direct_student_sweep.jsonl \
  --data "$TRAIN_DATA" \
  --epochs 100 \
  --batch-size 8192
```

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

### 3. Monte Carlo / batch propagation

```bash
sbatch hpc/slurm_mc_array.sbatch \
  --mc-backend auto \
  --gpu-sh-degree 24 \
  --out-dir "$LUNARIS_OUTPUT_DIR/monte_carlo/mc_run"
```

Backend selection is explicit and recorded in Monte Carlo outputs:

- `--mc-backend auto` prefers the safe GPU path when available and records any
  fallback.
- `--mc-backend cpu_sh` uses the full CPU spherical-harmonic path and is the
  recommended high-fidelity truth/reference backend.
- `--mc-backend gpu_sh` selects the true Numba CUDA classic-SH path. The current
  supported GPU SH tier is degree 24; higher `--gpu-sh-degree` requests fall back
  to CPU SH without silently clipping the degree.
- `--mc-backend gpu_st_lrps_potential` uses the scalar-potential ST-LRPS artifact
  and autograd residual acceleration on PyTorch CUDA.
- `--mc-backend gpu_st_lrps_direct` uses direct residual acceleration with a
  no-grad PyTorch CUDA forward pass. Keep it experimental until orbit-level
  validation shows acceptable drift for the target scenario set.

For 512-orbit GPU Monte Carlo, use `auto` or an explicit ST-LRPS GPU backend for
throughput runs, and keep `cpu_sh` high-degree runs as validation/truth jobs.
Do not describe high-degree SH as a true GPU baseline unless the output metadata
shows `actual_mc_backend=gpu_sh` and an `actual_sh_degree` at the requested tier.

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
