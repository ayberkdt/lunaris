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
  --out-dir "$LUNARIS_OUTPUT_DIR/monte_carlo/mc_run"
```

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

The provided Slurm scripts write their logs under `outputs/slurm/` (e.g.
`outputs/slurm/train_%j.out`), relative to the submit directory. Run products
(checkpoints, metrics, plots) should go to `LUNARIS_OUTPUT_DIR` on scratch via
the `--out-dir`/`--output-dir` flags shown above. Do not modify scripts to write
outputs inside source directories such as `src/lunaris/surrogate/st_lrps/` or
`src/lunaris/core/`.
