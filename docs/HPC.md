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

### From a fresh clone (one command)

On a cluster **login node** (the downloads need internet access, which compute
nodes often lack), right after `git clone`:

```bash
bash hpc/bootstrap.sh --profile uhem_altay
```

It runs the manual steps below in order and ends with an OK / WARN / FAIL summary
and one next step. It copies the site profile to `hpc/cluster.env` only when that
file does not exist (an existing one is never overwritten) and refuses CRLF line
endings. It prints `spart` (UHeM), `sinfo -s`, `sacctmgr`, `module avail CUDA` and `quota -s` for
the profile's `>>> FILL IN` values. It builds the venv with `hpc/setup_env.sh`
(skipped when the venv already has `lunaris-train`). It downloads the
`full-gravity` data preset into `LUNARIS_DATA_DIR`: the NAIF SPICE kernels
(`naif0012.tls`, `de440.bsp`, `pck00011.tpc`, `gm_de440.tpc`, the DE440 Moon
frame kernels) and the GRAIL GL1800F gravity field (`jggrx_1800f_sha.tab` + its
label), each checked against the SHA-256 in `data/data_sources.json`. Every file
of the preset must then be present and valid; `lunaris-data verify` alone lets
optional entries be missing. Last, it runs `hpc/preflight.sh`. A rerun is safe:
verified files are not downloaded again. Options: `--skip-env`, `--skip-data`,
`--dry-run`; `LUNARIS_DATA_PRESET` picks another preset.

ST-LRPS cloud suites and finished training runs (for `--init-from` or
evaluation) cannot be downloaded. Copy them from the workstation that has them:

```bash
bash hpc/stage_artifacts.sh <user>@<login-host> \
    --suite outputs/datasets/cloud_suites/<suite id> \
    --model outputs/training/<run dir>
```

Suites land in `$LUNARIS_DATA_DIR/datasets/cloud_suites/<name>`, runs in
`$LUNARIS_OUTPUT_DIR/training/<name>` (defaults `lunaris_data` and
`lunaris_outputs` under the remote `$HOME`, as in the Altay profile;
`--remote-data` / `--remote-outputs` change them). The SHA-256 list travels with
the files and the cluster checks it with `sha256sum -c` right after unpacking.
An artifact that already exists on the cluster is not touched. Runs on Linux,
macOS and Git Bash on Windows (needs `ssh`, `tar`, `sha256sum`); `--dry-run`
lists what would be copied. A training run's `config.json` keeps the absolute
paths of the machine that trained it, so pass `--data`, `--val-data`, ... with
the cluster paths when you continue from it.

### Manual steps

```bash
# 1. Configure your cluster once: account, partition, modules, venv, storage.
cp hpc/cluster.env.example hpc/cluster.env
$EDITOR hpc/cluster.env
#    Keep LF line endings: hpc/preflight.sh fails on a CRLF cluster.env.
#    Pre-filled site profiles live in hpc/clusters/ (currently UHeM Altay):
#    cp hpc/clusters/uhem_altay.env.example hpc/cluster.env
#    The Altay profile marks values this repo cannot confirm with ">>> FILL IN",
#    each with the command that finds the value (module avail, sinfo -s, ...).

# 2. Build the headless environment once (CPython 3.11 only). Creates the venv at
#    LUNARIS_VENV (default $HOME/venvs/lunaris) and installs the hash-pinned HPC
#    lock, then the package. Nothing to copy back when the default path is used.
bash hpc/setup_env.sh

# 3. Sanity-check, then submit. submit.sh injects partition/account/qos/gres
#    from cluster.env, so you never edit those into the .sbatch files.
#    Run the preflight inside a short GPU allocation for its GPU checks (see
#    "Pre-submit check" below). Optional: LUNARIS_PREFLIGHT_VERIFY_DATA=1 also
#    runs the strict data verification (recommended before the first real job).
bash hpc/preflight.sh
hpc/submit.sh train -- \
    --data "$LUNARIS_DATA_DIR/datasets/st_lrps_cloud_suite.h5" \
    --epochs 300 --batch-size 8192 \
    --out-dir "$LUNARIS_OUTPUT_DIR/training/run1"
```

> **The strong model needs no flags.** `TrainConfig` defaults *are* the strong
> benchmark-candidate profile (physical radial-decay encoding, residual SIREN
> blocks, 3-band multi-scale SIREN, altitude-balanced + radial/cross +
> direction losses, hybrid checkpoint metric) — there is no separate YAML/JSON
> "strong model" config to point at. The command above already trains it. For
> a provenance-tracked, paper-posture production run use the committed
> scenario `hpc/scenarios/st_lrps_strong_model_production.jsonl` (array `0-0`).

`hpc/cluster.env` is git-ignored, so site-specific values never get committed.
Every `hpc/*.sbatch` job sources `hpc/env_template.sh`, which loads `cluster.env`,
runs the requested `module load`s, and activates your venv/conda env before the
headless entry point runs. The manual installation and per-workflow details
below still apply when you need finer control.

Scheduler resources can also be centralized there with
`LUNARIS_CPUS_PER_TASK`, `LUNARIS_MEM`, `LUNARIS_TIME`, and `LUNARIS_SIGNAL`.
Leave them empty to keep each `.sbatch` default. Flags passed directly to
`hpc/submit.sh` are appended last and override the site defaults.

`env_template.sh` sets no shell options; each `.sbatch` sets `set -euo pipefail`
before sourcing it. Failures it cannot recover from (a `module load` that fails,
a missing venv, a per-job scratch directory that cannot be created) are collected
in `LUNARIS_ENV_FAILS` and abort the job. `LUNARIS_ENV_STRICT=0` collects them
without aborting; `hpc/preflight.sh` and `hpc/setup_env.sh` use that to report
every problem. Threads follow `--cpus-per-task` (1 outside Slurm), and
`LUNARIS_THREADS` overrides the OpenMP, MKL, OpenBLAS and numba thread counts
together. Each job gets its own `TMPDIR` under `LUNARIS_SCRATCH_DIR` (default
`${TMPDIR:-/tmp}`), plus a `NUMBA_CACHE_DIR` inside it unless one is already
exported; the directory is removed when the job script exits.

> **HDF5 on parallel filesystems.** `env_template.sh` exports
> `HDF5_USE_FILE_LOCKING=FALSE` by default. HDF5 ≥ 1.10 takes POSIX file locks
> on open, and Lustre/GPFS/NFS mounts commonly reject them with
> `OSError: unable to lock file` even for read-only access. Training only reads
> its HDF5 clouds (read-only + SWMR), so disabling locking is safe. If your
> site needs locking, export a different value before submitting — the template
> never overrides an existing setting.

## Installation

The recommended setup registers the package and its console commands
(`lunaris-train`, `lunaris-eval`, `lunaris-benchmark`, `lunaris-batch`,
`lunaris-data`, …) in an isolated, GUI-free environment.

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

`hpc/setup_env.sh` performs this same install into `LUNARIS_VENV`, and stops
unless the interpreter is CPython 3.11. The lock pins `torch==2.13.0`, built for
CUDA 13, which needs an NVIDIA driver `>= 580` on the GPU nodes. For an older
driver, run `LUNARIS_TORCH_INDEX_URL=<PyTorch wheel index for your driver> bash hpc/setup_env.sh`:
torch then comes from that index, and the lock's torch, triton, `nvidia-*` and
`cuda-*` pins are skipped (and removed first from a reused venv). The CUDA 12.6
index also publishes torch 2.13.0 (driver `>= 525`). For a driver from 450 to
524 the CUDA 11.8 index stops at torch 2.7.1, so name the version too:
`LUNARIS_TORCH_INDEX_URL=https://download.pytorch.org/whl/cu118 LUNARIS_TORCH_VERSION=2.7.1 bash hpc/setup_env.sh`
(the CPU suite passes with torch 2.7.1 on CPython 3.11; checked 2026-10-23).
`hpc/preflight.sh`, run inside a GPU job, prints the command that fits the
node's driver.

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

`hpc/env_template.sh` sets the storage defaults that the Slurm jobs source. They
are `$HOME/lunaris_data` and `$HOME/lunaris_outputs`; point the variables at your
scratch or project area in `hpc/cluster.env` (or export them before `sbatch`):

```bash
export LUNARIS_DATA_DIR="${LUNARIS_DATA_DIR:-$HOME/lunaris_data}"
export LUNARIS_OUTPUT_DIR="${LUNARIS_OUTPUT_DIR:-$HOME/lunaris_outputs}"
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

### Pre-submit check (`hpc/preflight.sh`)

`hpc/preflight.sh` loads the environment the way a job does, runs every check,
prints `PASS`, `WARN` or `FAIL` for each, and ends with one `PREFLIGHT VERDICT`.
It exits 1 only when a check FAILED; warnings never change the exit code. It checks:

- `hpc/cluster.env` exists and has LF line endings;
- scheduler settings (`LUNARIS_PARTITION`, `LUNARIS_ACCOUNT`, `LUNARIS_QOS`) when `sbatch` is on PATH;
- the venv: CPython 3.11, `import lunaris`, and the entry points `lunaris-train`, `lunaris-eval`, `lunaris-benchmark`, `lunaris-batch`, `lunaris-data`;
- `numpy`, `h5py` and `torch` imports;
- data under `LUNARIS_DATA_DIR` (the ephemeris kernels are required; the gravity file and HDF5 clouds are WARN), and the output directory (writable, free space);
- with `LUNARIS_PREFLIGHT_VERIFY_DATA=1`, `lunaris-data verify --strict --runtime` (slow).

The GPU checks are `nvidia-smi`, the torch CUDA build against the driver, a CUDA
kernel launch, and numba's libNVVM (WARN only). They run when a GPU is visible or
when one is required: `LUNARIS_PREFLIGHT_REQUIRE_GPU=1`, `LUNARIS_REQUIRE_CUDA=1`,
or a Slurm GPU allocation (`SLURM_JOB_GPUS`, `SLURM_GPUS` or `SLURM_GPUS_ON_NODE`).
When a GPU is required and none is visible, they FAIL; otherwise they are skipped
with a WARN. Run it inside a short interactive GPU allocation on UHeM:

```bash
srun -p a100q --gres=gpu:1 --cpus-per-task=4 --time=00:10:00 --pty bash
bash hpc/preflight.sh
```

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

> **Walltime and interruption.** The training templates
> (`slurm_train_stlrps.sbatch`, `slurm_train_scenario_array.sbatch`) send
> `SIGTERM` to the batch shell 30 minutes before the walltime
> (`#SBATCH --signal=B:TERM@1800`). The shell relays `TERM` and `USR1` to the
> trainer, which treats them like Ctrl+C: it finishes the current epoch, saves
> `ckpt_last.pt`, marks the run manifest `interrupted` with a resume hint, and
> exits with status 75. The lead time must exceed one epoch plus its validation,
> or the job is killed before it saves; for slow epochs raise it with
> `LUNARIS_SIGNAL="B:TERM@3600"` in `hpc/cluster.env`. The templates also pass
> `--requeue` and `--open-mode=append` (one log across restarts).

The exit status of `lunaris-train` (a job's exit status is the same value):

| Exit | Meaning |
|------|---------|
| `0` | run completed |
| `75` (`EX_TEMPFAIL`) | stopped by a signal after `ckpt_last.pt` was saved; resumable |
| `1` | run failed (NaN/Inf loss, aborted periodic evaluation), or an uncaught error |

A non-zero exit is recorded by Slurm as a failed job, so an interrupted run is
not reported as `COMPLETED`. The job prints the resume command on exit 75:

- **Single training job:** use the command the job (and the trainer's interrupt
  message) prints: the original arguments plus `--resume-from <run_dir>`, e.g.
  `hpc/submit.sh train -- <original flags> --resume-from <run_dir> --epochs <total>`.
  `--epochs` is the total target epoch count. Resume locks only the architecture
  and scaler fields; optimisation settings (batch size, learning rate, loss
  weights) come from the resuming command line, so a bare
  `--resume-from <run_dir>` falls back to the defaults for them. The trainer logs
  every setting that differs from the resumed run. A bare resume does restore the
  run's data files (including independent `--train-data`/`--val-data` files).
- **Scenario array:** resubmit the same array with `--resume` after the JSONL path.
  Each task continues from its own `checkpoints/ckpt_last.pt` and starts fresh when
  there is none. The launcher reports exit 75 as `STOPPED` (resumable), not `FAILED`.
- **Automatic:** set `LUNARIS_AUTO_REQUEUE=1` in `hpc/cluster.env`, and a job that
  exits 75 requeues itself with `scontrol requeue`. A single training job must name
  its run directory with `--out-dir` (the template exits 64 otherwise). A requeued
  attempt resumes from `<out-dir>/checkpoints/ckpt_last.pt`, or starts over with the
  original arguments if that file is missing.

> **One GPU per training job.** The trainer uses a single device (`cuda:0`);
> there is no DDP/multi-GPU path. Requesting `--gres=gpu:2` wastes an
> allocation. The templates also export `LUNARIS_REQUIRE_CUDA=1`, so a job
> whose torch cannot see CUDA (wrong module, CPU-only wheel) fails the trainer's
> own preflight check (`cuda_visibility`) instead of silently crawling through
> the walltime on CPU. `--allow-preflight-fail` and `--skip-preflight` are the
> escape hatches.

> **Throughput hints (Linux compute nodes).** The Windows-only
> HDF5-multiworker restriction does not apply on Linux: with the RAM-preload
> path active (`--preload-data`, or `auto` for small datasets) you can raise
> `--num-workers` toward the allocated CPUs (e.g. 6 of `--cpus-per-task=8`).
> Prefer preload or independent train/val files on parallel filesystems —
> large single-file streaming reads are expensive on Lustre/GPFS. If you
> enable periodic evaluation during training, pass
> `--periodic-eval-device cpu` on single-GPU allocations so the evaluation
> subprocess does not compete with training for VRAM.

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
| `st_lrps_resume_denemesi_reproduction.jsonl` | `0-0` | Frozen reproduction of the successful May 2026 512x5, 2-band SH20 -> SH200 run |
| `st_lrps_strong_model_production.jsonl` | `0-0` | Production strong-model run: TrainConfig defaults + `--run-preset paper` |
| `st_lrps_potential_autograd_paper_ablation_A0_to_A6.jsonl` | `0-20` | Ablation ladder A0→A6 × 3 seeds, generated from the `lunaris-ablation` registry |
| `st_lrps_potential_autograd_capacity_sweep_A6_full.jsonl` | `0-4` | A6-full architecture-size sweep (4×256 … 5×768) |
| `st_lrps_potential_autograd_encoding_and_loss_sweep.jsonl` | `0-6` | 5×512 direction-weight + input-encoding matrix |
| `st_lrps_runtime_benchmark_smoke.jsonl` | `0-1` | CPU/CUDA timing smoke checks |

> **The ablation ladder ends at the production model.** A0–A5 train on raw
> XYZ (`--model-preset baseline_raw`) and add one component per step at the
> shipped loss settings (direction weight 0.1). A6 adds the
> `recommended_physical_radial_decay` encoding, so A6 is the TrainConfig
> default — the same configuration as `st_lrps_strong_model_production.jsonl`
> without `--run-preset paper`. The file is generated from the
> `lunaris-ablation` registry (`python tools/hpc/generate_ablation_scenarios.py`);
> do not edit it by hand. Lines are seed-major: tasks 0–6 are one complete
> ladder (seed 42), 7–13 seed 43, 14–20 seed 44. Every line sets the same
> `--split-seed`, so all runs use one data split. Do not pass `--seed` or
> `--split-seed` as a common flag: it would override every line.
> The capacity and encoding/loss sweeps still use raw XYZ with direction
> weight 0.2; they are separate studies, not the A6 model.

> **Trajectory-aware screening campaign.** `st_lrps_trajectory_doe_phase1.jsonl`
> (array `0-9`) and `st_lrps_trajectory_doe_foldover.jsonl` (array `0-6`) are
> generated by `python tools/st_lrps_doe.py scenarios`; do not edit them by
> hand. Phase 1 is a 2^(6-3) fractional factorial over six training options
> plus two anchors at the shipped settings; the fold-over adds the mirrored
> runs, and together they form a resolution-IV design. Every line already
> carries its seed, split seed and reduced-fidelity budget (48 epochs,
> warmup-stable-decay schedule, capped batches), so pass only the data flags
> (`--train-data`, `--val-data`, `--suite-manifest`) as common flags.
> `python tools/st_lrps_doe.py analyze --out-root <dir>` reads the run
> histories and reports main effects with Lenth's pseudo standard error.

> **Frozen reproduction versus current defaults.** The historical
> `resume_denemesi` artifact used two frequency bands (`[13.7, 42.4]`) and
> several schedule/loss values that no longer all match `TrainConfig` defaults.
> Use `st_lrps_resume_denemesi_reproduction.jsonl` when the goal is to repeat
> that model, rather than follow the evolving strong-model profile.

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

# A0–A6 paper ablation × 3 seeds (21 tasks; 0-6 is one complete ladder)
sbatch --array=0-20 hpc/slurm_train_scenario_array.sbatch \
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
`$LUNARIS_OUTPUT_DIR/training/<scenario_name>/`. The launcher falls back to
`<repo root>/outputs/training/<scenario_name>/` only when neither `--output-root`
nor `LUNARIS_OUTPUT_DIR` is set; the job templates always set `LUNARIS_OUTPUT_DIR`
(default `$HOME/lunaris_outputs`). Before
launching, the launcher writes `scenario.json`, `scenario_command.txt`, and
`scenario_environment.json` into that directory for provenance. Outputs are never
written under `src/`.

**Log naming.** The array job writes
`lunaris_scenario_%A_%a.out` / `lunaris_scenario_%A_%a.err` to the submit
directory, where `%A` is the array job ID and `%a` the task ID.

**Resume / overwrite.** A task refuses to start if its output directory already
exists and is non-empty, so an accidental resubmission cannot clobber a finished
run. Two escape hatches:

- `--resume` (recommended after a walltime kill; a requeued array attempt adds it
  itself and drops `--overwrite`/`--force`): when the task's output
  directory contains `checkpoints/ckpt_last.pt` (or `ckpt_best.pt`), the
  launcher relaunches with `--resume-from <run_dir>` instead of `--out`, so
  every array task continues from its own last completed epoch. Tasks without
  a previous run start fresh. A directory that holds only this scenario's own
  provenance and no checkpoint (a task Slurm requeued before its first save)
  is refused unless `--restart-if-no-checkpoint` is also given; the requeued
  array attempt passes it, so such a task restarts from scratch. The original `scenario.json` provenance is kept;
  each resume attempt adds timestamped `scenario_resume_command_*.txt` /
  `scenario_resume_environment_*.json` files.
- `--force` (alias `--overwrite`): reuse the directory and OVERWRITE the run.
  Mutually exclusive with `--resume`.

To preview without launching, run the launcher directly with `--dry-run`
(the printed `mode` line shows whether a task would start fresh or resume):

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

`hpc/slurm_benchmark_gpu.sbatch` requests one GPU and checks CUDA before it starts
(`LUNARIS_REQUIRE_CUDA` defaults to 1). It always appends `--require-st-lrps`, so
the job fails rather than dropping ST-LRPS from the comparison when no valid model
directory is found. Without `--out` or `--output-dir`, it writes to
`$LUNARIS_OUTPUT_DIR/gravity_benchmark/<jobid>`. It sends SIGTERM 5 minutes before
the walltime (`B:TERM@300`). The benchmark has no checkpoint of its own, but
`--resume` skips scenarios already in `per_scenario_metrics.csv`; on a non-zero
exit the job prints the command to continue with `--output-dir <dir> --resume`.

### 3. Batch propagation / injection dispersion ensembles

```bash
hpc/submit.sh batch -- \
  --sampling-method sobol_scrambled \
  --batch-backend auto \
  --sh-degree 24
```

`hpc/submit.sh batch` requests a GPU only when the arguments ask for a GPU backend
(`numba_cuda_sh`, `torch_cuda_sh`, `gpu_st_lrps_*`, or `auto` with `--use-gpu true`,
which is the `lunaris-batch` default). Add `--gres=gpu:1` before `--` to force one.
If `LUNARIS_GRES` is empty, `submit.sh` warns that no GPU is requested. Calling
`sbatch` on the template directly skips this logic, since the template has no
`--gres` line of its own.

As an array job (`#SBATCH --array=0-9`, one CPU per task) the script sets each
task's seed to your `--seed` (default 42) plus the task ID, and writes
`$LUNARIS_OUTPUT_DIR/ensemble/batch_<arrayjob>_<task>.h5`, so a second submission
never overwrites the first. The same base seed reproduces the same ensembles; use
a base seed spaced by at least the array size for a new campaign. Unscrambled
`--sampling-method sobol` is refused for array jobs, because it ignores the seed
and every task would repeat one ensemble; use `sobol_scrambled`. The script's
`--seed` and `--batch-output-path` take precedence over any you pass. To write
elsewhere, set `LUNARIS_OUTPUT_DIR` (the batch CLI's own output flag is
`--batch-output-path`, not `--out-dir`, which the batch path accepts but ignores).

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

Dataset/spatial-cloud generation (step 1) ships a dedicated CPU-only template,
`hpc/slurm_generate_dataset.sbatch`, submitted as `hpc/submit.sh generate`.
Because the generator is a numba/NumPy pipeline, that job requests no GPU:
`submit.sh` skips the `LUNARIS_GRES` injection for it and prefers
`LUNARIS_CPU_PARTITION` (when set in `cluster.env`) over the GPU queue in
`LUNARIS_PARTITION`. Unless you pass `--workers`, the job uses `$SLURM_CPUS_PER_TASK`
workers. ST-LRPS evaluation (step 3) still has no dedicated
template — run it through the headless CLI, or copy one of the `.sbatch`
files and swap in the relevant command:

```bash
# Dataset / spatial cloud generation (flags: see --help)
hpc/submit.sh generate -- \
  --degree-max 200 --degree-min 20 --n-samples 10000000 \
  --alt-range 100 1000 --workers 16 --format h5 \
  --out "$LUNARIS_DATA_DIR/datasets/train_hybrid_10M.h5"
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
