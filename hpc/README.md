# Lunaris HPC quick start

This folder contains everything needed to run ST-LRPS (the GPU-heavy
surrogate-gravity workflow) on a Slurm cluster. Full reference:
[`../docs/HPC.md`](../docs/HPC.md).

## TL;DR — three steps

```bash
# 1. Configure your cluster once (account, partition, modules, venv, storage).
cp hpc/cluster.env.example hpc/cluster.env
$EDITOR hpc/cluster.env
#    On UHeM Altay, start from the pre-filled profile instead:
#    cp hpc/clusters/uhem_altay.env.example hpc/cluster.env

# 2. Build the headless environment once (CPython 3.11 only). Creates the venv at
#    LUNARIS_VENV (default $HOME/venvs/lunaris) and installs the pinned HPC lock.
bash hpc/setup_env.sh

# 3. Check (inside a GPU allocation for the GPU checks), then submit.
bash hpc/preflight.sh
hpc/submit.sh train -- \
    --data "$LUNARIS_DATA_DIR/datasets/st_lrps_cloud_suite.h5" \
    --epochs 300 --batch-size 8192 \
    --out-dir "$LUNARIS_OUTPUT_DIR/training/run1"
```

`hpc/submit.sh` reads scheduler placement and optional resource overrides from
`hpc/cluster.env`, so you never edit those into the `.sbatch` files. Everything
before `--` is passed to `sbatch`; everything after `--` goes to the program.

## Files

| File | Purpose |
|------|---------|
| `cluster.env.example` | Site-config template — copy to `cluster.env` (git-ignored) and fill in once. |
| `clusters/uhem_altay.env.example` | Pre-filled site profile for UHeM Altay (a100q, 1×A100, `$HOME` storage) — copy to `cluster.env`. |
| `env_template.sh` | Sourced by every job: loads `cluster.env`, modules, and activates the env. Fails the job on a collected setup error unless `LUNARIS_ENV_STRICT=0`. |
| `setup_env.sh` | Run once on a login node (CPython 3.11): creates the venv at `LUNARIS_VENV`, installs the hash-pinned HPC lock, then the package. |
| `preflight.sh` | Pre-submit check: runs every check, prints PASS/WARN/FAIL and one verdict (imports, entry points, GPU/CUDA, storage). |
| `submit.sh` | Submit helper: injects scheduler placement from `cluster.env`; requests a GPU only where needed; refuses absolute paths left by an unset `$LUNARIS_DATA_DIR`/`$LUNARIS_OUTPUT_DIR`. |
| `slurm_train_stlrps.sbatch` | ST-LRPS training (`lunaris-train`). Exit 75 = stopped and resumable; requeues with `LUNARIS_AUTO_REQUEUE=1`. |
| `slurm_train_scenario_array.sbatch` | Reproducible scenario sweeps (array jobs). A restarted task resumes with `--resume`. |
| `slurm_benchmark_gpu.sbatch` | Orbit-level gravity benchmark (`lunaris-benchmark`). Requires CUDA and an ST-LRPS model directory. |
| `slurm_batch_array.sbatch` | Batch propagation / uncertainty ensembles (`lunaris-batch`). One CPU per task; a GPU only when the arguments ask for a GPU backend (via `submit.sh`). |
| `slurm_generate_dataset.sbatch` | CPU-only ST-LRPS dataset / spatial-cloud generation (no GPU request). |
| `scenarios/*.jsonl` | Self-describing experiment sweeps (ablations, capacity, encoding/loss). |

Scenarios use JSONL rather than YAML because the launcher validates one
self-contained scenario per line and records the resolved JSON beside every
run. Slurm settings stay in `.sbatch`/`cluster.env`, avoiding a second,
unvalidated configuration source.

## Common jobs

To reproduce the successful local `resume_denemesi` architecture and training
run, keep all four HDF5 files in the same validated SH20 -> SH200 suite:

```bash
SUITE_DIR="$LUNARIS_DATA_DIR/datasets/second_run_100_1000km_10Mp_20260520_203208"
hpc/submit.sh scenario --array=0-0 --time=48:00:00 -- \
    hpc/scenarios/st_lrps_resume_denemesi_reproduction.jsonl \
    --train-data "$SUITE_DIR/train_hybrid_10M.h5" \
    --val-data   "$SUITE_DIR/val_uniform_2M.h5" \
    --test-data  "$SUITE_DIR/test_uniform_2M.h5" \
    --ood-data   "$SUITE_DIR/ood_combined_1000k.h5"
```

Resume after a walltime stop by using the same command with `--resume` after
the JSONL path. With `LUNARIS_AUTO_REQUEUE=1` in `cluster.env`, a job that exits
75 (stopped by the walltime signal, checkpoint saved) requeues itself; a single
training job also needs an explicit `--out-dir`. See
[`../docs/HPC.md`](../docs/HPC.md#running-on-slurm) for the exit codes. For persistent site defaults, set `LUNARIS_CPUS_PER_TASK`,
`LUNARIS_MEM`, `LUNARIS_TIME`, and `LUNARIS_SIGNAL` in `hpc/cluster.env`.
Explicit options passed to `hpc/submit.sh` override those defaults.

```bash
# Paper ablation A0–A6 as a 7-task array:
hpc/submit.sh scenario --array=0-6 -- \
    hpc/scenarios/st_lrps_potential_autograd_paper_ablation_A0_to_A6.jsonl \
    --train-data "$LUNARIS_DATA_DIR/datasets/train.h5" \
    --val-data   "$LUNARIS_DATA_DIR/datasets/val.h5" \
    --test-data  "$LUNARIS_DATA_DIR/datasets/test.h5" \
    --ood-data   "$LUNARIS_DATA_DIR/datasets/ood_high.h5" \
    --epochs 300 --batch-size 8192 --split-policy spatial_block

# CPU-only dataset / spatial-cloud generation (skips LUNARIS_GRES; uses
# LUNARIS_CPU_PARTITION over LUNARIS_PARTITION when set):
hpc/submit.sh generate -- \
    --degree-max 200 --degree-min 20 --n-samples 10000000 \
    --alt-range 100 1000 --workers 16 --format h5 \
    --out "$LUNARIS_DATA_DIR/datasets/train_hybrid_10M.h5"

# Gravity benchmark against a trained run:
hpc/submit.sh benchmark -- --gpu-batch-compare \
    --st-lrps-model-dir "$LUNARIS_OUTPUT_DIR/training/run1" \
    --output-dir "$LUNARIS_OUTPUT_DIR/gravity_benchmark/run1"
```

Preview any submission without queuing it by prefixing `DRYRUN=1`:

```bash
DRYRUN=1 hpc/submit.sh train -- --epochs 300
```
