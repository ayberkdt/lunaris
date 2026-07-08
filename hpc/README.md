# Lunaris HPC quick start

This folder contains everything needed to run ST-LRPS (the GPU-heavy
surrogate-gravity workflow) on a Slurm cluster. Full reference:
[`../docs/HPC.md`](../docs/HPC.md).

## TL;DR — three steps

```bash
# 1. Configure your cluster once (account, partition, modules, venv, storage).
cp hpc/cluster.env.example hpc/cluster.env
$EDITOR hpc/cluster.env

# 2. Build the headless environment once (creates a venv, installs .[hpc]).
bash hpc/setup_env.sh
#    then paste the printed LUNARIS_VENV=... line into hpc/cluster.env

# 3. Check, then submit.
bash hpc/preflight.sh
hpc/submit.sh train -- \
    --data "$LUNARIS_DATA_DIR/datasets/st_lrps_cloud_suite.h5" \
    --epochs 300 --batch-size 8192 \
    --out-dir "$LUNARIS_OUTPUT_DIR/training/run1"
```

`hpc/submit.sh` reads the partition / account / qos / gres from
`hpc/cluster.env`, so you never edit those into the `.sbatch` files. Everything
before `--` is passed to `sbatch`; everything after `--` goes to the program.

## Files

| File | Purpose |
|------|---------|
| `cluster.env.example` | Site-config template — copy to `cluster.env` (git-ignored) and fill in once. |
| `env_template.sh` | Sourced by every job: loads `cluster.env`, modules, and activates the env. |
| `setup_env.sh` | Run once on a login node: creates the venv and installs `.[hpc]`. |
| `preflight.sh` | Pre-submit sanity check (imports, entry points, CUDA, storage). |
| `submit.sh` | Submit helper: injects scheduler placement from `cluster.env`. |
| `slurm_train_stlrps.sbatch` | ST-LRPS training (`lunaris-train`). |
| `slurm_train_scenario_array.sbatch` | Reproducible scenario sweeps (array jobs). |
| `slurm_benchmark_gpu.sbatch` | Orbit-level gravity benchmark (`lunaris-benchmark`). |
| `slurm_batch_array.sbatch` | Batch propagation / uncertainty ensembles (`lunaris-batch`). |
| `scenarios/*.jsonl` | Self-describing experiment sweeps (ablations, capacity, encoding/loss). |

## Common jobs

```bash
# Paper ablation A0–A6 as a 7-task array:
hpc/submit.sh scenario --array=0-6 -- \
    hpc/scenarios/st_lrps_potential_autograd_paper_ablation_A0_to_A6.jsonl \
    --train-data "$LUNARIS_DATA_DIR/datasets/train.h5" \
    --val-data   "$LUNARIS_DATA_DIR/datasets/val.h5" \
    --test-data  "$LUNARIS_DATA_DIR/datasets/test.h5" \
    --ood-data   "$LUNARIS_DATA_DIR/datasets/ood_high.h5" \
    --epochs 300 --batch-size 8192 --split-policy spatial_block

# Gravity benchmark against a trained run:
hpc/submit.sh benchmark -- --gpu-batch-compare \
    --st-lrps-model-dir "$LUNARIS_OUTPUT_DIR/training/run1" \
    --output-dir "$LUNARIS_OUTPUT_DIR/gravity_benchmark/run1"
```

Preview any submission without queuing it by prefixing `DRYRUN=1`:

```bash
DRYRUN=1 hpc/submit.sh train -- --epochs 300
```
