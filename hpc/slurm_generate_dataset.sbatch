#!/usr/bin/env bash
#SBATCH --job-name=lunaris-generate
#SBATCH --output=lunaris_generate_%j.out
#SBATCH --error=lunaris_generate_%j.err
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=12:00:00
# CPU-only job on purpose: the spatial-cloud generator is a numba/NumPy
# pipeline (no --gres line, and `hpc/submit.sh generate` skips the
# LUNARIS_GRES injection so a GPU site profile does not leak a GPU request
# into it). Match `--workers` to --cpus-per-task for full utilization.

set -euo pipefail
# Slurm runs a spooled copy of this script, so $0 does not point into hpc/:
# take the folder from hpc/submit.sh (LUNARIS_HPC_DIR), else the submit directory.
_hpc_dir="${LUNARIS_HPC_DIR:-${SLURM_SUBMIT_DIR:-.}/hpc}"
[[ -f "${_hpc_dir}/env_template.sh" ]] || _hpc_dir="$(dirname "$0")"
source "${_hpc_dir}/env_template.sh"

# Example (10M-point suite piece; see the generator --help for the full menu):
#   hpc/submit.sh generate -- \
#       --degree-max 200 --degree-min 20 --n-samples 10000000 \
#       --alt-range 100 1000 --workers 16 --format h5 \
#       --out "$LUNARIS_DATA_DIR/datasets/train_hybrid_10M.h5"
python -m lunaris.surrogate.st_lrps.data.spatial_cloud_generator "$@"
