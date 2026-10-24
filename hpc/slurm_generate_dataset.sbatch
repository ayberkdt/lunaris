#!/usr/bin/env bash
#SBATCH --job-name=lunaris-generate
#SBATCH --output=lunaris_generate_%j.out
#SBATCH --error=lunaris_generate_%j.err
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=12:00:00
# One Python process on one node (no srun/MPI). Stated explicitly: the UHeM
# job examples give -N and -n on every job.
#SBATCH --nodes=1
#SBATCH --ntasks=1
# CPU-only job on purpose: the spatial-cloud generator is a numba/NumPy
# pipeline (no --gres line, and `hpc/submit.sh generate` skips the
# LUNARIS_GRES injection so a GPU site profile does not leak a GPU request
# into it). Unless you pass --workers, the generator gets one worker per
# allocated CPU (--cpus-per-task, 16 here).

set -euo pipefail
# Slurm runs a spooled copy of this script, so $0 does not point into hpc/:
# take the folder from submit.sh (LUNARIS_HPC_DIR), else the submit directory.
_hpc_dir="${LUNARIS_HPC_DIR:-${SLURM_SUBMIT_DIR:-.}/hpc}"
[[ -f "${_hpc_dir}/env_template.sh" ]] || _hpc_dir="$(dirname "$0")"
source "${_hpc_dir}/env_template.sh"

# Example (10M-point suite piece; see the generator --help for the full menu):
#   hpc/submit.sh generate -- \
#       --degree-max 200 --degree-min 20 --n-samples 10000000 \
#       --alt-range 100 1000 --format h5 \
#       --out "$LUNARIS_DATA_DIR/datasets/train_hybrid_10M.h5"
# Without --workers the generator would use every CPU it may run on, and a node
# shared with other jobs would be oversubscribed.
args=("$@")
workers="$(lunaris_arg_value --workers "$@" || true)"
if [[ -z "${workers}" ]]; then
  args+=(--workers "${SLURM_CPUS_PER_TASK:-1}")
elif [[ "${workers}" =~ ^[0-9]+$ && -n "${SLURM_CPUS_PER_TASK:-}" && "${workers}" -gt "${SLURM_CPUS_PER_TASK}" ]]; then
  echo "warning: --workers ${workers} exceeds the ${SLURM_CPUS_PER_TASK} allocated CPUs" >&2
fi
python -m lunaris.surrogate.st_lrps.data.spatial_cloud_generator ${args[@]+"${args[@]}"}
