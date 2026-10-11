#!/usr/bin/env bash
#SBATCH --job-name=lunaris-train
#SBATCH --output=lunaris_train_%j.out
#SBATCH --error=lunaris_train_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
# Send SIGTERM to the batch shell (B:) 10 minutes before the walltime limit so
# the trainer can finish the current epoch, save ckpt_last, and mark the run
# "interrupted" (resumable) instead of being killed mid-epoch.
#SBATCH --signal=B:TERM@600

set -euo pipefail
# Slurm runs a spooled copy of this script, so $0 does not point into hpc/:
# take the folder from hpc/submit.sh (LUNARIS_HPC_DIR), else the submit directory.
_hpc_dir="${LUNARIS_HPC_DIR:-${SLURM_SUBMIT_DIR:-.}/hpc}"
[[ -f "${_hpc_dir}/env_template.sh" ]] || _hpc_dir="$(dirname "$0")"
source "${_hpc_dir}/env_template.sh"

# This template requests a GPU, so a silent CPU fallback (wrong CUDA module /
# CPU-only torch wheel) must abort at preflight instead of burning walltime.
export LUNARIS_REQUIRE_CUDA="${LUNARIS_REQUIRE_CUDA:-1}"

# exec: the trainer must be the process that receives the walltime SIGTERM
# (--signal=B:TERM goes to the batch shell only).
exec lunaris-train "$@"
