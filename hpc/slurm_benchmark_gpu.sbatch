#!/usr/bin/env bash
#SBATCH --job-name=lunaris-bench
#SBATCH --output=lunaris_benchmark_%j.out
#SBATCH --error=lunaris_benchmark_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00

set -euo pipefail
# Slurm runs a spooled copy of this script, so $0 does not point into hpc/:
# take the folder from hpc/submit.sh (LUNARIS_HPC_DIR), else the submit directory.
_hpc_dir="${LUNARIS_HPC_DIR:-${SLURM_SUBMIT_DIR:-.}/hpc}"
[[ -f "${_hpc_dir}/env_template.sh" ]] || _hpc_dir="$(dirname "$0")"
source "${_hpc_dir}/env_template.sh"

lunaris-benchmark "$@"
