#!/usr/bin/env bash
#SBATCH --job-name=lunaris-batch
#SBATCH --output=lunaris_batch_%A_%a.out
#SBATCH --error=lunaris_batch_%A_%a.err
#SBATCH --array=0-9
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00

set -euo pipefail
# Slurm runs a spooled copy of this script, so $0 does not point into hpc/:
# take the folder from hpc/submit.sh (LUNARIS_HPC_DIR), else the submit directory.
_hpc_dir="${LUNARIS_HPC_DIR:-${SLURM_SUBMIT_DIR:-.}/hpc}"
[[ -f "${_hpc_dir}/env_template.sh" ]] || _hpc_dir="$(dirname "$0")"
source "${_hpc_dir}/env_template.sh"

# Give every array task its own RNG seed and output file so the tasks run
# distinct ensembles instead of repeating identical work and clobbering one
# output path. Falls back to task 0 for a plain (non-array) `sbatch` submission.
# These flags are appended last, so — with argparse "last wins" — the array
# script owns the seed and output path; to relocate the output, set
# LUNARIS_OUTPUT_DIR rather than passing --batch-output-path.
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
# Unscrambled Sobol ignores --seed, so every array task would run the same ensemble.
if [[ "${SLURM_ARRAY_TASK_COUNT:-1}" -gt 1 && " $* " =~ [[:space:]]--sampling-method[[:space:]=]sobol[[:space:]] ]]; then
  echo "error: --sampling-method sobol ignores the seed; use sobol_scrambled for array jobs" >&2
  exit 2
fi
mkdir -p "${LUNARIS_OUTPUT_DIR}/ensemble"

lunaris-batch "$@" \
  --seed "$((42 + TASK_ID))" \
  --batch-output-path "${LUNARIS_OUTPUT_DIR}/ensemble/batch_${TASK_ID}.h5"
