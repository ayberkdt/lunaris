#!/usr/bin/env bash
#SBATCH --job-name=lunaris-batch
#SBATCH --output=lunaris_batch_%A_%a.out
#SBATCH --error=lunaris_batch_%A_%a.err
#SBATCH --array=0-9
# CPU backends run samples sequentially on one core; extra CPUs would sit idle.
# This template carries no --gres line: `hpc/submit.sh batch` adds LUNARIS_GRES
# only when the program arguments ask for a GPU backend (see its usage text).
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=06:00:00
# One Python process on one node (no srun/MPI). Stated explicitly: the UHeM
# job examples give -N and -n on every job.
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -euo pipefail
# Slurm runs a spooled copy of this script, so $0 does not point into hpc/:
# take the folder from submit.sh (LUNARIS_HPC_DIR), else the submit directory.
_hpc_dir="${LUNARIS_HPC_DIR:-${SLURM_SUBMIT_DIR:-.}/hpc}"
[[ -f "${_hpc_dir}/env_template.sh" ]] || _hpc_dir="$(dirname "$0")"
source "${_hpc_dir}/env_template.sh"

# Give every array task its own RNG seed and output file so the tasks run
# distinct ensembles instead of repeating identical work and clobbering one
# output path. Falls back to task 0 for a plain (non-array) `sbatch` submission.
#
#   seed        = <base seed> + <array task id>. The base seed is your --seed
#                 (lunaris-batch's default, 42, when you give none). The same
#                 base seed reproduces the same ensembles on a second
#                 submission; pass a different --seed (spaced by at least the
#                 array size, or the tasks of two campaigns overlap) for new
#                 ones. The computed seed is printed below.
#   output file = $LUNARIS_OUTPUT_DIR/ensemble/batch_<array job id>_<task id>.h5,
#                 so a second submission never overwrites the first. This flag
#                 is appended last and wins; to relocate the output set
#                 LUNARIS_OUTPUT_DIR instead of passing --batch-output-path.
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
JOB_TAG="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
BASE_SEED="$(lunaris_arg_value --seed "$@" || true)"
BASE_SEED="${BASE_SEED:-42}"
if ! [[ "${BASE_SEED}" =~ ^-?[0-9]+$ ]]; then
  echo "error: --seed must be an integer, got '${BASE_SEED}'" >&2; exit 64
fi
# Unscrambled Sobol ignores --seed, so every array task would run the same ensemble.
if [[ "${SLURM_ARRAY_TASK_COUNT:-1}" -gt 1 && " $* " =~ [[:space:]]--sampling-method[[:space:]=]sobol[[:space:]] ]]; then
  echo "error: --sampling-method sobol ignores the seed; use sobol_scrambled for array jobs" >&2; exit 2
fi
mkdir -p "${LUNARIS_OUTPUT_DIR}/ensemble"

# The user's own --seed is replaced by the per-task seed derived from it.
lunaris_drop_flag --seed 1 "$@"
TASK_SEED="$((BASE_SEED + TASK_ID))"
OUT_PATH="${LUNARIS_OUTPUT_DIR}/ensemble/batch_${JOB_TAG}_${TASK_ID}.h5"
echo "[lunaris] batch task ${TASK_ID}: seed ${TASK_SEED} (base ${BASE_SEED} + task ${TASK_ID}), output ${OUT_PATH}"

lunaris-batch ${LUNARIS_ARGS_OUT[@]+"${LUNARIS_ARGS_OUT[@]}"} \
  --seed "${TASK_SEED}" \
  --batch-output-path "${OUT_PATH}"
