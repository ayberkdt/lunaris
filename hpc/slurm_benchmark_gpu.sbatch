#!/usr/bin/env bash
#SBATCH --job-name=lunaris-bench
#SBATCH --output=lunaris_benchmark_%j.out
#SBATCH --error=lunaris_benchmark_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
# SIGTERM to the batch shell 5 minutes before walltime. The benchmark has no
# checkpoint of its own, but it keeps finished scenarios: the script prints how
# to continue (same --output-dir plus --resume) before Slurm's kill.
#SBATCH --signal=B:TERM@300

set -euo pipefail
# Slurm runs a spooled copy of this script, so $0 does not point into hpc/:
# take the folder from submit.sh (LUNARIS_HPC_DIR), else the submit directory.
_hpc_dir="${LUNARIS_HPC_DIR:-${SLURM_SUBMIT_DIR:-.}/hpc}"
[[ -f "${_hpc_dir}/env_template.sh" ]] || _hpc_dir="$(dirname "$0")"
source "${_hpc_dir}/env_template.sh"

# This template requests a GPU, so a silent CPU fallback (wrong CUDA module /
# CPU-only torch wheel) must abort here. The benchmark CLI does not read
# LUNARIS_REQUIRE_CUDA, so check explicitly.
export LUNARIS_REQUIRE_CUDA="${LUNARIS_REQUIRE_CUDA:-1}"
[[ "${LUNARIS_REQUIRE_CUDA}" != "1" ]] || lunaris_assert_cuda

# Concurrent jobs must not share the CLI's relative default (outputs/gravity_benchmark):
# unless the caller names an output directory, use one per job.
args=("$@")
# --out is the config-mode spelling (`--config`); --output-dir the flag-mode one.
out_dir="$(lunaris_arg_value --out "$@" || lunaris_arg_value --output-dir "$@" || true)"
if [[ -z "${out_dir}" ]]; then
  out_dir="${LUNARIS_OUTPUT_DIR}/gravity_benchmark/${SLURM_JOB_ID:-local}"
  args+=(--output-dir "${out_dir}")
fi
# Without a model directory the benchmark would pick the newest local run, or
# drop st_lrps from the comparison with only a warning; make that an error.
args+=(--require-st-lrps)
echo "[lunaris] benchmark output directory: ${out_dir}"

# Not exec: the walltime SIGTERM reaches only this shell, so relay it and report.
rc=0
lunaris_run_stoppable lunaris-benchmark "${args[@]}" || rc=$?
if [[ ${rc} -ne 0 ]]; then
  echo "[lunaris] benchmark ended with exit ${rc}. To continue, resubmit the same arguments with --output-dir ${out_dir} --resume" >&2
fi
exit "${rc}"
