#!/usr/bin/env bash
#
# Thin submit helper. Picks the right hpc/*.sbatch template for a named job and
# injects scheduler placement and optional resource overrides from
# hpc/cluster.env, so you never have to edit account or partition into each
# .sbatch file by hand.
#
# Usage:
#   hpc/submit.sh <job> [extra sbatch flags...] -- [program args...]
#
#   <job> is one of: train | scenario | benchmark | batch | generate
#   Anything before `--` is passed to sbatch (e.g. --array=0-6, --time=48:00:00).
#   Anything after  `--` is forwarded to the job's program ($@ in the .sbatch).
#
# Examples:
#   hpc/submit.sh train -- --epochs 300 --batch-size 8192 \
#       --out-dir "$LUNARIS_OUTPUT_DIR/training/run1"
#
#   hpc/submit.sh scenario --array=0-6 -- \
#       hpc/scenarios/st_lrps_potential_autograd_paper_ablation_A0_to_A6.jsonl \
#       --train-data "$LUNARIS_DATA_DIR/datasets/train.h5" \
#       --val-data   "$LUNARIS_DATA_DIR/datasets/val.h5" \
#       --epochs 300 --batch-size 8192 --split-policy spatial_block
#
#   hpc/submit.sh benchmark -- --gpu-batch-compare \
#       --st-lrps-model-dir "$LUNARIS_OUTPUT_DIR/training/run1"
#
# Add DRYRUN=1 to print the resolved sbatch command without submitting.

set -euo pipefail

_HPC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"

usage() {
  sed -n '2,30p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  exit "${1:-64}"
}

[[ $# -lt 1 ]] && usage 64
job="$1"; shift

# `needs_gpu=0` jobs skip the LUNARIS_GRES injection below, so a GPU site
# profile (e.g. LUNARIS_GRES="gpu:1") cannot leak a GPU request into a
# CPU-only job.
needs_gpu=1
case "${job}" in
  train)     script="slurm_train_stlrps.sbatch" ;;
  scenario)  script="slurm_train_scenario_array.sbatch" ;;
  benchmark) script="slurm_benchmark_gpu.sbatch" ;;
  batch)     script="slurm_batch_array.sbatch" ;;
  generate)  script="slurm_generate_dataset.sbatch"; needs_gpu=0 ;;
  -h|--help) usage 0 ;;
  *) echo "error: unknown job '${job}' (use train|scenario|benchmark|batch|generate)" >&2; usage 64 ;;
esac
script_path="${_HPC_DIR}/${script}"
[[ -f "${script_path}" ]] || { echo "error: missing ${script_path}" >&2; exit 66; }

# Split args at the first `--`: before -> sbatch flags, after -> program args.
sbatch_extra=()
prog_args=()
seen_sep=0
for arg in "$@"; do
  if [[ ${seen_sep} -eq 0 && "${arg}" == "--" ]]; then
    seen_sep=1; continue
  fi
  if [[ ${seen_sep} -eq 0 ]]; then sbatch_extra+=("${arg}"); else prog_args+=("${arg}"); fi
done

# Load cluster.env for scheduler placement and resource values.
if [[ -f "${_HPC_DIR}/cluster.env" ]]; then
  set -a; source "${_HPC_DIR}/cluster.env"; set +a
fi

sbatch_flags=()
# CPU-only jobs prefer LUNARIS_CPU_PARTITION (when set) over the GPU queue in
# LUNARIS_PARTITION, so a GPU site profile doesn't route them to a GPU queue.
partition="${LUNARIS_PARTITION:-}"
[[ ${needs_gpu} -eq 0 && -n "${LUNARIS_CPU_PARTITION:-}" ]] && partition="${LUNARIS_CPU_PARTITION}"
[[ -n "${partition}" ]] && sbatch_flags+=("--partition=${partition}")
[[ -n "${LUNARIS_ACCOUNT:-}"   ]] && sbatch_flags+=("--account=${LUNARIS_ACCOUNT}")
[[ -n "${LUNARIS_QOS:-}"       ]] && sbatch_flags+=("--qos=${LUNARIS_QOS}")
[[ -n "${LUNARIS_GRES:-}" && ${needs_gpu} -eq 1 ]] && sbatch_flags+=("--gres=${LUNARIS_GRES}")
[[ -n "${LUNARIS_CPUS_PER_TASK:-}" ]] && sbatch_flags+=("--cpus-per-task=${LUNARIS_CPUS_PER_TASK}")
[[ -n "${LUNARIS_MEM:-}"            ]] && sbatch_flags+=("--mem=${LUNARIS_MEM}")
[[ -n "${LUNARIS_TIME:-}"           ]] && sbatch_flags+=("--time=${LUNARIS_TIME}")
[[ -n "${LUNARIS_SIGNAL:-}"         ]] && sbatch_flags+=("--signal=${LUNARIS_SIGNAL}")
# User-supplied sbatch flags come last so they can override cluster.env defaults.
# Guard empty-array expansion for bash < 4.4 under `set -u`.
sbatch_flags+=(${sbatch_extra[@]+"${sbatch_extra[@]}"})

cmd=(sbatch ${sbatch_flags[@]+"${sbatch_flags[@]}"} "${script_path}" ${prog_args[@]+"${prog_args[@]}"})

echo "+ ${cmd[*]}"
if [[ "${DRYRUN:-0}" == "1" ]]; then
  echo "(DRYRUN=1: not submitted)"
  exit 0
fi
command -v sbatch >/dev/null 2>&1 || { echo "error: sbatch not found on PATH" >&2; exit 69; }
exec "${cmd[@]}"
