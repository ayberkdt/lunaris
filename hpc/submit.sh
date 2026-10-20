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
# GPU rule: train, scenario and benchmark always request LUNARIS_GRES (their
# .sbatch files carry --gres=gpu:1). generate never does. batch requests it
# only when its program arguments ask for a GPU:
#   --batch-backend numba_cuda_sh | torch_cuda_sh | gpu_st_lrps_potential |
#                   gpu_st_lrps_third_body                       -> GPU
#   --batch-backend auto (the default) with --use-gpu true (the
#                   lunaris-batch default)                       -> GPU
#   anything else (cpu_sh, torch_cpu_sh, auto with --use-gpu false) -> no GPU
# Add --gres=gpu:1 before `--` to force a GPU request for a batch job.
#
# Variables in program arguments are expanded by YOUR shell before this script
# runs, so load the site settings first (the jobs read the same file):
#   set -a; . hpc/cluster.env; set +a
# A program argument that is an absolute path such as /datasets/... or
# /training/... is refused: it is what an unset $LUNARIS_DATA_DIR or
# $LUNARIS_OUTPUT_DIR leaves behind.
#
# Examples:
#   set -a; . hpc/cluster.env; set +a
#
#   hpc/submit.sh train -- --epochs 300 --batch-size 8192 \
#       --data "$LUNARIS_DATA_DIR/datasets/st_lrps_cloud_suite.h5" \
#       --out-dir "$LUNARIS_OUTPUT_DIR/training/run1"
#
#   hpc/submit.sh scenario --array=0-20 -- \
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
  # Print the leading comment block (everything up to the first non-comment line).
  awk 'NR > 1 { if (/^#/) { sub(/^# ?/, ""); print } else exit }' "${BASH_SOURCE[0]}"
  exit "${1:-64}"
}

[[ $# -lt 1 ]] && usage 64
job="$1"; shift

# `needs_gpu=0` jobs skip the LUNARIS_GRES injection below, so a GPU site
# profile (e.g. LUNARIS_GRES="gpu:1") cannot leak a GPU request into a
# CPU-only job. `batch` decides from its program arguments (see below).
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

# Refuse program arguments that look like an unset $LUNARIS_DATA_DIR or
# $LUNARIS_OUTPUT_DIR: `"$LUNARIS_DATA_DIR/datasets/x.h5"` becomes `/datasets/x.h5`
# when the variable is empty in this shell (cluster.env is read below, too late
# for the expansion). Both `--flag value` and `--flag=value` forms are checked.
for arg in ${prog_args[@]+"${prog_args[@]}"}; do
  value="${arg#--*=}"
  case "${value}" in
    /datasets|/datasets/*|/training|/training/*|/outputs|/outputs/*| \
    /gravity_benchmark|/gravity_benchmark/*|/evaluations|/evaluations/*| \
    /ensemble|/ensemble/*|/runtime|/runtime/*| \
    /gravity_models|/gravity_models/*|/ephemeris_models|/ephemeris_models/*| \
    /topography_models|/topography_models/*)
      echo "error: program argument '${arg}' is an absolute path at the filesystem root." >&2
      echo "       An empty \$LUNARIS_DATA_DIR or \$LUNARIS_OUTPUT_DIR in this shell produces this." >&2
      echo "       Load the site settings first:  set -a; . ${_HPC_DIR}/cluster.env; set +a" >&2
      exit 64 ;;
  esac
done

# batch: request a GPU only when the program arguments ask for one.
batch_wants_gpu() {
  # lunaris-batch defaults: --batch-backend auto, --use-gpu true.
  local backend="auto" use_gpu="true" prev="" arg
  for arg in "$@"; do
    case "${prev}" in
      --batch-backend) backend="${arg}" ;;
      --use-gpu)       use_gpu="${arg}" ;;
    esac
    case "${arg}" in
      --batch-backend=*) backend="${arg#*=}" ;;
      --use-gpu=*)       use_gpu="${arg#*=}" ;;
    esac
    prev="${arg}"
  done
  case "${backend}" in
    numba_cuda_sh|torch_cuda_sh|gpu_st_lrps_potential|gpu_st_lrps_third_body) return 0 ;;
    auto) ;;
    *) return 1 ;;
  esac
  # 'auto' follows --use-gpu (str2bool spellings).
  case "${use_gpu,,}" in 1|true|t|yes|y|on) return 0 ;; esac
  return 1
}
batch_gpu_note=""
if [[ "${job}" == "batch" ]]; then
  if batch_wants_gpu ${prog_args[@]+"${prog_args[@]}"}; then
    needs_gpu=1; batch_gpu_note="gpu"
  else
    needs_gpu=0
  fi
fi

# Load cluster.env for scheduler placement and resource values.
if [[ -f "${_HPC_DIR}/cluster.env" ]]; then
  set -a; source "${_HPC_DIR}/cluster.env"; set +a
else
  echo "warning: ${_HPC_DIR}/cluster.env not found: no account/partition/gres is injected." >&2
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
# The templates source hpc/env_template.sh; Slurm runs them from its spool
# folder, so hand them this folder explicitly.
sbatch_flags+=("--export=ALL,LUNARIS_HPC_DIR=${_HPC_DIR}")
# User-supplied sbatch flags come last so they can override cluster.env defaults.
# Guard empty-array expansion for bash < 4.4 under `set -u`.
sbatch_flags+=(${sbatch_extra[@]+"${sbatch_extra[@]}"})

# A batch job that asked for a GPU backend but has no gres to hold one (the
# batch template carries no --gres line of its own).
if [[ "${batch_gpu_note}" == "gpu" && -z "${LUNARIS_GRES:-}" ]]; then
  gres_given=0
  for arg in ${sbatch_extra[@]+"${sbatch_extra[@]}"}; do
    [[ "${arg}" == --gres=* || "${arg}" == --gpus* ]] && gres_given=1
  done
  if [[ ${gres_given} -eq 0 ]]; then
    echo "warning: the batch arguments ask for a GPU backend but LUNARIS_GRES is empty and no --gres was given: no GPU is requested." >&2
  fi
fi

cmd=(sbatch ${sbatch_flags[@]+"${sbatch_flags[@]}"} "${script_path}" ${prog_args[@]+"${prog_args[@]}"})

echo "+ ${cmd[*]}"
if [[ "${DRYRUN:-0}" == "1" ]]; then
  echo "(DRYRUN=1: not submitted)"
  exit 0
fi
command -v sbatch >/dev/null 2>&1 || { echo "error: sbatch not found on PATH" >&2; exit 69; }
exec "${cmd[@]}"
