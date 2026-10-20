#!/usr/bin/env bash
#SBATCH --job-name=lunaris-scenario
#SBATCH --output=lunaris_scenario_%A_%a.out
#SBATCH --error=lunaris_scenario_%A_%a.err
# --- Resource defaults ------------------------------------------------------ #
# Do not edit these per site: partition, account, QOS, gres and the resource
# sizes come from hpc/cluster.env through `hpc/submit.sh`, and any flag passed
# to `hpc/submit.sh` / `sbatch` overrides them. ST-LRPS training is GPU-heavy
# and single-GPU (the trainer drives one device), so one GPU is requested.
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
# Send SIGTERM 30 minutes before walltime so each training task can finish its
# current epoch (including validation), save ckpt_last, and exit 75 (resumable)
# instead of dying mid-epoch. An epoch plus validation longer than this lead
# time is killed unsaved: raise it (LUNARIS_SIGNAL="B:TERM@3600") for long epochs.
#SBATCH --signal=B:TERM@1800
# Allow `scontrol requeue` (LUNARIS_AUTO_REQUEUE=1) and keep one log across restarts.
#SBATCH --requeue
#SBATCH --open-mode=append

set -euo pipefail

# Source the cluster-agnostic environment template (paths, threads, modules).
# Slurm runs a spooled copy of this script, so $0 does not point into hpc/:
# take the folder from submit.sh (LUNARIS_HPC_DIR), else the submit directory.
_hpc_dir="${LUNARIS_HPC_DIR:-${SLURM_SUBMIT_DIR:-.}/hpc}"
[[ -f "${_hpc_dir}/env_template.sh" ]] || _hpc_dir="$(dirname "$0")"
source "${_hpc_dir}/env_template.sh"

# This template requests a GPU, so a silent CPU fallback (wrong CUDA module /
# CPU-only torch wheel) must abort at preflight instead of burning walltime.
export LUNARIS_REQUIRE_CUDA="${LUNARIS_REQUIRE_CUDA:-1}"

# The scenario JSONL file is the first positional argument; everything after it
# is forwarded to the launcher as common flags applied to every array task.
if [[ $# -lt 1 ]]; then
  echo "usage: sbatch --array=0-N $0 <scenarios.jsonl> [common flags...]" >&2
  exit 64
fi

echo "==================== Scenario array diagnostics ===================="
echo "Job ID            : ${SLURM_JOB_ID:-<none>} (array ${SLURM_ARRAY_JOB_ID:-?}:${SLURM_ARRAY_TASK_ID:-?}, restart ${SLURM_RESTART_COUNT:-0})"
echo "Host              : $(hostname)"
echo "Working directory : ${LUNARIS_PROJECT_ROOT:-$PWD}"
echo "Python            : $(python --version 2>&1) @ $(command -v python)"
echo "Threads           : ${LUNARIS_THREADS} (cpus-per-task ${SLURM_CPUS_PER_TASK:-?})"
echo "--- Package versions ---"
python - <<'PY' || true
import importlib
for mod in ("numpy", "torch"):
    try:
        m = importlib.import_module(mod)
        print(f"{mod:>8}: {getattr(m, '__version__', '?')}")
    except Exception as exc:  # pragma: no cover - diagnostics only
        print(f"{mod:>8}: not available ({exc})")
try:
    import torch
    print(f"   cuda: available={torch.cuda.is_available()} "
          f"device_count={torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"   gpu0: {torch.cuda.get_device_name(0)}")
except Exception as exc:  # pragma: no cover - diagnostics only
    print(f"   cuda: query failed ({exc})")
PY
echo "--- nvidia-smi ---"
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || echo "nvidia-smi not available"
echo "===================================================================="

cd "${LUNARIS_PROJECT_ROOT:-$PWD}"

# The resume form of the arguments: --overwrite/--force would wipe the run a
# stopped attempt just saved (and the launcher rejects them together with
# --resume), so they are dropped; --resume continues each task from its own
# checkpoint and starts fresh when there is none.
lunaris_drop_flag --overwrite 0 "$@"
resume_args=(${LUNARIS_ARGS_OUT[@]+"${LUNARIS_ARGS_OUT[@]}"})
lunaris_drop_flag --force 0 ${resume_args[@]+"${resume_args[@]}"}
resume_args=(${LUNARIS_ARGS_OUT[@]+"${LUNARIS_ARGS_OUT[@]}"})
lunaris_drop_flag --resume 0 ${resume_args[@]+"${resume_args[@]}"}
resume_args=(${LUNARIS_ARGS_OUT[@]+"${LUNARIS_ARGS_OUT[@]}"})
# --restart-if-no-checkpoint: a task requeued by Slurm itself (node failure,
# preemption) before its first checkpoint starts again instead of being refused.
lunaris_drop_flag --restart-if-no-checkpoint 0 ${resume_args[@]+"${resume_args[@]}"}
resume_args=(${LUNARIS_ARGS_OUT[@]+"${LUNARIS_ARGS_OUT[@]}"} --resume --restart-if-no-checkpoint)

args=("$@")
# A requeued attempt (SLURM_RESTART_COUNT > 0) is such a continuation.
if [[ "${SLURM_RESTART_COUNT:-0}" -gt 0 ]]; then
  args=("${resume_args[@]}")
  echo "[lunaris] restart ${SLURM_RESTART_COUNT}: launcher runs with --resume"
fi

# Select and launch the scenario matching $SLURM_ARRAY_TASK_ID. Not exec: the
# script must see the exit status (75 = stopped by a signal after saving
# ckpt_last). lunaris_run_stoppable relays the walltime SIGTERM/SIGUSR1, which
# reach only this shell, to the launcher, which relays them to the trainer.
rc=0
lunaris_run_stoppable python tools/hpc/run_training_scenario.py "${args[@]}" || rc=$?
_resume_text="$(printf '%q ' "${resume_args[@]}")"
lunaris_after_run "${rc}" "hpc/submit.sh scenario --array=${SLURM_ARRAY_TASK_ID:-0} -- ${_resume_text% }"
