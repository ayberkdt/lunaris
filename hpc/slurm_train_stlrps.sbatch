#!/usr/bin/env bash
#SBATCH --job-name=lunaris-train
#SBATCH --output=lunaris_train_%j.out
#SBATCH --error=lunaris_train_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
# One Python process on one node (no srun/MPI). Stated explicitly: the UHeM
# job examples give -N and -n on every job.
#SBATCH --nodes=1
#SBATCH --ntasks=1
# Send SIGTERM to the batch shell (B:) 30 minutes before the walltime limit so
# the trainer can finish the current epoch (including its validation), save
# ckpt_last, and exit 75 (resumable) instead of being killed mid-epoch. If one
# epoch plus validation takes longer than this lead time the job is killed
# unsaved: raise it (LUNARIS_SIGNAL="B:TERM@3600" in cluster.env) for long epochs.
#SBATCH --signal=B:TERM@1800
# Allow `scontrol requeue` (LUNARIS_AUTO_REQUEUE=1) and keep one log across restarts.
#SBATCH --requeue
#SBATCH --open-mode=append

set -euo pipefail
# Slurm runs a spooled copy of this script, so $0 does not point into hpc/:
# take the folder from submit.sh (LUNARIS_HPC_DIR), else the submit directory.
_hpc_dir="${LUNARIS_HPC_DIR:-${SLURM_SUBMIT_DIR:-.}/hpc}"
[[ -f "${_hpc_dir}/env_template.sh" ]] || _hpc_dir="$(dirname "$0")"
source "${_hpc_dir}/env_template.sh"

# This template requests a GPU, so a silent CPU fallback (wrong CUDA module /
# CPU-only torch wheel) must abort at preflight instead of burning walltime.
export LUNARIS_REQUIRE_CUDA="${LUNARIS_REQUIRE_CUDA:-1}"

# The run directory is what a resume needs: --out / --out-dir (the trainer's two
# spellings of one option), or the --resume-from the caller already gave.
run_dir="$(lunaris_arg_value --out "$@" || lunaris_arg_value --out-dir "$@" || lunaris_arg_value --resume-from "$@" || true)"
if [[ "${LUNARIS_AUTO_REQUEUE:-0}" == "1" && -z "${run_dir}" ]]; then
  echo "error: LUNARIS_AUTO_REQUEUE=1 needs an explicit --out-dir, otherwise a requeued run cannot find its checkpoint" >&2
  exit 64
fi

args=("$@")
# A requeued attempt (SLURM_RESTART_COUNT > 0) continues from the checkpoint the
# stopped attempt saved; without one it starts over with the original arguments.
if [[ "${SLURM_RESTART_COUNT:-0}" -gt 0 ]] && ! lunaris_arg_value --resume-from "$@" >/dev/null; then
  if [[ -n "${run_dir}" && -f "${run_dir}/checkpoints/ckpt_last.pt" ]]; then
    args+=(--resume-from "${run_dir}")
    echo "[lunaris] restart ${SLURM_RESTART_COUNT}: resuming from ${run_dir}/checkpoints/ckpt_last.pt"
  else
    echo "[lunaris] restart ${SLURM_RESTART_COUNT}: no checkpoint under '${run_dir}', starting from the original arguments"
  fi
fi

# Not exec: the script must see the exit status. lunaris_run_stoppable relays the
# walltime SIGTERM/SIGUSR1 (which reach only this shell) to the trainer.
rc=0
lunaris_run_stoppable lunaris-train ${args[@]+"${args[@]}"} || rc=$?
# The resume command repeats the original arguments: resume locks only the
# architecture, so a bare --resume-from would fall back to default optimisation
# settings (batch size, learning rate, loss weights).
lunaris_drop_flag --resume-from 1 "$@"
_resume_text=""
if [[ ${#LUNARIS_ARGS_OUT[@]} -gt 0 ]]; then _resume_text="$(printf '%q ' "${LUNARIS_ARGS_OUT[@]}")"; fi
lunaris_after_run "${rc}" "hpc/submit.sh train -- ${_resume_text}--resume-from ${run_dir:-<run_dir>}"
