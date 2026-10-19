#!/usr/bin/env bash
#
# Cluster-agnostic environment shared by every hpc/*.sbatch job. It reads your
# site-specific values from hpc/cluster.env (git-ignored; copy it from
# hpc/cluster.env.example or a profile in hpc/clusters/ and fill it in once),
# sizes the thread pools to the Slurm allocation, gives the job its own
# scratch/cache directories, loads the modules and activates the Python
# environment so the jobs "just work" without per-script edits.
#
# This file is SOURCED and does not set shell options: every .sbatch sets
# `set -euo pipefail` before sourcing it. Failures it cannot recover from
# (a module that will not load, a missing venv) are collected in
# LUNARIS_ENV_FAILS and, unless LUNARIS_ENV_STRICT=0 (hpc/preflight.sh does that
# so it can report every problem), abort the job right here with a message.

# Resolve this script's directory so cluster.env is found regardless of CWD.
_LUNARIS_HPC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"

LUNARIS_ENV_FAILS=()
LUNARIS_ENV_WARNS=()
_lunaris_env_fail() {
  LUNARIS_ENV_FAILS+=("$*")
  if [[ "${LUNARIS_ENV_STRICT:-1}" == "1" ]]; then echo "error: $*" >&2; fi
  return 0
}
_lunaris_env_warn() {
  LUNARIS_ENV_WARNS+=("$*")
  if [[ "${LUNARIS_ENV_STRICT:-1}" == "1" ]]; then echo "warning: $*" >&2; fi
  return 0
}

# 1. Load site configuration if present (does not override values already set in
#    the environment: `sbatch --export` or a manual `export` still wins for the
#    `: "${VAR:=...}"` entries).
if [[ -f "${_LUNARIS_HPC_DIR}/cluster.env" ]]; then
  # shellcheck disable=SC1091
  set -a
  source "${_LUNARIS_HPC_DIR}/cluster.env"
  set +a
else
  _lunaris_env_warn "${_LUNARIS_HPC_DIR}/cluster.env not found: no site settings (account, modules, venv, storage) are applied. Copy hpc/cluster.env.example or a profile from hpc/clusters/ to hpc/cluster.env."
fi

# 2. Project checkout and external storage.
#    LUNARIS_DATA_DIR    is read by the framework to locate external data.
#    LUNARIS_OUTPUT_DIR  is a convenience the example jobs pass to --out-dir /
#                        --output-dir
#                        (the framework does not read it directly).
#    Both default to $HOME so they exist everywhere; point them at your site's
#    scratch/project area in cluster.env when $HOME has a small quota.
export LUNARIS_PROJECT_ROOT="${LUNARIS_PROJECT_ROOT:-$(cd "${_LUNARIS_HPC_DIR}/.." && pwd)}"
export LUNARIS_DATA_DIR="${LUNARIS_DATA_DIR:-$HOME/lunaris_data}"
export LUNARIS_OUTPUT_DIR="${LUNARIS_OUTPUT_DIR:-$HOME/lunaris_outputs}"

# 3. Threads follow the allocation (--cpus-per-task), not a fixed number; outside
#    Slurm they fall back to 1. LUNARIS_THREADS overrides all four at once.
export LUNARIS_THREADS="${LUNARIS_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"
export OMP_NUM_THREADS="${LUNARIS_THREADS}"
export MKL_NUM_THREADS="${LUNARIS_THREADS}"
export OPENBLAS_NUM_THREADS="${LUNARIS_THREADS}"
export NUMBA_NUM_THREADS="${LUNARIS_THREADS}"

# 4. Per-job scratch: TMPDIR and the numba cache live in a directory named after
#    the job, so concurrent jobs never share (or corrupt) a cache. The root is
#    LUNARIS_SCRATCH_DIR from cluster.env, default ${TMPDIR:-/tmp}. An already
#    exported NUMBA_CACHE_DIR wins. The directory is removed when the script that
#    sourced this file exits (so the job scripts do not `exec` their program).
if [[ -z "${LUNARIS_JOB_SCRATCH:-}" ]]; then
  export LUNARIS_SCRATCH_DIR="${LUNARIS_SCRATCH_DIR:-${TMPDIR:-/tmp}}"
  export LUNARIS_JOB_SCRATCH="${LUNARIS_SCRATCH_DIR}/lunaris_${USER:-user}_${SLURM_JOB_ID:-$$}"
  if mkdir -p "${LUNARIS_JOB_SCRATCH}" 2>/dev/null && [[ -w "${LUNARIS_JOB_SCRATCH}" ]]; then
    export TMPDIR="${LUNARIS_JOB_SCRATCH}"
    export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-${LUNARIS_JOB_SCRATCH}/numba_cache}"
    trap 'rm -rf "${LUNARIS_JOB_SCRATCH}"' EXIT
  else
    _lunaris_env_fail "cannot create the per-job scratch directory ${LUNARIS_JOB_SCRATCH} (check LUNARIS_SCRATCH_DIR=${LUNARIS_SCRATCH_DIR})"
  fi
fi

# HDF5 >= 1.10 acquires POSIX file locks by default, which parallel filesystems
# (Lustre, GPFS, some NFS mounts) frequently do not support — reads then fail
# with "unable to lock file". Lunaris training only ever READS its HDF5 clouds
# (read-only + SWMR), so disabling locking is safe here. Site override wins.
export HDF5_USE_FILE_LOCKING="${HDF5_USE_FILE_LOCKING:-FALSE}"

# 5. Load the Environment Modules listed in cluster.env (LUNARIS_MODULES), e.g.
#    "Python/Python-3.11.6-... CUDA/12.4". A `module` command that does not
#    exist, a `module load` that fails, and a module that is missing from
#    `module list` afterwards are all reported instead of skipped.
if [[ -n "${LUNARIS_MODULES:-}" ]]; then
  if ! command -v module >/dev/null 2>&1; then
    # Batch shells often miss the shell function; try the usual init files.
    # Their scripts are not `set -u` clean, so relax it while sourcing.
    _lunaris_nounset=0
    case "$-" in *u*) _lunaris_nounset=1 ;; esac
    set +u
    for _init in /etc/profile.d/modules.sh /etc/profile.d/lmod.sh \
                 /usr/share/Modules/init/bash /usr/share/lmod/lmod/init/bash; do
      if [[ -f "${_init}" ]]; then
        # shellcheck disable=SC1090
        source "${_init}" >/dev/null 2>&1 || true
      fi
      command -v module >/dev/null 2>&1 && break
    done
    [[ ${_lunaris_nounset} -eq 1 ]] && set -u
  fi
  if ! command -v module >/dev/null 2>&1; then
    _lunaris_env_fail "LUNARIS_MODULES is set (${LUNARIS_MODULES}) but the 'module' command does not exist in this shell"
  else
    for _mod in ${LUNARIS_MODULES}; do
      if ! module load "${_mod}"; then
        _lunaris_env_fail "module load ${_mod} failed (find valid names with: module avail)"
      fi
    done
    # Environment Modules 3.x exits 0 on an unknown module; compare against the
    # terse list (names are matched case-insensitively, with or without version).
    _lunaris_loaded="$(module -t list 2>&1 || true)"
    for _mod in ${LUNARIS_MODULES}; do
      _found=0
      while IFS= read -r _line; do
        _line_l="${_line,,}"
        _mod_l="${_mod,,}"
        if [[ "${_line_l}" == "${_mod_l}" || "${_line_l}" == "${_mod_l}"/* ]]; then _found=1; break; fi
      done <<< "${_lunaris_loaded}"
      if [[ ${_found} -eq 0 ]]; then
        _lunaris_env_warn "module ${_mod} is not in 'module list' after loading it"
      fi
    done
  fi
fi

# 6. Activate the Python environment: a conda environment when LUNARIS_CONDA_ENV
#    is set, otherwise the venv directory LUNARIS_VENV (default $HOME/venvs/lunaris,
#    the one hpc/setup_env.sh creates).
export LUNARIS_VENV="${LUNARIS_VENV:-$HOME/venvs/lunaris}"
_lunaris_nounset=0
case "$-" in *u*) _lunaris_nounset=1 ;; esac
if [[ -n "${LUNARIS_CONDA_ENV:-}" ]]; then
  if command -v conda >/dev/null 2>&1; then
    set +u
    # `conda activate` needs the shell hook in a non-interactive job.
    eval "$(conda shell.bash hook)"
    conda activate "${LUNARIS_CONDA_ENV}" || _lunaris_env_fail "conda activate ${LUNARIS_CONDA_ENV} failed"
  else
    _lunaris_env_fail "LUNARIS_CONDA_ENV=${LUNARIS_CONDA_ENV} is set but 'conda' is not on PATH (load its module first)"
  fi
elif [[ -f "${LUNARIS_VENV}/bin/activate" ]]; then
  set +u
  # shellcheck disable=SC1091
  source "${LUNARIS_VENV}/bin/activate"
elif [[ "${LUNARIS_VENV}" == */bin/activate && -f "${LUNARIS_VENV}" ]]; then
  # An older cluster.env named the activate script itself; accept that form.
  set +u
  # shellcheck disable=SC1090
  source "${LUNARIS_VENV}"
else
  _lunaris_env_fail "venv not found: ${LUNARIS_VENV}/bin/activate (create it with hpc/setup_env.sh, or set LUNARIS_VENV / LUNARIS_CONDA_ENV in hpc/cluster.env)"
fi
[[ ${_lunaris_nounset} -eq 1 ]] && set -u

# First-time setup is automated by hpc/setup_env.sh, which creates the venv and
# installs the hash-pinned HPC lock plus the package (see docs/HPC.md). The
# unpinned alternative (version ranges, not reproducible) is the editable
# install  pip install -e ".[hpc]"  (the .[hpc] extra: PyTorch + h5py, no GUI).

# --------------------------------------------------------------------------- #
# Helpers for the job scripts
# --------------------------------------------------------------------------- #

# lunaris_arg_value <--flag> <args...>
#   Prints the value of the LAST `--flag VALUE` / `--flag=VALUE` among the args
#   (argparse "last wins") and returns 0; returns 1 when the flag is absent.
lunaris_arg_value() {
  local flag="$1"; shift
  local prev="" found=1 value="" arg
  for arg in "$@"; do
    if [[ "${prev}" == "${flag}" ]]; then
      value="${arg}"; found=0
    elif [[ "${arg}" == "${flag}="* ]]; then
      value="${arg#"${flag}="}"; found=0
    fi
    prev="${arg}"
  done
  if [[ ${found} -eq 0 ]]; then printf '%s' "${value}"; fi
  return ${found}
}

# lunaris_drop_flag <--flag> <takes-value 0|1> <args...>
#   Sets the array LUNARIS_ARGS_OUT to the args without `--flag` (and, when
#   takes-value is 1, without its value or `--flag=VALUE` form).
lunaris_drop_flag() {
  local flag="$1" takes_value="$2"; shift 2
  local skip=0 arg
  LUNARIS_ARGS_OUT=()
  for arg in "$@"; do
    if [[ ${skip} -eq 1 ]]; then skip=0; continue; fi
    if [[ "${arg}" == "${flag}" ]]; then skip=${takes_value}; continue; fi
    if [[ ${takes_value} -eq 1 && "${arg}" == "${flag}="* ]]; then continue; fi
    LUNARIS_ARGS_OUT+=("${arg}")
  done
  return 0
}

# lunaris_run_stoppable <command...>
#   Runs the command as a child of this shell and relays the walltime
#   SIGTERM/SIGUSR1 (`#SBATCH --signal=B:...` reaches only the batch shell) to
#   it, then returns the child's own exit status. This replaces `exec`, so the
#   script can look at the status afterwards (see lunaris_after_run).
_LUNARIS_CHILD=""
_LUNARIS_RELAYED=0
_lunaris_relay() {
  _LUNARIS_RELAYED=1
  if [[ -n "${_LUNARIS_CHILD}" ]]; then kill -s "$1" "${_LUNARIS_CHILD}" 2>/dev/null || true; fi
  return 0
}
lunaris_run_stoppable() {
  local rc=0 again=0
  _LUNARIS_RELAYED=0
  trap '_lunaris_relay TERM' TERM
  trap '_lunaris_relay USR1' USR1
  "$@" &
  _LUNARIS_CHILD=$!
  wait "${_LUNARIS_CHILD}" || rc=$?
  # A relayed signal ends `wait` early (status > 128) while the child is still
  # saving its checkpoint: keep waiting for it.
  while [[ ${rc} -gt 128 ]] && kill -0 "${_LUNARIS_CHILD}" 2>/dev/null; do
    rc=0
    wait "${_LUNARIS_CHILD}" || rc=$?
  done
  # The child may have exited between the early return and the liveness check;
  # collect its status once more (127: already collected, keep what we have).
  if [[ ${rc} -gt 128 && ${_LUNARIS_RELAYED} -eq 1 ]]; then
    wait "${_LUNARIS_CHILD}" || again=$?
    [[ ${again} -ne 127 ]] && rc=${again}
  fi
  _LUNARIS_CHILD=""
  trap - TERM USR1
  return "${rc}"
}

# lunaris_after_run <exit-status> <resume-command-text>
#   Exit-status 75 (EX_TEMPFAIL) is the trainer's "stopped by a signal after
#   saving ckpt_last" code. It prints the resume command and, when
#   LUNARIS_AUTO_REQUEUE=1, requeues this job (the requeued run resumes itself,
#   see the train sbatch files). Always exits the script with the status.
lunaris_after_run() {
  local rc="$1" hint="$2"
  if [[ "${rc}" -eq 75 ]]; then
    echo "[lunaris] stopped by a signal, checkpoint saved (exit 75)."
    echo "[lunaris] resume with: ${hint}"
    if [[ "${LUNARIS_AUTO_REQUEUE:-0}" == "1" ]]; then
      if [[ -n "${SLURM_JOB_ID:-}" ]] && scontrol requeue "${SLURM_JOB_ID}"; then
        echo "[lunaris] job ${SLURM_JOB_ID} requeued; the next attempt resumes from the saved checkpoint."
      else
        echo "[lunaris] scontrol requeue failed; resubmit with the resume command above." >&2
      fi
    fi
  fi
  exit "${rc}"
}

# lunaris_assert_cuda
#   Fails when torch cannot see a CUDA device (wrong CUDA module, CPU-only wheel).
lunaris_assert_cuda() {
  if ! python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)"; then
    echo "error: this job needs a GPU, but torch cannot see a CUDA device (wrong module, driver too old for the torch build, or no --gres?)" >&2
    return 1
  fi
}

if [[ "${LUNARIS_ENV_STRICT:-1}" == "1" && ${#LUNARIS_ENV_FAILS[@]} -gt 0 ]]; then
  return 1
fi
