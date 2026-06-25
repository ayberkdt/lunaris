#!/usr/bin/env bash
set -euo pipefail

# Cluster-agnostic environment shared by every hpc/*.sbatch job. It reads your
# site-specific values from hpc/cluster.env (git-ignored; copy it from
# hpc/cluster.env.example and fill it in once), then loads modules and activates
# the Python environment so the jobs "just work" without per-script edits.
#
# Anything not set in cluster.env is silently skipped, so this template stays
# safe to source on a laptop or a login node with nothing configured yet.

# Resolve this script's directory so cluster.env is found regardless of CWD.
_LUNARIS_HPC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"

# 1. Load site configuration if present (does not override values already set in
#    the environment — `sbatch --export` or a manual `export` still wins).
if [[ -f "${_LUNARIS_HPC_DIR}/cluster.env" ]]; then
  # shellcheck disable=SC1091
  set -a
  source "${_LUNARIS_HPC_DIR}/cluster.env"
  set +a
fi

# 2. Project checkout and external storage.
#    LUNARIS_DATA_DIR    is read by the framework to locate external data.
#    LUNARIS_OUTPUT_DIR  is a convenience the example jobs pass to --out-dir /
#                        --output-dir (the framework does not read it directly).
export LUNARIS_PROJECT_ROOT="${LUNARIS_PROJECT_ROOT:-$(cd "${_LUNARIS_HPC_DIR}/.." && pwd)}"
export LUNARIS_DATA_DIR="${LUNARIS_DATA_DIR:-/scratch/$USER/lunaris_data}"
export LUNARIS_OUTPUT_DIR="${LUNARIS_OUTPUT_DIR:-/scratch/$USER/lunaris_outputs}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

# 3. Load Environment Modules listed in cluster.env (LUNARIS_MODULES), e.g.
#    "cuda/12.1 cudnn/8.9". No-op when the variable is empty or `module` is
#    unavailable.
if [[ -n "${LUNARIS_MODULES:-}" ]] && command -v module >/dev/null 2>&1; then
  for _mod in ${LUNARIS_MODULES}; do
    module load "${_mod}"
  done
fi

# 4. Activate the Python environment: prefer an explicit venv activate script,
#    fall back to a conda environment name. No-op when neither is set.
if [[ -n "${LUNARIS_VENV:-}" ]]; then
  # shellcheck disable=SC1090
  source "${LUNARIS_VENV}"
elif [[ -n "${LUNARIS_CONDA_ENV:-}" ]] && command -v conda >/dev/null 2>&1; then
  # `conda activate` needs the shell hook in a non-interactive job.
  eval "$(conda shell.bash hook)"
  conda activate "${LUNARIS_CONDA_ENV}"
fi

# First-time setup is automated by hpc/setup_env.sh, which creates the venv and
# runs the headless editable install:  pip install -e ".[hpc]"
# (the .[hpc] extra installs PyTorch + h5py and omits all GUI packages).
