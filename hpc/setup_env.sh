#!/usr/bin/env bash
#
# One-shot, run-once-per-cluster setup for the headless Lunaris / ST-LRPS HPC
# stack. Creates a GUI-free virtual environment and installs the .[hpc] extra
# (PyTorch + h5py, no Qt). Run this on a login node, NOT inside a compute job.
#
#   cp hpc/cluster.env.example hpc/cluster.env   # then fill it in
#   bash hpc/setup_env.sh
#
# By default the venv is created at $HOME/venvs/lunaris. Override with:
#   LUNARIS_VENV_DIR=/path/to/venv bash hpc/setup_env.sh
# After it finishes, point cluster.env at the venv it printed:
#   LUNARIS_VENV="$HOME/venvs/lunaris/bin/activate"

set -euo pipefail

_HPC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
_PROJECT_ROOT="$(cd "${_HPC_DIR}/.." && pwd)"

# Load modules from cluster.env (e.g. a CUDA/Python module) before building the
# venv, so the interpreter and toolchain match the cluster.
if [[ -f "${_HPC_DIR}/cluster.env" ]]; then
  set -a; source "${_HPC_DIR}/cluster.env"; set +a
fi
if [[ -n "${LUNARIS_MODULES:-}" ]] && command -v module >/dev/null 2>&1; then
  for _mod in ${LUNARIS_MODULES}; do module load "${_mod}"; done
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${LUNARIS_VENV_DIR:-$HOME/venvs/lunaris}"

echo "==> Python      : $("${PYTHON_BIN}" --version 2>&1) @ $(command -v "${PYTHON_BIN}")"
echo "==> Project root: ${_PROJECT_ROOT}"
echo "==> Venv target : ${VENV_DIR}"

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "==> Creating virtual environment ..."
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
else
  echo "==> Reusing existing virtual environment."
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip

echo "==> Installing headless HPC extra (editable):  pip install -e \".[hpc]\""
echo "    NOTE: for CUDA builds, install torch from the matching PyTorch index"
echo "    first (see docs/HPC.md), then re-run this script."
cd "${_PROJECT_ROOT}"
python -m pip install -e ".[hpc]"

echo
echo "==> Verifying headless entry points ..."
for cmd in lunaris-train lunaris-eval lunaris-benchmark lunaris-batch lunaris-data; do
  if command -v "${cmd}" >/dev/null 2>&1; then
    echo "    ok   ${cmd}"
  else
    echo "    MISSING ${cmd}" >&2
  fi
done

echo
echo "Done. Add this line to hpc/cluster.env so jobs activate the venv:"
echo "  LUNARIS_VENV=\"${VENV_DIR}/bin/activate\""
echo "Then run a pre-submit check:  bash hpc/preflight.sh"
