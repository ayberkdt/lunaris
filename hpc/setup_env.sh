#!/usr/bin/env bash
#
# One-shot, run-once-per-cluster setup for the headless Lunaris / ST-LRPS HPC
# stack. Creates a GUI-free virtual environment and installs the hash-pinned HPC
# lock (locks/requirements-hpc-linux-py311.lock.txt: torch + h5py + the engine
# core, no Qt) and then the package itself without resolving dependencies again.
# Run this on a login node, NOT inside a compute job.
#
#   cp hpc/cluster.env.example hpc/cluster.env   # then fill it in
#   bash hpc/setup_env.sh
#
# The venv is LUNARIS_VENV (hpc/cluster.env), default $HOME/venvs/lunaris; the jobs
# and hpc/preflight.sh use the same variable, so there is nothing to paste back.
# The lock targets CPython 3.11 on Linux x86_64 (locks/README.md): load a Python
# 3.11 module first (LUNARIS_MODULES) or point PYTHON_BIN at a 3.11 interpreter.
#
# The lock pins torch 2.13 built for CUDA 13, which needs an NVIDIA driver >= 580
# on the GPU nodes. For a node with an older driver, install a torch build for an
# older CUDA from a PyTorch wheel index instead:
#
#   LUNARIS_TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126 bash hpc/setup_env.sh
#
# (use the index that matches the driver and publishes the pinned torch version;
# see pytorch.org). Then torch comes from that index and the lock's torch, triton,
# nvidia-* and cuda-* pins are skipped.

set -euo pipefail

_HPC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
_PROJECT_ROOT="$(cd "${_HPC_DIR}/.." && pwd)"
_LOCK="${_PROJECT_ROOT}/locks/requirements-hpc-linux-py311.lock.txt"

# cluster.env, modules and the per-job scratch/TMPDIR (pip builds in it) come
# from the shared environment template. The venv does not exist yet, so only
# that one failure is expected; anything else (a module that will not load)
# stops the setup.
export LUNARIS_ENV_STRICT=0
# shellcheck disable=SC1091
source "${_HPC_DIR}/env_template.sh"
_i=0
while [[ ${_i} -lt ${#LUNARIS_ENV_FAILS[@]} ]]; do
  case "${LUNARIS_ENV_FAILS[${_i}]}" in
    "venv not found"*) ;;
    *) echo "error: ${LUNARIS_ENV_FAILS[${_i}]}" >&2; exit 1 ;;
  esac
  _i=$((_i + 1))
done
if [[ -n "${LUNARIS_CONDA_ENV:-}" ]]; then
  echo "error: LUNARIS_CONDA_ENV is set, but this script builds the venv LUNARIS_VENV. Clear LUNARIS_CONDA_ENV in hpc/cluster.env, or install into the conda env by hand (docs/HPC.md)." >&2
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${LUNARIS_VENV}"

echo "==> Python      : $("${PYTHON_BIN}" --version 2>&1) @ $(command -v "${PYTHON_BIN}")"
echo "==> Project root: ${_PROJECT_ROOT}"
echo "==> Venv target : ${VENV_DIR}"
echo "==> Lock        : ${_LOCK}"

_pyver="$("${PYTHON_BIN}" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
if [[ "${_pyver}" != "3.11" ]]; then
  echo "error: ${PYTHON_BIN} is Python ${_pyver}, but the HPC lock targets CPython 3.11. Load a Python 3.11 module (LUNARIS_MODULES) or set PYTHON_BIN." >&2
  exit 1
fi
[[ -f "${_LOCK}" ]] || { echo "error: lock file not found: ${_LOCK}" >&2; exit 66; }

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "==> Creating virtual environment ..."
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
else
  echo "==> Reusing existing virtual environment."
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip

cd "${_PROJECT_ROOT}"
if [[ -n "${LUNARIS_TORCH_INDEX_URL:-}" ]]; then
  _torch_pin="$(awk -F'[ =]+' '/^torch==/ {print $1 "==" $2; exit}' "${_LOCK}")"
  # In a reused venv the lock's build (same version, no local tag) already
  # satisfies the pin, so pip would keep it: remove it first, with the lock's
  # CUDA 13 runtime packages (nvidia-*, cuda-*) that the other build replaces.
  _old_stack="$(python -m pip freeze 2>/dev/null | awk -F'[=@ ]' 'tolower($1) ~ /^(torch|triton|nvidia-|cuda-)/ {print $1}')"
  if [[ -n "${_old_stack}" ]]; then
    echo "==> Removing the installed torch stack so the ${LUNARIS_TORCH_INDEX_URL} build replaces it:"
    echo "    ${_old_stack//$'\n'/ }"
    # shellcheck disable=SC2086
    python -m pip uninstall -y ${_old_stack}
  fi
  echo "==> Installing ${_torch_pin} from ${LUNARIS_TORCH_INDEX_URL}"
  python -m pip install --index-url "${LUNARIS_TORCH_INDEX_URL}" "${_torch_pin}"
  # The rest of the lock, without the torch stack that the index just provided.
  _filtered="${TMPDIR}/lunaris_hpc_lock_without_torch.txt"
  awk '
    /^[A-Za-z0-9][A-Za-z0-9._-]*==/ {
      name = $1; sub(/==.*/, "", name)
      skip = (name == "torch" || name == "triton" || name ~ /^nvidia-/ || name ~ /^cuda-/)
    }
    !skip { print }
  ' "${_LOCK}" > "${_filtered}"
  echo "==> Installing the rest of the HPC lock (hash-verified)"
  python -m pip install --require-hashes --no-deps -r "${_filtered}"
else
  echo "==> Installing the HPC lock (hash-verified)"
  python -m pip install --require-hashes -r "${_LOCK}"
fi

echo "==> Installing the package itself (editable, dependencies already installed)"
python -m pip install --no-deps -e .

echo
echo "==> Installed torch build"
python - <<'PY'
import torch

print(f"    torch {torch.__version__}, built for CUDA {torch.version.cuda}")
PY

echo
echo "==> Verifying headless entry points ..."
_missing=0
for cmd in lunaris-train lunaris-eval lunaris-benchmark lunaris-batch lunaris-data; do
  if command -v "${cmd}" >/dev/null 2>&1; then
    echo "    ok   ${cmd}"
  else
    echo "    MISSING ${cmd}" >&2
    _missing=1
  fi
done

echo
echo "Done. Jobs activate ${VENV_DIR} through LUNARIS_VENV."
if [[ "${VENV_DIR}" != "${HOME}/venvs/lunaris" ]]; then
  echo "This is not the default path: set it in hpc/cluster.env:"
  echo "  LUNARIS_VENV=\"${VENV_DIR}\""
fi
echo "Then run a pre-submit check:  bash hpc/preflight.sh"
exit "${_missing}"
