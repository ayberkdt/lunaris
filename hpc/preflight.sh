#!/usr/bin/env bash
#
# Pre-submit sanity check. Run this on a login node, or better inside a short
# interactive GPU allocation, BEFORE launching real Slurm jobs, so a
# misconfigured environment fails in seconds instead of after a queued GPU job
# starts.
#
#   bash hpc/preflight.sh
#
# Every check runs (nothing aborts the script) and prints PASS, WARN or FAIL
# with the reason. The script ends with one verdict and exits 1 if any check
# FAILED; WARN never changes the exit code.
#
#   LUNARIS_PREFLIGHT_REQUIRE_GPU=1   run the GPU checks and FAIL when no GPU is
#                                    visible (default: GPU checks run only when
#                                    a GPU is visible; a Slurm GPU allocation or
#                                    LUNARIS_REQUIRE_CUDA=1 also require one)
#   LUNARIS_PREFLIGHT_VERIFY_DATA=1   also run `lunaris-data verify --strict
#                                    --runtime` (slow; recommended once before
#                                    the first real job)
#
# GPU checks (nvidia-smi, torch CUDA build vs driver, a CUDA kernel launch, numba
# libNVVM) need a GPU node: on UHeM use a short interactive job, e.g.
#   srun -p a100q --gres=gpu:1 --cpus-per-task=4 --time=00:10:00 --pty bash

set -uo pipefail

_HPC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"

case "${1:-}" in
  -h|--help)
    awk 'NR > 1 { if (/^#/) { sub(/^# ?/, ""); print } else exit }' "${BASH_SOURCE[0]}"
    exit 0 ;;
esac

n_pass=0; n_warn=0; n_fail=0
fail_lines=()
pass() { echo "  [PASS] $*"; n_pass=$((n_pass + 1)); }
warn() { echo "  [WARN] $*"; n_warn=$((n_warn + 1)); }
fail() { echo "  [FAIL] $*"; n_fail=$((n_fail + 1)); fail_lines+=("$*"); }
section() { echo; echo "-- $* --"; }

# Load the environment exactly like a real job does, but report problems
# instead of aborting at the first one.
export LUNARIS_ENV_STRICT=0
# shellcheck disable=SC1091
source "${_HPC_DIR}/env_template.sh"

echo "==================== Lunaris HPC preflight ===================="
echo "Host         : $(hostname)"
echo "Project root : ${LUNARIS_PROJECT_ROOT}"
echo "Python       : $(python --version 2>&1) @ $(command -v python || echo '<none>')"
echo "Data dir     : ${LUNARIS_DATA_DIR}"
echo "Output dir   : ${LUNARIS_OUTPUT_DIR}"
if [[ -n "${LUNARIS_CONDA_ENV:-}" ]]; then
  echo "Environment  : conda env ${LUNARIS_CONDA_ENV}"
else
  echo "Environment  : venv ${LUNARIS_VENV}"
fi
echo "Job scratch  : ${LUNARIS_JOB_SCRATCH:-<none>}"

# ---- 1. Site configuration ------------------------------------------------ #
section "site configuration"
if [[ -f "${_HPC_DIR}/cluster.env" ]]; then
  pass "hpc/cluster.env found"
  if grep -q $'\r' "${_HPC_DIR}/cluster.env"; then
    fail "hpc/cluster.env has CRLF line endings: every value ends in a carriage return. Fix: sed -i 's/\\r\$//' hpc/cluster.env"
  else
    pass "hpc/cluster.env has LF line endings"
  fi
else
  fail "hpc/cluster.env not found: no account, modules, venv or storage settings are applied. Copy hpc/cluster.env.example (or a profile from hpc/clusters/) to hpc/cluster.env"
fi
if [[ -n "${LUNARIS_MODULES:-}" ]]; then
  pass "modules requested: ${LUNARIS_MODULES}"
else
  warn "LUNARIS_MODULES is empty: no modules are loaded (fine only if Python and CUDA come from the venv)"
fi
_i=0
while [[ ${_i} -lt ${#LUNARIS_ENV_FAILS[@]} ]]; do fail "${LUNARIS_ENV_FAILS[${_i}]}"; _i=$((_i + 1)); done
_i=0
while [[ ${_i} -lt ${#LUNARIS_ENV_WARNS[@]} ]]; do
  case "${LUNARIS_ENV_WARNS[${_i}]}" in
    *"cluster.env not found"*) ;;  # already reported above
    *) warn "${LUNARIS_ENV_WARNS[${_i}]}" ;;
  esac
  _i=$((_i + 1))
done
if [[ ${#LUNARIS_ENV_FAILS[@]} -eq 0 && -n "${LUNARIS_MODULES:-}" ]]; then
  pass "modules loaded without error"
fi

# ---- 2. Scheduler --------------------------------------------------------- #
section "scheduler"
if command -v sbatch >/dev/null 2>&1; then
  pass "sbatch on PATH"
  [[ -n "${LUNARIS_PARTITION:-}" ]] && pass "LUNARIS_PARTITION=${LUNARIS_PARTITION}" \
    || warn "LUNARIS_PARTITION is empty: submit.sh adds no --partition (the cluster default queue is used; find yours with: sinfo -s)"
  [[ -n "${LUNARIS_ACCOUNT:-}" ]] && pass "LUNARIS_ACCOUNT=${LUNARIS_ACCOUNT}" \
    || warn "LUNARIS_ACCOUNT is empty: submit.sh adds no --account (find yours with: sacctmgr show assoc user=\$USER format=account,qos)"
  [[ -n "${LUNARIS_QOS:-}" ]] && pass "LUNARIS_QOS=${LUNARIS_QOS}" \
    || warn "LUNARIS_QOS is empty: submit.sh adds no --qos (fine if your account has a default QOS)"
else
  warn "sbatch not found: this is not a Slurm host, so scheduler settings are not checked"
fi

# ---- 3. Python environment ------------------------------------------------ #
section "python environment"
_python_path="$(command -v python || true)"
if [[ -z "${_python_path}" ]]; then
  fail "no 'python' on PATH: load the Python module and create the venv (hpc/setup_env.sh)"
else
  if [[ -z "${LUNARIS_CONDA_ENV:-}" && "${_python_path}" != "${LUNARIS_VENV}"/* ]]; then
    warn "python is ${_python_path}, not from the venv ${LUNARIS_VENV}"
  else
    pass "python ${_python_path}"
  fi
  _pyver="$(python -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || echo '?')"
  if [[ "${_pyver}" == "3.11" ]]; then
    pass "python ${_pyver} (the HPC lock targets CPython 3.11)"
  else
    warn "python ${_pyver}: the HPC lock locks/requirements-hpc-linux-py311.lock.txt targets CPython 3.11; other versions need an unpinned install"
  fi
  if python -c "import lunaris" 2>/dev/null; then
    pass "import lunaris ($(python -c 'import lunaris; print(getattr(lunaris, "__version__", "?"))' 2>/dev/null))"
  else
    fail "cannot 'import lunaris': run hpc/setup_env.sh and check LUNARIS_VENV in hpc/cluster.env"
  fi
fi
for cmd in lunaris-train lunaris-eval lunaris-benchmark lunaris-batch lunaris-data; do
  if command -v "${cmd}" >/dev/null 2>&1; then
    pass "entry point ${cmd}"
  else
    fail "entry point ${cmd} not on PATH (the package is not installed in this environment)"
  fi
done
pass "threads: ${LUNARIS_THREADS} (SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-unset})"

# ---- 4. Libraries and GPU ------------------------------------------------- #
section "libraries and GPU"
_truthy() { case "${1,,}" in 1|true|yes|on) return 0 ;; *) return 1 ;; esac; }
_active() { [[ -n "${1:-}" && "${1}" != "-1" ]]; }

gpu_visible=0
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L 2>/dev/null | grep -q '^GPU '; then
  gpu_visible=1
fi
gpu_required=0
_truthy "${LUNARIS_PREFLIGHT_REQUIRE_GPU:-0}" && gpu_required=1
_truthy "${LUNARIS_REQUIRE_CUDA:-0}" && gpu_required=1
for _v in "${SLURM_JOB_GPUS:-}" "${SLURM_GPUS:-}" "${SLURM_GPUS_ON_NODE:-}"; do
  _active "${_v}" && gpu_required=1
done

run_gpu=0
if [[ ${gpu_visible} -eq 1 || ${gpu_required} -eq 1 ]]; then run_gpu=1; fi
if [[ ${run_gpu} -eq 0 ]]; then
  warn "no GPU visible on this host: GPU checks skipped (expected on a login node; run inside a GPU job or set LUNARIS_PREFLIGHT_REQUIRE_GPU=1)"
fi

if [[ -n "${_python_path}" ]]; then
  while IFS=$'\t' read -r level msg; do
    case "${level}" in
      PASS) pass "${msg}" ;;
      WARN) warn "${msg}" ;;
      FAIL) fail "${msg}" ;;
      *)    [[ -n "${level}" ]] && echo "         ${level} ${msg}" ;;
    esac
  done < <(PF_GPU="${run_gpu}" python - <<'PY' 2>&1
import os
import shutil
import subprocess


def out(level, msg):
    print(f"{level}\t{msg}", flush=True)


for mod in ("numpy", "h5py"):
    try:
        m = __import__(mod)
        out("PASS", f"{mod} {getattr(m, '__version__', '?')}")
    except Exception as exc:
        out("FAIL", f"cannot import {mod}: {exc}")

try:
    import torch
except Exception as exc:
    out("FAIL", f"cannot import torch: {exc} (check the HPC lock install)")
    torch = None

if torch is not None:
    cuda_build = torch.version.cuda
    out("PASS", f"torch {torch.__version__} (built for CUDA {cuda_build})")

if os.environ.get("PF_GPU") == "1":
    # nvidia-smi: is there a driver and a GPU?
    driver = None
    smi = shutil.which("nvidia-smi")
    if smi is None:
        out("FAIL", "nvidia-smi not found: not a GPU node, or no NVIDIA driver (request --gres=gpu:1 and the right partition)")
    else:
        r = subprocess.run(
            [smi, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=120, check=False,
        )
        lines = [ln.strip() for ln in r.stdout.splitlines() if ln.strip()]
        if r.returncode != 0 or not lines:
            out("FAIL", f"nvidia-smi found no GPU (exit {r.returncode}): {r.stderr.strip() or r.stdout.strip()}")
        else:
            name, driver, mem = [p.strip() for p in lines[0].split(",")][:3]
            out("PASS", f"nvidia-smi: {len(lines)} GPU(s), gpu0={name}, driver {driver}, {mem}")

    if torch is not None:
        # Driver vs the CUDA build torch was compiled for. Minimum Linux driver
        # per CUDA major (NVIDIA CUDA compatibility notes): 11.x >= 450,
        # 12.x >= 525, 13.x >= 580.
        need = {11: 450, 12: 525, 13: 580}
        if torch.version.cuda is None:
            out("FAIL", "torch is a CPU-only build (torch.version.cuda is None): reinstall from the HPC lock or a CUDA index (LUNARIS_TORCH_INDEX_URL)")
        elif driver is not None:
            cuda_major = int(torch.version.cuda.split(".")[0])
            driver_major = int(driver.split(".")[0])
            if cuda_major not in need:
                out("WARN", f"torch CUDA {torch.version.cuda}: no minimum driver known for CUDA {cuda_major}, driver is {driver}")
            elif driver_major < need[cuda_major]:
                out("FAIL", (
                    f"driver {driver} is too old for torch built with CUDA {torch.version.cuda} "
                    f"(needs >= {need[cuda_major]}): install a torch build for an older CUDA "
                    "(LUNARIS_TORCH_INDEX_URL=<pytorch wheel index> bash hpc/setup_env.sh) or use a node with a newer driver"
                ))
            else:
                out("PASS", f"driver {driver} >= {need[cuda_major]} required by torch CUDA {torch.version.cuda}")

        if torch.cuda.is_available():
            out("PASS", f"torch sees {torch.cuda.device_count()} CUDA device(s): {torch.cuda.get_device_name(0)}, capability {torch.cuda.get_device_capability(0)}")
            try:
                total = float((torch.ones(1024, device="cuda") * 2).sum().item())
                out("PASS" if total == 2048.0 else "FAIL", f"CUDA kernel launch (sum of 1024 x 2 = {total})")
            except Exception as exc:
                out("FAIL", f"CUDA kernel launch failed: {exc}")
        else:
            out("FAIL", "torch.cuda.is_available() is False: wrong CUDA module, driver too old for this torch build, or no GPU allocated")

    # numba CUDA needs libNVVM (the numba_cuda_sh batch backend); training does not.
    try:
        import numba
        from numba import cuda
        from numba.cuda.cudadrv import nvvm

        version = nvvm.NVVM().get_version()
        out("PASS", f"numba {numba.__version__} CUDA libNVVM {version}, cuda.is_available()={cuda.is_available()}")
    except Exception as exc:
        out("WARN", f"numba CUDA / libNVVM unavailable: {exc} (only the numba_cuda_sh batch backend needs it; load a CUDA toolkit module or use torch_cuda_sh)")
PY
)
fi

# ---- 5. Storage ----------------------------------------------------------- #
section "storage"
if [[ -d "${LUNARIS_DATA_DIR}" ]]; then
  pass "data dir exists: ${LUNARIS_DATA_DIR}"
else
  warn "data dir missing: ${LUNARIS_DATA_DIR} (create it, then run: lunaris-data download --group ephemeris)"
fi
# Files, not just directories. The layout is data_sources.json's groups:
# ephemeris -> ephemeris_models/, gravity -> gravity_models/, datasets -> datasets/.
# data_sources.json also accepts each file under its text-wrapped alias (<name>.txt).
for rel in ephemeris_models/naif0012.tls ephemeris_models/de440.bsp ephemeris_models/pck00011.tpc; do
  if [[ -s "${LUNARIS_DATA_DIR}/${rel}" || -s "${LUNARIS_DATA_DIR}/${rel}.txt" ]]; then
    pass "data file ${rel}"
  else
    fail "data file missing or empty: ${LUNARIS_DATA_DIR}/${rel} (lunaris-data download --group ephemeris)"
  fi
done
if [[ -s "${LUNARIS_DATA_DIR}/gravity_models/jggrx_1800f_sha.tab" || -s "${LUNARIS_DATA_DIR}/gravity_models/jggrx_1800f_sha.tab.txt" ]]; then
  pass "data file gravity_models/jggrx_1800f_sha.tab"
else
  warn "lunar gravity file missing: ${LUNARIS_DATA_DIR}/gravity_models/jggrx_1800f_sha.tab (needed by lunar spherical-harmonic runs: lunaris-data download --group gravity)"
fi
# Depth 3 reaches a staged suite: datasets/cloud_suites/<suite id>/*.h5.
_n_h5="$(find "${LUNARIS_DATA_DIR}/datasets" -maxdepth 3 -name '*.h5' -size +0 2>/dev/null | wc -l)"
if [[ "${_n_h5}" -gt 0 ]]; then
  pass "${_n_h5} non-empty HDF5 cloud file(s) under ${LUNARIS_DATA_DIR}/datasets"
else
  warn "no *.h5 under ${LUNARIS_DATA_DIR}/datasets: training and evaluation jobs need cloud-suite files (generate or copy them there)"
fi
if mkdir -p "${LUNARIS_OUTPUT_DIR}" 2>/dev/null && [[ -w "${LUNARIS_OUTPUT_DIR}" ]]; then
  pass "output dir writable: ${LUNARIS_OUTPUT_DIR}"
  _free_kb="$(df -Pk "${LUNARIS_OUTPUT_DIR}" 2>/dev/null | awk 'NR == 2 {print $4}')"
  if [[ -n "${_free_kb}" ]]; then
    _free_gb=$((_free_kb / 1024 / 1024))
    if [[ ${_free_gb} -lt 20 ]]; then
      warn "only ${_free_gb} GB free under ${LUNARIS_OUTPUT_DIR} (df; a user quota is not visible here, check it with: quota -s)"
    else
      pass "${_free_gb} GB free under ${LUNARIS_OUTPUT_DIR} (df; a user quota is not visible here, check it with: quota -s)"
    fi
  fi
else
  fail "output dir not writable: ${LUNARIS_OUTPUT_DIR}"
fi

# Strict data verification (opt-in): proves the SPICE/gravity assets are present
# and runtime-readable, which the file checks above do not.
if _truthy "${LUNARIS_PREFLIGHT_VERIFY_DATA:-0}"; then
  if command -v lunaris-data >/dev/null 2>&1 && lunaris-data verify --strict --runtime; then
    pass "strict runtime data verification (lunaris-data verify --strict --runtime)"
  else
    fail "strict runtime data verification failed (lunaris-data verify --strict --runtime)"
  fi
else
  warn "strict data verification skipped (set LUNARIS_PREFLIGHT_VERIFY_DATA=1 to enable)"
fi

# ---- Verdict -------------------------------------------------------------- #
echo
echo "=============================================================="
echo "Summary: ${n_pass} PASS, ${n_warn} WARN, ${n_fail} FAIL"
if [[ ${n_fail} -eq 0 ]]; then
  echo "PREFLIGHT VERDICT: PASS (ready to submit with hpc/submit.sh; review the WARN lines above)"
else
  echo "PREFLIGHT VERDICT: FAIL" >&2
  for line in "${fail_lines[@]}"; do echo "  - ${line}" >&2; done
fi
echo "=============================================================="
[[ ${n_fail} -eq 0 ]]
