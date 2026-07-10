#!/usr/bin/env bash
#
# Pre-submit sanity check. Run this on a login node (or as a tiny interactive
# job) BEFORE launching real Slurm jobs, so a misconfigured environment fails in
# seconds instead of after a queued GPU job starts.
#
#   bash hpc/preflight.sh
#
# Exit code is non-zero if any required check fails. CUDA-not-available is a
# warning (you may be on a CPU login node), not a hard failure.

set -uo pipefail

_HPC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"

# Activate modules + environment exactly like a real job does.
# shellcheck disable=SC1091
source "${_HPC_DIR}/env_template.sh"

fail=0
ok()   { echo "  [ ok ] $*"; }
warn() { echo "  [warn] $*"; }
bad()  { echo "  [FAIL] $*"; fail=1; }

echo "==================== Lunaris HPC preflight ===================="
echo "Host        : $(hostname)"
echo "Project root: ${LUNARIS_PROJECT_ROOT}"
echo "Python      : $(python --version 2>&1) @ $(command -v python || echo '<none>')"
echo "Data dir    : ${LUNARIS_DATA_DIR}"
echo "Output dir  : ${LUNARIS_OUTPUT_DIR}"
echo "--------------------------------------------------------------"

# 1. Python + core import.
if python -c "import lunaris" 2>/dev/null; then
  ok "import lunaris"
else
  bad "cannot 'import lunaris' — run hpc/setup_env.sh and set LUNARIS_VENV in cluster.env"
fi

# 2. Headless entry points.
for cmd in lunaris-train lunaris-eval lunaris-benchmark lunaris-batch lunaris-data; do
  if command -v "${cmd}" >/dev/null 2>&1; then
    ok "entry point ${cmd}"
  else
    bad "entry point ${cmd} not on PATH"
  fi
done

# 3. Torch + CUDA visibility. A login-node check remains a warning by default,
#    but an explicit GPU demand (LUNARIS_REQUIRE_CUDA or a scheduler GPU
#    allocation) fails closed just like the trainer's preflight gate.
python - <<'PY'
import os
import sys

def _active(value):
    return str(value or '').strip() not in ('', '-1')

required = str(os.environ.get('LUNARIS_REQUIRE_CUDA', '')).strip().lower() in {
    '1', 'true', 'yes', 'on'
}
allocated = any(
    key in os.environ and _active(os.environ.get(key))
    for key in ('SLURM_JOB_GPUS', 'SLURM_GPUS', 'SLURM_GPUS_ON_NODE', 'CUDA_VISIBLE_DEVICES')
)
try:
    import torch
except Exception as exc:  # pragma: no cover - diagnostics only
    print(f"  [FAIL] import torch failed: {exc}")
    sys.exit(3)
print(f"  [ ok ] torch {torch.__version__}")
if torch.cuda.is_available():
    print(f"  [ ok ] CUDA available: {torch.cuda.device_count()} device(s), "
          f"gpu0={torch.cuda.get_device_name(0)}")
else:
    if required or allocated:
        print("  [FAIL] CUDA is required by this environment, but torch cannot see a CUDA device")
        sys.exit(4)
    print("  [warn] CUDA not available here (fine on a CPU login node; "
          "GPU jobs must run on a GPU partition)")
PY
torch_rc=$?
[[ ${torch_rc} -eq 3 ]] && bad "torch is not importable — check the .[hpc] install"
[[ ${torch_rc} -eq 4 ]] && bad "CUDA is required, but torch cannot see a CUDA device"
[[ ${torch_rc} -ne 0 && ${torch_rc} -ne 3 && ${torch_rc} -ne 4 ]] && \
  bad "torch/CUDA diagnostics failed (exit ${torch_rc})"

# 4. Storage: data dir present, output dir writable.
if [[ -d "${LUNARIS_DATA_DIR}" ]]; then
  ok "data dir exists: ${LUNARIS_DATA_DIR}"
else
  warn "data dir missing: ${LUNARIS_DATA_DIR} (create it / run 'lunaris-data download')"
fi
if mkdir -p "${LUNARIS_OUTPUT_DIR}" 2>/dev/null && [[ -w "${LUNARIS_OUTPUT_DIR}" ]]; then
  ok "output dir writable: ${LUNARIS_OUTPUT_DIR}"
else
  bad "output dir not writable: ${LUNARIS_OUTPUT_DIR}"
fi

# 5. Optional strict data verification (opt-in). The default checks above only
#    confirm the data dir exists; they do not prove the SPICE/gravity assets are
#    present and runtime-readable. Set LUNARIS_PREFLIGHT_VERIFY_DATA=1 to run the
#    full strict runtime check here and fail closed before spending queue time.
if [[ "${LUNARIS_PREFLIGHT_VERIFY_DATA:-0}" == "1" ]]; then
  if lunaris-data verify --strict --runtime; then
    ok "strict runtime data verification passed"
  else
    bad "strict runtime data verification failed (lunaris-data verify --strict --runtime)"
  fi
else
  warn "strict data verification skipped (set LUNARIS_PREFLIGHT_VERIFY_DATA=1 to enable)"
fi

# 6. Scheduler tooling.
command -v sbatch >/dev/null 2>&1 && ok "sbatch on PATH" || warn "sbatch not found (not on a Slurm host?)"

echo "--------------------------------------------------------------"
if [[ ${fail} -eq 0 ]]; then
  echo "PREFLIGHT PASSED — ready to submit (see hpc/submit.sh)."
else
  echo "PREFLIGHT FAILED — fix the [FAIL] items above before submitting." >&2
fi
echo "=============================================================="
exit ${fail}
