#!/usr/bin/env bash
#SBATCH --job-name=lunaris-scenario
#SBATCH --output=lunaris_scenario_%A_%a.out
#SBATCH --error=lunaris_scenario_%A_%a.err
# --- Cluster-specific resource directives (edit for your site) -------------- #
# Request a GPU by default; ST-LRPS training is GPU-heavy. Adjust the partition,
# QOS, account, and resource sizes to match your cluster, or override any of
# them at submit time with `sbatch --gres=... --time=... ...`.
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
# Send SIGTERM 10 minutes before walltime so each training task can finish its
# current epoch, save ckpt_last, and exit resumable instead of dying mid-epoch.
#SBATCH --signal=B:TERM@600
# #SBATCH --partition=gpu
# #SBATCH --qos=normal
# Do NOT hard-code an account here; pass `--account=<your_account>` at submit
# time or set SBATCH_ACCOUNT in your environment.

set -euo pipefail

# Source the cluster-agnostic environment template (paths, threads, modules).
source "$(dirname "$0")/env_template.sh"

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
echo "Job ID            : ${SLURM_JOB_ID:-<none>} (array ${SLURM_ARRAY_JOB_ID:-?}:${SLURM_ARRAY_TASK_ID:-?})"
echo "Host              : $(hostname)"
echo "Working directory : ${LUNARIS_PROJECT_ROOT:-$PWD}"
echo "Python            : $(python --version 2>&1) @ $(command -v python)"
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

# Select and launch the scenario matching $SLURM_ARRAY_TASK_ID.
python tools/hpc/run_training_scenario.py "$@"
