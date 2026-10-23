#!/usr/bin/env bash
#
# One command after `git clone` on a cluster LOGIN node. The data downloads need
# internet access, which compute nodes often do not have.
#
#   bash hpc/bootstrap.sh --profile uhem_altay
#
# Steps, each reported OK / WARN / FAIL in the summary at the end:
#   1. site config   copy hpc/clusters/<profile>.env.example to hpc/cluster.env
#                    (only when cluster.env does not exist; never overwritten)
#                    and refuse CRLF line endings
#   2. discovery     print the scheduler values the profile leaves as ">>> FILL IN"
#                    (spart, sinfo, sacctmgr, module avail CUDA, quota); read-only
#   3. environment   bash hpc/setup_env.sh, skipped when the venv already has
#                    lunaris-train (run hpc/setup_env.sh yourself to reinstall)
#   4. data          lunaris-data download --preset full-gravity into
#                    $LUNARIS_DATA_DIR: the NAIF SPICE kernels and the GRAIL
#                    GL1800F gravity field (jggrx_1800f_sha.tab), SHA-256 checked;
#                    then every file of the preset must be present and valid
#   5. preflight     bash hpc/preflight.sh (its GPU checks only warn on a login node)
#
# Options:
#   --profile NAME   site profile in hpc/clusters/ (uhem_altay, ...); needed only
#                    while hpc/cluster.env does not exist
#   --skip-env       skip step 3
#   --skip-data      skip step 4
#   --dry-run        print the commands of steps 3-5 instead of running them
#
# LUNARIS_DATA_PRESET selects another lunaris-data preset (default full-gravity).
# Exit status 0 when no step failed, 1 otherwise. A rerun is safe: hash-verified
# files are not downloaded again and an existing cluster.env is kept.

set -euo pipefail

_HPC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
_PROJECT_ROOT="$(cd "${_HPC_DIR}/.." && pwd)"
_CLUSTER_ENV="${_HPC_DIR}/cluster.env"

usage() {
  awk 'NR > 1 { if (/^#/) { sub(/^# ?/, ""); print } else exit }' "${BASH_SOURCE[0]}"
  exit "${1:-64}"
}

profile=""
skip_env=0
skip_data=0
dry_run=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile)    [[ $# -ge 2 ]] || { echo "error: --profile needs a name" >&2; exit 64; }
                  profile="$2"; shift 2 ;;
    --profile=*)  profile="${1#*=}"; shift ;;
    --skip-env)   skip_env=1; shift ;;
    --skip-data)  skip_data=1; shift ;;
    --dry-run)    dry_run=1; shift ;;
    -h|--help)    usage 0 ;;
    *) echo "error: unknown option '$1'" >&2; usage 64 ;;
  esac
done
# Accept "uhem_altay", "uhem_altay.env.example" and a path to the profile.
profile="$(basename "${profile}")"
profile="${profile%.env.example}"

data_preset="${LUNARIS_DATA_PRESET:-full-gravity}"

_names=()
_states=()
_notes=()
_record() { _names+=("$1"); _states+=("$2"); _notes+=("$3"); }
_step() { echo; echo "==> [$1/5] $2"; }
_run() {
  echo "+ $*"
  if [[ ${dry_run} -eq 1 ]]; then return 0; fi
  "$@"
}

# --------------------------------------------------------------------------- #
_step 1 "site config (hpc/cluster.env)"
if [[ -f "${_CLUSTER_ENV}" ]]; then
  echo "kept the existing ${_CLUSTER_ENV}"
  [[ -n "${profile}" ]] && echo "(--profile ${profile} is ignored: cluster.env already exists)"
  config_state="OK"; config_note="existing cluster.env kept"
elif [[ -z "${profile}" ]]; then
  echo "error: ${_CLUSTER_ENV} does not exist. Name a site profile:" >&2
  for p in "${_HPC_DIR}"/clusters/*.env.example; do
    [[ -e "${p}" ]] && echo "  bash hpc/bootstrap.sh --profile $(basename "${p}" .env.example)" >&2
  done
  echo "or copy hpc/cluster.env.example to hpc/cluster.env and fill it in." >&2
  exit 64
else
  src="${_HPC_DIR}/clusters/${profile}.env.example"
  if [[ ! -f "${src}" ]]; then
    echo "error: no site profile ${src}" >&2
    exit 64
  fi
  if [[ ${dry_run} -eq 1 ]]; then
    echo "+ cp ${src} ${_CLUSTER_ENV}"
    config_state="OK"; config_note="dry run: would copy ${profile}.env.example"
  else
    cp "${src}" "${_CLUSTER_ENV}"
    echo "copied ${src} -> ${_CLUSTER_ENV}"
    config_state="OK"; config_note="created from ${profile}.env.example"
  fi
fi
_cfg_file="${_CLUSTER_ENV}"
[[ -f "${_cfg_file}" ]] || _cfg_file="${_HPC_DIR}/clusters/${profile}.env.example"
# A CRLF copy leaves a carriage return in every value (partition "a100q\r").
# -U: grep builds for Windows (Git Bash) drop the CR without it; no-op on Linux.
if grep -qU $'\r' "${_cfg_file}"; then
  config_state="FAIL"; config_note="CRLF line endings; fix with: sed -i 's/\r\$//' hpc/cluster.env"
fi
_record "site config" "${config_state}" "${config_note}"
if [[ "${config_state}" == "FAIL" ]]; then
  # Sourcing a CRLF file breaks on its first compound line; stop before it.
  echo "error: ${config_note}" >&2
  echo "Not ready. Next: fix that, then rerun  bash hpc/bootstrap.sh"
  exit 1
fi

# Site values for the remaining steps. `set -a` exports them like the jobs do.
set +u
set -a
# shellcheck disable=SC1090
source "${_cfg_file}"
set +a
set -u
export LUNARIS_DATA_DIR="${LUNARIS_DATA_DIR:-$HOME/lunaris_data}"
export LUNARIS_OUTPUT_DIR="${LUNARIS_OUTPUT_DIR:-$HOME/lunaris_outputs}"
export LUNARIS_VENV="${LUNARIS_VENV:-$HOME/venvs/lunaris}"

# --------------------------------------------------------------------------- #
_step 2 "discovery (values for the >>> FILL IN lines of cluster.env)"
_empty=()
[[ -z "${LUNARIS_ACCOUNT:-}" ]] && _empty+=("LUNARIS_ACCOUNT")
[[ -z "${LUNARIS_CPU_PARTITION:-}" ]] && _empty+=("LUNARIS_CPU_PARTITION")
[[ -z "${LUNARIS_QOS:-}" ]] && _empty+=("LUNARIS_QOS")
case "${LUNARIS_MODULES:-}" in *[Cc][Uu][Dd][Aa]*) ;; *) _empty+=("CUDA module in LUNARIS_MODULES") ;; esac
if command -v spart >/dev/null 2>&1; then
  # UHeM: the queues this account may use, their load, and the CPU-hours left.
  echo "--- spart (your queues, free nodes, project CPU-hours)"
  spart 2>&1 || true
fi
if command -v sinfo >/dev/null 2>&1; then
  echo "--- sinfo -s (partitions; a CPU one goes to LUNARIS_CPU_PARTITION)"
  sinfo -s 2>&1 || true
  if [[ -n "${LUNARIS_PARTITION:-}" ]]; then
    echo "--- ${LUNARIS_PARTITION}: time limit, GPUs, nodes"
    sinfo -p "${LUNARIS_PARTITION}" -o "%P %l %G %D" 2>&1 || true
  fi
else
  echo "sinfo not found: not a Slurm login node?"
fi
if command -v sacctmgr >/dev/null 2>&1; then
  echo "--- sacctmgr (account and QOS of ${USER:-you})"
  sacctmgr -n -P show assoc user="${USER:-}" format=account,partition,qos 2>&1 || true
fi
if command -v module >/dev/null 2>&1; then
  # Site shell code: relax set -u for it, as hpc/env_template.sh does.
  set +u
  # The profile's Python module name was not confirmed from UHeM's docs (their
  # module list page is gone): the lock needs a CPython 3.11 one.
  echo "--- module avail Python (a 3.11 one goes first in LUNARIS_MODULES)"
  module -t avail Python 2>&1 | grep -i "3\.11" | head -n 20 || true
  echo "--- module avail CUDA (name for LUNARIS_MODULES)"
  module -t avail CUDA 2>&1 | head -n 20 || true
  set -u
else
  echo "module command not found in this shell: run  module avail CUDA  by hand"
fi
if command -v quota >/dev/null 2>&1; then
  echo "--- quota -s (\$HOME holds data and outputs unless cluster.env moves them)"
  quota -s 2>&1 || true
fi
if [[ ${#_empty[@]} -gt 0 ]]; then
  _record "discovery" "WARN" "still empty: ${_empty[*]} (fill in from the lists above)"
else
  _record "discovery" "OK" "no >>> FILL IN value left empty"
fi

# --------------------------------------------------------------------------- #
_step 3 "environment (${LUNARIS_VENV})"
if [[ ${skip_env} -eq 1 ]]; then
  _record "environment" "WARN" "skipped (--skip-env)"
elif [[ -x "${LUNARIS_VENV}/bin/lunaris-train" ]]; then
  echo "venv already has lunaris-train: not reinstalled"
  _record "environment" "OK" "existing venv kept"
elif _run bash "${_HPC_DIR}/setup_env.sh"; then
  _record "environment" "OK" "$([[ ${dry_run} -eq 1 ]] && echo "dry run" || echo "venv built from the HPC lock")"
else
  _record "environment" "FAIL" "hpc/setup_env.sh failed (see its output above)"
fi

# Steps 4 and 5 run inside the job environment (modules + venv).
env_ready=0
if [[ ${dry_run} -eq 0 && -x "${LUNARIS_VENV}/bin/python" ]]; then
  export LUNARIS_ENV_STRICT=0
  set +u
  # shellcheck disable=SC1091
  source "${_HPC_DIR}/env_template.sh"
  set -u
  [[ ${#LUNARIS_ENV_FAILS[@]} -eq 0 ]] && env_ready=1
  for _f in ${LUNARIS_ENV_FAILS[@]+"${LUNARIS_ENV_FAILS[@]}"}; do echo "error: ${_f}" >&2; done
fi

# --------------------------------------------------------------------------- #
_step 4 "data (lunaris-data preset ${data_preset} -> ${LUNARIS_DATA_DIR})"
if [[ ${skip_data} -eq 1 ]]; then
  _record "data" "WARN" "skipped (--skip-data)"
elif [[ ${dry_run} -eq 1 ]]; then
  _run lunaris-data download --preset "${data_preset}"
  _record "data" "OK" "dry run"
elif [[ ${env_ready} -eq 0 ]]; then
  _record "data" "FAIL" "no working environment (step 3)"
else
  mkdir -p "${LUNARIS_DATA_DIR}"
  _run lunaris-data download --preset "${data_preset}" || true
  # `lunaris-data verify` lets optional entries be missing; here every file of
  # the preset (the GL1800F gravity field included) must be present and valid.
  if python - "${data_preset}" <<'PY'
import sys

from lunaris.cli.data import (
    find_manifest,
    load_manifest,
    resolve_data_root,
    select_preset_datasets,
    verify_entry_detail,
)

ok_states = ("valid", "present")
root = resolve_data_root()
bad = 0
for entry in select_preset_datasets(load_manifest(find_manifest()), sys.argv[1]):
    detail = verify_entry_detail(entry, root)
    states = [detail["status"]] + [
        c["status"] for c in detail["companions"] if c["required"]
    ]
    good = all(s in ok_states for s in states)
    bad += not good
    print(f"  {'OK  ' if good else 'FAIL'} {entry['name']:28s} {'/'.join(states):18s} {detail['target']}")
sys.exit(1 if bad else 0)
PY
  then
    _record "data" "OK" "every ${data_preset} file present, SHA-256 verified"
  else
    _record "data" "FAIL" "a ${data_preset} file is missing or fails its hash (rerun to retry)"
  fi
fi

# --------------------------------------------------------------------------- #
_step 5 "preflight (hpc/preflight.sh)"
if [[ ${dry_run} -eq 1 ]]; then
  _run bash "${_HPC_DIR}/preflight.sh"
  _record "preflight" "OK" "dry run"
elif [[ ${env_ready} -eq 0 ]]; then
  _record "preflight" "FAIL" "no working environment (step 3)"
elif bash "${_HPC_DIR}/preflight.sh"; then
  _record "preflight" "OK" "no FAIL (GPU checks need a GPU job)"
else
  _record "preflight" "FAIL" "see the FAIL lines of hpc/preflight.sh above"
fi

# --------------------------------------------------------------------------- #
echo
echo "==> Summary"
failed=0
for i in "${!_names[@]}"; do
  printf '  %-4s  %-12s  %s\n' "${_states[$i]}" "${_names[$i]}" "${_notes[$i]}"
  [[ "${_states[$i]}" == "FAIL" ]] && failed=1
done
echo
if [[ ${failed} -eq 1 ]]; then
  echo "Not ready. Next: fix the FAIL lines above, then rerun  bash hpc/bootstrap.sh"
  exit 1
fi
_n_h5="$(find "${LUNARIS_DATA_DIR}/datasets" -maxdepth 3 -name '*.h5' -size +0 2>/dev/null | wc -l || true)"
if [[ ${dry_run} -eq 0 && "${_n_h5}" -eq 0 ]]; then
  echo "Ready, but no ST-LRPS cloud suite yet. Next: copy it from your workstation with"
  echo "  bash hpc/stage_artifacts.sh <user>@<login-host> --suite <suite dir> [--model <run dir>]"
  echo "or generate one on the cluster: hpc/submit.sh generate -- ... (hpc/README.md)."
else
  _srun="srun -p ${LUNARIS_PARTITION:-<gpu partition>}${LUNARIS_ACCOUNT:+ -A ${LUNARIS_ACCOUNT}}${LUNARIS_QOS:+ --qos=${LUNARIS_QOS}} --gres=${LUNARIS_GRES:-gpu:1}${LUNARIS_CPUS_PER_TASK:+ -c ${LUNARIS_CPUS_PER_TASK}} -t 10"
  echo "Ready. Next: check a GPU node with  ${_srun} bash hpc/preflight.sh"
  echo "then submit a job (hpc/README.md)."
fi
