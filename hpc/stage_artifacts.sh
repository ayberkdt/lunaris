#!/usr/bin/env bash
#
# Copy the artifacts that cannot be downloaded (ST-LRPS cloud suites, finished
# training runs for --init-from / evaluation) from this workstation to a cluster,
# and prove the copy with SHA-256. Run it on the workstation (Linux, macOS, or
# Git Bash on Windows); it needs ssh, tar and sha256sum.
#
#   bash hpc/stage_artifacts.sh <user>@<login-host> \
#       --suite outputs/datasets/cloud_suites/<suite id> \
#       --model outputs/training/<run dir>
#
# Each --suite DIR goes to <remote data>/datasets/cloud_suites/<name of DIR>,
# each --model DIR to <remote outputs>/training/<name of DIR>. Both options repeat.
#
# Options:
#   --suite DIR            a cloud-suite directory (HDF5 files)
#   --model DIR            a finished training run directory
#   --remote-data DIR      remote LUNARIS_DATA_DIR    (default lunaris_data, under the remote $HOME)
#   --remote-outputs DIR   remote LUNARIS_OUTPUT_DIR  (default lunaris_outputs, under the remote $HOME)
#   --dry-run              list what would be copied; connect to nothing
#
# The defaults match hpc/clusters/uhem_altay.env.example ($HOME/lunaris_data and
# $HOME/lunaris_outputs). An artifact that already exists on the cluster is not
# touched: delete or rename it there first. One ssh connection per artifact: the
# hash list travels with the files and the cluster checks it (sha256sum -c) right
# after unpacking. Exit status 0 when every artifact arrived intact.

set -euo pipefail

usage() {
  awk 'NR > 1 { if (/^#/) { sub(/^# ?/, ""); print } else exit }' "${BASH_SOURCE[0]}"
  exit "${1:-64}"
}

host=""
remote_data="lunaris_data"
remote_outputs="lunaris_outputs"
dry_run=0
kinds=()
paths=()
_need_value() { [[ $2 -ge 2 ]] || { echo "error: $1 needs a value" >&2; exit 64; }; }
while [[ $# -gt 0 ]]; do
  case "$1" in
    --suite)            _need_value "$1" $#; kinds+=(suite); paths+=("$2"); shift 2 ;;
    --suite=*)          kinds+=(suite); paths+=("${1#*=}"); shift ;;
    --model)            _need_value "$1" $#; kinds+=(model); paths+=("$2"); shift 2 ;;
    --model=*)          kinds+=(model); paths+=("${1#*=}"); shift ;;
    --remote-data)      _need_value "$1" $#; remote_data="$2"; shift 2 ;;
    --remote-data=*)    remote_data="${1#*=}"; shift ;;
    --remote-outputs)   _need_value "$1" $#; remote_outputs="$2"; shift 2 ;;
    --remote-outputs=*) remote_outputs="${1#*=}"; shift ;;
    --dry-run)          dry_run=1; shift ;;
    -h|--help)          usage 0 ;;
    -*)                 echo "error: unknown option '$1'" >&2; usage 64 ;;
    *)
      [[ -z "${host}" ]] || { echo "error: more than one host ('${host}', '$1')" >&2; exit 64; }
      host="$1"; shift ;;
  esac
done
[[ -n "${host}" ]] || { echo "error: name the cluster login: <user>@<login-host>" >&2; usage 64; }
[[ ${#paths[@]} -gt 0 ]] || { echo "error: nothing to copy: give at least one --suite or --model" >&2; exit 64; }
for tool in tar sha256sum; do
  command -v "${tool}" >/dev/null 2>&1 || { echo "error: ${tool} not found on PATH" >&2; exit 69; }
done
[[ ${dry_run} -eq 1 ]] || command -v ssh >/dev/null 2>&1 || { echo "error: ssh not found on PATH" >&2; exit 69; }

work="$(mktemp -d)"
trap 'rm -rf "${work}"' EXIT

results=()
failed=0
for i in "${!paths[@]}"; do
  src="${paths[$i]%/}"
  if [[ ! -d "${src}" ]]; then
    results+=("FAIL  ${src}  (not a directory)"); failed=1; continue
  fi
  name="$(basename "${src}")"
  parent="$(cd "$(dirname "${src}")" && pwd)"
  if [[ "${kinds[$i]}" == "suite" ]]; then
    dest="${remote_data%/}/datasets/cloud_suites"
  else
    dest="${remote_outputs%/}/training"
  fi
  n_files="$(find "${src}" -type f | wc -l | tr -d ' ')"
  size="$(du -sh "${src}" | cut -f1)"
  echo
  echo "==> ${kinds[$i]} ${name}: ${n_files} files, ${size} -> ${host}:${dest}/${name}"
  if [[ ${dry_run} -eq 1 ]]; then
    results+=("DRY   ${dest}/${name}  (${n_files} files, ${size})"); continue
  fi

  # Hash list with paths relative to the destination folder, beside the artifact.
  sums="${name}.sha256"
  (cd "${parent}" && find "${name}" -type f -print0 | sort -z | xargs -0 sha256sum) > "${work}/${sums}"
  printf -v q_dest '%q' "${dest}"
  printf -v q_name '%q' "${name}"
  printf -v q_sums '%q' "${sums}"
  remote="set -e; d=${q_dest}; [ ! -e \"\$d\"/${q_name} ] || { echo \"exists: \$d/\"${q_name} >&2; exit 17; }"
  remote+="; mkdir -p \"\$d\" && tar -xf - -C \"\$d\" && cd \"\$d\" && sha256sum -c --quiet ${q_sums}"
  rc=0
  tar -cf - -C "${work}" "${sums}" -C "${parent}" "${name}" | ssh "${host}" "${remote}" || rc=$?
  case "${rc}" in
    0)  results+=("OK    ${dest}/${name}  (${n_files} files, SHA-256 checked on the cluster)") ;;
    17) results+=("SKIP  ${dest}/${name}  (already on the cluster; not overwritten)"); failed=1 ;;
    *)  results+=("FAIL  ${dest}/${name}  (ssh/tar/sha256sum exit ${rc})"); failed=1 ;;
  esac
done

echo
echo "==> Summary"
for line in "${results[@]}"; do echo "  ${line}"; done
echo
if [[ ${dry_run} -eq 1 ]]; then
  echo "Dry run: nothing copied. Next: rerun without --dry-run."
elif [[ ${failed} -eq 1 ]]; then
  echo "Not every artifact arrived. Next: fix the SKIP/FAIL lines above and rerun with only those."
  exit 1
else
  echo "All artifacts arrived intact. Next, on the cluster: lunaris-data validate --data <file.h5> for each suite file."
fi
