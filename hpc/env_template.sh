#!/usr/bin/env bash
set -euo pipefail

# Adjust these paths for your cluster.
export LUNARIS_PROJECT_ROOT="${LUNARIS_PROJECT_ROOT:-$PWD}"
export LUNARIS_DATA_DIR="${LUNARIS_DATA_DIR:-$LUNARIS_PROJECT_ROOT/data}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

# Example:
# module load cuda/12.1
# source "$HOME/venvs/lunaris/bin/activate"
# pip install -e "$LUNARIS_PROJECT_ROOT.[hpc]"
