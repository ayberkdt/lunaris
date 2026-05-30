#!/usr/bin/env bash
set -euo pipefail

# Lightweight, cluster-agnostic environment template sourced by the hpc/*.sbatch
# jobs. Adapt the paths and commands below to your cluster; do not hard-code a
# specific site/module name here.

# Project checkout and external storage.
#   LUNARIS_DATA_DIR    is read by the framework to locate external data.
#   LUNARIS_OUTPUT_DIR  is a convenience the example jobs pass to --out-dir /
#                       --output-dir (the framework does not read it directly).
export LUNARIS_PROJECT_ROOT="${LUNARIS_PROJECT_ROOT:-$PWD}"
export LUNARIS_DATA_DIR="${LUNARIS_DATA_DIR:-/scratch/$USER/lunaris_data}"
export LUNARIS_OUTPUT_DIR="${LUNARIS_OUTPUT_DIR:-/scratch/$USER/lunaris_outputs}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

# First-time setup (run once): load your cluster's modules, activate an
# environment, then install the headless extra. Adapt the module names below.
# module load cuda/12.1
# source "$HOME/venvs/lunaris/bin/activate"
# cd "$LUNARIS_PROJECT_ROOT"
# python -m pip install -e ".[hpc]"
