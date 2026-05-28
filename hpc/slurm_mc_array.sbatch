#!/usr/bin/env bash
#SBATCH --job-name=lunaris-mc
#SBATCH --output=outputs/slurm/mc_%A_%a.out
#SBATCH --error=outputs/slurm/mc_%A_%a.err
#SBATCH --array=0-9
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00

set -euo pipefail
source "$(dirname "$0")/env_template.sh"

lunaris-mc "$@"
