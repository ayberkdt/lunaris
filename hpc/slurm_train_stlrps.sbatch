#!/usr/bin/env bash
#SBATCH --job-name=lunaris-train
#SBATCH --output=outputs/slurm/train_%j.out
#SBATCH --error=outputs/slurm/train_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00

set -euo pipefail
source "$(dirname "$0")/env_template.sh"

lunaris-train "$@"
