#!/bin/bash
#SBATCH --job-name=orbit_val
#SBATCH --output=outputs/slurm_logs/orbit_val_%j.out
#SBATCH --error=outputs/slurm_logs/orbit_val_%j.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=gpu          # TODO: Adapt to your cluster's partition
#SBATCH --gres=gpu:1             # TODO: Adapt if specific GPUs are needed
#SBATCH --account=your_account   # TODO: Replace with your actual project account

# This is a template script for orbit validation and comparing gravity models.
# Adapt it to your specific cluster.

# Load modules (Adapt to your cluster)
# module purge
# module load cuda/12.1.1
# module load miniconda3

# Activate environment
# source activate st_lrps

mkdir -p outputs/slurm_logs

# Run orbit validation / compare gravity models
echo "Starting Orbit Validation at $(date)"
python -m st_lrps.evaluation.compare_gravity_models --out-dir outputs/validation/orbit_validation_run
echo "Orbit Validation completed at $(date)"
