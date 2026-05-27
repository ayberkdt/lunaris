#!/bin/bash
#SBATCH --job-name=mc_sims
#SBATCH --output=outputs/slurm_logs/mc_runner_%j.out
#SBATCH --error=outputs/slurm_logs/mc_runner_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16       # Monte Carlo usually benefits from multiple CPU threads
#SBATCH --mem=64G
#SBATCH --partition=compute      # TODO: Adapt to your cluster's compute/CPU or GPU partition
#SBATCH --account=your_account   # TODO: Replace with your actual project account

# This is a template script for running Monte Carlo / batch propagation.
# Adapt it to your specific cluster.

# Load modules (Adapt to your cluster)
# module purge
# module load miniconda3

# Activate environment
# source activate st_lrps

mkdir -p outputs/slurm_logs

# Run Monte Carlo runner
echo "Starting Monte Carlo simulation at $(date)"
python mc_runner.py --out-dir outputs/mc_results/mc_run
echo "Monte Carlo simulation completed at $(date)"
