#!/bin/bash
#SBATCH --job-name=st_lrps_train
#SBATCH --output=outputs/slurm_logs/train_%j.out
#SBATCH --error=outputs/slurm_logs/train_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=gpu          # TODO: Adapt to your cluster's GPU partition
#SBATCH --gres=gpu:1             # TODO: Adapt if specific GPUs are needed (e.g., gpu:a100:1)
#SBATCH --account=your_account   # TODO: Replace with your actual project account

# This is a template script for running ST-LRPS training on a Slurm cluster.
# DO NOT claim this script is universally valid for all HPC systems; adapt it to your specific cluster.

# Load required modules
# TODO: Uncomment and adapt the following lines based on your cluster's module system.
# module purge
# module load gcc/11.2.0
# module load cuda/12.1.1        # Example CUDA 12.1 load. Adapt as needed.
# module load miniconda3

# Activate Conda environment
# source activate st_lrps

# Ensure output directory exists
mkdir -p outputs/slurm_logs

# Run training
# Note: Outputs are directed to the outputs/ directory, NOT inside st_lrps/.
echo "Starting ST-LRPS training job at $(date)"
python -m lunaris.surrogate.st_lrps.training.cli --out-dir outputs/training/st_lrps_train_run
echo "Training job completed at $(date)"
