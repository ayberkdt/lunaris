#!/bin/bash
#SBATCH --job-name=st_lrps_eval
#SBATCH --output=outputs/slurm_logs/eval_%j.out
#SBATCH --error=outputs/slurm_logs/eval_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=gpu          # TODO: Adapt to your cluster's partition
#SBATCH --gres=gpu:1             # TODO: Adapt if specific GPUs are needed
#SBATCH --account=your_account   # TODO: Replace with your actual project account

# This is a template script for evaluating a trained ST-LRPS model.
# Adapt it to your specific cluster.

# Load modules (Adapt to your cluster)
# module purge
# module load cuda/12.1.1        # Example CUDA load
# module load miniconda3

# Activate environment
# source activate st_lrps

mkdir -p outputs/slurm_logs

# Run evaluation
# Assumes the trained model is located in outputs/training/st_lrps_train_run
echo "Starting ST-LRPS evaluation at $(date)"
python -m lunaris.surrogate.st_lrps.evaluation.cli --model-dir outputs/training/st_lrps_train_run --out-dir outputs/evaluations/st_lrps_eval_run
echo "Evaluation completed at $(date)"
