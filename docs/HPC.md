# HPC and Cluster Deployment

This guide outlines how to deploy and run ST-LRPS on High-Performance Computing (HPC) clusters, specifically those using the Slurm workload manager.

## Environment Setup

Cluster nodes generally run "headless" (without graphical displays) and may encounter issues if you attempt to install GUI frameworks like PyQt6 or PySide6. Therefore, ST-LRPS provides dedicated configuration files for HPC environments.

### Option A: Using Conda (Recommended)

1. Load your cluster's Conda/Miniconda module:
   ```bash
   module load miniconda3  # Adapt to your cluster's module name
   ```
2. Edit `environment.yml` to uncomment and select the `pytorch-cuda` version that matches the CUDA module available on your cluster (e.g., `pytorch-cuda=12.1`).
3. Create and activate the environment:
   ```bash
   conda env create -f environment.yml
   conda activate lunaris
   ```

### Option B: Using Pip / Virtual Environment

If you prefer pip, use `requirements_hpc.txt` which specifically excludes GUI dependencies:

```bash
python -m venv lunaris_env
source lunaris_env/bin/activate
pip install -r requirements_hpc.txt
```

## Running Headless CLI Commands

Instead of using the Studio UI, HPC users must use the headless CLI entry points. Below are examples of launching specific tasks.

### 1. Training

```bash
python -m lunaris.surrogate.st_lrps.training.cli \
    --out-dir outputs/training/st_lrps_train_run \
    --epochs 100 \
    --batch-size 8192
```

### 2. Evaluation

Evaluations check the quality of a trained model:

```bash
python -m lunaris.surrogate.st_lrps.evaluation.cli \
    --model-dir outputs/training/st_lrps_train_run \
    --out-dir outputs/evaluations/st_lrps_eval_run
```

### 3. Orbit Validation (Gravity Models)

For validating orbital propagation across different gravity models and surrogates:

```bash
python -m lunaris.surrogate.st_lrps.evaluation.compare_gravity_models \
    --out-dir outputs/validation/orbit_validation_run
```

### 4. Monte Carlo / Batch Propagation

Run the Monte Carlo simulator headlessly for batch processing (CPU parallelized or GPU vectorized depending on your configuration):

```bash
lunaris-mc --out-dir outputs/monte_carlo/mc_run
```

## Using Slurm Job Scripts

You can submit background jobs using the provided template scripts located in the `slurm_examples/` directory.

**Important:** These are *templates*. You **must** open them and adapt the placeholders (marked with `TODO`) to match your specific cluster's partition names, account names, and available module names.

- **Training:** `sbatch slurm_examples/train_st_lrps.slurm`
- **Evaluation:** `sbatch slurm_examples/eval_st_lrps.slurm`
- **Orbit Validation:** `sbatch slurm_examples/orbit_validation.slurm`
- **Monte Carlo:** `sbatch slurm_examples/mc_runner.slurm`

### Output Policy

All Slurm scripts and headless CLI examples are designed to write outputs into the repository-level `outputs/` folder (e.g., `outputs/slurm_logs/`).
Please do not modify scripts to write output files inside source code directories such as `src/lunaris/surrogate/st_lrps/` or `src/lunaris/core/`.
