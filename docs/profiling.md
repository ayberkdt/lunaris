# ST-LRPS Runtime Profiling

ST-LRPS runtime profiling measures inference bottlenecks before optimization. It does not change physics, model architecture, checkpoint contents, loss functions, validation metrics, or propagation algorithms.

> **Read this before comparing ST-LRPS to the SH kernel.** ST-LRPS is a
> high-throughput **batch** gravity backend, not a low-latency single-trajectory
> CPU replacement. A single trajectory through `propagate()` runs the surrogate as
> an interpreted PyTorch + autograd closure (not `@njit`), paying Python/autograd
> overhead on every RHS call, so it is **expected to be slower** than the
> spherical-harmonic kernel on one CPU trajectory. The surrogate only amortizes
> that overhead across a large GPU batch (Monte Carlo / ensemble). A fair
> "surrogate vs SH" timing compares the **GPU batch** backend at matched batch
> sizes — never one CPU trajectory, which measures the wrong path and will
> mislead any speedup table.

The profiler measures:

- model loading and checkpoint/config/scaler load phases
- single-point latency and batched throughput
- full acceleration inference, including the autograd-gradient path for
  `potential_autograd` artifacts
- potential-only forward timing as a low-risk proxy for forward cost when the
  artifact predicts scalar potential
- direct residual-acceleration timing for `force_direct` artifacts; potential
  timing is reported as unavailable for those artifacts
- CPU or CUDA runtime behavior
- chunk-size sensitivity
- CUDA memory allocation and reservation when available
- optional classic spherical-harmonic timing when the local gravity file is available

## Recommended Command

```bash
python -m lunaris.surrogate.st_lrps.runtime.profiling \
    --model-dir outputs/training/st_lrps_train_xxx \
    --batch-sizes 1,16,128,1024,8192 \
    --n-warmup 10 \
    --n-repeat 50 \
    --out-dir outputs/runtime/st_lrps_runtime_xxx
```

## Synthetic Query Mode

Synthetic mode is the default and does not require dataset files. It samples random Moon-centered positions in SI meters with uniformly distributed directions and uniformly sampled altitude:

```bash
python -m lunaris.surrogate.st_lrps.runtime.profiling \
    --model-dir outputs/training/st_lrps_train_xxx \
    --input-source synthetic \
    --alt-min-km 100 \
    --alt-max-km 2000 \
    --out-dir outputs/runtime/st_lrps_runtime_xxx
```

## Dataset Query Mode

Dataset mode samples the first three columns as `x,y,z` positions from an HDF5 dataset without loading the full file:

```bash
python -m lunaris.surrogate.st_lrps.runtime.profiling \
    --model-dir outputs/training/st_lrps_train_xxx \
    --input-source dataset \
    --data data/spatial_cloud_train.h5 \
    --dataset-name data \
    --batch-sizes 1024,8192,32768 \
    --out-dir outputs/runtime/st_lrps_dataset_runtime_xxx
```

## CPU And CUDA

Use `--device cpu`, `--device cuda`, or `--device auto`. CUDA timings synchronize before and after measured calls so asynchronous kernels are not underreported. Warmup calls are excluded from steady-state statistics.

## Interpreting CUDA Memory Logs

Training logs report PyTorch allocator memory, not full `nvidia-smi` process memory. `cuda_mem=a/bMiB` means current allocated/current reserved memory. `peak=c/dMiB` means peak allocated/peak reserved memory since the current train or validation phase started. `total=eMiB` is physical GPU VRAM. Use `nvidia-smi -l 1` to inspect utilization and full process memory; low current allocation after a batch does not necessarily mean the GPU was idle.

## Interpreting Studio Live Loss Plots

The Studio live loss plots can use log-y scale for positive loss curves. Smoothing is display-only and never changes history files or training metrics. Auxiliary physics terms are shown separately from the main loss overview when present, and missing metrics are expected early in training or when a feature is disabled.

## Batch And Chunk Effects

Batch size 1 measures latency. Large batch sizes measure throughput. Use `--chunk-sizes` to understand whether runtime chunking is limiting throughput or reducing memory pressure:

```bash
python -m lunaris.surrogate.st_lrps.runtime.profiling \
    --model-dir outputs/training/st_lrps_train_xxx \
    --batch-sizes 1024,8192,32768 \
    --chunk-sizes none,512,1024,4096 \
    --out-dir outputs/runtime/st_lrps_chunks_xxx
```

Batch propagation workflows should prefer batched force evaluation when throughput improves at larger batch sizes. If p95 timing is much higher than median timing, runtime jitter or memory pressure may be present.

## Optional Classic SH Comparison

Classic spherical-harmonic timing is optional:

```bash
python -m lunaris.surrogate.st_lrps.runtime.profiling \
    --model-dir outputs/training/st_lrps_train_xxx \
    --compare-classic-sh \
    --classic-sh-degree 60 \
    --out-dir outputs/runtime/st_lrps_vs_sh_xxx
```

If the local gravity coefficient file is unavailable, ST-LRPS profiling still runs and the classic SH comparison is skipped with a warning.

## Batch Propagation Backend Profiling

Batch propagation profiling must record the sampling method, requested backend,
and resolved backend. Use `lunaris-batch --mc-backend ...` to compare:

```bash
lunaris-batch \
    --sampling-method sobol_scrambled \
    --mc-backend auto \
    --gpu-sh-degree 24 \
    --n-samples 128 \
    --mc-dt-s 60 \
    --mc-output-path outputs/monte_carlo/profile_auto.h5
```

Important interpretation rules:

- The current true classic-SH GPU tier is degree 24. Requests above degree 24
  fall back to CPU SH and must be reported as fallback, not GPU high-degree SH.
- Batch outputs include `sampling_method`, `requested_mc_backend`, `actual_mc_backend`,
  `requested_sh_degree`, `actual_sh_degree`, `runtime_model_kind`,
  `fallback_reason`, CUDA device name when available, dtype, integrator, and step
  size metadata.
- `gpu_st_lrps_potential` keeps the scalar-potential/autograd path and is the
  physically cleaner ST-LRPS runtime.
- `gpu_st_lrps_direct` avoids autograd and should benchmark faster, but it has no
  conservative-field guarantee and remains experimental until curl and orbit
  validation pass for the target regime.

## Generated Outputs

When `--out-dir` is provided, the profiler writes:

- `runtime_profile.json`
- `runtime_profile.csv`
- `runtime_profile_summary.md`
- `runtime_profile_latency.png` if matplotlib is available
- `runtime_profile_throughput.png` if matplotlib is available

These are generated outputs. The canonical location is `outputs/runtime/<profile_name>/`. External scratch storage is also fine; do not commit generated profiling products.

## Direct-Force Runtime Notes

`force_direct` artifacts are trained with `lunaris-train-force-direct` and
evaluated with `lunaris-eval-force-direct`. They target faster acceleration
inference by predicting residual acceleration directly, but they are not scalar
potential models. Treat speedup numbers as runtime diagnostics only until curl
and orbit-level validation have been run.
