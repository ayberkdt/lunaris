# ST-LRPS Runtime Profiling

ST-LRPS runtime profiling measures inference bottlenecks before optimization. It does not change physics, model architecture, checkpoint contents, loss functions, validation metrics, or propagation algorithms.

The profiler measures:

- model loading and checkpoint/config/scaler load phases
- single-point latency and batched throughput
- full acceleration inference, including the autograd-gradient path
- potential-only forward timing as a low-risk proxy for forward cost
- CPU or CUDA runtime behavior
- chunk-size sensitivity
- CUDA memory allocation and reservation when available
- optional classic spherical-harmonic timing when the local gravity file is available

## Recommended Command

```bash
python -m st_lrps.runtime.profiling \
    --model-dir runs/st_lrps_train_xxx \
    --batch-sizes 1,16,128,1024,8192 \
    --n-warmup 10 \
    --n-repeat 50 \
    --out-dir results/profiling/st_lrps_runtime
```

## Synthetic Query Mode

Synthetic mode is the default and does not require dataset files. It samples random Moon-centered positions in SI meters with uniformly distributed directions and uniformly sampled altitude:

```bash
python -m st_lrps.runtime.profiling \
    --model-dir runs/st_lrps_train_xxx \
    --input-source synthetic \
    --alt-min-km 100 \
    --alt-max-km 2000 \
    --out-dir results/profiling/st_lrps_runtime
```

## Dataset Query Mode

Dataset mode samples the first three columns as `x,y,z` positions from an HDF5 dataset without loading the full file:

```bash
python -m st_lrps.runtime.profiling \
    --model-dir runs/st_lrps_train_xxx \
    --input-source dataset \
    --data data/spatial_cloud_train.h5 \
    --dataset-name data \
    --batch-sizes 1024,8192,32768 \
    --out-dir results/profiling/st_lrps_dataset_runtime
```

## CPU And CUDA

Use `--device cpu`, `--device cuda`, or `--device auto`. CUDA timings synchronize before and after measured calls so asynchronous kernels are not underreported. Warmup calls are excluded from steady-state statistics.

## Batch And Chunk Effects

Batch size 1 measures latency. Large batch sizes measure throughput. Use `--chunk-sizes` to understand whether runtime chunking is limiting throughput or reducing memory pressure:

```bash
python -m st_lrps.runtime.profiling \
    --model-dir runs/st_lrps_train_xxx \
    --batch-sizes 1024,8192,32768 \
    --chunk-sizes none,512,1024,4096 \
    --out-dir results/profiling/st_lrps_chunks
```

Monte Carlo workflows should prefer batched force evaluation when throughput improves at larger batch sizes. If p95 timing is much higher than median timing, runtime jitter or memory pressure may be present.

## Optional Classic SH Comparison

Classic spherical-harmonic timing is optional:

```bash
python -m st_lrps.runtime.profiling \
    --model-dir runs/st_lrps_train_xxx \
    --compare-classic-sh \
    --classic-sh-degree 60 \
    --out-dir results/profiling/st_lrps_vs_sh
```

If the local gravity coefficient file is unavailable, ST-LRPS profiling still runs and the classic SH comparison is skipped with a warning.

## Generated Outputs

When `--out-dir` is provided, the profiler writes:

- `runtime_profile.json`
- `runtime_profile.csv`
- `runtime_profile_summary.md`
- `runtime_profile_latency.png` if matplotlib is available
- `runtime_profile_throughput.png` if matplotlib is available

These are generated outputs. Keep them under ignored locations such as `results/`, `outputs/`, `profiling_results/`, or external scratch storage, and do not commit them.
