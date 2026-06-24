# Lunar Gravity Validation

This module validates lunar gravity models by comparing candidate force models
against a declared high-degree spherical-harmonic numerical reference. It can
evaluate classical lower-degree spherical-harmonic models and, when an artifact
directory is provided, the ST-LRPS residual-potential surrogate.

## Harness Entry Point

Run the gravity benchmark CLI as:

```bash
python -m lunaris.surrogate.st_lrps.evaluation.compare_gravity_models --help
```

The same harness is exposed by `lunaris-benchmark` and by the ST-LRPS Studio
under **Analysis -> Orbit-Level Benchmark**.

## Reference Hierarchy

- **Numerical reference**: high-degree spherical harmonics, commonly SH200 for
  paper-safe orbit benchmarks.
- **Baseline models**: lower-degree spherical-harmonic models such as SH20,
  SH60, SH80, SH120, and SH160.
- **Optional learned model**: ST-LRPS residual-potential surrogate, enabled by
  providing a trained artifact directory.

## Validation Modes

Supported validation modes include:

- CPU smoke validation
- random scenario propagation
- ST-LRPS force-sample trajectory mode
- GPU batch comparison
- full SH-vs-ST-LRPS comparison

## Metrics

Gravity validation runs report:

- runtime_s
- runtime_rel_to_truth
- rms_pos_err_km
- final_pos_err_km
- max_pos_err_km
- p95_pos_err_km
- rms_vel_err_ms
- final_vel_err_ms
- radial_rms_km
- along_rms_km
- cross_rms_km
- radial_max_km
- along_max_km
- cross_max_km
- final_alt_err_km
- rms_alt_err_km
- max_abs_alt_err_km
- min_alt_model_km
- min_alt_truth_km
- status

## Example Commands

CPU smoke:

```bash
python -m lunaris.surrogate.st_lrps.evaluation.compare_gravity_models \
    --random-scenarios 3 --duration-days 0.01 \
    --models sh20,sh80 --truth sh200 \
    --output-dir outputs/gravity_benchmark/smoke_cpu
```

GPU batch smoke:

```bash
python -m lunaris.surrogate.st_lrps.evaluation.compare_gravity_models \
    --random-scenarios 5 --duration-days 0.05 \
    --truth sh200 \
    --gpu-models sh200,sh60,sh20,st_lrps \
    --gpu-batch-compare --rk4-dt-s 10 \
    --output-dir outputs/gravity_benchmark/smoke_gpu_batch_compare
```

## ST-LRPS Note

ST-LRPS comparison is optional and is treated as learned residual-potential
surrogate validation. Provide a trained artifact directory through the harness
options when comparing it against the spherical-harmonic reference. This README
documents validation behavior and output schemas rather than package internals.

## Generated Outputs

Validation outputs should be written under the repository-level `outputs/`
directory, usually `outputs/gravity_benchmark/`, or an external scratch path. Do
not commit generated plots, cached truth trajectories, metrics tables, reports,
checkpoints, or trained model artifacts; the `outputs/` tree is git-ignored.

## Implementation Layout

The public command module is:

```text
src/lunaris/surrogate/st_lrps/evaluation/compare_gravity_models.py
```

Implementation modules live in:

```text
src/lunaris/surrogate/st_lrps/evaluation/_gravity_benchmark/
```

- `types.py`: shared dataclasses and result types
- `compute.py`: propagation and error computation
- `metrics.py`: metric aggregation
- `modes.py`: validation modes such as CPU smoke, random scenarios, and GPU
  batch compare
- `plotting.py`: figures
- `results_io.py`: metrics tables and report I/O

Use the public command module for CLI and automation entry points.
