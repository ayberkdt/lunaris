# Lunar Gravity Validation

This module validates lunar gravity models by comparing lower-fidelity models and optional ST-LRPS against a high-fidelity spherical-harmonic truth/reference model.

## Current Harness

The CLI harness now lives inside the ST-LRPS package at:
`src/lunaris/surrogate/st_lrps/evaluation/compare_gravity_models.py`

It is also wired into the ST-LRPS Studio under **Analysis → Orbit-Level Benchmark**.

Run it as:

```bash
python -m lunaris.surrogate.st_lrps.evaluation.compare_gravity_models --help
```

## Reference Hierarchy

- **Truth model**: high-degree spherical harmonics, usually SH200 in the current harness.
- **Baseline models**: lower-degree spherical-harmonic models such as SH20, SH60, SH80, SH120, SH160.
- **Optional learned model**: ST-LRPS residual-potential surrogate, when an artifact directory is provided.

## Validation Modes

Current validation modes at a high level:
- CPU smoke validation
- random scenario propagation
- ST-LRPS force sample trajectory mode
- GPU batch comparison
- full SH-vs-ST-LRPS comparison

## Metrics

Current and expected metrics for gravity validation runs:
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

ST-LRPS comparison is optional and is treated as learned residual-potential surrogate validation. Provide a trained artifact directory through the harness options when comparing it against the spherical-harmonic reference. This README intentionally documents validation behavior rather than unstable package internals.

## Generated Outputs

Validation outputs should be written under the repository-level `outputs/` directory (the canonical location is `outputs/gravity_benchmark/`) or an external scratch path. Do not commit generated plots, cached truth trajectories, metrics tables, reports, checkpoints, or trained model artifacts; the `outputs/` tree is git-ignored.

## Future Refactor Target

An intended future split of `src/lunaris/surrogate/st_lrps/evaluation/compare_gravity_models.py` includes:
- `src/lunaris/surrogate/st_lrps/evaluation/orbit_benchmark/scenarios.py`
- `src/lunaris/surrogate/st_lrps/evaluation/orbit_benchmark/metrics.py`
- `src/lunaris/surrogate/st_lrps/evaluation/orbit_benchmark/runners.py`
- `src/lunaris/surrogate/st_lrps/evaluation/orbit_benchmark/reports.py`
- `src/lunaris/surrogate/st_lrps/evaluation/orbit_benchmark/schemas.py`

(This is a future plan, not current implementation.)
