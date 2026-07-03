# Backend Capability Matrix

Last verified: 2026-06-27

This document summarizes the batch propagation backend capability surface after
the modular refactor. The executable source of truth remains the code:

- Capability registry: `src/lunaris/core/backend_capabilities.py`
- Selection policy: `src/lunaris/batch/backend_policy.py`
- Batch package adapter: `src/lunaris/batch/backend_policy.py`

Do not update this table without checking those modules and the backend policy
tests. The table is documentation; the registry and resolver decide runtime
behavior.

## Registered Backends

| Backend | Family | Device | Integrator | SH degree limit | Dtypes | Supported force models | Notes |
|---|---|---|---|---|---|---|---|
| `cpu_sh` | classic SH | CPU | adaptive DOP853 | coefficient file / memory | float64 | SH, third-body Sun/Earth, Earth J2, SRP, albedo, thermal IR, solid tides, 1PN relativity | Full-fidelity CPU fallback and default non-GPU route. |
| `numba_cuda_sh` | classic SH | CUDA | fixed-step RK4 | 24 | float64 | SH, third-body Sun/Earth, Earth J2, SRP, 1PN relativity | Low-degree screening backend. The degree-24 ceiling is a CUDA workspace limit, not a physical limit. |
| `torch_cuda_sh` | classic SH | CUDA | fixed-step RK4 | coefficient file / VRAM / batch | float32, float64 | SH only | High-degree GPU SH route. Any added perturbation causes an explicit recorded fallback. |
| `torch_cpu_sh` | classic SH | CPU | fixed-step RK4 | coefficient file / memory | float32, float64 | SH only | CUDA-free validation route for the torch SH evaluator. |
| `gpu_st_lrps_potential` | ST-LRPS | CUDA | fixed-step RK4 | surrogate artifact | float32, float64 | ST-LRPS gravity only | Uses potential-autograd runtime artifacts. Added perturbations fall back to CPU. |
| `gpu_st_lrps_direct` | ST-LRPS | CUDA | fixed-step RK4 | surrogate artifact | float32, float64 | ST-LRPS gravity only | Uses direct residual-acceleration artifacts. No scalar-potential fallback is allowed. |
| `cpu_st_lrps` | ST-LRPS | CPU | adaptive DOP853 | surrogate artifact | float64 | ST-LRPS gravity plus CPU perturbations | CPU path used when ST-LRPS GPU is unavailable or incompatible with requested physics. |
| `auto` | meta | auto | resolved at runtime | resolved at runtime | resolved at runtime | resolved at runtime | Request name only; `resolve_batch_backend_policy()` picks a concrete backend. |

## Selection Rules

Classic SH selection is centralized in `select_classic_sh_backend()` and then
consumed by `resolve_batch_backend_policy()`.

- Explicit `cpu_sh` always selects CPU.
- Explicit `torch_cuda_sh` requires PyTorch CUDA and SH-only physics.
- Explicit `numba_cuda_sh` requires Numba CUDA, degree <= 24, and physics
  supported by the Numba CUDA kernel.
- If `numba_cuda_sh` is requested above degree 24, the
  `sh_fallback_policy` controls whether the resolver errors, tries
  `torch_cuda_sh`, or falls back to CPU.
- `auto` prefers compatible GPU backends when `use_gpu=True`; high-degree
  classic SH tries `torch_cuda_sh` before CPU.
- GPU paths never silently disable requested perturbations. Unsupported physics
  produces either a hard error or a recorded fallback.

## Provenance Fields

Batch archives should keep these fields aligned with the backend plan:

| Field | Meaning |
|---|---|
| `requested_batch_backend` | User/config request before fallback. |
| `actual_batch_backend` | Concrete backend used for propagation. |
| `requested_sh_degree` | Requested classic-SH degree; never clipped silently. |
| `actual_sh_degree` | Degree actually evaluated by the selected backend, when applicable. |
| `fallback_applied` | Whether backend selection substituted a different backend. |
| `fallback_reason` | Human-readable reason for fallback. Empty when no fallback occurred. |
| `backend_family` | `classic_sh`, `st_lrps`, or `meta` before resolution. |
| `backend_implementation` | Implementation family such as `numba_cuda`, `torch`, or `numpy_numba_cpu`. |
