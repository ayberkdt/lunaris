# Backend Capability Matrix

Last verified: 2026-07-05

> The former `gpu_st_lrps_direct` backend (direct residual-acceleration ST-LRPS
> artifacts) was removed from main and archived in the
> `experimental/force-direct-archive` branch. The supported CUDA ST-LRPS
> backends are the conservative potential path and the third-body hybrid.

## GPU qualification

CPU is the accuracy reference; GPU backends must be re-qualified against it on a
recurring cadence, not just once. Two mechanisms:

- **CI (`.github/workflows/cuda-nightly.yml`)** runs the `requires_cuda` suite
  weekly, but only once a self-hosted runner with the `gpu` label is registered
  (Settings → Actions → Runners). Until then it stays queued and never blocks
  PRs.
- **Local (`tools/gpu_qualification.py`)** runs the same suite on the local
  device and writes a timestamped, commit-stamped log under the git-ignored
  `outputs/gpu_qualification/`. It exits non-zero on failure and uses `pytest -rs`
  so a run that merely *skipped* (no CUDA visible) is distinguishable from one
  that truly validated the GPU.

  Schedule it weekly with Windows Task Scheduler:

  ```powershell
  schtasks /Create /SC WEEKLY /D SUN /TN "Lunaris GPU qualification" ^
    /TR "python \"%CD%\tools\gpu_qualification.py\"" /ST 06:00
  ```

  or with cron on Linux (`0 6 * * 0 cd /path/to/lunaris && python tools/gpu_qualification.py`).

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
| `cpu_sh` | classic SH | CPU | adaptive DOP853 | coefficient file / memory | float64 | SH, third-body Sun/Earth, Earth J2, SRP, albedo, thermal IR, solid tides, selected 1PN corrections | Complete supported-force CPU route and default non-GPU route. “Complete” means every currently implemented force flag is routable; surface-radiation/eclipse models remain disclosed engineering approximations in `force_model_fidelity`. |
| `numba_cuda_sh` | classic SH | CUDA | fixed-step RK4 | 24 | float64 | SH, third-body Sun/Earth, Earth J2, SRP, selected 1PN corrections | Low-degree screening backend. The degree-24 ceiling is a CUDA workspace limit, not a physical limit. SRP uses a cannonball Cr*A/m area model; shadowing is reduced-fidelity on this path: cylindrical Moon umbra, no Earth eclipse. Archives disclose this as `srp_force_model` and `srp_shadow_model`. |
| `torch_cuda_sh` | classic SH | CUDA | fixed-step RK4 | coefficient file / VRAM / batch | float32, float64 | SH only | High-degree GPU SH route. Any added perturbation causes an explicit recorded fallback. |
| `torch_cpu_sh` | classic SH | CPU | fixed-step RK4 | coefficient file / memory | float32, float64 | SH only | CUDA-free validation route for the torch SH evaluator. |
| `gpu_st_lrps_potential` | ST-LRPS | CUDA | fixed-step RK4 | surrogate artifact | float32, float64 | ST-LRPS gravity only | Uses potential-autograd runtime artifacts. Added perturbations fall back to CPU. |
| `gpu_st_lrps_third_body` | ST-LRPS | CUDA | fixed-step RK4 | surrogate artifact | float32, float64 | ST-LRPS gravity plus Sun/Earth third-body | Uses potential-autograd runtime artifacts plus analytic vectorized Battin F(q) third-body terms. Other perturbations fall back to CPU. |
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
- ST-LRPS `auto` upgrades to `gpu_st_lrps_third_body` only when the active
  extras are Sun/Earth third-body and PyTorch CUDA is available.
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
| `requested_dt_s` | User/config requested fixed-step RK4 substep. For CPU adaptive runs this is recorded as the batch request, not a solver step. |
| `effective_dt_s` | Realized fixed-step RK4 substep after aligning the snapshot grid; present for fixed-step backends. |
| `steps_per_snapshot` | Number of RK4 substeps per recorded snapshot for fixed-step backends. |
| `effective_output_dt_s` | Realized snapshot spacing after forcing the final epoch to land exactly on `duration_s`. |
| `srp_shadow_model` | SRP eclipse/shadow implementation used by the backend when SRP is active. |
| `srp_shadow_model_fidelity` | Whether the canonical implemented conical-shadow model or a recorded backend approximation was used. Historical machine values are compatibility labels, not claims of complete physical fidelity. |
| `srp_force_model` | SRP area/attitude force model, currently `cannonball_cr_area_over_mass` rather than attitude-dependent flat plate. |
| `relativity_model` | Relativity scope label, currently `selected_1pn_corrections`, not full relativistic N-body dynamics. |
| `force_model_fidelity` | Nested copy of active non-gravity fidelity labels for paper/report consumers. |

The compatibility value `fidelity_class="full"` means that a CPU route accepts
all currently implemented force flags. It does not mean that Lunaris implements
every physical effect or that each enabled model is exact; per-force approximation
labels and limitations remain authoritative.
