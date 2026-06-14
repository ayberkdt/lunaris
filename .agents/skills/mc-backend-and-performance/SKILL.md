---
name: mc-backend-and-performance
description: >-
  Reason about Lunaris Monte Carlo / propagation backend selection and
  performance the Lunaris way — through the backend policy, not generic GPU
  advice. Use when asked to "speed up Monte Carlo", "run on CUDA", "why did it
  fall back to CPU", "torch_cuda_sh vs numba_cuda_sh", "is it actually using the
  GPU", "profile this", "high-degree SH on GPU", or to change dt/chunk/dtype for
  throughput. Enforces profile-first, requested-vs-actual backend, supported
  physics, fallback provenance, and CPU-reference accuracy. NOT for adding new
  perturbation math (use astrodynamics-validation) or ST-LRPS training throughput
  alone (use st-lrps-training). Do NOT pull in CuPy/RAPIDS/Warp/Lightning.
---

# Monte Carlo Backend & Performance

Lunaris already has a deliberate backend policy. The failure mode here is generic
"just put it on the GPU" advice that bypasses that policy, silently drops physics,
or adds an unwanted dependency. Optimize *within* the policy.

## Invocation

Auto-trigger; inline. Pair with `astrodynamics-validation` when an optimization
could change numerical results, and `independent-review` before merging a
performance claim.

## Backend map (canonical: `src/lunaris/core/mc_backend_policy.py`, `backend_capabilities.py`, `docs/ARCHITECTURE.md`)

- **Single source of truth for selection** is `select_classic_sh_backend()`,
  consumed by `resolve_mc_backend_policy()`. Do not re-derive routing elsewhere.
- Classic-SH GPU backends are **distinct**:
  - `numba_cuda_sh` (alias `gpu_sh`) — Numba CUDA RK4; degree ≤ 24 is a
    **kernel-workspace** limit (`GPU_SH_MAX_DEGREE`, `_GPU_WS=26`), not physical.
  - `torch_cuda_sh` — PyTorch RK4 (`core/torch_sh_propagator.py`); arbitrary
    degree, gravity-only first form. Degree > 24 with PyTorch CUDA → `torch_cuda_sh`.
- ST-LRPS GPU path: `core/torch_batch_propagator.py`. CPU path: full-fidelity
  DOP853 per sample.
- `requested_sh_degree` is **never clipped**; CPU fallback is recorded, never
  silent, and a CPU run is never labeled with a GPU backend/device.

## Procedure

1. **Profile first.** Identify the real hot path with evidence
   (`docs/profiling.md`); do not optimize by guess.
2. **Identify requested vs actual backend.** Read the run's `MCBackendPlan`
   provenance (`actual_mc_backend`, `actual_device`, `fallback_applied`,
   `fallback_reason`). "It's slow" is often a CPU fallback, not a GPU that's slow.
3. **Verify supported physics.** `torch_cuda_sh`/ST-LRPS GPU paths are
   gravity-only; an active perturbation forces an explicit fallback. Never flip a
   physics flag off just to reach a GPU path.
4. **Account for warm-up & transfer.** Exclude Numba JIT compile and the first
   CUDA launch from steady-state timing; account for host⇄device copies.
5. **Compare equals.** Same scenario, dt, dtype, degree on both sides. Match the
   accuracy target (float64 vs float32) before comparing speed.
6. **Report speed AND error together.** Cross-check against the CPU reference
   (`mc-backend-and-performance` pairs with `astrodynamics-validation`): an
   optimization that loses physics or accuracy is rejected regardless of speedup.
7. **Memory/chunking.** Use the existing VRAM-aware chunking
   (`torch_sh_propagator` chunk preflight); chunk size changes memory only, never
   results.

## Verification

- `python -m pytest tests/test_mc_gpu_policy.py tests/test_classic_sh_policy.py tests/test_backend_capabilities.py tests/test_mc_backend_dispatch.py -q`
- For numerical safety: `tests/test_torch_sh_mc_propagator.py` (CPU/CUDA agreement,
  chunk/batch invariance). CUDA tests skip cleanly without a device — report the
  skip, don't claim a GPU result you didn't run.

## Stop conditions

- An optimization requires disabling a perturbation or loosening accuracy to win →
  stop; that is not a valid speedup.
- A proposal adds CuPy/RAPIDS/Warp/cuDF or PyTorch Lightning → stop; justify
  against the existing torch/numba policy first (almost always unnecessary).

## Output

A performance memo: profiled hot path, requested-vs-actual backend, matched-config
speed numbers WITH the accuracy/error numbers, memory behavior, and any provenance
that was previously misleading.

## Acceptance

Decision is routed through the policy; requested vs actual backend is explicit; no
physics silently dropped; speed reported with error; no new heavy GPU dependency;
relevant backend + numerical tests pass (or skips are reported honestly).
