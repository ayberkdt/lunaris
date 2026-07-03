# Optimization Roadmap

Date: 2026-06-25

This roadmap records the current ST-LRPS and batch propagation backend decision. It is
intentionally conservative: do not claim a faster or more accurate method as the
default until runtime and orbit-level validation have both been run for the same
artifact, hardware, force model, and scenario set.

## Executive Recommendation

1. Keep `potential_autograd` as the validation-safe ST-LRPS runtime.
2. Keep `force_direct` available for deployment and bulk-throughput experiments,
   but mark it experimental until acceleration, curl, and orbit drift are
   validated against the target truth model.
3. Use `sampling_method="sobol_scrambled"` or `sampling_method="lhs"` for
   validation/coverage batches, and `sampling_method="random"` for classical
   random uncertainty draws. Pair that with `batch_backend="auto"` for
   ordinary batch runs, explicit `cpu_sh` for high-fidelity truth/reference,
   explicit `numba_cuda_sh`/`gpu_sh` for the degree-24 Numba CUDA screening tier,
   explicit `torch_cuda_sh` for high-degree gravity-only PyTorch CUDA runs, and
   explicit ST-LRPS GPU backends for throughput.
4. Do not remove the Numba CUDA degree-24 limit by raising a constant. The
   Numba evaluator uses fixed `(26 x 26)` per-thread Legendre workspaces; that
   ceiling belongs to `numba_cuda_sh` and its legacy alias `gpu_sh`, not to the
   entire classic-SH GPU family.
5. For requested classic-SH degrees above 24, route through
   `select_classic_sh_backend()`: use `torch_cuda_sh` when PyTorch CUDA is
   available and the requested physics is gravity-only, otherwise fall back
   explicitly to CPU SH according to policy. No silent clipping, and no
   "GPU SH100" label unless `actual_batch_backend` records `torch_cuda_sh`.

## Method Selection Audit

| Option | Expected speed | Accuracy / physics | Complexity | Recommendation |
|--------|----------------|--------------------|------------|----------------|
| A: `potential_autograd` | Medium. Autograd is expensive, especially for small batches and CPU. | Best current ST-LRPS physical structure because acceleration is the gradient of a learned scalar residual potential. | Already supported. | Keep as default validation/runtime baseline. |
| B: `force_direct` | High. It avoids input-gradient autograd and can run under `torch.no_grad()`. | No potential output and no conservative-field guarantee. Needs curl and orbit validation. | Already supported by the new direct runtime. | Keep experimental; use for deployment throughput after validation. |
| C: Distilled direct-force student | High if the student is small and batched. | Could inherit useful behavior from a potential teacher, but still needs field and orbit checks. | Moderate training/evaluation work. | Benchmark next; this is the most promising speed path. |
| D: Hybrid runtime | High for bulk batch propagation while preserving validation with SH or potential runs. | Strong practical compromise if fallbacks and domain metadata are clear. | Low-to-moderate, mostly policy and reporting. | Recommended operating model for 512-orbit studies. |
| E: Alternative architectures | Unknown without new experiments. | Could improve extrapolation or invariances, but risks scope creep. | Medium to high. | Postpone until baseline/direct-student evidence is collected. |

## 512-Orbit Batch Ensemble Policy

Recommended production workflow:

- Run throughput sweeps with `batch_backend="auto"` or
  `batch_backend="gpu_st_lrps_potential"` when a validated potential artifact is
  available.
- Prefer `sampling_method="sobol_scrambled"` or `sampling_method="lhs"` for
  validation coverage; use `sampling_method="random"` when estimating a true
  ensemble impact probability under a stated uncertainty distribution.
- Use `batch_backend="gpu_st_lrps_direct"` only for deployment-style experiments
  until drift and curl validation pass.
- Run smaller high-degree `batch_backend="cpu_sh"` truth/reference batches to
  quantify the error envelope.
- Use `batch_backend="gpu_sh"` only as the legacy alias for `numba_cuda_sh`
  (degree <= 24). For high-degree GPU classic SH, request `torch_cuda_sh`
  explicitly or let `batch_backend="auto"` select it when the run is gravity-only
  and PyTorch CUDA is available. Requests that cannot be served on GPU must
  record the CPU fallback in requested-vs-actual metadata.

## Implemented Now

- Added `batch_backend` selection with the supported values `auto`, `cpu_sh`,
  `gpu_sh`, `gpu_st_lrps_potential`, and `gpu_st_lrps_direct`.
- Added explicit requested/actual backend metadata, requested/actual SH degree,
  GPU SH capability metadata, runtime model kind, CUDA device name, dtype, and
  fallback reason.
- Removed silent degree clipping in the Numba classic-SH CUDA pack builder.
  Direct use of the low-level Numba CUDA propagator with degree >24 now raises a
  clear runtime error; the high-level policy selects `torch_cuda_sh` for
  compatible high-degree gravity-only runs, or records an explicit CPU fallback.
- Kept `potential_autograd` and `force_direct` runtime paths working, with direct
  force inference using a no-grad torch path through `SurrogateGravityModel`.
- Added UI and CLI support for the explicit backend names.
- Added a measured non-conservativeness (curl) diagnostic for `force_direct`
  artifacts. `lunaris.surrogate.st_lrps.evaluation.force_direct_eval` now reports
  `conservativeness_metrics` (finite-difference `curl a` magnitude plus the
  scale-free `nonconservative_ratio = ||antisym(J)||_F / ||J||_F in [0, 1]`),
  so the "not conservative by construction" caveat is now quantified per artifact
  rather than only asserted.
- Added an orbit-level drift harness,
  `lunaris.surrogate.st_lrps.evaluation.orbit_drift`
  (`python -m lunaris.surrogate.st_lrps.evaluation.orbit_drift --model-dir <run>
  [--ref-model-dir <potential_run>]`). It propagates the same circular orbit
  under two acceleration models with identical RK4 settings and reports position/
  velocity drift, plus an `energy_drift` diagnostic whose secular trend exposes
  the non-conservative content at the orbit level. Both halves of the
  `force_direct` validation gate (local curl + orbit drift/energy) are now
  tooled and unit-tested against analytic Kepler fields; the **remaining gate
  action is to run them on a contract-valid `force_direct` artifact** (paired
  with its potential counterpart) and record the thresholds before the
  experimental label is dropped.

## Benchmark Status

Baseline files live under `outputs/optimization/baseline_profile/`.

The current local workstation probe found PyTorch CUDA on an NVIDIA GeForce GTX
1660 Ti, but Numba CUDA was unavailable in this environment. As a result, the
classic-SH CUDA benchmark could not run here. The available local ST-LRPS
artifact also lacks the newer `artifact_contract` block required by the runtime
benchmark, so ST-LRPS timing is recorded as blocked rather than fabricated.

Because of those constraints, no before/after speedup claim is made in this
roadmap. The next benchmark pass should use a contract-valid potential artifact,
a contract-valid force-direct artifact trained on the same dataset/degree
contract, and a machine where both PyTorch CUDA and Numba CUDA are available.

## What To Benchmark Next

> **Launch these as reproducible scenario sweeps.** The capacity, encoding/loss,
> and force-direct student experiments below are committed as self-describing
> JSONL sweeps under `hpc/scenarios/` and submitted with
> `hpc/slurm_train_scenario_array.sbatch` (see
> [docs/HPC.md](HPC.md#1b-st-lrps-scenario-arrays-reproducible-sweeps)). Each run
> records its scenario, command, and environment for provenance. `force_direct`
> students are trained from `st_lrps_force_direct_student_sweep.jsonl` and remain
> experimental until acceleration, curl, and orbit drift are validated; they are
> never folded into the scalar-potential A0–A6 ablation matrix.

- ST-LRPS runtime: `potential_autograd` vs `force_direct`, CPU and CUDA, batch
  sizes `1, 16, 128, 512, 1024, 8192`.
- Batch classic SH: Numba CUDA degrees `0, 2, 10, 20, 24`; PyTorch CUDA
  high-degree gravity-only runs `50, 100, 200`; batch sizes `32, 128, 512,
  2048`; all with CPU same-degree correctness checks and requested-vs-actual
  backend metadata.
- High-degree truth: CPU SH50/SH100/SH200 against ST-LRPS potential and direct
  force over representative 1-day and 5-day orbit sets.
- Direct-force validity: acceleration absolute/relative error, curl penalty,
  radial/along-track/cross-track RMS, P95/P99/max error, instability/impact
  counts, and any domain fallback counts.

## Postponed

- Higher-degree Numba CUDA SH kernels beyond the current degree-24 workspace
  tier. PyTorch CUDA already covers gravity-only high-degree classic SH; a Numba
  extension would still need generated max-degree specializations or a
  streaming/row-wise recurrence and should not be attempted as a constant
  change.
- CUDA graphs or `torch.compile` for ST-LRPS. These may help repeated fixed
  shapes, but should follow baseline timings and must preserve autograd for
  `potential_autograd`.
- New surrogate architectures such as SIREN variants, curl-penalized vector
  fields, equivariant encodings, FNO, or DeepONet. They require a separate
  experiment plan and should not replace the current runtime by default.

## Delete Or Hide

- Hide or relabel any UI/report wording that implies high-degree classic SH is
  computed on GPU when metadata shows CPU fallback, or when the actual backend is
  not the backend family being claimed.
- Do not publish `force_direct` scientific accuracy claims without orbit-level
  validation.
- Do not keep legacy benchmark artifacts in the default runtime path unless they
  carry the required artifact contract or are explicitly loaded as legacy.
