---
name: astrodynamics-validation
description: >-
  Validate the physical and numerical correctness of Lunaris force models,
  frames, and integrators — beyond "the test passes". Use when adding or
  changing a perturbation (spherical harmonics, third-body, SRP, Earth J2,
  relativity, solid tides), a frame/time transform, a quaternion rotation, an
  integrator, or a CPU/GPU kernel, and when an orbit "drifts", "blows up",
  "has the wrong sign", or "CPU and GPU disagree". Trigger on units, frames
  (inertial vs Moon-fixed), time systems, potential-vs-acceleration sign,
  finite-difference checks, convergence, and CPU/GPU numerical agreement. NOT
  for backend selection/throughput (use mc-backend-and-performance) or ST-LRPS
  evidence claims (use st-lrps-evidence-audit).
---

# Astrodynamics & Numerical Validation

A passing unit test proves the code runs, not that the physics is right. This
skill is the procedure for establishing that a force model, frame transform, or
integrator is physically and numerically correct in Lunaris' conventions.

## Invocation

Auto-trigger; inline. Pair with `lunaris-architecture-guardian` when the change
also moves code between layers, and with `independent-review` for adversarial
sign-off.

## Lunaris conventions (verify against the code, do not assume)

- **Units:** strict SI (m, m/s, m/s², s, kg). Constants come from
  `lunaris.common.constants` (`MU_MOON`, `R_MOON`, `MU_SUN`, `MU_EARTH`, `AU`, …).
- **Frames:** SH gravity is evaluated in the **Moon-fixed (body-fixed)** frame;
  the state is integrated in the **Moon-centered inertial** frame. The bridge is
  the `q_i2f` quaternion (scalar-first `[w,x,y,z]`) from `lunaris.physics.ephemeris`.
  Rotate inertial→fixed with `q_i2f`; rotate the fixed-frame acceleration back to
  inertial with the **conjugate** quaternion. (See `core/dynamics.py`,
  `core/mc_propagator.py::_quat_rot_cuda`, `core/torch_sh_propagator.py`.)
- **Potential vs acceleration sign:** `a = +∇U` with the geodesy potential sign
  convention used in `physics/spherical_harmonics.py`; ST-LRPS residual uses
  `Δa = a_sign · ∇ΔU` (see `docs/ST_LRPS_VALIDATION_HYGIENE.md`). Confirm the sign
  against the existing kernel, not from memory.
- **Reference evaluator:** `physics/spherical_harmonics.py::GravityModel.accel_fixed`
  is the trusted CPU SH reference for cross-checks.

## Required repository sources

- `docs/ARCHITECTURE.md` (perturbation-flag table), `docs/PERTURBATION_BUDGET.md`.
- The kernel under test in `lunaris/physics/` and its caller in `lunaris/core/`.
- Existing physics tests (`tests/test_dynamics.py`, `tests/test_torch_sh_mc_propagator.py`,
  `tests/test_solid_tides.py`, `tests/test_lunar_albedo.py`, `tests/test_ephemeris.py`).

## Procedure (read `checklists/physics-review.md` for the full gate)

1. **Dimensional check.** Every term reduces to m/s². Flag any constant whose
   units you cannot trace to `common.constants`.
2. **Frame & rotation.** Confirm where each quantity lives (inertial vs fixed),
   that `q_i2f` is applied at the correct epoch, and that the inverse uses the
   conjugate — not the same quaternion. For multi-stage integrators, the frame
   must be re-evaluated at each stage epoch (not approximated with one epoch).
3. **Sign & limiting cases.** Point-mass limit (zero non-central coeffs) must
   reduce to `-µ r / |r|³`. Symmetry (e.g. zonal-only field) must hold.
4. **Finite-difference gradient check.** For any potential→acceleration path,
   compare analytic `a` against a central finite difference of `U` at several
   radii/latitudes; agreement to ~1e-6 relative (f64) is the bar. Never accept a
   symbolic derivation into a kernel without this check.
5. **Integrator convergence.** Halve `dt`; confirm the expected order (RK4 → ~16×
   error reduction). Confirm tolerance sensitivity for adaptive paths (DOP853).
6. **CPU/GPU agreement.** Compare the new path against the CPU reference at
   matched dtype: float64 strict (~1e-9 rel), float32 separate looser bound
   (~1e-6 rel). Measure tolerances; do not invent generous ones to pass.
7. **Model on/off & unsupported combos.** Toggling the flag must change the
   result; unsupported combinations must error or fall back explicitly, never
   silently drop a force.

## Verification commands

- `python -m pytest tests/test_dynamics.py tests/test_torch_sh_mc_propagator.py -q`
- A short ad-hoc finite-difference script comparing analytic vs numerical
  gradient (keep it under `outputs/` or a scratch path, not `src/`).

## Stop conditions

- Analytic vs finite-difference gradient disagree beyond tolerance → the kernel
  is wrong; stop and fix the derivation, do not loosen the tolerance.
- A constant or sign cannot be traced to the canonical source → stop and resolve
  before claiming correctness.

## Output

A correctness memo: units table, frame/sign findings, finite-difference and
convergence numbers (with the tolerances and how they were chosen), CPU/GPU
agreement figures, and any limiting-case checks.

## Acceptance

Dimensions consistent; frames/signs proven against the canonical kernel;
finite-difference and convergence checks pass at measured tolerances; CPU/GPU
agreement demonstrated; no silently dropped physics.
