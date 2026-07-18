# Force-Model Validation Memo (Phase 0/1 + Test Suite)

Scope: the perturbation force models and ephemeris application used by the
Lunaris equations of motion (`src/lunaris/core/dynamics/`,
`src/lunaris/physics/*`, and the CUDA batch propagation path
`src/lunaris/core/batch_propagator.py`). This memo records what was changed, why,
and the measured evidence that the result is physically and numerically correct
to an industry/research standard. It follows the `astrodynamics-validation`
procedure (dimensions, frames/signs, finite-difference gradients, convergence,
CPU/GPU agreement, limiting cases).

## 1. Conventions (verified against the kernels, not assumed)

| Quantity | Convention | Source of truth |
|---|---|---|
| Units | strict SI (m, m/s, m/s², s, kg) | `lunaris.common.constants` |
| Integration frame | Moon-centred inertial (J2000-like) | `core/dynamics/engine.py` |
| Gravity/tide eval frame | Moon-fixed (body-fixed) | `physics/spherical_harmonics.py` |
| Frame bridge | `q_i2f` (scalar-first `[w,x,y,z]`); inverse via **conjugate** | `physics/ephemeris.py` |
| Potential → accel | `a = +∇U` (geodesy sign) | confirmed by finite difference, §3 |

## 2. Changes made

### 2.1 Third-body differential gravity → cancellation-free Battin form
The differential (tidal) third-body acceleration
`a = mu[(r_tb − r_sc)/|r_tb − r_sc|³ − r_tb/|r_tb|³]` is the small difference of
two large, nearly-equal vectors (for a lunar orbiter `|r_sc|/|r_tb|` ≈ 5e-3 for
Earth, ≈1e-5 for Sun). Evaluated as a literal subtraction it loses leading
digits. It is now evaluated with the Battin `F(q)` device (the form GMAT and
other professional propagators use), which is mathematically identical but
free of the cancellation:

```
q  = (|r_sc|² − 2 r_sc·r_tb)/|r_tb|²
F  = q(3 + 3q + q²)/(1 + (1+q)^{3/2})      # ≡ (1+q)^{3/2} − 1, without the subtraction
a  = −mu/|d|³ (r_sc + r_tb·F),   d = r_tb − r_sc
```

Applied identically to the CPU kernel
(`physics/third_body_effects.py::accel_third_body_numba`) and the CUDA kernel
(`batch_propagator.py::_third_body_cuda`) to preserve CPU/GPU parity. Both backends
are float64.

**Evidence** (relative error vs a 50-significant-digit `decimal` reference on the
exact float64 inputs):

| Geometry | direct difference | Battin `F(q)` | improvement |
|---|---|---|---|
| LLO / Earth (axial) | 2.86e-15 | 1.00e-17 | 285× |
| LLO / Earth (oblique) | 1.95e-14 | 5.21e-16 | 38× |
| LLO / Sun | **1.13e-11** | 5.30e-16 | **21361×** |
| High lunar orbit / Earth | 2.09e-15 | 3.40e-16 | 6× |

The Sun term, previously the worst-conditioned, is now at machine precision.
Locked by `tests/test_third_body_precision.py`.

### 2.2 GPU ephemeris vec3 interpolation aligned to the CPU scheme
The CPU runtime sampler (`physics/ephemeris.py::interp_vec3_safe`) uses
constant / linear / Catmull-Rom by table size, but the CUDA `_interp3_cuda` was
linear-only — a backend inconsistency in how the *same* Sun/Earth ephemeris table
is applied (~1e-9 relative scheme difference between integration nodes). The CUDA
kernel now mirrors `interp_vec3_safe` exactly (the quaternion path already matched
CPU SLERP). Validated by reproducing the device logic as a `numba.njit` mirror and
comparing against the CPU sampler across table sizes and the full time span:
max discrepancy ≈ 1 ULP at position scale (≈1.5e-16 relative), i.e. ~7 orders of
magnitude tighter than before. Locked by
`tests/test_ephemeris_interpolation.py`. End-to-end CPU/GPU parity remains covered
by `tests/test_real_asset_cpu_gpu_validation.py` on a CUDA host.

### 2.3 CPU SRP + conical eclipse — reviewed, **no change**
Verified empirically (magnitude vs closed form, direction away from Sun,
inverse-square `|a|·d²` invariance, umbra→0, penumbra continuity/monotonicity in
[0,1]). The runtime `|r_earth|² > 1e12` test that gates Earth-shadow is a robust
detection of a real vs collapsed (`(1,3)` zero-row) Earth table and was
intentionally left in place; replacing it with a build-time flag would change
behaviour when the ephemeris carries a real Earth vector while no Earth
perturbation is enabled. Locked by `tests/test_srp_eclipse.py`.

SRP is a cannonball `Cr*A/m` area model, not an attitude-dependent flat-plate
force. Batch manifests record this as `srp_force_model`.

Numba CUDA SRP is intentionally lower fidelity for screening: cylindrical Moon
umbra and no Earth eclipse. Batch archives record this under `srp_shadow_model`
and `srp_shadow_model_fidelity`; CPU/GPU SRP parity claims must account for it.

### 2.4 Selected 1PN relativity scope and ephemeris state consistency
The relativity flag enables selected 1PN corrections: central-body
Schwarzschild, external-body differential Schwarzschild, and de Sitter/geodetic
terms when ephemeris tables are available. It is not full relativistic N-body
dynamics: EIH terms, Lense-Thirring frame dragging, J2-relativistic coupling,
and clock/time-dilation models are outside the current scope. Batch manifests
record this as `relativity_model=selected_1pn_corrections`.

For the external terms, Sun/Earth velocity is now the analytic derivative of
the same clamped Catmull-Rom position interpolant used at that integration
epoch. The CPU and CUDA paths implement the same polynomial derivative, so the
external 1PN state is C1 within an ephemeris cell instead of mixing a smooth
position with a piecewise-constant finite-difference velocity.

### 2.5 Surface-radiation eclipse fidelity — reviewed, **no change**
Albedo and equilibrium thermal-IR facet sums use one Moon-center Earth-shadow
factor rather than per-facet occultation. This is an engineering approximation
documented by the kernels and recorded in batch manifests through
`force_model_fidelity`; high-accuracy surface-radiation claims must call it out.

## 3. Finite-difference gradient checks (`a = +∇U`)

Central differences of each potential vs the analytic acceleration kernel; bar is
~1e-6 relative in float64. Conservative fields additionally checked curl-free
(symmetric acceleration Jacobian).

| Force | analytic vs FD (relative) | curl-free |
|---|---|---|
| Solid tide, degree 2 (Earth & Sun) | ~2e-10 | yes |
| Solid tide, degree 3 (Earth & Sun) | ~4e-10 | yes |
| Earth J2 differential | ~1e-10 | yes |
| SH gravity → point-mass limit (zero coeffs) | 3.5e-16 | — |
| SH zonal field axial symmetry | <1e-10 | — |

Locked by `tests/test_force_gradients.py`.

### 3.1 Solid-tide system convention

The elastic solid tide (`physics/solid_tides.py`) is an additive, time-varying
perturbation on the static SH field. It assumes the loaded gravity coefficients
are a **tide-free** field, so the permanent tidal deformation is not double
counted. GRAIL GRGM / jggrx products are tide-free, which is the intended
pairing. A *zero-tide* or *mean-tide* coefficient set would need its permanent
tidal term converted to tide-free before use. Lunaris does not auto-detect a
file's tide system — this is a per-product modelling assumption, documented here
and in the module, not enforced in code.

Deliberately deferred (not oversights): tidal time lag / dissipation
(imaginary Love number, k2/Q — only matters for multi-month arcs), and
propagation of k2 measurement uncertainty (a UQ input, not a force change).
Ocean and thermal tides do not apply to the Moon.

### 3.2 Adaptive SH degree blend — policy and quantitative cost

The altitude-aware dual-fidelity blend (`GravityModel.accel_adaptive`) smoothly
transitions the evaluated SH degree with altitude. It is a **speed option for
exploratory use only**: reference / paper runs must use a single fixed degree so
the result's error is attributable to one model. This is enforced fail-closed —
enabling adaptive degree under a paper-safe / benchmark / strict posture raises
(`core/dynamics/preparation.prepare_adaptive_gravity_policy`), it is never a
silent downgrade.

The cost is not limited to symplectic methods. The production selector
quantizes the degree downward at discrete altitude thresholds, so the
acceleration is a discontinuous function of position at every threshold.
Adaptive-step integrators (DOP853/RK45) absorb the jump through their error
control rather than breaking, but the local error estimator sees each crossing
as a model change: step rejections can rise near thresholds and small phase
artifacts can enter the trajectory. This is the second reason (besides error
attributability) that reference runs use a single fixed degree. A
potential-level spectral taper whose derivative is included in the acceleration
would remove the discontinuity at its source; that is a physics-changing
design decision, deliberately not implemented alongside the current kernels.

Quantitative study (`tools/blend_error_study.py`, synthetic degree-64 field,
5–400 km, transition band 50–250 km):

| Metric | Blend | Hard degree switch |
|---|---|---|
| Max rel. accel error vs fixed degree | ~1.7e-5 | ~4.3e-5 |
| Max residual gradient (discontinuity proxy, per km) | ~5.4e-7 | ~1.1e-5 |

The blend is ~20× smoother across the transition than a hard switch, but it is
still a two-degree approximation of the full field (~1.7e-5 relative; ~9 m
integrated end-of-orbit position difference over one period against the fixed
degree). Hence the fixed-degree requirement for reference runs. Re-run the study
with `python tools/blend_error_study.py` (artifacts land under the git-ignored
`outputs/blend_study/`).

## 4. Integration invariants

- **Energy conservation:** circular lunar orbit, RK4 over one period at
  `dt = T/4000` — specific-energy drift ≤ 1e-9 relative; closure to start ≤ 1e-3·R.
- **Convergence order:** RK4 global error vs the analytic circular orbit halves
  by ~16× when `dt` is halved (measured ratio within [12, 20]).
- **Breakdown consistency:** `get_acceleration_breakdown` component norm equals
  the RHS acceleration magnitude for a single-force config (rel 1e-12).
- **Model on/off & unsupported combos:** toggling relativity changes the RHS;
  SH-without-model, third-body-without-ephemeris, Earth-J2-without-params,
  tides-k3-without-k2, and SRP-with-zero-area all raise. Never a silent drop.

Locked by `tests/test_dynamics_invariants.py` and
`tests/test_force_limiting_cases.py`.

## 5. Regression status

All physics, core, batch, parity, and event suites pass after the changes
(no GPU on the validation host; CUDA kernels compile-guarded and exercised by the
end-to-end parity tests on a CUDA host). New coverage: 45 tests across the five
files above.

Reproduction probes used during the review live under `outputs/` (git-ignored):
`tb_cancellation_probe.py`, `tb_deployed_verify.py`, `srp_eclipse_probe.py`,
`gradient_sign_probe*.py`, `gpu_interp_mirror_verify.py`.
