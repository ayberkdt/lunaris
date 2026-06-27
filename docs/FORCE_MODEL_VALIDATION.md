# Force-Model Validation Memo (Phase 0/1 + Test Suite)

Scope: the perturbation force models and ephemeris application used by the
Lunaris equations of motion (`src/lunaris/core/dynamics/`,
`src/lunaris/physics/*`, and the CUDA Monte-Carlo path
`src/lunaris/core/mc_propagator.py`). This memo records what was changed, why,
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
(`mc_propagator.py::_third_body_cuda`) to preserve CPU/GPU parity. Both backends
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

### 2.3 SRP + conical eclipse — reviewed, **no change**
Verified empirically (magnitude vs closed form, direction away from Sun,
inverse-square `|a|·d²` invariance, umbra→0, penumbra continuity/monotonicity in
[0,1]). The runtime `|r_earth|² > 1e12` test that gates Earth-shadow is a robust
detection of a real vs collapsed (`(1,3)` zero-row) Earth table and was
intentionally left in place; replacing it with a build-time flag would change
behaviour when the ephemeris carries a real Earth vector while no Earth
perturbation is enabled. Locked by `tests/test_srp_eclipse.py`.

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

All physics, core, Monte-Carlo, parity, and event suites pass after the changes
(no GPU on the validation host; CUDA kernels compile-guarded and exercised by the
end-to-end parity tests on a CUDA host). New coverage: 45 tests across the five
files above.

Reproduction probes used during the review live under `outputs/` (git-ignored):
`tb_cancellation_probe.py`, `tb_deployed_verify.py`, `srp_eclipse_probe.py`,
`gradient_sign_probe*.py`, `gpu_interp_mirror_verify.py`.
