# Common Layer Cleanup Plan

Status legend: ☐ todo · ◐ in progress · ☑ done

Scope: audit of `src/lunaris/common/` plus a sweep of the rest of `src/lunaris/`
for SSOT violations and structures that belong in the shared layer.

## Findings

### F1 — SSOT violation: hardcoded speed of light in `core/mc_propagator.py` (HIGH)
The CUDA device kernels hardcode `c_light = 299_792_458.0` instead of using the
SSOT constant:
- `src/lunaris/core/mc_propagator.py:611` (`_relativity_1pn_cuda`)
- `src/lunaris/core/mc_propagator.py:664` (`_de_sitter_cuda`)

Every CPU path (`relativity_effects.py`, `surface_effects.py`, `thermal_ir.py`,
`lunar_albedo.py`, `dynamics.py`) takes `C_LIGHT` from `common.constants`. The
module already imports other constants from `common.constants` at line 60.
Numba CUDA device functions capture a module-level Python float as a
compile-time constant, so referencing `C_LIGHT` is safe. Matters for CPU/GPU
numerical consistency: the constant must be changeable from one place.

### F2 — `common/__init__.py` API inconsistency (MEDIUM)
`_LAZY_MODULES` (`src/lunaris/common/__init__.py:32`) lists only `math_utils`,
`time_utils`, `montecarlo_defs`. The layer also ships `lunar_data.py`,
`paths.py`, `hashing.py` — real shared modules (`lunar_data` exists precisely to
keep physics→core off the ST-LRPS subsystem). They are not attribute-accessible
via the package `__getattr__`/`__dir__` and are absent from the package
docstring (`src/lunaris/common/__init__.py:17-20`), which is now stale.

### F3 — intra-`common` minor inconsistencies (LOW)
- `time_utils.py:282`: `US_PER_DAY = 86_400_000_000` redefined inside
  `jd_to_date_tuple`, shadowing the module constant at `time_utils.py:77`.
  njit can read the module global; the local can be removed.
- `math_utils.py:54-56`: docstring references nonexistent `EPS2_1E18` / `EPS2_*`;
  the real names in `constants.py` are `EPS_1E18` etc. Docstring drift.

### F4 — hardcoded `86400` instead of `DAY_S` (LOW, mechanical)
`common.constants.DAY_S` exists but ~30 sites use literal `/86400.0` /
`*86400.0`. core/physics should use `DAY_S`; UI/plot/viz is lower value.

### Deliberately NOT moved to common
`core/torch_frame.py` quaternion torch helpers stay in core — `torch` is an
optional dependency and `common` must stay torch-free. Math/grid/quaternion/COE,
time, hashing, and the MC output grid are already centralized correctly.

## Parallel workstreams (disjoint file sets — safe to run concurrently)

### A — SSOT c_light (HIGH) ☑
Files: `core/mc_propagator.py`
- Add `C_LIGHT` to the `common.constants` import (line 60).
- Replace the two `c_light = 299_792_458.0` with `c_light = C_LIGHT`.
- Verify: `python -c "import lunaris.core.mc_propagator"` and the relativity
  tests (`tests/test_de_sitter_precession.py`, `tests/test_external_schwarzschild.py`).

### B — common API consistency + intra-common cleanups (MEDIUM) ☑
Files: `common/__init__.py`, `common/time_utils.py`, `common/math_utils.py` (docstring only)
- Register `lunar_data`, `paths`, `hashing` in `_LAZY_MODULES`; refresh the
  package docstring.
- Remove the local `US_PER_DAY` in `time_utils.jd_to_date_tuple`.
- Fix the `EPS2_*` → `EPS_*` docstring drift in `math_utils`.
- Verify: `python -c "import lunaris.common; lunaris.common.paths; lunaris.common.hashing; lunaris.common.lunar_data"`.

### C — DAY_S sweep in core/physics (LOW) ☐
Files: `core/mc_runner.py`, `core/monte_carlo_engine.py`, `core/config.py`, `physics/ephemeris.py`
- Replace literal `86400` with `DAY_S` (add import). Verify: `pytest tests/ -q`.

### D — DAY_S sweep in UI/analysis/viz (LOW, optional) ☐
Files: `analysis/**`, `visualization/**`, `ui/**`, `surrogate/**/_gravity_benchmark/**`
- Same `DAY_S` substitution. Open question: whether this is worth the churn.

Sequencing: A and B first (highest value, independent). C/D are mechanical and
can follow or run in parallel. No workstream changes constant *values*, so there
are no cross-stream conflicts.

### E — math_utils completeness & hygiene (MEDIUM) ☑
Files: `common/math_utils.py`, `tests/test_math_utils.py`
Audit of `math_utils` for missing/under-optimized helpers:
- Hygiene: `_rv_to_coe_kernel` inclination now uses the new `safe_acos`
  (replaced `clamp(x=..., lo=-1, hi=1)` int-literal kwarg call); `_quat_slerp`
  threshold references `constants.NEARLY_UNIT`; grid kernel uses `EPS_1E12`
  instead of a `1e-12` literal.
- Added the helpers the module docstring already advertised but did not
  implement: `dot3`, `cross3`, `vec3_normalize`, `safe_acos`, and the signed
  `wrap_angle_pi` (companion to `wrap_angle_2pi`).
- Added `coe_to_rv` — the inverse of `rv_to_coe_select` (perifocal / Vallado
  R3-R1-R3), supporting elliptical and hyperbolic conics. This function had been
  reimplemented inside `tests/test_math_utils.py`; the test now aliases the
  library version, and new tests cover the primitives, a hyperbolic round-trip,
  and input-validation guards.
- Refreshed `__all__` and the module docstring. Verify: `pytest
  tests/test_math_utils.py -q` (93 passed).
- Deliberately left intact: the validated `_rv_to_coe_kernel` internals (not
  rewritten to call the new primitives) to avoid risk in a hot, well-tested
  kernel; torch quaternion helpers stay in `core` (common is torch-free).
