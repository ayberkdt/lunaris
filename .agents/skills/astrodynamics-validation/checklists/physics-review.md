# Physics & Numerical Review Checklist

Work top to bottom. A force model / frame / integrator change is "validated" only
when every applicable line has concrete evidence — not "the unit test is green".

## 1. Units & constants
- [ ] Every term reduces to m/s² (write out the dimensional reduction once).
- [ ] All physical constants come from `lunaris.common.constants`
      (`MU_MOON`, `R_MOON`, `MU_SUN`, `MU_EARTH`, `AU`, `P_SUN_1AU`, `R_EARTH_MEAN`).
- [ ] No locally redefined constant with a different value.

## 2. Frames & rotations
- [ ] Each quantity's frame is identified: inertial (integration) vs Moon-fixed
      (SH evaluation).
- [ ] `q_i2f` (scalar-first `[w,x,y,z]`) rotates inertial→fixed; the **conjugate**
      rotates the fixed-frame acceleration back to inertial.
- [ ] Multi-stage integrators re-evaluate the frame quaternion at the **correct
      stage epoch** (not a single start/end epoch approximation).
- [ ] Cross-check the convention against `core/mc_propagator.py::_quat_rot_cuda`
      and `core/torch_sh_propagator.py` rather than memory.

## 3. Signs & limiting cases
- [ ] Potential→acceleration sign matches `physics/spherical_harmonics.py`
      (and ST-LRPS `Δa = a_sign·∇ΔU`).
- [ ] Point-mass limit (all non-central coeffs zero) reduces to `-µ r / |r|³`.
- [ ] A zonal-only field is axisymmetric; a single tesseral term has the expected
      longitude dependence.

## 4. Finite-difference gradient check (mandatory for any U→a path)
- [ ] Compare analytic `a` to a central finite difference of `U` at several radii
      (e.g. 50/100/500/2000 km) and latitudes incl. near-pole.
- [ ] Agreement ~1e-6 relative (float64). If it fails, the kernel is wrong — fix
      the derivation, do **not** loosen the tolerance.

## 5. Integrator behavior
- [ ] Halving `dt` reduces error at the expected order (RK4 → ~16×).
- [ ] Adaptive (DOP853) tolerance sensitivity is sane (tighter tol → smaller error,
      converging).
- [ ] Energy / element drift over a representative span is bounded and explained.

## 6. CPU/GPU agreement
- [ ] New path vs CPU reference (`GravityModel.accel_fixed`) at matched dtype.
- [ ] float64 strict (~1e-9 rel); float32 reported separately (~1e-6 rel).
- [ ] Tolerances were **measured**, not chosen to pass.

## 7. Activation & unsupported combinations
- [ ] Toggling the model flag changes the result (the term is actually wired).
- [ ] Unsupported flag combinations error or fall back **explicitly** (recorded),
      never silently drop a force.
- [ ] Documented model limitations (`docs/ARCHITECTURE.md`,
      `docs/PERTURBATION_BUDGET.md`) still hold.

## Evidence to capture
Units reduction; frame/sign findings; finite-difference table (radius, lat,
analytic, numerical, rel err); convergence numbers; CPU/GPU agreement figures with
the chosen tolerances and why.
