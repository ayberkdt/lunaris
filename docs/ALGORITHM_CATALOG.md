<!-- GENERATED FILE - do not edit by hand.
     Edit docs/algorithms/algorithm_registry.yaml and run
     `python tools/algorithm_registry.py generate`. -->

# Lunaris Algorithm Catalogue

Human-readable view of the algorithm-traceability registry. The source of truth is [`docs/algorithms/algorithm_registry.yaml`](algorithms/algorithm_registry.yaml); this file is generated. See [`docs/ALGORITHM_TRACEABILITY_POLICY.md`](ALGORITHM_TRACEABILITY_POLICY.md) for the naming, citation and classification policy.

**3 entries.**

Implementation class: delegated_library (1), exact (1), exact_reformulation (1)

Verification status: identifier_verified_content_pending (1), verified_primary_source (2)

## Index

| ID | Method | Class | Verification |
| --- | --- | --- | --- |
| [`LUNARIS-ALG-INT-001`](#lunarisalgint001) | Dormand-Prince 8(5,3) explicit Runge-Kutta method as delegated to SciPy DOP853 | delegated_library | verified_primary_source |
| [`LUNARIS-ALG-SUM-001`](#lunarisalgsum001) | Kahan compensated summation | exact | verified_primary_source |
| [`LUNARIS-ALG-TB-001`](#lunarisalgtb001) | Battin F(q) cancellation-resistant differential third-body formulation | exact_reformulation | identifier_verified_content_pending |

## Third-body gravity (TB)

<a id="lunarisalgtb001"></a>
### LUNARIS-ALG-TB-001 -- Battin F(q) cancellation-resistant differential third-body formulation

- **Slug**: `battin_fq_differential_third_body`
- **Category**: numerical_algorithm | **Domain**: TB | **Status**: active
- **Classification**: exact_reformulation
- **Verification**: identifier_verified_content_pending | **Scientific status**: implemented_and_tested
- **Primary reference**: `Battin1999Astrodynamics` -- Battin, 1999. "An Introduction to the Mathematics and Methods of Astrodynamics" (edition Revised) [ISBN: 978-1-56347-342-5]
- **Verification notes**: Book metadata verified: Richard H. Battin, "An Introduction to the Mathematics and Methods of Astrodynamics", Revised Edition, AIAA Education Series, 1999, ISBN 978-1-56347-342-5. The implemented device rewrites F = (1+q)^{3/2} - 1 as q(3+3q+q^2)/(1+(1+q)^{3/2}) to avoid the subtractive cancellation of the direct differential form; this identity is elementary and was checked independently, and the F(q)/Encke device is attributed to Battin by GMAT's MathSpec and multiple modern propagators. PENDING: exact Battin chapter/section/equation number (physical copy needed) before this may be promoted to verified_primary_source.
- **Mathematical contract**:
  - Inputs: Moon-centred spacecraft and third-body position vectors and the third-body gravitational parameter mu (SI units).
  - Outputs: differential (tidal) third-body acceleration in m/s^2
  - Exactness: exact_algebraic_reformulation
  - Preserves: algebraically equal to mu*[(r_tb-r_sc)/|r_tb-r_sc|^3 - r_tb/|r_tb|^3]
  - Preserves: removes catastrophic cancellation for small |r_sc|/|r_tb|
- **Implementing symbols**:
  - `src/lunaris/physics/third_body_effects.py` -- `accel_third_body_numba` (numba_implementation)
  - `src/lunaris/physics/third_body_effects.py` -- `calc_3rd_body_accel` (api_entry_point)
  - `src/lunaris/core/torch_third_body.py` -- `third_body_accel_batch` (torch_implementation)
  - `src/lunaris/core/batch_propagator.py` -- `_third_body_cuda` (cuda_implementation)
- **Lunaris modifications**:
  - float64 numba CPU kernel, torch batch mirror, and CUDA kernel
  - Moon-centred origin and Lunaris ephemeris/frame contracts
- **Assumptions**:
  - point-mass third body
  - both vectors share the same origin and frame
- **Limitations**:
  - does not model the third body's non-spherical gravity
  - returns a zero vector inside a 1 m singularity guard
- **Validated by**:
  - `tests/test_third_body_precision.py`
  - `tests/test_third_body_effects.py`
  - `tests/test_torch_third_body.py`
- **Notes**: This is a numerically stable reformulation of the SAME differential acceleration, not a different physical force model. Do not describe it as a distinct perturbation.

## Compensated summation (SUM)

<a id="lunarisalgsum001"></a>
### LUNARIS-ALG-SUM-001 -- Kahan compensated summation

- **Slug**: `kahan_compensated_summation`
- **Category**: numerical_algorithm | **Domain**: SUM | **Status**: active
- **Classification**: exact
- **Verification**: verified_primary_source | **Scientific status**: implemented_and_tested
- **Primary reference**: `Kahan1965ReducingErrors` -- Kahan, 1965. "Pracniques: Further Remarks on Reducing Truncation Errors" (pages 40) [DOI: 10.1145/363707.363723]
- **Verification notes**: W. Kahan, "Pracniques: Further Remarks on Reducing Truncation Errors", Communications of the ACM 8(1):40, January 1965. DOI 10.1145/363707.363723 verified via the ACM Digital Library and Wikidata; the compensated-sum recurrence (running sum, carried error term, corrected addend) matches the one-page original exactly.
- **Mathematical contract**:
  - Inputs: running sum, carried compensation term, next addend
  - Outputs: updated running sum and updated compensation term
  - Exactness: exact_error_free_transformation
  - Preserves: recovers low-order bits lost to rounding during accumulation
  - Preserves: reduces summation error growth from O(n*eps) toward O(eps)
- **Implementing symbols**:
  - `src/lunaris/physics/spherical_harmonics.py` -- `_kahan_sum_step` (numba_implementation)
- **Lunaris modifications**:
  - Applied per-step inside the high-degree spherical-harmonic acceleration accumulation; the algorithm itself is unchanged.
- **Assumptions**:
  - IEEE-754 round-to-nearest float64 arithmetic
  - compiler/JIT does not re-associate the compensation subtraction
- **Limitations**:
  - does not help when individual addends are themselves inaccurate
  - guards against cancellation across many terms, not within one term
- **Validated by**:
  - `tests/test_spherical_harmonics.py`
  - `tests/test_independent_sh_validation.py`
- **Notes**: Numba does not reorder these floating-point operations, so the compensation term survives compilation; this is why the recurrence is kept as an explicit four-line helper rather than a bare Python sum.

## Integrators (INT)

<a id="lunarisalgint001"></a>
### LUNARIS-ALG-INT-001 -- Dormand-Prince 8(5,3) explicit Runge-Kutta method as delegated to SciPy DOP853

- **Slug**: `dop853_adaptive_runge_kutta_delegation`
- **Category**: integrator | **Domain**: INT | **Status**: active
- **Classification**: delegated_library
- **Verification**: verified_primary_source | **Scientific status**: implemented_and_tested
- **Primary reference**: `Virtanen2020SciPy` -- Virtanen, 2020. "{SciPy} 1.0: Fundamental Algorithms for Scientific Computing in {Python}" (section scipy.integrate.solve_ivp method='DOP853'; pages 261-272) [DOI: 10.1038/s41592-019-0686-2]
- **Verification notes**: Lunaris delegates adaptive integration to scipy.integrate.solve_ivp; the DOP853 method is SciPy's Python port of Hairer's DOP853 (explicit Runge-Kutta of order 8(5,3) with a 5th/3rd-order error estimator and 7th-order dense output). SciPy paper DOI 10.1038/s41592-019-0686-2 verified (Nature Methods 17(3):261-272, 2020); the official SciPy DOP853 page confirms the order-8(5,3) Dormand-Prince identity and cites Hairer, Norsett & Wanner (1993). Method-origin book ISBN 978-3-540-56670-0 verified; exact Hairer section (II) pending physical-copy confirmation.
- **Mathematical contract**:
  - Inputs: Right-hand-side callable f(t, y), initial state, integration span and absolute/relative tolerances (SI units).
  - Outputs: State trajectory sampled on the requested output grid.
  - Exactness: adaptive_error_controlled_approximation
  - Preserves: embedded 5(3) error estimate drives adaptive step control
  - Preserves: 7th-order dense output for off-node sampling
- **Implementing symbols**:
  - `src/lunaris/core/propagation/integrators/scipy.py` -- `_resolve_scipy_method` (delegation_wrapper)
  - `src/lunaris/core/propagation/scipy_runner.py` -- `run_scipy_propagation` (cpu_implementation)
  - `src/lunaris/common/integrator_methods.py` -- `(module)` (config_surface)
- **Lunaris modifications**:
  - Lunaris only selects and configures the SciPy method (token resolution, tolerance policy, chunked output grid); the stepper itself is unmodified.
- **Assumptions**:
  - non-stiff dynamics over each integration chunk
  - right-hand side is sufficiently smooth for high-order accuracy
- **Limitations**:
  - not symplectic; long-horizon energy behaviour is not structurally bounded
  - stiff regimes should use an implicit method (Radau/BDF) instead
- **Validated by**:
  - `tests/test_integrators.py`
  - `tests/test_solver_policy_and_telemetry.py`
  - `tests/test_solver_tolerance_defaults.py`
- **Notes**: Additional SciPy methods (RK45, RK23, Radau, BDF, LSODA) share this delegation surface and will receive their own ids during Phase 2.
