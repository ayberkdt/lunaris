<!-- GENERATED FILE - do not edit by hand.
     Edit docs/algorithms/algorithm_registry.yaml and run
     `python tools/algorithm_registry.py generate`. -->

# Lunaris Algorithm Catalogue

Human-readable view of the algorithm-traceability registry. The source of truth is [`docs/algorithms/algorithm_registry.yaml`](algorithms/algorithm_registry.yaml); this file is generated. See [`docs/ALGORITHM_TRACEABILITY_POLICY.md`](ALGORITHM_TRACEABILITY_POLICY.md) for the naming, citation and classification policy.

**8 entries.**

Implementation class: delegated_library (1), exact (3), exact_reformulation (1), heuristic (2), standard_implementation (1)

Verification status: identifier_verified_content_pending (1), unverifiable (2), verified_primary_source (5)

## Index

| ID | Method | Class | Verification |
| --- | --- | --- | --- |
| [`LUNARIS-ALG-INT-001`](#lunarisalgint001) | Dormand-Prince 8(5,3) explicit Runge-Kutta method as delegated to SciPy DOP853 | delegated_library | verified_primary_source |
| [`LUNARIS-ALG-SH-001`](#lunarisalgsh001) | Standard forward-column recursion of fully-normalized associated Legendre functions | exact | verified_primary_source |
| [`LUNARIS-ALG-SH-002`](#lunarisalgsh002) | Spherical-harmonic gravitational acceleration via the classical spherical-coordinate geopotential gradient | standard_implementation | verified_primary_source |
| [`LUNARIS-ALG-SUM-001`](#lunarisalgsum001) | Kahan compensated summation | exact | verified_primary_source |
| [`LUNARIS-ALG-TB-001`](#lunarisalgtb001) | Battin F(q) cancellation-resistant differential third-body formulation | exact_reformulation | identifier_verified_content_pending |
| [`LUNARIS-ALG-TB-002`](#lunarisalgtb002) | Newtonian point-mass (monopole) central-body gravitational acceleration | exact | verified_primary_source |
| [`LUNARIS-HEUR-SH-001`](#lunarisheursh001) | Pole-stable spherical-harmonic order truncation (stable-m limit) | heuristic | unverifiable |
| [`LUNARIS-HEUR-SH-002`](#lunarisheursh002) | Degree-switched spherical-harmonic evaluation with a cubic-Hermite (smoothstep) altitude blend | heuristic | unverifiable |

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
- **See also**: [`LUNARIS-ALG-TB-002`](#lunarisalgtb002)
- **Notes**: This is a numerically stable reformulation of the SAME differential acceleration, not a different physical force model. Do not describe it as a distinct perturbation.

<a id="lunarisalgtb002"></a>
### LUNARIS-ALG-TB-002 -- Newtonian point-mass (monopole) central-body gravitational acceleration

- **Slug**: `newtonian_point_mass_central_gravity`
- **Category**: physical_model | **Domain**: TB | **Status**: active
- **Classification**: exact
- **Verification**: verified_primary_source | **Scientific status**: implemented_and_tested
- **Primary reference**: `MontenbruckGill2000SatelliteOrbits` -- Montenbruck, 2000. "Satellite Orbits: Models, Methods and Applications" (chapter 3 (Force Models); section 3.1 (point-mass / two-body acceleration)) [DOI: 10.1007/978-3-642-58351-3]
- **Verification notes**: Montenbruck & Gill (2000), ISBN 978-3-540-67280-7. The central-body monopole acceleration a = -mu * r / |r|^3 is the elementary Newtonian two-body term introduced in Chapter 3; book identifiers verified as for LUNARIS-ALG-SH-002.
- **Mathematical contract**:
  - Inputs: position vector relative to the central body (m) and mu (m^3/s^2)
  - Outputs: monopole gravitational acceleration (m/s^2)
  - Exactness: exact
  - Preserves: inverse-square central force
  - Preserves: identical to the degree-0 term of the spherical-harmonic field
- **Implementing symbols**:
  - `src/lunaris/physics/spherical_harmonics.py` -- `compute_point_mass_acceleration` (numba_implementation)
  - `src/lunaris/physics/third_body_effects.py` -- `calc_central_body_accel` (api_entry_point)
- **Lunaris modifications**:
  - singularity guard at small radius
- **Assumptions**:
  - spherically symmetric central body
- **Limitations**:
  - ignores all non-spherical and third-body perturbations by construction
- **Validated by**:
  - `tests/test_spherical_harmonics.py`
  - `tests/test_dynamics.py`
- **See also**: [`LUNARIS-ALG-SH-002`](#lunarisalgsh002)

## Spherical-harmonic gravity (SH)

<a id="lunarisalgsh001"></a>
### LUNARIS-ALG-SH-001 -- Standard forward-column recursion of fully-normalized associated Legendre functions

- **Slug**: `fully_normalized_alf_forward_column_recursion`
- **Category**: numerical_algorithm | **Domain**: SH | **Status**: active
- **Classification**: exact
- **Verification**: verified_primary_source | **Scientific status**: implemented_and_tested
- **Primary reference**: `HolmesFeatherstone2002Legendre` -- Holmes, 2002. "A Unified Approach to the {Clenshaw} Summation and the Recursive Computation of Very High Degree and Order Normalised Associated {Legendre} Functions" (section Standard forward-column recursion of normalised ALFs; pages 279-299) [DOI: 10.1007/s00190-002-0216-2]
- **Verification notes**: S. A. Holmes and W. E. Featherstone, "A unified approach to the Clenshaw summation and the recursive computation of very high degree and order normalised associated Legendre functions", Journal of Geodesy 76(5): 279-299, 2002. DOI 10.1007/s00190-002-0216-2 verified via Springer. The implemented diagonal seed sqrt((2n+1)/2n), sub-diagonal sqrt(2n+1) and vertical coefficients A[n,m]=sqrt((2n-1)(2n+1)/((n-m)(n+m))), B[n,m]=sqrt((2n+1)(n-m-1)(n+m-1)/((2n-3)(n+m)(n-m))) are exactly the standard forward-column recursion presented in this paper (also Colombo 1981).
- **Mathematical contract**:
  - Inputs: maximum degree, sin/cos of geocentric latitude
  - Outputs: fully-normalized associated Legendre functions P_nm and their latitude derivatives dP_nm/dphi
  - Exactness: exact_recurrence
  - Preserves: 4-pi (geodesy) full normalization
  - Preserves: numerical stability to very high degree/order
- **Implementing symbols**:
  - `src/lunaris/physics/spherical_harmonics.py` -- `build_legendre_coeffs` (reference_implementation)
  - `src/lunaris/physics/spherical_harmonics.py` -- `_compute_legendre_polynomials_inplace` (numba_implementation)
  - `src/lunaris/physics/spherical_harmonics.py` -- `_apply_legendre_normalization` (numba_implementation)
- **Lunaris modifications**:
  - Geodesy convention: no Condon-Shortley phase, real-harmonic sqrt(2) scaling for m>0 (matches GRAIL/EGM/GRACE/ICGEM coefficient definitions).
  - pole-safe per-order truncation via the stable-m limit (see LUNARIS-HEUR-SH-001)
- **Assumptions**:
  - IEEE-754 float64 arithmetic
  - coefficients supplied in the same normalization convention
- **Limitations**:
  - Condon-Shortley phase must be excluded by the coefficient provider
- **Validated by**:
  - `tests/test_spherical_harmonics.py`
  - `tests/test_independent_sh_validation.py`
  - `tests/test_sh_convention_lock.py`
- **See also**: [`LUNARIS-ALG-SH-002`](#lunarisalgsh002)
- **Notes**: The absence of the Condon-Shortley phase is a convention, not a bug: applying (-1)^m would corrupt tesseral/sectoral terms while leaving zonal (J2) tests unaffected.

<a id="lunarisalgsh002"></a>
### LUNARIS-ALG-SH-002 -- Spherical-harmonic gravitational acceleration via the classical spherical-coordinate geopotential gradient

- **Slug**: `spherical_harmonic_gravity_acceleration`
- **Category**: physical_model | **Domain**: SH | **Status**: active
- **Classification**: standard_implementation
- **Verification**: verified_primary_source | **Scientific status**: implemented_and_tested
- **Primary reference**: `MontenbruckGill2000SatelliteOrbits` -- Montenbruck, 2000. "Satellite Orbits: Models, Methods and Applications" (chapter 3 (Force Models); section 3.2 (gravitational field / spherical-harmonic expansion)) [DOI: 10.1007/978-3-642-58351-3]
- **Verification notes**: O. Montenbruck and E. Gill, "Satellite Orbits: Models, Methods and Applications", Springer, 2000. ISBN 978-3-540-67280-7 and DOI 10.1007/978-3-642-58351-3 verified; Chapter 3 (Force Models) confirmed from the publisher table of contents, with the gravitational spherical-harmonic expansion in Section 3.2. Lunaris implements the classical spherical-coordinate gradient of the truncated geopotential (radial and meridional basis vectors, dP/dphi from LUNARIS-ALG-SH-001), which is physically the exact gradient of the same expansion.
- **Mathematical contract**:
  - Inputs: body-fixed position (m), reference radius, mu, and fully-normalized C_nm/S_nm coefficient blocks
  - Outputs: gravitational acceleration in the body-fixed frame (m/s^2)
  - Exactness: exact_gradient_of_truncated_field
  - Preserves: curl-free (gradient of a scalar potential)
  - Preserves: degree-0 term reduces to the point-mass monopole
- **Implementing symbols**:
  - `src/lunaris/physics/spherical_harmonics.py` -- `_compute_sh_acceleration_serial` (numba_implementation)
  - `src/lunaris/physics/spherical_harmonics.py` -- `sh_accel_fixed_numba` (numba_implementation)
  - `src/lunaris/physics/spherical_harmonics.py` -- `GravityModel` (api_entry_point)
  - `src/lunaris/physics/torch_spherical_harmonics.py` -- `TorchSHGravityEvaluator.acceleration` (torch_implementation)
  - `src/lunaris/core/batch_propagator.py` -- `_sh_accel_cuda` (cuda_implementation)
- **Lunaris modifications**:
  - Kahan-compensated accumulation in the serial kernel (see LUNARIS-ALG-SUM-001)
  - optional parallel fastmath kernel above a degree threshold
  - pole-safe evaluation via the stable-m limit (see LUNARIS-HEUR-SH-001)
- **Assumptions**:
  - position expressed in the body-fixed frame of the coefficient set
  - coefficients in the 4-pi normalization without Condon-Shortley phase
- **Limitations**:
  - truncation error grows if the degree is below the field's spectral content
- **Validated by**:
  - `tests/test_spherical_harmonics.py`
  - `tests/test_gravity_propagation_parity.py`
  - `tests/test_sh_frame_invariants.py`
  - `tests/test_independent_sh_validation.py`
- **See also**: [`LUNARIS-ALG-SH-001`](#lunarisalgsh001), [`LUNARIS-ALG-SUM-001`](#lunarisalgsum001), [`LUNARIS-HEUR-SH-001`](#lunarisheursh001), [`LUNARIS-HEUR-SH-002`](#lunarisheursh002)

<a id="lunarisheursh001"></a>
### LUNARIS-HEUR-SH-001 -- Pole-stable spherical-harmonic order truncation (stable-m limit)

- **Slug**: `pole_stable_order_truncation`
- **Category**: heuristic | **Domain**: SH | **Status**: active
- **Classification**: heuristic
- **Verification**: unverifiable | **Scientific status**: implemented_and_tested
- **Primary reference**: Lunaris-specific; no external primary reference.
- **Mathematical contract**:
  - Inputs: cos(geocentric latitude) and maximum degree
  - Outputs: highest order m for which cos(phi)^m stays above the float64 floor
  - Exactness: conservative_truncation
  - Preserves: skips only orders whose sectoral terms have underflowed to zero
- **Implementing symbols**:
  - `src/lunaris/physics/spherical_harmonics.py` -- `_compute_stable_m_limit` (numba_implementation)
- **Lunaris modifications**:
  - Lunaris-specific policy m < LOG_UNDERFLOW_LIMIT / ln|cos phi|, clamped to [0, max_degree], with exact-pole and near-equator special cases.
- **Assumptions**:
  - underflowed high-order sectoral terms contribute negligibly
- **Limitations**:
  - performance/stability policy, not a change to the physical field
- **Validated by**:
  - `tests/test_spherical_harmonics.py`
  - `tests/test_sh_frame_invariants.py`
- **See also**: [`LUNARIS-ALG-SH-002`](#lunarisalgsh002)
- **Notes**: Project-specific heuristic; no external primary source. Justified by the float64 underflow floor rather than a published algorithm.

<a id="lunarisheursh002"></a>
### LUNARIS-HEUR-SH-002 -- Degree-switched spherical-harmonic evaluation with a cubic-Hermite (smoothstep) altitude blend

- **Slug**: `degree_switched_smoothstep_altitude_blend`
- **Category**: heuristic | **Domain**: SH | **Status**: active
- **Classification**: heuristic
- **Verification**: unverifiable | **Scientific status**: implemented_and_tested
- **Primary reference**: Lunaris-specific; no external primary reference.
- **Mathematical contract**:
  - Inputs: position, two evaluation degrees, and a blend altitude band
  - Outputs: acceleration that transitions C1-continuously between a low and a high evaluation degree across the band
  - Exactness: continuous_policy_blend
  - Preserves: C1 continuity of the blended acceleration across the altitude band
- **Implementing symbols**:
  - `src/lunaris/physics/spherical_harmonics.py` -- `sh_accel_adaptive_blend_numba` (numba_implementation)
  - `src/lunaris/physics/spherical_harmonics.py` -- `_apply_smoothstep` (numba_implementation)
- **Lunaris modifications**:
  - The blend weight uses the standard cubic-Hermite smoothstep 3t^2-2t^3; the degree schedule and altitude band are a Lunaris performance policy.
- **Assumptions**:
  - the high-degree field is only needed at low altitude
- **Limitations**:
  - the blended result is a policy, not the exact field at intermediate degrees
- **Validated by**:
  - `tests/test_spherical_harmonics.py`
- **See also**: [`LUNARIS-ALG-SH-002`](#lunarisalgsh002)
- **Notes**: The smoothstep polynomial 3t^2-2t^3 is a standard cubic-Hermite S-curve; the degree-switching schedule itself is a Lunaris-specific heuristic.

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
- **See also**: [`LUNARIS-ALG-SH-002`](#lunarisalgsh002)
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
