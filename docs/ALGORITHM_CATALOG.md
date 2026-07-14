<!-- GENERATED FILE - do not edit by hand.
     Edit docs/algorithms/algorithm_registry.yaml and run
     `python tools/algorithm_registry.py generate`. -->

# Lunaris Algorithm Catalogue

Human-readable view of the algorithm-traceability registry. The source of truth is [`docs/algorithms/algorithm_registry.yaml`](algorithms/algorithm_registry.yaml); this file is generated. See [`docs/ALGORITHM_TRACEABILITY_POLICY.md`](ALGORITHM_TRACEABILITY_POLICY.md) for the naming, citation and classification policy.

**18 entries.**

Implementation class: adaptation (1), delegated_library (4), exact (8), exact_reformulation (1), heuristic (3), standard_implementation (1)

Verification status: identifier_verified_content_pending (2), unverifiable (3), verified_primary_source (13)

## Index

| ID | Method | Class | Verification |
| --- | --- | --- | --- |
| [`LUNARIS-ALG-INT-001`](#lunarisalgint001) | Dormand-Prince 8(5,3) explicit Runge-Kutta method as delegated to SciPy DOP853 | delegated_library | verified_primary_source |
| [`LUNARIS-ALG-INT-002`](#lunarisalgint002) | Classical fourth-order Runge-Kutta method (RK4) | exact | verified_primary_source |
| [`LUNARIS-ALG-INT-003`](#lunarisalgint003) | Classical fourth-order Runge-Kutta-Nystrom method (RKN4) | exact | identifier_verified_content_pending |
| [`LUNARIS-ALG-INT-004`](#lunarisalgint004) | Gragg-Bulirsch-Stoer extrapolation with the modified midpoint rule (exposed as the RK8 token) | exact | verified_primary_source |
| [`LUNARIS-ALG-INT-005`](#lunarisalgint005) | Velocity Verlet (Stormer-Verlet) symplectic method | exact | verified_primary_source |
| [`LUNARIS-ALG-INT-006`](#lunarisalgint006) | Recursive triple-jump symmetric composition of velocity Verlet (Y4/Y6/Y8) | adaptation | verified_primary_source |
| [`LUNARIS-ALG-INT-007`](#lunarisalgint007) | Position-Extended Forest-Ruth-Like (PEFRL) fourth-order symplectic integrator | exact | verified_primary_source |
| [`LUNARIS-ALG-INT-008`](#lunarisalgint008) | Dormand-Prince 5(4) embedded Runge-Kutta method as delegated to SciPy RK45 | delegated_library | verified_primary_source |
| [`LUNARIS-ALG-INT-009`](#lunarisalgint009) | Bogacki-Shampine 3(2) embedded Runge-Kutta method as delegated to SciPy RK23 | delegated_library | verified_primary_source |
| [`LUNARIS-ALG-INT-010`](#lunarisalgint010) | Implicit stiff ODE solvers (Radau IIA, BDF, LSODA) as delegated to SciPy | delegated_library | verified_primary_source |
| [`LUNARIS-ALG-SH-001`](#lunarisalgsh001) | Standard forward-column recursion of fully-normalized associated Legendre functions | exact | verified_primary_source |
| [`LUNARIS-ALG-SH-002`](#lunarisalgsh002) | Spherical-harmonic gravitational acceleration via the classical spherical-coordinate geopotential gradient | standard_implementation | verified_primary_source |
| [`LUNARIS-ALG-SUM-001`](#lunarisalgsum001) | Kahan compensated summation | exact | verified_primary_source |
| [`LUNARIS-ALG-TB-001`](#lunarisalgtb001) | Battin F(q) cancellation-resistant differential third-body formulation | exact_reformulation | identifier_verified_content_pending |
| [`LUNARIS-ALG-TB-002`](#lunarisalgtb002) | Newtonian point-mass (monopole) central-body gravitational acceleration | exact | verified_primary_source |
| [`LUNARIS-HEUR-EVT-001`](#lunarisheurevt001) | In-step event-time localization by bisection with a false-position final correction | heuristic | unverifiable |
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

<a id="lunarisalgint002"></a>
### LUNARIS-ALG-INT-002 -- Classical fourth-order Runge-Kutta method (RK4)

- **Slug**: `classical_rk4_fixed_step`
- **Category**: integrator | **Domain**: INT | **Status**: active
- **Classification**: exact
- **Verification**: verified_primary_source | **Scientific status**: implemented_and_tested
- **Primary reference**: `HairerNorsettWanner1993ODEsI` -- Hairer, 1993. "Solving Ordinary Differential Equations I: Nonstiff Problems" (edition Second Revised; chapter II (Runge-Kutta and Extrapolation Methods); section II.1 (the first Runge-Kutta methods)) [ISBN: 978-3-540-56670-0]
- **Verification notes**: The implemented tableau (k1..k4 with the standard 1/2, 1/2, 1 nodes and the 1/6,1/3,1/3,1/6 weights) is the classical fourth-order Runge-Kutta method presented in Hairer, Norsett & Wanner (1993), Chapter II. Book ISBN 978-3-540-56670-0 verified via Springer; exact page pending physical copy but the tableau identity is unambiguous.
- **Mathematical contract**:
  - Inputs: RHS f(t, y), state y, step h
  - Outputs: state advanced by one step of size h
  - Exactness: fourth_order_accurate
  - Preserves: 4th-order local/global accuracy for smooth RHS
- **Implementing symbols**:
  - `src/lunaris/core/propagation/integrators/rk.py` -- `_rk4_step_full` (cpu_implementation)
- **Assumptions**:
  - first-order ODE form y' = f(t, y)
- **Limitations**:
  - not symplectic; fixed step has no error control
- **Validated by**:
  - `tests/test_integrators.py`
  - `tests/test_batched_fixed_step.py`

<a id="lunarisalgint003"></a>
### LUNARIS-ALG-INT-003 -- Classical fourth-order Runge-Kutta-Nystrom method (RKN4)

- **Slug**: `classical_rkn4_fixed_step`
- **Category**: integrator | **Domain**: INT | **Status**: active
- **Classification**: exact
- **Verification**: identifier_verified_content_pending | **Scientific status**: implemented_and_tested
- **Primary reference**: `HairerNorsettWanner1993ODEsI` -- Hairer, 1993. "Solving Ordinary Differential Equations I: Nonstiff Problems" (edition Second Revised; chapter II (Runge-Kutta and Extrapolation Methods); section Runge-Kutta-Nystrom methods for second-order ODEs) [ISBN: 978-3-540-56670-0]
- **Verification notes**: Runge-Kutta-Nystrom methods for y'' = f(t, y) are treated in Hairer, Norsett & Wanner (1993), Chapter II. The implemented step uses r_mid with 0.125 h^2 k1, r_end with 0.5 h^2 k2, position update r + h v + (h^2/6)(k1+2 k2) and velocity update v + (h/6)(k1+4 k2+k4), reusing k2=k3. Book ISBN verified; PENDING confirmation that these coefficients match the specific classical RKN4 tableau in the cited section (physical copy needed) rather than a different 4th-order RKN pair.
- **Mathematical contract**:
  - Inputs: acceleration a(t, y), packed position/velocity state, step h
  - Outputs: position/velocity advanced by one step
  - Exactness: fourth_order_accurate
  - Preserves: direct second-order integration without reduction overhead
- **Implementing symbols**:
  - `src/lunaris/core/propagation/integrators/rk.py` -- `_rkn4_step` (cpu_implementation)
- **Lunaris modifications**:
  - single acceleration evaluation reused for k2 and k3
- **Assumptions**:
  - dynamics expressible as y'' = a(t, y) (force independent of velocity)
- **Limitations**:
  - inapplicable to velocity-dependent forces (e.g. relativistic drag terms)
- **Validated by**:
  - `tests/test_integrators.py`
  - `tests/test_batched_fixed_step.py`

<a id="lunarisalgint004"></a>
### LUNARIS-ALG-INT-004 -- Gragg-Bulirsch-Stoer extrapolation with the modified midpoint rule (exposed as the RK8 token)

- **Slug**: `gragg_bulirsch_stoer_extrapolation`
- **Category**: integrator | **Domain**: INT | **Status**: active
- **Classification**: exact
- **Verification**: verified_primary_source | **Scientific status**: implemented_and_tested
- **Primary reference**: `HairerNorsettWanner1993ODEsI` -- Hairer, 1993. "Solving Ordinary Differential Equations I: Nonstiff Problems" (edition Second Revised; chapter II (Runge-Kutta and Extrapolation Methods); section II.9 (extrapolation methods; Gragg modified midpoint)) [ISBN: 978-3-540-56670-0]
- **Verification notes**: The code applies Gragg's symmetric modified-midpoint rule over the sub-step sequence (2,4,6,8) and combines the results with the polynomial (Bulirsch-Stoer / Deuflhard) extrapolation recurrence T[i] += (T[i]-T[i-1])/((n_i/n_{i-k})^2 - 1). This is the Gragg-Bulirsch-Stoer extrapolation method (Hairer, Norsett & Wanner 1993, Chapter II, extrapolation methods), NOT an 8th-order Runge-Kutta tableau. Book ISBN verified.
- **Mathematical contract**:
  - Inputs: RHS f(t, y), state y, macro-step h
  - Outputs: high-order extrapolated state after one macro-step
  - Exactness: high_order_extrapolation
  - Preserves: even-order error expansion of the symmetric midpoint rule
  - Preserves: high effective order via extrapolation to zero sub-step
- **Implementing symbols**:
  - `src/lunaris/core/propagation/integrators/rk.py` -- `_rk8_step_full` (cpu_implementation)
  - `src/lunaris/core/propagation/integrators/rk.py` -- `_modified_midpoint` (cpu_implementation)
- **Lunaris modifications**:
  - fixed (2,4,6,8) sub-step sequence rather than an adaptive tableau
- **Assumptions**:
  - smooth RHS (extrapolation assumes an asymptotic error expansion)
- **Limitations**:
  - NAMING: the accepted token is RK8, but this is a Gragg-Bulirsch-Stoer extrapolation method, not a Runge-Kutta order-8 formula. See the traceability findings note.
- **Validated by**:
  - `tests/test_integrators.py`
  - `tests/test_batched_fixed_step.py`
- **Notes**: Naming finding: the RK8 alias is misleading; the implemented algorithm is Gragg-Bulirsch-Stoer extrapolation. The behaviour is correct and this entry does not change it; only the documented identity is corrected.

<a id="lunarisalgint005"></a>
### LUNARIS-ALG-INT-005 -- Velocity Verlet (Stormer-Verlet) symplectic method

- **Slug**: `velocity_verlet_stormer_verlet`
- **Category**: integrator | **Domain**: INT | **Status**: active
- **Classification**: exact
- **Verification**: verified_primary_source | **Scientific status**: implemented_and_tested
- **Primary reference**: `HairerLubichWanner2006Geometric` -- Hairer, 2006. "Geometric Numerical Integration: Structure-Preserving Algorithms for Ordinary Differential Equations" (edition Second; chapter I (Examples and Numerical Experiments); section The Stormer-Verlet method) [ISBN: 978-3-540-30663-4]
- **Verification notes**: The half-kick / drift / half-kick sequence (v_half = v + 0.5 h a0; r1 = r + h v_half; v1 = v_half + 0.5 h a1) is the velocity form of the Stormer-Verlet method treated in Hairer, Lubich & Wanner, "Geometric Numerical Integration", 2nd ed. ISBN 978-3-540-30663-4 verified via Springer/Amazon. Historical origins: Stormer, Verlet (1967, position form) and Swope et al. (1982, velocity form).
- **Mathematical contract**:
  - Inputs: acceleration a(t, y), packed state, step h
  - Outputs: symplectically advanced position/velocity
  - Exactness: second_order_symplectic
  - Preserves: symplectic and time-reversible
  - Preserves: bounded energy error for Hamiltonian systems
- **Implementing symbols**:
  - `src/lunaris/core/propagation/integrators/symplectic.py` -- `_vv_step` (cpu_implementation)
- **Assumptions**:
  - separable Hamiltonian with velocity-independent force
- **Limitations**:
  - only second order; use a composition method for higher order
- **Validated by**:
  - `tests/test_integrators.py`
  - `tests/test_symplectic_strict_guard.py`
- **See also**: [`LUNARIS-ALG-INT-006`](#lunarisalgint006), [`LUNARIS-ALG-INT-007`](#lunarisalgint007)
- **Notes**: Accepted aliases VV, VERLET, STORMER_VERLET and LEAPFROG all resolve to this method.

<a id="lunarisalgint006"></a>
### LUNARIS-ALG-INT-006 -- Recursive triple-jump symmetric composition of velocity Verlet (Y4/Y6/Y8)

- **Slug**: `recursive_triple_jump_symplectic_composition`
- **Category**: integrator | **Domain**: INT | **Status**: active
- **Classification**: adaptation
- **Verification**: verified_primary_source | **Scientific status**: implemented_and_tested
- **Primary reference**: `Yoshida1990Symplectic` -- Yoshida, 1990. "Construction of Higher Order Symplectic Integrators" (section Triple-jump composition construction; pages 262-268) [DOI: 10.1016/0375-9601(90)90092-3]
- **Verification notes**: H. Yoshida, "Construction of higher order symplectic integrators", Physics Letters A 150(5-7):262-268, 1990. DOI 10.1016/0375-9601(90) 90092-3 verified via ScienceDirect/ADS. The code composes velocity-Verlet sub-steps with the triple-jump weights x1 = 1/(2 - 2^{1/(2k+1)}), x0 = -2^{1/(2k+1)}/(2 - 2^{1/(2k+1)}). Y4 (k=1) reproduces the Forest-Ruth / Yoshida 4th-order integrator. NAMING: Y6/Y8 use this recursive triple-jump construction, NOT Yoshida's optimized minimal-stage 6th/8th-order coefficient sets (which the same paper obtains by solving the order conditions numerically).
- **Mathematical contract**:
  - Inputs: acceleration a(t, y), packed state, step h, composition order
  - Outputs: symplectically advanced state of order 2k
  - Exactness: order_2k_symplectic
  - Preserves: symplectic and time-reversible at every composition level
- **Implementing symbols**:
  - `src/lunaris/core/propagation/integrators/symplectic.py` -- `_composition_weights` (cpu_implementation)
  - `src/lunaris/core/propagation/integrators/symplectic.py` -- `_composed_step` (cpu_implementation)
  - `src/lunaris/core/propagation/integrators/symplectic.py` -- `_y6_step` (cpu_implementation)
- **Lunaris modifications**:
  - recursive triple-jump rather than optimized minimal-stage coefficients
- **Assumptions**:
  - separable Hamiltonian with velocity-independent force
- **Limitations**:
  - Y6/Y8 have larger error constants than Yoshida's optimized solution sets because the recursive triple jump uses more sub-steps.
- **Validated by**:
  - `tests/test_integrators.py`
  - `tests/test_symplectic_strict_guard.py`
- **See also**: [`LUNARIS-ALG-INT-005`](#lunarisalgint005)
- **Notes**: Naming clarification: only Y4 is literally Yoshida's coefficient set; Y6/Y8 are the recursive triple-jump (Suzuki fractal) construction. The methods are valid symplectic integrators of the stated order.

<a id="lunarisalgint007"></a>
### LUNARIS-ALG-INT-007 -- Position-Extended Forest-Ruth-Like (PEFRL) fourth-order symplectic integrator

- **Slug**: `pefrl_position_extended_forest_ruth`
- **Category**: integrator | **Domain**: INT | **Status**: active
- **Classification**: exact
- **Verification**: verified_primary_source | **Scientific status**: implemented_and_tested
- **Primary reference**: `Omelyan2002PEFRL` -- Omelyan, 2002. "Optimized {Forest-Ruth-} and {Suzuki-}like Algorithms for Integration of Motion in Many-Body Systems" (section PEFRL coefficients; pages 188-202) [DOI: 10.1016/S0010-4655(02)00451-4]
- **Verification notes**: I. P. Omelyan, I. M. Mryglod & R. Folk, "Optimized Forest-Ruth- and Suzuki-like algorithms for integration of motion in many-body systems", Computer Physics Communications 146(2):188-202, 2002. DOI 10.1016/S0010-4655(02)00451-4 verified. The implemented constants xi=0.1786178958448091, lambda=-0.2123418310626054, chi=-0.06626458266981849 are exactly the published PEFRL coefficients.
- **Mathematical contract**:
  - Inputs: acceleration a(t, y), packed state, step h
  - Outputs: symplectically advanced state
  - Exactness: fourth_order_symplectic
  - Preserves: symplectic with a markedly smaller error constant than Yoshida4
  - Preserves: drift fractions telescope to 1 and velocity kicks sum to 1
- **Implementing symbols**:
  - `src/lunaris/core/propagation/integrators/symplectic.py` -- `_pefrl_step` (cpu_implementation)
- **Assumptions**:
  - separable Hamiltonian with velocity-independent force
- **Limitations**:
  - four force evaluations per step
- **Validated by**:
  - `tests/test_integrators.py`
  - `tests/test_symplectic_strict_guard.py`
- **See also**: [`LUNARIS-ALG-INT-005`](#lunarisalgint005)

<a id="lunarisalgint008"></a>
### LUNARIS-ALG-INT-008 -- Dormand-Prince 5(4) embedded Runge-Kutta method as delegated to SciPy RK45

- **Slug**: `rk45_dormand_prince_delegation`
- **Category**: integrator | **Domain**: INT | **Status**: active
- **Classification**: delegated_library
- **Verification**: verified_primary_source | **Scientific status**: implemented_and_tested
- **Primary reference**: `DormandPrince1980Embedded` -- Dormand, 1980. "A Family of Embedded {Runge-Kutta} Formulae" (section RK5(4) embedded pair; pages 19-26) [DOI: 10.1016/0771-050X(80)90013-3]
- **Verification notes**: SciPy's RK45 is the Dormand-Prince 5(4) embedded pair (Dormand & Prince, J. Comput. Appl. Math. 6(1):19-26, 1980; DOI 10.1016/0771-050X(80) 90013-3 verified). Lunaris selects the method through the SciPy delegation surface; SciPy is the actual implementation (Virtanen et al. 2020).
- **Mathematical contract**:
  - Inputs: RHS f(t, y), initial state, span, tolerances
  - Outputs: adaptively stepped trajectory on the output grid
  - Exactness: adaptive_error_controlled_approximation
  - Preserves: embedded 4th-order error estimate drives step control
- **Implementing symbols**:
  - `src/lunaris/core/propagation/integrators/scipy.py` -- `_resolve_scipy_method` (delegation_wrapper)
  - `src/lunaris/common/integrator_methods.py` -- `(module)` (config_surface)
- **Lunaris modifications**:
  - method selection and tolerance policy only
- **Assumptions**:
  - non-stiff dynamics
- **Limitations**:
  - lower order than DOP853 for tight tolerances
- **Validated by**:
  - `tests/test_integrators.py`
  - `tests/test_solver_policy_and_telemetry.py`
- **See also**: [`LUNARIS-ALG-INT-001`](#lunarisalgint001)

<a id="lunarisalgint009"></a>
### LUNARIS-ALG-INT-009 -- Bogacki-Shampine 3(2) embedded Runge-Kutta method as delegated to SciPy RK23

- **Slug**: `rk23_bogacki_shampine_delegation`
- **Category**: integrator | **Domain**: INT | **Status**: active
- **Classification**: delegated_library
- **Verification**: verified_primary_source | **Scientific status**: implemented_and_tested
- **Primary reference**: `BogackiShampine1989Pair` -- Bogacki, 1989. "A 3(2) Pair of {Runge-Kutta} Formulas" (section 3(2) FSAL pair; pages 321-325) [DOI: 10.1016/0893-9659(89)90079-7]
- **Verification notes**: SciPy's RK23 is the Bogacki-Shampine 3(2) FSAL pair (Bogacki & Shampine, Applied Mathematics Letters 2(4):321-325, 1989; DOI 10.1016/0893-9659(89)90079-7 verified). SciPy is the delegated implementation (Virtanen et al. 2020).
- **Mathematical contract**:
  - Inputs: RHS f(t, y), initial state, span, tolerances
  - Outputs: adaptively stepped trajectory on the output grid
  - Exactness: adaptive_error_controlled_approximation
  - Preserves: embedded 2nd-order error estimate; FSAL reuse
- **Implementing symbols**:
  - `src/lunaris/core/propagation/integrators/scipy.py` -- `_resolve_scipy_method` (delegation_wrapper)
  - `src/lunaris/common/integrator_methods.py` -- `(module)` (config_surface)
- **Lunaris modifications**:
  - method selection and tolerance policy only
- **Assumptions**:
  - non-stiff dynamics; loose tolerances
- **Limitations**:
  - low order; inaccurate for tight tolerances
- **Validated by**:
  - `tests/test_integrators.py`
  - `tests/test_solver_policy_and_telemetry.py`
- **See also**: [`LUNARIS-ALG-INT-001`](#lunarisalgint001)

<a id="lunarisalgint010"></a>
### LUNARIS-ALG-INT-010 -- Implicit stiff ODE solvers (Radau IIA, BDF, LSODA) as delegated to SciPy

- **Slug**: `scipy_stiff_solver_delegation`
- **Category**: integrator | **Domain**: INT | **Status**: active
- **Classification**: delegated_library
- **Verification**: verified_primary_source | **Scientific status**: implemented_and_tested
- **Primary reference**: `HairerWanner1996ODEsII` -- Hairer, 1996. "Solving Ordinary Differential Equations II: Stiff and Differential-Algebraic Problems" (edition Second Revised; chapter IV-V (stiff Runge-Kutta and multistep methods); section Radau IIA and BDF methods) [DOI: 10.1007/978-3-642-05221-7]
- **Verification notes**: SciPy's Radau (order-5 Radau IIA) and BDF solvers implement methods from Hairer & Wanner, "Solving Ordinary Differential Equations II", 2nd ed. (ISBN 978-3-540-60452-5, DOI 10.1007/978-3-642-05221-7 verified). LSODA wraps the ODEPACK automatic stiff/non-stiff switcher (Hindmarsh; Petzold 1983). All are reached through the SciPy delegation surface (Virtanen et al. 2020); these stiff options are available but rarely used for gravity-only propagation.
- **Mathematical contract**:
  - Inputs: RHS f(t, y), initial state, span, tolerances (Jacobian optional)
  - Outputs: adaptively stepped trajectory on the output grid
  - Exactness: implicit_adaptive_approximation
  - Preserves: A-stability suitable for stiff systems
- **Implementing symbols**:
  - `src/lunaris/core/propagation/integrators/scipy.py` -- `_resolve_scipy_method` (delegation_wrapper)
  - `src/lunaris/common/integrator_methods.py` -- `(module)` (config_surface)
- **Lunaris modifications**:
  - method selection and tolerance policy only
- **Assumptions**:
  - problem may be stiff
- **Limitations**:
  - LSODA-specific ODEPACK origin (Petzold 1983) recorded in prose only
- **Validated by**:
  - `tests/test_integrators.py`
  - `tests/test_solver_policy_and_telemetry.py`
- **See also**: [`LUNARIS-ALG-INT-001`](#lunarisalgint001)
- **Notes**: Grouped delegation entry; SciPy is the verified delegated implementation and the method origins are cited per solver.

## Event handling (EVT)

<a id="lunarisheurevt001"></a>
### LUNARIS-HEUR-EVT-001 -- In-step event-time localization by bisection with a false-position final correction

- **Slug**: `in_step_event_root_localization`
- **Category**: heuristic | **Domain**: EVT | **Status**: active
- **Classification**: heuristic
- **Verification**: unverifiable | **Scientific status**: implemented_and_tested
- **Primary reference**: Lunaris-specific; no external primary reference.
- **Mathematical contract**:
  - Inputs: a single fixed-step interval with a sign change in the event function g, plus a sub-step re-integrator
  - Outputs: refined event time and state inside the interval
  - Exactness: bracketed_root_localization
  - Preserves: keeps the bracket around the sign change
  - Preserves: does not perturb the main integrator's state advancement
- **Implementing symbols**:
  - `src/lunaris/core/propagation/events.py` -- `_refine_event_time_bisect` (cpu_implementation)
  - `src/lunaris/core/propagation/events.py` -- `_event_crossed` (cpu_implementation)
- **Lunaris modifications**:
  - Sub-steps are re-integrated from the interval start so symplectic stepping is unchanged; only the reported event time/state is refined.
  - fail-closed on non-finite bracket or callback values
- **Assumptions**:
  - exactly one sign change of g within the interval
- **Limitations**:
  - Policy, not a library root-finder: the underlying method is standard bisection with a false-position (regula-falsi) final linear-in-g correction; the in-step re-integration wrapper is Lunaris-specific.
- **Validated by**:
  - `tests/test_events.py`
  - `tests/test_event_outcome_contract.py`
- **Notes**: Terminal-event outcome selection follows SciPy solve_ivp parity semantics (earliest terminal crossing). The root find itself is elementary bisection + false position, hence no external primary source is claimed.
