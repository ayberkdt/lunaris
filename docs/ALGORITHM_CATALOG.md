<!-- GENERATED FILE - do not edit by hand.
     Edit docs/algorithms/algorithm_registry.yaml and run
     `python tools/algorithm_registry.py generate`. -->

# Lunaris Algorithm Catalogue

Human-readable view of the algorithm-traceability registry. The source of truth is [`docs/algorithms/algorithm_registry.yaml`](algorithms/algorithm_registry.yaml); this file is generated. See [`docs/ALGORITHM_TRACEABILITY_POLICY.md`](ALGORITHM_TRACEABILITY_POLICY.md) for the naming, citation and classification policy.

**51 entries.**

Implementation class: adaptation (10), delegated_library (8), exact (16), exact_reformulation (1), heuristic (7), standard_implementation (9)

Verification status: identifier_verified_content_pending (2), unverifiable (7), verified_primary_source (41), verified_secondary_source (1)

## Index

| ID | Method | Class | Verification |
| --- | --- | --- | --- |
| [`LUNARIS-ALG-EPH-001`](#lunarisalgeph001) | SPICE-built fixed-grid ephemeris and lunar-orientation tables | delegated_library | verified_primary_source |
| [`LUNARIS-ALG-FRM-001`](#lunarisalgfrm001) | Spherical linear interpolation of unit quaternions (SLERP) | exact | verified_primary_source |
| [`LUNARIS-ALG-FRM-002`](#lunarisalgfrm002) | Scalar-first (Hamilton) unit-quaternion active vector rotation | standard_implementation | verified_primary_source |
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
| [`LUNARIS-ALG-INTP-001`](#lunarisalgintp001) | Clamped uniform Catmull-Rom cubic interpolation of vector time series | exact | verified_primary_source |
| [`LUNARIS-ALG-INTP-002`](#lunarisalgintp002) | Cubic Hermite interpolation of ephemeris position and velocity states | exact | verified_primary_source |
| [`LUNARIS-ALG-ML-001`](#lunarisalgml001) | Sinusoidal Representation Network (SIREN) | exact | verified_primary_source |
| [`LUNARIS-ALG-ML-002`](#lunarisalgml002) | Random Fourier feature input encoding | exact | verified_primary_source |
| [`LUNARIS-ALG-OE-001`](#lunarisalgoe001) | Classical orbital-element conversions (state vector to/from Keplerian elements) | standard_implementation | verified_primary_source |
| [`LUNARIS-ALG-OPT-001`](#lunarisalgopt001) | AdamW (decoupled weight decay) optimizer as delegated to PyTorch | delegated_library | verified_primary_source |
| [`LUNARIS-ALG-OPT-002`](#lunarisalgopt002) | GradNorm adaptive multi-task loss balancing | adaptation | verified_primary_source |
| [`LUNARIS-ALG-OPT-003`](#lunarisalgopt003) | Sobolev training (derivative-supervision loss) | adaptation | verified_primary_source |
| [`LUNARIS-ALG-PHZ-001`](#lunarisalgphz001) | Tangential Gauss variational equation for along-track phase drift with RIC error decomposition | standard_implementation | verified_primary_source |
| [`LUNARIS-ALG-SAMP-001`](#lunarisalgsamp001) | Scrambled Sobol low-discrepancy sequence sampling | delegated_library | verified_primary_source |
| [`LUNARIS-ALG-SAMP-002`](#lunarisalgsamp002) | Latin hypercube sampling | delegated_library | verified_primary_source |
| [`LUNARIS-ALG-SH-001`](#lunarisalgsh001) | Standard forward-column recursion of fully-normalized associated Legendre functions | exact | verified_primary_source |
| [`LUNARIS-ALG-SH-002`](#lunarisalgsh002) | Spherical-harmonic gravitational acceleration via the classical spherical-coordinate geopotential gradient | standard_implementation | verified_primary_source |
| [`LUNARIS-ALG-SUM-001`](#lunarisalgsum001) | Kahan compensated summation | exact | verified_primary_source |
| [`LUNARIS-ALG-TB-001`](#lunarisalgtb001) | Battin F(q) cancellation-resistant differential third-body formulation | exact_reformulation | identifier_verified_content_pending |
| [`LUNARIS-ALG-TB-002`](#lunarisalgtb002) | Newtonian point-mass (monopole) central-body gravitational acceleration | exact | verified_primary_source |
| [`LUNARIS-ALG-UQ-001`](#lunarisalguq001) | Empirical state covariance and eigendecomposed three-sigma position ellipsoid | standard_implementation | verified_secondary_source |
| [`LUNARIS-ALG-UQ-002`](#lunarisalguq002) | Wilson score confidence interval for impact proportion | exact | verified_primary_source |
| [`LUNARIS-DATA-CST-001`](#lunarisdatacst001) | CODATA 2018 recommended fundamental physical constants | standard_implementation | verified_primary_source |
| [`LUNARIS-DATA-EPH-001`](#lunarisdataeph001) | JPL DE440 planetary and lunar ephemeris | standard_implementation | verified_primary_source |
| [`LUNARIS-DATA-GRAV-001`](#lunarisdatagrav001) | GRAIL GL1800F lunar spherical-harmonic gravity field (JGGRX_1800F) | standard_implementation | verified_primary_source |
| [`LUNARIS-HEUR-EVT-001`](#lunarisheurevt001) | In-step event-time localization by bisection with a false-position final correction | heuristic | unverifiable |
| [`LUNARIS-HEUR-FRZ-001`](#lunarisheurfrz001) | Lunaris thresholded frozen-orbit candidate screening and validation gate | heuristic | unverifiable |
| [`LUNARIS-HEUR-IMP-001`](#lunarisheurimp001) | Outer-sphere rejection and terrain-height bisection for batched impact localization | heuristic | unverifiable |
| [`LUNARIS-HEUR-INTP-001`](#lunarisheurintp001) | Latitude-clamped, longitude-periodic planetary grid sampling with missing-value fallback | heuristic | unverifiable |
| [`LUNARIS-HEUR-ML-001`](#lunarisheurml001) | Multi-band SIREN variants and physics-informed input encodings (Lunaris-specific) | heuristic | unverifiable |
| [`LUNARIS-HEUR-SH-001`](#lunarisheursh001) | Pole-stable spherical-harmonic order truncation (stable-m limit) | heuristic | unverifiable |
| [`LUNARIS-HEUR-SH-002`](#lunarisheursh002) | Degree-switched spherical-harmonic evaluation with a cubic-Hermite (smoothstep) altitude blend | heuristic | unverifiable |
| [`LUNARIS-HEUR-SH-003`](#lunarisheursh003) | Spectrum-weighted spherical-harmonic truncation-degree recommendation | adaptation | verified_primary_source |
| [`LUNARIS-MODEL-J2E-001`](#lunarismodelj2e001) | Differential Earth-J2 oblateness perturbation in a Moon-centred frame | adaptation | verified_primary_source |
| [`LUNARIS-MODEL-RAD-001`](#lunarismodelrad001) | Cannonball solar radiation pressure with a dual-cone (conical) umbra/penumbra shadow | adaptation | verified_primary_source |
| [`LUNARIS-MODEL-RAD-002`](#lunarismodelrad002) | Faceted lunar albedo (shortwave) radiation-pressure model | adaptation | verified_primary_source |
| [`LUNARIS-MODEL-RAD-003`](#lunarismodelrad003) | Faceted lunar thermal-infrared (longwave) radiation-pressure model | adaptation | verified_primary_source |
| [`LUNARIS-MODEL-REL-001`](#lunarismodelrel001) | Schwarzschild first post-Newtonian acceleration (central body) | exact | verified_primary_source |
| [`LUNARIS-MODEL-REL-002`](#lunarismodelrel002) | de Sitter (geodesic) precession acceleration | exact | verified_primary_source |
| [`LUNARIS-MODEL-REL-003`](#lunarismodelrel003) | External-body differential first post-Newtonian correction (Schwarzschild + de Sitter) | adaptation | verified_primary_source |
| [`LUNARIS-MODEL-TID-001`](#lunarismodeltid001) | Elastic lunar solid-body tide (degree-2 and degree-3 Love-number disturbing potential) | adaptation | verified_primary_source |
| [`LUNARIS-STD-FRM-001`](#lunarisstdfrm001) | Lunar principal-axis (PA) body-fixed frame realized by DE440 | standard_implementation | verified_primary_source |

## Third-body gravity (TB)

<a id="lunarisalgtb001"></a>
### LUNARIS-ALG-TB-001 -- Battin F(q) cancellation-resistant differential third-body formulation

- **Slug**: `battin_fq_differential_third_body`
- **Category**: numerical_algorithm | **Domain**: TB | **Status**: active
- **Classification**: exact_reformulation
- **Verification**: identifier_verified_content_pending | **Scientific status**: implemented_and_tested
- **Primary reference**: `Battin1999Astrodynamics` -- Battin, 1999. "An Introduction to the Mathematics and Methods of Astrodynamics" (edition Revised) [ISBN: 978-1-56347-342-5]
- **Verification notes**: Book metadata verified: Richard H. Battin, "An Introduction to the Mathematics and Methods of Astrodynamics", Revised Edition, AIAA Education Series, 1999, ISBN 978-1-56347-342-5. The implemented device rewrites F = (1+q)^{3/2} - 1 as q(3+3q+q^2)/(1+(1+q)^{3/2}) to avoid the subtractive cancellation of the direct differential form; this identity is elementary and was checked independently, and the F(q)/Encke device is attributed to Battin by GMAT's MathSpec and multiple modern propagators. Narrowed 2026-07-18: Gkolias & Colombo (arXiv:2104.01240, ref [25]) cite the Battin-Giorgi/Encke q-device to Battin (1999 rev. ed.) pp. 448-450, 490-494, 529-530, and Vallado (2007) Ch. 8 cites Battin (1987:450) for Encke rectification; no online source exposes the exact F(q) equation number. PENDING: exact Battin chapter/section/equation number (physical copy needed) before this may be promoted to verified_primary_source.
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
  - `src/lunaris/validation/gravity_reference/independent_field_oracle.py` -- `_associated_legendre_all` (reference_implementation)
  - `src/lunaris/validation/gravity_reference/independent_field_oracle.py` -- `normalized_alf` (reference_implementation)
  - `src/lunaris/surrogate/st_lrps/data/spatial_cloud_generator.py` -- `run_generation` (delegation_wrapper)
  - `src/lunaris/surrogate/st_lrps/data/spatial_cloud_generator.py` -- `run_suite_generation` (delegation_wrapper)
  - `src/lunaris/surrogate/st_lrps/data/spatial_cloud_generator.py` -- `_run_active_refinement` (delegation_wrapper)
  - `src/lunaris/core/batch_propagator.py` -- `_build_grav_pack` (delegation_wrapper)
- **Lunaris modifications**:
  - Geodesy convention: no Condon-Shortley phase, real-harmonic sqrt(2) scaling for m>0 (matches GRAIL/EGM/GRACE/ICGEM coefficient definitions).
  - pole-safe per-order truncation via the stable-m limit (see LUNARIS-HEUR-SH-001)
  - analytic polar-axis limit: inside the pole-safe cutoff (rho^2 < 1e-24) the transverse components are replaced by the removable-singularity m=1 limit a_x = sum_n (mu/r^2)(R/r)^n sqrt(n(n+1)(2n+1)/2) sigma_n C_n1 (a_y with S_n1; sigma_n = 1 north, (-1)^(n+1) south), derived as the theta -> 0 limit of the same truncated expansion; implemented identically in the numba and torch backends (2026-07-18)
- **Assumptions**:
  - IEEE-754 float64 arithmetic
  - coefficients supplied in the same normalization convention
- **Limitations**:
  - Condon-Shortley phase must be excluded by the coefficient provider
  - Reliable in IEEE-754 float64 only to degree ~1900 (Holmes & Featherstone 2002): above that, the sectoral seed cos(phi)^m can underflow before the recursion re-enters the oscillatory region (n >= m / cos(phi)), silently zeroing physically recovering columns; the Holmes-Featherstone global 1e-280 scaling would then be required. At the GRAIL maximum N=1800 the measured margin between the underflow cut and the oscillatory boundary is >= ~50 orders at every latitude (worst near cos(phi) ~ 0.4), verified against pyshtools at N=1800 (2026-07-14).
- **Validated by**:
  - `tests/test_spherical_harmonics.py`
  - `tests/test_independent_sh_validation.py`
  - `tests/test_sh_convention_lock.py`
  - `tests/test_sh_high_degree_stability.py`
  - `tests/test_sh_pole_axis.py`
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
  - `src/lunaris/physics/spherical_harmonics.py` -- `_axis_transverse_m1` (numba_implementation)
  - `src/lunaris/physics/spherical_harmonics.py` -- `GravityModel` (api_entry_point)
  - `src/lunaris/physics/torch_spherical_harmonics.py` -- `TorchSHGravityEvaluator.acceleration` (torch_implementation)
  - `src/lunaris/core/batch_propagator.py` -- `_sh_accel_cuda` (cuda_implementation)
  - `src/lunaris/physics/spherical_harmonics.py` -- `(module)` (numba_implementation)
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
  - `tests/test_sh_high_degree_stability.py`
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
  - The negligibility assumption holds only while the underflow cut stays below the oscillatory-region boundary m ~ N cos(phi), i.e. for N <= ~1900 in float64. Measured at N=1800 (2026-07-14): full-m and stable-m kernels agree to 3.6e-14 worst-case across latitudes, pyshtools cross-check 1e-9 class. For N > ~1900 the cut would drop physically recovering columns and Holmes-Featherstone scaling is needed instead.
- **Validated by**:
  - `tests/test_spherical_harmonics.py`
  - `tests/test_sh_frame_invariants.py`
  - `tests/test_sh_high_degree_stability.py`
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
  - Outputs: acceleration that transitions C0-continuously (piecewise smooth) between a low and a high evaluation degree across the band
  - Exactness: continuous_policy_blend
  - Preserves: C0 continuity of the blended acceleration across the altitude band
- **Implementing symbols**:
  - `src/lunaris/physics/spherical_harmonics.py` -- `sh_accel_adaptive_blend_numba` (numba_implementation)
  - `src/lunaris/physics/spherical_harmonics.py` -- `_apply_smoothstep` (numba_implementation)
- **Lunaris modifications**:
  - The blend weight uses the standard cubic-Hermite smoothstep 3t^2-2t^3; the degree schedule and altitude band are a Lunaris performance policy.
- **Assumptions**:
  - the high-degree field is only needed at low altitude
- **Limitations**:
  - the blended result is a policy, not the exact field at intermediate degrees
  - not C1 in general: the spatial derivative jumps where the discrete degree ladder switches its blend pair inside the band
  - non-conservative inside the transition band: the acceleration-level blend omits the (U_hi - U_lo) * grad(w) term of the blended-potential gradient, so the field is not curl-free there and must not back symplectic or energy-conservation studies (the propagator's symplectic guard rejects adaptive-degree gravity)
  - not routed into the production RHS (reachable only via accel_adaptive)
- **Validated by**:
  - `tests/test_spherical_harmonics.py`
- **See also**: [`LUNARIS-ALG-SH-002`](#lunarisalgsh002)
- **Notes**: The smoothstep polynomial 3t^2-2t^3 is a standard cubic-Hermite S-curve; the degree-switching schedule itself is a Lunaris-specific heuristic.

<a id="lunarisheursh003"></a>
### LUNARIS-HEUR-SH-003 -- Spectrum-weighted spherical-harmonic truncation-degree recommendation

- **Slug**: `spectrum_weighted_gravity_degree_recommendation`
- **Category**: heuristic | **Domain**: SH | **Status**: active
- **Classification**: adaptation
- **Verification**: verified_primary_source | **Scientific status**: implemented_and_tested
- **Primary reference**: `Lemoine2014GRGM900C` -- Lemoine, 2014. "{GRGM900C}: A Degree 900 Lunar Gravity Model from {GRAIL} Primary and Extended Mission Data" (section Gravity-coefficient power spectrum and high-degree power-law constraint; pages 3382-3389) [DOI: 10.1002/2014GL060027]
- **Verification notes**: Lemoine et al. (2014), DOI 10.1002/2014GL060027, was verified through NASA NTRS record 20160005754. The paper reports the measured lunar gravity power spectrum and an explicit 3.6e-4/l^2 constraint above degree 600. Lunaris follows the spectrum-weighting construction, but its exponent p=1.7 is an empirical fit to the bundled JGGRX 1800F coefficients rather than a constant asserted by that paper.
- **Mathematical contract**:
  - Inputs: altitude, body reference radius, maximum degree, a fitted degree-power exponent, and an admissible discarded RMS tail fraction
  - Outputs: smallest degree whose modeled power-law acceleration tail is below the requested fraction of total non-spherical acceleration power
  - Exactness: model_calibrated_spectrum_heuristic
  - Preserves: recommendation is bounded by the available model degree
  - Preserves: tighter tail fractions never reduce the recommended degree
  - Preserves: disabling spectrum weighting preserves the attenuation-only behavior
- **Implementing symbols**:
  - `src/lunaris/common/math_utils.py` -- `recommended_sh_degree` (cpu_implementation)
  - `src/lunaris/core/propagation/diagnostics.py` -- `build_propagation_diagnostics` (api_entry_point)
- **Lunaris modifications**:
  - p=1.7 is fitted to JGGRX 1800F rather than copied from a terrestrial Kaula rule
  - the cumulative RMS acceleration tail, not a single coefficient, controls the threshold
- **Assumptions**:
  - the active lunar coefficient spectrum is represented adequately by the fitted power law
  - the requested altitude is outside the reference radius
- **Limitations**:
  - configuration recommendation only; not a guaranteed trajectory-error bound
  - the p=1.7 default is JGGRX-specific and must not be promoted as universal lunar physics
  - surrogate representation ceilings require a separate residual-band comparison
- **Validated by**:
  - `tests/test_math_utils.py`
  - `tests/test_band_share_analysis.py`
- **See also**: [`LUNARIS-ALG-SH-001`](#lunarisalgsh001), [`LUNARIS-ALG-SH-002`](#lunarisalgsh002)
- **Notes**: Primary-source verification covers the spectrum/power-law method. The fitted exponent and acceptance threshold are explicitly project-specific.

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

## Earth oblateness (J2) (J2E)

<a id="lunarismodelj2e001"></a>
### LUNARIS-MODEL-J2E-001 -- Differential Earth-J2 oblateness perturbation in a Moon-centred frame

- **Slug**: `differential_earth_j2_oblateness`
- **Category**: physical_model | **Domain**: J2E | **Status**: active
- **Classification**: adaptation
- **Verification**: verified_primary_source | **Scientific status**: implemented_and_tested
- **Primary reference**: `MontenbruckGill2000SatelliteOrbits` -- Montenbruck, 2000. "Satellite Orbits: Models, Methods and Applications" (chapter 3 (Force Models); section 3.2 (zonal harmonics / J2)) [DOI: 10.1007/978-3-642-58351-3]
- **Verification notes**: The vector J2 acceleration a = (3/2) J2 mu R^2 / r^5 * [(5 (r.k)^2/r^2 - 1) r - 2 (r.k) k] is the standard oblateness term (Montenbruck & Gill 2000, Chapter 3; book identifiers verified as for LUNARIS-ALG-SH-002). Lunaris applies it DIFFERENTIALLY for Earth in a Moon-centred inertial frame: a_diff = a_J2(Earth->SC) - a_J2(Earth->Moon).
- **Mathematical contract**:
  - Inputs: Moon-centred spacecraft and Earth positions, Earth mu, reference radius, J2, and the Earth spin-axis unit vector
  - Outputs: differential J2 acceleration in m/s^2
  - Exactness: exact_gradient_differential_form
  - Preserves: subtracts the origin (Moon) J2 acceleration for the relative EOM
- **Implementing symbols**:
  - `src/lunaris/physics/third_body_effects.py` -- `_accel_j2_oblate_unit_k` (numba_implementation)
  - `src/lunaris/physics/third_body_effects.py` -- `accel_j2_oblate_diff_numba` (numba_implementation)
  - `src/lunaris/physics/third_body_effects.py` -- `calc_j2_oblate_diff_accel` (api_entry_point)
- **Lunaris modifications**:
  - differential (Moon-centred) formulation of the standard J2 term
  - explicit unit spin-axis input
- **Assumptions**:
  - single zonal J2 term; Earth spin axis supplied as a unit vector
- **Limitations**:
  - only the J2 zonal harmonic of Earth (no higher zonals/tesserals)
- **Validated by**:
  - `tests/test_dynamics.py`
  - `tests/test_force_gradients.py`
- **See also**: [`LUNARIS-ALG-TB-001`](#lunarisalgtb001)
- **Notes**: A small perturbation on a lunar orbiter; included for completeness of the Earth-perturbation budget.

## Solid tides (TID)

<a id="lunarismodeltid001"></a>
### LUNARIS-MODEL-TID-001 -- Elastic lunar solid-body tide (degree-2 and degree-3 Love-number disturbing potential)

- **Slug**: `lunar_solid_tide_love_degree_2_3`
- **Category**: physical_model | **Domain**: TID | **Status**: active
- **Classification**: adaptation
- **Verification**: verified_primary_source | **Scientific status**: implemented_and_tested
- **Primary reference**: `PetitLuzum2010IERSConventions` -- Petit, 2010. "{IERS} Conventions (2010)" (chapter 6 (Effects of the solid Earth tides); section 6.1 (elastic solid Earth tides, Love-number formulation)) [ISBN: 978-3-89888-989-6]
- **Verification notes**: The Love-number disturbing potential dU_l = k_l mu_j/|R_j| (R/|r|)^(l+1) (R/|R_j|)^l P_l(c) and its gradient follow the elastic solid-tide formalism of IERS Conventions (2010), TN 36, Chapter 6. Lunaris applies this Earth-tide formalism to the MOON (lunar Love numbers k2/k3), an adaptation of the terrestrial convention; instantaneous elastic response only.
- **Mathematical contract**:
  - Inputs: Moon-fixed spacecraft and tide-raising-body vectors, body mu, lunar radius, Love number, and degree (2 or 3)
  - Outputs: solid-tide acceleration (m/s^2), gradient of the positive potential
  - Exactness: exact_gradient_of_elastic_tide_potential
  - Preserves: degree-2 and degree-3 Love-number response
- **Implementing symbols**:
  - `src/lunaris/physics/solid_tides.py` -- `solid_tide_potential_degree_numba` (numba_implementation)
  - `src/lunaris/physics/solid_tides.py` -- `accel_solid_tides_numba` (numba_implementation)
  - `src/lunaris/physics/solid_tides.py` -- `calc_solid_tide_accel` (api_entry_point)
  - `src/lunaris/physics/solid_tides.py` -- `legendre_p2` (numba_implementation)
  - `src/lunaris/physics/solid_tides.py` -- `legendre_p2_derivative` (numba_implementation)
  - `src/lunaris/physics/solid_tides.py` -- `legendre_p3` (numba_implementation)
  - `src/lunaris/physics/solid_tides.py` -- `legendre_p3_derivative` (numba_implementation)
  - `src/lunaris/physics/solid_tides.py` -- `solid_tide_accel_degree_numba` (numba_implementation)
- **Lunaris modifications**:
  - terrestrial IERS Love-tide formalism applied to lunar parameters
  - dimensionless radius ratios to avoid large intermediate powers
- **Assumptions**:
  - instantaneous elastic response (no lag, dissipation, ocean/thermal tides)
- **Limitations**:
  - no anelastic/frequency-dependent Love numbers; degrees 2 and 3 only
- **Validated by**:
  - `tests/test_solid_tides.py`
- **See also**: [`LUNARIS-ALG-SH-002`](#lunarisalgsh002)
- **Notes**: Lunaris uses the Earth-tide convention with lunar Love numbers; this is an adaptation, not a lunar-specific published tide model.

## Relativistic corrections (REL)

<a id="lunarismodelrel001"></a>
### LUNARIS-MODEL-REL-001 -- Schwarzschild first post-Newtonian acceleration (central body)

- **Slug**: `schwarzschild_1pn_central`
- **Category**: physical_model | **Domain**: REL | **Status**: active
- **Classification**: exact
- **Verification**: verified_primary_source | **Scientific status**: implemented_and_tested
- **Primary reference**: `PetitLuzum2010IERSConventions` -- Petit, 2010. "{IERS} Conventions (2010)" (chapter 10 (General relativistic models for space-time coordinates and equations of motion); section 10.3; eq. 10.12 (Schwarzschild term)) [ISBN: 978-3-89888-989-6]
- **Verification notes**: IERS Conventions (2010), IERS Technical Note No. 36, Eq. 10.12. Report number and ISBN 978-3-89888-989-6 verified via iers.org/ADS. The kernel a = (mu/(c^2 r^3)) [(4 mu/r - v^2) r + 4 (r.v) v] is the Schwarzschild part of Eq. 10.12 with PPN parameters beta = gamma = 1 (Lense-Thirring and geodesic terms handled separately/omitted).
- **Mathematical contract**:
  - Inputs: position (m), velocity (m/s), central-body mu (m^3/s^2)
  - Outputs: 1PN Schwarzschild acceleration correction (m/s^2)
  - Exactness: exact_1pn_with_beta_gamma_unity
  - Preserves: reproduces the relativistic perihelion advance
- **Implementing symbols**:
  - `src/lunaris/physics/relativity_effects.py` -- `schwarzschild_components` (numba_implementation)
  - `src/lunaris/physics/relativity_effects.py` -- `calc_schwarzschild_accel` (api_entry_point)
  - `src/lunaris/physics/relativity_effects.py` -- `RelativityModel` (api_entry_point)
  - `src/lunaris/physics/relativity_effects.py` -- `(module)` (numba_implementation)
  - `src/lunaris/core/dynamics/engine.py` -- `rhs` (delegation_wrapper)
  - `src/lunaris/core/dynamics/engine.py` -- `_rhs_kernel_numba` (delegation_wrapper)
  - `src/lunaris/core/dynamics/engine.py` -- `get_acceleration_vector_breakdown` (delegation_wrapper)
  - `src/lunaris/analysis/perturbation_budget/acceleration_budget.py` -- `non_gravity_vectors` (delegation_wrapper)
- **Lunaris modifications**:
  - PPN parameters fixed to general relativity (beta = gamma = 1)
- **Assumptions**:
  - single dominant central mass
  - weak-field, slow-motion (1PN) regime
- **Limitations**:
  - no frame-dragging (Lense-Thirring) term
- **Validated by**:
  - `tests/test_relativity_effects.py`
- **See also**: [`LUNARIS-MODEL-REL-002`](#lunarismodelrel002), [`LUNARIS-MODEL-REL-003`](#lunarismodelrel003)

<a id="lunarismodelrel002"></a>
### LUNARIS-MODEL-REL-002 -- de Sitter (geodesic) precession acceleration

- **Slug**: `de_sitter_geodesic_precession`
- **Category**: physical_model | **Domain**: REL | **Status**: active
- **Classification**: exact
- **Verification**: verified_primary_source | **Scientific status**: implemented_and_tested
- **Primary reference**: `PetitLuzum2010IERSConventions` -- Petit, 2010. "{IERS} Conventions (2010)" (chapter 10 (General relativistic models for space-time coordinates and equations of motion); section 10.3; eq. 10.12 (geodesic/de Sitter term)) [ISBN: 978-3-89888-989-6]
- **Verification notes**: IERS Conventions (2010), TN 36, Eq. 10.12 geodesic term. The kernel forms Omega_dS = (3/2) mu_body/(c^2 |R|^3) (R x V) and applies a = +2 Omega x v (prograde). Sign verified against the canonical ~19.2 mas/yr solar geodetic precession; regression in tests/test_de_sitter_precession.py.
- **Mathematical contract**:
  - Inputs: spacecraft velocity and the Moon->body position/velocity
  - Outputs: geodesic-precession acceleration (m/s^2)
  - Exactness: exact_1pn_geodesic
  - Preserves: prograde precession of the orbit plane at |Omega_dS|
- **Implementing symbols**:
  - `src/lunaris/physics/relativity_effects.py` -- `de_sitter_components` (numba_implementation)
  - `src/lunaris/core/batch_propagator.py` -- `_de_sitter_cuda` (cuda_implementation)
- **Lunaris modifications**:
  - Coriolis-like +2 Omega x v application in the Moon-centred frame
- **Assumptions**:
  - Moon orbits the external body in the weak-field regime
- **Limitations**:
  - geodesic term only; no Lense-Thirring
- **Validated by**:
  - `tests/test_de_sitter_precession.py`
  - `tests/test_relativity_effects.py`
- **See also**: [`LUNARIS-MODEL-REL-001`](#lunarismodelrel001)
- **Notes**: Sign is intentionally +2 Omega x v (prograde); a previous sign error was corrected and is locked by the precession regression test.

<a id="lunarismodelrel003"></a>
### LUNARIS-MODEL-REL-003 -- External-body differential first post-Newtonian correction (Schwarzschild + de Sitter)

- **Slug**: `external_body_differential_1pn`
- **Category**: physical_model | **Domain**: REL | **Status**: active
- **Classification**: adaptation
- **Verification**: verified_primary_source | **Scientific status**: implemented_and_tested
- **Primary reference**: `PetitLuzum2010IERSConventions` -- Petit, 2010. "{IERS} Conventions (2010)" (chapter 10 (General relativistic models for space-time coordinates and equations of motion); section 10.3; eq. 10.12) [ISBN: 978-3-89888-989-6]
- **Verification notes**: Combines the differential Schwarzschild correction (a_1PN(SC wrt body) - a_1PN(Moon wrt body)) with the de Sitter term (LUNARIS-MODEL-REL-002), both derived from IERS TN 36 Eq. 10.12. The differential Moon-centred composition is a Lunaris adaptation of the single-body IERS forms.
- **Mathematical contract**:
  - Inputs: spacecraft state and the external body Moon-centred state and mu
  - Outputs: differential external-body 1PN acceleration (m/s^2)
  - Exactness: differential_1pn
  - Preserves: subtracts the Moon-origin 1PN term for the relative EOM
- **Implementing symbols**:
  - `src/lunaris/physics/relativity_effects.py` -- `external_schwarzschild_diff_components` (numba_implementation)
  - `src/lunaris/physics/relativity_effects.py` -- `external_1pn_components` (numba_implementation)
  - `src/lunaris/physics/relativity_effects.py` -- `calc_external_1pn_accel` (api_entry_point)
  - `src/lunaris/core/batch_propagator.py` -- `_external_schwarzschild_diff_cuda` (cuda_implementation)
  - `src/lunaris/core/batch_propagator.py` -- `_external_1pn_cuda` (cuda_implementation)
- **Lunaris modifications**:
  - differential Moon-centred composition of the IERS single-body terms
- **Assumptions**:
  - external body in the weak-field regime
- **Limitations**:
  - no Lense-Thirring; external Schwarzschild uses beta = gamma = 1
- **Validated by**:
  - `tests/test_relativity_effects.py`
- **See also**: [`LUNARIS-MODEL-REL-001`](#lunarismodelrel001), [`LUNARIS-MODEL-REL-002`](#lunarismodelrel002)

## Radiation pressure (RAD)

<a id="lunarismodelrad001"></a>
### LUNARIS-MODEL-RAD-001 -- Cannonball solar radiation pressure with a dual-cone (conical) umbra/penumbra shadow

- **Slug**: `cannonball_srp_conical_shadow`
- **Category**: physical_model | **Domain**: RAD | **Status**: active
- **Classification**: adaptation
- **Verification**: verified_primary_source | **Scientific status**: implemented_and_tested
- **Primary reference**: `MontenbruckGill2000SatelliteOrbits` -- Montenbruck, 2000. "Satellite Orbits: Models, Methods and Applications" (chapter 3 (Force Models); section 3.4 (solar radiation pressure and shadow function)) [DOI: 10.1007/978-3-642-58351-3]
- **Verification notes**: The cannonball law a = (P0 Cr A/m)(AU/d)^2 nu u_hat(sun->sc) and the dual-cone (umbra + penumbra) shadow geometry with a finite solar disk follow Montenbruck & Gill (2000), Section 3.4 (identifiers verified as for LUNARIS-ALG-SH-002). Lunaris ADAPTS the penumbra: the fractional illumination uses a cubic-Hermite smoothstep across the umbra->penumbra band instead of the exact overlapping-disk area, and supports both Moon and Earth occulters (shadow = min of the two).
- **Mathematical contract**:
  - Inputs: Moon-centred spacecraft, Sun and Earth positions; radii; AU; P0; Cr; area; mass; eclipse toggles
  - Outputs: solar-radiation-pressure acceleration (m/s^2)
  - Exactness: cannonball_with_smoothstep_penumbra
  - Preserves: inverse-square flux falloff with heliocentric distance
  - Preserves: zero force inside the umbra
- **Implementing symbols**:
  - `src/lunaris/physics/solar_effects.py` -- `accel_srp` (numba_implementation)
  - `src/lunaris/physics/solar_effects.py` -- `_shadow_factor_conical` (numba_implementation)
  - `src/lunaris/physics/solar_effects.py` -- `compute_srp_accel` (api_entry_point)
- **Lunaris modifications**:
  - smoothstep penumbra transition instead of exact occulted-disk area
  - combined Moon+Earth shadow via the minimum shadow factor
- **Assumptions**:
  - spherical spacecraft (cannonball), single reflectivity coefficient Cr
- **Limitations**:
  - no attitude-dependent or multi-surface SRP
  - penumbra fraction is an approximation, not the exact geometric area
- **Validated by**:
  - `tests/test_solar_effects.py`
  - `tests/test_srp_eclipse.py`
- **See also**: [`LUNARIS-MODEL-RAD-002`](#lunarismodelrad002), [`LUNARIS-MODEL-RAD-003`](#lunarismodelrad003)

<a id="lunarismodelrad002"></a>
### LUNARIS-MODEL-RAD-002 -- Faceted lunar albedo (shortwave) radiation-pressure model

- **Slug**: `facet_lunar_albedo_radiation_pressure`
- **Category**: empirical_model | **Domain**: RAD | **Status**: active
- **Classification**: adaptation
- **Verification**: verified_primary_source | **Scientific status**: implemented_and_tested
- **Primary reference**: `Knocke1988EarthRadiation` -- Knocke, 1988. "Earth Radiation Pressure Effects on Satellites" (section Diffuse-body shortwave (albedo) radiation force; pages 577-587) [DOI: 10.2514/6.1988-4292]
- **Verification notes**: P. C. Knocke, J. C. Ries & B. D. Tapley, "Earth Radiation Pressure Effects on Satellites", AIAA Paper 88-4292, 1988, pp. 577-587. DOI 10.2514/6.1988-4292 verified via AIAA. Lunaris ADAPTS this diffuse-body shortwave (albedo) radiation-pressure model from Earth to the MOON, summing reflected-sunlight contributions over a lat/lon facet grid.
- **Mathematical contract**:
  - Inputs: spacecraft state, Sun direction, facet grid, albedo, spacecraft props
  - Outputs: albedo radiation-pressure acceleration (m/s^2)
  - Exactness: faceted_diffuse_reflection_sum
  - Preserves: only sunlit, visible facets contribute
- **Implementing symbols**:
  - `src/lunaris/physics/lunar_albedo.py` -- `albedo_single_facet_accel_numba` (numba_implementation)
  - `src/lunaris/physics/lunar_albedo.py` -- `accel_albedo_facets_numba` (numba_implementation)
  - `src/lunaris/physics/lunar_albedo.py` -- `calc_albedo_accel` (api_entry_point)
- **Lunaris modifications**:
  - Earth ERP formalism applied to lunar albedo over a lat/lon facet grid
- **Assumptions**:
  - Lambertian diffuse reflection; cannonball spacecraft response
- **Limitations**:
  - facet-resolution dependent; static albedo map
- **Validated by**:
  - `tests/test_lunar_albedo.py`
  - `tests/test_surface_effects.py`
- **See also**: [`LUNARIS-MODEL-RAD-003`](#lunarismodelrad003)

<a id="lunarismodelrad003"></a>
### LUNARIS-MODEL-RAD-003 -- Faceted lunar thermal-infrared (longwave) radiation-pressure model

- **Slug**: `facet_thermal_ir_radiation_pressure`
- **Category**: empirical_model | **Domain**: RAD | **Status**: active
- **Classification**: adaptation
- **Verification**: verified_primary_source | **Scientific status**: implemented_and_tested
- **Primary reference**: `Knocke1988EarthRadiation` -- Knocke, 1988. "Earth Radiation Pressure Effects on Satellites" (section Diffuse-body longwave (thermal) radiation force; pages 577-587) [DOI: 10.2514/6.1988-4292]
- **Verification notes**: Same primary source as LUNARIS-MODEL-RAD-002 (Knocke et al. 1988, DOI 10.2514/6.1988-4292 verified); this entry is the longwave (thermal-IR) counterpart, summing thermal re-emission over the lunar facet grid.
- **Mathematical contract**:
  - Inputs: spacecraft state, facet grid, thermal emission model, spacecraft props
  - Outputs: thermal-IR radiation-pressure acceleration (m/s^2)
  - Exactness: faceted_thermal_emission_sum
  - Preserves: integrates longwave emission over visible facets
- **Implementing symbols**:
  - `src/lunaris/physics/thermal_ir.py` -- `build_latlon_facets` (cpu_implementation)
  - `src/lunaris/physics/thermal_ir.py` -- `accel_thermal_ir_facets_numba` (numba_implementation)
  - `src/lunaris/physics/thermal_ir.py` -- `calc_thermal_ir_accel` (api_entry_point)
- **Lunaris modifications**:
  - Earth ERP thermal formalism applied to lunar surface facets
- **Assumptions**:
  - diffuse thermal re-emission; cannonball spacecraft response
- **Limitations**:
  - depends on the surface thermal model and facet resolution
- **Validated by**:
  - `tests/test_thermal_ir.py`
  - `tests/test_surface_effects.py`
- **See also**: [`LUNARIS-MODEL-RAD-002`](#lunarismodelrad002)

## Ephemeris and interpolation (EPH)

<a id="lunarisalgeph001"></a>
### LUNARIS-ALG-EPH-001 -- SPICE-built fixed-grid ephemeris and lunar-orientation tables

- **Slug**: `spice_ephemeris_orientation_tables`
- **Category**: data_product | **Domain**: EPH | **Status**: active
- **Classification**: delegated_library
- **Verification**: verified_primary_source | **Scientific status**: implemented_and_tested
- **Primary reference**: `Acton1996SpiceNaif` -- Acton, 1996. "Ancillary Data Services of {NASA}'s Navigation and Ancillary Information Facility" (section SPICE ancillary information system; pages 65-70) [DOI: 10.1016/0032-0633(95)00107-7]
- **Verification notes**: C. H. Acton, "Ancillary data services of NASA's Navigation and Ancillary Information Facility", Planetary and Space Science 44(1):65-70, 1996. DOI 10.1016/0032-0633(95)00107-7 verified via ScienceDirect/ADS. Lunaris samples SPICE ``spkezr`` states (km and km/s, converted to SI) onto uniform-grid tables of third-body positions/velocities and lunar-orientation quaternions; the underlying ephemeris data product is DE440 (see LUNARIS-DATA-EPH-001).
- **Mathematical contract**:
  - Inputs: duration, output step, and loaded SPICE kernels
  - Outputs: uniform-grid tables of body positions (m), velocities (m/s), and unit orientation quaternions, plus GM lookups
  - Exactness: sampled_from_reference_ephemeris
  - Preserves: values match SPICE at grid nodes
  - Preserves: orientation quaternions renormalized to unit length
- **Implementing symbols**:
  - `src/lunaris/physics/ephemeris.py` -- `build_tables` (cpu_implementation)
  - `src/lunaris/physics/ephemeris.py` -- `build_spice_tables` (delegation_wrapper)
  - `src/lunaris/physics/ephemeris.py` -- `EphemerisManager` (api_entry_point)
  - `src/lunaris/physics/ephemeris.py` -- `load_ephemeris_tables_npz` (api_entry_point)
- **Lunaris modifications**:
  - fixed-grid resampling for fast in-loop interpolation
  - fail-closed handling when kernels are missing
- **Assumptions**:
  - required SPICE kernels (incl. DE440) are loaded
- **Limitations**:
  - off-node values require interpolation (see LUNARIS-ALG-INTP-002, LUNARIS-ALG-FRM-001)
- **Validated by**:
  - `tests/test_ephemeris.py`
  - `tests/test_main_ephemeris_policy.py`
- **See also**: [`LUNARIS-DATA-EPH-001`](#lunarisdataeph001), [`LUNARIS-ALG-INTP-002`](#lunarisalgintp002), [`LUNARIS-ALG-FRM-001`](#lunarisalgfrm001)

<a id="lunarisdataeph001"></a>
### LUNARIS-DATA-EPH-001 -- JPL DE440 planetary and lunar ephemeris

- **Slug**: `jpl_de440_planetary_lunar_ephemeris`
- **Category**: data_product | **Domain**: EPH | **Status**: active
- **Classification**: standard_implementation
- **Verification**: verified_primary_source | **Scientific status**: implemented_and_tested
- **Primary reference**: `ParkFolkner2021DE440` -- Park, 2021. "The {JPL} Planetary and Lunar Ephemerides {DE440} and {DE441}" (section DE440 planetary and lunar ephemeris; pages 105) [DOI: 10.3847/1538-3881/abd414]
- **Verification notes**: R. S. Park, W. M. Folkner, J. G. Williams & D. H. Boggs, "The JPL Planetary and Lunar Ephemerides DE440 and DE441", The Astronomical Journal 161(3):105, 2021. DOI 10.3847/1538-3881/abd414 verified via IOP/ ADS. DE440 supplies the third-body positions, lunar orientation/libration and the GM values recorded in Lunaris constants provenance.
- **Mathematical contract**:
  - Inputs: body identifier and epoch (via SPICE kernels)
  - Outputs: reference positions/orientation and GM constants
  - Exactness: reference_data_product
  - Preserves: consistent GM and body-frame definitions across the toolchain
- **Implementing symbols**:
  - `src/lunaris/common/constants.py` -- `(module)` (config_surface)
  - `src/lunaris/physics/ephemeris.py` -- `get_body_gm_m3s2` (api_entry_point)
- **Lunaris modifications**:
  - consumed through SPICE kernels; not re-integrated by Lunaris
- **Assumptions**:
  - DE440 kernels are the loaded ephemeris source
- **Limitations**:
  - superseded fields would require a kernel swap, not a code change
- **Validated by**:
  - `tests/test_canonical_constants.py`
  - `tests/test_ephemeris.py`
- **See also**: [`LUNARIS-ALG-EPH-001`](#lunarisalgeph001), [`LUNARIS-STD-FRM-001`](#lunarisstdfrm001)
- **Notes**: GRAIL lunar gravity coefficients are defined in the DE440 principal-axis body frame (see LUNARIS-STD-FRM-001).

## Frame conventions (FRM)

<a id="lunarisalgfrm001"></a>
### LUNARIS-ALG-FRM-001 -- Spherical linear interpolation of unit quaternions (SLERP)

- **Slug**: `slerp_unit_quaternion_interpolation`
- **Category**: interpolation | **Domain**: FRM | **Status**: active
- **Classification**: exact
- **Verification**: verified_primary_source | **Scientific status**: implemented_and_tested
- **Primary reference**: `Shoemake1985QuaternionCurves` -- Shoemake, 1985. "Animating Rotation with Quaternion Curves" (section Spherical linear interpolation (slerp); pages 245-254) [DOI: 10.1145/325165.325242]
- **Verification notes**: K. Shoemake, "Animating rotation with quaternion curves", ACM SIGGRAPH Computer Graphics 19(3):245-254, 1985. DOI 10.1145/325165.325242 verified via the ACM Digital Library (note: the DOI stem is 325165, not 325334). The implemented weights s0=sin((1-t)theta)/sin(theta), s1=sin(t theta)/sin(theta) are the SLERP formula, with shortest-arc sign handling and a lerp+normalize fallback for near-parallel quaternions.
- **Mathematical contract**:
  - Inputs: two unit quaternions and a parameter t in [0,1]
  - Outputs: interpolated unit quaternion
  - Exactness: exact_constant_angular_velocity
  - Preserves: constant angular velocity along the shortest arc
  - Preserves: unit norm of the result
- **Implementing symbols**:
  - `src/lunaris/common/math_utils.py` -- `_quat_slerp` (numba_implementation)
  - `src/lunaris/common/math_utils.py` -- `interp_quat_slerp` (cpu_implementation)
  - `src/lunaris/physics/ephemeris.py` -- `interp_quat_safe` (cpu_implementation)
  - `src/lunaris/common/math_utils.py` -- `quat_slerp_np` (api_entry_point)
  - `src/lunaris/batch/requirements.py` -- `_impact_positions_fixed` (delegation_wrapper)
  - `src/lunaris/core/propagation/events.py` -- `r_i_to_bf` (delegation_wrapper)
- **Lunaris modifications**:
  - shortest-arc sign flip when the dot product is negative
  - lerp+normalize fallback below a near-parallel threshold
- **Assumptions**:
  - both inputs are unit quaternions
- **Limitations**:
  - not commutative; endpoints must be ordered
- **Validated by**:
  - `tests/test_math_utils.py`
  - `tests/test_numba_frame_slerp.py`
- **See also**: [`LUNARIS-ALG-FRM-002`](#lunarisalgfrm002)

<a id="lunarisalgfrm002"></a>
### LUNARIS-ALG-FRM-002 -- Scalar-first (Hamilton) unit-quaternion active vector rotation

- **Slug**: `hamilton_scalar_first_quaternion_rotation`
- **Category**: frame_time_convention | **Domain**: FRM | **Status**: active
- **Classification**: standard_implementation
- **Verification**: verified_primary_source | **Scientific status**: implemented_and_tested
- **Primary reference**: `Kuipers1999Quaternions` -- Kuipers, 1999. "Quaternions and Rotation Sequences: A Primer with Applications to Orbits, Aerospace, and Virtual Reality" (section Quaternion rotation operator) [ISBN: 978-0-691-10298-6]
- **Verification notes**: J. B. Kuipers, "Quaternions and Rotation Sequences", Princeton University Press, 1999, ISBN 978-0-691-10298-6 verified. The kernel uses the scalar-first (Hamilton) convention and the Rodrigues form v' = v + 2 q0 (q_vec x v) + 2 q_vec x (q_vec x v), equivalent to q * [0,v] * q_conj. This fixes Lunaris's storage/handedness convention.
- **Mathematical contract**:
  - Inputs: unit quaternion (scalar-first) and a 3-vector
  - Outputs: the vector rotated into the target frame
  - Exactness: exact
  - Preserves: length-preserving (orthogonal) rotation
  - Preserves: conjugate equals inverse for unit quaternions
- **Implementing symbols**:
  - `src/lunaris/common/math_utils.py` -- `quat_rotate_vec` (numba_implementation)
  - `src/lunaris/common/math_utils.py` -- `quat_rotate_np` (api_entry_point)
  - `src/lunaris/common/math_utils.py` -- `quat_conj` (cpu_implementation)
- **Lunaris modifications**:
  - Rodrigues-form evaluation avoiding explicit quaternion products
- **Assumptions**:
  - scalar-first storage (q0, q1, q2, q3) and unit norm
- **Limitations**:
  - Hamilton (not JPL) convention; callers must supply matching quaternions
- **Validated by**:
  - `tests/test_math_utils.py`
  - `tests/test_torch_frame.py`
- **See also**: [`LUNARIS-ALG-FRM-001`](#lunarisalgfrm001), [`LUNARIS-STD-FRM-001`](#lunarisstdfrm001)
- **Notes**: Convention is scalar-first Hamilton with active rotation; SPICE-provided lunar-orientation quaternions are consumed under this convention.

<a id="lunarisstdfrm001"></a>
### LUNARIS-STD-FRM-001 -- Lunar principal-axis (PA) body-fixed frame realized by DE440

- **Slug**: `lunar_principal_axis_body_frame`
- **Category**: frame_time_convention | **Domain**: FRM | **Status**: active
- **Classification**: standard_implementation
- **Verification**: verified_primary_source | **Scientific status**: implemented_and_tested
- **Primary reference**: `ParkFolkner2021DE440` -- Park, 2021. "The {JPL} Planetary and Lunar Ephemerides {DE440} and {DE441}" (section Lunar orientation and principal-axis frame; pages 105) [DOI: 10.3847/1538-3881/abd414]
- **Verification notes**: The lunar body-fixed frame used for spherical-harmonic gravity is the principal-axis (PA) frame defined by the DE440 lunar libration (Park et al. 2021, DOI 10.3847/1538-3881/abd414 verified), realized through SPICE PCK/BPC kernels. This is distinct from the mean-Earth/polar-axis (ME) frame; GRAIL SHADR coefficients are defined in the PA frame, which is why Lunaris evaluates the field there.
- **Mathematical contract**:
  - Inputs: inertial state and a lunar-orientation quaternion
  - Outputs: state expressed in the lunar principal-axis body frame
  - Exactness: frame_convention
  - Preserves: consistency between the gravity-coefficient frame and the rotation applied
- **Implementing symbols**:
  - `src/lunaris/common/frame_policy.py` -- `resolve_frame_policy` (config_surface)
  - `src/lunaris/core/torch_frame.py` -- `(module)` (config_surface)
- **Lunaris modifications**:
  - rotation realized with the scalar-first Hamilton quaternion (LUNARIS-ALG-FRM-002)
- **Assumptions**:
  - orientation quaternions are the DE440-consistent PA realization
- **Limitations**:
  - does not implement the mean-Earth/polar-axis (ME) frame
- **Validated by**:
  - `tests/test_frame_policy.py`
  - `tests/test_sh_convention_lock.py`
- **See also**: [`LUNARIS-ALG-FRM-002`](#lunarisalgfrm002), [`LUNARIS-DATA-EPH-001`](#lunarisdataeph001), [`LUNARIS-ALG-SH-002`](#lunarisalgsh002)
- **Notes**: PA vs ME is a common source of lunar-gravity frame errors; this entry records that Lunaris uses the PA frame consistent with the coefficient set.

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
  - `src/lunaris/surrogate/st_lrps/evaluation/_gravity_benchmark/modes.py` -- `run_random_scenario_mode` (delegation_wrapper)
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
  - `src/lunaris/core/propagation/integrators/symplectic.py` -- `(module)` (cpu_implementation)
  - `src/lunaris/core/propagation/integrators/fixed_step.py` -- `_accel_stepper` (delegation_wrapper)
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

## Interpolation (INTP)

<a id="lunarisalgintp001"></a>
### LUNARIS-ALG-INTP-001 -- Clamped uniform Catmull-Rom cubic interpolation of vector time series

- **Slug**: `clamped_catmull_rom_vec3_interpolation`
- **Category**: interpolation | **Domain**: INTP | **Status**: active
- **Classification**: exact
- **Verification**: verified_primary_source | **Scientific status**: implemented_and_tested
- **Primary reference**: `CatmullRom1974Splines` -- Catmull, 1974. "A Class of Local Interpolating Splines" (section A class of local interpolating splines; pages 317-326) [DOI: 10.1016/B978-0-12-079050-0.50020-5]
- **Verification notes**: E. Catmull and R. Rom, "A class of local interpolating splines", in Computer Aided Geometric Design (Barnhill & Riesenfeld, eds.), Academic Press, 1974, pp. 317-326. ISBN 978-0-12-079050-0 and chapter DOI 10.1016/B978-0-12-079050-0.50020-5 verified. The implemented basis weights 0.5*[(-f+2f^2-f^3),(2-5f^2+3f^3),(f+4f^2-3f^3),(-f^2+f^3)] are the standard uniform Catmull-Rom cubic; control-point indices are clamped at the table ends.
- **Mathematical contract**:
  - Inputs: time, uniform step dt, and an (N,3) table
  - Outputs: C1-continuous interpolated 3-vector
  - Exactness: exact_catmull_rom
  - Preserves: passes through the control points (interpolating)
  - Preserves: C1 continuity of the tangent
- **Implementing symbols**:
  - `src/lunaris/common/math_utils.py` -- `interp_vec3_catmull` (cpu_implementation)
  - `src/lunaris/core/torch_third_body.py` -- `interp_vec3_catmull_torch` (torch_implementation)
  - `src/lunaris/physics/ephemeris.py` -- `interp_vec3_safe` (delegation_wrapper)
  - `src/lunaris/core/torch_third_body.py` -- `earth_position` (delegation_wrapper)
  - `src/lunaris/core/torch_third_body.py` -- `sun_position` (delegation_wrapper)
- **Lunaris modifications**:
  - endpoint index clamping (clamped Catmull-Rom variant)
  - endpoint/degenerate-table fallbacks
- **Assumptions**:
  - uniformly spaced samples in time
- **Limitations**:
  - uniform parameterization (not the centripetal variant)
- **Validated by**:
  - `tests/test_math_utils.py`
  - `tests/test_ephemeris_interpolation.py`
- **See also**: [`LUNARIS-ALG-EPH-001`](#lunarisalgeph001)
- **Notes**: Retained as the explicit compatibility path for legacy in-memory position-only providers. Canonical schema-v2 ephemeris artifacts use LUNARIS-ALG-INTP-002.

<a id="lunarisalgintp002"></a>
### LUNARIS-ALG-INTP-002 -- Cubic Hermite interpolation of ephemeris position and velocity states

- **Slug**: `cubic_hermite_ephemeris_state_interpolation`
- **Category**: interpolation | **Domain**: INTP | **Status**: active
- **Classification**: exact
- **Verification**: verified_primary_source | **Scientific status**: implemented_and_tested
- **Primary reference**: `Hermite1878Interpolation` -- Hermite, 1878. "Sur la formule d'interpolation de Lagrange" (section interpolation formula; pages 70-79) [DOI: 10.1515/crelle-1878-18788405]
- **Verification notes**: Charles Hermite, "Sur la formule d'interpolation de Lagrange", Journal fÃ¼r die reine und angewandte Mathematik 84:70-79, 1878; DOI 10.1515/crelle-1878-18788405 verified. Lunaris uses the standard cubic endpoint-value/endpoint-derivative Hermite basis on each uniform SPICE state interval.
- **Mathematical contract**:
  - Inputs: uniform sample step, position table (m), and matched velocity table (m/s)
  - Outputs: interpolated position and its analytic time derivative
  - Exactness: exact_piecewise_cubic_hermite
  - Preserves: exact position and velocity at every table node
  - Preserves: C1 continuity across table nodes
  - Preserves: exact reproduction of cubic position histories
- **Implementing symbols**:
  - `src/lunaris/physics/ephemeris.py` -- `interp_vec3_hermite` (cpu_implementation)
  - `src/lunaris/core/torch_third_body.py` -- `interp_vec3_hermite_torch` (torch_implementation)
  - `src/lunaris/core/batch_propagator.py` -- `_interp3_cuda` (cuda_implementation)
- **Lunaris modifications**:
  - endpoint time clamping to the covered SPICE interval
  - schema-v2 artifact enforcement with legacy archives rejected
- **Assumptions**:
  - position and velocity use the same frame, observer, aberration correction, and epochs
  - uniformly spaced samples in SI units
- **Limitations**:
  - piecewise cubic interpolation is C1 but not generally C2
- **Validated by**:
  - `tests/test_ephemeris.py`
  - `tests/test_ephemeris_interpolation.py`
  - `tests/test_numba_ephemeris_hermite.py`
  - `tests/test_torch_third_body.py`
- **See also**: [`LUNARIS-ALG-EPH-001`](#lunarisalgeph001), [`LUNARIS-ALG-INTP-001`](#lunarisalgintp001)
- **Notes**: CPU, Numba CUDA, and Torch paths consume the same position+velocity table contract; the Catmull-Rom route is compatibility-only.

<a id="lunarisheurintp001"></a>
### LUNARIS-HEUR-INTP-001 -- Latitude-clamped, longitude-periodic planetary grid sampling with missing-value fallback

- **Slug**: `periodic_planetary_grid_sampling_policy`
- **Category**: interpolation | **Domain**: INTP | **Status**: active
- **Classification**: heuristic
- **Verification**: unverifiable | **Scientific status**: implemented_and_tested
- **Primary reference**: Lunaris-specific; no external primary reference.
- **Mathematical contract**:
  - Inputs: regular latitude/longitude raster or flat row-major storage, floating grid coordinates, DN scale/offset, and optional missing sentinel
  - Outputs: interpolated physical scalar or NaN for an unavailable nearest sample
  - Exactness: piecewise_bilinear_or_nearest_policy
  - Preserves: latitude indices clamp at the polar rows
  - Preserves: longitude indices wrap periodically across the raster seam
  - Preserves: scaling is applied in physical-value space
- **Implementing symbols**:
  - `src/lunaris/common/math_utils.py` -- `sample_grid_bilinear` (api_entry_point)
  - `src/lunaris/common/math_utils.py` -- `sample_2d_scaled_bilinear_kernel` (numba_implementation)
  - `src/lunaris/common/math_utils.py` -- `sample_2d_scaled_nearest` (api_entry_point)
- **Lunaris modifications**:
  - any missing member of a bilinear stencil triggers nearest-neighbor fallback
  - nearest indices use half-up rounding
- **Assumptions**:
  - regular equirectangular grid metadata and row-major latitude ordering
- **Limitations**:
  - no area weighting or spherical interpolation
  - discontinuous at missing-data fallback boundaries
  - clamping does not infer data beyond the supplied polar rows
- **Validated by**:
  - `tests/test_math_utils.py`
- **See also**: [`LUNARIS-ALG-INTP-001`](#lunarisalgintp001)
- **Notes**: Bilinear and nearest-neighbor interpolation are standard constructions; this record covers Lunaris' boundary, DN scaling, and missing-data policy, so no originality or external defining source is claimed.

## Orbital elements (OE)

<a id="lunarisalgoe001"></a>
### LUNARIS-ALG-OE-001 -- Classical orbital-element conversions (state vector to/from Keplerian elements)

- **Slug**: `classical_orbital_element_conversions`
- **Category**: numerical_algorithm | **Domain**: OE | **Status**: active
- **Classification**: standard_implementation
- **Verification**: verified_primary_source | **Scientific status**: implemented_and_tested
- **Primary reference**: `Vallado2013Fundamentals` -- Vallado, 2013. "Fundamentals of Astrodynamics and Applications" (edition Fourth; chapter 2 (Kepler's Equation and Kepler's Problem); section RV2COE and COE2RV algorithms) [ISBN: 978-1-881883-18-0]
- **Verification notes**: D. A. Vallado, "Fundamentals of Astrodynamics and Applications", 4th ed., Microcosm Press, 2013. ISBN 978-1-881883-18-0 verified. The conversions follow Vallado's RV2COE / COE2RV algorithms with the standard special-case handling (circular, equatorial). PENDING only the exact algorithm numbers from the physical copy; the method identity is the classical osculating-element set.
- **Mathematical contract**:
  - Inputs: position/velocity (m, m/s) and mu, or Keplerian elements and mu
  - Outputs: osculating Keplerian elements, or state vector
  - Exactness: exact_two_body_transformation
  - Preserves: two-body invariants (energy, angular momentum) at the conversion epoch
- **Implementing symbols**:
  - `src/lunaris/common/math_utils.py` -- `rv_to_coe_select` (cpu_implementation)
  - `src/lunaris/common/math_utils.py` -- `coe_to_rv` (cpu_implementation)
  - `src/lunaris/common/math_utils.py` -- `batch_y_to_elements` (cpu_implementation)
- **Lunaris modifications**:
  - branch handling for near-circular and near-equatorial edge cases
  - vectorized batch element extraction
- **Assumptions**:
  - instantaneous osculating (two-body) elements
- **Limitations**:
  - undefined elements (e.g. node for equatorial orbits) use documented fallbacks
- **Validated by**:
  - `tests/test_math_utils.py`

## Sampling / design of experiments (SAMP)

<a id="lunarisalgsamp001"></a>
### LUNARIS-ALG-SAMP-001 -- Scrambled Sobol low-discrepancy sequence sampling

- **Slug**: `scrambled_sobol_low_discrepancy_design`
- **Category**: sampling | **Domain**: SAMP | **Status**: active
- **Classification**: delegated_library
- **Verification**: verified_primary_source | **Scientific status**: implemented_and_tested
- **Primary reference**: `Sobol1967Distribution` -- Sobol', 1967. "On the Distribution of Points in a Cube and the Approximate Evaluation of Integrals" (section Sobol low-discrepancy sequence; pages 86-112) [DOI: 10.1016/0041-5553(67)90144-9]
- **Verification notes**: Sobol' sequences: I. M. Sobol', USSR Comput. Math. Math. Phys. 7(4): 86-112, 1967 (DOI 10.1016/0041-5553(67)90144-9 verified). Lunaris uses scipy.stats.qmc.Sobol (Owen-type scrambling when scramble=True); SciPy is the delegated implementation (Virtanen et al. 2020). The initial-state and spacecraft-property designs map the unit sequence to physical ranges.
- **Mathematical contract**:
  - Inputs: number of samples, dimension, seed, scramble flag
  - Outputs: low-discrepancy design in the unit hypercube (then mapped to ranges)
  - Exactness: low_discrepancy_quasi_random
  - Preserves: low discrepancy / better space filling than pseudo-random sampling
- **Implementing symbols**:
  - `src/lunaris/batch/sampling.py` -- `generate_standard_normal_design` (delegation_wrapper)
  - `src/lunaris/batch/sampling.py` -- `sample_initial_states` (api_entry_point)
  - `src/lunaris/analysis/frozen/search.py` -- `sobol_element_samples` (delegation_wrapper)
  - `src/lunaris/analysis/frozen/search.py` -- `stage0_samples` (delegation_wrapper)
  - `src/lunaris/batch/engine.py` -- `run` (delegation_wrapper)
  - `src/lunaris/batch/sampling.py` -- `_sobol_size_note` (config_surface)
  - `src/lunaris/surrogate/st_lrps/evaluation/_gravity_benchmark/compute.py` -- `generate_unit_samples` (delegation_wrapper)
  - `src/lunaris/surrogate/st_lrps/evaluation/_gravity_benchmark/compute.py` -- `_sobol_note` (config_surface)
  - `src/lunaris/surrogate/st_lrps/evaluation/_gravity_benchmark/results_io.py` -- `_sampling_metadata` (config_surface)
  - `src/lunaris/surrogate/st_lrps/evaluation/_gravity_benchmark/results_io.py` -- `prepare_scenarios` (config_surface)
- **Lunaris modifications**:
  - Deterministic (unscrambled) Sobol discards the pathological all-zero first point; scrambled Sobol retains it (Lunaris policy documented in the sampler).
- **Assumptions**:
  - independent standardized inputs before range mapping
- **Limitations**:
  - unscrambled Sobol has no variance estimate; use scrambled for UQ
- **Validated by**:
  - `tests/test_batch_sampling.py`
  - `tests/test_batch_sampling_designs.py`
- **See also**: [`LUNARIS-ALG-SAMP-002`](#lunarisalgsamp002)
- **Notes**: The all-zero-first-point discard for the unscrambled variant is a Lunaris policy, not part of the Sobol construction.

<a id="lunarisalgsamp002"></a>
### LUNARIS-ALG-SAMP-002 -- Latin hypercube sampling

- **Slug**: `latin_hypercube_sampling`
- **Category**: sampling | **Domain**: SAMP | **Status**: active
- **Classification**: delegated_library
- **Verification**: verified_primary_source | **Scientific status**: implemented_and_tested
- **Primary reference**: `McKayBeckmanConover1979LHS` -- McKay, 1979. "A Comparison of Three Methods for Selecting Values of Input Variables in the Analysis of Output from a Computer Code" (section Latin hypercube sampling; pages 239-245) [DOI: 10.1080/00401706.1979.10489755]
- **Verification notes**: Latin hypercube sampling: McKay, Beckman & Conover, Technometrics 21(2): 239-245, 1979 (DOI 10.1080/00401706.1979.10489755 verified). Lunaris uses scipy.stats.qmc.LatinHypercube (Virtanen et al. 2020, delegated implementation).
- **Mathematical contract**:
  - Inputs: number of samples, dimension, seed
  - Outputs: stratified design with one sample per axis stratum
  - Exactness: stratified_sampling
  - Preserves: one-dimensional stratification along every axis
- **Implementing symbols**:
  - `src/lunaris/batch/sampling.py` -- `generate_standard_normal_design` (delegation_wrapper)
  - `src/lunaris/analysis/frozen/search.py` -- `sobol_element_samples` (delegation_wrapper)
- **Lunaris modifications**:
  - mapped to standardized normal inputs before physical range mapping
- **Assumptions**:
  - independent standardized inputs
- **Limitations**:
  - stratification is per-axis; no guaranteed multi-dim low discrepancy
- **Validated by**:
  - `tests/test_batch_sampling.py`
  - `tests/test_batch_sampling_designs.py`
- **See also**: [`LUNARIS-ALG-SAMP-001`](#lunarisalgsamp001)

## Uncertainty quantification (UQ)

<a id="lunarisalguq001"></a>
### LUNARIS-ALG-UQ-001 -- Empirical state covariance and eigendecomposed three-sigma position ellipsoid

- **Slug**: `empirical_state_covariance_and_principal_ellipsoid`
- **Category**: diagnostic | **Domain**: UQ | **Status**: active
- **Classification**: standard_implementation
- **Verification**: verified_secondary_source | **Scientific status**: implemented_and_tested
- **Primary reference**: `Heckert2002NISTHandbook` -- Heckert, 2002. "{NIST/SEMATECH} e-Handbook of Statistical Methods" (chapter 6. Process or Product Monitoring and Control; section 6.3.4.1 sample covariance; 6.5.3.2 eigenstructure) [DOI: 10.18434/M32189]
- **Verification notes**: The NIST/SEMATECH handbook gives the ddof=1 sample covariance matrix for independent multivariate observations and describes covariance eigenstructure. It is used here as an authoritative verification reference, not as an originality claim for covariance or symmetric eigendecomposition.
- **Mathematical contract**:
  - Inputs: N by 6 inertial Cartesian states [m, m/s], or a 6 by 6 state covariance applied to standardized design coordinates by a Cholesky factor
  - Outputs: ddof=1 empirical 6 by 6 covariance, component standard deviations, and 3-sigma position semi-axes [m] with inertial principal-axis directions
  - Exactness: empirical_finite_ensemble_summary
  - Preserves: covariance is symmetrized before the ellipsoid eigendecomposition
  - Preserves: negative eigenvalues caused only by roundoff are clipped to zero
- **Implementing symbols**:
  - `src/lunaris/analysis/ensemble/statistics.py` -- `_cov6` (cpu_implementation)
  - `src/lunaris/analysis/ensemble/statistics.py` -- `_position_ellipsoid_axes` (cpu_implementation)
  - `src/lunaris/analysis/ensemble/statistics.py` -- `compute_ensemble_statistics` (api_entry_point)
  - `src/lunaris/common/batch_defs.py` -- `StateUncertainty` (config_surface)
  - `src/lunaris/batch/sampling.py` -- `sample_initial_states` (cpu_implementation)
- **Lunaris modifications**:
  - altitude summaries use the configured reference radius
  - invalid samples and, optionally, impacted samples are excluded
- **Assumptions**:
  - at least two valid ensemble members
  - ddof=1 is an unbiased covariance estimator only for independent draws
- **Limitations**:
  - LHS and Sobol ensembles are non-IID designs; their covariance is an empirical design summary, not an unbiased Monte Carlo estimator
  - a componentwise or 3-sigma ellipsoid is not a distribution-free coverage guarantee
- **Validated by**:
  - `tests/test_ensemble_statistics_validity.py`
  - `tests/test_uq_report.py`
- **See also**: [`LUNARIS-ALG-SAMP-001`](#lunarisalgsamp001), [`LUNARIS-ALG-SAMP-002`](#lunarisalgsamp002), [`LUNARIS-ALG-PHZ-001`](#lunarisalgphz001)
- **Notes**: Source scope is verification of the standard covariance/eigenstructure mathematics. Sample filtering, altitude summaries, and QMC interpretation warnings are Lunaris analysis policy.

<a id="lunarisalguq002"></a>
### LUNARIS-ALG-UQ-002 -- Wilson score confidence interval for impact proportion

- **Slug**: `wilson_score_binomial_interval`
- **Category**: diagnostic | **Domain**: UQ | **Status**: active
- **Classification**: exact
- **Verification**: verified_primary_source | **Scientific status**: implemented_and_tested
- **Primary reference**: `Wilson1927ScoreInterval` -- Wilson, 1927. "Probable Inference, the Law of Succession, and Statistical Inference" (pages 209-212) [DOI: 10.1080/01621459.1927.10502953]
- **Verification notes**: Publisher DOI metadata and the original 1927 paper verify author, title, journal, volume 22(158), pages 209-212, and the score-based binomial interval. Lunaris evaluates the uncorrected two-sided form at z=1.96.
- **Mathematical contract**:
  - Inputs: impact count k, valid-sample count n, and normal quantile z
  - Outputs: lower and upper binomial-proportion bounds clipped to [0, 1]
  - Exactness: exact_wilson_score_formula
  - Preserves: returns [0, 1] when n is zero
  - Preserves: reported bounds remain within the probability interval
- **Implementing symbols**:
  - `src/lunaris/analysis/ensemble/statistics.py` -- `_binomial_ci_wilson` (cpu_implementation)
  - `src/lunaris/analysis/ensemble/statistics.py` -- `compute_impact_statistics` (api_entry_point)
- **Lunaris modifications**:
  - fixed default z=1.96 is labelled as a nominal 95 percent interval
- **Assumptions**:
  - impact outcomes are interpreted as Bernoulli trials for interval reporting
- **Limitations**:
  - nominal frequentist coverage assumes an IID binomial sample; QMC designs and correlated uncertainty draws do not satisfy that interpretation
  - no continuity correction
- **Validated by**:
  - `tests/test_ensemble_statistics_validity.py`
  - `tests/test_uq_report.py`
- **See also**: [`LUNARIS-ALG-UQ-001`](#lunarisalguq001)
- **Notes**: This record attributes the interval formula only. Impact detection and the definition of a valid ensemble member are separate Lunaris contracts.

## Phase / perturbation diagnostics (PHZ)

<a id="lunarisalgphz001"></a>
### LUNARIS-ALG-PHZ-001 -- Tangential Gauss variational equation for along-track phase drift with RIC error decomposition

- **Slug**: `tangential_gauss_ve_phase_drift`
- **Category**: diagnostic | **Domain**: PHZ | **Status**: active
- **Classification**: standard_implementation
- **Verification**: verified_primary_source | **Scientific status**: implemented_and_tested
- **Primary reference**: `Vallado2013Fundamentals` -- Vallado, 2013. "Fundamentals of Astrodynamics and Applications" (edition Fourth; chapter 9 (Special Perturbation Techniques); section Gauss variational equations and the RSW/RIC frame) [ISBN: 978-1-881883-18-0]
- **Verification notes**: The diagnostic integrates the exact tangential Gauss variational equation for the semi-major axis, da/dt = 2 a^2 v / mu * a_tangential, to predict along-track (phase) drift from a measured tangential acceleration bias, and decomposes position error in the radial/in-track/cross-track (RIC, a.k.a. RSW) frame. Both are standard astrodynamics (Vallado 4th ed., ISBN 978-1-881883-18-0 verified); exact section numbers pending physical copy. NASA/TP-20250006484, Section 3.1.4, Eqs. 3.1-1 through 3.1-3, independently verifies the R, C = R x V, and I = C x R basis convention; it is a verification reference, not the claimed source of the Gauss VE.
- **Mathematical contract**:
  - Inputs: reference and test trajectories (position/velocity) and mu
  - Outputs: predicted phase drift, RIC error history, and tangential-bias diagnosis
  - Exactness: exact_gauss_ve_causal_test
  - Preserves: links tangential acceleration bias to secular along-track drift
- **Implementing symbols**:
  - `src/lunaris/surrogate/st_lrps/evaluation/phase_diagnostics.py` -- `predict_phase_drift_from_tangential_bias` (cpu_implementation)
  - `src/lunaris/surrogate/st_lrps/evaluation/phase_diagnostics.py` -- `diagnose_tangential_bias` (cpu_implementation)
  - `src/lunaris/surrogate/st_lrps/evaluation/phase_diagnostics.py` -- `compute_ric_error_history` (cpu_implementation)
  - `src/lunaris/surrogate/st_lrps/evaluation/phase_diagnostics.py` -- `osculating_sma` (cpu_implementation)
  - `src/lunaris/analysis/orbit_analysis.py` -- `_to_ric` (cpu_implementation)
- **Lunaris modifications**:
  - causal test tying surrogate acceleration bias to observed phase drift
- **Assumptions**:
  - near-circular osculating dynamics for the Gauss VE integration
- **Limitations**:
  - diagnostic only; does not alter propagation
- **Validated by**:
  - `tests/test_phase_diagnostics.py`
  - `tests/test_orbit_drift.py`
  - `tests/test_uq_report.py`
- **See also**: [`LUNARIS-ALG-OE-001`](#lunarisalgoe001)
- **Notes**: Central to the phase-drift analysis: along-track error is dominated by semi-major-axis drift from a tangential acceleration bias.

## Frozen-orbit search (FRZ)

<a id="lunarisheurfrz001"></a>
### LUNARIS-HEUR-FRZ-001 -- Lunaris thresholded frozen-orbit candidate screening and validation gate

- **Slug**: `thresholded_frozen_orbit_candidate_screening`
- **Category**: heuristic | **Domain**: FRZ | **Status**: active
- **Classification**: heuristic
- **Verification**: unverifiable | **Scientific status**: implemented_and_tested
- **Primary reference**: Lunaris-specific; no external primary reference.
- **Mathematical contract**:
  - Inputs: time histories of eccentricity, perilune altitude [m], inclination [rad], argument of periapsis [rad], termination state, and mission thresholds
  - Outputs: lower-is-better screening score, boundedness reasons, and a candidate or validated classification label
  - Exactness: mission_threshold_screening_heuristic
  - Preserves: impacts, domain exits, escapes, and non-finite core metrics cannot rank as frozen
  - Preserves: validated labels require explicit classical-SH long-horizon evidence
- **Implementing symbols**:
  - `src/lunaris/analysis/frozen/metrics.py` -- `compute_frozen_metrics` (cpu_implementation)
  - `src/lunaris/analysis/frozen/classify.py` -- `frozen_score` (cpu_implementation)
  - `src/lunaris/analysis/frozen/classify.py` -- `classify_candidate` (api_entry_point)
  - `src/lunaris/analysis/frozen/classify.py` -- `FrozenClassificationConfig` (config_surface)
- **Lunaris modifications**:
  - combines envelopes, secular trends, omega behavior, and eccentricity-vector loop drift
  - derives default drift thresholds from the requested mission duration
- **Assumptions**:
  - osculating elements are sampled densely enough to represent the screened arc
  - mission-specific thresholds are supplied or consciously accepted
- **Limitations**:
  - not a universal mathematical definition of a frozen orbit
  - a surrogate-only screen can produce candidate language, never validated frozen status
  - classification applies only to the simulated duration and force/configuration evidence
- **Validated by**:
  - `tests/test_frozen_classification.py`
  - `tests/test_frozen_search_pipeline.py`
- **See also**: [`LUNARIS-ALG-OE-001`](#lunarisalgoe001)
- **Notes**: No external primary source is claimed: the score, thresholds, and evidence gate are Lunaris policy. Scientific publications may motivate frozen-orbit searches but do not define this software-specific classifier.

## Impact / terrain (IMP)

<a id="lunarisheurimp001"></a>
### LUNARIS-HEUR-IMP-001 -- Outer-sphere rejection and terrain-height bisection for batched impact localization

- **Slug**: `terrain_segment_impact_localization`
- **Category**: heuristic | **Domain**: IMP | **Status**: active
- **Classification**: heuristic
- **Verification**: unverifiable | **Scientific status**: implemented_and_tested
- **Primary reference**: Lunaris-specific; no external primary reference.
- **Mathematical contract**:
  - Inputs: batched inertial segment endpoints [m], segment epoch/step [s], inertial-to-fixed frame, terrain raster, and impact altitude [m]
  - Outputs: per-segment hit mask and interpolation fraction alpha in [0, 1]
  - Exactness: fixed_iteration_segment_crossing_localization
  - Preserves: non-hit rows return alpha=1
  - Preserves: impacted state position and velocity use the same interpolation fraction
- **Implementing symbols**:
  - `src/lunaris/core/torch_frame.py` -- `terrain_segment_intersection` (torch_implementation)
  - `src/lunaris/core/batched_fixed_step.py` -- `_propagate_chunk` (torch_implementation)
- **Lunaris modifications**:
  - outer maximum-terrain sphere rejects impossible crossings
  - body-frame rotation is evaluated once at the segment midpoint
  - fixed-count bisection runs on the terrain-height residual
- **Assumptions**:
  - one resolved downward crossing within the fixed step
  - lunar rotation during one step is negligible relative to terrain relief
- **Limitations**:
  - transient dip-and-recovery crossings inside one step are not detected
  - crossing state is linearly interpolated between integrator endpoints
  - accuracy is bounded by the propagation step, raster, and midpoint-frame approximation
- **Validated by**:
  - `tests/test_torch_terrain_freeze.py`
- **See also**: [`LUNARIS-HEUR-EVT-001`](#lunarisheurevt001), [`LUNARIS-HEUR-INTP-001`](#lunarisheurintp001)
- **Notes**: This is a Lunaris batched-impact policy composed from elementary sphere intersection and bisection. It is not presented as a published terrain collision algorithm.

## Neural architectures (ML)

<a id="lunarisalgml001"></a>
### LUNARIS-ALG-ML-001 -- Sinusoidal Representation Network (SIREN)

- **Slug**: `siren_sinusoidal_representation_network`
- **Category**: neural_architecture | **Domain**: ML | **Status**: active
- **Classification**: exact
- **Verification**: verified_primary_source | **Scientific status**: implemented_and_tested
- **Primary reference**: `Sitzmann2020SIREN` -- Sitzmann, 2020. "Implicit Neural Representations with Periodic Activation Functions" (section SIREN architecture and initialization scheme) [DOI: 10.5555/3495724.3496350]
- **Verification notes**: V. Sitzmann, J. N. P. Martel, A. W. Bergman, D. B. Lindell & G. Wetzstein, "Implicit Neural Representations with Periodic Activation Functions", NeurIPS 2020 (arXiv:2006.09661; ACM DOI 10.5555/3495724. 3496350 verified). The sin(w0 x) activation and the paper's principled init (first layer uniform(-1/n_in, 1/n_in); hidden uniform( -sqrt(6/n_in)/w0, +sqrt(6/n_in)/w0)) match the reference exactly.
- **Mathematical contract**:
  - Inputs: coordinate input tensor (scaled spacecraft position)
  - Outputs: implicit-field output (gravity potential or acceleration)
  - Exactness: exact_architecture
  - Preserves: periodic activations with the paper's variance-preserving init
- **Implementing symbols**:
  - `src/lunaris/surrogate/st_lrps/networks/models.py` -- `Sine` (torch_implementation)
  - `src/lunaris/surrogate/st_lrps/networks/models.py` -- `siren_init_hidden_` (torch_implementation)
  - `src/lunaris/surrogate/st_lrps/networks/models.py` -- `SirenMLP` (torch_implementation)
  - `src/lunaris/surrogate/runtime/networks.py` -- `SirenMLP` (reference_implementation)
  - `src/lunaris/surrogate/st_lrps/networks/models.py` -- `SirenResBlock` (torch_implementation)
  - `src/lunaris/surrogate/st_lrps/networks/models.py` -- `siren_init_first_` (torch_implementation)
  - `src/lunaris/surrogate/st_lrps/networks/models.py` -- `build_model_from_config` (api_entry_point)
  - `src/lunaris/surrogate/runtime/networks.py` -- `_build_model_from_config` (api_entry_point)
- **Lunaris modifications**:
  - applied to lunar gravity-field regression (default w0 = 30)
- **Assumptions**:
  - inputs scaled consistently with the trained scalers
- **Limitations**:
  - init scheme assumes the sine activation and matching w0
- **Validated by**:
  - `tests/test_surrogate_architecture_upgrades.py`
  - `tests/test_surrogate_gravity.py`
- **See also**: [`LUNARIS-ALG-ML-002`](#lunarisalgml002), [`LUNARIS-HEUR-ML-001`](#lunarisheurml001)

<a id="lunarisalgml002"></a>
### LUNARIS-ALG-ML-002 -- Random Fourier feature input encoding

- **Slug**: `random_fourier_feature_encoding`
- **Category**: neural_architecture | **Domain**: ML | **Status**: active
- **Classification**: exact
- **Verification**: verified_primary_source | **Scientific status**: implemented_and_tested
- **Primary reference**: `Tancik2020FourierFeatures` -- Tancik, 2020. "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains" (section Gaussian random Fourier feature mapping) [OFFICIAL URL: https://arxiv.org/abs/2006.10739]
- **Verification notes**: M. Tancik et al., "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains", NeurIPS 2020 (arXiv:2006.10739, stable arXiv URL verified; NeurIPS proceedings hash 55053683...). The embedding maps x -> [cos(2 pi B x), sin(2 pi B x)] with B drawn from a Gaussian of scale sigma, the Gaussian RFF mapping of the paper.
- **Mathematical contract**:
  - Inputs: low-dimensional coordinate input and a fixed random matrix B
  - Outputs: high-dimensional sinusoidal feature embedding
  - Exactness: exact_encoding
  - Preserves: tunable frequency bandwidth via sigma
  - Preserves: deterministic given the stored seed
- **Implementing symbols**:
  - `src/lunaris/surrogate/st_lrps/networks/models.py` -- `FourierInputEmbedding` (torch_implementation)
  - `src/lunaris/surrogate/runtime/networks.py` -- `FourierInputEmbedding` (reference_implementation)
  - `src/lunaris/surrogate/runtime/networks.py` -- `PhysicsNet` (delegation_wrapper)
  - `src/lunaris/surrogate/st_lrps/training/config.py` -- `TrainConfig` (config_surface)
  - `src/lunaris/surrogate/st_lrps/training/config.py` -- `parse_args` (config_surface)
  - `src/lunaris/surrogate/st_lrps/training/engine.py` -- `_log_training_curriculum` (config_surface)
  - `src/lunaris/surrogate/st_lrps/ui/studio_parts/training_pages.py` -- `STLRPSTrainTab` (config_surface)
- **Lunaris modifications**:
  - optional raw-input append; seed/sigma recorded in the artifact config
- **Assumptions**:
  - B is fixed at initialization and persisted for inference parity
- **Limitations**:
  - sigma must be tuned to the target spectral content
- **Validated by**:
  - `tests/test_surrogate_architecture_upgrades.py`
  - `tests/test_surrogate_gravity.py`
- **See also**: [`LUNARIS-ALG-ML-001`](#lunarisalgml001)

<a id="lunarisheurml001"></a>
### LUNARIS-HEUR-ML-001 -- Multi-band SIREN variants and physics-informed input encodings (Lunaris-specific)

- **Slug**: `multiband_siren_and_physics_encodings`
- **Category**: neural_architecture | **Domain**: ML | **Status**: active
- **Classification**: heuristic
- **Verification**: unverifiable | **Scientific status**: implemented_and_tested
- **Primary reference**: Lunaris-specific; no external primary reference.
- **Mathematical contract**:
  - Inputs: scaled coordinates (and radius) of the query point
  - Outputs: multi-band SIREN features / physics-motivated encodings
  - Exactness: project_specific_architecture
  - Preserves: geometrically-spaced w0 bands aligned to the SH residual spectrum
- **Implementing symbols**:
  - `src/lunaris/surrogate/st_lrps/networks/models.py` -- `MultiScaleSirenMLP` (torch_implementation)
  - `src/lunaris/surrogate/st_lrps/networks/models.py` -- `AdditiveMultiBandSirenMLP` (torch_implementation)
  - `src/lunaris/surrogate/st_lrps/networks/models.py` -- `compute_harmonic_w0_bands` (torch_implementation)
  - `src/lunaris/surrogate/st_lrps/networks/models.py` -- `RealSHBasisEncoding` (torch_implementation)
  - `src/lunaris/surrogate/st_lrps/networks/models.py` -- `PhysicalRadialDecayEncoding` (torch_implementation)
- **Lunaris modifications**:
  - w0 bands spaced in log-degree to match spherical-harmonic spatial frequencies (n/R_moon); radial-decay and real-SH-basis encodings that inject gravity structure into the input.
- **Assumptions**:
  - the residual field spectrum is bounded by the target SH degree
- **Limitations**:
  - Lunaris-specific composition; builds on SIREN (LUNARIS-ALG-ML-001) but the multi-band schedule and physics encodings are not a single published method. The real-SH basis follows the standard SH definition (LUNARIS-ALG-SH-001).
- **Validated by**:
  - `tests/test_surrogate_architecture_upgrades.py`
- **See also**: [`LUNARIS-ALG-ML-001`](#lunarisalgml001), [`LUNARIS-ALG-SH-001`](#lunarisalgsh001)
- **Notes**: Heuristic architecture family; the SIREN backbone and Fourier features are registered separately with their primary sources.

## Optimization (OPT)

<a id="lunarisalgopt001"></a>
### LUNARIS-ALG-OPT-001 -- AdamW (decoupled weight decay) optimizer as delegated to PyTorch

- **Slug**: `adamw_optimizer_delegation`
- **Category**: optimization | **Domain**: OPT | **Status**: active
- **Classification**: delegated_library
- **Verification**: verified_primary_source | **Scientific status**: implemented_and_tested
- **Primary reference**: `LoshchilovHutter2019AdamW` -- Loshchilov, 2019. "Decoupled Weight Decay Regularization" (section Decoupled weight decay (AdamW)) [OFFICIAL URL: https://arxiv.org/abs/1711.05101]
- **Verification notes**: Training uses torch.optim.AdamW (fused on CUDA when available). AdamW is Loshchilov & Hutter, "Decoupled Weight Decay Regularization", ICLR 2019 (arXiv:1711.05101 verified); the base Adam update is Kingma & Ba, ICLR 2015 (arXiv:1412.6980). The optimizer itself is the PyTorch implementation; Lunaris only configures parameter groups and schedule.
- **Mathematical contract**:
  - Inputs: model parameter groups, learning rate, weight decay
  - Outputs: parameter updates with decoupled weight decay
  - Exactness: adaptive_first_order_optimizer
  - Preserves: weight decay decoupled from the adaptive gradient step
- **Implementing symbols**:
  - `src/lunaris/surrogate/st_lrps/training/engine.py` -- `_build_model_and_optim` (delegation_wrapper)
- **Lunaris modifications**:
  - parameter-group construction and learning-rate schedule only
- **Assumptions**:
  - gradients are finite (NaN guards elsewhere in the loop)
- **Limitations**:
  - delegated; Lunaris does not reimplement the update rule
- **Validated by**:
  - `tests/test_st_lrps_training_improvements.py`
  - `tests/test_surrogate_training_contracts.py`

<a id="lunarisalgopt002"></a>
### LUNARIS-ALG-OPT-002 -- GradNorm adaptive multi-task loss balancing

- **Slug**: `gradnorm_adaptive_loss_balancing`
- **Category**: optimization | **Domain**: OPT | **Status**: active
- **Classification**: adaptation
- **Verification**: verified_primary_source | **Scientific status**: implemented_and_tested
- **Primary reference**: `Chen2018GradNorm` -- Chen, 2018. "{GradNorm}: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks" (section GradNorm gradient-magnitude balancing) [OFFICIAL URL: https://arxiv.org/abs/1711.02257]
- **Verification notes**: Z. Chen, V. Badrinarayanan, C.-Y. Lee & A. Rabinovich, "GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks", ICML 2018 (arXiv:1711.02257 verified). Lunaris ADAPTS GradNorm to balance its gravity loss terms (value/derivative/auxiliary) by tuning their gradient magnitudes on shared backbone parameters.
- **Mathematical contract**:
  - Inputs: per-task losses and shared-parameter gradients
  - Outputs: adaptively updated per-task loss weights
  - Exactness: adaptive_loss_weighting
  - Preserves: balances gradient magnitudes across loss terms
- **Implementing symbols**:
  - `src/lunaris/surrogate/st_lrps/training/losses.py` -- `GradNormWeights` (torch_implementation)
- **Lunaris modifications**:
  - applied to Lunaris gravity loss terms and backbone parameter selection
- **Assumptions**:
  - a meaningful shared parameter set exists for gradient balancing
- **Limitations**:
  - adds a backward pass for the gradient-norm targets
- **Validated by**:
  - `tests/test_st_lrps_baseline_gradnorm_domain.py`
- **See also**: [`LUNARIS-ALG-OPT-003`](#lunarisalgopt003)

<a id="lunarisalgopt003"></a>
### LUNARIS-ALG-OPT-003 -- Sobolev training (derivative-supervision loss)

- **Slug**: `sobolev_derivative_supervision_loss`
- **Category**: optimization | **Domain**: OPT | **Status**: active
- **Classification**: adaptation
- **Verification**: verified_primary_source | **Scientific status**: implemented_and_tested
- **Primary reference**: `Czarnecki2017Sobolev` -- Czarnecki, 2017. "Sobolev Training for Neural Networks" (section Sobolev training with target derivatives) [DOI: 10.5555/3294996.3295182]
- **Verification notes**: W. M. Czarnecki, S. Osindero, M. Jaderberg, G. Swirszcz & R. Pascanu, "Sobolev Training for Neural Networks", NeurIPS 2017 (arXiv:1706.04859; ACM DOI 10.5555/3294996.3295182 verified). Lunaris ADAPTS this to gravity: the potential network is supervised on both the value and its input-gradient (the acceleration), i.e. derivative supervision.
- **Mathematical contract**:
  - Inputs: predicted field, target values, and target derivatives
  - Outputs: combined value + derivative loss
  - Exactness: derivative_matching_loss
  - Preserves: supervises the gradient (acceleration), not only the value (potential)
- **Implementing symbols**:
  - `src/lunaris/surrogate/st_lrps/training/losses.py` -- `SobolevLoss` (torch_implementation)
- **Lunaris modifications**:
  - gravity-specific weighting and altitude balancing of the derivative term
- **Assumptions**:
  - target derivatives (accelerations) are available for training samples
- **Limitations**:
  - requires derivative labels; higher memory for the gradient term
- **Validated by**:
  - `tests/test_st_lrps_training_improvements.py`
  - `tests/test_surrogate_training_contracts.py`
- **See also**: [`LUNARIS-ALG-OPT-002`](#lunarisalgopt002)
- **Notes**: This is the core physics-supervision mechanism: matching accelerations as input-gradients of the learned potential.

## Scientific data products (DATA)

<a id="lunarisdatagrav001"></a>
### LUNARIS-DATA-GRAV-001 -- GRAIL GL1800F lunar spherical-harmonic gravity field (JGGRX_1800F)

- **Slug**: `grail_gl1800f_lunar_gravity_field`
- **Category**: data_product | **Domain**: DATA | **Status**: active
- **Classification**: standard_implementation
- **Verification**: verified_primary_source | **Scientific status**: implemented_and_tested
- **Primary reference**: `GrailGL1800FSHADR` -- {JPL GRAIL Level-2 Team}, 2023. "{GRAIL} Lunar Gravity Field {GL1800F} (Spherical Harmonic Coefficients, {JGGRX\_1800F\_SHA})" (section JGGRX_1800F_SHA product) [OFFICIAL URL: https://pds-geosciences.wustl.edu/grail/grail-l-lgrs-5-rdr-v1/grail_1001/shadr/]
- **Verification notes**: The shipped coefficient set is GL1800F (file jggrx_1800f_sha), read directly from its PDS SHADR label: degree/order 1800, fully normalized, reference radius 1738.0 km, DE440 principal-axis frame, k2=0.024223, k3=0.0163. Attribution resolved (2026-07-18): the PDS label (updated 2025-05-16) credits the JPL Level-2 team with model developers R. S. Park, A. Berne, A. S. Konopliv, J. T. Keane, I. Matsuyama, F. Nimmo, M. Rovira-Navarro, M. P. Panning, M. Simons, D. J. Stevenson and R. C. Weber, and cites Park et al., "Thermal asymmetry in the Moon's mantle inferred from monthly tidal response", Nature 641(8065), 1188-1192 (2025), DOI 10.1038/s41586-025-08949-5 (DOI and author list verified; the label's k3=0.0163 matches the paper's k3=0.0163+-0.0007). The GRAIL high-resolution field methodology remains Konopliv et al. 2014 (DOI 10.1002/2013GL059066 verified).
- **Mathematical contract**:
  - Inputs: SHADR coefficient file path
  - Outputs: fully-normalized C_nm/S_nm blocks, reference radius, and GM for spherical-harmonic evaluation
  - Exactness: reference_gravity_coefficient_set
  - Preserves: fully-normalized (no Condon-Shortley) convention matching LUNARIS-ALG-SH-001
  - Preserves: DE440 principal-axis body frame (LUNARIS-STD-FRM-001)
- **Implementing symbols**:
  - `src/lunaris/loaders/io_gravity.py` -- `load_shadr_ascii` (cpu_implementation)
  - `src/lunaris/loaders/io_gravity.py` -- `load_gravity_model` (api_entry_point)
- **Lunaris modifications**:
  - degree truncation at load time; structural monopole handling
- **Assumptions**:
  - coefficients interpreted in the DE440 PA frame at radius 1738.0 km
- **Limitations**:
  - static field; no time-variable gravity
- **Validated by**:
  - `tests/test_loader_helpers.py`
  - `tests/test_gravity_reference_runner.py`
- **See also**: [`LUNARIS-ALG-SH-002`](#lunarisalgsh002), [`LUNARIS-STD-FRM-001`](#lunarisstdfrm001)
- **Notes**: This is the actual lunar gravity field Lunaris evaluates; the SH algorithm (LUNARIS-ALG-SH-002) consumes these coefficients.

## Physical constants (CST)

<a id="lunarisdatacst001"></a>
### LUNARIS-DATA-CST-001 -- CODATA 2018 recommended fundamental physical constants

- **Slug**: `codata_2018_fundamental_constants`
- **Category**: data_product | **Domain**: CST | **Status**: active
- **Classification**: standard_implementation
- **Verification**: verified_primary_source | **Scientific status**: implemented_and_tested
- **Primary reference**: `Tiesinga2021CODATA` -- Tiesinga, 2021. "{CODATA} Recommended Values of the Fundamental Physical Constants: 2018" (section Recommended values (CODATA 2018); pages 025010) [DOI: 10.1103/RevModPhys.93.025010]
- **Verification notes**: E. Tiesinga, P. J. Mohr, D. B. Newell & B. N. Taylor, "CODATA Recommended Values of the Fundamental Physical Constants: 2018", Reviews of Modern Physics 93, 025010 (2021). DOI 10.1103/RevModPhys.93.025010 verified via APS/ADS. Lunaris uses the CODATA 2018 gravitational constant G = 6.6743e-11 and the exact SI speed of light c = 299792458 m/s.
- **Mathematical contract**:
  - Inputs: none (constant definitions)
  - Outputs: G, c and derived radiation constants
  - Exactness: reference_constant_values
  - Preserves: single source of truth for physical constants across the toolchain
- **Implementing symbols**:
  - `src/lunaris/common/constants.py` -- `(module)` (config_surface)
- **Lunaris modifications**:
  - provenance strings record the source per constant
- **Assumptions**:
  - SI units throughout
- **Limitations**:
  - body GMs come from DE440, not CODATA (see LUNARIS-DATA-EPH-001)
- **Validated by**:
  - `tests/test_canonical_constants.py`
- **See also**: [`LUNARIS-DATA-EPH-001`](#lunarisdataeph001)
- **Notes**: c is the exact post-2019-SI definition; G is CODATA 2018.
