# ST_LRPS/core/state.py
# -*- coding: utf-8 -*-
"""
Orbital State & Coordinate Transformation Engine

This module serves as the single source of truth for the 6-DOF Cartesian state 
vector used in the lunar simulation. It provides a robust bridge between 
numerical integration arrays and geometric Keplerian representations.

Canonical State Definition (SI Units):
--------------------------------------
The system state is represented as a flat, 6-element float64 array:
$$ y = [r_x, r_y, r_z, v_x, v_y, v_z]^T $$

- Position components  : Meters [m]
- Velocity components  : Meters per second [m/s]

Key Functional Areas:
---------------------
1. Data Integrity & Validation:
   - Strict finiteness and dimensionality checks (NaN/Inf protection).
   - Domain-specific validation for orbital geometry (elliptic vs. hyperbolic).

2. Transformation Kernels:
   - Efficient COE (Keplerian) <-> Cartesian (r, v) conversions.
   - Perifocal (PQW) to Inertial (IJK) frame rotations.

3. High-Level Containers:
   - 'OrbitState'       : Object-oriented wrapper for physical state analysis.
   - 'ClassicalElements': Geometric representation for orbital design.

4. Integration Interface (Packaging):
   - Standardized pack/unpack routines for numerical ODE solvers.

Design Constraints & Scope:
---------------------------
- Computational Purity: Strictly numerical; no I/O, SPICE dependency, or side effects.
- Type Safety        : Utilizes TypeAlias and Overloads for enhanced developer experience.
- JIT Compatibility  : Designed to work seamlessly with Numba-accelerated dynamics.
"""


# =============================================================================
# 0.                              IMPORTS
# =============================================================================

from __future__ import annotations


from dataclasses import dataclass
from numpy.typing import ArrayLike, NDArray
from typing import Tuple, Union, overload, Literal

import math
import numpy as np

from lunaris.common.constants import (MU_MOON,
                              KM_TO_M, M_TO_KM,
                              EPS_1E12, EPS_1E15)


from lunaris.common.math_utils import norm3, wrap_angle_2pi, rv_to_coe_select

from lunaris.common.type_defs import Vec3



STATE_SIZE: int = 6


# =============================================================================
# 1.                  ARRAY VALIDATION & CONVERSION HELPERS
# =============================================================================

def _ensure_1d_float_array(
    source_data: ArrayLike, 
    expected_length: int, 
    *, 
    param_name: str
) -> NDArray[np.float64]:
    """
    Standardizes input into a 1D float64 NumPy array and validates data integrity.

    Converts the input to a flat array, verifies that the length matches the 
    expected dimension, and ensures no non-finite values (NaN/Inf) are present.
    """
    # Force conversion to float64 and flatten to 1D
    array_out = np.asarray(source_data, dtype=np.float64).ravel()

    if array_out.size != expected_length:
        raise ValueError(
            f"Input '{param_name}' must have exactly {expected_length} elements. "
            f"Got size: {array_out.size}."
        )

    if not np.isfinite(array_out).all():
        raise ValueError(
            f"Input '{param_name}' contains non-finite values (NaN or Inf). "
            "Computation aborted to prevent instability."
        )

    return array_out


def as_vec3(data: ArrayLike, name: str = "vector") -> NDArray[np.float64]:
    """Ensures input is a 3-element (x, y, z) float64 array."""
    return _ensure_1d_float_array(data, 3, param_name=name)


def _calculate_vec3_norm(vector: ArrayLike, name: str = "vector") -> float:
    """
    Computes the Euclidean norm (magnitude) of a 3-element vector.

    Note: Bridges the gap between array-based inputs and the scalar-optimized 
    math_utils.norm3 function.
    """
    v_clean = as_vec3(data=vector, name=name)
    # math_utils.norm3 expects three distinct scalars for maximum JIT speed
    return float(norm3(float(v_clean[0]), float(v_clean[1]), float(v_clean[2])))



# =============================================================================
# 2.                      STATE PACKING & UNPACKING 
# =============================================================================

def pack_orbital_state(position: ArrayLike, velocity: ArrayLike) -> NDArray[np.float64]:
    """
    Packs separate position and velocity vectors into a single 6-element state vector.

    Creates a standardized flat array suitable for numerical integration (ODE solvers).
    The resulting structure follows the convention: 
    $$ y = [r_x, r_y, r_z, v_x, v_y, v_z]^T $$

    Parameters
    ----------
    position : ArrayLike
        The Cartesian position vector [m].
    velocity : ArrayLike
        The Cartesian velocity vector [m/s].

    Returns
    -------
    NDArray[np.float64]
        A 6-element float64 state vector.
    """
    # Validate and clean inputs using our specialized helpers
    r_clean = as_vec3(position, name="position")
    v_clean = as_vec3(velocity, name="velocity")

    # Efficiently allocate and fill the 6-element vector
    state_vector = np.empty(STATE_SIZE, dtype=np.float64)
    state_vector[:3] = r_clean
    state_vector[3:] = v_clean
    
    return state_vector


def unpack_orbital_state(
    state_vector: ArrayLike, 
    *, 
    copy: bool = True
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Decomposes a 6-element state vector back into position and velocity components.

    Splits the combined vector $$ y $$ into its physical constituents $$ \vec{r} $$ and $$ \vec{v} $$.

    Parameters
    ----------
    state_vector : ArrayLike
        The 6-element state vector to be decomposed.
    copy : bool, optional
        If True (default), returns deep copies of the components.
        If False, returns memory-efficient views (use with caution during mutation).

    Returns
    -------
    Tuple[NDArray[np.float64], NDArray[np.float64]]
        The decoupled (position, velocity) vectors.
    """
    # Ensure input is a valid 6-element float64 state
    y_clean = _ensure_1d_float_array(state_vector, STATE_SIZE, param_name="state_vector")

    # Slice the vector into position and velocity
    position = y_clean[:3]
    velocity = y_clean[3:]

    if copy:
        return position.copy(), velocity.copy()
    
    return position, velocity



# =============================================================================
# 3.                            CORE CONTAINERS
# =============================================================================

@dataclass(slots=True)
class OrbitState:
    """
    Cartesian State Vector container (r, v) in the Body-Fixed or Inertial frame.
    
    This class acts as a high-level wrapper around raw NumPy arrays, providing 
    convenient access to orbital invariants like energy, angular momentum, 
    and Keplerian transformations.
    """
    position: Vec3       # Radius vector [m]
    velocity: Vec3       # Velocity vector [m/s]

    def __post_init__(self) -> None:
        """Validates input dimensions and types to ensure Numba compatibility."""
        self.position = as_vec3(self.position, "position")
        self.velocity = as_vec3(self.velocity, "velocity")

    @property
    def y(self) -> NDArray[np.float64]:
        """
        Returns the packed state vector [rx, ry, rz, vx, vy, vz].
        Standard format for ODE solvers.
        """
        return pack_orbital_state(self.position, self.velocity)

    @classmethod
    def from_y(cls, state_vector: ArrayLike) -> "OrbitState":
        """Factory: Creates an OrbitState instance from a flat (6,) array."""
        r, v = unpack_orbital_state(state_vector, copy=True)
        return cls(r, v)

    def copy(self) -> "OrbitState":
        """Returns a deep copy of the current state instance."""
        return OrbitState(self.position.copy(), self.velocity.copy())

    @property
    def r_mag(self) -> float:
        """Magnitude of the position vector (scalar distance) [m]."""
        return _calculate_vec3_norm(self.position, "position")

    @property
    def v_mag(self) -> float:
        """Magnitude of the velocity vector (scalar speed) [m/s]."""
        return _calculate_vec3_norm(self.velocity, "velocity")

    def compute_specific_energy(self, mu: float = float(MU_MOON)) -> float:
        """
        Calculates the specific orbital energy ($J/kg$).
        
        Formula:
        $$ \\epsilon = \\frac{v^2}{2} - \\frac{\\mu}{r} $$
        """
        # Using built-in max() to guard against singularity at the center
        r_safe = max(self.r_mag, 1e-6) 
        return 0.5 * (self.v_mag**2) - (mu / r_safe)

    def compute_angular_momentum(self) -> Vec3:
        """
        Calculates the specific angular momentum vector.
        
        Formula:
        $$ \\vec{h} = \\vec{r} \\times \\vec{v} $$
        """
        return np.cross(self.position, self.velocity)

    def to_keplerian(self, mu: float = float(MU_MOON)) -> "ClassicalElements":
        """
        Transforms the Cartesian state into Classical Orbital Elements (COE).
        Returns a ClassicalElements instance.
        """
        kepler_tuple = cartesian_to_keplerian(
            self.position, 
            self.velocity, 
            mu=mu,
            wrap_angles=True
        )
        return ClassicalElements(*kepler_tuple)
    

@dataclass(frozen=True, slots=True)
class ClassicalElements:
    """
    Classical Orbital Elements (COE) — Keplerian representation.
    
    This class defines the orbital geometry using six fundamental parameters. 
    All angular elements are stored in radians.
    """
    a: float     # Semi-major axis [m]
    e: float     # Eccentricity [-]
    inc: float   # Inclination [rad]
    raan: float  # Right Ascension of Ascending Node [rad]
    argp: float  # Argument of Perigee [rad]
    ta: float    # True Anomaly [rad]

    def normalized(self) -> "ClassicalElements":
        """
        Returns a new instance with all angular components wrapped to [0, 2π).
        Useful for long-term propagations where angles may accumulate.
        """
        return type(self)(
            self.a,
            self.e,
            wrap_angle_2pi(self.inc),
            wrap_angle_2pi(self.raan),
            wrap_angle_2pi(self.argp),
            wrap_angle_2pi(self.ta),
        )

    def to_cartesian(self, mu: float = float(MU_MOON)) -> Tuple[Vec3, Vec3]:
        """
        Transforms Keplerian elements into Cartesian Position and Velocity vectors.
        Returns a tuple of (r, v).
        """
        return keplerian_to_cartesian(
            self.a, self.e, self.inc, self.raan, self.argp, self.ta, 
            mu=mu
        )

    def to_state_vector(self, mu: float = float(MU_MOON)) -> NDArray[np.float64]:
        """
        Converts elements into a packed (6,) Cartesian state vector [r, v].
        Primarily used for initializing numerical integrators.
        """
        r, v = self.to_cartesian(mu=mu)
        return pack_orbital_state(r, v)

    def to_orbit_state(self, mu: float = float(MU_MOON)) -> "OrbitState":
        """
        Converts elements into a high-level OrbitState instance.
        Enables immediate access to energy, momentum, and other properties.
        """
        r, v = self.to_cartesian(mu=mu)
        return OrbitState(r, v)



# =============================================================================
# 4.                            COE <-> Cartesian
# =============================================================================

def _validate_gravitational_parameter(mu: float) -> float:
    """
    Validates the gravitational parameter (mu).
    
    Ensures mu is finite and strictly positive, as negative gravity 
    is physically non-permitted in this simulation context.
    """
    mu_val = float(mu)
    if not math.isfinite(mu_val) or mu_val <= 0.0:
        raise ValueError(f"Gravitational parameter (mu) must be finite and > 0. Got: {mu_val}")
    return mu_val


def _validate_orbital_geometry(semi_major_axis: float, eccentricity: float) -> Tuple[float, float]:
    """
    Validates the semi-major axis (a) and eccentricity (e) relationship.

    Enforces the following astrodynamics conventions:
    1. Eccentricity must be non-negative (e >= 0).
    2. Parabolic orbits (e = 1.0) are not supported via (a, e) parametrization.
    3. Elliptic orbits (e < 1): Semi-major axis must be positive (a > 0).
    4. Hyperbolic orbits (e > 1): Semi-major axis must be negative (a < 0).
    """
    a = float(semi_major_axis)
    e = float(eccentricity)

    if not (math.isfinite(a) and math.isfinite(e)):
        raise ValueError(f"Orbital elements (a, e) must be finite. Got a={a}, e={e}")

    if e < 0.0:
        raise ValueError(f"Eccentricity (e) cannot be negative. Got: {e}")

    # Parabolic guard: avoid division by zero or undefined states
    if abs(e - 1.0) < EPS_1E12:
        raise ValueError(
            "Parabolic orbits (e ≈ 1.0) are mathematically singular in (a, e) "
            "parametrization. Use a different state representation."
        )

    # Conic section consistency checks
    if e < 1.0 and a <= 0.0:
        raise ValueError(f"Elliptic orbit (e < 1) requires a positive semi-major axis (a > 0). Got a={a}")
    
    if e > 1.0 and a >= 0.0:
        raise ValueError(f"Hyperbolic orbit (e > 1) requires a negative semi-major axis (a < 0). Got a={a}")

    return a, e


def _ensure_finite_angles(*angles: float) -> Tuple[float, ...]:
    """
    Verifies that all provided angular values (inclination, RAAN, etc.) are finite.
    """
    validated = tuple(float(x) for x in angles)
    if not all(math.isfinite(x) for x in validated):
        raise ValueError("One or more orbital angles are non-finite (NaN or Inf).")
    return validated


def keplerian_to_cartesian(
    semi_major_axis: float,
    eccentricity: float,
    inclination: float,
    raan: float,
    argp: float,
    true_anomaly: float,
    mu: float = float(MU_MOON),
) -> Tuple[Vec3, Vec3]:
    """
    Transforms Classical Orbital Elements (COE) to Cartesian state vectors.

    Converts geometry into Position (r) and Velocity (v) in an inertial frame.
    Supports elliptic (e < 1) and hyperbolic (e > 1) trajectories.
    
    Rotation Logic:
    Perifocal (PQW) -> Inertial (IJK) via $$ R_z(\\Omega) R_x(i) R_z(\\omega) $$
    """
    # 1. Validation & Preprocessing
    mu_val = _validate_gravitational_parameter(mu)
    a, e = _validate_orbital_geometry(semi_major_axis, eccentricity)
    inc, raan_val, argp_val, ta = _ensure_finite_angles(inclination, raan, argp, true_anomaly)

    # 2. Perifocal Distance Calculation
    # Semi-latus rectum: p = a(1 - e^2)
    p = a * (1.0 - e**2)
    if not math.isfinite(p) or p <= 0.0:
        raise ValueError("Invalid orbital geometry: Semi-latus rectum (p) must be > 0.")

    cos_ta, sin_ta = math.cos(ta), math.sin(ta)
    
    # Radius magnitude in the orbital plane
    denom = 1.0 + e * cos_ta
    if abs(denom) <= EPS_1E15:
        raise ValueError("Trajectory singularity: 1 + e*cos(ta) is near zero (hyperbolic asymptote).")
    
    r_mag = p / denom

    # 3. Position and Velocity in Perifocal Frame (PQW)
    # Z-component is zero by definition in the perifocal plane
    r_pqw = np.array([r_mag * cos_ta, r_mag * sin_ta, 0.0], dtype=np.float64)

    v_factor = math.sqrt(mu_val / p)
    v_pqw = np.array([-v_factor * sin_ta, v_factor * (e + cos_ta), 0.0], dtype=np.float64)

    # 4. Construct Rotation Matrix (DCM)
    # Rotating from PQW to Inertial (IJK)
    cO, sO = math.cos(raan_val), math.sin(raan_val)
    ci, si = math.cos(inc), math.sin(inc)
    cw, sw = math.cos(argp_val), math.sin(argp_val)

    # Direction Cosine Matrix components
    dcm = np.empty((3, 3), dtype=np.float64)
    dcm[0, 0] = cO * cw - sO * sw * ci
    dcm[0, 1] = -cO * sw - sO * cw * ci
    dcm[0, 2] = sO * si
    dcm[1, 0] = sO * cw + cO * sw * ci
    dcm[1, 1] = -sO * sw + cO * cw * ci
    dcm[1, 2] = -cO * si
    dcm[2, 0] = sw * si
    dcm[2, 1] = cw * si
    dcm[2, 2] = ci

    # 5. Project to Inertial Frame
    r_inertial = dcm @ r_pqw
    v_inertial = dcm @ v_pqw

    return r_inertial, v_inertial


def keplerian_to_state_vector(
    semi_major_axis: float,
    eccentricity: float,
    inclination: float,
    raan: float,
    argp: float,
    true_anomaly: float,
    mu: float = float(MU_MOON),
) -> NDArray[np.float64]:
    """
    Converts Classical Orbital Elements (COE) directly to a packed Cartesian state vector.

    This is a convenience wrapper that combines coordinate transformation and 
    state packing into a single call, typically used to initialize ODE integrators.

    The resulting vector follows the convention:
    $$ y = [r_x, r_y, r_z, v_x, v_y, v_z]^T $$

    Parameters
    ----------
    semi_major_axis : float
        Semi-major axis [m].
    eccentricity : float
        Orbital eccentricity [-].
    inclination : float
        Inclination [rad].
    raan : float
        Right Ascension of Ascending Node [rad].
    argp : float
        Argument of Perigee [rad].
    true_anomaly : float
        True anomaly [rad].
    mu : float, optional
        Gravitational parameter [m^3/s^2]. Defaults to MU_MOON.

    Returns
    -------
    NDArray[np.float64]
        A packed (6,) state vector containing position and velocity.
    """
    # 1. Transform geometry to Cartesian vectors
    position, velocity = keplerian_to_cartesian(
        semi_major_axis, 
        eccentricity, 
        inclination, 
        raan, 
        argp, 
        true_anomaly, 
        mu=mu
    )

    # 2. Pack vectors into the standardized (6,) integration format
    return pack_orbital_state(position, velocity)


@overload
def create_state_from_keplerian(
    semi_major_axis: float, eccentricity: float, inclination: float,
    raan: float, argp: float, true_anomaly: float,
    mu: float = float(MU_MOON), *, return_array: Literal[False] = False,
) -> "OrbitState": ...

@overload
def create_state_from_keplerian(
    semi_major_axis: float, eccentricity: float, inclination: float,
    raan: float, argp: float, true_anomaly: float,
    mu: float = float(MU_MOON), *, return_array: Literal[True],
) -> NDArray[np.float64]: ...


def create_state_from_keplerian(
    semi_major_axis: float,
    eccentricity: float,
    inclination: float,
    raan: float,
    argp: float,
    true_anomaly: float,
    mu: float = float(MU_MOON),
    *,
    return_array: bool = False,
) -> Union["OrbitState", NDArray[np.float64]]:
    """
    Main entry point for creating orbital states from Keplerian elements.
    
    Can return either a high-level OrbitState object or a raw packed (6,) NumPy array.
    """
    r, v = keplerian_to_cartesian(
        semi_major_axis, eccentricity, inclination, 
        raan, argp, true_anomaly, mu=mu
    )

    if return_array:
        return pack_orbital_state(r, v)
    
    return OrbitState(r, v)


def cartesian_to_keplerian(
    position: Vec3,
    velocity: Vec3,
    *,
    mu: float = float(MU_MOON),
    wrap_angles: bool = True,
) -> Tuple[float, float, float, float, float, float]:
    """
    Transforms Cartesian Position (r) and Velocity (v) to Keplerian elements.
    
    Returns
    -------
    (a, e, i, Ω, ω, ν) : tuple
        Semi-major axis, eccentricity, inclination, RAAN, Arg. of Perigee, True Anomaly.
    """
    # Validate and standardize inputs
    r_vec = as_vec3(position, "position")
    v_vec = as_vec3(velocity, "velocity")
    mu_val = _validate_gravitational_parameter(mu)

    # Core conversion kernel (External optimized routine)
    a, e, inc, raan, argp, ta = rv_to_coe_select(r_vec, v_vec, mu_val, mode="coe6")

    # Optional angle normalization
    if wrap_angles:
        inc  = wrap_angle_2pi(inc) 
        raan = wrap_angle_2pi(raan)
        argp = wrap_angle_2pi(argp)
        ta   = wrap_angle_2pi(ta)

    return float(a), float(e), float(inc), float(raan), float(argp), float(ta)



# =============================================================================
# 5.                          Altitudes <-> (a,e)
# =============================================================================

def _ensure_positive_finite(value: float, name: str) -> float:
    """
    Ensures a value is a finite number and strictly greater than zero.
    Used for physical constants like radii and gravitational parameters.
    """
    val = float(value)
    if not math.isfinite(val) or val <= 0.0:
        raise ValueError(f"Parameter '{name}' must be finite and > 0. Got: {val}")
    return val


def _validate_elliptic_geometry(semi_major_axis: float, eccentricity: float) -> Tuple[float, float]:
    """
    Validates that the provided (a, e) parameters describe a valid elliptic orbit.
    
    Requirements:
    1. 0 <= eccentricity < 1
    2. semi_major_axis > 0
    """
    a = float(semi_major_axis)
    e = float(eccentricity)
    
    if not (math.isfinite(a) and math.isfinite(e)):
        raise ValueError(f"Orbital parameters must be finite. Got a={a}, e={e}")
        
    if not (0.0 <= e < 1.0):
        raise ValueError(f"Eccentricity must be in range [0, 1) for elliptic orbits. Got: {e}")
        
    if a <= 0.0:
        raise ValueError(f"Semi-major axis must be positive for elliptic orbits. Got: {a}")
        
    return a, e


def calculate_periapsis_apoapsis_radii(
    semi_major_axis: float, 
    eccentricity: float
) -> Tuple[float, float]:
    """
    Calculates periapsis and apoapsis radii from semi-major axis and eccentricity.

    Formulas:
    $$ r_p = a(1 - e) $$
    $$ r_a = a(1 + e) $$

    Returns
    -------
    periapsis_radius, apoapsis_radius : float [m]
    """
    a, e = _validate_elliptic_geometry(semi_major_axis, eccentricity)
    
    periapsis_radius = a * (1.0 - e)
    apoapsis_radius = a * (1.0 + e)
    
    return periapsis_radius, apoapsis_radius


def calculate_ae_from_radii(
    periapsis_radius: float, 
    apoapsis_radius: float
) -> Tuple[float, float]:
    """
    Computes semi-major axis and eccentricity from periapsis and apoapsis radii.

    Formulas:
    $$ a = \frac{r_p + r_a}{2} $$
    $$ e = \frac{r_a - r_p}{r_a + r_p} $$
    """
    rp = _ensure_positive_finite(periapsis_radius, "periapsis_radius")
    ra = _ensure_positive_finite(apoapsis_radius, "apoapsis_radius")

    # Safety: ensure apoapsis is actually the larger radius
    if ra < rp:
        rp, ra = ra, rp

    semi_major_axis = 0.5 * (rp + ra)
    eccentricity = (ra - rp) / (ra + rp)  # Safe since rp, ra > 0
    
    return semi_major_axis, eccentricity


def calculate_ae_from_altitudes(
    reference_radius: float, 
    periapsis_alt_km: float, 
    apoapsis_alt_km: float
) -> Tuple[float, float]:
    """
    Derives (a, e) from altitudes above a reference radius.

    Parameters
    ----------
    reference_radius : float [m]
    periapsis_alt_km : float [km]
    apoapsis_alt_km : float [km]
    """
    r_ref = _ensure_positive_finite(reference_radius, "reference_radius")

    # Convert altitudes to radii in meters
    periapsis_radius = r_ref + float(periapsis_alt_km) * KM_TO_M
    apoapsis_radius = r_ref + float(apoapsis_alt_km) * KM_TO_M
    
    return calculate_ae_from_radii(periapsis_radius, apoapsis_radius)


def calculate_altitudes_from_ae(
    reference_radius: float, 
    semi_major_axis: float, 
    eccentricity: float
) -> Tuple[float, float]:
    """
    Calculates periapsis and apoapsis altitudes [km] from orbital elements.
    """
    r_ref = _ensure_positive_finite(reference_radius, "reference_radius")

    periapsis_radius, apoapsis_radius = calculate_periapsis_apoapsis_radii(
        semi_major_axis, eccentricity
    )
    
    # Convert radii back to altitudes in km
    periapsis_alt_km = (periapsis_radius - r_ref) * M_TO_KM
    apoapsis_alt_km = (apoapsis_radius - r_ref) * M_TO_KM
    
    return periapsis_alt_km, apoapsis_alt_km



# =============================================================================
# 6.                                PUBLIC API
# =============================================================================

__all__ = (
    # --- Fundamental Constants ---
    "STATE_SIZE",                       # Length of the packed state vector (6,)

    # --- Core Data Containers ---
    "OrbitState",                       # Cartesian state wrapper (position, velocity)
    "ClassicalElements",                # Keplerian elements wrapper (a, e, i, Ω, ω, ν)

    # --- State Vector Management ---
    "pack_orbital_state",               # (r, v) -> packed (6,) array [rx, ry, rz, vx, vy, vz]
    "unpack_orbital_state",             # packed (6,) -> (r, v) components

    # --- Coordinate Transformations (COE <-> Cartesian) ---
    "keplerian_to_cartesian",           # Geometric elements -> Inertial vectors (r, v)
    "cartesian_to_keplerian",           # Inertial vectors (r, v) -> Geometric elements
    "keplerian_to_state_vector",        # COE -> Packed (6,) array
    "create_state_from_keplerian",      # Factory: Returns OrbitState or packed array

    # --- Elliptic Geometry & Altitude Utilities ---
    "calculate_periapsis_apoapsis_radii", # (a, e) -> (rp, ra) distance from center [m]
    "calculate_ae_from_radii",           # (rp, ra) -> (a, e) orbital parameters
    "calculate_ae_from_altitudes",       # (hp, ha) -> (a, e) using reference radius
    "calculate_altitudes_from_ae"        # (a, e) -> (hp, ha) height above reference [km]
)
