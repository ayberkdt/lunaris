# lunaris/common/constants.py
"""
Lunar Simulation Physical Constants & Reference Values
==============================================

This module serves as the Single Source of Truth (SSOT) for all physical,
astronomical, and mathematical constants used throughout the Lunar Simulation project.
It eliminates the use of "magic numbers" scattered across the codebase.

Architectural Design Principles
-------------------------------
1. Dependency-Free: This module is written in Pure Python. It deliberately
   avoids imports from `numpy`, `scipy`, or `numba`.
   - Reasoning: This ensures that constants can be imported directly into
     Numba JIT-compiled kernels (`@njit`) without triggering compilation overhead
     or type inference conflicts.
   
2. SI Units: Unless explicitly stated otherwise, all quantities conform to
   the International System of Units (SI):
   - Length: Metres (m)
   - Time: Seconds (s)
   - Mass: Kilograms (kg)
   - Angle: Radians (rad)

3. Traceability: Constants are derived from authoritative astronomical sources
   (e.g., JPL DE440 Ephemerides, IAU 2012, CODATA 2018) to ensure high-fidelity
   dynamical propagation.

Usage Guidelines
----------------
- Gravitational Parameters: Use the pre-defined `MU_*` (mu = G * M)
  constants instead of multiplying `G * Mass` manually. Ephemeris providers usually
  determine mu with higher precision than the individual mass.

- Time Systems: Constants like `DAY_S` and `MJD_J2000` define the continuous
  time grid (TDB/TT). This module does **not** account for UTC leap seconds.
  For high-precision UTC <-> TDB transformations, utilize the SPICE-based layer
  in `models/ephemeris.py`.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Final, Mapping


# =============================================================================
# 1.                          UNIVERSAL CONSTANTS
# =============================================================================

# Speed of light [m/s] (exact, by definition)
C_LIGHT: float = 299_792_458.0

# Stefan-Boltzmann constant [W m^-2 K^-4].
# Exact after the 2019 SI redefinition through fixed h, k_B, and c.
SIGMA_SB: float = 5.670_374_419e-8

# Newtonian gravitational constant [m^3 kg^-1 s^-2] (CODATA 2018)
G: float = 6.6743e-11


# =============================================================================
# 2.                            TIME CONSTANTS
# =============================================================================

# Seconds per day [s]
DAY_S: float = 86_400.0

# Days per Julian year [day]
JULIAN_YEAR_DAYS: float = 365.25

# J2000 epoch in Modified Julian Date:
#   2000-01-01 12:00:00 (TT)
MJD_J2000_TT: float = 51_544.5


# =============================================================================
# 3.                            MOON PARAMETERS
# =============================================================================

# Lunar gravitational parameter mu = GM [m^3/s^2]
MU_MOON: float = 4_904_869_500_000.0

# Reference radius used in many lunar gravity/topography products [m]
R_MOON: float = 1_738_000.0

# Mean lunar radius (useful for altitude-like quantities) [m]
R_MOON_MEAN: float = 1_737_400.0

# Sidereal rotation rate of the Moon [rad/s]
OMEGA_MOON: float = 2.6616995e-06


# =============================================================================
# 4.                            EARTH PARAMETERS
# =============================================================================

# Earth gravitational parameter [m^3/s^2] (Standard: 3.986004418e14)
MU_EARTH: float = 398_600_441_800_000.0

# Earth Equatorial Radius (WGS-84 semi-major axis) [m]
R_EARTH_EQUATORIAL: float = 6_378_137.0

# Earth Mean Radius (Volumetric mean / IERS standard approx) [m]
R_EARTH_MEAN: float = 6_371_000.0            # Often used for spherical approximation.

# Earth Mean Angular Velocity of Rotation [rad/s]
# Standard WGS-84 value: 7.2921150e-5 rad/s
OMEGA_EARTH: float = 7.292_115_0e-5


# =============================================================================
# 5.                             SOLAR SYSTEM
# =============================================================================

# Solar gravitational parameter [m^3/s^2]
MU_SUN: float = 1.32712440042e+20

# IAU 2015 nominal solar radius [m]
R_SUN_MEAN: float = 695_700_000.0  

# Astronomical Unit [m] (IAU 2012 exact)
AU: float = 149_597_870_700.0

# Solar radiation pressure at 1 AU [N/m^2] (typical nominal value)
P_SUN_1AU: float = 4.56 * 1e-06

# Solar flux at 1 AU [W/m^2], kept consistent with the nominal SRP pressure.
SOLAR_FLUX_1AU: float = P_SUN_1AU * C_LIGHT


# =============================================================================
# 6.                 MATHEMATICAL CONSTANTS & CONVERSIONS
# =============================================================================

# Keep PI as a plain float so this module stays dependency-free.
PI: float = 3.141592653589793

TWO_PI: float = 2.0 * PI
HALF_PI: float = 0.5 * PI

DEG2RAD: float = PI / 180.0
RAD2DEG: float = 180.0 / PI


KM_TO_M: float = 1_000.0
M_TO_KM: float = 1e-3

KM3_TO_M3: float = 1e9
M3_TO_KM3: float = 1e-9


NEARLY_UNIT: float = 0.9995

EPS_1E6 : float = 1e-6
EPS_1E12: float = 1e-12
EPS_1E15: float = 1e-15
EPS_1E18: float = 1e-18
EPS_1E24: float = 1e-24
EPS_1E30: float = 1e-30


# =============================================================================
# 7.                            METADATA & PROVENANCE
# =============================================================================

# Dictionary mapping constant names to their authoritative sources.
# This ensures scientific traceability for orbit determination audits.
CONSTANT_SOURCES: Final[Mapping[str, str]] = MappingProxyType({
    # --- Universal ---
    "C_LIGHT": "SI definition (exact)",
    "SIGMA_SB": "SI/CODATA exact value after 2019 SI redefinition",
    "G": "CODATA 2018",

    # --- Time / epochs (conventions) ---
    "DAY_S": "SI day = 86400 s (definition)",
    "JULIAN_YEAR_DAYS": "Julian year convention (365.25 d)",
    "MJD_J2000_TT": "J2000 epoch: 2000-01-01 12:00:00 TT => MJD 51544.5",

    # --- Sun / solar system ---
    "R_SUN_MEAN": "IAU 2015 nominal solar radius",
    "AU": "IAU 2012 exact definition",
    "MU_SUN": "JPL DE440 ephemeris (GM of Sun)",
    "P_SUN_1AU": "Nominal SRP at 1 AU (common astrodynamics reference value)",
    "SOLAR_FLUX_1AU": "Derived from nominal P_SUN_1AU * C_LIGHT",

    # --- Earth ---
    "MU_EARTH": "JPL DE440 geocentric GM",
    "R_EARTH_EQUATORIAL": "WGS-84 standards",
    "R_EARTH_MEAN": "IERS/WGS-84 mean radius convention used in project",
    "OMEGA_EARTH": "IERS/WGS-84 nominal Earth rotation rate",

    # --- Moon ---
    "MU_MOON": "JPL DE440 / lunar gravity model reference (GM)",
    "R_MOON": "LRO LOLA reference radius used by lunar products",
    "R_MOON_MEAN": "Mean lunar radius (project reference / product convention)",
    "OMEGA_MOON": "Derived from lunar sidereal period (ω = 2π/T)",
})


# =============================================================================
# 8.                       PUBLIC API DEFINITION (__all__)
# =============================================================================

# Defines the symbols exported when using: from lunaris.common.constants import *
__all__ = (
    # 1. Universal & Physical Constants
    "C_LIGHT",                   # Speed of light in vacuum [m/s]
    "SIGMA_SB",                   # Stefan-Boltzmann constant [W m^-2 K^-4]
    "G",                         # Newtonian constant of gravitation [m^3/(kg*s^2)]

    # 2. Time & Epoch Standards
    "DAY_S",                     # Seconds in a Julian day [s]
    "JULIAN_YEAR_DAYS",          # Days in a Julian year (365.25) [day]
    "MJD_J2000_TT",                 # Modified Julian Date of J2000.0 epoch

    # 3. Lunar System (The Moon)
    "MU_MOON",                   # Lunar gravitational parameter GM [m^3/s^2]
    "R_MOON",                    # Lunar reference/equatorial radius [m]
    "R_MOON_MEAN",               # Lunar mean volumetric radius [m]
    "OMEGA_MOON",                # Lunar mean rotation rate [rad/s]

    # 4. The Earth
    "MU_EARTH",                  # Earth gravitational parameter GM [m^3/s^2]
    "R_EARTH_EQUATORIAL",        # Earth equatorial radius (WGS-84) [m]
    "R_EARTH_MEAN",              # Earth mean radius [m]
    "OMEGA_EARTH",               # Earth mean angular velocity [rad/s]

    # 5. Solar Constants
    "MU_SUN",                    # Solar gravitational parameter GM [m^3/s^2]
    "R_SUN_MEAN",                # Solar nominal mean radius [m]
    "AU",                        # Astronomical Unit [m]
    "P_SUN_1AU",                 # Solar radiation pressure at 1 AU [N/m^2]
    "SOLAR_FLUX_1AU",            # Solar flux at 1 AU [W/m^2]

    # 6. Mathematics & Unit Conversions
    "PI",                        # π
    "TWO_PI",                    # 2π
    "HALF_PI",                   # π/2
    "DEG2RAD",                   # Degrees -> radians factor
    "RAD2DEG",                   # Radians -> degrees factor
    "KM_TO_M",                   # km -> m factor
    "M_TO_KM",                   # m -> km factor
    "KM3_TO_M3",                 # km^3 -> m^3 factor
    "M3_TO_KM3",                 # m^3 -> km^3 factor
    
    # 7. Numeric tolerances (scale-explicit)
    "NEARLY_UNIT",
    "EPS_1E6",
    "EPS_1E12",                  # generic small epsilon (angles, wrap, comparisons)
    "EPS_1E15",                  # norm / division guard epsilon
    "EPS_1E18",                  # squared-scale epsilon (e.g. r^2 near-origin)
    "EPS_1E24",
    "EPS_1E30",

    # 8. Project Metadata
    "CONSTANT_SOURCES",          # Scientific traceability mapping
)
