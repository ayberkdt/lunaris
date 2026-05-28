# LUNAR_SIMULATION/common/type_defs.py
"""
Type Definitions & Configuration Dataclasses
============================================

This module defines the **public data model** for the project: type aliases, configuration
dataclasses, and result containers. It is intended to be the single source of truth (SSOT)
for structures shared across layers (UI/CLI → core → models → post-processing).

Principles
----------
- **Stable surface area:** keep public types small, explicit, and documented.
- **Clear ownership:**
  - `TimeConfig`: time span and output sampling policy.
  - `PropagatorConfig`: numerical method selection, tolerances, safety limits, and run controls.
- `GravityConfig` / `AdaptiveDegreeConfig`: central-gravity backend selection and SH degree policy.
  - `PerturbationFlags`: enable/disable switches for force models.
  - `PropagationResult` / `SimulationHistory`: propagation outputs and analysis-friendly views.
- **Early failure:** dataclasses validate inputs in `__post_init__` to catch misconfiguration
  as close to the source as possible.
- **Dependency-light:** avoid importing heavy libraries here. (NumPy typing is acceptable;
  runtime-heavy modules should live elsewhere.)

Conventions
-----------
- Units are included in field names where ambiguity is likely (`*_km`, `*_m`, `*_s`).
- All dataclasses are `frozen=True, slots=True` where mutation is not required.
- Arrays are normalized to `float64` and validated for shape in result containers.

Time scales
-----------
Time strings may be interpreted analytically (e.g., without leap-second tables) depending on the
layer that parses them. This module stores configuration only; it does not define UTC↔TT behavior.
"""


# =============================================================================
# 0.                              IMPORTS
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, List, TypeAlias, Annotated

import numpy as np
import numpy.typing as npt

from .constants import DAY_S, R_MOON_MEAN


# =============================================================================
# 1.                            TYPE ALIASES
# =============================================================================

# ---- Base numeric dtype ----
F64 = np.float64

# ---- Base array type (float64) ----
F64Array: TypeAlias = npt.NDArray[np.float64]

# ---- Generic intent aliases (dimension by convention; not enforced by type checkers) ----
Arr1: TypeAlias = Annotated[F64Array, "1D float64 array"]
Arr2: TypeAlias = Annotated[F64Array, "2D float64 array"]

# ---- Semantic aliases (shape by convention; document intent for readers) ----
Vec3: TypeAlias = Annotated[F64Array, "shape=(3,)"]           # position, velocity, accel, thrust, etc.
Quat: TypeAlias = Annotated[F64Array, "shape=(4,)"]           # [w, x, y, z] convention
StateVector: TypeAlias = Annotated[F64Array, "shape=(6,)"]    # [rx, ry, rz, vx, vy, vz]
Matrix3x3: TypeAlias = Annotated[F64Array, "shape=(3,3)"]     # DCM, inertia tensor, etc.


# =============================================================================
# 2.                        PHYSICAL PROPERTIES
# =============================================================================

@dataclass(frozen=True, slots=True, kw_only=True)
class SpacecraftProps:
    """
    Spacecraft physical properties used by non-gravitational perturbations (e.g., SRP, drag).

    Notes
    -----
    - Immutable (`frozen=True`) to prevent accidental mutation during propagation.
    - `slots=True` reduces memory footprint and avoids accidental attribute typos.
    - `kw_only=True` prevents silent bugs from positional-argument ordering mistakes.
    """
    mass_kg: float = 1000.0   # wet mass [kg]
    area_m2: float = 2.0      # effective cross-sectional area [m^2]
    cd: float = 2.2           # drag coefficient [-]
    cr: float = 1.8           # SRP reflectivity coefficient [-]

    def __post_init__(self) -> None:
        if not (self.mass_kg > 0.0):
            raise ValueError(f"mass_kg must be > 0. Got {self.mass_kg!r}")
        if not (self.area_m2 >= 0.0):
            raise ValueError(f"area_m2 must be >= 0. Got {self.area_m2!r}")
        if not (self.cd >= 0.0):
            raise ValueError(f"cd must be >= 0. Got {self.cd!r}")
        if not (self.cr >= 0.0):
            raise ValueError(f"cr must be >= 0. Got {self.cr!r}")

    @property
    def ballistic_coefficient(self) -> float:
        """
        Ballistic coefficient: BC = m / (Cd * A) [kg/m^2].

        - Larger BC => less drag sensitivity.
        - If Cd*A == 0, BC is infinite (no drag coupling in this simplified model).
        """
        denom = self.cd * self.area_m2
        return float("inf") if denom <= 0.0 else (self.mass_kg / denom)



# =============================================================================
# 3.                GRAVITY CONFIG (CENTRAL BODY)
# =============================================================================

@dataclass(frozen=True, slots=True, kw_only=True)
class AdaptiveDegreeConfig:
    """
    Dynamic spherical-harmonic (SH) degree selection policy.

    Priority
    --------
    1) enabled == False     -> always use full degree (GravityConfig.degree)
    2) altitude_table set   -> use table lookup (piecewise-constant schedule)
    3) else                 -> use power-law heuristic

    Notes
    -----
    - This is a policy container only; evaluation logic belongs in the gravity model.
    - Quantization can reduce JIT/cache churn if kernels depend on degree.
    """
    enabled: bool = False

    # Power-law strategy parameters:
    # n(r) ~ floor(n_max * (R_ref / r) ** power)
    power: float = 2.5
    min_degree: int = 4
    quantization_step: int = 10

    # Table strategy:
    # Each row: (altitude_threshold_km, degree)
    # The engine selects the first row where current_alt_km <= threshold_km.
    altitude_table: Optional[Tuple[Tuple[float, int], ...]] = None

    def __post_init__(self) -> None:
        if self.power <= 0.0:
            raise ValueError(f"power must be > 0, got {self.power!r}")

        if self.min_degree < 0:
            raise ValueError(f"min_degree must be >= 0, got {self.min_degree!r}")

        if self.quantization_step < 1:
            raise ValueError(
                f"quantization_step must be >= 1, got {self.quantization_step!r}"
            )

        table = self.altitude_table
        if table is None:
            return

        if len(table) == 0:
            raise ValueError("altitude_table cannot be empty; use None to disable.")

        prev_alt = -1.0
        for i, row in enumerate(table):
            if len(row) != 2:
                raise ValueError(
                    f"Row {i}: expected (altitude_threshold_km, degree), got {row!r}"
                )
            alt_km, deg = row

            if alt_km < 0.0:
                raise ValueError(
                    f"Row {i}: altitude_threshold_km must be >= 0, got {alt_km!r}"
                )
            if deg < 0:
                raise ValueError(f"Row {i}: degree must be >= 0, got {deg!r}")

            # Strictly increasing thresholds to avoid ambiguous selection.
            if alt_km <= prev_alt:
                raise ValueError(
                    "altitude_table must be strictly increasing in altitude_threshold_km. "
                    f"Row {i} has {alt_km!r} after {prev_alt!r}."
                )
            prev_alt = alt_km


@dataclass(frozen=True, slots=True, kw_only=True)
class GravityConfig:
    """
    Central gravity model configuration.

    Notes
    -----
    - ``backend="classic_sh"`` uses the spherical-harmonics coefficient file in
      ``file_path`` and optional adaptive-degree rules.
    - ``backend="st_lrps"`` uses a trained neural surrogate stored in
      ``st_lrps_model_dir``.
    - ``degree`` is the maximum SH degree (N_max). If None, the SH loader may
      choose a default.
    - If adaptive degree is enabled, runtime degree is selected <= ``degree``
      (if provided).
    """
    file_path: str
    degree: Optional[int] = None
    use_mmap: bool = True
    backend: str = "classic_sh"
    st_lrps_model_dir: str = ""
    adaptive: AdaptiveDegreeConfig = field(default_factory=AdaptiveDegreeConfig)

    def __post_init__(self) -> None:
        path = self.file_path.strip() if self.file_path else ""
        backend = str(self.backend).strip().lower()
        if backend not in {"classic_sh", "st_lrps"}:
            raise ValueError(
                "GravityConfig.backend must be 'classic_sh' or 'st_lrps'. "
                f"Got {self.backend!r}"
            )

        if backend == "classic_sh" and not path:
            raise ValueError("GravityConfig.file_path cannot be empty when backend='classic_sh'.")

        surrogate_dir = self.st_lrps_model_dir.strip() if self.st_lrps_model_dir else ""
        if backend == "st_lrps" and not surrogate_dir:
            raise ValueError(
                "GravityConfig.st_lrps_model_dir cannot be empty when backend='st_lrps'."
            )

        if self.degree is not None and self.degree < 0:
            raise ValueError(f"GravityConfig.degree must be >= 0, got {self.degree!r}")

        # If a max degree is specified, ensure adaptive policy cannot request more than that.
        if self.degree is not None and self.adaptive.min_degree > self.degree:
            raise ValueError(
                f"adaptive.min_degree ({self.adaptive.min_degree}) cannot exceed "
                f"GravityConfig.degree ({self.degree})."
            )

    @property
    def uses_st_lrps(self) -> bool:
        """True when the central gravity backend is the neural surrogate."""

        return str(self.backend).strip().lower() == "st_lrps"



# =============================================================================
# 4.                           FORCE FLAGS
# =============================================================================

@dataclass(frozen=True, slots=True, kw_only=True)
class PerturbationFlags:
    """Enable/disable perturbation models used in the equations of motion."""

    # Gravity field
    enable_sh: bool = True

    # Ephemeris-dependent (third body / Earth gravity)
    enable_3rd_body_sun: bool = False
    enable_3rd_body_earth: bool = False
    enable_earth_j2: bool = False

    # Non-gravitational
    enable_srp: bool = False
    enable_albedo: bool = False
    enable_thermal: bool = False

    # High-fidelity options
    enable_tides_k2: bool = False
    enable_tides_k3: bool = False
    enable_relativity_1pn: bool = False

    def __post_init__(self) -> None:
        if self.enable_tides_k3 and not self.enable_tides_k2:
            raise ValueError("enable_tides_k3=True requires enable_tides_k2=True.")

    @property
    def enable_third_body(self) -> bool:
        return self.enable_3rd_body_sun or self.enable_3rd_body_earth or self.enable_earth_j2

    @property
    def enable_surface_forces(self) -> bool:
        return self.enable_albedo or self.enable_thermal

    @property
    def enable_tides(self) -> bool:
        return self.enable_tides_k2 or self.enable_tides_k3

    @property
    def tides_degree(self) -> int:
        if self.enable_tides_k3:
            return 3
        if self.enable_tides_k2:
            return 2
        return 0

    @property
    def tides_kind(self) -> str:
        return {0: "none", 2: "k2", 3: "k3"}[self.tides_degree]



# =============================================================================
# 5.                                   TIME 
# =============================================================================

@dataclass(frozen=True, slots=True, kw_only=True)
class TimeConfig:
    """
    Time span and output sampling grid.

    Ownership
    ---------
    This is the single owner of:
    - simulation duration
    - output sampling resolution (t_eval grid density)

    Conventions
    -----------
    - `duration_s` is the physical span of the propagation (seconds).
    - `output_dt_s` controls the spacing of *reported* samples, not internal solver micro-steps.
    - If `output_dt_s` is None, the engine may compute an output spacing from `samples_per_period`.

    Notes
    -----
    - This config is intentionally strict: it validates obvious misconfiguration early.
    - `start_date` is expected to be ISO-like.
    - Entry points normalize explicit offsets to UTC before ephemeris sampling.
    - Naive timestamps are interpreted as UTC for backward compatibility.
    """
    start_date: str = "2025-01-01T00:00:00Z"

    # Total propagation span [s]
    duration_s: float = DAY_S

    # Output sampling: either fixed dt or derived from orbital period.
    output_dt_s: Optional[float] = 60.0
    samples_per_period: int = 120

    # Soft cap to prevent unbounded output arrays for long runs.
    max_points_cap: int = 200_000

    # Time origin passed to RHS/events (often zero).
    t0_s: float = 0.0

    def __post_init__(self) -> None:
        # Defensive checks for obvious misconfiguration.
        if self.duration_s <= 0.0:
            raise ValueError(f"duration_s must be positive. Got {self.duration_s!r}")

        if self.output_dt_s is not None and self.output_dt_s <= 0.0:
            raise ValueError(f"output_dt_s must be positive if set. Got {self.output_dt_s!r}")

        if self.samples_per_period < 2:
            raise ValueError(f"samples_per_period must be >= 2. Got {self.samples_per_period!r}")

        if self.max_points_cap < 10:
            raise ValueError(f"max_points_cap too small. Got {self.max_points_cap!r}")

        # Strict contract: avoid ambiguous/meaningless derived sampling.
        if self.output_dt_s is None and self.samples_per_period < 2:
            raise ValueError(
                "output_dt_s is None requires samples_per_period >= 2 "
                "(used to derive an output grid from orbital period)."
            )

        # Strict safety: if output_dt_s is explicit, enforce the cap at config time.
        if self.output_dt_s is not None:
            # +1 to include both endpoints if you build t_eval as [t0, t0+dt, ..., t0+duration]
            est_points = int(self.duration_s / self.output_dt_s) + 1
            if est_points > self.max_points_cap:
                raise ValueError(
                    f"Requested output grid has ~{est_points} points, exceeds max_points_cap={self.max_points_cap}. "
                    "Increase output_dt_s, reduce duration_s, or raise max_points_cap."
                )

    @property
    def duration_days(self) -> float:
        """Convenience conversion: seconds -> days."""
        return self.duration_s / DAY_S



# =============================================================================
# 6.                            STATE DEFINITIONS
# =============================================================================

@dataclass(frozen=True, slots=True, kw_only=True)
class InitialState:
    """
    Initial Cartesian state vector (inertial frame).

    Units
    -----
    Position: meters [m]
    Velocity: meters per second [m/s]
    """
    x: float
    y: float
    z: float
    vx: float
    vy: float
    vz: float

    def to_array(self) -> StateVector:
        """
        Return a 6-element state array: [x, y, z, vx, vy, vz].

        Notes
        -----
        - Always returns a new float64 array.
        - Keep this out of hot loops if performance is critical.
        """
        return np.asarray((self.x, self.y, self.z, self.vx, self.vy, self.vz), dtype=np.float64)

    def r_vec(self) -> Vec3:
        """Return position vector [m] as a float64 array of shape (3,)."""
        return np.asarray((self.x, self.y, self.z), dtype=np.float64)

    def v_vec(self) -> Vec3:
        """Return velocity vector [m/s] as a float64 array of shape (3,)."""
        return np.asarray((self.vx, self.vy, self.vz), dtype=np.float64)



# =============================================================================
# 7.                                    EVENTS
# =============================================================================

@dataclass(frozen=True, slots=True, kw_only=True)
class EventConfig:
    """
    Event-related configuration (stopping/logging rules).

    This config describes WHAT to detect (impact, peri/apo crossings, eclipse),
    but it does not implement event functions itself.
    """
    detect_impact: bool = True
    impact_alt_km: float = 0.0

    # Non-terminal bookkeeping events
    enable_peri_apo_events: bool = True

    # Optional logging/guards
    detect_eclipse: bool = False
    stop_on_capture: bool = False

    def __post_init__(self) -> None:
        if self.impact_alt_km < 0.0:
            raise ValueError(f"impact_alt_km must be >= 0, got {self.impact_alt_km!r}")
        if not self.detect_impact and self.impact_alt_km != 0.0:
            raise ValueError(
                "impact_alt_km is only meaningful when detect_impact=True "
                f"(got detect_impact=False with impact_alt_km={self.impact_alt_km!r})."
            )



# =============================================================================
# 8.                       PROPAGATOR / SOLVER CONFIG
# =============================================================================

@dataclass(frozen=True, slots=True, kw_only=True)
class PropagatorConfig:
    """Numerical propagation configuration (SciPy solve_ivp + optional fixed-step methods)."""

    # Solver (SciPy)
    method: str = "DOP853"
    rtol: float = 1e-10
    atol: float = 1e-12

    # Fixed-step / symplectic (if used by your engine)
    symplectic_default: str = "YOSHIDA4"

    # Internal step-size logic
    use_nyquist_max_step: bool = True
    user_max_step_s: Optional[float] = None
    nyquist_safety_div: float = 4.0
    nyquist_v_margin: float = 1.10

    # Safety guard
    max_internal_steps: int = 1_000_000

    # Runtime / logging
    verbose: bool = True
    heartbeat_hours: float = 6.0

    # Cooperative stop
    stop_file: Optional[str] = None
    stop_event_in_scipy: bool = True

    # Chunking / checkpointing
    chunk_s: Optional[float] = None
    checkpoint_path: Optional[str] = None
    checkpoint_every_chunk: bool = False

    # Baseline run (e.g., 2-body comparison)
    compute_2body_baseline: bool = True
    baseline_rtol: float = 1e-12
    baseline_atol: float = 1e-14

    # Hybrid impact switch options (only if impact detection is enabled)
    hybrid_switch_alt_m: float = 11_000.0
    hybrid_kind: str = "bilinear"

    # Events
    events: EventConfig = field(default_factory=EventConfig)

    def __post_init__(self) -> None:
        if self.rtol <= 0.0 or self.atol <= 0.0:
            raise ValueError(f"rtol/atol must be > 0 (rtol={self.rtol!r}, atol={self.atol!r})")

        if self.nyquist_safety_div <= 0.0:
            raise ValueError(f"nyquist_safety_div must be > 0, got {self.nyquist_safety_div!r}")

        if self.nyquist_v_margin <= 0.0:
            raise ValueError(f"nyquist_v_margin must be > 0, got {self.nyquist_v_margin!r}")

        if self.user_max_step_s is not None and self.user_max_step_s <= 0.0:
            raise ValueError(f"user_max_step_s must be > 0, got {self.user_max_step_s!r}")

        if self.max_internal_steps < 100:
            raise ValueError(f"max_internal_steps too small, got {self.max_internal_steps!r}")

        if self.heartbeat_hours <= 0.0:
            raise ValueError(f"heartbeat_hours must be > 0, got {self.heartbeat_hours!r}")

        if self.chunk_s is not None and self.chunk_s <= 0.0:
            raise ValueError(f"chunk_s must be > 0 if set, got {self.chunk_s!r}")

        if self.compute_2body_baseline and (self.baseline_rtol <= 0.0 or self.baseline_atol <= 0.0):
            raise ValueError(
                "baseline_rtol/baseline_atol must be > 0 when compute_2body_baseline=True "
                f"(baseline_rtol={self.baseline_rtol!r}, baseline_atol={self.baseline_atol!r})"
            )

        if self.hybrid_switch_alt_m < 0.0:
            raise ValueError(f"hybrid_switch_alt_m must be >= 0, got {self.hybrid_switch_alt_m!r}")



# =============================================================================
# 9.                          RESULT CONTAINERS
# =============================================================================

@dataclass(slots=True)
class SimulationHistory:
    """
    Processed simulation data optimized for analysis/plotting/reporting.

    This is a convenience container with unit conversions applied:
    - time: seconds -> days
    - position: meters -> km
    - velocity: m/s -> km/s
    - altitude: meters -> km
    """
    t_days: F64Array
    pos_km: F64Array
    vel_km_s: F64Array
    alt_km: F64Array

    coe: Optional[Dict[str, F64Array]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Normalize dtype + contiguity for stable downstream behavior.
        self.t_days = np.ascontiguousarray(self.t_days, dtype=np.float64)
        self.pos_km = np.ascontiguousarray(self.pos_km, dtype=np.float64)
        self.vel_km_s = np.ascontiguousarray(self.vel_km_s, dtype=np.float64)
        self.alt_km = np.ascontiguousarray(self.alt_km, dtype=np.float64)

        # Defensive copies for mutable containers.
        self.metadata = dict(self.metadata)
        if self.coe is not None:
            self.coe = dict(self.coe)

        # Shape checks.
        if self.t_days.ndim != 1:
            raise ValueError(f"t_days must be 1D, got shape={self.t_days.shape!r}")

        n = int(self.t_days.shape[0])

        if self.alt_km.ndim != 1:
            raise ValueError(f"alt_km must be 1D, got shape={self.alt_km.shape!r}")
        if int(self.alt_km.shape[0]) != n:
            raise ValueError(f"Mismatch: t_days={n}, alt_km={int(self.alt_km.shape[0])}")

        if self.pos_km.ndim != 2 or int(self.pos_km.shape[1]) != 3:
            raise ValueError(f"pos_km must be (N, 3), got shape={self.pos_km.shape!r}")
        if int(self.pos_km.shape[0]) != n:
            raise ValueError(f"Mismatch: t_days={n}, pos_km={int(self.pos_km.shape[0])}")

        if self.vel_km_s.ndim != 2 or int(self.vel_km_s.shape[1]) != 3:
            raise ValueError(f"vel_km_s must be (N, 3), got shape={self.vel_km_s.shape!r}")
        if int(self.vel_km_s.shape[0]) != n:
            raise ValueError(f"Mismatch: t_days={n}, vel_km_s={int(self.vel_km_s.shape[0])}")


@dataclass(slots=True)
class PropagationResult:
    """
    Raw propagation result container (row-major).

    Conventions
    -----------
    - t: shape (N,)
    - y: shape (N, n_state) in row-major order
    - Use `y_col` for SciPy-style column-major view: shape (n_state, N)

    Notes
    -----
    - n_state is expected to be >= 6 (r,v); additional states are allowed.
    """
    t: F64Array
    y: F64Array
    ode: Any = None

    # Event outputs (mirrors SciPy solve_ivp format: one array per event function)
    t_events: List[F64Array] = field(default_factory=list)
    y_events: List[F64Array] = field(default_factory=list)

    # Optional impact bookkeeping
    impacted: bool = False
    t_impact_s: Optional[float] = None
    y_impact: Optional[F64Array] = None

    # Optional early stop bookkeeping
    stopped_early: bool = False
    stop_reason: Optional[str] = None
    t_stop_s: Optional[float] = None

    # Arbitrary diagnostics (step counts, wall time, error metrics, etc.)
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    # Optional baseline run output (e.g., 2-body reference)
    baseline: Optional["PropagationResult"] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        # Normalize main arrays.
        self.t = np.ascontiguousarray(self.t, dtype=np.float64)
        self.y = np.ascontiguousarray(self.y, dtype=np.float64)

        # Normalize event arrays.
        self.t_events = [np.asarray(te, dtype=np.float64) for te in self.t_events]
        self.y_events = [np.asarray(ye, dtype=np.float64) for ye in self.y_events]

        # Defensive copy for mutable dict.
        self.diagnostics = dict(self.diagnostics)

        # Shape checks.
        if self.t.ndim != 1:
            raise ValueError(f"t must be 1D, got shape={self.t.shape!r}")
        if self.y.ndim != 2:
            raise ValueError(f"y must be 2D (N, n_state), got shape={self.y.shape!r}")
        if int(self.y.shape[0]) != int(self.t.shape[0]):
            raise ValueError(f"Length mismatch: t={int(self.t.shape[0])}, y rows={int(self.y.shape[0])}")
        if int(self.y.shape[1]) < 6:
            raise ValueError(f"State must have at least 6 elements, got n_state={int(self.y.shape[1])}")

        # Optional impact state normalization.
        if self.y_impact is not None:
            self.y_impact = np.asarray(self.y_impact, dtype=np.float64)

    @property
    def y_col(self) -> F64Array:
        """Column-major view (n_state, N), matching SciPy's `sol.y` convention."""
        return self.y.T

    def to_history(
        self,
        *,
        r_ref_m: float = R_MOON_MEAN,
        coe: Optional[Dict[str, F64Array]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SimulationHistory:
        """
        Convert raw propagation output to a plotting-friendly history object.

        This method intentionally stays lightweight:
        - converts units (m->km, s->days)
        - computes altitude as ||r|| - r_ref
        - does NOT compute orbital elements unless provided by caller
        """
        pos_m = self.y[:, 0:3]
        vel_m_s = self.y[:, 3:6]

        t_days = self.t / DAY_S
        pos_km = pos_m / 1000.0
        vel_km_s = vel_m_s / 1000.0

        alt_km = (np.linalg.norm(pos_m, axis=1) - float(r_ref_m)) / 1000.0

        md = {} if metadata is None else dict(metadata)

        return SimulationHistory(
            t_days=t_days,
            pos_km=pos_km,
            vel_km_s=vel_km_s,
            alt_km=alt_km,
            coe=coe,
            metadata=md,
        )



# =============================================================================
# 10.                             PUBLIC API
# =============================================================================

__all__ = [

    # Type aliases (NumPy typing)
    "F64",                    # np.float64 dtype alias
    "F64Array",               # Generic float64 NumPy array (any shape)

    # Semantic array aliases (shape by convention)
    "Arr1",                   # 1D float64 array (generic)
    "Arr2",                   # 2D float64 array (generic)

    # Domain aliases (shape by convention)
    "Vec3",                   # 3-vector: (3,) position/velocity/accel/thrust, etc.
    "Quat",                   # Quaternion: (4,) [w, x, y, z] (scalar-first)
    "StateVector",            # State: (6,) [rx, ry, rz, vx, vy, vz]
    "Matrix3x3",              # 3x3 matrix: (3, 3) DCM/rotation/inertia

    # Physical properties
    "SpacecraftProps",        # Spacecraft mass/area/aero coefficients + derived BC

    # Gravity configuration 
    "GravityConfig",          # SH model file path + max degree + runtime options
    "AdaptiveDegreeConfig",   # Policy for runtime SH degree selection


    # Perturbation enable/disable flags
    "PerturbationFlags",      # Toggles: SH, 3rd bodies, SRP, albedo/thermal, tides, etc.


    # Time span and output grid
    "TimeConfig",             # Start time + duration + output sampling rules


    # Initial state (Cartesian inertial)
    "InitialState",          # x,y,z,vx,vy,vz with convenience array helpers


    # Event configuration (stop/log rules)
    "EventConfig",           # Impact/peri-apo/eclispe/capture detection toggles


    # Propagation / solver configuration
    "PropagatorConfig",      # Integrator settings, tolerances, step limits, logging


    # Results / data containers
    "PropagationResult",     # Raw solver output (t, y) + events + diagnostics
    "SimulationHistory",     # Post-processed output for plotting/reporting (km, days)
]
