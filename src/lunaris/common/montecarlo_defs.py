# lunaris/common/montecarlo_defs.py
"""
Monte Carlo Simulation Configuration Definitions
=================================================

Configuration dataclasses for MC uncertainty propagation.  All types follow
the project SSOT pattern: frozen dataclasses with __post_init__ validation.

Layers
------
- ``StateUncertainty``      : position/velocity covariance model.
- ``SpacecraftUncertainty`` : mass / Cd / Cr / area perturbations.
- ``MonteCarloConfig``      : top-level MC run configuration.
- ``MCRunResult``           : output container for ensemble trajectories.

Units
-----
- Positions : meters  [m]
- Velocities: m/s
- Mass      : kg
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from .type_defs import F64Array


# =============================================================================
# 1.                       STATE UNCERTAINTY
# =============================================================================

@dataclass(frozen=True, slots=True, kw_only=True)
class StateUncertainty:
    """
    Initial Cartesian state uncertainty model (position + velocity).

    Two modes
    ---------
    Diagonal (default):
        Isotropic 1-sigma for position (``sigma_r_m``) and velocity
        (``sigma_v_m_s``).  Covariance = diag(σ_r², σ_r², σ_r², σ_v², σ_v², σ_v²).

    Full covariance:
        Provide ``covariance_6x6`` to override the diagonal.  The matrix must
        be symmetric positive-semi-definite (6×6 float64).

    Notes
    -----
    - ``to_covariance()`` always returns a 6×6 matrix regardless of mode.
    - Cholesky factor is available via ``cholesky_factor()`` for sampling.
    """
    sigma_r_m: float = 1_000.0      # Position 1-sigma [m]
    sigma_v_m_s: float = 1.0        # Velocity 1-sigma [m/s]
    covariance_6x6: Optional[F64Array] = None  # Overrides diagonal if set

    def __post_init__(self) -> None:
        if self.sigma_r_m < 0.0:
            raise ValueError(f"sigma_r_m must be >= 0, got {self.sigma_r_m}")
        if self.sigma_v_m_s < 0.0:
            raise ValueError(f"sigma_v_m_s must be >= 0, got {self.sigma_v_m_s}")

        if self.covariance_6x6 is not None:
            cov = np.asarray(self.covariance_6x6, dtype=np.float64)
            if cov.ndim != 2 or cov.shape != (6, 6):
                raise ValueError(
                    f"covariance_6x6 must be shape (6, 6), got {cov.shape}"
                )
            if not np.allclose(cov, cov.T, atol=1e-10):
                raise ValueError("covariance_6x6 must be symmetric.")
            eigvals = np.linalg.eigvalsh(cov)
            if np.any(eigvals < -1e-10 * max(1.0, np.abs(eigvals).max())):
                raise ValueError("covariance_6x6 must be positive semi-definite.")

    def to_covariance(self) -> F64Array:
        """Return the 6×6 covariance matrix (always float64, C-contiguous)."""
        if self.covariance_6x6 is not None:
            return np.ascontiguousarray(self.covariance_6x6, dtype=np.float64)
        cov = np.zeros((6, 6), dtype=np.float64)
        sr2 = self.sigma_r_m ** 2
        sv2 = self.sigma_v_m_s ** 2
        for i in range(3):
            cov[i, i] = sr2
        for i in range(3, 6):
            cov[i, i] = sv2
        return cov

    def cholesky_factor(self) -> F64Array:
        """
        Lower-triangular Cholesky factor L such that P = L @ L.T.

        Used for efficient Gaussian sampling: x = mean + L @ z,  z ~ N(0,I).
        """
        P = self.to_covariance()
        # Regularise very small diagonal entries to avoid near-singular Cholesky.
        P = P + np.eye(6, dtype=np.float64) * 1e-30
        return np.linalg.cholesky(P)

    @property
    def sigma_summary(self) -> str:
        """Human-readable 1-sigma summary."""
        return (
            f"Δr={self.sigma_r_m / 1_000:.3f} km, "
            f"Δv={self.sigma_v_m_s:.3f} m/s"
        )


# =============================================================================
# 2.                    SPACECRAFT UNCERTAINTY
# =============================================================================

@dataclass(frozen=True, slots=True, kw_only=True)
class SpacecraftUncertainty:
    """
    1-sigma uncertainties for spacecraft physical properties.

    Zero sigma means the parameter is treated as deterministic (no perturbation).
    Truncated-normal sampling ensures physical positivity (mass > 0, Cd > 0, etc.)
    at engine level.
    """
    sigma_mass_kg: float = 0.0      # [kg]
    sigma_cd: float = 0.0           # dimensionless drag coefficient
    sigma_cr: float = 0.0           # dimensionless SRP reflectivity
    sigma_area_m2: float = 0.0      # [m²]

    def __post_init__(self) -> None:
        for attr in ("sigma_mass_kg", "sigma_cd", "sigma_cr", "sigma_area_m2"):
            v = getattr(self, attr)
            if v < 0.0:
                raise ValueError(f"SpacecraftUncertainty.{attr} must be >= 0, got {v}")

    @property
    def any_active(self) -> bool:
        """True if at least one parameter has non-zero uncertainty."""
        return any(
            v > 0.0
            for v in (self.sigma_mass_kg, self.sigma_cd, self.sigma_cr, self.sigma_area_m2)
        )


# =============================================================================
# 3.                       MONTE CARLO CONFIG
# =============================================================================

def validate_st_lrps_model_dir(path: str | Path) -> Path:
    """
    Verify that the given path is a valid ST-LRPS model directory.
    It must exist, contain config.json, and have a valid checkpoint.
    """
    st_lrps_path = Path(path).expanduser().resolve()
    if not st_lrps_path.is_dir():
        raise ValueError(f"st_lrps_model_dir must point to an existing trained ST-LRPS run directory, got {str(path)!r}")
    if not (st_lrps_path / "config.json").is_file():
        raise ValueError(f"st_lrps_model_dir must contain config.json from a trained ST-LRPS run, got {str(path)!r}")
    ckpt_dir = st_lrps_path / "checkpoints"
    has_best_or_last = (
        (ckpt_dir / "ckpt_best.pt").is_file()
        or (ckpt_dir / "ckpt_last.pt").is_file()
    )
    if not has_best_or_last:
        raise ValueError(f"st_lrps_model_dir must contain checkpoints/ckpt_best.pt or checkpoints/ckpt_last.pt, got {str(path)!r}")
    return st_lrps_path


@dataclass(frozen=True, slots=True, kw_only=True)
class MonteCarloConfig:
    """
    Top-level Monte Carlo simulation configuration.

    Routing
    -------
    ``use_gpu=True``  → GPU CUDA RK4 batch propagator (low/medium-fidelity physics).
    ``use_gpu=False`` → CPU multiprocessing via existing :func:`core.propagator.propagate`
                        (full-fidelity physics, slower for large N).

    GPU physics model
    -----------------
    - Point-mass + SH gravity up to ``gpu_sh_degree`` (≤ 24 supported by the GPU kernel).
    - Third-body Sun / Earth (if enabled in SimConfig flags).
    - SRP (if enabled in SimConfig flags).
    - 1PN relativity (if enabled in SimConfig flags).
    - Albedo / thermal / tides: CPU-only (not available on GPU path).

    Output
    ------
    Snapshots at ``output_dt_s`` intervals are written to disk in HDF5 or NPZ
    format to avoid VRAM / RAM exhaustion for large N or long durations.

    Validation
    ----------
    ``gpu_sh_degree`` is capped at 24 because the GPU kernel uses compile-time
    fixed workspace arrays sized for degree 24 (26×26 per thread).
    """
    # Ensemble
    n_samples: int = 1_000
    seed: int = 42

    # Uncertainty models
    state: StateUncertainty = field(default_factory=StateUncertainty)
    spacecraft: SpacecraftUncertainty = field(default_factory=SpacecraftUncertainty)

    # Backend selection
    use_gpu: bool = True
    gpu_device_id: int = 0
    gravity_mode_override: str = "follow_mission"
    st_lrps_model_dir: Optional[str] = None

    # GPU physics fidelity
    gpu_sh_degree: int = 10         # SH degree evaluated per-thread on GPU (0 = PM only)
    gpu_threads_per_block: int = 128

    # Fixed-step RK4 integration (GPU path)
    dt_s: float = 60.0              # RK4 step [s]

    # Output
    output_format: str = "hdf5"     # "hdf5" or "npz"
    output_path: str = "outputs/monte_carlo/mc_output.h5"
    max_vram_gb: float = 4.0        # VRAM budget (caps batch size automatically)

    # Statistical analysis
    compute_impact_probability: bool = True
    impact_alt_km: float = 0.0      # Impact detection threshold [km]
    sigma_levels: Tuple[float, ...] = (1.0, 2.0, 3.0)

    def __post_init__(self) -> None:
        if self.n_samples < 2:
            raise ValueError(f"n_samples must be >= 2, got {self.n_samples}")
        if self.dt_s <= 0.0:
            raise ValueError(f"dt_s must be > 0, got {self.dt_s}")
        if self.gravity_mode_override not in ("follow_mission", "classic_sh", "st_lrps"):
            raise ValueError(
                "gravity_mode_override must be one of: "
                "'follow_mission', 'classic_sh', 'st_lrps'. "
                f"Got {self.gravity_mode_override!r}"
            )
        st_lrps_model_dir = str(self.st_lrps_model_dir or "").strip()
        if self.gravity_mode_override == "st_lrps" and not st_lrps_model_dir:
            raise ValueError("st_lrps_model_dir cannot be empty when gravity_mode_override='st_lrps'.")
        if not (0 <= self.gpu_sh_degree <= 24):
            raise ValueError(
                f"gpu_sh_degree must be in [0, 24] (GPU kernel limit), "
                f"got {self.gpu_sh_degree}."
            )
        if not (32 <= self.gpu_threads_per_block <= 1024):
            raise ValueError(
                f"gpu_threads_per_block must be in [32, 1024], "
                f"got {self.gpu_threads_per_block}"
            )
        if self.output_format not in ("hdf5", "npz"):
            raise ValueError(
                f"output_format must be 'hdf5' or 'npz', got {self.output_format!r}"
            )
        if self.max_vram_gb <= 0.0:
            raise ValueError(f"max_vram_gb must be > 0, got {self.max_vram_gb}")
        if self.impact_alt_km < 0.0:
            raise ValueError(f"impact_alt_km must be >= 0, got {self.impact_alt_km}")

    @property
    def output_path_resolved(self) -> Path:
        return Path(self.output_path).expanduser().resolve()

    def effective_max_batch(self, state_bytes_per_sample: int = 96) -> int:
        """
        Maximum batch size that fits in ``max_vram_gb`` VRAM.

        Conservative: reserves 20 % headroom for GPU kernel overheads.
        The default estimate includes the propagated state vector plus the
        per-sample spacecraft properties and impact bookkeeping that the GPU
        backend also keeps resident on device memory.
        """
        budget = self.max_vram_gb * 1e9 * 0.80
        return max(1, int(budget / max(1, state_bytes_per_sample)))


# =============================================================================
# 4.                          RESULT CONTAINERS
# =============================================================================

@dataclass(slots=True)
class MCRunResult:
    """
    Ensemble simulation output.

    Shape conventions
    -----------------
    - ``t``  : (T,)       — output time grid [s]
    - ``Y``  : (T, N, 6)  — state ensemble [m, m/s]
    - ``sc_samples`` : (N, 4)  — sampled [mass_kg, cd, cr, area_m2] per run

    ``Y[k, i, :]`` is the state of sample ``i`` at time step ``k``.

    Notes
    -----
    - Only samples that did not impact are kept for full duration.
    - ``impact_mask[i]`` is True if sample ``i`` impacted before ``t[-1]``.
    - ``t_impact[i]`` is the impact time (NaN if no impact).
    """
    t: F64Array                              # (T,)
    Y: F64Array                              # (T, N, 6)
    sc_samples: F64Array                     # (N, 4) [mass_kg, area_m2, cd, cr]
    impact_mask: F64Array                    # (N,) bool-like float64 (0/1)
    t_impact: F64Array                       # (N,) NaN if no impact
    diagnostics: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.t = np.ascontiguousarray(self.t, dtype=np.float64)
        self.Y = np.ascontiguousarray(self.Y, dtype=np.float64)
        self.sc_samples = np.ascontiguousarray(self.sc_samples, dtype=np.float64)
        self.impact_mask = np.ascontiguousarray(self.impact_mask, dtype=np.float64)
        self.t_impact = np.ascontiguousarray(self.t_impact, dtype=np.float64)

        n_t = self.t.shape[0]
        n_samp = self.Y.shape[1] if self.Y.ndim == 3 else 0

        if self.Y.ndim != 3 or self.Y.shape[2] != 6:
            raise ValueError(
                f"Y must be (T, N, 6), got {self.Y.shape}"
            )
        if self.Y.shape[0] != n_t:
            raise ValueError(
                f"Y.shape[0]={self.Y.shape[0]} != len(t)={n_t}"
            )
        if self.sc_samples.shape != (n_samp, 4):
            raise ValueError(
                f"sc_samples must be (N, 4), got {self.sc_samples.shape}"
            )
        if self.impact_mask.shape != (n_samp,):
            raise ValueError(
                f"impact_mask must be (N,), got {self.impact_mask.shape}"
            )
        if self.t_impact.shape != (n_samp,):
            raise ValueError(
                f"t_impact must be (N,), got {self.t_impact.shape}"
            )

    @property
    def n_samples(self) -> int:
        return int(self.Y.shape[1])

    @property
    def n_steps(self) -> int:
        return int(self.t.shape[0])

    @property
    def n_survived(self) -> int:
        """Number of samples that completed without impact."""
        return int(np.sum(self.impact_mask == 0))

    @property
    def impact_fraction(self) -> float:
        """Fraction of samples that impacted the surface."""
        return float(np.mean(self.impact_mask > 0.5))

    def survived_trajectories(self) -> F64Array:
        """Return Y[:, mask, :] for non-impacting samples only."""
        mask = self.impact_mask < 0.5
        return self.Y[:, mask, :]


# =============================================================================
# 5.                         PUBLIC API
# =============================================================================

__all__ = [
    "StateUncertainty",
    "SpacecraftUncertainty",
    "MonteCarloConfig",
    "MCRunResult",
    "validate_st_lrps_model_dir",
]
