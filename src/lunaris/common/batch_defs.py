# lunaris/common/batch_defs.py
"""
Batch Propagation Configuration Definitions
===========================================

Configuration dataclasses for batch/ensemble injection-dispersion propagation.
Independent random sampling is one sampling design; Latin-hypercube and Sobol
designs are also available for validation-oriented batch propagation. All types follow
the project SSOT pattern: frozen dataclasses with __post_init__ validation.

Layers
------
- ``StateUncertainty``      : position/velocity covariance model.
- ``SpacecraftUncertainty`` : mass / Cd / Cr / area perturbations.
- ``BatchPropagationConfig``: top-level batch/ensemble run configuration.
- ``BatchPropagationResult``: output container for ensemble trajectories.

Units
-----
- Positions : meters  [m]
- Velocities: m/s
- Mass      : kg
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, get_args

import numpy as np

from .time_grid_contract import build_output_time_grid
from .type_defs import F64Array

SamplingMethod = Literal["random", "lhs", "sobol", "sobol_scrambled"]
BatchBackend = Literal[
    "auto",
    "cpu_sh",
    "numba_cuda_sh",
    "torch_cuda_sh",
    "torch_cpu_sh",
    "gpu_st_lrps_potential",
    "gpu_st_lrps_third_body",
]
GravityModeOverride = Literal["follow_mission", "classic_sh", "st_lrps"]
ShFallbackPolicy = Literal["compatible_gpu", "cpu", "error"]
TorchDtype = Literal["float32", "float64"]
OutputFormat = Literal["hdf5", "npz"]
ResultStorageMode = Literal["auto", "memory", "disk"]
BatchOutputMode = Literal["full", "summary_only"]
ImpactSurfaceMode = Literal["sphere", "terrain"]

BATCH_SAMPLING_METHODS: tuple[str, ...] = tuple(get_args(SamplingMethod))
BATCH_BACKENDS: tuple[str, ...] = tuple(get_args(BatchBackend))
GRAVITY_MODE_OVERRIDES: tuple[str, ...] = tuple(get_args(GravityModeOverride))
SH_FALLBACK_POLICIES: tuple[str, ...] = tuple(get_args(ShFallbackPolicy))
TORCH_DTYPES: tuple[str, ...] = tuple(get_args(TorchDtype))
OUTPUT_FORMATS: tuple[str, ...] = tuple(get_args(OutputFormat))
RESULT_STORAGE_MODES: tuple[str, ...] = tuple(get_args(ResultStorageMode))
BATCH_OUTPUT_MODES: tuple[str, ...] = tuple(get_args(BatchOutputMode))
IMPACT_SURFACE_MODES: tuple[str, ...] = tuple(get_args(ImpactSurfaceMode))
ST_LRPS_BATCH_BACKENDS: tuple[str, ...] = tuple(
    backend for backend in BATCH_BACKENDS if backend.startswith("gpu_st_lrps_")
)


def _choice_list(choices: tuple[str, ...]) -> str:
    return ", ".join(repr(choice) for choice in choices)


# =============================================================================
# 0.                    SHARED BATCH OUTPUT GRID
# =============================================================================


class TrajectoryArrayLike(Protocol):
    """Minimal eager/lazy trajectory surface consumed by batch result logic."""

    shape: tuple[int, ...]
    ndim: int
    dtype: np.dtype[Any]

    def __getitem__(self, key: Any) -> np.ndarray:
        ...

    def __array__(self, dtype: Any = None, copy: bool | None = None) -> np.ndarray:
        ...





def build_batch_output_grid(duration_s: float, output_dt_s: float) -> tuple[F64Array, int, float]:
    """Single source of truth for the batch/ensemble output time grid.

    Every batch backend (CPU DOP853, Numba CUDA classic SH, torch_cuda_sh /
    torch_cpu_sh, and the ST-LRPS torch path) must emit its snapshots on THIS
    grid so that ``Y[k]`` means the same epoch across backends and trajectories
    are index-comparable for benchmarking and error metrics.

    Contract
    --------
    - ``t[0]  == 0.0``                 (the initial state is always snapshot 0)
    - ``t[-1] == duration_s``          (exact final epoch; never an overshoot)
    - ``np.all(np.diff(t) > 0.0)``     (strictly increasing, uniform spacing)
    - ``len(t) == n_snaps + 1``

    Parameters
    ----------
    duration_s : total propagation span [s] (> 0).
    output_dt_s : maximum requested snapshot interval [s] (> 0). The realized
        interval ``snap_interval_s = duration_s / ceil(duration_s/output_dt_s)``
        never exceeds it, while the last sample lands exactly on ``duration_s``.

    Returns
    -------
    (t_out, n_snaps, snap_interval_s)
        ``t_out`` is the ``(n_snaps + 1,)`` grid; ``snap_interval_s`` is the
        realized spacing. Fixed-step backends derive their stepping from it:
        ``steps_per_snap = max(1, round(snap_interval_s / dt))`` and
        ``dt_eff = snap_interval_s / steps_per_snap`` so steps align with grid
        points exactly.
    """
    return build_output_time_grid(duration_s, output_dt_s)


def build_fixed_step_grid_metadata(
    duration_s: float,
    output_dt_s: float,
    dt_s: float,
) -> dict[str, float | int]:
    """Return provenance for fixed-step RK4 runs on the shared output grid.

    Fixed-step backends intentionally adjust the effective substep so each
    snapshot lands exactly on the canonical output grid. Keep the requested and
    realized values together so benchmark reports cannot confuse ``dt_s`` with
    the step that was actually integrated.
    """

    _t_out, n_snaps, snap_interval = build_batch_output_grid(duration_s, output_dt_s)
    requested_dt = float(dt_s)
    if requested_dt <= 0.0:
        raise ValueError(f"dt_s must be > 0, got {requested_dt!r}")
    steps_per_snapshot = max(1, int(round(snap_interval / requested_dt)))
    effective_dt = snap_interval / steps_per_snapshot
    return {
        "requested_dt_s": requested_dt,
        "effective_dt_s": float(effective_dt),
        "steps_per_snapshot": int(steps_per_snapshot),
        "requested_output_dt_s": float(output_dt_s),
        "effective_output_dt_s": float(snap_interval),
        "n_output_snapshots": int(n_snaps + 1),
    }


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
    covariance_6x6: F64Array | None = None  # Overrides diagonal if set

    def __post_init__(self) -> None:
        # NaN passes plain ``< 0`` comparisons (IEEE-754), so finiteness must be
        # checked explicitly before sign; otherwise NaN sigmas reach sampling
        # and Cholesky as silently "valid" values.
        if not math.isfinite(self.sigma_r_m) or self.sigma_r_m < 0.0:
            raise ValueError(f"sigma_r_m must be finite and >= 0, got {self.sigma_r_m}")
        if not math.isfinite(self.sigma_v_m_s) or self.sigma_v_m_s < 0.0:
            raise ValueError(f"sigma_v_m_s must be finite and >= 0, got {self.sigma_v_m_s}")

        if self.covariance_6x6 is not None:
            cov = np.asarray(self.covariance_6x6, dtype=np.float64)
            if cov.ndim != 2 or cov.shape != (6, 6):
                raise ValueError(
                    f"covariance_6x6 must be shape (6, 6), got {cov.shape}"
                )
            if not np.all(np.isfinite(cov)):
                raise ValueError("covariance_6x6 must contain only finite values.")
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
        return np.asarray(np.linalg.cholesky(P), dtype=np.float64)

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
            if not math.isfinite(v) or v < 0.0:
                raise ValueError(f"SpacecraftUncertainty.{attr} must be finite and >= 0, got {v}")

    @property
    def any_active(self) -> bool:
        """True if at least one parameter has non-zero uncertainty."""
        return any(
            v > 0.0
            for v in (self.sigma_mass_kg, self.sigma_cd, self.sigma_cr, self.sigma_area_m2)
        )


# =============================================================================
# 3.                       BATCH PROPAGATION CONFIG
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
class BatchPropagationConfig:
    """
    Top-level batch/ensemble propagation configuration.

    ``sampling_method="random"`` is the Monte Carlo sampling design. Use
    ``"lhs"``, ``"sobol"``, or ``"sobol_scrambled"`` when the goal is a
    space-filling validation or benchmark design rather than a purely random
    injection-dispersion draw.

    Routing
    -------
    ``use_gpu=True``  → request a GPU backend.
    ``use_gpu=False`` → CPU multiprocessing via existing :func:`core.propagation.propagator.propagate`
                        (all currently implemented force flags, slower for large N).

    ``batch_backend`` is the explicit backend selector.  The default ``"auto"``
    resolves the backend from ``use_gpu`` + ``gravity_mode_override``.
    Explicit values are ``"cpu_sh"``, ``"numba_cuda_sh"`` (degree <= 24 Numba
    CUDA screening kernel), ``"torch_cuda_sh"`` (high-degree PyTorch CUDA
    classic-SH, gravity-only),
    ``"torch_cpu_sh"`` (PyTorch CPU classic-SH, same evaluator as torch_cuda_sh),
    ``"gpu_st_lrps_potential"``, and ``"gpu_st_lrps_third_body"``.  The explicit
    value is recorded verbatim in provenance and is never silently rewritten to
    another backend name.

    GPU physics model
    -----------------
    - Point-mass + SH gravity up to the resolved ``sh_degree``.
    - Third-body Sun / Earth (if enabled in SimConfig flags).
    - SRP (if enabled in SimConfig flags); Numba CUDA uses the cannonball
      Cr*A/m area model with a reduced-fidelity cylindrical Moon umbra and no
      Earth eclipse, recorded in provenance.
    - Selected 1PN corrections (if enabled in SimConfig flags).
    - Albedo / thermal / tides: CPU-only (not available on GPU path).

    Output
    ------
    Snapshots at ``output_dt_s`` intervals are written to disk in HDF5 or NPZ
    format to avoid VRAM / RAM exhaustion for large N or long durations.

    Validation
    ----------
    The current Numba CUDA SH kernel is a true GPU implementation only through
    degree 24 because it uses compile-time fixed workspace arrays sized for
    degree 24 (26×26 per thread).  Higher requested degrees are accepted here
    so the backend resolver can fall back explicitly to CPU instead of silently
    clipping the request.
    """
    # Ensemble
    n_samples: int = 1_000
    seed: int = 42
    sampling_method: SamplingMethod = "random"

    # Injection-dispersion models
    state: StateUncertainty = field(default_factory=StateUncertainty)
    spacecraft: SpacecraftUncertainty = field(default_factory=SpacecraftUncertainty)

    # Backend selection
    use_gpu: bool = True
    batch_backend: BatchBackend = "auto"
    gpu_device_id: int = 0
    gravity_mode_override: GravityModeOverride = "follow_mission"
    st_lrps_model_dir: str | None = None

    # GPU physics fidelity
    sh_degree: int = 10         # requested SH degree (numba_cuda_sh true SH only through 24)
    gpu_threads_per_block: int = 128

    # High-degree classic-SH fallback policy when an explicit ``numba_cuda_sh``
    # request exceeds the degree-24 kernel limit: "compatible_gpu" (try
    # torch_cuda_sh, else CPU), "cpu" (force CPU), or "error" (raise instead of
    # substituting).  The requested degree is never clipped.
    sh_fallback_policy: ShFallbackPolicy = "compatible_gpu"

    # Torch backend controls (torch_cuda_sh and GPU ST-LRPS paths).
    torch_dtype: TorchDtype = "float64"    # "float32" or "float64" for torch backends
    torch_sh_chunk_size: int = 0    # 0 = auto (VRAM-aware); else samples per GPU chunk

    # Fixed-step RK4 integration (GPU path)
    dt_s: float = 60.0              # RK4 step [s]

    # Output
    output_format: OutputFormat = "hdf5"     # "hdf5" or "npz"
    output_path: str = "outputs/ensemble/batch_output.h5"
    max_vram_gb: float = 4.0        # VRAM budget (caps batch size automatically)
    result_storage_mode: ResultStorageMode = "auto"  # "auto", "memory", or "disk"
    max_result_memory_gb: float = 1.0
    # R23 screening output: "full" keeps every trajectory; "summary_only" keeps
    # the versioned per-sample screening summary plus the top-K full histories
    # (the (T, N, 6) ensemble tensor is never materialized or archived).
    output_mode: BatchOutputMode = "full"       # "full" or "summary_only"
    summary_top_k: int = 16         # full histories retained in summary mode

    # Statistical analysis
    detect_impact: bool | None = None
    compute_impact_statistics: bool | None = None
    compute_impact_probability: bool | None = None
    impact_alt_km: float = 0.0      # Impact detection threshold [km]
    impact_surface_mode: ImpactSurfaceMode = "sphere"  # "sphere" or "terrain"
    sigma_levels: tuple[float, ...] = (1.0, 2.0, 3.0)

    def __post_init__(self) -> None:
        if self.n_samples < 2:
            raise ValueError(f"n_samples must be >= 2, got {self.n_samples}")
        if self.sampling_method not in BATCH_SAMPLING_METHODS:
            raise ValueError(
                f"sampling_method must be one of: {_choice_list(BATCH_SAMPLING_METHODS)}. "
                f"Got {self.sampling_method!r}"
            )
        if not math.isfinite(self.dt_s) or self.dt_s <= 0.0:
            raise ValueError(f"dt_s must be finite and > 0, got {self.dt_s}")
        if self.gravity_mode_override not in GRAVITY_MODE_OVERRIDES:
            raise ValueError(
                f"gravity_mode_override must be one of: {_choice_list(GRAVITY_MODE_OVERRIDES)}. "
                f"Got {self.gravity_mode_override!r}"
            )
        if self.batch_backend not in BATCH_BACKENDS:
            raise ValueError(
                f"batch_backend must be one of: {_choice_list(BATCH_BACKENDS)}. "
                f"Got {self.batch_backend!r}"
            )
        if self.sh_fallback_policy not in SH_FALLBACK_POLICIES:
            raise ValueError(
                f"sh_fallback_policy must be one of: {_choice_list(SH_FALLBACK_POLICIES)}. "
                f"Got {self.sh_fallback_policy!r}"
            )
        if str(self.torch_dtype).lower() not in TORCH_DTYPES:
            raise ValueError(
                f"torch_dtype must be one of: {_choice_list(TORCH_DTYPES)}. "
                f"Got {self.torch_dtype!r}"
            )
        if int(self.torch_sh_chunk_size) < 0:
            raise ValueError(
                f"torch_sh_chunk_size must be >= 0 (0 = auto), got {self.torch_sh_chunk_size}"
            )
        st_lrps_model_dir = str(self.st_lrps_model_dir or "").strip()
        if (
            self.gravity_mode_override == "st_lrps"
            or self.batch_backend in ST_LRPS_BATCH_BACKENDS
        ) and not st_lrps_model_dir:
            raise ValueError(
                "st_lrps_model_dir cannot be empty when ST-LRPS batch gravity is requested."
            )
        if int(self.sh_degree) < 0:
            raise ValueError(
                f"sh_degree must be >= 0; backend policy handles GPU support limits. "
                f"got {self.sh_degree}."
            )
        if not (32 <= self.gpu_threads_per_block <= 1024):
            raise ValueError(
                f"gpu_threads_per_block must be in [32, 1024], "
                f"got {self.gpu_threads_per_block}"
            )
        if self.output_format not in OUTPUT_FORMATS:
            raise ValueError(
                f"output_format must be one of: {_choice_list(OUTPUT_FORMATS)}. "
                f"Got {self.output_format!r}"
            )
        if self.result_storage_mode not in RESULT_STORAGE_MODES:
            raise ValueError(
                f"result_storage_mode must be one of: {_choice_list(RESULT_STORAGE_MODES)}. "
                f"Got {self.result_storage_mode!r}"
            )
        if self.result_storage_mode == "disk" and self.output_format != "hdf5":
            raise ValueError("result_storage_mode='disk' requires output_format='hdf5'.")
        if self.output_mode not in BATCH_OUTPUT_MODES:
            raise ValueError(
                f"output_mode must be one of: {_choice_list(BATCH_OUTPUT_MODES)}. "
                f"Got {self.output_mode!r}"
            )
        if int(self.summary_top_k) < 1:
            raise ValueError(f"summary_top_k must be >= 1, got {self.summary_top_k}")
        if not math.isfinite(self.max_result_memory_gb) or self.max_result_memory_gb <= 0.0:
            raise ValueError(
                f"max_result_memory_gb must be finite and > 0, got {self.max_result_memory_gb}"
            )
        if not math.isfinite(self.max_vram_gb) or self.max_vram_gb <= 0.0:
            raise ValueError(f"max_vram_gb must be finite and > 0, got {self.max_vram_gb}")
        if not math.isfinite(self.impact_alt_km) or self.impact_alt_km < 0.0:
            raise ValueError(f"impact_alt_km must be finite and >= 0, got {self.impact_alt_km}")
        if self.impact_surface_mode not in IMPACT_SURFACE_MODES:
            raise ValueError(
                f"impact_surface_mode must be one of: {_choice_list(IMPACT_SURFACE_MODES)}. "
                f"Got {self.impact_surface_mode!r}"
            )
        if self.compute_impact_probability is not None:
            warnings.warn(
                "compute_impact_probability is deprecated; use detect_impact and "
                "compute_impact_statistics.",
                DeprecationWarning,
                stacklevel=2,
            )

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

    @property
    def impact_detection_enabled(self) -> bool:
        if self.detect_impact is not None:
            return bool(self.detect_impact)
        if self.compute_impact_probability is not None:
            return bool(self.compute_impact_probability)
        return True

    @property
    def impact_surface_terrain_enabled(self) -> bool:
        """True when terrain-aware impact freeze is requested AND detection is on.

        The kernels still require a usable topography payload at runtime; this
        flag only reflects the requested policy, not data availability.
        """
        return self.impact_detection_enabled and self.impact_surface_mode == "terrain"

    @property
    def impact_statistics_enabled(self) -> bool:
        if self.compute_impact_statistics is not None:
            return bool(self.compute_impact_statistics)
        if self.compute_impact_probability is not None:
            return bool(self.compute_impact_probability)
        return True

    def estimated_result_bytes(self, n_steps: int) -> int:
        """Estimated bytes for an eager ``(T, N, 6)`` float64 trajectory."""
        return int(n_steps) * int(self.n_samples) * 6 * np.dtype(np.float64).itemsize


# =============================================================================
# 4.                          RESULT CONTAINERS
# =============================================================================


def _as_bool_mask(value: Any, name: str) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D mask, got {arr.shape}")
    if arr.dtype == np.bool_:
        return np.ascontiguousarray(arr, dtype=np.bool_)
    return np.ascontiguousarray(np.asarray(arr, dtype=np.float64) > 0.5, dtype=np.bool_)


@dataclass(slots=True)
class BatchPropagationResult:
    """
    Ensemble simulation output.

    Shape conventions
    -----------------
    - ``t``  : (T,)       — output time grid [s]
    - ``Y``  : (T, N, 6)  — state ensemble [m, m/s]
    - ``sc_samples`` : (N, 4)  — sampled [mass_kg, area_m2, cd, cr] per run

    ``Y[k, i, :]`` is the state of sample ``i`` at time step ``k``.

    Notes
    -----
    - Only samples that did not impact are kept for full duration.
    - ``impact_mask[i]`` is True if sample ``i`` impacted before ``t[-1]``.
    - ``t_impact[i]`` is the impact time (NaN if no impact).
    """
    t: F64Array                              # (T,)
    Y: np.ndarray | TrajectoryArrayLike      # ndarray or read-only lazy trajectory view
    sc_samples: F64Array                     # (N, 4) [mass_kg, area_m2, cd, cr]
    impact_mask: np.ndarray                  # (N,) boolean impact mask
    t_impact: F64Array                       # (N,) NaN if no impact
    diagnostics: dict = field(default_factory=dict)
    # (N,) boolean validity mask: True=valid sample, False=failed/invalid (e.g.
    # a CPU sample that raised and was NaN-filled under allow_sample_failures).
    # ``None`` means "all samples valid" — the batch GPU/torch paths never
    # partially fail. Lets consumers exclude failed samples from ensemble
    # statistics instead of treating NaN-filled trajectories as if they were real
    # states. Archive writers may store this as compact 0/1 values; the in-memory
    # API exposes bools.
    valid_mask: np.ndarray | None = None
    impact_position_inertial_m: F64Array | None = None
    impact_position_fixed_m: F64Array | None = None
    archive_path: str | None = None

    def __post_init__(self) -> None:
        self.t = np.ascontiguousarray(self.t, dtype=np.float64)
        is_lazy = bool(getattr(self.Y, "_lunaris_lazy_trajectory", False))
        if not is_lazy:
            self.Y = np.ascontiguousarray(self.Y, dtype=np.float64)
        self.sc_samples = np.ascontiguousarray(self.sc_samples, dtype=np.float64)
        self.impact_mask = _as_bool_mask(self.impact_mask, "impact_mask")
        self.t_impact = np.ascontiguousarray(self.t_impact, dtype=np.float64)
        if self.valid_mask is not None:
            self.valid_mask = _as_bool_mask(self.valid_mask, "valid_mask")
        if self.impact_position_inertial_m is not None:
            self.impact_position_inertial_m = np.ascontiguousarray(
                self.impact_position_inertial_m, dtype=np.float64
            )
        if self.impact_position_fixed_m is not None:
            self.impact_position_fixed_m = np.ascontiguousarray(
                self.impact_position_fixed_m, dtype=np.float64
            )

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
        if self.valid_mask is not None and self.valid_mask.shape != (n_samp,):
            raise ValueError(
                f"valid_mask must be (N,), got {self.valid_mask.shape}"
            )
        for name, value in (
            ("impact_position_inertial_m", self.impact_position_inertial_m),
            ("impact_position_fixed_m", self.impact_position_fixed_m),
        ):
            if value is not None and value.shape != (n_samp, 3):
                raise ValueError(f"{name} must be (N, 3), got {value.shape}")

    @property
    def n_samples(self) -> int:
        return int(self.Y.shape[1])

    @property
    def n_valid(self) -> int:
        """Number of valid (non-failed) samples; all samples when no mask is set."""
        if self.valid_mask is None:
            return int(self.Y.shape[1])
        return int(np.sum(self.valid_mask))

    def valid_sample_mask(self) -> np.ndarray:
        """Canonical boolean validity mask used by analysis and reporting."""
        if self.valid_mask is None:
            return np.ones(self.n_samples, dtype=bool)
        return np.asarray(self.valid_mask, dtype=bool)

    @property
    def n_steps(self) -> int:
        return int(self.t.shape[0])

    @property
    def n_survived(self) -> int:
        """Number of samples that completed without impact."""
        return int(np.sum(self.valid_sample_mask() & ~self.impact_mask))

    @property
    def impact_fraction(self) -> float:
        """Fraction of samples that impacted the surface."""
        valid = self.valid_sample_mask()
        n_valid = int(np.sum(valid))
        if n_valid == 0:
            raise ValueError("impact fraction is undefined: no valid samples")
        return float(np.sum(valid & self.impact_mask) / n_valid)

    def survived_trajectories(self) -> F64Array:
        """Return Y[:, mask, :] for non-impacting samples only."""
        mask = self.valid_sample_mask() & ~self.impact_mask
        return self.Y[:, mask, :]

    @property
    def is_lazy(self) -> bool:
        return bool(getattr(self.Y, "_lunaris_lazy_trajectory", False))


# =============================================================================
# 5.                         PUBLIC API
# =============================================================================

__all__ = [
    "build_batch_output_grid",
    "build_fixed_step_grid_metadata",
    "BATCH_BACKENDS",
    "BATCH_OUTPUT_MODES",
    "GRAVITY_MODE_OVERRIDES",
    "IMPACT_SURFACE_MODES",
    "OUTPUT_FORMATS",
    "BATCH_SAMPLING_METHODS",
    "RESULT_STORAGE_MODES",
    "SH_FALLBACK_POLICIES",
    "ST_LRPS_BATCH_BACKENDS",
    "TORCH_DTYPES",
    "BatchBackend",
    "BatchOutputMode",
    "StateUncertainty",
    "SpacecraftUncertainty",
    "BatchPropagationConfig",
    "BatchPropagationResult",
    "GravityModeOverride",
    "ImpactSurfaceMode",
    "OutputFormat",
    "ResultStorageMode",
    "SamplingMethod",
    "ShFallbackPolicy",
    "TorchDtype",
    "TrajectoryArrayLike",
    "validate_st_lrps_model_dir",
]
