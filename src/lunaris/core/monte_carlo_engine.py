# lunaris.core.monte_carlo_engine
"""
Batch / Monte Carlo Dispatch Engine
===================================

This module is the single entry point for running batch ensemble propagation.
Monte Carlo is the default random sampling method; LHS and Sobol designs are
available for validation-oriented coverage.
It handles:

1. Sample generation  — random, LHS, or Sobol Gaussian designs for the 6D state and
   optional spacecraft property perturbations.
2. Backend dispatch   — routes to GPUBatchPropagator (CUDA) or
   CPUBatchPropagator (multiprocessing) based on ``MonteCarloConfig.use_gpu``
   and hardware availability.
3. VRAM / RAM budget  — automatically tiles large ensembles into sub-batches
   that fit within ``mc_cfg.max_vram_gb``.
4. Streaming output   — snapshot data is written to HDF5 or NPZ at
   ``output_dt_s`` intervals to avoid memory exhaustion.
5. Progress reporting — optional structured ``progress_callback(payload)`` hook.

Usage example
-------------
::

    from lunaris.core.config import load_default_config
    from lunaris.common.montecarlo_defs import MonteCarloConfig, StateUncertainty
    from lunaris.core.monte_carlo_engine import MonteCarloEngine

    sim_cfg = load_default_config()
    mc_cfg  = MonteCarloConfig(
        n_samples=500,
        state=StateUncertainty(sigma_r_m=500.0, sigma_v_m_s=0.5),
        use_gpu=True,
        gpu_sh_degree=10,
    )

    engine = MonteCarloEngine(sim_cfg, mc_cfg)
    result = engine.run()          # MCRunResult

Architecture note
-----------------
This module is **layer 3** (core); it may import from ``common`` and
``common``, ``physics``, and lower-level ``core`` helpers but must not import
from ``analysis`` or UI packages.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import time
import warnings
from collections.abc import Callable, Iterator
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from lunaris.common.constants import DAY_S, MU_MOON, R_MOON
from lunaris.common.math_utils import quat_rotate_np, quat_slerp_np
from lunaris.common.montecarlo_defs import (
    BATCH_SAMPLING_METHODS,
    MCRunResult,
    MonteCarloConfig,
    StateUncertainty,
    build_mc_output_grid,
)
from lunaris.common.type_defs import F64Array

if TYPE_CHECKING:
    from lunaris.core.mc_backend_policy import MCBackendPlan

# Canonical backend name -> human-readable run label. Sourced from the resolved
# plan's ``actual_backend`` (the single source of truth for what executes) so a
# CPU run is never labeled GPU. ``torch_cpu_sh`` and ``torch_cuda_sh`` share one
# propagator class, which is exactly why class-name inference is unsafe here.
_BACKEND_DISPLAY_NAMES = {
    "cpu_sh": "CPU",
    "cpu_st_lrps": "CPU-ST-LRPS",
    "numba_cuda_sh": "GPU-CLASSIC-SH",
    "torch_cuda_sh": "GPU-TORCH-SH",
    "torch_cpu_sh": "CPU-TORCH-SH",
    "gpu_st_lrps_potential": "GPU-ST-LRPS",
    "gpu_st_lrps_direct": "GPU-ST-LRPS",
}

# Manifest fields that a schema-v2 Monte Carlo archive must carry. Enforced by
# ``load_mc_result(strict=True)`` so a truncated, hand-edited, or pre-contract
# archive cannot masquerade as a complete, provenance-bearing v2 result. Every
# listed field is always written with a non-null value by ``MonteCarloEngine.run``.
REQUIRED_ARCHIVE_V2_FIELDS: tuple[str, ...] = (
    "archive_schema_version",
    "n_samples",
    "seed",
    "duration_s",
    "output_dt_s",
    "backend",
    "requested_mc_backend",
    "actual_mc_backend",
    "mc_backend",
    "detect_impact",
    "compute_impact_statistics",
)


def _sha256_file(path: Any) -> str | None:
    """Return the SHA-256 hex digest of a file, or ``None`` when unavailable.

    Used to stamp artifact / coefficient / kernel provenance into the archive
    manifest. Never raises: missing or unreadable files yield ``None`` so
    provenance capture cannot abort a completed run.
    """
    if not path:
        return None
    try:
        p = Path(path)
        if not p.is_file():
            return None
        digest = hashlib.sha256()
        with p.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except Exception:
        return None

# =============================================================================
# 0.                    LOCAL BOOTSTRAP / COMPAT HELPERS
# =============================================================================

def _state_to_array(state_like: Any) -> np.ndarray:
    """
    Convert the configured nominal state to a plain row-major float64 vector.

    MC sampling works in Cartesian state space, so we accept the same state
    container styles as the single-run pipeline: ``InitialState``,
    ``OrbitState``-like objects exposing ``.y``, or raw array-likes.
    """

    if state_like is None:
        raise ValueError("Nominal state is None.")

    if hasattr(state_like, "to_array"):
        arr = np.asarray(state_like.to_array(), dtype=np.float64).reshape(-1)
    elif hasattr(state_like, "y"):
        arr = np.asarray(state_like.y, dtype=np.float64).reshape(-1)
    else:
        arr = np.asarray(state_like, dtype=np.float64).reshape(-1)

    if arr.size < 6:
        raise ValueError(f"Nominal state must contain at least 6 elements, got {arr.size}.")
    return np.ascontiguousarray(arr[:6], dtype=np.float64)


def _active_physics_capabilities(sim_cfg: Any) -> list[str]:
    """Return the canonical force models active in ``sim_cfg.flags`` for provenance.

    Recorded in the run manifest so a reader can see exactly which physics the run
    requested — and, paired with ``actual_mc_backend``, whether the chosen backend
    could honor them.
    """
    from lunaris.core.backend_capabilities import FORCE_MODEL_FLAG_ATTR

    flags = getattr(sim_cfg, "flags", None)
    if flags is None:
        return []
    active: list[str] = []
    for canonical, attr in FORCE_MODEL_FLAG_ATTR.items():
        if bool(getattr(flags, attr, False)):
            active.append(canonical)
    return active


def _surface_topography_requested(surface_provider: Any, topo_grid: Any) -> bool:
    """
    Return True when the MC run needs Moon-fixed ephemeris because terrain is active.

    Topography can influence both surface-force sampling and impact detection, so
    the MC bootstrap mirrors the main runner by treating terrain availability as
    an ephemeris requirement even when third-body vectors are disabled.
    """

    return bool(surface_provider is not None or topo_grid is not None)


def _need_ephemeris(cfg: Any, *, topo_requested: bool) -> bool:
    """
    Match the main runner's ephemeris policy for consistent physics coverage.

    The Monte Carlo path should not secretly use a different decision tree than
    the single-run path.  Repeating the logic locally keeps this core module
    self-contained while preserving the same "SH/topography implies q_i2f"
    behavior.
    """

    flags = cfg.flags
    physics_need = (
        flags.enable_sh
        or flags.enable_3rd_body_sun
        or flags.enable_3rd_body_earth
        or flags.enable_earth_j2
        or flags.enable_srp
        or flags.enable_albedo
        or flags.enable_thermal
        or flags.enable_surface_forces
        or flags.enable_tides_k2
        or flags.enable_tides_k3
        or flags.enable_relativity_1pn
    )
    return bool(physics_need or topo_requested)


def _need_body_vectors(cfg: Any) -> bool:
    """
    Return True only when Sun/Earth position tables are physically required.

    SH-only or topo-only Monte Carlo runs still need the Moon-fixed attitude
    quaternion table, but they do not need Sun/Earth vectors.  Using this split
    keeps ephemeris initialization lighter and avoids misleading SPICE warnings.
    """

    flags = cfg.flags
    thermal_mode = (
        str(getattr(getattr(cfg, "thermal", None), "thermal_mode", "constant_temperature")).strip().lower()
    )
    thermal_needs_sun = bool(
        flags.enable_thermal
        and thermal_mode in {"equilibrium", "equilibrium_temperature", "instantaneous_equilibrium"}
    )
    return bool(
        flags.enable_3rd_body_sun
        or flags.enable_3rd_body_earth
        or flags.enable_earth_j2
        or flags.enable_srp
        or flags.enable_albedo
        or thermal_needs_sun
        or flags.enable_tides_k2
        or flags.enable_tides_k3
        or flags.enable_relativity_1pn
    )


def _build_ephemeris_manager(cfg: Any) -> Any:
    """
    Build an ``EphemerisManager`` using the same buffered timeline as main.py.

    A small duration buffer protects interpolation near the last requested
    sample, which is especially helpful in Monte Carlo runs where many samples
    stop at slightly different times due to impact events.
    """

    from lunaris.physics.ephemeris import EphemerisManager

    start_utc = str(cfg.time.start_date).strip()
    if not start_utc:
        raise ValueError("cfg.time.start_date is empty.")

    time_cfg = replace(cfg.time, duration_s=float(cfg.time.duration_s) + 0.1 * DAY_S)
    spice_cfg = replace(cfg.spice, include_third_body=_need_body_vectors(cfg))
    return EphemerisManager.from_time_and_spice(
        time_cfg,
        spice_cfg,
        auto_fix_kernel_paths=True,
        need_moon_fixed_rotation=True,
    )


def _impact_positions_fixed(
    ephem: Any,
    t_impact: np.ndarray,
    positions_inertial: np.ndarray,
) -> np.ndarray:
    """Rotate per-sample inertial impact positions into the Moon-fixed frame.

    Returns an all-NaN array when no ephemeris is available. Without the
    inertial->Moon-fixed rotation table there is no physically meaningful
    Moon-fixed impact position, so the engine must NOT fabricate one by treating
    inertial coordinates as if they were body-fixed: that silently produces a
    wrong geographic impact distribution (lat/lon) with no error signal.
    Consumers detect the NaN and skip lat/lon reporting instead.
    """
    out = np.full_like(positions_inertial, np.nan, dtype=np.float64)
    if ephem is None:
        return out
    finite = np.isfinite(t_impact) & np.isfinite(positions_inertial).all(axis=1)
    if not np.any(finite):
        return out

    provider = ephem.get_data_provider()
    q_tab = np.asarray(
        provider.get("q_i2f_tab", provider.get("rot_table")),
        dtype=np.float64,
    )
    dt_s = float(provider.get("dt_s", provider.get("dt")))
    for idx in np.where(finite)[0]:
        u = max(0.0, float(t_impact[idx]) / max(dt_s, 1e-12))
        i0 = min(int(math.floor(u)), q_tab.shape[0] - 1)
        if i0 >= q_tab.shape[0] - 1:
            q = q_tab[-1]
        else:
            q = quat_slerp_np(q_tab[i0], q_tab[i0 + 1], u - i0)
        out[idx] = quat_rotate_np(q, positions_inertial[idx])
    return out


# =============================================================================
# 1.                      SAMPLE GENERATION
# =============================================================================

def _sobol_size_note(method: str, n_samples: int) -> str:
    if str(method).startswith("sobol") and n_samples > 0:
        power = 1 << int(math.ceil(math.log2(max(1, int(n_samples)))))
        if power != int(n_samples):
            return (
                f"{method} generated {power} base-2 design points and kept the "
                f"first {int(n_samples)}."
            )
    return ""


def generate_standard_normal_design(
    n_samples: int,
    n_dim: int,
    method: str,
    seed: int,
    rng: np.random.Generator | None = None,
) -> F64Array:
    """
    Generate standardized normal samples for ensemble propagation.

    ``random`` preserves the historical Monte Carlo draw path. Space-filling
    methods generate unit-hypercube designs and transform them with the inverse
    normal CDF so the existing covariance machinery can be reused.
    """

    method = str(method or "random")
    if method not in BATCH_SAMPLING_METHODS:
        raise ValueError(
            "sampling_method must be one of: "
            + ", ".join(repr(item) for item in BATCH_SAMPLING_METHODS)
            + f". Got {method!r}"
        )
    if n_samples <= 0:
        raise ValueError(f"n_samples must be > 0, got {n_samples}")
    if n_dim <= 0:
        raise ValueError(f"n_dim must be > 0, got {n_dim}")

    if method == "random":
        active_rng = rng if rng is not None else np.random.default_rng(int(seed))
        return np.ascontiguousarray(
            active_rng.standard_normal((int(n_samples), int(n_dim))),
            dtype=np.float64,
        )

    from scipy import special
    from scipy.stats import qmc

    if method == "lhs":
        unit = qmc.LatinHypercube(d=int(n_dim), seed=int(seed)).random(int(n_samples))
    else:
        scramble = method == "sobol_scrambled"
        sampler = qmc.Sobol(
            d=int(n_dim),
            scramble=scramble,
            seed=int(seed) if scramble else None,
        )
        m = int(math.ceil(math.log2(max(1, int(n_samples)))))
        unit = sampler.random_base2(m=m)[: int(n_samples)]

    eps = np.finfo(np.float64).eps
    clipped = np.clip(np.asarray(unit, dtype=np.float64), eps, 1.0 - eps)
    return np.ascontiguousarray(special.ndtri(clipped), dtype=np.float64)


def sample_initial_states(
    nominal_state: F64Array,         # (6,) [x,y,z,vx,vy,vz]
    uncertainty: StateUncertainty,
    n_samples: int,
    rng: np.random.Generator,
    *,
    sampling_method: str = "random",
    seed: int = 0,
    standard_normal_samples: F64Array | None = None,
) -> F64Array:
    """
    Draw N Gaussian samples around the nominal state.

    ``sampling_method`` can be ``random`` (classical Monte Carlo), ``lhs``,
    ``sobol``, or ``sobol_scrambled``. Non-random methods are transformed into
    standard-normal samples before the covariance factor is applied.

    Returns
    -------
    Y0 : (N, 6) float64 perturbed initial states
    """
    L = uncertainty.cholesky_factor()           # (6, 6) lower-triangular
    if standard_normal_samples is None:
        Z = generate_standard_normal_design(n_samples, 6, sampling_method, seed, rng)
    else:
        Z = np.asarray(standard_normal_samples, dtype=np.float64)
        if Z.shape != (int(n_samples), 6):
            raise ValueError(
                f"standard_normal_samples must be ({n_samples}, 6), got {Z.shape}"
            )
    delta = Z @ L.T                             # (N, 6) perturbation
    return np.ascontiguousarray(
        nominal_state[None, :] + delta, dtype=np.float64
    )


def sample_spacecraft_props(
    nominal_mass: float,
    nominal_area: float,
    nominal_cd: float,
    nominal_cr: float,
    uncertainty: Any,               # SpacecraftUncertainty
    n_samples: int,
    rng: np.random.Generator,
    *,
    sampling_method: str = "random",
    seed: int = 0,
    standard_normal_samples: F64Array | None = None,
) -> F64Array:
    """
    Sample spacecraft physical properties (truncated normal at zero).

    Returns
    -------
    sc_samples : (N, 4) float64 — columns [mass_kg, area_m2, cd, cr]
    """
    sc = np.zeros((n_samples, 4), dtype=np.float64)
    if standard_normal_samples is not None:
        z_sc = np.asarray(standard_normal_samples, dtype=np.float64)
        if z_sc.shape != (int(n_samples), 4):
            raise ValueError(
                f"standard_normal_samples must be ({n_samples}, 4), got {z_sc.shape}"
            )
    elif str(sampling_method or "random") == "random":
        z_sc = None
    else:
        z_sc = generate_standard_normal_design(n_samples, 4, sampling_method, seed)

    def _trunc_normal(mu: float, sigma: float, col: int) -> np.ndarray:
        """Sample with sigma; clip at 0.01 * mu to keep values positive."""
        if sigma <= 0.0:
            return np.full(n_samples, mu, dtype=np.float64)
        if z_sc is None:
            raw = rng.normal(mu, sigma, n_samples)
        else:
            raw = mu + sigma * z_sc[:, col]
        return np.clip(raw, 0.01 * max(mu, 1e-30), None)

    sc[:, 0] = _trunc_normal(nominal_mass, float(getattr(uncertainty, "sigma_mass_kg", 0.0)), 0)
    sc[:, 1] = _trunc_normal(nominal_area, float(getattr(uncertainty, "sigma_area_m2", 0.0)), 1)
    sc[:, 2] = _trunc_normal(nominal_cd,   float(getattr(uncertainty, "sigma_cd",     0.0)), 2)
    sc[:, 3] = _trunc_normal(nominal_cr,   float(getattr(uncertainty, "sigma_cr",     0.0)), 3)

    return sc


# =============================================================================
# 2.               OUTPUT WRITERS (HDF5 / NPZ)
# =============================================================================

def _metadata_value_to_jsonable(value: Any) -> Any:
    """
    Convert runtime metadata values into JSON-safe primitives.

    Monte Carlo archives are often reopened long after the run completed, so
    even lightweight metadata such as seed, cadence, and backend selection is
    worth preserving in a transport-safe form across both HDF5 and NPZ outputs.
    """

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_metadata_value_to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _metadata_value_to_jsonable(val) for key, val in value.items()}
    return str(value)


def _decode_archive_metadata(raw: Any) -> dict[str, Any]:
    """
    Decode a metadata payload stored inside an archive.

    The helper is intentionally permissive: missing or malformed metadata
    should not block result loading.
    """

    if raw is None:
        return {}

    try:
        if isinstance(raw, np.ndarray):
            raw = raw.item()
        text = str(raw).strip()
        if not text:
            return {}
        decoded = json.loads(text)
        return decoded if isinstance(decoded, dict) else {}
    except Exception:
        return {}


def _decode_metadata_value(raw: Any) -> Any:
    """Normalize HDF5/NPZ metadata while preserving ordinary strings."""
    if isinstance(raw, np.generic):
        raw = raw.item()
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    if isinstance(raw, str):
        text = raw.strip()
        if text.startswith(("{", "[")):
            try:
                return json.loads(text)
            except Exception:
                return raw
        return raw
    if isinstance(raw, np.ndarray):
        return [_decode_metadata_value(item) for item in raw.tolist()]
    return _metadata_value_to_jsonable(raw)


# Fraction of the host RAM that is *free right now* an eager result (or a single
# per-batch host buffer) may occupy. Leaves headroom for the OS, other processes,
# and transient copies so a generous ``max_result_memory_gb`` — or a busy host —
# cannot push the run into swap/OOM. Applied only when psutil can measure RAM.
_HOST_MEMORY_SAFETY_FACTOR = 0.8


def _available_host_memory_bytes() -> int | None:
    """Bytes of host RAM available right now, or ``None`` if it cannot be measured.

    Uses psutil when present; degrades gracefully (returns ``None``) so the memory
    safety factor is a best-effort guard, never a hard dependency.
    """
    try:
        import psutil
    except Exception:
        return None
    try:
        return int(psutil.virtual_memory().available)
    except Exception:
        return None


def _resolve_result_storage(
    mc_cfg: MonteCarloConfig,
    n_steps: int,
) -> tuple[str, int, int]:
    """Resolve eager versus disk-backed result storage before opening a writer."""
    result_bytes = mc_cfg.estimated_result_bytes(n_steps)
    memory_limit_bytes = int(float(mc_cfg.max_result_memory_gb) * (1024.0 ** 3))
    # Effective host budget: the configured cap, further bounded by a safety
    # fraction of the RAM actually free now (when measurable). This is the budget
    # the auto storage decision and the per-batch host buffer must both respect.
    available = _available_host_memory_bytes()
    host_budget_bytes = memory_limit_bytes
    if available is not None:
        host_budget_bytes = min(memory_limit_bytes, int(available * _HOST_MEMORY_SAFETY_FACTOR))
    storage_mode = str(mc_cfg.result_storage_mode)
    if storage_mode == "auto":
        storage_mode = (
            "disk"
            if mc_cfg.output_format == "hdf5" and result_bytes > host_budget_bytes
            else "memory"
        )
    if storage_mode not in {"memory", "disk"}:
        raise ValueError(f"Unsupported result storage mode: {storage_mode!r}")
    if (
        mc_cfg.output_format == "npz"
        and result_bytes > host_budget_bytes
        and mc_cfg.result_storage_mode == "auto"
    ):
        limit_gib = host_budget_bytes / (1024.0 ** 3)
        budget_note = (
            f"host memory safety budget ({limit_gib:.2f} GiB = "
            f"{_HOST_MEMORY_SAFETY_FACTOR:.0%} of free RAM)"
            if available is not None and host_budget_bytes < memory_limit_bytes
            else f"max_result_memory_gb={mc_cfg.max_result_memory_gb:g}"
        )
        raise MemoryError(
            "Estimated eager Monte Carlo trajectory size "
            f"({result_bytes / (1024.0 ** 3):.2f} GiB) exceeds {budget_note}. "
            "Use HDF5 output for disk-backed streaming or explicitly choose "
            "result_storage_mode='memory'."
        )
    if (
        storage_mode == "memory"
        and available is not None
        and result_bytes > int(available * _HOST_MEMORY_SAFETY_FACTOR)
    ):
        warnings.warn(
            "Eager Monte Carlo result "
            f"({result_bytes / (1024.0 ** 3):.2f} GiB) exceeds the host memory "
            f"safety budget ({_HOST_MEMORY_SAFETY_FACTOR:.0%} of "
            f"{available / (1024.0 ** 3):.2f} GiB free RAM); the run may exhaust "
            "host memory. Prefer HDF5 disk-backed output for large ensembles.",
            RuntimeWarning,
            stacklevel=2,
        )
    return storage_mode, result_bytes, host_budget_bytes


def _allocate_result_buffer(
    storage_mode: str,
    writer_buffer: Any,
    shape: tuple[int, int, int],
) -> np.ndarray | None:
    """Allocate the full ensemble only for the explicit eager result path."""
    if storage_mode == "disk":
        return None
    if storage_mode != "memory":
        raise ValueError(f"Unsupported result storage mode: {storage_mode!r}")
    if isinstance(writer_buffer, np.ndarray):
        if tuple(writer_buffer.shape) != tuple(shape):
            raise ValueError(
                f"Writer result buffer must have shape {shape}, "
                f"got {writer_buffer.shape}"
            )
        return writer_buffer
    return np.empty(shape, dtype=np.float64)


class HDF5TrajectoryView:
    """Read-only, path-backed view of an HDF5 trajectory dataset."""

    _lunaris_lazy_trajectory = True

    def __init__(self, path: str | Path, dataset: str = "Y") -> None:
        self.path = Path(path).expanduser().resolve()
        self.dataset = str(dataset)
        try:
            import h5py
        except ImportError:
            raise ImportError("h5py required for lazy HDF5 MC results.") from None
        with h5py.File(str(self.path), "r") as f:
            ds = f[self.dataset]
            self.shape = tuple(int(v) for v in ds.shape)
            self.ndim = int(ds.ndim)
            self.dtype = np.dtype(ds.dtype)

    def __getitem__(self, key: Any) -> np.ndarray:
        import h5py

        with h5py.File(str(self.path), "r") as f:
            return np.asarray(f[self.dataset][key], dtype=np.float64)

    def iter_epoch_sample_blocks(
        self,
        sample_indices: np.ndarray,
    ) -> Iterator[np.ndarray]:
        """Yield one ``(n_samples, 6)`` block per epoch using one file handle."""
        import h5py

        indices = np.asarray(sample_indices, dtype=np.int64)
        if indices.ndim != 1:
            raise ValueError("sample_indices must be one-dimensional")
        if indices.size > 1 and np.any(np.diff(indices) <= 0):
            raise ValueError("sample_indices must be strictly increasing")

        with h5py.File(str(self.path), "r") as f:
            dataset = f[self.dataset]
            for epoch in range(self.shape[0]):
                yield np.asarray(dataset[epoch, indices, :], dtype=np.float64)

    def __array__(self, dtype: Any = None, copy: bool | None = None) -> np.ndarray:
        import h5py

        with h5py.File(str(self.path), "r") as f:
            arr = np.asarray(f[self.dataset], dtype=dtype or np.float64)
        if copy:
            return arr.copy()
        return arr


class _HDF5Writer:
    """Atomic HDF5 writer with sample-axis batch streaming."""

    def __init__(
        self,
        path: Path,
        n_samples: int,
        t_grid: np.ndarray,
        n_state: int = 6,
    ) -> None:
        try:
            import h5py
            self._h5py = h5py
        except ImportError:
            raise ImportError(
                "h5py is required for HDF5 output. "
                "Install via:  pip install h5py"
            ) from None

        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = path
        self._part_path = Path(f"{path}.part")
        if self._part_path.exists():
            self._part_path.unlink()
        self._f = h5py.File(str(self._part_path), "w")
        self._n = n_samples
        self._s = n_state
        self._t = np.ascontiguousarray(t_grid, dtype=np.float64)
        self._next_sample = 0
        self._final_payload_written = False
        self._ds_t = self._f.create_dataset("t", data=self._t)
        self._ds_Y = self._f.create_dataset(
            "Y", shape=(len(self._t), n_samples, n_state),
            dtype=np.float64,
            chunks=(min(len(self._t), 16), min(n_samples, 128), n_state),
            compression="lzf",
        )

    def write_sample_batch(self, start: int, end: int, Y: np.ndarray) -> None:
        """Write ``Y[:, start:end, :]`` without retaining the full ensemble."""
        start = int(start)
        end = int(end)
        if start != self._next_sample:
            raise ValueError(
                "HDF5 sample batches must be contiguous and ordered: "
                f"expected start={self._next_sample}, got {start}"
            )
        if end <= start or end > self._n:
            raise ValueError(
                f"HDF5 sample batch [{start}:{end}] is outside [0:{self._n}]"
            )
        expected = (len(self._t), int(end - start), self._s)
        if tuple(Y.shape) != expected:
            raise ValueError(f"HDF5 batch must have shape {expected}, got {Y.shape}")
        self._ds_Y[:, start:end, :] = Y
        self._next_sample = end

    @property
    def memory_buffer(self) -> None:
        return None

    def write_metadata(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            try:
                payload = _metadata_value_to_jsonable(v)
                if payload is None:
                    continue
                if isinstance(payload, (dict, list)):
                    payload = json.dumps(payload, sort_keys=True)
                self._f.attrs[k] = payload
            except Exception as exc:
                raise ValueError(
                    f"Could not write HDF5 metadata field {k!r}"
                ) from exc

    def write_final(
        self,
        sc_samples: np.ndarray,
        impact_flags: np.ndarray,
        t_impact: np.ndarray,
        valid_mask: np.ndarray,
        impact_position_inertial_m: np.ndarray,
        impact_position_fixed_m: np.ndarray,
    ) -> None:
        expected_shapes = {
            "sc_samples": (self._n, 4),
            "impact_flags": (self._n,),
            "t_impact": (self._n,),
            "valid_mask": (self._n,),
            "impact_position_inertial_m": (self._n, 3),
            "impact_position_fixed_m": (self._n, 3),
        }
        values = {
            "sc_samples": sc_samples,
            "impact_flags": impact_flags,
            "t_impact": t_impact,
            "valid_mask": valid_mask,
            "impact_position_inertial_m": impact_position_inertial_m,
            "impact_position_fixed_m": impact_position_fixed_m,
        }
        for name, expected in expected_shapes.items():
            actual = tuple(np.shape(values[name]))
            if actual != expected:
                raise ValueError(
                    f"HDF5 final payload {name!r} must have shape "
                    f"{expected}, got {actual}"
                )
        self._f.create_dataset("sc_samples",  data=sc_samples)
        self._f.create_dataset("impact_flags", data=impact_flags)
        self._f.create_dataset("t_impact",    data=t_impact)
        self._f.create_dataset("valid_mask", data=valid_mask)
        self._f.create_dataset(
            "impact_position_inertial_m", data=impact_position_inertial_m
        )
        self._f.create_dataset("impact_position_fixed_m", data=impact_position_fixed_m)
        self._final_payload_written = True

    def finalize(self) -> None:
        if self._next_sample != self._n:
            raise RuntimeError(
                "Cannot finalize incomplete HDF5 trajectory stream: "
                f"wrote samples [0:{self._next_sample}], expected [0:{self._n}]"
            )
        if not self._final_payload_written:
            raise RuntimeError(
                "Cannot finalize HDF5 archive before final result arrays are written"
            )
        self._f.attrs["archive_schema_version"] = 2
        self._f.flush()
        self._f.close()
        self._part_path.replace(self._path)

    def abort(self) -> None:
        try:
            self._f.close()
        except Exception:
            pass
        try:
            self._part_path.unlink(missing_ok=True)
        except Exception:
            pass


class _NPZWriter:
    """Writes an eager trajectory archive in one compressed NPZ operation."""

    def __init__(
        self,
        path: Path,
        n_samples: int,
        t_grid: np.ndarray,
        n_state: int = 6,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = path
        self._t = np.ascontiguousarray(t_grid, dtype=np.float64)
        self._Y = np.empty((len(self._t), n_samples, n_state), dtype=np.float64)
        self._metadata: dict[str, Any] = {}

    def write_sample_batch(self, start: int, end: int, Y: np.ndarray) -> None:
        self._Y[:, start:end, :] = Y

    @property
    def memory_buffer(self) -> np.ndarray:
        return self._Y

    def write_metadata(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            self._metadata[str(key)] = _metadata_value_to_jsonable(value)

    def write_final(
        self,
        sc_samples: np.ndarray,
        impact_flags: np.ndarray,
        t_impact: np.ndarray,
        valid_mask: np.ndarray,
        impact_position_inertial_m: np.ndarray,
        impact_position_fixed_m: np.ndarray,
    ) -> None:
        np.savez_compressed(
            str(self._path),
            t=self._t,
            Y=self._Y,
            sc_samples=sc_samples,
            impact_flags=impact_flags,
            t_impact=t_impact,
            valid_mask=valid_mask,
            impact_position_inertial_m=impact_position_inertial_m,
            impact_position_fixed_m=impact_position_fixed_m,
            metadata_json=np.asarray(json.dumps(self._metadata, sort_keys=True), dtype=np.str_),
        )

    def finalize(self) -> None:
        pass

    def abort(self) -> None:
        pass


def _make_writer(
    mc_cfg: MonteCarloConfig,
    n_samples: int,
    t_grid: np.ndarray,
) -> Any:
    """Factory: return the appropriate writer based on output_format."""
    p = mc_cfg.output_path_resolved
    if mc_cfg.output_format == "hdf5":
        return _HDF5Writer(p, n_samples, t_grid)
    return _NPZWriter(p, n_samples, t_grid)


# =============================================================================
# 3.               MONTE CARLO ENGINE
# =============================================================================

class MonteCarloEngine:
    """
    Orchestrates a full Monte Carlo orbital uncertainty propagation run.

    Workflow
    --------
    1. ``__init__``: validate configs, select backend (GPU / CPU).
    2. ``run()``:
       a. Draw N initial state samples + spacecraft property samples.
       b. Open output writer.
       c. For each sub-batch (VRAM-bounded):
          - Transfer arrays to device (GPU) or dispatch workers (CPU).
          - Iterate over time steps; write snapshots to disk.
       d. Aggregate impact statistics.
       e. Return ``MCRunResult``.

    Parameters
    ----------
    sim_cfg : SimConfig
        Full simulation configuration (physics flags, gravity, ephemeris, …).
    mc_cfg : MonteCarloConfig
        Monte Carlo parameters (N, uncertainties, GPU flags, output format).
    dynamics_engine : optional pre-built DynamicsEngine
        If None, the engine builds one from ``sim_cfg``.
    progress_callback : optional ``f(payload: dict)``
        Receives structured progress payloads containing stage, percent,
        done/total scenario counts, and ETA hints suitable for UI progress bars.
    """

    def __init__(
        self,
        sim_cfg: Any,                       # config.SimConfig
        mc_cfg: MonteCarloConfig,
        dynamics_engine: Any = None,        # core.dynamics.DynamicsEngine
        surface_provider: Any = None,
        topo_grid: Any = None,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self._sim_cfg = sim_cfg
        self._mc      = mc_cfg
        self._cb      = progress_callback
        self._surface_provider = surface_provider
        self._topo_grid = topo_grid
        self._backend_note = ""
        self._backend_plan: MCBackendPlan | None = None
        if self._topo_grid is None and self._surface_provider is not None and hasattr(self._surface_provider, "grids"):
            try:
                self._topo_grid = self._surface_provider.grids().topo
            except Exception:
                self._topo_grid = None
        self._dyn     = dynamics_engine or self._build_dynamics()

    def _publish_progress(
        self,
        *,
        stage: str,
        stage_fraction: float,
        total_samples: int,
        done_samples: float,
        elapsed_s: float,
        backend: str,
        batch_index: int | None = None,
        batch_count: int | None = None,
        detail: str = "",
    ) -> None:
        """
        Emit a structured progress payload for the UI layer.

        Progress is modeled in phases rather than as a single opaque counter:
        sampling, propagating, and writing each occupy a weighted slice of the
        full run.  This keeps the progress bar visually honest and avoids the
        "stuck at 99%" anti-pattern.
        """

        if self._cb is None:
            return

        stage_offsets = {
            "sampling": (0.00, 0.05),
            "propagating": (0.05, 0.90),
            "writing": (0.95, 0.05),
            "finalizing": (0.995, 0.005),
        }
        offset, weight = stage_offsets.get(stage, (0.0, 1.0))
        stage_fraction = max(0.0, min(1.0, float(stage_fraction)))
        overall_fraction = max(0.0, min(1.0, offset + weight * stage_fraction))
        eta_s: float | None = None
        if overall_fraction > 1.0e-6:
            eta_s = max(0.0, float(elapsed_s) * (1.0 - overall_fraction) / overall_fraction)

        payload = {
            "stage": str(stage),
            "percent": round(overall_fraction * 100.0, 3),
            "fraction": overall_fraction,
            "done_samples": float(done_samples),
            "total_samples": int(total_samples),
            "elapsed_s": round(float(elapsed_s), 3),
            "eta_s": (round(float(eta_s), 3) if eta_s is not None else None),
            "backend": str(backend),
            "detail": str(detail),
        }
        if batch_index is not None:
            payload["batch_index"] = int(batch_index)
        if batch_count is not None:
            payload["batch_count"] = int(batch_count)

        self._cb(payload)

    # ----------------------------------------------------------------
    # Internal: build dynamics engine from SimConfig
    # ----------------------------------------------------------------

    def _build_dynamics(self) -> Any:
        """
        Lazily build a DynamicsEngine from the stored SimConfig.

        The MC path intentionally reuses the same gravity / ephemeris bootstrap
        policy as the single-run path so users do not hit "works in Run, breaks
        in Monte Carlo" divergences.
        """
        from lunaris.core.dynamics import DynamicsEngine

        cfg = self._sim_cfg
        mc_backend = str(getattr(self._mc, "mc_backend", "auto") or "auto")
        mc_forces_classic_sh = mc_backend in {"cpu_sh", "gpu_sh", "numba_cuda_sh", "torch_cuda_sh", "torch_cpu_sh"}
        mc_forces_st_lrps = mc_backend in {"gpu_st_lrps_potential", "gpu_st_lrps_direct"}
        grav_model = None
        ephem_manager = None
        use_st_lrps_gravity = False
        surface_provider = self._surface_provider
        topo_requested = _surface_topography_requested(surface_provider, self._topo_grid)

        if bool(cfg.flags.enable_sh):
            try:
                use_st_lrps_gravity = (
                    mc_forces_st_lrps
                    or (
                        not mc_forces_classic_sh
                        and bool(getattr(cfg.gravity, "uses_st_lrps", False))
                    )
                )
                if use_st_lrps_gravity:
                    from lunaris.surrogate.runtime_adapter import SurrogateGravityModel

                    # Prioritize the MC-specific ST-LRPS run directory if provided.
                    st_lrps_dir = self._mc.st_lrps_model_dir or cfg.gravity.st_lrps_model_dir

                    from lunaris.common.montecarlo_defs import validate_st_lrps_model_dir
                    valid_dir = validate_st_lrps_model_dir(st_lrps_dir)

                    grav_model = SurrogateGravityModel.from_model_dir(
                        str(valid_dir),
                        mu_override=float(MU_MOON),
                        r_ref_override=float(R_MOON),
                        device_preference="cpu",
                    )
                else:
                    from lunaris.physics.spherical_harmonics import GravityModel

                    requested_degree = int(cfg.gravity.degree) if cfg.gravity.degree is not None else None

                    # The classic-SH GPU batch paths (numba_cuda_sh /
                    # torch_cuda_sh / torch_cpu_sh, or use_gpu auto) evaluate SH
                    # up to mc.gpu_sh_degree. Load coefficients to at least that
                    # degree (clamped to the file's own max by the loader) so a
                    # high gpu_sh_degree is not rejected by the propagator
                    # preflight merely because the mission's nominal degree is
                    # lower. Pure-CPU runs keep the mission degree unchanged so
                    # their physics is not silently altered.
                    mc_gpu_degree = int(getattr(self._mc, "gpu_sh_degree", 0) or 0)
                    gpu_sh_path_requested = (
                        mc_backend in {"gpu_sh", "numba_cuda_sh", "torch_cuda_sh", "torch_cpu_sh"}
                        or (mc_backend == "auto" and bool(getattr(self._mc, "use_gpu", False)))
                    )
                    if gpu_sh_path_requested and mc_gpu_degree > 0:
                        requested_degree = (
                            mc_gpu_degree
                            if requested_degree is None
                            else max(requested_degree, mc_gpu_degree)
                        )

                    # GravityModel already exposes the full dynamics gravity
                    # contract (degree_max, R_ref_m, GM_m3s2, Cnm ... ws).
                    grav_model = GravityModel.from_file(
                        path=str(cfg.gravity.file_path),
                        requested_degree=requested_degree,
                    )
            except Exception as exc:
                raise RuntimeError(
                    f"Monte Carlo bootstrap failed: Could not load gravity model.\n"
                    f"ST-LRPS mode: {getattr(cfg.gravity, 'uses_st_lrps', False)}\n"
                    f"Error: {exc}"
                ) from exc

        if _need_ephemeris(cfg, topo_requested=topo_requested):
            try:
                ephem_manager = _build_ephemeris_manager(cfg)
            except Exception as exc:
                raise RuntimeError(
                    f"Monte Carlo bootstrap failed: Could not load ephemeris.\n"
                    f"Error: {exc}"
                ) from exc

        earth_j2 = getattr(cfg, "earth_j2", None)
        thermal = getattr(cfg, "thermal", None)
        solid_tides = getattr(cfg, "solid_tides", None)

        return DynamicsEngine(
            sc_props=cfg.spacecraft,
            flags=cfg.flags,
            gravity_model=grav_model,
            gravity_adaptive=(
                None if use_st_lrps_gravity
                else getattr(cfg.gravity, "adaptive", None)
            ),
            ephem_manager=ephem_manager,
            surface_provider=surface_provider,
            earth_j2=earth_j2,
            thermal=thermal,
            solid_tides=solid_tides,
            allow_identity_rotation=(ephem_manager is None),
        )

    # ----------------------------------------------------------------
    # Internal: select and initialise backend
    # ----------------------------------------------------------------

    def _resolve_topo_payload(self) -> dict[str, Any] | None:
        """Topography payload for terrain-aware impact freeze, or ``None``.

        Built only when the config requests ``impact_surface_mode='terrain'`` AND
        a topography grid/provider is available; otherwise the batch backends keep
        the constant-sphere impact freeze (zero behaviour change). The payload is
        the same POD contract the CPU ground-truth event consumes, so all backends
        share one terrain definition.
        """
        if not bool(getattr(self._mc, "impact_surface_terrain_enabled", False)):
            return None

        prov = self._surface_provider
        if prov is not None and hasattr(prov, "topo_payload"):
            try:
                payload = prov.topo_payload()
            except Exception:
                payload = None
            if payload is not None and payload.get("dn", None) is not None:
                return payload

        if self._topo_grid is not None:
            from lunaris.loaders.io_surface import _grid_topo_payload
            return _grid_topo_payload(self._topo_grid)
        return None

    def _build_propagator(self) -> Any:
        """
        Instantiate the appropriate batch propagator using the backend policy.

        Backend selection is fully delegated to
        ``core.mc_backend_policy.resolve_mc_backend_policy`` so the routing
        logic is testable in isolation without constructing a full engine.
        """
        from lunaris.core.mc_backend_policy import MCBackend, resolve_mc_backend_policy
        from lunaris.core.mc_propagator import CPUBatchPropagator

        plan = resolve_mc_backend_policy(self._mc, self._sim_cfg)
        self._backend_plan = plan

        # Terrain-aware impact freeze payload (None unless requested + available).
        # Shared across every batch backend so they agree on the surface.
        topo_payload = self._resolve_topo_payload()

        # Emit all warnings produced by the policy resolver
        for w in plan.warnings:
            warnings.warn(w, RuntimeWarning, stacklevel=2)
            self._backend_note = w  # keep the most recent one for the run log

        # Log the resolved plan
        plan.log_summary()

        # ----------------------------------------------------------------
        # GPU ST-LRPS path — PyTorch fixed-step RK4
        # ----------------------------------------------------------------
        if plan.final_backend == MCBackend.GPU_ST_LRPS:
            try:
                from lunaris.core.torch_batch_propagator import (
                    TorchBatchPropagator,
                    TorchSTLRPSPreflightError,
                )

                grav_model = getattr(self._dyn, "grav", None)
                if grav_model is None or getattr(grav_model, "model_kind", None) != "st_lrps":
                    raise RuntimeError(
                        "GPU ST-LRPS backend selected but no SurrogateGravityModel "
                        "is attached to the dynamics engine."
                    )
                deg_min = getattr(grav_model, "degree_min", "?")
                deg_max = getattr(grav_model, "degree_max", "?")
                print(
                    f"[MC][GPU-STLRPS] Loading surrogate: degree_min={deg_min}  "
                    f"degree_max={deg_max}  model_dir={grav_model.model_dir}",
                    flush=True,
                )
                actual_runtime_kind = str(
                    getattr(getattr(grav_model, "_force_runtime", None), "runtime_model_kind", "")
                    or getattr(grav_model, "config", {}).get("runtime_model_kind", "")
                ).strip()
                expected_runtime_kind = str(getattr(plan, "runtime_model_kind", "") or "").strip()
                if (
                    expected_runtime_kind
                    and actual_runtime_kind
                    and actual_runtime_kind != expected_runtime_kind
                ):
                    raise TorchSTLRPSPreflightError(
                        "GPU ST-LRPS artifact kind mismatch: backend policy expects "
                        f"{expected_runtime_kind!r}, loaded runtime is {actual_runtime_kind!r}."
                    )
                prop_kwargs: dict[str, Any] = {
                    "surrogate_model": grav_model,
                    "mc_cfg": self._mc,
                    "device_id": int(getattr(self._mc, "gpu_device_id", 0)),
                }
                constructor_params = inspect.signature(TorchBatchPropagator).parameters
                if "ephem" in constructor_params:
                    prop_kwargs["ephem"] = getattr(self._dyn, "ephem", None)
                if "allow_identity_rotation" in constructor_params:
                    prop_kwargs["allow_identity_rotation"] = bool(
                        getattr(self._dyn, "allow_identity_rotation", False)
                    )
                if "topo_payload" in constructor_params and topo_payload is not None:
                    prop_kwargs["topo_payload"] = topo_payload
                return TorchBatchPropagator(**prop_kwargs)
            except TorchSTLRPSPreflightError:
                raise
            except Exception as exc:
                note = (
                    f"[MC] GPU ST-LRPS backend initialization failed ({exc}). "
                    "Falling back to the CPU full-fidelity backend."
                )
                self._backend_note = note
                warnings.warn(note, RuntimeWarning, stacklevel=2)
                self._downgrade_plan_to_cpu(plan, note)

        # ----------------------------------------------------------------
        # GPU classic-SH path — Numba CUDA fixed-step RK4 (torch_cuda_sh's sibling)
        # ----------------------------------------------------------------
        elif plan.final_backend == MCBackend.GPU_CLASSIC_SH:
            try:
                from lunaris.core.mc_propagator import GPUBatchPropagator

                gpu_kwargs: dict[str, Any] = {}
                if "topo_payload" in inspect.signature(GPUBatchPropagator).parameters and topo_payload is not None:
                    gpu_kwargs["topo_payload"] = topo_payload
                return GPUBatchPropagator(
                    self._dyn,
                    self._mc,
                    self._sim_cfg.flags,
                    **gpu_kwargs,
                )
            except Exception as exc:
                note = (
                    f"[MC] GPU classic-SH backend initialization failed ({exc}). "
                    "Falling back to the CPU full-fidelity backend."
                )
                self._backend_note = note
                warnings.warn(note, RuntimeWarning, stacklevel=2)
                self._downgrade_plan_to_cpu(plan, note)

        # ----------------------------------------------------------------
        # GPU torch classic-SH path — PyTorch fixed-step RK4 (high-degree)
        # ----------------------------------------------------------------
        elif plan.final_backend == MCBackend.GPU_TORCH_SH:
            from lunaris.core.torch_sh_propagator import (
                TorchSHBatchPropagator,
                TorchSHPreflightError,
            )

            try:
                return TorchSHBatchPropagator(
                    self._dyn,
                    self._mc,
                    self._sim_cfg.flags,
                    device=f"cuda:{int(getattr(self._mc, 'gpu_device_id', 0) or 0)}",
                    topo_payload=topo_payload,
                )
            except TorchSHPreflightError:
                # Hard contract violation (degree above the coefficient file,
                # unsupported physics, missing model). Never silently fall back —
                # surface it so the requested degree is not quietly reduced.
                raise
            except Exception as exc:
                note = (
                    f"[MC] torch_cuda_sh backend initialization failed ({exc}). "
                    "Falling back to the CPU full-fidelity backend."
                )
                self._backend_note = note
                warnings.warn(note, RuntimeWarning, stacklevel=2)
                self._downgrade_plan_to_cpu(plan, note)

        # ----------------------------------------------------------------
        # Torch CPU classic-SH path — PyTorch fixed-step RK4 on CPU
        # ----------------------------------------------------------------
        elif plan.final_backend == MCBackend.TORCH_CPU_SH:
            from lunaris.core.torch_sh_propagator import (
                TorchSHBatchPropagator,
                TorchSHPreflightError,
            )

            try:
                return TorchSHBatchPropagator(
                    self._dyn,
                    self._mc,
                    self._sim_cfg.flags,
                    device="cpu",
                    topo_payload=topo_payload,
                )
            except TorchSHPreflightError:
                raise
            except Exception as exc:
                note = (
                    f"[MC] torch_cpu_sh backend initialization failed ({exc}). "
                    "Falling back to the CPU full-fidelity backend."
                )
                self._backend_note = note
                warnings.warn(note, RuntimeWarning, stacklevel=2)
                self._downgrade_plan_to_cpu(plan, note)

        # ----------------------------------------------------------------
        # CPU path (default / fallback)
        # ----------------------------------------------------------------
        return CPUBatchPropagator(
            self._sim_cfg,
            self._mc,
            dynamics_template=self._dyn,
            surface_provider=self._surface_provider,
            topo_grid=self._topo_grid,
        )

    @staticmethod
    def _downgrade_plan_to_cpu(plan: Any, reason: str) -> None:
        """Rewrite a backend plan to CPU after a GPU propagator failed to build.

        Keeps provenance honest: a run that actually executes on CPU must not be
        labeled with a GPU backend, device, or integrator (task §13).
        """
        from lunaris.core.mc_backend_policy import MCBackend

        plan.final_backend = MCBackend.CPU
        plan.use_gpu = False
        plan.actual_backend = "cpu_st_lrps" if plan.gravity_backend == "st_lrps" else "cpu_sh"
        plan.actual_sh_degree = None
        plan.actual_device = "cpu"
        plan.cuda_device_name = None
        plan.dtype = "float64"
        plan.integrator = "adaptive (DOP853)"
        plan.fallback_applied = True
        plan.fallback_reason = reason
        # Refresh family/implementation labels for the new actual backend.
        try:
            from lunaris.core.backend_capabilities import get_capabilities

            caps = get_capabilities(plan.actual_backend)
            plan.backend_family = caps.family
            plan.backend_implementation = caps.implementation
        except Exception:
            pass

    # ----------------------------------------------------------------
    # Public: run
    # ----------------------------------------------------------------

    def run(self) -> MCRunResult:
        """
        Execute the full Monte Carlo simulation.

        Returns
        -------
        MCRunResult
            Ensemble trajectories, spacecraft samples, impact bookkeeping.
        """
        mc  = self._mc
        cfg = self._sim_cfg
        N   = int(mc.n_samples)

        t_wall0 = time.perf_counter()
        rng = np.random.default_rng(int(mc.seed))
        self._publish_progress(
            stage="sampling",
            stage_fraction=0.0,
            total_samples=N,
            done_samples=0.0,
            elapsed_s=0.0,
            backend="pending",
            detail="Preparing ensemble sample set",
        )

        # -----------------------------------------------------------------
        # 1) Generate samples
        # -----------------------------------------------------------------
        nominal = _state_to_array(cfg.initial_state)   # (6,)
        sampling_method = str(getattr(mc, "sampling_method", "random") or "random")
        qmc_design_note = _sobol_size_note(sampling_method, N)
        if sampling_method == "random":
            joint_standard_normals = None
        else:
            joint_standard_normals = generate_standard_normal_design(
                N,
                10,
                sampling_method,
                int(mc.seed),
            )

        Y0 = sample_initial_states(
            nominal,
            mc.state,
            N,
            rng,
            sampling_method=sampling_method,
            seed=int(mc.seed),
            standard_normal_samples=(
                None if joint_standard_normals is None else joint_standard_normals[:, :6]
            ),
        )
        sc_samples = sample_spacecraft_props(
            nominal_mass=float(cfg.spacecraft.mass_kg),
            nominal_area=float(cfg.spacecraft.area_m2),
            nominal_cd=float(cfg.spacecraft.cd),
            nominal_cr=float(cfg.spacecraft.cr),
            uncertainty=mc.spacecraft,
            n_samples=N,
            rng=rng,
            sampling_method=sampling_method,
            seed=int(mc.seed) + 1,
            standard_normal_samples=(
                None if joint_standard_normals is None else joint_standard_normals[:, 6:10]
            ),
        )

        masses = sc_samples[:, 0]
        areas  = sc_samples[:, 1]
        cds    = sc_samples[:, 2]
        crs    = sc_samples[:, 3]
        self._publish_progress(
            stage="sampling",
            stage_fraction=1.0,
            total_samples=N,
            done_samples=0.0,
            elapsed_s=time.perf_counter() - t_wall0,
            backend="pending",
            detail=f"Samples generated ({sampling_method})",
        )

        # -----------------------------------------------------------------
        # 2) Propagator + output storage contract
        # -----------------------------------------------------------------
        prop   = self._build_propagator()

        # Fail-fast: validate gravity model contract before entering the sample
        # loop.  Without this check the CPU propagator catches the same missing-
        # attribute error N times and prints N identical "Sample i failed" lines.
        if hasattr(prop, "validate_gravity_assets"):
            prop.validate_gravity_assets()

        # Human-readable backend label derived from the *resolved plan's* actual
        # backend — never from the propagator class name. The class name cannot
        # distinguish torch_cuda_sh from torch_cpu_sh (same TorchSHBatchPropagator
        # class) and would mislabel a CPU run as GPU. After a GPU-build failure the
        # plan has already been downgraded to CPU (see _downgrade_plan_to_cpu), so
        # this label stays consistent with what actually executes.
        _plan_actual = str(getattr(self._backend_plan, "actual_backend", "") or "")
        backend_name = _BACKEND_DISPLAY_NAMES.get(_plan_actual, "CPU")
        backend_diag = prop.diagnostics_snapshot() if hasattr(prop, "diagnostics_snapshot") else {}
        backend_plan = getattr(self, "_backend_plan", None)
        requested_max_batch = mc.effective_max_batch()
        if hasattr(prop, "recommended_max_batch"):
            max_batch = int(prop.recommended_max_batch(requested_max_batch))
        else:
            max_batch = int(requested_max_batch)

        duration_s  = float(cfg.time.duration_s)
        output_dt_s = float(cfg.time.output_dt_s or mc.dt_s * 10)
        t_out_contract, _, _ = build_mc_output_grid(duration_s, output_dt_s)
        storage_mode, result_bytes, memory_limit_bytes = _resolve_result_storage(
            mc,
            len(t_out_contract),
        )
        writer = _make_writer(mc, N, t_out_contract)

        print(
            f"[MC] N={N}  backend={backend_name}  "
            f"T={duration_s / DAY_S:.2f} d  "
            f"step={mc.dt_s:.1f} s  snap={output_dt_s:.1f} s",
            flush=True,
        )
        if self._backend_note:
            print(self._backend_note, flush=True)
        if backend_diag:
            device_name = str(backend_diag.get("device_name", "")).strip()
            tpb = backend_diag.get("threads_per_block")
            if device_name:
                print(
                    f"[MC] runtime device={device_name}  tpb={tpb}  "
                    f"batch_cap~{max_batch}",
                    flush=True,
                )

        # -----------------------------------------------------------------
        # 3) Sub-batch loop (VRAM + host-RAM budget)
        # -----------------------------------------------------------------
        # The per-batch host buffer is (T, b_n, 6) float64. A batch that fits in
        # VRAM can still exhaust host RAM for long / high-cadence runs because the
        # VRAM cap only accounts for a single state vector per sample, not the full
        # snapshot history kept on the host. Bound the sample batch by the host
        # memory budget as well so max_batch never blows out resident memory. The
        # budget already folds in the available-RAM safety factor (see
        # _resolve_result_storage), so a busy host tightens the batch cap too.
        host_bytes_per_sample = len(t_out_contract) * 6 * np.dtype(np.float64).itemsize
        host_batch_cap = max(
            1, int(memory_limit_bytes / max(1, host_bytes_per_sample))
        )
        if host_batch_cap < max_batch:
            print(
                f"[MC] Host-RAM cap reduced batch {max_batch} -> {host_batch_cap} "
                f"(per-batch host buffer ~{host_bytes_per_sample / 1e6:.1f} MB/sample "
                f"x T={len(t_out_contract)}).",
                flush=True,
            )
            max_batch = host_batch_cap

        n_batches = math.ceil(N / max_batch)
        self._publish_progress(
            stage="propagating",
            stage_fraction=0.0,
            total_samples=N,
            done_samples=0.0,
            elapsed_s=time.perf_counter() - t_wall0,
            backend=backend_name,
            batch_count=n_batches if n_batches > 0 else None,
            detail="Propagation starting",
        )

        # Result arrays stay eager only in memory mode. Disk mode writes each
        # sample batch directly into the final HDF5 trajectory dataset.
        t_out_ref = t_out_contract
        writer_buffer = getattr(writer, "memory_buffer", None)
        Y_all = _allocate_result_buffer(
            storage_mode,
            writer_buffer,
            (len(t_out_ref), N, 6),
        )
        impact_all   = np.zeros(N, dtype=np.float64)
        t_impact_all = np.full(N, np.nan, dtype=np.float64)
        valid_all = np.zeros(N, dtype=np.float64)
        impact_position_inertial = np.full((N, 3), np.nan, dtype=np.float64)
        impact_position_fixed = np.full((N, 3), np.nan, dtype=np.float64)

        # Throughput accumulators across engine sub-batches. Aggregated as
        # total_state_steps / total_propagation_time (NOT an average of per-batch
        # rates) so the recorded diagnostics match the work actually done.
        _agg_raw_steps = 0
        _agg_active_steps = 0
        _agg_elapsed_s = 0.0

        for b_idx in range(n_batches):
            b_start = b_idx * max_batch
            b_end   = min(N, b_start + max_batch)
            b_n     = b_end - b_start

            print(
                f"[MC] Batch {b_idx + 1}/{n_batches}  "
                f"samples {b_start}–{b_end - 1}",
                flush=True,
            )

            # Loop variables are bound as defaults: the callback is invoked
            # synchronously within this iteration's propagate() call, but binding
            # makes that explicit and silences B023 (late-binding closure).
            def _batch_progress(
                local_fraction: float,
                _b_start: int = b_start,
                _b_n: int = b_n,
                _b_idx: int = b_idx,
            ) -> None:
                effective_done = float(_b_start) + float(_b_n) * max(0.0, min(1.0, float(local_fraction)))
                self._publish_progress(
                    stage="propagating",
                    stage_fraction=(effective_done / max(N, 1)),
                    total_samples=N,
                    done_samples=effective_done,
                    elapsed_s=time.perf_counter() - t_wall0,
                    backend=backend_name,
                    batch_index=_b_idx + 1,
                    batch_count=n_batches,
                    detail=f"Batch {_b_idx + 1}/{n_batches}",
                )

            try:
                t_b, Y_b, imp_b, t_imp_b = prop.propagate(
                    Y0[b_start:b_end],
                    masses[b_start:b_end],
                    areas[b_start:b_end],
                    cds[b_start:b_end],
                    crs[b_start:b_end],
                    duration_s=duration_s,
                    output_dt_s=output_dt_s,
                    callback=_batch_progress,
                )
            except Exception:
                writer.abort()
                raise

            # Accumulate this batch's throughput counters (only backends that
            # expose them populate these keys; others contribute nothing).
            if hasattr(prop, "diagnostics_snapshot"):
                _bd = prop.diagnostics_snapshot()
                _agg_raw_steps += int(_bd.get("total_raw_state_steps", 0) or 0)
                _agg_active_steps += int(_bd.get("total_active_state_steps", 0) or 0)
                _agg_elapsed_s += float(_bd.get("propagation_elapsed_s", 0.0) or 0.0)

            # Resample to reference grid if needed
            if len(t_b) == len(t_out_ref) and np.allclose(t_b, t_out_ref, rtol=1e-6):
                Y_ref = np.ascontiguousarray(Y_b, dtype=np.float64)
            else:
                # Linear interpolation to reference grid
                Y_ref = np.empty((len(t_out_ref), b_n, 6), dtype=np.float64)
                for j in range(b_n):
                    for c in range(6):
                        Y_ref[:, j, c] = np.interp(
                            t_out_ref, t_b, Y_b[:, j, c]
                        )

            impact_all[b_start:b_end] = imp_b
            valid_b = np.isfinite(Y_ref).all(axis=(0, 2))
            valid_all[b_start:b_end] = valid_b.astype(np.float64)

            batch_impact_positions = np.full((b_n, 3), np.nan, dtype=np.float64)
            if hasattr(prop, "last_impact_positions_inertial"):
                candidate_positions = np.asarray(
                    prop.last_impact_positions_inertial(), dtype=np.float64
                )
                if candidate_positions.shape == (b_n, 3):
                    batch_impact_positions[:] = candidate_positions
            for j in range(b_n):
                if (
                    imp_b[j] > 0.5
                    and not np.isfinite(batch_impact_positions[j]).all()
                ):
                    if np.isfinite(t_imp_b[j]):
                        hit_idx = int(np.argmin(np.abs(t_b - float(t_imp_b[j]))))
                    else:
                        radii = np.linalg.norm(Y_b[:, j, :3], axis=1)
                        hits = np.where(
                            radii
                            <= float(R_MOON) + float(mc.impact_alt_km) * 1_000.0
                        )[0]
                        hit_idx = int(hits[0]) if hits.size else len(t_b) - 1
                        t_imp_b[j] = float(t_b[hit_idx])
                    batch_impact_positions[j] = Y_b[hit_idx, j, :3]
            t_impact_all[b_start:b_end] = t_imp_b
            impact_position_inertial[b_start:b_end] = batch_impact_positions
            impact_position_fixed[b_start:b_end] = _impact_positions_fixed(
                getattr(self._dyn, "ephem", None),
                np.asarray(t_imp_b, dtype=np.float64),
                batch_impact_positions,
            )

            try:
                writer.write_sample_batch(b_start, b_end, Y_ref)
            except Exception:
                writer.abort()
                raise
            if Y_all is not None and Y_all is not writer_buffer:
                Y_all[:, b_start:b_end, :] = Y_ref

            self._publish_progress(
                stage="propagating",
                stage_fraction=(float(b_end) / max(N, 1)),
                total_samples=N,
                done_samples=float(b_end),
                elapsed_s=time.perf_counter() - t_wall0,
                backend=backend_name,
                batch_index=b_idx + 1,
                batch_count=n_batches,
                detail=f"Batch {b_idx + 1}/{n_batches} complete",
            )

        # -----------------------------------------------------------------
        # 3b) Refresh diagnostics AFTER propagation (throughput is only known
        #     post-run) and fold in the cross-batch aggregate. The pre-run
        #     snapshot above carried static device info but no throughput.
        # -----------------------------------------------------------------
        if hasattr(prop, "diagnostics_snapshot"):
            backend_diag = dict(prop.diagnostics_snapshot())
            if _agg_elapsed_s > 0.0:
                backend_diag["total_raw_state_steps"] = _agg_raw_steps
                backend_diag["total_active_state_steps"] = _agg_active_steps
                backend_diag["propagation_elapsed_s"] = _agg_elapsed_s
                backend_diag["raw_batch_state_steps_per_second"] = _agg_raw_steps / _agg_elapsed_s
                backend_diag["active_state_steps_per_second"] = _agg_active_steps / _agg_elapsed_s

        # -----------------------------------------------------------------
        # 4) Finalize archive metadata
        # -----------------------------------------------------------------
        # Collect ST-LRPS provenance metadata when the surrogate backend is active.
        _grav_model = getattr(self._dyn, "grav", None)
        _st_lrps_meta: dict[str, Any] = {}
        if getattr(_grav_model, "model_kind", None) == "st_lrps":
            _st_lrps_meta = {
                "gravity_backend": "st_lrps",
                "st_lrps_model_dir": str(getattr(_grav_model, "model_dir", "") or ""),
                "st_lrps_degree_min": getattr(_grav_model, "degree_min", None),
                "st_lrps_degree_max": getattr(_grav_model, "degree_max", None),
                "effective_degree_max": getattr(_grav_model, "effective_degree_max", None),
                "runtime_model_kind": str(
                    getattr(getattr(_grav_model, "_force_runtime", None), "runtime_model_kind", "")
                    or getattr(_grav_model, "config", {}).get("runtime_model_kind", "potential_autograd")
                ),
            }

        # Collect backend-plan provenance for the archive
        try:
            _plan = backend_plan
            if _plan is None:
                from lunaris.core.mc_backend_policy import resolve_mc_backend_policy as _resolve
                _plan = _resolve(mc, self._sim_cfg)
            actual_sh_degree = backend_diag.get("actual_gpu_sh_degree", backend_diag.get("gpu_sh_degree"))
            if actual_sh_degree is None and _grav_model is not None:
                actual_sh_degree = getattr(_grav_model, "effective_degree_max", getattr(_grav_model, "degree", None))
            _plan_meta: dict[str, Any] = {
                "requested_mc_backend": getattr(_plan, "requested_backend", "auto"),
                "actual_mc_backend": getattr(_plan, "actual_backend", _plan.final_backend.value),
                "mc_backend": _plan.final_backend.value,
                "backend_family": getattr(_plan, "backend_family", ""),
                "backend_implementation": backend_diag.get("backend_implementation")
                    or getattr(_plan, "backend_implementation", ""),
                "requested_use_gpu": bool(mc.use_gpu),
                "final_use_gpu": _plan.use_gpu,
                "plan_gravity_backend": _plan.gravity_backend,   # renamed: avoids collision with _st_lrps_meta["gravity_backend"]
                "requested_device": getattr(_plan, "requested_device", ""),
                "actual_device": backend_diag.get("device_name") or getattr(_plan, "actual_device", ""),
                "requested_sh_degree": getattr(_plan, "requested_sh_degree", int(mc.gpu_sh_degree)),
                "actual_sh_degree": actual_sh_degree,
                "gpu_sh_max_degree": getattr(_plan, "gpu_sh_max_degree", None),
                "gpu_sh_supported_tiers": list(getattr(_plan, "gpu_sh_supported_tiers", ())),
                "runtime_model_kind": _st_lrps_meta.get(
                    "runtime_model_kind",
                    getattr(_plan, "runtime_model_kind", None),
                ),
                "torch_cuda_available": _plan.torch_cuda_available,
                "numba_cuda_available": _plan.numba_cuda_available,
                "cuda_device_name": backend_diag.get("device_name") or getattr(_plan, "cuda_device_name", None),
                "dtype": backend_diag.get("dtype") or getattr(_plan, "dtype", "float64"),
                "state_dtype": backend_diag.get("state_dtype")
                    or backend_diag.get("dtype")
                    or getattr(_plan, "dtype", "float64"),
                "model_dtype": backend_diag.get("model_dtype"),
                "acceleration_output_dtype": backend_diag.get("acceleration_output_dtype"),
                "frame_mode": backend_diag.get("frame_mode", "unknown"),
                "integrator": backend_diag.get("integrator") or _plan.integrator,
                "batch_size": max_batch,
                "chunk_size": backend_diag.get("chunk_size", max_batch),
                "fallback_applied": bool(getattr(_plan, "fallback_applied", False)),
                "fallback_reason": (
                    getattr(_plan, "fallback_reason", "")
                    if bool(getattr(_plan, "fallback_applied", False))
                    else ""
                ),
                "selection_reason": getattr(_plan, "reason", ""),
                "physics_capabilities": _active_physics_capabilities(self._sim_cfg),
            }
        except Exception:
            # Even on the degenerate provenance path the required v2 manifest
            # fields must be present (and non-null) so the archive still loads
            # under load_mc_result(strict=True).
            _fallback_backend = str(getattr(mc, "mc_backend", "auto") or "auto")
            _plan_meta = {
                "requested_mc_backend": _fallback_backend,
                "actual_mc_backend": _fallback_backend,
                "mc_backend": _fallback_backend,
                "requested_sh_degree": int(mc.gpu_sh_degree),
            }

        # Artifact + coefficient + kernel hash provenance: a path string alone is
        # not reproducible evidence. Stamp content hashes so a reader can verify
        # exactly which weights, gravity coefficients, and GPU kernel produced
        # this archive. _sha256_file never raises (missing file -> None, dropped).
        _provenance_hashes: dict[str, Any] = {}
        if getattr(_grav_model, "model_kind", None) == "st_lrps":
            _force_runtime = getattr(_grav_model, "_force_runtime", None)
            _ckpt_path = getattr(_force_runtime, "checkpoint_path", None)
            if _ckpt_path:
                _provenance_hashes["st_lrps_checkpoint_sha256"] = _sha256_file(_ckpt_path)
            _model_dir = getattr(_grav_model, "model_dir", None)
            if _model_dir:
                _provenance_hashes["st_lrps_config_sha256"] = _sha256_file(
                    Path(_model_dir) / "config.json"
                )
            _run_manifest = getattr(_force_runtime, "run_manifest", {}) or {}
            for _key in ("checkpoint_hash", "scaler_hash", "training_config_hash"):
                _val = _run_manifest.get(_key)
                if _val:
                    _provenance_hashes[f"st_lrps_{_key}"] = _val
        _grav_file = getattr(getattr(cfg, "gravity", None), "file_path", None)
        if _grav_file:
            _provenance_hashes["sh_coefficient_sha256"] = _sha256_file(_grav_file)
        try:
            _provenance_hashes["kernel_module"] = str(getattr(type(prop), "__module__", "") or "")
            _provenance_hashes["kernel_source_sha256"] = _sha256_file(
                inspect.getsourcefile(type(prop))
            )
        except Exception:
            pass

        try:
            writer.write_metadata(
                archive_schema_version=2,
                n_samples=N,
                seed=int(mc.seed),
                sampling_method=sampling_method,
                sampling_note=qmc_design_note,
                duration_s=duration_s,
                output_dt_s=output_dt_s,
                requested_backend="GPU" if bool(mc.use_gpu) else "CPU",
                gpu_sh_degree=int(mc.gpu_sh_degree),
                backend=backend_name,
                backend_note=self._backend_note,
                backend_diagnostics=backend_diag,
                result_storage_mode=storage_mode,
                estimated_result_bytes=result_bytes,
                detect_impact=bool(mc.impact_detection_enabled),
                compute_impact_statistics=bool(mc.impact_statistics_enabled),
                impact_frame_available=bool(getattr(self._dyn, "ephem", None) is not None),
                **_provenance_hashes,
                **_st_lrps_meta,
                **_plan_meta,
            )
            writer.write_final(
                sc_samples,
                impact_all,
                t_impact_all,
                valid_all,
                impact_position_inertial,
                impact_position_fixed,
            )
            self._publish_progress(
                stage="writing",
                stage_fraction=1.0,
                total_samples=N,
                done_samples=float(N),
                elapsed_s=time.perf_counter() - t_wall0,
                backend=backend_name,
                batch_index=n_batches,
                batch_count=n_batches,
                detail="Finalizing archive",
            )
            writer.finalize()
        except Exception:
            writer.abort()
            raise

        t_wall = time.perf_counter() - t_wall0
        valid_bool = valid_all > 0.5
        n_valid = int(np.sum(valid_bool))
        n_hit = int(np.sum(valid_bool & (impact_all > 0.5)))
        impact_fraction = float(n_hit) / n_valid if n_valid else math.nan
        print(
            f"[MC] Done. Wall={t_wall:.1f}s  "
            f"impacts={n_hit}/{n_valid} "
            f"({100.0 * impact_fraction:.1f}%)",
            flush=True,
        )
        self._publish_progress(
            stage="finalizing",
            stage_fraction=1.0,
            total_samples=N,
            done_samples=float(N),
            elapsed_s=t_wall,
            backend=backend_name,
            batch_index=n_batches,
            batch_count=n_batches,
            detail="Run completed",
        )

        # -----------------------------------------------------------------
        # 5) Build result
        # -----------------------------------------------------------------
        if storage_mode == "disk":
            Y_result: Any = HDF5TrajectoryView(mc.output_path_resolved)
        else:
            if Y_all is None:
                raise RuntimeError("Eager MC result buffer was not initialized.")
            Y_result = Y_all
        n_failed = int(np.sum(valid_all < 0.5))

        return MCRunResult(
            t=t_out_ref,
            Y=Y_result,
            sc_samples=sc_samples,
            impact_mask=impact_all,
            t_impact=t_impact_all,
            valid_mask=valid_all,
            impact_position_inertial_m=impact_position_inertial,
            impact_position_fixed_m=impact_position_fixed,
            archive_path=str(mc.output_path_resolved),
            diagnostics={
                "wall_time_s": float(t_wall),
                "n_samples": N,
                "sampling_method": sampling_method,
                "sampling_note": qmc_design_note,
                "n_valid_samples": n_valid,
                "n_failed_samples": n_failed,
                "n_impacts": n_hit,
                "impact_fraction": impact_fraction,
                "impact_detection_enabled": bool(mc.impact_detection_enabled),
                "impact_statistics_enabled": bool(mc.impact_statistics_enabled),
                "impact_frame_available": bool(getattr(self._dyn, "ephem", None) is not None),
                "backend": backend_name,
                "backend_note": self._backend_note,
                "output_path": str(mc.output_path_resolved),
                "backend_diagnostics": backend_diag,
                # Throughput metrics from the batched propagator (if available)
                "raw_batch_state_steps_per_second": backend_diag.get("raw_batch_state_steps_per_second"),
                "active_state_steps_per_second": backend_diag.get("active_state_steps_per_second"),
                "requested_mc_backend": _plan_meta.get("requested_mc_backend"),
                "actual_mc_backend": _plan_meta.get("actual_mc_backend"),
                "requested_sh_degree": _plan_meta.get("requested_sh_degree"),
                "actual_sh_degree": _plan_meta.get("actual_sh_degree"),
                "runtime_model_kind": _plan_meta.get("runtime_model_kind"),
                "fallback_reason": _plan_meta.get("fallback_reason"),
                "selection_reason": _plan_meta.get("selection_reason"),
                "result_storage_mode": storage_mode,
            },
        )


# =============================================================================
# 4.             LOADER: read back a previously saved run
# =============================================================================

def _infer_valid_mask_from_dataset(dataset: Any, chunk_size: int = 256) -> np.ndarray:
    """Infer legacy validity without materializing the full trajectory."""
    n_samples = int(dataset.shape[1])
    valid = np.zeros(n_samples, dtype=np.float64)
    for start in range(0, n_samples, chunk_size):
        end = min(n_samples, start + chunk_size)
        block = np.asarray(dataset[:, start:end, :], dtype=np.float64)
        valid[start:end] = np.isfinite(block).all(axis=(0, 2)).astype(np.float64)
    return valid


def _validate_archive_v2_manifest(metadata: dict[str, Any]) -> None:
    """Enforce required manifest fields for schema-v2 Monte Carlo archives.

    Pre-v2 / legacy archives (missing ``archive_schema_version`` or < 2) are
    exempt and loaded best-effort. A v2 archive missing any required field is
    rejected so incomplete provenance never passes silently as a valid result.
    """
    raw_version = metadata.get("archive_schema_version")
    if raw_version is None:
        return
    try:
        version = int(raw_version)
    except (TypeError, ValueError):
        return
    if version < 2:
        return
    missing = [f for f in REQUIRED_ARCHIVE_V2_FIELDS if f not in metadata]
    if missing:
        raise ValueError(
            f"Monte Carlo archive declares schema v{version} but is missing required "
            f"manifest field(s): {', '.join(sorted(missing))}. The archive is incomplete "
            "or was not produced by a current MonteCarloEngine run. Pass strict=False to "
            "load it as a best-effort legacy archive."
        )


def load_mc_result(path: str, *, lazy: bool = False, strict: bool = True) -> MCRunResult:
    """
    Reload a saved ``MCRunResult`` from HDF5 or NPZ file.

    Parameters
    ----------
    path : str
        Path produced by ``MonteCarloEngine.run()``.
    lazy : bool
        Return a disk-backed, read-only trajectory view instead of loading the
        full ``Y`` ensemble into memory (HDF5 only).
    strict : bool
        When True (default), schema-v2 archives must carry every field in
        ``REQUIRED_ARCHIVE_V2_FIELDS`` or a ``ValueError`` is raised. Pre-v2
        archives are always loaded best-effort. Pass ``strict=False`` to load a
        partial/legacy archive without manifest enforcement.

    Returns
    -------
    MCRunResult
    """
    p = Path(path).expanduser().resolve()
    suffix = p.suffix.lower()

    if suffix in (".h5", ".hdf5"):
        try:
            import h5py
        except ImportError:
            raise ImportError("h5py required to read HDF5 MC output.") from None
        with h5py.File(str(p), "r") as f:
            t_arr  = np.asarray(f["t"],           dtype=np.float64)
            Y_arr: Any = (
                HDF5TrajectoryView(p)
                if lazy
                else np.asarray(f["Y"], dtype=np.float64)
            )
            sc     = np.asarray(f["sc_samples"],  dtype=np.float64)
            imask  = np.asarray(f["impact_flags"], dtype=np.float64)
            t_imp  = np.asarray(f["t_impact"],    dtype=np.float64)
            valid = (
                np.asarray(f["valid_mask"], dtype=np.float64)
                if "valid_mask" in f
                else _infer_valid_mask_from_dataset(f["Y"])
            )
            impact_i = (
                np.asarray(f["impact_position_inertial_m"], dtype=np.float64)
                if "impact_position_inertial_m" in f
                else None
            )
            impact_f = (
                np.asarray(f["impact_position_fixed_m"], dtype=np.float64)
                if "impact_position_fixed_m" in f
                else None
            )
            diagnostics = {
                str(key): _decode_metadata_value(value)
                for key, value in dict(f.attrs).items()
            }
        if strict:
            _validate_archive_v2_manifest(diagnostics)
        return MCRunResult(
            t=t_arr, Y=Y_arr, sc_samples=sc,
            impact_mask=imask, t_impact=t_imp,
            valid_mask=valid,
            impact_position_inertial_m=impact_i,
            impact_position_fixed_m=impact_f,
            archive_path=str(p),
            diagnostics=diagnostics,
        )

    if suffix == ".npz":
        with np.load(str(p), allow_pickle=False) as data:
            diagnostics = {}
            if "metadata_json" in data.files:
                diagnostics = _decode_archive_metadata(data["metadata_json"])
            if strict:
                _validate_archive_v2_manifest(diagnostics)
            return MCRunResult(
                t=data["t"],
                Y=data["Y"],
                sc_samples=data["sc_samples"],
                impact_mask=data["impact_flags"],
                t_impact=data["t_impact"],
                valid_mask=(
                    data["valid_mask"]
                    if "valid_mask" in data.files
                    else np.isfinite(data["Y"]).all(axis=(0, 2)).astype(np.float64)
                ),
                impact_position_inertial_m=(
                    data["impact_position_inertial_m"]
                    if "impact_position_inertial_m" in data.files
                    else None
                ),
                impact_position_fixed_m=(
                    data["impact_position_fixed_m"]
                    if "impact_position_fixed_m" in data.files
                    else None
                ),
                archive_path=str(p),
                diagnostics=diagnostics,
            )

    raise ValueError(f"Unrecognised MC output format: {suffix!r} (expected .h5 or .npz)")


def mc_entry() -> int:
    """Console-script entry point for batch/Monte Carlo ensemble propagation."""
    from lunaris.core.mc_runner import main as _mc_main

    return int(_mc_main())


def batch_entry() -> int:
    """Console-script alias for the batch propagation terminology."""
    return mc_entry()


# =============================================================================
# 5.                        PUBLIC API
# =============================================================================

__all__ = [
    "MonteCarloEngine",
    "generate_standard_normal_design",
    "sample_initial_states",
    "sample_spacecraft_props",
    "HDF5TrajectoryView",
    "load_mc_result",
    "mc_entry",
    "batch_entry",
]
