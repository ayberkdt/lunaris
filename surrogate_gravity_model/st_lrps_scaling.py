# -*- coding: utf-8 -*-
"""Origin-fixed isometric scaling for the lunar potential surrogate."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Tuple

import h5py
import numpy as np
import torch

try:
    from .dataset_parameters import MU_MOON_SI, R_MOON_SI, is_lunar_body_signature
except ImportError:  # pragma: no cover - script execution fallback
    from dataset_parameters import MU_MOON_SI, R_MOON_SI, is_lunar_body_signature


logger = logging.getLogger(__name__)

@dataclass
class IsometricScaleParams:
    """Per-axis mean + single global characteristic scale for one quantity."""
    mean: List[float]       # per-axis mean (centroid)
    scale: float            # single global characteristic scale

    def to_tensors(self, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        mean_t = torch.tensor(self.mean, device=device, dtype=dtype, requires_grad=False)
        scale_t = torch.tensor([self.scale], device=device, dtype=dtype, requires_grad=False)
        return mean_t, scale_t

@dataclass
class ScalerPack:
    """Bundle of isometric scalers for inputs (x) and targets (u, a)."""
    x: IsometricScaleParams
    u: IsometricScaleParams
    a: IsometricScaleParams

    def save_json(self, path: Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)

    @staticmethod
    def load_json(path: Path) -> "ScalerPack":
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return ScalerPack(
            x=IsometricScaleParams(**d["x"]),
            u=IsometricScaleParams(**d["u"]),
            a=IsometricScaleParams(**d["a"]),
        )

    @staticmethod
    def load(path: Path, device: torch.device, dtype: torch.dtype) -> "ScalerPack":
        """Load the historical scaler.json format and cache device tensors."""
        return ScalerPack.load_json(path).to_tensors(device=device, dtype=dtype)

    def to_tensors(self, device: torch.device, dtype: torch.dtype) -> "ScalerPack":
        self._x_mean, self._x_scale = self.x.to_tensors(device, dtype)
        self._u_mean, self._u_scale = self.u.to_tensors(device, dtype)
        self._a_mean, self._a_scale = self.a.to_tensors(device, dtype)
        return self

    def _ensure_tensors(self, ref: torch.Tensor) -> None:
        # Re-create cached tensors whenever device or dtype changes (e.g. CPU→CUDA).
        needs = (
            not hasattr(self, "_x_mean")
            or self._x_mean.device != ref.device
            or self._x_mean.dtype != ref.dtype
        )
        if needs:
            self.to_tensors(device=ref.device, dtype=ref.dtype)

    def scale_x(self, x: torch.Tensor) -> torch.Tensor:
        self._ensure_tensors(x)
        return (x - self._x_mean) / self._x_scale

    def scale_u(self, u: torch.Tensor) -> torch.Tensor:
        self._ensure_tensors(u)
        return (u - self._u_mean) / self._u_scale

    def unscale_u(self, u_scaled: torch.Tensor) -> torch.Tensor:
        self._ensure_tensors(u_scaled)
        return u_scaled * self._u_scale + self._u_mean

    def scale_a(self, a: torch.Tensor) -> torch.Tensor:
        self._ensure_tensors(a)
        return (a - self._a_mean) / self._a_scale

    def unscale_a(self, a_scaled: torch.Tensor) -> torch.Tensor:
        self._ensure_tensors(a_scaled)
        return a_scaled * self._a_scale + self._a_mean

class OnlineIsometricStats:
    """Streaming Welford mean + running max-norm for isometric scale fitting."""
    def __init__(self, dim: int):
        self.dim = int(dim)
        self.n = 0
        self.mean = np.zeros(self.dim, dtype=np.float64)
        self.M2 = np.zeros(self.dim, dtype=np.float64)
        self.max_norm: float = 0.0

    def update(self, batch: np.ndarray) -> None:
        batch = np.asarray(batch, dtype=np.float64)
        if batch.ndim == 1:
            batch = batch.reshape(1, -1)
        n_b = batch.shape[0]
        if n_b == 0:
            return

        # Welford for mean
        mean_b = np.mean(batch, axis=0)
        m2_b = np.sum((batch - mean_b) ** 2, axis=0)
        n_new = self.n + n_b
        delta = mean_b - self.mean
        self.mean += delta * (n_b / n_new)
        self.M2 += m2_b + (delta ** 2) * (self.n * n_b / n_new)
        self.n = n_new

        # Running max-norm (centered around current mean estimate)
        centered = batch - self.mean  # use latest mean estimate
        norms = np.linalg.norm(centered, axis=1)
        batch_max = float(np.max(norms))
        if batch_max > self.max_norm:
            self.max_norm = batch_max

    def finalize(
        self,
        eps: float = 1e-12,
        *,
        mode: str = "max",
        multiplier: float = 1.0,
    ) -> Tuple[np.ndarray, float]:
        """
        Return ``(mean, scale)`` using a physically motivated single-scalar rule.

        Parameters
        ----------
        mode:
            ``"max"`` keeps the historical max-norm behaviour.

            ``"rms"`` uses the population RMS norm around the mean multiplied
            by ``multiplier``.

            ``"hybrid"`` uses ``min(max_norm, multiplier * rms_norm)``. This
            is the preferred setting for targets because it remains isotropic
            while preventing a handful of extreme samples from shrinking the
            entire learning signal.
        multiplier:
            Expansion factor applied to RMS-based scales. Values around ``6``
            work well here: they still cover the bulk of the target dynamic
            range without letting rare outliers dominate.
        """

        rms_scale = 0.0
        if self.n > 0:
            variances = self.M2 / float(self.n)
            rms_scale = float(np.sqrt(np.sum(variances)))

        max_scale = self.max_norm if self.max_norm > eps else 0.0
        rms_scaled = max(eps, float(multiplier) * max(rms_scale, eps))

        mode_l = str(mode).strip().lower()
        if mode_l == "max":
            scale = max(max_scale, eps)
        elif mode_l == "rms":
            scale = rms_scaled
        elif mode_l == "hybrid":
            if max_scale > eps:
                scale = min(max_scale, rms_scaled)
            else:
                scale = rms_scaled
        else:
            raise ValueError(f"Unknown scale mode: {mode!r}")

        return self.mean, float(max(scale, eps))


# --- HDF5 helpers & DataLoader ---

def compute_base_potential(x_phys: torch.Tensor, mu: float, a_sign: float, degree_min: int = -1) -> torch.Tensor:
    """
    If degree_min >= 0, point-mass is already excluded from the dataset.
    """
    if degree_min >= 0:
        return torch.zeros((x_phys.shape[0], 1), device=x_phys.device, dtype=x_phys.dtype)
    r_norm = torch.norm(x_phys, dim=1, keepdim=True).clamp(min=1.0)
    return a_sign * (mu / r_norm)

def compute_base_accel(x_phys: torch.Tensor, mu: float, degree_min: int = -1) -> torch.Tensor:
    if degree_min >= 0:
        return torch.zeros((x_phys.shape[0], 3), device=x_phys.device, dtype=x_phys.dtype)
    r_norm = torch.norm(x_phys, dim=1, keepdim=True).clamp(min=1.0)
    return -mu * x_phys / (r_norm ** 3)


# --- GradNorm loss balancing (Chen et al. 2018) ---
# Equalises ‖∂L_U/∂W‖ and ‖∂L_a/∂W‖ at the last hidden layer.
# Amortised: expensive autograd only every update_interval steps.

def fit_scaler_streaming(
    h5_path: Path,
    dset_name: str,
    meta: "DatasetMeta",
    use_si: bool,
    mu_si: float,
    a_sign: float,
    n_fit: int = 500_000,
    seed: int = 0,
    chunk_rows: int = 131_072,
    degree_min: int = -1,
) -> "ScalerPack":
    """Stream-fit isometric scalers on residuals ΔU/Δa (baseline already subtracted)."""
    logger.info(f"Fitting isometric scaler on {n_fit:,} rows from '{h5_path.name}'...")
    logger.info(f"  Residual mode: subtracting point-mass baseline (mu_si={mu_si:.6e}, a_sign={a_sign:+.1f})")
    rng = np.random.default_rng(seed)
    
    x_stats = OnlineIsometricStats(3)
    u_stats = OnlineIsometricStats(1)   # will receive ΔU = U - U_base
    a_stats = OnlineIsometricStats(3)   # will receive Δa = a - a_base

    max_r_from_origin: float = 0.0  # max ‖x‖ tracked independently to fix origin at Moon CoM

    with h5py.File(h5_path, "r", libver="latest", swmr=True) as f:
        ds = f[dset_name]
        total_rows = int(ds.shape[0])
        rows_to_use = min(int(n_fit), total_rows)

        seen_rows = 0
        while seen_rows < rows_to_use:
            block_size = min(chunk_rows, rows_to_use - seen_rows)
            start_idx = int(rng.integers(0, max(total_rows - block_size, 1)))

            arr = np.asarray(ds[start_idx : start_idx + block_size, :], dtype=np.float64)

            x = arr[:, 0:3]
            u = arr[:, 3:4]
            a = arr[:, 4:7]

            if use_si and meta.unit_system == "canonical":
                x, u, a = meta.convert_xyz_U_a_to_si(x, u, a)

            # Track max ‖x‖ from origin (not from running mean) for SH-correct scaling
            batch_max_r = float(np.max(np.linalg.norm(x, axis=1)))
            if batch_max_r > max_r_from_origin:
                max_r_from_origin = batch_max_r

            # Subtract baseline so scaler is fitted on residuals
            x_t = torch.as_tensor(x, dtype=torch.float64)
            u_base = compute_base_potential(x_t, mu_si, a_sign, degree_min).numpy()   # (B, 1)
            a_base = compute_base_accel(x_t, mu_si, degree_min).numpy()              # (B, 3)

            delta_u = u - u_base    # residual potential
            delta_a = a - a_base    # residual acceleration

            x_stats.update(x)
            u_stats.update(delta_u)
            a_stats.update(delta_a)

            seen_rows += block_size

    # x_mean is fixed to [0,0,0]: shifting the coordinate origin away from Moon's CoM
    # breaks the 1/r symmetry that SH expansions depend on.
    # x_scale = max ‖x‖ (from origin, not from data mean) preserves ΔU isotropy.
    x_mean = np.zeros(3, dtype=np.float64)
    x_scale = max(max_r_from_origin, 1e-12)

    u_mean, u_scale = u_stats.finalize(mode="max")
    a_mean, a_scale = a_stats.finalize(mode="max")

    logger.info(f"  x : mean=[0,0,0] (fixed -> Moon CoM), max_r={x_scale:.3e} m")
    logger.info(f"  dU: mean={u_mean[0]:.3e}, char_scale={u_scale:.3e}")
    logger.info(f"  da: mean_norm={np.linalg.norm(a_mean):.3e}, char_scale={a_scale:.3e}")
    logger.info("Isometric scaler fitting complete (residual mode).")

    return ScalerPack(
        x=IsometricScaleParams(mean=x_mean.tolist(), scale=float(x_scale)),
        u=IsometricScaleParams(mean=u_mean.tolist(), scale=float(u_scale)),
        a=IsometricScaleParams(mean=a_mean.tolist(), scale=float(a_scale)),
    )


__all__ = [
    'IsometricScaleParams', 'ScalerPack', 'OnlineIsometricStats',
    'compute_base_potential', 'compute_base_accel', 'fit_scaler_streaming',
]
