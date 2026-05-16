#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
st_lrps_force_model.py - Propagator-ready inference API for the lunar residual potential surrogate.

Usage:
    from st_lrps_force_model import load_surrogate_force_model

    fm = load_surrogate_force_model("runs/st_lrps_train_20240101_120000")
    delta_u = fm.predict_residual_potential(x_m)   # DeltaU in m^2/s^2
    delta_a = fm.predict_residual_accel(x_m)       # Delta_a in m/s^2
    a_total = fm.predict_total_accel(x_m, base_accel_fn)  # a_SH20 + Delta_a

Frame warning: x_m must be in the same Moon-centered inertial frame used when
generating the SH dataset (typically MCMF / PA frame). Mixing frames will produce
physically wrong accelerations with no error signal.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np
import torch
import torch.nn as nn

try:
    from .st_lrps_models import build_model_from_config
    from .st_lrps_scaling import ScalerPack, compute_base_accel, compute_base_potential
    from .dataset_parameters import MU_MOON_SI, R_MOON_SI
except ImportError:
    from st_lrps_models import build_model_from_config
    from st_lrps_scaling import ScalerPack, compute_base_accel, compute_base_potential
    from dataset_parameters import MU_MOON_SI, R_MOON_SI


def _resolve_run_dir(model_dir: Union[str, Path]) -> Path:
    """
    Accept run dir, checkpoint dir, or direct checkpoint path.
    Returns the run directory (parent of checkpoints/).
    """
    p = Path(model_dir).expanduser().resolve()
    if p.is_file():
        # Direct checkpoint path -> run dir is grandparent
        return p.parent.parent
    if p.is_dir() and p.name == "checkpoints":
        # checkpoints/ dir -> run dir is parent
        return p.parent
    # Assume it's already the run dir
    return p


def _find_checkpoint(run_dir: Path) -> Path:
    """Prefer ckpt_best.pt, fall back to ckpt_last.pt."""
    ckpt_dir = run_dir / "checkpoints"
    for name in ("ckpt_best.pt", "ckpt_last.pt"):
        p = ckpt_dir / name
        if p.exists():
            return p
    raise FileNotFoundError(
        f"No checkpoint found in {ckpt_dir}. "
        "Expected ckpt_best.pt or ckpt_last.pt."
    )


def _to_tensor(x: Union[np.ndarray, torch.Tensor], device: torch.device) -> torch.Tensor:
    """Accept numpy or torch, return float32 tensor on device with shape (N,3)."""
    if isinstance(x, torch.Tensor):
        t = x.to(device=device, dtype=torch.float32)
    else:
        t = torch.from_numpy(np.asarray(x, dtype=np.float32)).to(device)
    if t.ndim == 1:
        if t.shape[0] != 3:
            raise ValueError(f"1-D input must have shape (3,). Got {t.shape}.")
        t = t.unsqueeze(0)  # (1,3)
    if t.ndim != 2 or t.shape[1] != 3:
        raise ValueError(f"Input must have shape (3,) or (N,3). Got {t.shape}.")
    return t


class SurrogateForceModel:
    """
    Loaded surrogate gravity force model for propagator integration.

    All methods accept positions in SI metres (Moon-centred inertial frame,
    matching the frame used during SH dataset generation).

    Attributes
    ----------
    degree_min : int
        Minimum SH degree of the analytical baseline the surrogate sits on top of.
        If degree_min < 0, the surrogate predicts the full potential field.
    mu_si : float
        Lunar GM in SI [m^3/s^2].
    a_sign : float
        Sign convention for a = a_sign * grad(U). Typically +1 or -1.
    device : torch.device
        Inference device.
    """

    def __init__(
        self,
        model: nn.Module,
        scaler: ScalerPack,
        cfg: dict,
        device: torch.device,
        chunk_size: int = 8192,
    ):
        self.model = model.eval()
        self.scaler = scaler
        self.cfg = cfg
        self.device = device
        self.chunk_size = int(chunk_size)

        self.mu_si = float(cfg.get("resolved_mu_si", MU_MOON_SI))
        self.a_sign = float(cfg.get("resolved_a_sign", 1.0))
        self.degree_min = int(cfg.get("degree_min", -1))
        self.r_ref_m = float(cfg.get("resolved_r_ref_m", R_MOON_SI))

    def _predict_chunk(self, x_t: torch.Tensor) -> tuple:
        """Forward + autograd for one chunk. Returns (delta_u_np, delta_a_np)."""
        x_scaled = self.scaler.scale_x(x_t).requires_grad_(True)
        delta_u_scaled = self.model(x_scaled)

        grad_delta_u = torch.autograd.grad(
            outputs=delta_u_scaled,
            inputs=x_scaled,
            grad_outputs=torch.ones_like(delta_u_scaled),
            create_graph=False,
            retain_graph=False,
            only_inputs=True,
        )[0]  # (B,3)

        # Chain rule: Delta_a = a_sign * grad(DeltaU_scaled) * (u_scale / x_scale)
        scaler_factor = self.scaler._u_scale / self.scaler._x_scale
        delta_a = self.a_sign * grad_delta_u * scaler_factor  # (B,3)
        delta_u = self.scaler.unscale_u(delta_u_scaled)       # (B,1)

        return delta_u.detach().cpu().numpy(), delta_a.detach().cpu().numpy()

    def _chunked_predict(
        self, x: Union[np.ndarray, torch.Tensor]
    ) -> tuple:
        """Chunked inference over arbitrary-length inputs."""
        x_t = _to_tensor(x, self.device)
        N = x_t.shape[0]
        u_out = np.empty((N, 1), dtype=np.float64)
        a_out = np.empty((N, 3), dtype=np.float64)

        for s in range(0, N, self.chunk_size):
            e = min(s + self.chunk_size, N)
            du, da = self._predict_chunk(x_t[s:e])
            u_out[s:e] = du
            a_out[s:e] = da

        return u_out, a_out

    def predict_residual_potential(
        self, x_m: Union[np.ndarray, torch.Tensor]
    ) -> np.ndarray:
        """
        Predict residual gravitational potential DeltaU(x) in m^2/s^2.

        Parameters
        ----------
        x_m : array-like, shape (3,) or (N,3)
            Moon-centred position(s) in metres.

        Returns
        -------
        delta_u : np.ndarray, shape (N,) or scalar
            Residual potential in m^2/s^2.
        """
        single = np.asarray(x_m).ndim == 1
        du, _ = self._chunked_predict(x_m)
        result = du.reshape(-1)
        return float(result[0]) if single else result

    def predict_residual_accel(
        self, x_m: Union[np.ndarray, torch.Tensor]
    ) -> np.ndarray:
        """
        Predict residual acceleration Delta_a = a_sign * grad(DeltaU) in m/s^2.

        Parameters
        ----------
        x_m : array-like, shape (3,) or (N,3)
            Moon-centred position(s) in metres.

        Returns
        -------
        delta_a : np.ndarray, shape (3,) or (N,3)
            Residual acceleration in m/s^2.
        """
        single = np.asarray(x_m).ndim == 1
        _, da = self._chunked_predict(x_m)
        return da[0] if single else da

    def predict_total_accel(
        self,
        x_m: Union[np.ndarray, torch.Tensor],
        base_accel_fn: Optional[Callable] = None,
    ) -> np.ndarray:
        """
        Predict total acceleration a_total = a_base(x) + Delta_a_NN(x).

        Parameters
        ----------
        x_m : array-like, shape (3,) or (N,3)
            Moon-centred position(s) in metres.
        base_accel_fn : callable, optional
            base_accel_fn(x_m) -> np.ndarray of shape (N,3).
            Should return the base SH(degree_min) acceleration.
            If None and degree_min < 0: uses point-mass formula with self.mu_si.
            If None and degree_min >= 0: raises ValueError (base model required).

        Returns
        -------
        a_total : np.ndarray, shape (3,) or (N,3)
            Total acceleration in m/s^2.
        """
        single = np.asarray(x_m).ndim == 1
        x_arr = np.asarray(x_m, dtype=np.float64)
        if x_arr.ndim == 1:
            x_arr = x_arr[None, :]

        _, da = self._chunked_predict(x_arr)

        if base_accel_fn is not None:
            a_base = np.asarray(base_accel_fn(x_arr), dtype=np.float64)
        elif self.degree_min < 0:
            # Point-mass approximation: a = -mu * r / |r|^3
            r_norm = np.linalg.norm(x_arr, axis=1, keepdims=True)
            r_norm = np.maximum(r_norm, 1.0)
            a_base = -self.mu_si * x_arr / r_norm ** 3
        else:
            raise ValueError(
                f"degree_min={self.degree_min}: a base_accel_fn(x) -> SH({self.degree_min}) "
                "acceleration must be provided for residual-mode total prediction. "
                "The point-mass approximation is not accurate enough for SH degree > 0 baselines."
            )

        a_total = a_base + da.astype(np.float64)
        return a_total[0] if single else a_total


def load_surrogate_force_model(
    model_dir: Union[str, Path],
    device: str = "auto",
    chunk_size: int = 8192,
) -> SurrogateForceModel:
    """
    Load a trained surrogate force model from a run directory.

    Accepts:
    - run directory:     runs/st_lrps_train_YYYYMMDD_HHMMSS
    - checkpoint dir:    runs/.../checkpoints
    - direct ckpt path: runs/.../checkpoints/ckpt_best.pt

    Parameters
    ----------
    model_dir : str or Path
        Path to run directory, checkpoint directory, or checkpoint file.
    device : str
        "auto" (GPU if available), "cpu", "cuda", or "mps".
    chunk_size : int
        Batch size for chunked inference. Reduce for low-memory GPUs.

    Returns
    -------
    SurrogateForceModel
        Ready-to-use force model.
    """
    run_dir = _resolve_run_dir(model_dir)
    ckpt_path_input = Path(model_dir).expanduser().resolve()
    if ckpt_path_input.is_file():
        ckpt_path = ckpt_path_input
    else:
        ckpt_path = _find_checkpoint(run_dir)

    cfg_path = run_dir / "config.json"
    sc_path = run_dir / "scaler.json"

    if not cfg_path.exists():
        raise FileNotFoundError(f"config.json not found in run dir: {run_dir}")
    if not sc_path.exists():
        raise FileNotFoundError(f"scaler.json not found in run dir: {run_dir}")

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    # Resolve device
    dev_str = str(device).lower()
    if dev_str == "auto":
        if torch.cuda.is_available():
            dev = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            dev = torch.device("mps")
        else:
            dev = torch.device("cpu")
    else:
        dev = torch.device(dev_str)

    scaler = ScalerPack.load(sc_path, device=dev, dtype=torch.float32)

    model = build_model_from_config(cfg, device=dev, dtype=torch.float32)
    model.eval()

    try:
        ckpt = torch.load(ckpt_path, map_location=dev, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=dev)

    state_dict = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state_dict, strict=True)

    # Prefer resolved physics from checkpoint if available (enriched by recent engine changes)
    for key in ("resolved_mu_si", "resolved_a_sign", "resolved_r_ref_m", "degree_min"):
        if key not in cfg and key in ckpt:
            cfg[key] = ckpt[key]

    return SurrogateForceModel(model=model, scaler=scaler, cfg=cfg, device=dev, chunk_size=chunk_size)


__all__ = ["SurrogateForceModel", "load_surrogate_force_model"]


if __name__ == "__main__":
    import argparse
    import sys

    ap = argparse.ArgumentParser(description="Smoke-test st_lrps_force_model.py")
    ap.add_argument("model_dir", help="Run dir, checkpoint dir, or .pt file")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--n", type=int, default=10)
    args = ap.parse_args()

    fm = load_surrogate_force_model(args.model_dir, device=args.device)
    print(f"Loaded: degree_min={fm.degree_min}, mu_si={fm.mu_si:.4e}, a_sign={fm.a_sign:+.1f}")
    print("Frame warning: input must be Moon-centred inertial frame matching SH dataset generation.")

    try:
        from dataset_parameters import R_MOON_SI as _R_REF
    except ImportError:
        _R_REF = 1.7374e6

    rng = np.random.default_rng(0)
    r = _R_REF + rng.uniform(30e3, 120e3, (args.n, 1))
    dirs = rng.standard_normal((args.n, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    x = (r * dirs).astype(np.float32)

    du = fm.predict_residual_potential(x)
    da = fm.predict_residual_accel(x)
    print(f"dU range: [{du.min():.3e}, {du.max():.3e}] m^2/s^2")
    print(f"|da| range: [{np.linalg.norm(da, axis=1).min():.3e}, {np.linalg.norm(da, axis=1).max():.3e}] m/s^2")
    print("Smoke test passed.")
