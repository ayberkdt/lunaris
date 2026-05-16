#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
st_lrps_evaluate.py  –  Evaluate a trained residual gravity model from st_lrps_train.py.

Required artifacts in --model-dir:
  config.json   – architecture, resolved_mu_si, resolved_a_sign, degree_min
  scaler.json   – isometric scale parameters for x, ΔU, Δa
  checkpoints/ckpt_best.pt

Prediction pipeline:
  x_scaled = (x - [0,0,0]) / max‖x‖      (origin-fixed isometric scaling)
  ΔU_scaled = model(x_scaled)
  Δa = a_sign · ∇(ΔU_scaled) · (u_scale / x_scale)   [isometric chain rule]
  U_total = U_base + unscale(ΔU_scaled)
  a_total = a_base + Δa

Note: torch.no_grad() is intentionally NOT used — Δa = ∇ΔU requires input gradients.

Metrics saved:
  MAE, RMSE, robust relative error, L∞ for U and |a|;
  vectorial angular error; radial/cross-radial (approx RTN) decomposition;
  altitude-binned RMSE and MAPE (bar charts + CSV);
  OOD table for ±10 % beyond training altitude band.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import h5py
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

try:
    from .dataset_parameters import (
        MU_MOON_SI,
        R_MOON_SI,
        is_lunar_body_signature,
        looks_like_lunar_run_config,
    )
except Exception:  # pragma: no cover
    import sys as _sys
    from pathlib import Path as _Path

    _HERE = _Path(__file__).resolve().parent
    if str(_HERE) not in _sys.path:
        _sys.path.insert(0, str(_HERE))
    from dataset_parameters import (  # type: ignore
        MU_MOON_SI,
        R_MOON_SI,
        is_lunar_body_signature,
        looks_like_lunar_run_config,
    )


# -----------------------------
# Device utilities
# -----------------------------
def get_device(prefer: str = "auto") -> torch.device:
    """Select device (auto/cpu/cuda/mps)."""
    prefer = prefer.lower()
    if prefer == "cpu":
        return torch.device("cpu")
    if prefer == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    if prefer == "mps":
        if getattr(torch.backends, "mps", None) is None or not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available.")
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _sync(device: torch.device) -> None:
    """Best-effort device sync for accurate timing."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        try:
            torch.mps.synchronize()
        except Exception:
            pass


# --- Shared model/scaler implementation ---
try:
    from .st_lrps_models import FourierInputEmbedding, MLP, PhysicsNet, Sine, SirenMLP, build_model_from_config
    from .st_lrps_data import DatasetMeta
    from .st_lrps_scaling import IsometricScaleParams, ScalerPack, compute_base_accel, compute_base_potential
except ImportError:  # pragma: no cover - script execution fallback
    from st_lrps_models import FourierInputEmbedding, MLP, PhysicsNet, Sine, SirenMLP, build_model_from_config
    from st_lrps_data import DatasetMeta
    from st_lrps_scaling import IsometricScaleParams, ScalerPack, compute_base_accel, compute_base_potential


def infer_r_ref_m_from_dataset(path: Path, dataset_name: str = "data") -> Optional[float]:
    """Read the lunar reference radius from dataset metadata when available."""
    if Path(path).suffix.lower() not in {".h5", ".hdf5"}:
        return None
    try:
        meta = DatasetMeta.from_h5(Path(path))
    except Exception:
        return None
    return float(meta.r_ref_m) if meta.r_ref_m is not None else None


def _read_eval_dataset_meta(path: Path, dataset_name: str = "data") -> Dict[str, Any]:
    """Return evaluation dataset metadata in the legacy evaluator dict shape."""
    if Path(path).suffix.lower() not in {".h5", ".hdf5"}:
        return {"unit_system": "unknown"}
    meta = DatasetMeta.from_h5(Path(path))
    central_body = (
        meta.raw_attrs.get("central_body")
        or ((meta.cloud_config or {}).get("central_body") if meta.cloud_config is not None else None)
        or meta.raw_attrs.get("body")
        or meta.raw_attrs.get("target_body")
    )
    out = {
        "unit_system": meta.unit_system,
        "central_body": central_body,
        "mu_si": meta.mu_si,
        "r_ref_m": meta.r_ref_m,
        "DU_m": meta.DU_m,
        "TU_s": meta.TU_s,
        "VU_m_s": meta.VU_m_s,
        "requested_degree": meta.requested_degree,
        "degree_min": meta.degree_min,
        "degree_max": meta.degree_max,
        "target_mode": meta.target_mode,
        "columns": meta.columns,
        "alt_min_km": meta.alt_min_km,
        "alt_max_km": meta.alt_max_km,
    }
    # Preserve suite/OOD attrs that DatasetMeta does not model explicitly.
    # ood_combined.h5 relies on these to split lower and upper OOD rows exactly.
    for key, value in meta.raw_attrs.items():
        if key not in out:
            if isinstance(value, bytes):
                try:
                    value = value.decode("utf-8")
                except UnicodeDecodeError:
                    pass
            out[key] = value
    return out


def load_checkpoint_state(path: Path, device: torch.device) -> Dict[str, Any]:
    """Load a training checkpoint across PyTorch versions."""
    try:
        state = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        state = torch.load(path, map_location=device)
    if isinstance(state, dict) and isinstance(state.get("model"), dict):
        return state["model"]
    return state


def _discover_h5_dataset_name(path: Path, preferred: str = "data") -> str:
    with h5py.File(path, "r") as handle:
        if preferred in handle and isinstance(handle[preferred], h5py.Dataset):
            return preferred
        for name, value in handle.items():
            if isinstance(value, h5py.Dataset) and value.ndim == 2 and value.shape[1] >= 7:
                return str(name)
    raise KeyError(f"No 2D [N, >=7] dataset found in {path}")


def iter_h5_batches(
    path: Path,
    dataset_name: str,
    *,
    batch_size: int,
    start: int = 0,
    end: Optional[int] = None,
) -> Iterator[np.ndarray]:
    with h5py.File(path, "r") as handle:
        ds = handle[dataset_name]
        stop = int(ds.shape[0]) if end is None else min(int(end), int(ds.shape[0]))
        for lo in range(int(start), stop, int(batch_size)):
            hi = min(lo + int(batch_size), stop)
            yield np.asarray(ds[lo:hi, :], dtype=np.float64)


def iter_pt_batches(
    path: Path,
    *,
    batch_size: int,
    start: int = 0,
    end: Optional[int] = None,
) -> Iterator[np.ndarray]:
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        for key in ("data", "array", "samples"):
            if key in obj:
                obj = obj[key]
                break
    arr = obj.detach().cpu().numpy() if isinstance(obj, torch.Tensor) else np.asarray(obj)
    stop = int(arr.shape[0]) if end is None else min(int(end), int(arr.shape[0]))
    for lo in range(int(start), stop, int(batch_size)):
        hi = min(lo + int(batch_size), stop)
        yield np.asarray(arr[lo:hi, :], dtype=np.float64)


def _canonical_to_si_batch(
    x: np.ndarray,
    u: np.ndarray,
    a: np.ndarray,
    DU_m: float,
    TU_s: float,
    VU_m_s: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_si = np.asarray(x, dtype=np.float64) * float(DU_m)
    u_si = np.asarray(u, dtype=np.float64) * (float(VU_m_s) ** 2)
    a_si = np.asarray(a, dtype=np.float64) * (float(DU_m) / (float(TU_s) ** 2))
    return x_si, u_si, a_si


def _accel_error_radial_cross_components(
    err_vec: np.ndarray,
    x_phys: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Radial/cross-radial error split plus approximate T/N axes without velocity."""
    r = np.asarray(x_phys, dtype=np.float64)
    e = np.asarray(err_vec, dtype=np.float64)
    r_norm = np.linalg.norm(r, axis=1, keepdims=True)
    r_hat = r / np.clip(r_norm, 1e-12, None)
    radial = np.sum(e * r_hat, axis=1)
    cross_vec = e - radial[:, None] * r_hat
    cross = np.linalg.norm(cross_vec, axis=1)

    z_hat = np.zeros_like(r_hat)
    z_hat[:, 2] = 1.0
    approx_t = np.cross(z_hat, r_hat)
    t_norm = np.linalg.norm(approx_t, axis=1, keepdims=True)
    fallback = t_norm[:, 0] < 1e-10
    if np.any(fallback):
        x_axis = np.zeros_like(r_hat[fallback])
        x_axis[:, 0] = 1.0
        approx_t[fallback] = np.cross(x_axis, r_hat[fallback])
        t_norm[fallback] = np.linalg.norm(approx_t[fallback], axis=1, keepdims=True)
    approx_t = approx_t / np.clip(t_norm, 1e-12, None)
    approx_n = np.cross(r_hat, approx_t)
    approx_t_err = np.sum(e * approx_t, axis=1)
    approx_n_err = np.sum(e * approx_n, axis=1)
    return radial, cross, approx_t_err, approx_n_err


# -----------------------------
# Core: forward + grad (FIX-3: hierarchical residual)
# -----------------------------
def predict_u_and_a(
    model: nn.Module,
    scaler: ScalerPack,
    x_phys: torch.Tensor,
    a_sign: float = 1.0,
    mu_si: float = MU_MOON_SI,
    degree_min: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Predict total U and a via hierarchical residual superposition.

    Returns U_pred_phys (B,1) and a_pred_phys (B,3):
      U_total = U_base + unscale(ΔU_scaled)
      a_total = a_base + a_sign · ∇(ΔU_scaled) · (u_scale/x_scale)
    """
    u_base = compute_base_potential(x_phys, mu_si, a_sign, degree_min)   # (B,1)
    a_base = compute_base_accel(x_phys, mu_si, degree_min)               # (B,3)

    x_scaled = scaler.scale_x(x_phys).requires_grad_(True)   # (B,3)
    delta_u_scaled = model(x_scaled)                          # (B,1)

    grad_delta_u_scaled = torch.autograd.grad(
        outputs=delta_u_scaled,
        inputs=x_scaled,
        grad_outputs=torch.ones_like(delta_u_scaled),
        create_graph=False,
        retain_graph=False,
        only_inputs=True,
    )[0]  # (B,3)

    # Isometric chain rule: scalar factor (u_scale/x_scale) preserves ∇U isotropy
    delta_a_phys = float(a_sign) * grad_delta_u_scaled * (scaler._u_scale / scaler._x_scale)
    delta_u_phys = scaler.unscale_u(delta_u_scaled)           # (B,1)

    u_total = u_base + delta_u_phys
    a_total = a_base + delta_a_phys

    return u_total, a_total


# -----------------------------
# Metrics
# -----------------------------
@dataclass
class MetricPack:
    mae: float
    rmse: float
    rel_mean_pct: float
    rel_p50_pct: float
    rel_p90_pct: float
    nrmse_pct: float
    linf: float
    rel_floor_abs: float


def infer_relative_floor_abs(
    ref: np.ndarray,
    *,
    eps: float = 1e-12,
    floor_fraction: float = 1e-2,
    percentile: float = 90.0,
) -> float:
    """
    Derive a stable denominator floor for residual-field relative errors.

    Classical MAPE explodes whenever the reference crosses zero, which happens
    frequently for residual gravity fields. That makes the post-training report
    look catastrophically bad even when the absolute error is physically small.

    We therefore compute a *dataset-scale* floor from the reference amplitudes
    and use ``max(|ref|, floor)`` as the denominator for relative metrics.
    This keeps the metric honest for meaningful signals while preventing a tiny
    number of near-zero residual points from dominating the summary.
    """

    ref_abs = np.abs(np.asarray(ref, dtype=np.float64).reshape(-1))
    ref_abs = ref_abs[np.isfinite(ref_abs)]
    if ref_abs.size == 0:
        return float(eps)

    anchor = max(
        float(np.percentile(ref_abs, percentile)),
        float(np.sqrt(np.mean(ref_abs ** 2))),
        float(eps),
    )
    return float(max(eps, floor_fraction * anchor))


def bounded_relative_error_pct(
    pred: np.ndarray,
    ref: np.ndarray,
    *,
    rel_floor_abs: float,
) -> np.ndarray:
    """
    Compute bounded percentage relative error using a dataset-scale floor.
    """

    pred_arr = np.asarray(pred, dtype=np.float64)
    ref_arr = np.asarray(ref, dtype=np.float64)
    denom = np.maximum(np.abs(ref_arr), float(rel_floor_abs))
    return 100.0 * np.abs(pred_arr - ref_arr) / denom


def compute_metrics(
    err: np.ndarray,
    ref: np.ndarray,
    eps: float = 1e-12,
    rel_floor_abs: Optional[float] = None,
) -> MetricPack:
    err = np.asarray(err, dtype=np.float64).reshape(-1)
    ref = np.asarray(ref, dtype=np.float64).reshape(-1)
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    linf = float(np.max(np.abs(err)))
    floor_abs = infer_relative_floor_abs(ref, eps=eps) if rel_floor_abs is None else float(rel_floor_abs)
    rel = np.abs(err) / np.maximum(np.abs(ref), floor_abs)
    rel_pct = 100.0 * rel
    rms_ref = float(np.sqrt(np.mean(ref ** 2)))
    nrmse_pct = float(100.0 * rmse / max(rms_ref, floor_abs))
    return MetricPack(
        mae=mae,
        rmse=rmse,
        rel_mean_pct=float(np.mean(rel_pct)),
        rel_p50_pct=float(np.percentile(rel_pct, 50)),
        rel_p90_pct=float(np.percentile(rel_pct, 90)),
        nrmse_pct=nrmse_pct,
        linf=linf,
        rel_floor_abs=floor_abs,
    )


def altitude_km(x: np.ndarray, r_ref_m: float) -> np.ndarray:
    r = np.linalg.norm(x, axis=1)
    return (r - float(r_ref_m)) / 1000.0


def _build_ood_region_masks(
    alt_km: np.ndarray,
    *,
    alt_lo: float,
    alt_hi: float,
    margin_fraction: float = 0.10,
) -> Dict[str, Any]:
    """
    Build immediate OOD masks just outside the training altitude band.
    """

    alt_flat = np.asarray(alt_km, dtype=np.float64).reshape(-1)
    span = max(0.0, float(alt_hi) - float(alt_lo))
    margin = float(margin_fraction) * span
    lower_lo = max(0.0, float(alt_lo) - margin)
    upper_hi = float(alt_hi) + margin

    return {
        "margin_km": margin,
        "lower_bounds_km": [lower_lo, float(alt_lo)],
        "in_band_bounds_km": [float(alt_lo), float(alt_hi)],
        "upper_bounds_km": [float(alt_hi), upper_hi],
        "lower_ood": (alt_flat >= lower_lo) & (alt_flat < float(alt_lo)),
        "in_band": (alt_flat >= float(alt_lo)) & (alt_flat <= float(alt_hi)),
        "upper_ood": (alt_flat > float(alt_hi)) & (alt_flat <= upper_hi),
    }


def spatial_rmse_by_altitude(
    alt_km: np.ndarray,
    err: np.ndarray,
    bin_km: float,
) -> Dict[str, Any]:
    alt_km = np.asarray(alt_km, dtype=np.float64).reshape(-1)
    err = np.asarray(err, dtype=np.float64).reshape(-1)

    lo = float(np.nanmin(alt_km))
    hi = float(np.nanmax(alt_km))
    if not np.isfinite(lo) or not np.isfinite(hi):
        return {"bin_km": float(bin_km), "bins": []}

    start = math.floor(lo / bin_km) * bin_km
    stop = math.ceil(hi / bin_km) * bin_km
    edges = np.arange(start, stop + bin_km, bin_km, dtype=np.float64)

    bins_out: List[Dict[str, Any]] = []
    for i in range(len(edges) - 1):
        a0, a1 = edges[i], edges[i + 1]
        mask = (alt_km >= a0) & (alt_km < a1)
        n = int(np.sum(mask))
        if n == 0:
            continue
        rmse = float(np.sqrt(np.mean(err[mask] ** 2)))
        bins_out.append({"alt_km_lo": float(a0), "alt_km_hi": float(a1), "n": n, "rmse": rmse})

    return {"bin_km": float(bin_km), "bins": bins_out}


def spatial_mape_by_altitude(
    alt_km: np.ndarray,
    ref: np.ndarray,
    pred: np.ndarray,
    bin_km: float,
    eps: float = 1e-12,
    rel_floor_abs: Optional[float] = None,
) -> Dict[str, Any]:
    """Bounded mean absolute percentage error per altitude bin."""
    alt_km = np.asarray(alt_km, dtype=np.float64).reshape(-1)
    ref    = np.asarray(ref,    dtype=np.float64).reshape(-1)
    pred   = np.asarray(pred,   dtype=np.float64).reshape(-1)
    floor_abs = infer_relative_floor_abs(ref, eps=eps) if rel_floor_abs is None else float(rel_floor_abs)
    rel_pct = bounded_relative_error_pct(pred, ref, rel_floor_abs=floor_abs)

    lo = float(np.nanmin(alt_km))
    hi = float(np.nanmax(alt_km))
    if not np.isfinite(lo) or not np.isfinite(hi):
        return {"bin_km": float(bin_km), "bins": []}

    start = math.floor(lo / bin_km) * bin_km
    stop  = math.ceil(hi / bin_km) * bin_km
    edges = np.arange(start, stop + bin_km, bin_km, dtype=np.float64)

    bins_out: List[Dict[str, Any]] = []
    for i in range(len(edges) - 1):
        a0, a1 = edges[i], edges[i + 1]
        mask = (alt_km >= a0) & (alt_km < a1)
        n = int(np.sum(mask))
        if n == 0:
            continue
        seg = rel_pct[mask]
        bins_out.append({
            "alt_km_lo": float(a0),
            "alt_km_hi": float(a1),
            "n": n,
            "mape_pct": float(np.mean(seg)),
            "p50_pct":  float(np.percentile(seg, 50)),
            "p90_pct":  float(np.percentile(seg, 90)),
            "rel_floor_abs": float(floor_abs),
        })

    return {"bin_km": float(bin_km), "bins": bins_out}


# -----------------------------
# Plotting
# -----------------------------
def apply_professional_style():
    try:
        import matplotlib.pyplot as plt
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams.update({
            "font.family": "sans-serif",
            "axes.labelsize": 11,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "legend.fontsize": 10,
            "figure.autolayout": True
        })
    except Exception:
        pass

def save_parity_plot(y_true: np.ndarray, y_pred: np.ndarray, path: Path, title: str) -> None:
    apply_professional_style()
    import matplotlib.pyplot as plt
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    n = y_true.size
    if n > 200_000:
        idx = np.random.default_rng(0).choice(n, size=200_000, replace=False)
        y_true = y_true[idx]
        y_pred = y_pred[idx]

    plt.figure(figsize=(7, 6))
    plt.scatter(y_true, y_pred, s=3, alpha=0.5, color="#4C72B0", edgecolor="none", label="Predictions")
    mn = float(min(y_true.min(), y_pred.min()))
    mx = float(max(y_true.max(), y_pred.max()))
    plt.plot([mn, mx], [mn, mx], color="#C44E52", linestyle="--", linewidth=2.0, label="Perfect Agreement (y=x)")
    
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r2 = 1 - (ss_res / max(ss_tot, 1e-12))
    plt.plot([], [], ' ', label=f"R² = {r2:.4f}")
    
    plt.xlabel("True Values", labelpad=8)
    plt.ylabel("Predicted Values", labelpad=8)
    plt.title(title, pad=12)
    plt.legend(frameon=True, fancybox=True, shadow=True, loc="best")
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()

def save_scatter_altitude(
    alt_km: np.ndarray,
    rel_err_pct: np.ndarray,
    path: Path,
    title: str,
) -> None:
    apply_professional_style()
    import matplotlib.pyplot as plt
    alt_km = np.asarray(alt_km).reshape(-1)
    rel_err_pct = np.asarray(rel_err_pct).reshape(-1)

    n = alt_km.size
    if n > 300_000:
        idx = np.random.default_rng(1).choice(n, size=300_000, replace=False)
        alt_km = alt_km[idx]
        rel_err_pct = rel_err_pct[idx]

    plt.figure(figsize=(8, 6))
    plt.scatter(alt_km, rel_err_pct, s=3, alpha=0.4, color="#55A868", edgecolor="none")
    mean_err = np.mean(rel_err_pct)
    plt.axhline(mean_err, color="#C44E52", linestyle="--", linewidth=2.0, label=f"Mean Error: {mean_err:.4f}%")
    plt.xlabel("Altitude [km]", labelpad=8)
    plt.ylabel("Relative Error [%]", labelpad=8)
    plt.title(title, pad=12)
    plt.yscale("log")
    plt.legend(frameon=True, fancybox=True, shadow=True)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()

def save_log_hist(errors: np.ndarray, path: Path, title: str) -> None:
    apply_professional_style()
    import matplotlib.pyplot as plt
    e = np.asarray(errors, dtype=np.float64).reshape(-1)
    e = np.abs(e)
    e = e[np.isfinite(e)]
    e = e[e > 0]
    if e.size == 0:
        return

    plt.figure(figsize=(8, 5))
    lo = np.percentile(e, 0.1)
    hi = np.percentile(e, 99.9)
    lo = max(lo, 1e-18)
    hi = max(hi, lo * 10)
    bins = np.logspace(np.log10(lo), np.log10(hi), 80)
    
    plt.hist(e, bins=bins, color="#8172B3", edgecolor="white", alpha=0.85)
    mean_e = np.mean(e)
    median_e = np.median(e)
    plt.axvline(mean_e, color="#C44E52", linestyle="--", linewidth=1.5, label=f"Mean: {mean_e:.2e}")
    plt.axvline(median_e, color="#2C3E50", linestyle=":", linewidth=1.5, label=f"Median: {median_e:.2e}")
    
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Absolute Error", labelpad=8)
    plt.ylabel("Frequency", labelpad=8)
    plt.title(title, pad=12)
    plt.legend(frameon=True, fancybox=True, shadow=True)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()

def save_pct_error_hist(
    rel_err_pct: np.ndarray,
    path: Path,
    title: str,
    percentiles: Tuple[float, float] = (0.1, 99.9),
) -> None:
    """Histogram of percentage relative error with log-frequency axis."""
    apply_professional_style()
    e = np.asarray(rel_err_pct, dtype=np.float64).reshape(-1)
    e = np.abs(e)
    e = e[np.isfinite(e)]
    if e.size == 0:
        return

    plt.figure(figsize=(8, 5))
    lo = max(np.percentile(e, percentiles[0]), 1e-6)
    hi = max(np.percentile(e, percentiles[1]), lo * 10)
    bins = np.logspace(np.log10(lo), np.log10(hi), 80)
    plt.hist(e, bins=bins, color="#4C72B0", edgecolor="white", alpha=0.85)

    # Shade percentile bands
    p50 = float(np.percentile(e, 50))
    p90 = float(np.percentile(e, 90))
    p99 = float(np.percentile(e, 99))
    plt.axvline(p50,  color="#2ECC71", linestyle="--", linewidth=1.5, label=f"P50: {p50:.3f}%")
    plt.axvline(p90,  color="#E67E22", linestyle="--", linewidth=1.5, label=f"P90: {p90:.3f}%")
    plt.axvline(p99,  color="#C44E52", linestyle="--", linewidth=1.5, label=f"P99: {p99:.3f}%")
    mean_v = float(np.mean(e))
    plt.axvline(mean_v, color="#8E44AD", linestyle=":",  linewidth=1.5, label=f"Mean: {mean_v:.3f}%")

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Relative Error [%]", labelpad=8)
    plt.ylabel("Frequency", labelpad=8)
    plt.title(title, pad=12)
    plt.legend(frameon=True, fancybox=True, shadow=True)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def save_binned_mae_pct(
    alt_km: np.ndarray,
    rel_err_pct: np.ndarray,
    path: Path,
    title: str,
    bin_km: float = 10.0,
) -> None:
    """Bar chart of altitude-binned mean absolute percentage error (MAPE)."""
    apply_professional_style()
    alt = np.asarray(alt_km, dtype=np.float64).reshape(-1)
    err = np.abs(np.asarray(rel_err_pct, dtype=np.float64).reshape(-1))

    lo = math.floor(float(np.nanmin(alt)) / bin_km) * bin_km
    hi = math.ceil(float(np.nanmax(alt)) / bin_km) * bin_km
    edges = np.arange(lo, hi + bin_km, bin_km)
    if len(edges) < 2:
        return

    centers, maes, p25s, p75s, counts = [], [], [], [], []
    for i in range(len(edges) - 1):
        mask = (alt >= edges[i]) & (alt < edges[i + 1])
        n = int(np.sum(mask))
        if n == 0:
            continue
        seg = err[mask]
        centers.append(0.5 * (edges[i] + edges[i + 1]))
        maes.append(float(np.mean(seg)))
        p25s.append(float(np.percentile(seg, 25)))
        p75s.append(float(np.percentile(seg, 75)))
        counts.append(n)

    if not centers:
        return

    centers = np.array(centers)
    maes    = np.array(maes)
    p25s    = np.array(p25s)
    p75s    = np.array(p75s)
    counts  = np.array(counts)

    fig, ax1 = plt.subplots(figsize=(9, 5))
    bar_w = bin_km * 0.7
    ax1.bar(centers, maes, width=bar_w, color="#4C72B0", alpha=0.75, label="MAPE [%]")
    # IQR error bars
    ax1.errorbar(
        centers, maes,
        yerr=[np.clip(maes - p25s, 0, None), np.clip(p75s - maes, 0, None)],
        fmt="none", ecolor="#2C3E50", elinewidth=1.2, capsize=3, label="IQR"
    )
    ax1.set_xlabel("Altitude [km]", labelpad=8)
    ax1.set_ylabel("Mean Absolute % Error", labelpad=8)
    ax1.set_title(title, pad=12)

    # Secondary axis: sample count
    ax2 = ax1.twinx()
    ax2.step(np.append(centers - bar_w / 2, centers[-1] + bar_w / 2),
             np.append(counts, counts[-1]),
             where="post", color="#95A5A6", linewidth=1.2, linestyle="--", alpha=0.7, label="Count")
    ax2.set_ylabel("Sample count", labelpad=8)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, frameon=True, fancybox=True, shadow=True, fontsize=9)

    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_hist_angular_deg(deg: np.ndarray, path: Path, title: str) -> None:
    apply_professional_style()
    import matplotlib.pyplot as plt
    d = np.asarray(deg, dtype=np.float64).reshape(-1)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return

    plt.figure(figsize=(8, 5))
    bins = np.linspace(0.0, 180.0, 181)
    plt.hist(d, bins=bins, color="#64B5CD", edgecolor="white", alpha=0.85)
    
    mean_d = np.mean(d)
    median_d = np.median(d)
    plt.axvline(mean_d, color="#C44E52", linestyle="--", linewidth=1.5, label=f"Mean: {mean_d:.2f}°")
    plt.axvline(median_d, color="#2C3E50", linestyle=":", linewidth=1.5, label=f"Median: {median_d:.2f}°")
    
    plt.yscale("log")
    plt.xlabel("Angular Error [deg]", labelpad=8)
    plt.ylabel("Frequency", labelpad=8)
    plt.title(title, pad=12)
    plt.legend(frameon=True, fancybox=True, shadow=True)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()

# -----------------------------
# Performance benchmark
# -----------------------------
def benchmark_throughput(
    model: nn.Module,
    scaler: ScalerPack,
    x_np: np.ndarray,
    device: torch.device,
    batch_size: int,
    a_sign: float,
    mu_si: float,
    degree_min: int = -1,
    warmup: int = 10,
    iters: int = 30,
) -> float:
    """Points/sec for (U + grad) with given batch_size."""
    model = model.to(device)
    model.eval()

    x_np = np.asarray(x_np, dtype=np.float32)
    n = x_np.shape[0]
    if n == 0:
        return float("nan")

    take = min(n, batch_size * max(1, iters))
    x_np = x_np[:take]

    for _ in range(warmup):
        xb = torch.from_numpy(x_np[:batch_size]).to(device=device, dtype=torch.float32)
        _ = predict_u_and_a(model, scaler, xb, a_sign=a_sign, mu_si=mu_si, degree_min=degree_min)
        _sync(device)

    _sync(device)
    t0 = time.perf_counter()

    done = 0
    for k in range(iters):
        s = (k * batch_size) % x_np.shape[0]
        e = min(s + batch_size, x_np.shape[0])
        xb = torch.from_numpy(x_np[s:e]).to(device=device, dtype=torch.float32)
        _ = predict_u_and_a(model, scaler, xb, a_sign=a_sign, mu_si=mu_si, degree_min=degree_min)
        done += (e - s)

    _sync(device)
    t1 = time.perf_counter()
    dt = max(t1 - t0, 1e-12)
    return float(done / dt)


def benchmark_latency_ms(
    model: nn.Module,
    scaler: ScalerPack,
    x_np: np.ndarray,
    device: torch.device,
    a_sign: float,
    mu_si: float,
    degree_min: int = -1,
    warmup: int = 25,
    iters: int = 200,
) -> float:
    """Single-point latency (ms) for one evaluation (batch_size=1)."""
    model = model.to(device)
    model.eval()

    x_np = np.asarray(x_np, dtype=np.float32)
    if x_np.shape[0] == 0:
        return float("nan")

    x1 = x_np[0:1, :]

    for _ in range(warmup):
        xb = torch.from_numpy(x1).to(device=device, dtype=torch.float32)
        _ = predict_u_and_a(model, scaler, xb, a_sign=a_sign, mu_si=mu_si, degree_min=degree_min)
        _sync(device)

    _sync(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        xb = torch.from_numpy(x1).to(device=device, dtype=torch.float32)
        _ = predict_u_and_a(model, scaler, xb, a_sign=a_sign, mu_si=mu_si, degree_min=degree_min)
    _sync(device)
    t1 = time.perf_counter()

    dt = max(t1 - t0, 1e-12)
    return float(1000.0 * dt / iters)


# -----------------------------
# Main evaluation
# -----------------------------
def evaluate(
    model_dir: Path,
    data_path: Path,
    out_dir: Path,
    device: torch.device,
    batch_size: int,
    a_sign: float,
    r_ref_m: float,
    alt_bin_km: float,
    dataset_name: str = "data",
    start: int = 0,
    end: Optional[int] = None,
    max_points_for_plots: int = 500_000,
) -> None:
    model_dir = model_dir.resolve()
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg_path = model_dir / "config.json"
    sc_path = model_dir / "scaler.json"
    ckpt_path = model_dir / "checkpoints" / "ckpt_best.pt"
    if not ckpt_path.exists():
        # Fallback: use ckpt_last.pt if ckpt_best.pt was never written (e.g. short training runs)
        _last = model_dir / "checkpoints" / "ckpt_last.pt"
        if _last.exists():
            print(f"[eval] ckpt_best.pt not found; falling back to ckpt_last.pt")
            ckpt_path = _last
    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)
    if not sc_path.exists():
        raise FileNotFoundError(sc_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    scaler = ScalerPack.load(sc_path, device=device, dtype=torch.float32)

    # Read dataset metadata for unit conversion and consistency checks
    ds_meta = _read_eval_dataset_meta(data_path, dataset_name=dataset_name)
    ds_unit_system = str(ds_meta.get("unit_system", "unknown")).lower()
    ds_DU_m: Optional[float] = None
    ds_TU_s: Optional[float] = None
    ds_VU_m_s: Optional[float] = None
    if ds_unit_system == "canonical":
        try:
            ds_DU_m = float(ds_meta["DU_m"])
            ds_TU_s = float(ds_meta["TU_s"])
            ds_VU_m_s = float(ds_meta["VU_m_s"])
        except (KeyError, TypeError, ValueError):
            raise ValueError(
                "Canonical dataset requires DU_m, TU_s, and VU_m_s for SI conversion. "
                f"Got DU_m={ds_meta.get('DU_m')}, TU_s={ds_meta.get('TU_s')}, VU_m_s={ds_meta.get('VU_m_s')}."
            )

    def _as_optional_int(value: Any) -> Optional[int]:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _as_optional_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    mu_si = float(cfg.get("resolved_mu_si", MU_MOON_SI))
    # Backward-compat: old models stored degree_min only inside dataset_meta
    _dm_top = cfg.get("degree_min")
    _dm_meta = (cfg.get("dataset_meta") or {}).get("degree_min")
    degree_min = int(_dm_top if _dm_top is not None else (_dm_meta if _dm_meta is not None else -1))
    a_sign_resolved = float(cfg.get("resolved_a_sign", a_sign))
    if a_sign_resolved != a_sign:
        print(f"[INFO] Overriding CLI a_sign={a_sign} with config.json resolved_a_sign={a_sign_resolved}")
        a_sign = a_sign_resolved

    # Training altitude range - used to build OOD metric table
    _dm_meta_block = cfg.get("dataset_meta") or {}
    _train_alt_min_km: Optional[float] = None
    _train_alt_max_km: Optional[float] = None
    try:
        _v = _dm_meta_block.get("alt_min_km")
        if _v is not None:
            _train_alt_min_km = float(_v)
        _v = _dm_meta_block.get("alt_max_km")
        if _v is not None:
            _train_alt_max_km = float(_v)
    except (TypeError, ValueError):
        pass
    # Fallback: ood_combined.h5 files store train_alt_min_km / train_alt_max_km as attrs
    if _train_alt_min_km is None:
        try:
            _v2 = ds_meta.get("train_alt_min_km")
            if _v2 is not None:
                _train_alt_min_km = float(_v2)
        except (TypeError, ValueError):
            pass
    if _train_alt_max_km is None:
        try:
            _v2 = ds_meta.get("train_alt_max_km")
            if _v2 is not None:
                _train_alt_max_km = float(_v2)
        except (TypeError, ValueError):
            pass
    # Try top-level config too (some configs store altitude_min_km / altitude_max_km)
    if _train_alt_min_km is None:
        try:
            _v2 = cfg.get("altitude_min_km")
            if _v2 is not None:
                _train_alt_min_km = float(_v2)
        except (TypeError, ValueError):
            pass
    if _train_alt_max_km is None:
        try:
            _v2 = cfg.get("altitude_max_km")
            if _v2 is not None:
                _train_alt_max_km = float(_v2)
        except (TypeError, ValueError):
            pass

    model_meta = cfg.get("dataset_meta") or {}
    model_degree_max = _as_optional_int(cfg.get("degree_max"))
    if model_degree_max is None:
        model_degree_max = _as_optional_int(model_meta.get("degree_max"))
    if model_degree_max is None:
        model_degree_max = _as_optional_int(model_meta.get("requested_degree"))

    ds_degree_max = _as_optional_int(ds_meta.get("degree_max"))
    if ds_degree_max is None:
        ds_degree_max = _as_optional_int(ds_meta.get("requested_degree"))
    if model_degree_max is not None and ds_degree_max is not None and model_degree_max != ds_degree_max:
        raise ValueError(
            f"Model degree_max={model_degree_max} does not match dataset degree_max={ds_degree_max}. "
            "Evaluation refuses to continue because the residual harmonic band would be inconsistent."
        )

    model_body = str(cfg.get("central_body") or model_meta.get("central_body") or "").strip().lower() or None
    ds_body = str(ds_meta.get("central_body", "") or "").strip().lower() or None
    if model_body and model_body not in {"moon", "lunar", "selene"}:
        raise ValueError(f"Model config declares non-lunar central_body={model_body!r}.")
    if ds_body and ds_body not in {"moon", "lunar", "selene"}:
        raise ValueError(f"Evaluation dataset declares non-lunar central_body={ds_body!r}.")
    if model_body and ds_body and model_body != ds_body:
        raise ValueError(
            f"Model central_body={model_body!r} does not match evaluation dataset central_body={ds_body!r}."
        )

    model_mu = _as_optional_float(cfg.get("resolved_mu_si"))
    if model_mu is None:
        model_mu = _as_optional_float(model_meta.get("mu_si"))
    model_r_ref = _as_optional_float(cfg.get("resolved_r_ref_m"))
    if model_r_ref is None:
        model_r_ref = _as_optional_float(cfg.get("r_ref_m"))
    if model_r_ref is None:
        model_r_ref = _as_optional_float(model_meta.get("r_ref_m"))
    ds_mu = _as_optional_float(ds_meta.get("mu_si"))
    ds_r_ref = _as_optional_float(ds_meta.get("r_ref_m"))

    if not is_lunar_body_signature(mu_si=model_mu, r_ref_m=model_r_ref):
        raise ValueError(
            f"Model lunar body signature is inconsistent (mu_si={model_mu!r}, r_ref_m={model_r_ref!r})."
        )
    if ds_body is None and ds_mu is None and ds_r_ref is None:
        raise ValueError("Evaluation dataset is missing lunar body metadata (central_body, mu_si, r_ref_m).")
    if ds_mu is not None or ds_r_ref is not None:
        if not is_lunar_body_signature(mu_si=ds_mu, r_ref_m=ds_r_ref):
            raise ValueError(
                f"Evaluation dataset lunar body signature is inconsistent (mu_si={ds_mu!r}, r_ref_m={ds_r_ref!r})."
            )

    # Check dataset vs model degree_min consistency
    _ds_dm_raw = ds_meta.get("degree_min")
    if _ds_dm_raw is not None:
        try:
            ds_degree_min = int(_ds_dm_raw)
            if ds_degree_min != degree_min:
                raise ValueError(
                    f"Model degree_min={degree_min} does not match dataset degree_min={ds_degree_min}. "
                    "Evaluation refuses to continue because residual baselines would be inconsistent."
                )
        except (TypeError, ValueError):
            raise
    ds_target_mode = str(ds_meta.get("target_mode", "")).strip().lower() or None
    if ds_target_mode is not None:
        expected_target_mode = "residual" if degree_min >= 0 else "full"
        if ds_target_mode != expected_target_mode:
            raise ValueError(
                f"Model expects target_mode={expected_target_mode!r} but dataset declares "
                f"target_mode={ds_target_mode!r}. Use a matching evaluation dataset."
            )
        print(f"[info] Dataset target_mode: {ds_target_mode} | model degree_min: {degree_min}")

    # ---------- Metadata comparison summary (H.1) ----------
    print("\n=============== EVALUATION CONTRACT ===============")
    print(f"{'Field':<22}  {'Model':>26}  {'Dataset':>26}")
    print("-" * 78)
    _m_body   = str(cfg.get("central_body") or model_meta.get("central_body") or "?").strip()
    _d_body   = str(ds_meta.get("central_body", "") or "?").strip()
    _m_dm     = str(degree_min)
    _d_dm     = str(ds_meta.get("degree_min", "?"))
    _m_dmax   = str(model_degree_max if model_degree_max is not None else "?")
    _d_dmax   = str(ds_degree_max if ds_degree_max is not None else "?")
    _m_tmode  = str(cfg.get("target_mode") or ("residual" if degree_min >= 0 else "full"))
    _d_tmode  = str(ds_meta.get("target_mode", "?")).strip() or "?"
    _m_mu     = f"{mu_si:.6e}"
    _d_mu     = f"{ds_mu:.6e}" if ds_mu is not None else "?"
    _m_rref   = f"{model_r_ref:.6e}" if model_r_ref is not None else "?"
    _d_rref   = f"{ds_r_ref:.6e}"  if ds_r_ref  is not None else "?"
    for label, mv, dv in [
        ("central_body",   _m_body,  _d_body),
        ("degree_min",     _m_dm,    _d_dm),
        ("degree_max",     _m_dmax,  _d_dmax),
        ("target_mode",    _m_tmode, _d_tmode),
        ("mu_si (m^3/s^2)",  _m_mu,    _d_mu),
        ("r_ref_m (m)",    _m_rref,  _d_rref),
    ]:
        ok = "OK" if mv == dv else "WARN"
        if label in ("mu_si (m^3/s^2)", "r_ref_m (m)"):
            ok = "OK"  # numeric tolerance already checked above
        print(f"  {label:<20}  {mv:>26}  {dv:>26}  [{ok}]")
    print("====================================================\n")

    model = build_model_from_config(cfg, device=device)
    model.eval()

    state = load_checkpoint_state(ckpt_path, device=device)
    _load_result = model.load_state_dict(state, strict=True)
    load_report = {
        "missing_keys": list(_load_result.missing_keys),
        "unexpected_keys": list(_load_result.unexpected_keys),
    }
    if load_report["missing_keys"] or load_report["unexpected_keys"]:
        print(f"[WARN] Checkpoint key mismatch: {load_report}")

    suffix = data_path.suffix.lower()
    if suffix in [".h5", ".hdf5"]:
        dset_name = dataset_name
        try:
            with h5py.File(data_path, "r") as f:
                _ = f[dset_name]
        except Exception:
            dset_name = _discover_h5_dataset_name(data_path, preferred=dataset_name)
        batch_iter = iter_h5_batches(data_path, dset_name, batch_size=batch_size, start=start, end=end)
    elif suffix == ".pt":
        batch_iter = iter_pt_batches(data_path, batch_size=batch_size, start=start, end=end)
    else:
        raise ValueError("Unsupported data format. Use .h5/.hdf5 or .pt")

    u_true_all: List[np.ndarray] = []
    u_pred_all: List[np.ndarray] = []
    a_true_mag_all: List[np.ndarray] = []
    a_pred_mag_all: List[np.ndarray] = []
    alt_all: List[np.ndarray] = []
    x_all: List[np.ndarray] = []           # positions for RTN decomposition
    a_err_vec_all: List[np.ndarray] = []   # vectorial acceleration error for RTN
    ang_all: List[np.ndarray] = []
    a_pred_vec_all: List[np.ndarray] = []  # full predicted acceleration vector
    a_true_vec_all: List[np.ndarray] = []  # full true acceleration vector

    # bounded buffers for plots
    u_err_plot: List[np.ndarray] = []
    a_rel_plot: List[np.ndarray] = []
    ang_deg_plot: List[np.ndarray] = []

    # streaming angular stats
    ang_sum_deg = 0.0
    ang_max_deg = 0.0
    ang_sum_cossim = 0.0
    ang_count = 0

    total = 0
    for arr in batch_iter:
        x = arr[:, 0:3]
        u_true = arr[:, 3:4]
        a_true = arr[:, 4:7]

        # Convert canonical → SI if needed (model was trained in SI)
        if ds_unit_system == "canonical" and ds_DU_m is not None:
            x, u_true, a_true = _canonical_to_si_batch(x, u_true, a_true, ds_DU_m, ds_TU_s, ds_VU_m_s)

        xb = torch.from_numpy(x).to(device=device, dtype=torch.float32)
        u_pred_t, a_pred_t = predict_u_and_a(model, scaler, xb, a_sign=a_sign, mu_si=mu_si, degree_min=degree_min)

        u_pred = u_pred_t.detach().cpu().numpy()
        a_pred = a_pred_t.detach().cpu().numpy()

        a_true_mag = np.linalg.norm(a_true, axis=1, keepdims=True)
        a_pred_mag = np.linalg.norm(a_pred, axis=1, keepdims=True)

        # ---- NEW: vectorial angular error ----
        eps = 1e-18
        dot = np.sum(a_pred * a_true, axis=1)
        na_p = np.linalg.norm(a_pred, axis=1)
        na_t = np.linalg.norm(a_true, axis=1)
        denom = np.maximum(na_p * na_t, eps)
        cossim = np.clip(dot / denom, -1.0, 1.0)
        ang_deg = np.degrees(np.arccos(cossim))  # [0, 180]

        ang_sum_deg += float(np.sum(ang_deg))
        ang_max_deg = max(ang_max_deg, float(np.max(ang_deg)) if ang_deg.size else 0.0)
        ang_sum_cossim += float(np.sum(cossim))
        ang_count += int(ang_deg.size)

        alt_km = altitude_km(x, r_ref_m=r_ref_m).reshape(-1, 1)

        u_true_all.append(u_true)
        u_pred_all.append(u_pred)
        a_true_mag_all.append(a_true_mag)
        a_pred_mag_all.append(a_pred_mag)
        alt_all.append(alt_km)
        x_all.append(x.astype(np.float32))
        a_err_vec_all.append((a_pred - a_true).astype(np.float32))
        ang_all.append(ang_deg.reshape(-1, 1))
        a_pred_vec_all.append(a_pred.astype(np.float32))
        a_true_vec_all.append(a_true.astype(np.float32))

        # plot buffers
        if sum(v.shape[0] for v in u_err_plot) < max_points_for_plots:
            u_err_plot.append((u_pred - u_true).reshape(-1, 1))
            rel_a = bounded_relative_error_pct(a_pred_mag, a_true_mag, rel_floor_abs=1e-12)
            a_rel_plot.append(rel_a.reshape(-1, 1))
            ang_deg_plot.append(ang_deg.reshape(-1, 1))

        total += x.shape[0]

    u_true = np.concatenate(u_true_all, axis=0).reshape(-1, 1)
    u_pred = np.concatenate(u_pred_all, axis=0).reshape(-1, 1)
    a_true_mag = np.concatenate(a_true_mag_all, axis=0).reshape(-1, 1)
    a_pred_mag = np.concatenate(a_pred_mag_all, axis=0).reshape(-1, 1)
    alt_km_all = np.concatenate(alt_all, axis=0).reshape(-1, 1)
    x_all_np = np.concatenate(x_all, axis=0)                    # (N, 3) positions
    a_err_vec_np = np.concatenate(a_err_vec_all, axis=0)        # (N, 3) vectorial error
    ang_deg_all = np.concatenate(ang_all, axis=0).reshape(-1)

    u_err = (u_pred - u_true).reshape(-1)
    a_mag_err = (a_pred_mag - a_true_mag).reshape(-1)
    u_rel_floor_abs = infer_relative_floor_abs(u_true.reshape(-1))
    a_rel_floor_abs = infer_relative_floor_abs(a_true_mag.reshape(-1))

    # ---- Masked angular error (exclude near-zero residuals below the direction floor) ----
    direction_floor_abs = float(cfg.get("direction_loss_floor_abs", 3e-6))
    a_true_vec_np = np.concatenate(a_true_vec_all, axis=0)   # (N,3)
    a_pred_vec_np = np.concatenate(a_pred_vec_all, axis=0)   # (N,3)
    a_true_norms = np.linalg.norm(a_true_vec_np, axis=1)
    dir_mask = a_true_norms > direction_floor_abs
    mask_frac = float(np.mean(dir_mask.astype(np.float64)))
    masked_ang_deg = ang_deg_all[dir_mask] if dir_mask.any() else np.array([], dtype=np.float64)

    # ---- Total-field angular error (residual mode: approximate with point-mass base) ----
    if degree_min < 0:
        # In full-field mode, ang_deg_all already IS the total angular error.
        total_ang_note = "same_as_residual_ang_in_full_field_mode"
        ang_total_deg: Optional[np.ndarray] = None
    else:
        # Residual mode: add point-mass baseline (degree_min=-1) to approximate totals.
        # True total requires SH(degree_min) which needs the GFC model at eval time.
        x_t_all_for_base = torch.from_numpy(x_all_np.astype(np.float32)).to(device)
        a_base_np = compute_base_accel(x_t_all_for_base, mu_si, degree_min=-1).cpu().numpy()
        a_total_pred_np = a_base_np.astype(np.float64) + a_pred_vec_np.astype(np.float64)
        a_total_true_np = a_base_np.astype(np.float64) + a_true_vec_np.astype(np.float64)
        eps_ang = 1e-18
        dot_tot = np.sum(a_total_pred_np * a_total_true_np, axis=1)
        na_tot_p = np.linalg.norm(a_total_pred_np, axis=1)
        na_tot_t = np.linalg.norm(a_total_true_np, axis=1)
        cossim_tot = np.clip(dot_tot / np.maximum(na_tot_p * na_tot_t, eps_ang), -1.0, 1.0)
        ang_total_deg = np.degrees(np.arccos(cossim_tot))
        total_ang_note = "point_mass_base_approximation"

    # ---- Norm-binned angular error by ||a_true|| magnitude ----
    NORM_BINS = [0.0, 1e-10, 1e-9, 1e-8, 1e-7, float("inf")]
    norm_bin_labels = ["<1e-10", "1e-10-1e-9", "1e-9-1e-8", "1e-8-1e-7", ">1e-7"]
    norm_binned_ang: List[Dict[str, Any]] = []
    for _i_bin, (_lo_n, _hi_n) in enumerate(zip(NORM_BINS[:-1], NORM_BINS[1:])):
        _nb_mask = (a_true_norms >= _lo_n) & (a_true_norms < _hi_n)
        _n_bin = int(np.sum(_nb_mask))
        if _n_bin == 0:
            norm_binned_ang.append({"bin": norm_bin_labels[_i_bin], "N": 0})
            continue
        _ang_bin = ang_deg_all[_nb_mask]
        norm_binned_ang.append({
            "bin": norm_bin_labels[_i_bin],
            "norm_range_m_s2": [float(_lo_n), float(_hi_n)],
            "N": _n_bin,
            "mean_deg": float(np.mean(_ang_bin)),
            "median_deg": float(np.median(_ang_bin)),
            "p90_deg": float(np.percentile(_ang_bin, 90)),
            "p99_deg": float(np.percentile(_ang_bin, 99)),
        })

    # Radial / cross-radial directional diagnostics (approximate T/N, no velocity available)
    a_r, a_cross, approx_t, approx_n = _accel_error_radial_cross_components(
        a_err_vec_np.astype(np.float64), x_all_np.astype(np.float64)
    )
    directional_metrics: Dict[str, Any] = {
        "frame": "approximate_rtn_like_without_velocity",
        "accel_err_radial_mae": float(np.mean(np.abs(a_r))),
        "accel_err_radial_rmse": float(np.sqrt(np.mean(a_r ** 2))),
        "accel_err_cross_radial_mae": float(np.mean(np.abs(a_cross))),
        "accel_err_cross_radial_rmse": float(np.sqrt(np.mean(a_cross ** 2))),
        "approx_T_rmse": float(np.sqrt(np.mean(approx_t ** 2))),
        "approx_N_rmse": float(np.sqrt(np.mean(approx_n ** 2))),
    }

    # OOD table - test points around or outside the training altitude band.
    ood_table: Optional[Dict[str, Any]] = None
    if _train_alt_min_km is not None and _train_alt_max_km is not None:
        alt_lo = float(_train_alt_min_km)
        alt_hi = float(_train_alt_max_km)
        mask_pack = _build_ood_region_masks(alt_km_all.reshape(-1), alt_lo=alt_lo, alt_hi=alt_hi, margin_fraction=0.10)

        def _region_stats(mask: np.ndarray, label: str) -> Dict[str, Any]:
            n = int(np.sum(mask))
            if n == 0:
                return {"region": label, "N": 0}
            ue = u_err[mask]
            ae = a_mag_err[mask]
            u_metrics = compute_metrics(ue, u_true.reshape(-1)[mask], rel_floor_abs=u_rel_floor_abs)
            a_metrics = compute_metrics(ae, a_true_mag.reshape(-1)[mask], rel_floor_abs=a_rel_floor_abs)
            ang_region = ang_deg_all[mask]
            return {
                "region": label,
                "N": n,
                "RMSE_U": float(u_metrics.rmse),
                "MAE_U": float(u_metrics.mae),
                "RMSE_accel_norm": float(a_metrics.rmse),
                "MAE_accel_norm": float(a_metrics.mae),
                "robust_rel_accel_mean_pct": float(a_metrics.rel_mean_pct),
                "robust_rel_accel_p90_pct": float(a_metrics.rel_p90_pct),
                "angular_error_mean_deg": float(np.mean(ang_region)),
                "angular_error_p90_deg": float(np.percentile(ang_region, 90.0)),
            }

        _ds_role = str(ds_meta.get("dataset_role", "") or "").strip().lower()
        _low_n = _as_optional_int(ds_meta.get("ood_low_n"))
        _high_n = _as_optional_int(ds_meta.get("ood_high_n"))
        _n_total = int(alt_km_all.reshape(-1).shape[0])
        _has_exact_ood_split = (
            _ds_role == "ood_combined"
            and _low_n is not None
            and _high_n is not None
            and _low_n >= 0
            and _high_n >= 0
            and (_low_n + _high_n) == _n_total
        )

        if _has_exact_ood_split:
            mask_low = np.zeros((_n_total,), dtype=bool)
            mask_high = np.zeros((_n_total,), dtype=bool)
            mask_low[: int(_low_n)] = True
            mask_high[int(_low_n): int(_low_n) + int(_high_n)] = True
            mask_combined = mask_low | mask_high

            low_range = [
                _as_optional_float(ds_meta.get("ood_low_alt_min_km")),
                _as_optional_float(ds_meta.get("ood_low_alt_max_km")),
            ]
            high_range = [
                _as_optional_float(ds_meta.get("ood_high_alt_min_km")),
                _as_optional_float(ds_meta.get("ood_high_alt_max_km")),
            ]
            low_label = (
                f"{low_range[0]:.1f}-{low_range[1]:.1f} km"
                if low_range[0] is not None and low_range[1] is not None
                else "lower OOD rows"
            )
            high_label = (
                f"{high_range[0]:.1f}-{high_range[1]:.1f} km"
                if high_range[0] is not None and high_range[1] is not None
                else "upper OOD rows"
            )
            ood_table = {
                "train_alt_range_km": [alt_lo, alt_hi],
                "classification": "metadata_row_ranges",
                "definitions": {
                    "lower_ood_rows": [0, int(_low_n)],
                    "upper_ood_rows": [int(_low_n), int(_low_n) + int(_high_n)],
                    "lower_ood_km": low_range,
                    "upper_ood_km": high_range,
                    "in_band_km": [alt_lo, alt_hi],
                },
                "lower_ood": _region_stats(mask_low, low_label),
                "in_band": _region_stats(np.zeros((_n_total,), dtype=bool), f"{alt_lo:.1f}-{alt_hi:.1f} km"),
                "upper_ood": _region_stats(mask_high, high_label),
                "combined_ood": _region_stats(mask_combined, "lower + upper OOD"),
                "ood_combined_meta": {
                    "ood_low_n_generated": int(_low_n),
                    "ood_high_n_generated": int(_high_n),
                    "ood_low_alt_range_km": low_range,
                    "ood_high_alt_range_km": high_range,
                },
            }
        else:
            mask_in = mask_pack["in_band"]
            mask_above = mask_pack["upper_ood"]
            below_thresh = float(mask_pack["lower_bounds_km"][0])
            above_thresh = float(mask_pack["upper_bounds_km"][1])

            ood_table = {
                "train_alt_range_km": [alt_lo, alt_hi],
                "classification": "altitude_margin",
                "ood_margin_pct": 10.0,
                "ood_margin_km": float(mask_pack["margin_km"]),
                "definitions": {
                    "lower_ood_km": mask_pack["lower_bounds_km"],
                    "in_band_km": mask_pack["in_band_bounds_km"],
                    "upper_ood_km": mask_pack["upper_bounds_km"],
                },
                "lower_ood": _region_stats(mask_pack["lower_ood"], f"{mask_pack['lower_bounds_km'][0]:.1f}-{mask_pack['lower_bounds_km'][1]:.1f} km"),
                "in_band": _region_stats(mask_pack["in_band"], f"{mask_pack['in_band_bounds_km'][0]:.1f}-{mask_pack['in_band_bounds_km'][1]:.1f} km"),
                "upper_ood": _region_stats(mask_pack["upper_ood"], f"{mask_pack['upper_bounds_km'][0]:.1f}-{mask_pack['upper_bounds_km'][1]:.1f} km"),
                "in_train": _region_stats(mask_in, f"{below_thresh:.1f}-{above_thresh:.1f} km"),
                "above_train": _region_stats(mask_above, f"> {above_thresh:.1f} km"),
            }
        # If the dataset is an ood_combined file with embedded split metadata, annotate.
        if _ds_role == "ood_combined" and not _has_exact_ood_split:
            try:
                ood_table["ood_combined_meta"] = {
                    "ood_low_n_generated": int(ds_meta["ood_low_n"]) if "ood_low_n" in ds_meta else None,
                    "ood_high_n_generated": int(ds_meta["ood_high_n"]) if "ood_high_n" in ds_meta else None,
                    "ood_low_alt_range_km": [
                        float(ds_meta["ood_low_alt_min_km"]) if "ood_low_alt_min_km" in ds_meta else None,
                        float(ds_meta["ood_low_alt_max_km"]) if "ood_low_alt_max_km" in ds_meta else None,
                    ],
                    "ood_high_alt_range_km": [
                        float(ds_meta["ood_high_alt_min_km"]) if "ood_high_alt_min_km" in ds_meta else None,
                        float(ds_meta["ood_high_alt_max_km"]) if "ood_high_alt_max_km" in ds_meta else None,
                    ],
                }
            except Exception:
                pass

    # Assemble angular_metrics block (richer than old a_vectorial)
    _ang_masked_mean = float(np.mean(masked_ang_deg)) if masked_ang_deg.size > 0 else None
    _ang_masked_p50 = float(np.median(masked_ang_deg)) if masked_ang_deg.size > 0 else None
    _ang_masked_p90 = float(np.percentile(masked_ang_deg, 90)) if masked_ang_deg.size > 0 else None
    _angular_metrics: Dict[str, Any] = {
        "residual_all": {
            "mean_deg": float(ang_sum_deg / max(ang_count, 1)),
            "max_deg": float(ang_max_deg),
            "mean_cossim": float(ang_sum_cossim / max(ang_count, 1)),
            "p50_deg": float(np.median(ang_deg_all)),
            "p90_deg": float(np.percentile(ang_deg_all, 90)),
            "p99_deg": float(np.percentile(ang_deg_all, 99)),
        },
        "residual_masked": {
            "direction_floor_abs": direction_floor_abs,
            "mask_frac": mask_frac,
            "N_masked": int(dir_mask.sum()),
            "mean_deg": _ang_masked_mean,
            "p50_deg": _ang_masked_p50,
            "p90_deg": _ang_masked_p90,
        },
        "norm_binned": norm_binned_ang,
    }
    if degree_min >= 0 and ang_total_deg is not None:
        _angular_metrics["total_approx"] = {
            "note": total_ang_note,
            "mean_deg": float(np.mean(ang_total_deg)),
            "p50_deg": float(np.median(ang_total_deg)),
            "p90_deg": float(np.percentile(ang_total_deg, 90)),
            "p99_deg": float(np.percentile(ang_total_deg, 99)),
            "max_deg": float(np.max(ang_total_deg)),
        }
    else:
        _angular_metrics["total_approx"] = {"note": total_ang_note}

    metrics: Dict[str, Any] = {
        "U": compute_metrics(u_err, u_true.reshape(-1), rel_floor_abs=u_rel_floor_abs).__dict__,
        "|a|": compute_metrics(a_mag_err, a_true_mag.reshape(-1), rel_floor_abs=a_rel_floor_abs).__dict__,
        "a_vectorial": {
            "mean_deg": float(ang_sum_deg / max(1, ang_count)),
            "max_deg": float(ang_max_deg),
            "mean_cossim": float(ang_sum_cossim / max(1, ang_count)),
        },
        "angular_metrics": _angular_metrics,
        "a_directional": directional_metrics,
        "ood_table": ood_table,
        "model_degree_min": int(degree_min),
        "model_degree_max": (int(model_degree_max) if model_degree_max is not None else None),
        "dataset_degree_min": _as_optional_int(ds_meta.get("degree_min")),
        "dataset_degree_max": (int(ds_degree_max) if ds_degree_max is not None else None),
        "central_body": (model_body or ds_body or "moon"),
        "decomposition_frame": "approximate_rtn_like_without_velocity",
        "n_points": int(u_true.shape[0]),
        "device": str(device),
        "a_sign": float(a_sign),
        "mu_si": float(mu_si),
        "r_ref_m": float(r_ref_m),
        "load_report": load_report,
    }

    spatial_u = spatial_rmse_by_altitude(alt_km_all.reshape(-1), u_err, bin_km=alt_bin_km)
    spatial_a = spatial_rmse_by_altitude(alt_km_all.reshape(-1), a_mag_err, bin_km=alt_bin_km)
    spatial_u_mape = spatial_mape_by_altitude(
        alt_km_all.reshape(-1), u_true.reshape(-1), u_pred.reshape(-1),
        bin_km=alt_bin_km, rel_floor_abs=u_rel_floor_abs,
    )
    spatial_a_mape = spatial_mape_by_altitude(
        alt_km_all.reshape(-1), a_true_mag.reshape(-1), a_pred_mag.reshape(-1),
        bin_km=alt_bin_km, rel_floor_abs=a_rel_floor_abs,
    )

    plots_dir = out_dir
    plots_dir.mkdir(parents=True, exist_ok=True)

    save_parity_plot(
        y_true=u_true.reshape(-1),
        y_pred=u_pred.reshape(-1),
        path=plots_dir / "parity_U.png",
        title="Potential: prediction vs truth",
    )

    rel_a_all = bounded_relative_error_pct(a_pred_mag, a_true_mag, rel_floor_abs=a_rel_floor_abs)
    save_scatter_altitude(
        alt_km=alt_km_all.reshape(-1),
        rel_err_pct=rel_a_all.reshape(-1),
        path=plots_dir / "scatter_relerr_accel_vs_alt.png",
        title="|a| bounded relative error vs altitude",
    )

    save_log_hist(u_err, plots_dir / "hist_abs_err_U_log.png", "U absolute error histogram")
    save_log_hist(a_mag_err, plots_dir / "hist_abs_err_accelmag_log.png", "|a| absolute error histogram")

    # Percentage error histograms
    rel_u_pct = bounded_relative_error_pct(u_pred.reshape(-1), u_true.reshape(-1), rel_floor_abs=u_rel_floor_abs)
    save_pct_error_hist(
        rel_u_pct,
        plots_dir / "hist_rel_err_U_pct.png",
        "U bounded relative error distribution",
    )
    save_pct_error_hist(
        rel_a_all.reshape(-1),
        plots_dir / "hist_rel_err_accel_pct.png",
        "|a| bounded relative error distribution",
    )

    # Altitude-binned MAPE bar charts
    save_binned_mae_pct(
        alt_km_all.reshape(-1),
        rel_u_pct,
        plots_dir / "binned_mape_U_vs_alt.png",
        "U bounded mean absolute % error by altitude",
        bin_km=alt_bin_km,
    )
    save_binned_mae_pct(
        alt_km_all.reshape(-1),
        rel_a_all.reshape(-1),
        plots_dir / "binned_mape_accel_vs_alt.png",
        "|a| bounded mean absolute % error by altitude",
        bin_km=alt_bin_km,
    )

    # angular error histogram
    if ang_deg_plot:
        ang_plot = np.concatenate(ang_deg_plot, axis=0).reshape(-1)
        save_hist_angular_deg(ang_plot, plots_dir / "hist_angular_err_deg.png", "Angular error (deg) histogram")

    # masked angular error histogram (excludes near-zero residuals below direction floor)
    if masked_ang_deg.size > 0:
        save_hist_angular_deg(
            masked_ang_deg,
            plots_dir / "angular_error_masked.png",
            "Masked Residual Angular Error (||da_true|| > floor)",
        )

    # OOD bar chart: accel RMSE for lower_ood / in_band / upper_ood
    if ood_table is not None:
        try:
            _ood_labels, _ood_vals = [], []
            for _rk in ("lower_ood", "in_band", "upper_ood"):
                _r = ood_table.get(_rk, {})
                _ood_labels.append(_r.get("region", _rk))
                _ood_vals.append(float(_r.get("RMSE_accel_norm", 0.0)))
            if any(v > 0 for v in _ood_vals):
                apply_professional_style()
                _colors = ["#E74C3C", "#2ECC71", "#3498DB"]
                _fig, _ax = plt.subplots(figsize=(7, 4.5))
                _bars = _ax.bar(_ood_labels, _ood_vals, color=_colors, alpha=0.85, edgecolor="white")
                _ax.set_ylabel("Accel RMSE [m/s²]")
                _ax.set_title("|a| RMSE by Altitude Region (OOD ±10%)", pad=12)
                for _b, _v in zip(_bars, _ood_vals):
                    if _v > 0:
                        _ax.text(_b.get_x() + _b.get_width() / 2, _v * 1.02, f"{_v:.3e}",
                                 ha="center", va="bottom", fontsize=8)
                _fig.tight_layout()
                _fig.savefig(plots_dir / "ood_bar_accel_rmse.png", dpi=300, bbox_inches="tight")
                plt.close(_fig)
        except Exception as _ood_plot_err:
            print(f"[warn] OOD bar chart failed: {_ood_plot_err}")

    # --- Benchmark: throughput + latency ---
    bench_n = min(int(total), 200_000)

    if suffix in [".h5", ".hdf5"]:
        with h5py.File(data_path, "r") as f:
            dsn = _discover_h5_dataset_name(data_path, preferred=dataset_name)
            x_for_bench = np.asarray(f[dsn][0:bench_n, 0:3], dtype=np.float32, order="C")
    else:
        obj = torch.load(data_path, map_location="cpu")
        t = obj["data"] if isinstance(obj, dict) and "data" in obj else obj
        x_for_bench = t[0:bench_n, 0:3].float().contiguous().numpy()

    throughput_points_per_sec: Dict[str, float] = {}
    latency_ms_batch1: Dict[str, float] = {}

    # current device
    throughput_points_per_sec[str(device)] = benchmark_throughput(
        model, scaler, x_for_bench, device=device, batch_size=min(batch_size, 8192),
        a_sign=a_sign, mu_si=mu_si, degree_min=degree_min,
    )
    latency_ms_batch1[str(device)] = benchmark_latency_ms(
        model, scaler, x_for_bench, device=device,
        a_sign=a_sign, mu_si=mu_si, degree_min=degree_min,
    )

    # CPU baseline
    cpu = torch.device("cpu")
    scaler_cpu = ScalerPack.load(model_dir / "scaler.json", device=cpu, dtype=torch.float32)
    model_cpu = build_model_from_config(cfg, device=cpu)
    model_cpu.load_state_dict(state, strict=True)

    throughput_points_per_sec["cpu"] = benchmark_throughput(
        model_cpu, scaler_cpu, x_for_bench, device=cpu, batch_size=min(4096, batch_size),
        a_sign=a_sign, mu_si=mu_si, degree_min=degree_min,
    )
    latency_ms_batch1["cpu"] = benchmark_latency_ms(
        model_cpu, scaler_cpu, x_for_bench, device=cpu,
        a_sign=a_sign, mu_si=mu_si, degree_min=degree_min,
    )

    report = {
        "metrics": metrics,
        "evaluation_contract": {
            "model_degree_min": int(degree_min),
            "model_degree_max": (int(model_degree_max) if model_degree_max is not None else None),
            "dataset_degree_min": _as_optional_int(ds_meta.get("degree_min")),
            "dataset_degree_max": (int(ds_degree_max) if ds_degree_max is not None else None),
            "central_body": (model_body or ds_body or "moon"),
            "train_alt_range_km": ([_train_alt_min_km, _train_alt_max_km] if _train_alt_min_km is not None and _train_alt_max_km is not None else None),
            "ood_band_definitions": (ood_table.get("definitions") if isinstance(ood_table, dict) else None),
            "directional_frame": "approximate_rtn_like_without_velocity",
        },
        "spatial_breakdown": {
            "U_rmse_by_alt":  spatial_u,
            "|a|_rmse_by_alt": spatial_a,
            "U_mape_by_alt":  spatial_u_mape,
            "|a|_mape_by_alt": spatial_a_mape,
        },
        "throughput_points_per_sec": throughput_points_per_sec,
        "latency_ms_batch1": latency_ms_batch1,
        "artifacts": {
            "model_dir": str(model_dir),
            "data_path": str(data_path),
            "config": str(cfg_path),
            "scaler": str(sc_path),
            "checkpoint": str(ckpt_path),
            "out_dir": str(out_dir),
        },
    }
    (out_dir / "eval_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    def write_bins_csv(bins: Dict[str, Any], path: Path, extra_cols: List[str] = []) -> None:
        header = "alt_km_lo,alt_km_hi,n,rmse"
        if extra_cols:
            header += "," + ",".join(extra_cols)
        rows = [header]
        for b in bins.get("bins", []):
            row = f"{b['alt_km_lo']},{b['alt_km_hi']},{b['n']},{b.get('rmse', '')}"
            for col in extra_cols:
                row += f",{b.get(col, '')}"
            rows.append(row)
        path.write_text("\n".join(rows), encoding="utf-8")

    def write_mape_csv(bins: Dict[str, Any], path: Path) -> None:
        rows = ["alt_km_lo,alt_km_hi,n,mape_pct,p50_pct,p90_pct"]
        for b in bins.get("bins", []):
            rows.append(
                f"{b['alt_km_lo']},{b['alt_km_hi']},{b['n']},"
                f"{b.get('mape_pct', '')},{b.get('p50_pct', '')},{b.get('p90_pct', '')}"
            )
        path.write_text("\n".join(rows), encoding="utf-8")

    write_bins_csv(spatial_u, out_dir / "spatial_rmse_U.csv")
    write_bins_csv(spatial_a, out_dir / "spatial_rmse_accelmag.csv")
    write_mape_csv(spatial_u_mape, out_dir / "spatial_mape_U.csv")
    write_mape_csv(spatial_a_mape, out_dir / "spatial_mape_accel.csv")

    # --- metrics_summary.csv ---
    _ms_rows = ["metric,mae,rmse,rel_mean_pct,rel_p50_pct,rel_p90_pct,nrmse_pct,linf"]
    for _key in ("U", "|a|"):
        _d = metrics[_key]
        _ms_rows.append(
            f"{_key},{_d['mae']},{_d['rmse']},{_d['rel_mean_pct']},"
            f"{_d['rel_p50_pct']},{_d['rel_p90_pct']},{_d['nrmse_pct']},{_d['linf']}"
        )
    (out_dir / "metrics_summary.csv").write_text("\n".join(_ms_rows), encoding="utf-8")

    # --- altitude_binned_metrics.csv (combined U + accel RMSE + MAPE) ---
    _ab_rows = ["alt_km_lo,alt_km_hi,n,rmse_U,rmse_accel,mape_U_pct,mape_accel_pct,mape_U_p90_pct,mape_accel_p90_pct"]
    _rmse_u_bins  = {(b["alt_km_lo"], b["alt_km_hi"]): b for b in spatial_u.get("bins", [])}
    _rmse_a_bins  = {(b["alt_km_lo"], b["alt_km_hi"]): b for b in spatial_a.get("bins", [])}
    _mape_u_bins  = {(b["alt_km_lo"], b["alt_km_hi"]): b for b in spatial_u_mape.get("bins", [])}
    _mape_a_bins  = {(b["alt_km_lo"], b["alt_km_hi"]): b for b in spatial_a_mape.get("bins", [])}
    _all_bin_keys = sorted(set(_rmse_u_bins) | set(_rmse_a_bins) | set(_mape_u_bins) | set(_mape_a_bins))
    for _k in _all_bin_keys:
        _ru = _rmse_u_bins.get(_k, {}); _ra = _rmse_a_bins.get(_k, {})
        _mu = _mape_u_bins.get(_k, {}); _ma = _mape_a_bins.get(_k, {})
        _n = _ru.get("n", _ra.get("n", _mu.get("n", _ma.get("n", 0))))
        _ab_rows.append(
            f"{_k[0]},{_k[1]},{_n},"
            f"{_ru.get('rmse','')},"
            f"{_ra.get('rmse','')},"
            f"{_mu.get('mape_pct','')},"
            f"{_ma.get('mape_pct','')},"
            f"{_mu.get('p90_pct','')},"
            f"{_ma.get('p90_pct','')}"
        )
    (out_dir / "altitude_binned_metrics.csv").write_text("\n".join(_ab_rows), encoding="utf-8")

    # --- ood_metrics.csv ---
    if ood_table is not None:
        _ood_header = "region,N,RMSE_U,MAE_U,RMSE_accel_norm,MAE_accel_norm,robust_rel_accel_mean_pct,robust_rel_accel_p90_pct,angular_error_mean_deg,angular_error_p90_deg"
        _ood_rows = [_ood_header]
        for _region_key in ("lower_ood", "in_band", "upper_ood"):
            _r = ood_table.get(_region_key, {})
            if _r.get("N", 0) > 0:
                _ood_rows.append(
                    f"{_r.get('region','')},{_r['N']},"
                    f"{_r.get('RMSE_U','')},{_r.get('MAE_U','')},"
                    f"{_r.get('RMSE_accel_norm','')},{_r.get('MAE_accel_norm','')},"
                    f"{_r.get('robust_rel_accel_mean_pct','')},{_r.get('robust_rel_accel_p90_pct','')},"
                    f"{_r.get('angular_error_mean_deg','')},{_r.get('angular_error_p90_deg','')}"
                )
            else:
                _ood_rows.append(f"{_region_key},0,,,,,,,,")
        (out_dir / "ood_metrics.csv").write_text("\n".join(_ood_rows), encoding="utf-8")

    # --- acceleration_decomposition.csv ---
    _ad_rows = [
        "component,mae,rmse",
        f"radial,{directional_metrics['accel_err_radial_mae']},{directional_metrics['accel_err_radial_rmse']}",
        f"cross_radial_norm,{directional_metrics['accel_err_cross_radial_mae']},{directional_metrics['accel_err_cross_radial_rmse']}",
        f"approx_T,{directional_metrics.get('approx_T_rmse','')},{directional_metrics.get('approx_T_rmse','')}",
        f"approx_N,{directional_metrics.get('approx_N_rmse','')},{directional_metrics.get('approx_N_rmse','')}",
    ]
    (out_dir / "acceleration_decomposition.csv").write_text("\n".join(_ad_rows), encoding="utf-8")

    print("\n==================== EVAL SUMMARY ====================")
    print(f"Model dir : {model_dir}")
    print(f"Data      : {data_path}")
    print(f"Device    : {device}")
    print(f"Points    : {metrics['n_points']}")
    print(f"a_sign    : {metrics['a_sign']:+.1f}")
    print(f"mu_si     : {metrics['mu_si']:.6e} m^3/s^2")
    def _fmt(d: Dict[str, Any]) -> str:
        return (f"MAE={d['mae']:.4e}  RMSE={d['rmse']:.4e}  "
                f"Rel(mean)={d['rel_mean_pct']:.3f}%  NRMSE={d['nrmse_pct']:.3f}%  L_inf={d['linf']:.4e}")
    print("--- U ---")
    print("  " + _fmt(metrics["U"]))
    print("--- |a| ---")
    print("  " + _fmt(metrics["|a|"]))
    print("--- a vectorial ---")
    av = metrics["a_vectorial"]
    print(f"  mean_deg={av['mean_deg']:.3f} deg  max_deg={av['max_deg']:.3f} deg  "
          f"mean_cossim={av['mean_cossim']:.6f}")
    # Print altitude-binned bounded relative-error summary (first 5 bins)
    print("--- U relative error by altitude (first 5 bins) ---")
    for b in spatial_u_mape.get("bins", [])[:5]:
        print(f"  [{b['alt_km_lo']:.0f}-{b['alt_km_hi']:.0f} km]  "
              f"Mean={b['mape_pct']:.3f}%  P90={b['p90_pct']:.3f}%  n={b['n']}")
    print("--- |a| relative error by altitude (first 5 bins) ---")
    for b in spatial_a_mape.get("bins", [])[:5]:
        print(f"  [{b['alt_km_lo']:.0f}-{b['alt_km_hi']:.0f} km]  "
              f"Mean={b['mape_pct']:.3f}%  P90={b['p90_pct']:.3f}%  n={b['n']}")
    print("--- a directional decomposition (radial / cross-radial; approx T/N without velocity) ---")
    directional = metrics.get("a_directional", {})
    if directional:
        print(f"  radial:        MAE={directional['accel_err_radial_mae']:.4e}  RMSE={directional['accel_err_radial_rmse']:.4e}")
        print(f"  cross-radial:  MAE={directional['accel_err_cross_radial_mae']:.4e}  RMSE={directional['accel_err_cross_radial_rmse']:.4e}")
        print(f"  approx_T RMSE: {directional['approx_T_rmse']:.4e}  |  approx_N RMSE: {directional['approx_N_rmse']:.4e}")
    print("--- OOD table (+/-10% beyond training altitude band) ---")
    ood = metrics.get("ood_table")
    if ood:
        print(f"  Training band: {ood['train_alt_range_km'][0]:.0f}-{ood['train_alt_range_km'][1]:.0f} km")
        for key in ("lower_ood", "in_band", "upper_ood"):
            row = ood[key]
            if row["N"] > 0:
                print(f"  {row['region']:40s}  n={row['N']:7d}  "
                      f"|a|_RMSE={row['RMSE_accel_norm']:.4e}  U_RMSE={row['RMSE_U']:.4e}  "
                      f"Rel={row['robust_rel_accel_mean_pct']:.3f}%")
            else:
                print(f"  {key:40s}  n=0 (no samples in this region)")
    else:
        print("  [skipped: alt_min_km / alt_max_km not found in model config]")
    print("--- Throughput (points/sec; U+grad) ---")
    for k, v in throughput_points_per_sec.items():
        print(f"{k:>8s}: {v:,.0f}")
    print("--- Latency (ms; batch_size=1; U+grad) ---")
    for k, v in latency_ms_batch1.items():
        print(f"{k:>8s}: {v:,.3f} ms")
    print(f"\nSaved: {out_dir / 'eval_report.json'}")
    print("Plots:")
    for p in sorted(plots_dir.glob("*.png")):
        print(f"  {p.name}")
    print("======================================================\n")


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=None, help="Directory containing config.json, scaler.json, and checkpoints/ckpt_best.pt or ckpt_last.pt. If omitted, auto-detects the newest lunar-compatible run near this script or via env ST_LRPS_MODEL_DIR.")
    ap.add_argument("--data", default=None, help="Test dataset path (.h5/.hdf5/.pt). If omitted, auto-detect the newest lunar-compatible dataset near model-dir.")
    ap.add_argument("--test-data", default=None, help="Optional independent in-band test dataset. Evaluated into <out>/test when provided.")
    ap.add_argument("--ood-data", default=None, help="Optional OOD/extrapolation dataset. Evaluated into <out>/ood when provided.")
    ap.add_argument("--use-config-datasets", action="store_true", help="Use test_data_path / ood_data_path from config.json when explicit paths are not supplied.")
    ap.add_argument("--dataset-name", default="data", help="HDF5 dataset name (default: data, auto-fallback to first dataset).")
    ap.add_argument("--out", default=None, help="Output directory (default: <model-dir>/eval_results).")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument("--a-sign", type=float, default=1.0, help="Fallback a_sign if config.json lacks resolved_a_sign. +1 for geodesy, -1 for physics.")
    ap.add_argument("--r-ref-m", type=float, default=None, help="Reference radius for altitude (meters). If omitted, auto-infer from dataset metadata and fall back to the lunar reference radius.")
    ap.add_argument("--alt-bin-km", type=float, default=50.0, help="Altitude bin size for spatial RMSE breakdown (km).")
    ap.add_argument("--start", type=int, default=0, help="Start row for evaluation.")
    ap.add_argument("--end", type=int, default=None, help="End row (exclusive) for evaluation.")
    ap.add_argument("--max-points-for-plots", type=int, default=500_000, help="Cap number of points kept for plots.")
    return ap.parse_args()


def _dataset_path_from_config(model_dir: Path, key: str) -> Optional[Path]:
    cfg_path = model_dir / "config.json"
    if not cfg_path.is_file():
        return None
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    # Try the exact key first, then the key without '_path' suffix (e.g. test_data_path -> test_data)
    for k in (key, key.replace("_path", "") if key.endswith("_path") else None):
        if k is None:
            continue
        value = cfg.get(k)
        if value:
            path = Path(str(value)).expanduser()
            if not path.is_absolute():
                path = (model_dir / path).resolve()
            if path.exists():
                return path
    return None


def _auto_find_testset(model_dir: Path) -> Optional[Path]:
    """Fallback: try to recover a test dataset from config.json."""
    return _dataset_path_from_config(model_dir, "test_data_path")


def main() -> None:
    args = parse_args()

    # --- model dir auto-detect ---
    model_dir: Optional[Path] = Path(args.model_dir).resolve() if args.model_dir else None
    if model_dir is None:
        env_md = os.environ.get("ST_LRPS_MODEL_DIR")
        if env_md:
            model_dir = Path(env_md).resolve()

    if model_dir is None:
        # start search near this script
        script_dir = Path(__file__).resolve().parent
        found_md = _auto_find_model_dir(script_dir)
        if found_md is not None:
            model_dir = found_md
            print(f"[auto] Using model-dir: {model_dir}")

    if model_dir is None:
        raise SystemExit(
            "Missing --model-dir and auto-detect failed."
            "Fix:"
            "  python st_lrps_evaluate.py --model-dir path\to\run_dir [--data path\to\test.h5]"
            "or set env var (PowerShell):"
            "  $env:ST_LRPS_MODEL_DIR='C:\\path\\to\\run_dir'"
        )

    if not model_dir.exists():
        raise FileNotFoundError(model_dir)

    # --- data auto-detect (existing behavior) ---
    data_path = Path(args.data).resolve() if args.data else None
    if data_path is None:
        if args.use_config_datasets:
            data_path = _dataset_path_from_config(model_dir, "test_data_path")
        if data_path is None:
            found = _auto_find_testset(model_dir)
            if found is not None:
                data_path = found
        if data_path is None:
            raise SystemExit(
                "No --data provided and auto-detect failed. "
                "Provide --data path/to/test.h5 or use --use-config-datasets with a config.json "
                "that contains test_data_path."
            )
        print(f"[auto] Using dataset: {data_path}")

    
    # --- reference radius (altitude) ---
    r_ref_m_resolved: Optional[float] = float(args.r_ref_m) if args.r_ref_m is not None else None
    if r_ref_m_resolved is None:
        inferred = infer_r_ref_m_from_dataset(data_path, dataset_name=str(args.dataset_name))
        if inferred is not None:
            r_ref_m_resolved = float(inferred)
            print(f"[auto] Using r_ref_m from dataset meta: {r_ref_m_resolved:g} m")
        else:
            r_ref_m_resolved = float(MOON_R_REF_M_DEFAULT)
            print(f"[auto] r_ref_m not found in dataset meta; falling back to lunar reference radius: {r_ref_m_resolved:g} m")

    out_dir = Path(args.out).resolve() if args.out else (model_dir / "eval_results")
    device = get_device(args.device)

    jobs: List[Tuple[str, Path, Path]] = [("primary", data_path, out_dir)]
    test_path = Path(args.test_data).resolve() if args.test_data else None
    ood_path = Path(args.ood_data).resolve() if args.ood_data else None
    if args.use_config_datasets:
        test_path = test_path or _dataset_path_from_config(model_dir, "test_data_path")
        ood_path = ood_path or _dataset_path_from_config(model_dir, "ood_data_path")
    if test_path is not None and test_path != data_path:
        if not test_path.exists():
            raise FileNotFoundError(test_path)
        jobs.append(("test", test_path, out_dir / "test"))
    if ood_path is not None:
        if not ood_path.exists():
            raise FileNotFoundError(ood_path)
        jobs.append(("ood", ood_path, out_dir / "ood"))

    for label, job_data_path, job_out_dir in jobs:
        print(f"[eval:{label}] dataset={job_data_path}")
        evaluate(
            model_dir=model_dir,
            data_path=job_data_path,
            out_dir=job_out_dir,
            device=device,
            batch_size=int(args.batch_size),
            a_sign=float(args.a_sign),
            r_ref_m=float(r_ref_m_resolved),
            alt_bin_km=float(args.alt_bin_km),
            dataset_name=str(args.dataset_name),
            start=int(args.start),
            end=(int(args.end) if args.end is not None else None),
            max_points_for_plots=int(args.max_points_for_plots),
        )


if __name__ == "__main__":
    main()
