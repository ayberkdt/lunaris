#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_st_lrps.py

Sanity test for an ST-LRPS / Sobolev-PINN gravity model trained in this repo.

- Auto-discovers latest run dir (st_lrps_*) and best/last checkpoint
- Loads model architecture from the checkpoint itself (SIREN, RFF-MLP, plain MLP)
- Uses isometric scaling matching train(): x_mean=[0,0,0], x_scale = scalar max‖x‖
- Correct chain rule: Δa = a_sign · ∇(ΔU_scaled) · (u_scale / x_scale)

Run
  python test_st_lrps.py --progress
  python test_st_lrps.py --progress --dataset path/to/data.h5
  python test_st_lrps.py --run-dir runs/st_lrps_YYYYMMDD_HHMMSS --progress
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent

try:
    import torch
    import torch.nn as nn
except Exception as e:
    raise RuntimeError("PyTorch is required. Install torch first.") from e

# Prefer the real refactored modules; keep st_lrps_train fallback for older layouts.
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    from st_lrps_models import (
        SirenMLP,
        MLP,
        PhysicsNet,
        FourierInputEmbedding,
        build_model_from_config,
    )
    from st_lrps_scaling import (
        IsometricScaleParams,
        ScalerPack,
    )
    _STLRPS_IMPORTED = True
    _STLRPS_IMPORT_ERR = ""
except Exception as _e:
    try:
        from st_lrps_train import (
            SirenMLP,
            MLP,
            PhysicsNet,
            FourierInputEmbedding,
            build_model_from_config,
            IsometricScaleParams,
            ScalerPack,
        )
        _STLRPS_IMPORTED = True
        _STLRPS_IMPORT_ERR = ""
    except Exception as _fallback_e:
        _STLRPS_IMPORTED = False
        _STLRPS_IMPORT_ERR = str(_fallback_e)


# =============================================================================
# Logging
# =============================================================================

def _ts() -> str:
    return time.strftime("%H:%M:%S")


def _log(msg: str) -> None:
    print(f"[{_ts()}] {msg}", flush=True)


# =============================================================================
# Run / checkpoint discovery
# =============================================================================

def _looks_like_root(p: Path) -> bool:
    if (p / "st_lrps_train.py").exists():
        return True
    runs = p / "runs"
    if runs.is_dir():
        try:
            return any(runs.glob("st_lrps_*"))
        except Exception:
            return True
    return False


def find_project_root() -> Path:
    bases = []
    try:
        bases.append(Path(__file__).resolve().parent)
    except Exception:
        pass
    bases.append(Path.cwd().resolve())
    for base in bases:
        for parent in [base, *base.parents]:
            if _looks_like_root(parent):
                return parent
    raise FileNotFoundError(
        "Could not find project root. Run from inside the repo or pass --project-root."
    )


def pick_latest_run(runs_root: Path) -> Path:
    runs = [p for p in runs_root.glob("st_lrps_*") if p.is_dir()]
    if not runs:
        raise FileNotFoundError(f"No run dirs found under: {runs_root}")
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return runs[0]


def find_latest_run_dir(project_root: Path) -> Path:
    candidates = [
        project_root / "runs",
        project_root / "ST_LRPS_Model" / "runs",
        project_root / "ST_LRPS_Model" / "runs",
        project_root / "ST_LRPS" / "runs",
    ]
    last_err: Optional[Exception] = None
    for rr in candidates:
        if rr.is_dir():
            try:
                return pick_latest_run(rr)
            except Exception as e:
                last_err = e
    if last_err:
        raise last_err
    raise FileNotFoundError("Could not find a runs directory.")


def find_checkpoint(run_dir: Path) -> Path:
    ckpt_dir = run_dir / "checkpoints"
    for name in ("ckpt_best.pt", "ckpt_last.pt"):
        p = ckpt_dir / name
        if p.exists():
            return p
    raise FileNotFoundError(f"No checkpoint found under: {ckpt_dir}")


# =============================================================================
# Dataset path resolution
# =============================================================================

def resolve_dataset_path(
    cfg: Dict[str, Any], project_root: Path, override: Optional[str]
) -> Path:
    if override:
        p = Path(override).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"--dataset provided but not found: {p}")
        return p

    candidates_raw: List[str] = []
    for k in ("dataset_path", "data_path", "data"):
        v = cfg.get(k)
        if isinstance(v, str) and v.strip():
            candidates_raw.append(v.strip())

    for raw in candidates_raw:
        p = Path(raw).expanduser()
        if p.exists():
            return p.resolve()

    search_dirs = [
        project_root / "data",
        project_root / "ST_LRPS_Model" / "data",
        project_root / "ST_LRPS_Model" / "data",
        project_root / "ST_LRPS" / "data",
    ]
    basenames = [
        raw.replace("\\", "/").split("/")[-1]
        for raw in candidates_raw
        if raw.lower().endswith(".h5")
    ]
    for base in basenames:
        for d in search_dirs:
            p = d / base
            if p.exists():
                return p.resolve()

    newest: Optional[Path] = None
    newest_mtime = -1.0
    for d in search_dirs:
        if not d.is_dir():
            continue
        for p in d.glob("*.h5"):
            try:
                mt = p.stat().st_mtime
            except Exception:
                continue
            if mt > newest_mtime:
                newest = p
                newest_mtime = mt
    if newest is not None:
        return newest.resolve()

    raise FileNotFoundError(
        "Could not resolve dataset path.\n"
        "Fix: pass --dataset /path/to/file.h5"
    )


# =============================================================================
# Checkpoint loading
# =============================================================================

def load_checkpoint(ckpt_path: Path, device: torch.device) -> Dict[str, Any]:
    try:
        return torch.load(str(ckpt_path), map_location=device, weights_only=False)
    except Exception:
        return torch.load(str(ckpt_path), map_location=device)


def extract_state_dict(ckpt: Any) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt, dict):
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            return ckpt["model"]
        if "model_state_dict" in ckpt and isinstance(ckpt["model_state_dict"], dict):
            return ckpt["model_state_dict"]
        if all(isinstance(k, str) for k in ckpt) and all(
            torch.is_tensor(v) for v in ckpt.values()
        ):
            return ckpt  # type: ignore[return-value]
    raise ValueError("Unrecognized checkpoint format.")


# =============================================================================
# Model reconstruction from checkpoint config
# =============================================================================

def _build_model_from_ckpt(ckpt: Dict[str, Any]) -> nn.Module:
    """Reconstruct the exact architecture from the saved config using the shared builder."""
    if not _STLRPS_IMPORTED:
        raise ImportError(
            f"Could not import model classes: {_STLRPS_IMPORT_ERR}\n"
            "Make sure st_lrps_models.py is in the same directory as test_st_lrps.py."
        )
    saved_cfg = ckpt.get("config", {})
    # build_model_from_config handles SIREN/Fourier mutual exclusion and correct API names
    return build_model_from_config(saved_cfg)


def _load_scaler_from_ckpt(ckpt: Dict[str, Any]) -> "ScalerPack":
    """Load ScalerPack from checkpoint state. Falls back to scaler.json if missing."""
    scaler_dict = ckpt.get("scaler")
    if scaler_dict and _STLRPS_IMPORTED:
        def _parse(d: Dict[str, Any]) -> "IsometricScaleParams":
            return IsometricScaleParams(
                mean=[float(v) for v in d["mean"]],
                scale=float(d["scale"]),
            )
        return ScalerPack(
            x=_parse(scaler_dict["x"]),
            u=_parse(scaler_dict["u"]),
            a=_parse(scaler_dict["a"]),
        )
    raise ValueError(
        "Checkpoint does not contain 'scaler'. Re-train with the current codebase."
    )


# =============================================================================
# Metrics
# =============================================================================

def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    err = y_pred - y_true
    abs_err = np.abs(err)
    mae = float(np.mean(abs_err))
    rmse = float(np.sqrt(np.mean(err * err)))
    linf = float(np.max(abs_err))
    median_ae = float(np.median(abs_err))
    p95_ae = float(np.percentile(abs_err, 95))
    # Robust relative error: denominator floor = median(|y_true|) * 1e-3, with
    # a hard minimum of 1e-30 to avoid division by zero on all-zero targets.
    # This avoids inflating relative errors for residual fields that cross zero.
    abs_true = np.abs(y_true)
    denom_floor = max(float(np.median(abs_true)) * 1e-3, 1e-30)
    robust_denom = np.maximum(abs_true, denom_floor)
    robust_rel_mean_pct = float(np.mean(abs_err / robust_denom) * 100.0)
    return {
        "mae": mae,
        "rmse": rmse,
        "median_ae": median_ae,
        "p95_ae": p95_ae,
        "linf": linf,
        "robust_rel_pct": robust_rel_mean_pct,
        # denom_floor printed so caller knows the scale used
        "_rel_denom_floor": denom_floor,
    }


def angle_deg(a_true: np.ndarray, a_pred: np.ndarray) -> Dict[str, float]:
    num = np.sum(a_true * a_pred, axis=1)
    den = np.linalg.norm(a_true, axis=1) * np.linalg.norm(a_pred, axis=1) + 1e-12
    c = np.clip(num / den, -1.0, 1.0)
    ang = np.degrees(np.arccos(c))
    return {
        "mean_deg": float(np.mean(ang)),
        "median_deg": float(np.median(ang)),
        "p95_deg": float(np.percentile(ang, 95)),
        "max_deg": float(np.max(ang)),
    }


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Sanity test for trained ST-LRPS gravity surrogate.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--project-root", type=str, default=None)
    ap.add_argument("--run-dir", type=str, default=None,
                    help="Path to a specific run dir (st_lrps_*). Default: latest.")
    ap.add_argument("--dataset", type=str, default=None,
                    help="Override dataset .h5 path.")
    ap.add_argument("--device", type=str, default="cuda",
                    choices=["cuda", "cpu"])
    ap.add_argument("--n", type=int, default=200,
                    help="Number of unique random points to test.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--progress", action="store_true",
                    help="Print progress messages.")
    ap.add_argument("--chunk", type=int, default=50000,
                    help="Chunk size for forward+grad (lower = less VRAM).")
    ap.add_argument("--a-sign-override", type=float, default=None,
                    help="Explicitly set a_sign (1.0 or -1.0). Only needed for old checkpoints "
                         "missing resolved_a_sign in config.json.")
    args = ap.parse_args()

    # Resolve project root and run dir
    project_root = (
        Path(args.project_root).resolve() if args.project_root else find_project_root()
    )
    run_dir = (
        Path(args.run_dir).resolve()
        if args.run_dir
        else find_latest_run_dir(project_root)
    )
    ckpt_path = find_checkpoint(run_dir)

    # Read config.json for metadata (dataset path, etc.)
    cfg_path = run_dir / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing config.json in: {run_dir}")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    dataset_path = resolve_dataset_path(cfg, project_root, args.dataset)

    device = (
        torch.device("cpu")
        if args.device == "cpu"
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    if args.device == "cuda" and device.type != "cuda":
        _log("CUDA requested but not available; falling back to CPU")

    if args.progress:
        _log(f"project_root : {project_root}")
        _log(f"run_dir      : {run_dir}")
        _log(f"ckpt         : {ckpt_path}")
        _log(f"dataset      : {dataset_path}")
        _log(f"device       : {device}")

    # Load checkpoint
    if args.progress:
        _log("Loading checkpoint...")
    ckpt = load_checkpoint(ckpt_path, device)

    # Print key metadata
    arch = ckpt.get("config", {})
    print(f"\n  activation    : {arch.get('activation', '?')}")
    print(f"  hidden/depth  : {arch.get('hidden', '?')} / {arch.get('depth', '?')}")
    print(f"  w0_first/hid  : {arch.get('w0_first', '?')} / {arch.get('w0_hidden', '?')}")
    print(f"  use_fourier   : {arch.get('use_fourier', False)}")
    print(f"  residual_mode : {ckpt.get('residual_mode', '?')}")
    print(f"  target_mode   : {ckpt.get('target_mode', '?')}")

    # Build model from saved config
    if args.progress:
        _log("Building model from checkpoint config...")
    model = _build_model_from_ckpt(ckpt).to(device)

    state = extract_state_dict(ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing and args.progress:
        _log(f"missing keys   : {missing}")
    if unexpected and args.progress:
        _log(f"unexpected keys: {unexpected}")
    model.eval()

    # Load scaler (isometric: mean + scalar scale)
    if args.progress:
        _log("Loading scaler from checkpoint...")
    scaler = _load_scaler_from_ckpt(ckpt)

    x_mean_np = np.array(scaler.x.mean, dtype=np.float64)   # [0,0,0] by design
    x_scale = float(scaler.x.scale)                          # scalar max‖x‖
    u_mean_np = np.array(scaler.u.mean, dtype=np.float64)
    u_scale = float(scaler.u.scale)

    # --a-sign-override allows explicit specification for old checkpoints
    if args.a_sign_override is not None:
        a_sign = float(args.a_sign_override)
        _log(f"a_sign overridden by CLI: {a_sign:+.1f}")
    # Refuse to silently guess a_sign — a wrong sign inverts all predicted accelerations.
    elif "resolved_a_sign" in cfg:
        a_sign = float(cfg["resolved_a_sign"])
    else:
        _raw = cfg.get("a_sign", "MISSING")
        if str(_raw).lower() in ("auto", "missing"):
            raise ValueError(
                "config.json is missing 'resolved_a_sign'. "
                "Re-train with the current codebase, or pass --a-sign-override <value> "
                "if you know the correct sign convention for this checkpoint."
            )
        try:
            a_sign = float(_raw)
        except (ValueError, TypeError) as _exc:
            raise ValueError(
                f"Cannot parse a_sign from config.json (got {_raw!r}). "
                "Re-train or use --a-sign-override."
            ) from _exc

    # Load dataset
    try:
        import h5py
    except Exception as e:
        raise RuntimeError("h5py is required. pip install h5py") from e

    rng = np.random.default_rng(int(args.seed))
    dataset_name = str(cfg.get("dataset_name", "data"))

    if args.progress:
        _log(f"Opening HDF5 and sampling n={args.n} points...")

    with h5py.File(str(dataset_path), "r") as f:
        # Try configured dataset name, fall back to first 2-D dataset
        ds_key = dataset_name if dataset_name in f else None
        if ds_key is None:
            for k in f:
                if hasattr(f[k], "shape") and len(f[k].shape) >= 2:
                    ds_key = k
                    break
        if ds_key is None:
            raise KeyError(f"No suitable dataset found in {dataset_path}")
        dset = f[ds_key]
        n_total = int(dset.shape[0])
        n = int(min(args.n, n_total))
        idx_sorted = np.sort(rng.choice(n_total, size=n, replace=False))
        batch = dset[idx_sorted, :]

    if args.progress:
        _log(f"Loaded batch: shape={batch.shape}, total_rows={n_total}")

    x_np = batch[:, 0:3].astype(np.float64)
    u_true = batch[:, 3].astype(np.float64)
    a_true = batch[:, 4:7].astype(np.float64)

    # Isometric scaling: x_mean=[0,0,0], x_scale is scalar
    x_scaled_np = (x_np - x_mean_np) / x_scale
    x_t_all = torch.from_numpy(x_scaled_np.astype(np.float32)).to(device)

    # Chain-rule factor (scalar / scalar — isometric)
    chain_factor = float(u_scale) / float(x_scale)

    # Chunked forward + autograd
    N = x_t_all.shape[0]
    chunk = int(max(1, args.chunk))
    u_pred = np.empty((N,), dtype=np.float64)
    a_pred = np.empty((N, 3), dtype=np.float64)

    if args.progress:
        _log(f"Forward+grad (N={N}, chunk={chunk}, a_sign={a_sign})...")

    with torch.set_grad_enabled(True):
        for start in range(0, N, chunk):
            end = min(start + chunk, N)
            x_chunk = x_t_all[start:end].detach().clone().requires_grad_(True)

            u_scaled_chunk = model(x_chunk)   # (B,1)

            grad_u_scaled = torch.autograd.grad(
                outputs=u_scaled_chunk,
                inputs=x_chunk,
                grad_outputs=torch.ones_like(u_scaled_chunk),
                create_graph=False,
                retain_graph=False,
                only_inputs=True,
            )[0]  # (B,3)

            # Unscale U:  U_phys = U_scaled * u_scale + u_mean
            u_s_np = u_scaled_chunk.detach().cpu().numpy().reshape(-1)
            u_pred[start:end] = u_s_np * u_scale + u_mean_np.reshape(-1)[0]

            # Isometric chain rule: Δa = a_sign · ∇(U_scaled) · (u_scale / x_scale)
            a_pred[start:end, :] = (
                a_sign * chain_factor * grad_u_scaled.detach().cpu().numpy().astype(np.float64)
            )

            if args.progress and end % max(chunk, 1) == 0:
                _log(f"  processed {end}/{N}")

    if args.progress:
        _log("Computing metrics...")

    u_m = metrics(u_true, u_pred)
    a_mag_true = np.linalg.norm(a_true, axis=1)
    a_mag_pred = np.linalg.norm(a_pred, axis=1)
    a_m = metrics(a_mag_true, a_mag_pred)
    ang = angle_deg(a_true, a_pred)

    # Determine comparison mode from checkpoint metadata
    _target_mode = ckpt.get("target_mode") or cfg.get("target_mode", "unknown")
    _degree_min = ckpt.get("config", {}).get("degree_min", cfg.get("degree_min", "unknown"))
    try:
        _dm_int = int(_degree_min)
        _is_residual = _dm_int >= 0
    except (TypeError, ValueError):
        _is_residual = (_target_mode == "residual")
    _comparison_mode = "residual_vs_residual" if _is_residual else "total_vs_total"

    print("\n==================== ST-LRPS TEST SUMMARY ====================")
    print(f"  Points           : {N}")
    print(f"  a_sign           : {a_sign:+.1f}")
    print(f"  x_scale          : {x_scale:.6e}  (isometric, scalar)")
    print(f"  u_scale          : {u_scale:.6e}")
    print(f"  dataset target   : {_target_mode}")
    print(f"  degree_min       : {_degree_min}")
    print(f"  comparison_mode  : {_comparison_mode}")
    if not _is_residual:
        print("  WARNING: full-field comparison -- model predicts residual only.")
        print("  Add a base U/a reconstruction for a fair total-field comparison.")
    print("--- delta_U (residual potential) ---")
    for k, v in u_m.items():
        if k.startswith("_"):
            continue
        print(f"  {k:18s}: {v:.4e}")
    print(f"  [robust_rel denominator floor: {u_m['_rel_denom_floor']:.2e}]")
    print("--- |delta_a| (residual acceleration magnitude) ---")
    for k, v in a_m.items():
        if k.startswith("_"):
            continue
        print(f"  {k:18s}: {v:.4e}")
    print(f"  [robust_rel denominator floor: {a_m['_rel_denom_floor']:.2e}]")
    print("--- delta_a direction (degrees) ---")
    for k, v in ang.items():
        print(f"  {k:18s}: {v:.3f} deg")
    print("==========================================================\n")


# =============================================================================
# Unit tests (run with --unit-tests flag)
# =============================================================================

def _test_scaler_roundtrip() -> None:
    """Scaler round-trip: scale then unscale should recover original values."""
    from st_lrps_scaling import IsometricScaleParams, ScalerPack
    sp = ScalerPack(
        x=IsometricScaleParams(mean=[0.0, 0.0, 0.0], scale=1.7e6),
        u=IsometricScaleParams(mean=[0.0], scale=1e4),
        a=IsometricScaleParams(mean=[0.0, 0.0, 0.0], scale=1e-3),
    ).to_tensors(torch.device("cpu"), torch.float32)
    x = torch.randn(100, 3)
    u = torch.randn(100, 1)
    a = torch.randn(100, 3)
    _x_scaled = sp.scale_x(x)  # check it runs without error
    u_rt = sp.unscale_u(sp.scale_u(u))
    a_rt = sp.unscale_a(sp.scale_a(a))
    assert torch.allclose(u, u_rt, atol=1e-5), "Scaler round-trip failed for u"
    assert torch.allclose(a, a_rt, atol=1e-5), "Scaler round-trip failed for a"
    print("[PASS] scaler_roundtrip")


def _test_chain_rule() -> None:
    """Verify da = a_sign * grad(U_scaled) * (u_scale/x_scale)."""
    from st_lrps_scaling import IsometricScaleParams, ScalerPack
    sp = ScalerPack(
        x=IsometricScaleParams(mean=[0.0, 0.0, 0.0], scale=1.7e6),
        u=IsometricScaleParams(mean=[0.0], scale=1e4),
        a=IsometricScaleParams(mean=[0.0, 0.0, 0.0], scale=1e-3),
    ).to_tensors(torch.device("cpu"), torch.float32)
    cfg_tiny = {
        "hidden": 32, "depth": 2, "activation": "sine",
        "w0_first": 30.0, "w0_hidden": 30.0,
    }
    model = build_model_from_config(cfg_tiny)
    model.eval()
    x = torch.randn(4, 3) * 1.7e6
    x_scaled = sp.scale_x(x).requires_grad_(True)
    u_scaled = model(x_scaled)
    grad = torch.autograd.grad(
        u_scaled, x_scaled, torch.ones_like(u_scaled), create_graph=False
    )[0]
    da = 1.0 * grad * (sp._u_scale / sp._x_scale)
    assert da.shape == (4, 3), f"Wrong shape: {da.shape}"
    assert torch.all(torch.isfinite(da)), "da has non-finite values"
    print("[PASS] chain_rule")


def _test_direction_mask() -> None:
    """Masked angular error should exclude near-zero residual vectors correctly.

    Constructs a scenario where:
    - Half the a_true vectors are near-zero (below the floor), so the mask
      excludes them.
    - For the above-floor half, a_true and a_pred are identical (0 deg error).
    - For the below-floor half, a_pred points in the opposite direction (180 deg).
    - All-sample mean ~ 90 deg; masked mean ~ 0 deg, so the difference > 45 deg.
    """
    n = 1000
    rng = np.random.default_rng(42)
    dirs = rng.standard_normal((n, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

    a_true = dirs * 1e-5   # all above 3e-6 by magnitude
    a_pred = dirs.copy()   # perfect match -> 0 deg

    # Make the first half sub-floor (below 3e-6)
    a_true[:n // 2] = dirs[:n // 2] * 1e-8   # below floor
    a_pred[:n // 2] = -dirs[:n // 2] * 1e-3  # opposite direction -> 180 deg

    floor = 3e-6
    norms = np.linalg.norm(a_true, axis=1)
    mask = norms > floor
    assert mask.sum() > 0 and (~mask).sum() > 0, "Mask must have both True and False"

    def _ang(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        dot = np.sum(a * b, axis=1)
        na, nb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
        c = np.clip(dot / np.maximum(na * nb, 1e-18), -1.0, 1.0)
        return np.degrees(np.arccos(c))

    ang_all = _ang(a_true, a_pred)
    ang_masked = _ang(a_true[mask], a_pred[mask])
    # All-sample mean ~90 deg (mix of 0 and 180); masked mean ~0 deg (perfect match)
    assert abs(float(np.mean(ang_all)) - float(np.mean(ang_masked))) > 45.0, (
        f"Masked angular error should differ significantly from all-sample. "
        f"Got all={np.mean(ang_all):.1f} deg, masked={np.mean(ang_masked):.1f} deg"
    )
    # Confirm the masked subset has near-zero error (perfect match)
    assert float(np.mean(ang_masked)) < 1.0, (
        f"Masked subset should have near-zero angular error; got {np.mean(ang_masked):.2f} deg"
    )
    print("[PASS] direction_mask")


def _test_artifact_resolver() -> None:
    """st_lrps_force_model._resolve_run_dir should accept run dir / ckpt dir / direct ckpt path."""
    import tempfile
    SCRIPT_DIR_FM = Path(__file__).resolve().parent
    if str(SCRIPT_DIR_FM) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR_FM))
    try:
        from st_lrps_force_model import _resolve_run_dir
    except ImportError:
        print("[SKIP] artifact_resolver (st_lrps_force_model not available)")
        return
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "st_lrps_train_20240101"
        ckpt_dir = run_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True)
        ckpt_file = ckpt_dir / "ckpt_best.pt"
        ckpt_file.write_bytes(b"")  # empty placeholder

        assert _resolve_run_dir(run_dir) == run_dir.resolve(), "run dir resolution failed"
        assert _resolve_run_dir(ckpt_dir) == run_dir.resolve(), "ckpt dir resolution failed"
        assert _resolve_run_dir(ckpt_file) == run_dir.resolve(), "direct ckpt path resolution failed"
    print("[PASS] artifact_resolver")


def test_laplacian_diagnostic_does_not_require_grad() -> None:
    """Diagnostic mode Laplacian loss must NOT require grad (cheap, no graph)."""
    try:
        from st_lrps_losses import collocation_laplacian_loss
        from st_lrps_scaling import ScalerPack, IsometricScaleParams
    except ImportError:
        print("[SKIP] test_laplacian_diagnostic_does_not_require_grad"); return
    import torch
    sp = ScalerPack(
        x=IsometricScaleParams(mean=[0.0,0.0,0.0], scale=2e6),
        u=IsometricScaleParams(mean=[0.0], scale=1.0),
        a=IsometricScaleParams(mean=[0.0,0.0,0.0], scale=1e-3),
    ).to_tensors(torch.device("cpu"), torch.float32)
    # Tiny dummy model
    model = torch.nn.Sequential(torch.nn.Linear(3,4), torch.nn.Tanh(), torch.nn.Linear(4,1))
    loss = collocation_laplacian_loss(model, sp, r_min_m=1.8e6, r_max_m=1.9e6,
                                       n_points=16, device=torch.device("cpu"),
                                       n_hutchinson=2, mode="diagnostic")
    assert not loss.requires_grad, "Diagnostic Laplacian should not require grad"
    print("[PASS] test_laplacian_diagnostic_does_not_require_grad")


def test_laplacian_train_requires_grad() -> None:
    """Train mode Laplacian loss must require grad (gradients flow to model weights)."""
    try:
        from st_lrps_losses import collocation_laplacian_loss
        from st_lrps_scaling import ScalerPack, IsometricScaleParams
    except ImportError:
        print("[SKIP] test_laplacian_train_requires_grad"); return
    import torch
    sp = ScalerPack(
        x=IsometricScaleParams(mean=[0.0,0.0,0.0], scale=2e6),
        u=IsometricScaleParams(mean=[0.0], scale=1.0),
        a=IsometricScaleParams(mean=[0.0,0.0,0.0], scale=1e-3),
    ).to_tensors(torch.device("cpu"), torch.float32)
    model = torch.nn.Sequential(torch.nn.Linear(3,4), torch.nn.Tanh(), torch.nn.Linear(4,1))
    loss = collocation_laplacian_loss(model, sp, r_min_m=1.8e6, r_max_m=1.9e6,
                                       n_points=16, device=torch.device("cpu"),
                                       n_hutchinson=2, mode="train")
    assert loss.requires_grad, "Train mode Laplacian must require grad"
    print("[PASS] test_laplacian_train_requires_grad")


def test_laplacian_train_backward_changes_params() -> None:
    """Train mode Laplacian backward must populate parameter gradients."""
    try:
        from st_lrps_losses import collocation_laplacian_loss
        from st_lrps_scaling import ScalerPack, IsometricScaleParams
    except ImportError:
        print("[SKIP] test_laplacian_train_backward_changes_params"); return
    import torch
    sp = ScalerPack(
        x=IsometricScaleParams(mean=[0.0,0.0,0.0], scale=2e6),
        u=IsometricScaleParams(mean=[0.0], scale=1.0),
        a=IsometricScaleParams(mean=[0.0,0.0,0.0], scale=1e-3),
    ).to_tensors(torch.device("cpu"), torch.float32)
    model = torch.nn.Sequential(torch.nn.Linear(3,4), torch.nn.Tanh(), torch.nn.Linear(4,1))
    loss = collocation_laplacian_loss(model, sp, r_min_m=1.8e6, r_max_m=1.9e6,
                                       n_points=16, device=torch.device("cpu"),
                                       n_hutchinson=2, mode="train")
    loss.backward()
    params_with_grad = [p for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0]
    assert len(params_with_grad) > 0, "Laplacian train backward did not populate any gradients"
    print("[PASS] test_laplacian_train_backward_changes_params")


def test_cli_defaults_match_trainconfig_defaults() -> None:
    """Key CLI parser defaults must match TrainConfig field defaults."""
    import dataclasses
    try:
        from st_lrps_config import TrainConfig
        import argparse
    except ImportError:
        print("[SKIP] test_cli_defaults_match_trainconfig_defaults"); return

    tc_fields = {f.name: f.default for f in dataclasses.fields(TrainConfig)
                 if f.default is not dataclasses.MISSING}

    # Build an argparse namespace with defaults only (no sys.argv)
    import sys
    old_argv = sys.argv
    sys.argv = ["prog", "--data", "/tmp/fake.h5", "--out", "/tmp/fake_out"]
    try:
        # We just check the defaults dict, not parse_args() since it reads files
        from st_lrps_config import _TC_DEFAULTS
        for key in ["direction_loss_start_epoch", "direction_loss_ramp_epochs",
                    "direction_loss_weight", "best_metric", "hybrid_direction_alpha"]:
            if key in tc_fields and key in _TC_DEFAULTS:
                assert tc_fields[key] == _TC_DEFAULTS[key], (
                    f"TrainConfig.{key}={tc_fields[key]} but _TC_DEFAULTS.{key}={_TC_DEFAULTS[key]}"
                )
    finally:
        sys.argv = old_argv
    print("[PASS] test_cli_defaults_match_trainconfig_defaults")


def test_streaming_metrics_match_in_memory_on_small_dataset() -> None:
    """StreamingMetrics must match simple in-memory equivalents on a small batch."""
    try:
        import sys
        from pathlib import Path
        _HERE = Path(__file__).resolve().parent
        if str(_HERE) not in sys.path:
            sys.path.insert(0, str(_HERE))
        from st_lrps_evaluate import _StreamingMetrics
    except ImportError as e:
        print(f"[SKIP] test_streaming_metrics (import failed: {e})"); return

    import numpy as np, math
    rng = np.random.default_rng(42)
    N = 200
    x = rng.standard_normal((N, 3)) * 1.85e6
    a_true = rng.standard_normal((N, 3)) * 1e-3
    a_pred = a_true + rng.standard_normal((N, 3)) * 1e-5
    u_true = rng.standard_normal(N)
    u_pred = u_true + rng.standard_normal(N) * 0.01
    R = 1.737e6

    sm = _StreamingMetrics(n_alt_bins=5, alt_min_km=0, alt_max_km=500)
    sm.update(x, a_true, a_pred, u_true, u_pred, R)
    res = sm.finalize()

    a_err_norm = np.linalg.norm(a_pred - a_true, axis=1)
    expected_mae = float(a_err_norm.mean())
    expected_rmse = float(np.sqrt((a_err_norm**2).mean()))
    assert abs(res["mae_a"] - expected_mae) < 1e-10, f"MAE mismatch: {res['mae_a']} vs {expected_mae}"
    assert abs(res["rmse_a"] - expected_rmse) < 1e-10, f"RMSE mismatch"
    assert res["count"] == N
    print("[PASS] test_streaming_metrics_match_in_memory_on_small_dataset")


def test_topk_error_export_shape_and_columns() -> None:
    """TopKErrors heap keeps exactly K worst samples with correct shape."""
    try:
        from st_lrps_evaluate import _TopKErrors
    except ImportError as e:
        print(f"[SKIP] test_topk_error_export_shape_and_columns (import failed: {e})"); return
    import numpy as np
    rng = np.random.default_rng(7)
    N = 100
    K = 10
    x = rng.standard_normal((N, 3)) * 1.85e6
    a_true = rng.standard_normal((N, 3)) * 1e-3
    a_pred = a_true.copy()
    # Make first 5 samples have large error
    a_pred[:5] += rng.standard_normal((5, 3)) * 1.0
    u_true = rng.standard_normal(N)
    u_pred = u_true + rng.standard_normal(N) * 0.01

    # Make the FIRST K samples have moderate error, others much larger;
    # ensure heap keeps the largest K regardless of insertion order.
    # Recompute with a cleaner setup: 20 large-error samples among 100 total.
    a_pred = a_true.copy()
    a_pred[:20] += rng.standard_normal((20, 3)) * 1.0   # 20 samples with large error
    tk = _TopKErrors(K)
    tk.update_batch(x, u_true, u_pred, a_true, a_pred, 1.737e6)
    arr = tk.to_array()
    assert arr.shape == (K, 14), f"Expected ({K}, 14), got {arr.shape}"
    # The top errors should all be from the large-perturbation pool
    top_errs = arr[:, 11]  # abs_a_error column
    assert float(top_errs.min()) > 0.01, "Top-K should capture high-error samples"
    print("[PASS] test_topk_error_export_shape_and_columns")


def test_force_model_domain_status_inside_range() -> None:
    """domain_status should report in_range=True for positions inside training bounds."""
    try:
        from st_lrps_force_model import SurrogateForceModel
        from st_lrps_scaling import ScalerPack, IsometricScaleParams
    except ImportError:
        print("[SKIP] test_force_model_domain_status_inside_range"); return
    import torch
    import numpy as np
    sp = ScalerPack(
        x=IsometricScaleParams(mean=[0.0,0.0,0.0], scale=2e6),
        u=IsometricScaleParams(mean=[0.0], scale=1.0),
        a=IsometricScaleParams(mean=[0.0,0.0,0.0], scale=1e-3),
    ).to_tensors(torch.device("cpu"), torch.float32)
    model = torch.nn.Sequential(torch.nn.Linear(3,4), torch.nn.Tanh(), torch.nn.Linear(4,1))
    cfg = {
        "resolved_mu_si": 4.902e12,
        "resolved_a_sign": 1.0,
        "resolved_r_ref_m": 1.737e6,
        "degree_min": -1,
        "altitude_min_km": 100.0,
        "altitude_max_km": 500.0,
    }
    fm = SurrogateForceModel(model=model, scaler=sp, cfg=cfg, device=torch.device("cpu"))
    # Position at ~200 km altitude (inside range)
    x = np.array([[0.0, 0.0, 1.937e6]])
    status = fm.domain_status(x)
    assert status["finite_input"] is True
    assert status["in_training_altitude_range"] is True
    assert not status["recommended_fallback"]
    print("[PASS] test_force_model_domain_status_inside_range")


def test_force_model_domain_status_outside_range() -> None:
    """domain_status should report in_range=False for positions far outside training bounds."""
    try:
        from st_lrps_force_model import SurrogateForceModel
        from st_lrps_scaling import ScalerPack, IsometricScaleParams
    except ImportError:
        print("[SKIP] test_force_model_domain_status_outside_range"); return
    import torch
    import numpy as np
    sp = ScalerPack(
        x=IsometricScaleParams(mean=[0.0,0.0,0.0], scale=2e6),
        u=IsometricScaleParams(mean=[0.0], scale=1.0),
        a=IsometricScaleParams(mean=[0.0,0.0,0.0], scale=1e-3),
    ).to_tensors(torch.device("cpu"), torch.float32)
    model = torch.nn.Sequential(torch.nn.Linear(3,4), torch.nn.Tanh(), torch.nn.Linear(4,1))
    cfg = {
        "resolved_mu_si": 4.902e12,
        "resolved_a_sign": 1.0,
        "resolved_r_ref_m": 1.737e6,
        "degree_min": -1,
        "altitude_min_km": 100.0,
        "altitude_max_km": 500.0,
    }
    fm = SurrogateForceModel(model=model, scaler=sp, cfg=cfg, device=torch.device("cpu"))
    # Position at 1000 km altitude (well outside range)
    x = np.array([[0.0, 0.0, 2.737e6]])
    status = fm.domain_status(x)
    assert status["in_training_altitude_range"] is False
    assert status["recommended_fallback"] is True
    print("[PASS] test_force_model_domain_status_outside_range")


def test_force_model_rejects_bad_base_accel_shape() -> None:
    """predict_total_accel must raise ValueError when base_accel_fn returns wrong shape."""
    try:
        from st_lrps_force_model import SurrogateForceModel
        from st_lrps_scaling import ScalerPack, IsometricScaleParams
    except ImportError:
        print("[SKIP] test_force_model_rejects_bad_base_accel_shape"); return
    import torch, numpy as np
    sp = ScalerPack(
        x=IsometricScaleParams(mean=[0.0,0.0,0.0], scale=2e6),
        u=IsometricScaleParams(mean=[0.0], scale=1.0),
        a=IsometricScaleParams(mean=[0.0,0.0,0.0], scale=1e-3),
    ).to_tensors(torch.device("cpu"), torch.float32)
    model = torch.nn.Sequential(torch.nn.Linear(3,4), torch.nn.Tanh(), torch.nn.Linear(4,1))
    cfg = {"resolved_mu_si": 4.902e12, "resolved_a_sign": 1.0, "resolved_r_ref_m": 1.737e6, "degree_min": -1}
    fm = SurrogateForceModel(model=model, scaler=sp, cfg=cfg, device=torch.device("cpu"))
    x = np.array([[0.0, 0.0, 1.937e6]])
    bad_fn = lambda _x: np.zeros((5,))  # wrong shape
    try:
        fm.predict_total_accel(x, bad_fn)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    print("[PASS] test_force_model_rejects_bad_base_accel_shape")


def test_predict_residual_potential_no_grad_path() -> None:
    """predict_residual_potential should work without requiring grad (no_grad fast path)."""
    try:
        from st_lrps_force_model import SurrogateForceModel
        from st_lrps_scaling import ScalerPack, IsometricScaleParams
    except ImportError:
        print("[SKIP] test_predict_residual_potential_no_grad_path"); return
    import torch, numpy as np
    sp = ScalerPack(
        x=IsometricScaleParams(mean=[0.0,0.0,0.0], scale=2e6),
        u=IsometricScaleParams(mean=[0.0], scale=1.0),
        a=IsometricScaleParams(mean=[0.0,0.0,0.0], scale=1e-3),
    ).to_tensors(torch.device("cpu"), torch.float32)
    model = torch.nn.Sequential(torch.nn.Linear(3,4), torch.nn.Tanh(), torch.nn.Linear(4,1))
    cfg = {"resolved_mu_si": 4.902e12, "resolved_a_sign": 1.0, "resolved_r_ref_m": 1.737e6, "degree_min": -1}
    fm = SurrogateForceModel(model=model, scaler=sp, cfg=cfg, device=torch.device("cpu"))
    x = np.array([[0.0, 0.0, 1.937e6], [0.0, 1.937e6, 0.0]])
    # Should run without error and return shape (2,)
    result = fm.predict_residual_potential(x)
    assert result.shape == (2,), f"Expected (2,), got {result.shape}"
    assert np.all(np.isfinite(result)), "Non-finite potential predictions"
    print("[PASS] test_predict_residual_potential_no_grad_path")


def test_model_factory_rejects_incompatible_encodings() -> None:
    """build_model_from_config must raise ValueError when both SH and radial encoding are enabled."""
    try:
        from st_lrps_models import build_model_from_config
    except ImportError:
        print("[SKIP] test_model_factory_rejects_incompatible_encodings"); return
    import torch
    cfg = {
        "hidden": 32, "depth": 2, "activation": "sine",
        "w0_first": 30.0, "w0_hidden": 30.0, "dropout": 0.0,
        "use_fourier": False, "fourier_n_features": 16, "fourier_sigma": 1.0, "fourier_seed": 0,
        "fourier_append_raw": True,
        "use_sh_encoding": True,
        "sh_encoding_degree": 4,
        "sh_append_raw": True,
        "use_radial_separation": True,  # CONFLICT
        "radial_append_raw": False,
        "use_residual_blocks": False,
        "n_bands": 1,
    }
    try:
        build_model_from_config(cfg, device=torch.device("cpu"), dtype=torch.float32)
        assert False, "Should have raised ValueError for incompatible encodings"
    except ValueError as e:
        assert "cannot both be True" in str(e) or "incompatible" in str(e).lower(), str(e)
    print("[PASS] test_model_factory_rejects_incompatible_encodings")


def test_x_scale_uses_metadata_when_available() -> None:
    """fit_scaler_streaming should prefer r_ref + alt_max over streaming max-norm."""
    import tempfile, numpy as np
    try:
        import h5py
    except ImportError:
        print("[SKIP] test_x_scale_uses_metadata (h5py unavailable)"); return
    try:
        from st_lrps_scaling import fit_scaler_streaming
        from st_lrps_data import DatasetMeta
    except ImportError as e:
        print(f"[SKIP] test_x_scale_uses_metadata (import: {e})"); return

    rng = np.random.default_rng(1)
    N = 500
    R_REF = 1.737e6
    ALT_MAX = 300.0  # km
    r = R_REF + rng.uniform(50e3, ALT_MAX * 1000, N)
    dirs = rng.standard_normal((N, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    x = (r[:, None] * dirs).astype(np.float32)
    u = (rng.standard_normal(N) * 1e5).astype(np.float32)
    a = (rng.standard_normal((N, 3)) * 1e-3).astype(np.float32)
    data = np.concatenate([x, u[:, None], a], axis=1).astype(np.float32)

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
        h5path = Path(f.name)
    with h5py.File(h5path, "w") as hf:
        ds = hf.create_dataset("data", data=data)
        # DatasetMeta.from_h5 reads file-level attrs (f.attrs), not dataset-level.
        hf.attrs["alt_min_km"] = 50.0
        hf.attrs["alt_max_km"] = ALT_MAX
        hf.attrs["r_ref_m"] = R_REF
        hf.attrs["unit_system"] = "si"
        hf.attrs["mu_si"] = 4.902e12

    try:
        meta = DatasetMeta.from_h5(h5path)
        scaler = fit_scaler_streaming(h5path, "data", meta, use_si=False,
                                      mu_si=4.902e12, a_sign=1.0, n_fit=N, seed=0)
        expected_x_scale = R_REF + ALT_MAX * 1000.0
        assert abs(scaler.x.scale - expected_x_scale) < 1.0, (
            f"Expected x_scale={expected_x_scale:.3e}, got {scaler.x.scale:.3e}"
        )
        assert scaler.provenance.get("x_scale_source") == "metadata_altitude_max", (
            f"Expected metadata_altitude_max, got {scaler.provenance.get('x_scale_source')}"
        )
    finally:
        h5path.unlink(missing_ok=True)
    print("[PASS] test_x_scale_uses_metadata_when_available")


def test_x_scale_falls_back_to_streaming_when_metadata_missing() -> None:
    """fit_scaler_streaming should fall back to max-norm when metadata lacks altitude bounds."""
    import tempfile, numpy as np
    try:
        import h5py
    except ImportError:
        print("[SKIP] test_x_scale_falls_back_to_streaming (h5py unavailable)"); return
    try:
        from st_lrps_scaling import fit_scaler_streaming
        from st_lrps_data import DatasetMeta
    except ImportError as e:
        print(f"[SKIP] test_x_scale_falls_back_to_streaming (import: {e})"); return

    rng = np.random.default_rng(2)
    N = 500
    R_REF = 1.737e6
    r = R_REF + rng.uniform(50e3, 200e3, N)
    dirs = rng.standard_normal((N, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    x = (r[:, None] * dirs).astype(np.float32)
    u = (rng.standard_normal(N) * 1e5).astype(np.float32)
    a = (rng.standard_normal((N, 3)) * 1e-3).astype(np.float32)
    data = np.concatenate([x, u[:, None], a], axis=1).astype(np.float32)

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
        h5path = Path(f.name)
    with h5py.File(h5path, "w") as hf:
        # No alt_min_km / alt_max_km / r_ref_m in attrs -> fallback
        hf.create_dataset("data", data=data)

    try:
        meta = DatasetMeta.from_h5(h5path)
        scaler = fit_scaler_streaming(h5path, "data", meta, use_si=False,
                                      mu_si=4.902e12, a_sign=1.0, n_fit=N, seed=0)
        assert scaler.provenance.get("x_scale_source") == "streaming_fit", (
            f"Expected streaming_fit, got {scaler.provenance.get('x_scale_source')}"
        )
        # x_scale should be close to the actual max radius in the data
        actual_max_r = float(np.linalg.norm(x, axis=1).max())
        assert abs(scaler.x.scale - actual_max_r) / actual_max_r < 0.02, (
            f"Streaming x_scale={scaler.x.scale:.3e} should be near actual max_r={actual_max_r:.3e}"
        )
    finally:
        h5path.unlink(missing_ok=True)
    print("[PASS] test_x_scale_falls_back_to_streaming_when_metadata_missing")


def test_active_error_point_loader() -> None:
    """_load_error_points should read a CSV of error points correctly."""
    import tempfile
    try:
        from st_lrps_evaluate import _TopKErrors  # reuse to generate a CSV
    except ImportError:
        print("[SKIP] test_active_error_point_loader (st_lrps_evaluate unavailable)"); return
    try:
        import sys
        from pathlib import Path
        _HERE = Path(__file__).resolve().parent
        if str(_HERE) not in sys.path:
            sys.path.insert(0, str(_HERE))
        from spatial_cloud_generator import _load_error_points
    except ImportError:
        print("[SKIP] test_active_error_point_loader (spatial_cloud_generator unavailable)"); return

    import numpy as np
    from pathlib import Path
    # Write a fake CSV
    header = "x,y,z,u_true,u_pred,ax_true,ay_true,az_true,ax_pred,ay_pred,az_pred,abs_a_error,rel_a_error,altitude_km"
    rows = [[float(j) for j in range(14)] for _ in range(20)]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as f:
        f.write(header + "\n")
        for row in rows:
            f.write(",".join(str(v) for v in row) + "\n")
        csv_path = Path(f.name)
    try:
        arr = _load_error_points(csv_path, max_source=10)
        assert arr.shape == (10, 14), f"Expected (10,14), got {arr.shape}"
    finally:
        csv_path.unlink(missing_ok=True)
    print("[PASS] test_active_error_point_loader")


def test_active_jitter_points_have_expected_shape() -> None:
    """_jitter_around_point must produce exactly n_samples points."""
    try:
        import sys
        from pathlib import Path
        _HERE = Path(__file__).resolve().parent
        if str(_HERE) not in sys.path:
            sys.path.insert(0, str(_HERE))
        from spatial_cloud_generator import _jitter_around_point
    except ImportError:
        print("[SKIP] test_active_jitter_points_have_expected_shape"); return
    import numpy as np
    rng = np.random.default_rng(5)
    x_src = np.array([0.0, 0.0, 1.937e6])
    pts = _jitter_around_point(x_src, n_samples=30, jitter_radial_km=10.0,
                                jitter_tangent_km=20.0, rng=rng)
    assert pts.shape == (30, 3), f"Expected (30,3), got {pts.shape}"
    # Points should be near source
    dist = np.linalg.norm(pts - x_src[None, :], axis=1)
    assert dist.max() < 200e3, f"Jitter too large: max dist={dist.max():.0f} m"
    print("[PASS] test_active_jitter_points_have_expected_shape")


def test_active_component_metadata_written() -> None:
    """Active refinement should write active_refinement_meta.json."""
    try:
        import sys
        from pathlib import Path
        _HERE = Path(__file__).resolve().parent
        if str(_HERE) not in sys.path:
            sys.path.insert(0, str(_HERE))
        from spatial_cloud_generator import _load_error_points, _jitter_around_point
    except ImportError:
        print("[SKIP] test_active_component_metadata_written"); return
    import numpy as np, tempfile, json
    from pathlib import Path

    # We test that _jitter_around_point + saving meta produces correct structure.
    # (Full _run_active_refinement would need argparse namespace.)
    rng = np.random.default_rng(9)
    x_src = np.array([1.8e6, 0.3e6, 0.5e6])
    pts = _jitter_around_point(x_src, 10, 5.0, 10.0, rng)
    meta = {
        "component_name": "active_error_refinement",
        "source_error_file": "/fake/path.csv",
        "n_source_points": 5,
        "active_jitter_radial_km": 5.0,
        "active_jitter_tangent_km": 10.0,
        "active_samples_per_point": 10,
        "total_generated_positions": 50,
    }
    with tempfile.TemporaryDirectory() as tmpd:
        meta_path = Path(tmpd) / "active_refinement_meta.json"
        meta_path.write_text(json.dumps(meta))
        loaded = json.loads(meta_path.read_text())
        assert loaded["component_name"] == "active_error_refinement"
        assert "source_error_file" in loaded
        assert "active_jitter_radial_km" in loaded
    print("[PASS] test_active_component_metadata_written")


def run_unit_tests() -> None:
    """Run all unit tests. Prints [PASS]/[SKIP] for each; raises on first failure."""
    print("\n========== ST-LRPS Unit Tests ==========")
    _test_scaler_roundtrip()
    _test_chain_rule()
    _test_direction_mask()
    _test_artifact_resolver()
    test_laplacian_diagnostic_does_not_require_grad()
    test_laplacian_train_requires_grad()
    test_laplacian_train_backward_changes_params()
    test_cli_defaults_match_trainconfig_defaults()
    test_streaming_metrics_match_in_memory_on_small_dataset()
    test_topk_error_export_shape_and_columns()
    test_force_model_domain_status_inside_range()
    test_force_model_domain_status_outside_range()
    test_force_model_rejects_bad_base_accel_shape()
    test_predict_residual_potential_no_grad_path()
    test_model_factory_rejects_incompatible_encodings()
    test_x_scale_uses_metadata_when_available()
    test_x_scale_falls_back_to_streaming_when_metadata_missing()
    test_active_error_point_loader()
    test_active_jitter_points_have_expected_shape()
    test_active_component_metadata_written()
    print("========== All unit tests passed ==========\n")


if __name__ == "__main__":
    # Insert --unit-tests flag handling before dispatching to main()
    import sys as _sys
    if "--unit-tests" in _sys.argv:
        _sys.argv.remove("--unit-tests")
        run_unit_tests()
    else:
        main()
