#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
spatial_cloud_analysis.py

Analyze a generated spatial point-cloud dataset:
    columns = [x, y, z, U, ax, ay, az]

Supports:
- HDF5 produced by spatial_cloud_generator.py (dataset name: "data")
- PyTorch .pt produced by spatial_cloud_generator.py (dict with "data" tensor)

Outputs:
- Console summary (what points were analyzed + key statistics)
- Optional plots (PNG) into an output folder
- Optional JSON summary (machine-readable)

Why this exists
---------------
Generator produces *big* datasets (millions of rows). This script is designed to:
- sample efficiently without loading everything (especially for .h5)
- show which subset / region is actually analyzed (reproducible)
- provide quick sanity checks: uniformity, altitude band, U and |a| ranges, etc.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _human_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    x = float(n)
    for u in units:
        if x < 1024.0:
            return f"{x:.2f} {u}"
        x /= 1024.0
    return f"{x:.2f} PB"


def _safe_float(d: Dict[str, object], key: str, default: float) -> float:
    try:
        v = d.get(key, default)
        return float(v)  # type: ignore[arg-type]
    except Exception:
        return float(default)


def _np_stats(x: np.ndarray) -> Dict[str, float]:
    x = np.asarray(x)
    return {
        "min": float(np.nanmin(x)),
        "p01": float(np.nanpercentile(x, 1.0)),
        "p05": float(np.nanpercentile(x, 5.0)),
        "p50": float(np.nanpercentile(x, 50.0)),
        "p95": float(np.nanpercentile(x, 95.0)),
        "p99": float(np.nanpercentile(x, 99.0)),
        "max": float(np.nanmax(x)),
        "mean": float(np.nanmean(x)),
        "std": float(np.nanstd(x)),
    }


def _rms(x: np.ndarray) -> float:
    """Root-mean-square with NaN-safe handling for compact field summaries."""
    x = np.asarray(x, dtype=np.float64)
    return float(np.sqrt(np.nanmean(x * x)))


def _safe_div(num: float, den: float, default: float = 0.0) -> float:
    if not np.isfinite(num) or not np.isfinite(den) or abs(den) <= 1e-30:
        return float(default)
    return float(num / den)


def _entropy_score(counts: np.ndarray) -> float:
    """Return normalized Shannon entropy in [0, 1] for occupancy balance."""
    counts = np.asarray(counts, dtype=np.float64)
    total = float(np.sum(counts))
    if total <= 0.0 or counts.size <= 1:
        return 0.0
    p = counts[counts > 0.0] / total
    h = -float(np.sum(p * np.log(p)))
    return float(h / math.log(float(counts.size)))


def _bin_balance(values: np.ndarray, lo: float, hi: float, n_bins: int = 10) -> Dict[str, object]:
    """Occupancy metrics for a scalar coordinate such as altitude."""
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.nanmin(values))
        hi = float(np.nanmax(values))
    if hi <= lo:
        hi = lo + 1.0
    counts, edges = np.histogram(values, bins=int(max(1, n_bins)), range=(lo, hi))
    mean_count = float(np.mean(counts)) if counts.size else 0.0
    cv = _safe_div(float(np.std(counts)), mean_count, default=0.0)
    return {
        "range": [float(lo), float(hi)],
        "bin_edges": [float(v) for v in edges],
        "counts": [int(v) for v in counts],
        "min_count": int(np.min(counts)) if counts.size else 0,
        "max_count": int(np.max(counts)) if counts.size else 0,
        "empty_bins": int(np.count_nonzero(counts == 0)),
        "coefficient_of_variation": float(cv),
        "entropy_score": _entropy_score(counts),
    }


def _optional_float_attr(raw: Dict[str, object], *keys: str) -> Optional[float]:
    for key in keys:
        if key in raw:
            try:
                return float(raw[key])  # type: ignore[arg-type]
            except Exception:
                continue
    return None


def _optional_str_attr(raw: Dict[str, object], *keys: str) -> Optional[str]:
    for key in keys:
        if key in raw:
            try:
                v = raw[key]
                if isinstance(v, bytes):
                    v = v.decode("utf-8")
                s = str(v).strip()
                if s:
                    return s
            except Exception:
                continue
    return None


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CloudMeta:
    n_total: int
    columns: Tuple[str, ...]
    unit_system: str  # "si" | "canonical" | unknown
    mu_si: Optional[float]
    r_ref_m: Optional[float]
    DU_m: Optional[float]
    TU_s: Optional[float]
    VU_m_s: Optional[float]
    raw: Dict[str, object]
    # Optional metadata fields from modern generator output
    degree_min: Optional[int] = None
    degree_max: Optional[int] = None
    target_mode: Optional[str] = None
    central_body: Optional[str] = None
    a_sign_convention: Optional[str] = None


def _load_h5_meta(path: Path) -> Tuple["h5py.File", "h5py.Dataset", CloudMeta]:
    import h5py  # type: ignore

    f = h5py.File(str(path), "r")
    dset = f["data"]
    attrs = {str(k): f.attrs[k] for k in f.attrs.keys()}

    cols = ("x", "y", "z", "U", "ax", "ay", "az")
    if "columns" in attrs:
        try:
            # stored as a string like "[x,y,z,U,ax,ay,az]"
            s = str(attrs["columns"])
            s = s.strip().strip("[]")
            cols = tuple([c.strip() for c in s.split(",") if c.strip()]) or cols
        except Exception:
            pass

    unit_system = str(attrs.get("unit_system", "unknown")).lower()

    def _parse_int_attr(key: str) -> Optional[int]:
        val = attrs.get(key)
        if val is None:
            return None
        try:
            return int(val)
        except (ValueError, TypeError):
            return None

    def _parse_str_attr(key: str) -> Optional[str]:
        val = attrs.get(key)
        if val is None:
            return None
        try:
            s = val.decode("utf-8") if isinstance(val, bytes) else str(val)
            return s.strip() or None
        except Exception:
            return None

    meta = CloudMeta(
        n_total=int(dset.shape[0]),
        columns=tuple(cols),
        unit_system=unit_system,
        mu_si=_safe_float(attrs, "mu_si", math.nan) if "mu_si" in attrs else None,
        r_ref_m=_safe_float(attrs, "r_ref_m", math.nan) if "r_ref_m" in attrs else None,
        DU_m=_safe_float(attrs, "DU_m", math.nan) if "DU_m" in attrs else None,
        TU_s=_safe_float(attrs, "TU_s", math.nan) if "TU_s" in attrs else None,
        VU_m_s=_safe_float(attrs, "VU_m_s", math.nan) if "VU_m_s" in attrs else None,
        raw=attrs,
        degree_min=_parse_int_attr("degree_min"),
        degree_max=_parse_int_attr("degree_max"),
        target_mode=_parse_str_attr("target_mode"),
        central_body=_parse_str_attr("central_body"),
        a_sign_convention=_parse_str_attr("a_sign_convention"),
    )
    return f, dset, meta


def _load_pt(path: Path) -> Tuple[np.ndarray, CloudMeta]:
    import torch  # type: ignore

    obj = torch.load(str(path), map_location="cpu")
    if not isinstance(obj, dict) or "data" not in obj:
        raise ValueError("Unsupported .pt format. Expected dict with key 'data'.")
    data = obj["data"]
    if hasattr(data, "detach"):
        data = data.detach().cpu()
    arr = np.asarray(data)

    raw_meta = obj.get("meta", {}) if isinstance(obj, dict) else {}
    if not isinstance(raw_meta, dict):
        raw_meta = {}

    cols = tuple(obj.get("columns", ["x", "y", "z", "U", "ax", "ay", "az"]))
    if len(cols) != 7:
        cols = ("x", "y", "z", "U", "ax", "ay", "az")

    unit_system = str(raw_meta.get("unit_system", "unknown")).lower()

    def _safe_int_meta(key: str) -> Optional[int]:
        val = raw_meta.get(key)
        if val is None:
            return None
        try:
            return int(val)
        except (ValueError, TypeError):
            return None

    def _safe_str_meta(key: str) -> Optional[str]:
        val = raw_meta.get(key)
        if val is None:
            return None
        try:
            return str(val).strip() or None
        except Exception:
            return None

    meta = CloudMeta(
        n_total=int(arr.shape[0]),
        columns=cols,
        unit_system=unit_system,
        mu_si=float(raw_meta["mu_si"]) if "mu_si" in raw_meta else None,
        r_ref_m=float(raw_meta["r_ref_m"]) if "r_ref_m" in raw_meta else None,
        DU_m=float(raw_meta["DU_m"]) if "DU_m" in raw_meta else None,
        TU_s=float(raw_meta["TU_s"]) if "TU_s" in raw_meta else None,
        VU_m_s=float(raw_meta["VU_m_s"]) if "VU_m_s" in raw_meta else None,
        raw=raw_meta,
        degree_min=_safe_int_meta("degree_min"),
        degree_max=_safe_int_meta("degree_max"),
        target_mode=_safe_str_meta("target_mode"),
        central_body=_safe_str_meta("central_body"),
        a_sign_convention=_safe_str_meta("a_sign_convention"),
    )
    return arr, meta


# ---------------------------------------------------------------------------
# Sampling strategy
# ---------------------------------------------------------------------------
def _sample_rows_h5_contiguous(
    dset,  # h5py.Dataset
    n_sample: int,
    seed: int,
    block_size: int = 50_000,
) -> np.ndarray:
    """
    Efficient sampling for huge HDF5 arrays:
    - read several random contiguous blocks
    - sample rows from each block
    """
    n_total = int(dset.shape[0])
    n_sample = int(min(n_sample, n_total))
    rng = np.random.default_rng(int(seed))

    if n_sample <= 0:
        return np.empty((0, int(dset.shape[1])), dtype=np.float64)

    # Choose number of blocks so each block contributes ~uniformly
    n_blocks = max(1, min(50, (n_sample + 9999) // 10_000))
    block_size = int(min(block_size, max(10_000, n_total // max(1, n_blocks))))
    starts = rng.integers(0, max(1, n_total - block_size), size=n_blocks, endpoint=False)

    out = []
    remaining = n_sample
    for i, s in enumerate(starts):
        take = int(min(remaining, max(1, n_sample // n_blocks)))
        if i == (n_blocks - 1):
            take = remaining
        block = np.asarray(dset[int(s) : int(s) + block_size, :])
        idx = rng.choice(block.shape[0], size=take, replace=False if take <= block.shape[0] else True)
        out.append(block[idx, :])
        remaining -= take
        if remaining <= 0:
            break

    return np.concatenate(out, axis=0)


def _apply_region_filter(
    X: np.ndarray,
    *,
    r_ref_m: Optional[float],
    DU_m: Optional[float],
    unit_system: str,
    alt_min_km: Optional[float],
    alt_max_km: Optional[float],
) -> Tuple[np.ndarray, Dict[str, object]]:
    """
    Filter points by altitude band (km) if requested.
    Works for SI and canonical datasets (needs DU_m).
    """
    info: Dict[str, object] = {"filter": "none"}
    if alt_min_km is None and alt_max_km is None:
        return X, info

    xyz = X[:, 0:3]
    r = np.linalg.norm(xyz, axis=1)

    if unit_system == "canonical":
        if DU_m is None or r_ref_m is None or not np.isfinite(DU_m) or not np.isfinite(r_ref_m):
            raise ValueError("Canonical dataset needs DU_m and r_ref_m in metadata to filter by altitude.")
        r_si = r * float(DU_m)
        alt_km = (r_si - float(r_ref_m)) / 1000.0
    else:
        # assume SI
        if r_ref_m is None or not np.isfinite(r_ref_m):
            raise ValueError(
                "SI dataset needs r_ref_m in metadata to filter by altitude. "
                "Refusing to invent a reference radius from the sample cloud."
            )
        alt_km = (r - float(r_ref_m)) / 1000.0

    lo = -np.inf if alt_min_km is None else float(alt_min_km)
    hi = np.inf if alt_max_km is None else float(alt_max_km)

    m = (alt_km >= lo) & (alt_km <= hi)
    info = {"filter": "altitude_km", "alt_min_km": lo, "alt_max_km": hi, "kept": int(np.count_nonzero(m)), "total": int(len(m))}
    return X[m, :], info


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _make_plots(
    X: np.ndarray,
    meta: CloudMeta,
    outdir: Path,
    scatter_n: int,
) -> Dict[str, str]:
    outdir.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")  # headless
    import matplotlib.pyplot as plt  # type: ignore
    
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except Exception:
        pass
    plt.rcParams.update({
        "font.family": "sans-serif",
        "axes.labelsize": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "figure.autolayout": True
    })

    paths: Dict[str, str] = {}

    xyz = X[:, 0:3]
    r = np.linalg.norm(xyz, axis=1)

    alt_km = None
    if meta.r_ref_m is not None and np.isfinite(meta.r_ref_m):
        if meta.unit_system == "canonical" and meta.DU_m is not None and np.isfinite(meta.DU_m):
            r_si = r * float(meta.DU_m)
            alt_km = (r_si - float(meta.r_ref_m)) / 1000.0
        elif meta.unit_system != "canonical":
            alt_km = (r - float(meta.r_ref_m)) / 1000.0

    U = X[:, 3]
    a = X[:, 4:7]
    a_norm = np.linalg.norm(a, axis=1)

    if scatter_n > 0:
        rng = np.random.default_rng(0)
        n = min(int(scatter_n), X.shape[0])
        idx = rng.choice(X.shape[0], size=n, replace=False)
        xs, ys, zs = xyz[idx, 0], xyz[idx, 1], xyz[idx, 2]
        cval = alt_km[idx] if alt_km is not None else r[idx]

        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection="3d")
        ax.set_facecolor('white')
        p = ax.scatter(xs, ys, zs, s=2, alpha=0.6, c=cval, cmap="viridis", edgecolor='none')
        cbar = fig.colorbar(p, ax=ax, shrink=0.6, pad=0.08)
        cbar.set_label("Altitude [km]" if alt_km is not None else "Radius |r|", weight='bold')
        ax.set_title("Spatial Distribution of Sampled Points", pad=15)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_zlabel("z [m]")
        fn = outdir / "scatter_3d.png"
        fig.savefig(fn, dpi=300, bbox_inches='tight')
        plt.close(fig)
        paths["scatter_3d"] = str(fn)

    def _hist(data: np.ndarray, title: str, xlabel: str, fname: str, color: str) -> None:
        fig = plt.figure(figsize=(8, 5))
        counts, bins, patches = plt.hist(data, bins=80, color=color, edgecolor="white", alpha=0.85)
        mean_val = np.nanmean(data)
        median_val = np.nanmedian(data)
        plt.axvline(mean_val, color='#C44E52', linestyle='--', linewidth=1.5, label=f"Mean: {mean_val:.2e}")
        plt.axvline(median_val, color='#2C3E50', linestyle=':', linewidth=1.5, label=f"Median: {median_val:.2e}")
        plt.title(title, pad=12)
        plt.xlabel(xlabel, labelpad=8)
        plt.ylabel("Frequency", labelpad=8)
        plt.legend(frameon=True, fancybox=True, shadow=True)
        fn = outdir / fname
        fig.savefig(fn, dpi=300, bbox_inches='tight')
        plt.close(fig)
        paths[fname.replace(".png", "")] = str(fn)

    _is_residual = (meta.target_mode == "residual") or (
        getattr(meta, "degree_min", None) is not None and getattr(meta, "degree_min", -1) >= 0
    )
    _pot_title = "Residual Potential (dU) Distribution" if _is_residual else "Gravitational Potential (U) Distribution"
    _pot_xlabel = "dU (residual potential)" if _is_residual else "U (potential)"
    _acc_title = "Residual Acceleration (|da|) Distribution" if _is_residual else "Acceleration Magnitude (|a|) Distribution"
    _acc_xlabel = "|da| (residual accel)" if _is_residual else "|a| (acceleration magnitude)"
    _hist(U, _pot_title, _pot_xlabel, "hist_U.png", "#4C72B0")
    _hist(a_norm, _acc_title, _acc_xlabel, "hist_a_norm.png", "#55A868")
    if alt_km is not None:
        _hist(alt_km, "Altitude Distribution", "Altitude [km]", "hist_alt_km.png", "#8172B3")
    else:
        _hist(r, "Radius Distribution", "Radius |r|", "hist_r.png", "#8172B3")

    def _proj(ix: int, iy: int, title: str, fname: str) -> None:
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111)
        ax.scatter(xyz[:, ix], xyz[:, iy], s=1, alpha=0.15, color="#2C3E50")
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_title(title, pad=12)
        axes_labels = ["x [m]", "y [m]", "z [m]"]
        ax.set_xlabel(axes_labels[ix])
        ax.set_ylabel(axes_labels[iy])
        ax.grid(True, linestyle="--", alpha=0.6)
        fn = outdir / fname
        fig.savefig(fn, dpi=300, bbox_inches='tight')
        plt.close(fig)
        paths[fname.replace(".png", "")] = str(fn)

    _proj(0, 1, "Equatorial Projection (x-y)", "proj_xy.png")
    _proj(0, 2, "Polar Projection (x-z)", "proj_xz.png")
    _proj(1, 2, "Polar Projection (y-z)", "proj_yz.png")

    return paths


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze a spatial SH point-cloud (.h5 or .pt).")
    p.add_argument("input", type=str, nargs="?", default=None,
                   help="Path to .h5/.hdf5 or .pt dataset. If omitted, the script will try to auto-detect the newest dataset in common output folders (runs/, outputs/, data/, datasets/, out/).")
    p.add_argument("--sample", type=int, default=200_000, help="How many rows to analyze (sampled).")
    p.add_argument("--seed", type=int, default=123, help="Sampling seed (reproducible).")

    p.add_argument("--alt-min-km", type=float, default=None, help="Optional: filter analyzed sample by altitude >= this.")
    p.add_argument("--alt-max-km", type=float, default=None, help="Optional: filter analyzed sample by altitude <= this.")

    p.add_argument("--no-plots", action="store_true", help="Do not create plots.")
    p.add_argument("--scatter-n", type=int, default=50_000, help="How many points to draw in the 3D scatter plot.")
    p.add_argument("--outdir", type=str, default=None, help="Output folder for plots / summary.json. If omitted, defaults to outputs/dataset_reports/<dataset>_<timestamp>.")
    p.add_argument("--dump-json", action="store_true", help="Write summary.json into outdir.")
    return p.parse_args()


def _try_import_dataset_parameters(script_dir: Path):
    """
    Best-effort import of the local lunar dataset_parameters module.
    Returns the imported module or None.
    """
    import sys
    here = str(script_dir)
    if here not in sys.path:
        sys.path.insert(0, here)
    try:
        import dataset_parameters  # type: ignore
        return dataset_parameters
    except Exception:
        return None


def _autodetect_input(script_dir: Path) -> Path | None:
    """
    If user doesn't pass an input path, try to find the newest dataset file.
    Search order:
      1) SPATIAL_CLOUD_INPUT env var
      2) dataset_parameters hints (if available)
      3) newest *.h5/*.hdf5/*.pt in common folders
    """
    import os

    env = os.environ.get("SPATIAL_CLOUD_INPUT")
    if env:
        p = Path(env).expanduser()
        if p.exists():
            return p.resolve()

    dp = _try_import_dataset_parameters(script_dir)
    if dp is not None:
        # Look for a few common places users store output paths.
        possible = []
        for name in ("DEFAULT_DATASET_CONFIG", "DATASET_CONFIG", "CONFIG"):
            cfg = getattr(dp, name, None)
            if isinstance(cfg, dict):
                for k in ("out_path", "output_path", "output_file", "cloud_path", "dataset_path", "last_output"):
                    v = cfg.get(k)
                    if isinstance(v, str):
                        possible.append(v)
        for name in ("DEFAULT_OUT_PATH", "OUTPUT_PATH", "CLOUD_PATH", "DATASET_PATH", "LAST_OUTPUT_PATH"):
            v = getattr(dp, name, None)
            if isinstance(v, str):
                possible.append(v)

        for s in possible:
            p = Path(s).expanduser()
            if not p.is_absolute():
                # interpret relative to repo root (one level above script_dir)
                p = (script_dir.parent / p).resolve()
            if p.exists():
                return p

    # Common folders (relative to cwd and script location)
    search_roots = []
    cwd = Path.cwd()
    search_roots += [
        cwd, cwd / "runs", cwd / "outputs", cwd / "data", cwd / "datasets", cwd / "out",
        script_dir, script_dir / "runs", script_dir / "outputs", script_dir / "data", script_dir / "datasets", script_dir / "out",
        script_dir.parent, script_dir.parent / "runs", script_dir.parent / "outputs", script_dir.parent / "data", script_dir.parent / "datasets", script_dir.parent / "out",
    ]

    exts = ("*.h5", "*.hdf5", "*.pt")
    found: list[Path] = []
    for root in search_roots:
        if root.exists() and root.is_dir():
            for pat in exts:
                found.extend(root.glob(pat))

    if not found:
        return None

    def _looks_lunar(p: Path) -> bool:
        dp = _try_import_dataset_parameters(script_dir)
        checker = getattr(dp, "is_lunar_body_signature", None) if dp is not None else None
        suffix = p.suffix.lower()
        try:
            if suffix in (".h5", ".hdf5"):
                import h5py  # type: ignore

                with h5py.File(str(p), "r") as f:
                    attrs = f.attrs
                    body = str(attrs.get("central_body", "") or "").strip().lower()
                    if body in {"moon", "lunar", "selene"}:
                        return True
                    if checker is not None:
                        mu_si = float(attrs["mu_si"]) if "mu_si" in attrs else None
                        r_ref_m = float(attrs["r_ref_m"]) if "r_ref_m" in attrs else None
                        return bool(checker(mu_si=mu_si, r_ref_m=r_ref_m))
            if suffix == ".pt":
                import torch  # type: ignore

                obj = torch.load(str(p), map_location="cpu")
                meta = obj.get("meta", {}) if isinstance(obj, dict) else {}
                if isinstance(meta, dict):
                    body = str(meta.get("central_body", "") or "").strip().lower()
                    if body in {"moon", "lunar", "selene"}:
                        return True
                    if checker is not None:
                        mu_si = float(meta["mu_si"]) if "mu_si" in meta else None
                        r_ref_m = float(meta["r_ref_m"]) if "r_ref_m" in meta else None
                        return bool(checker(mu_si=mu_si, r_ref_m=r_ref_m))
        except Exception:
            return False
        return False

    ranked = sorted(found, key=lambda p: (_looks_lunar(p), p.stat().st_mtime), reverse=True)
    for cand in ranked:
        if _looks_lunar(cand):
            return cand.resolve()
    return None


def main() -> None:
    args = parse_args()

    # Anchor input auto-detection at the st_lrps package root (one level up from
    # data/), preserving the pre-reorg search location.
    script_dir = Path(__file__).resolve().parents[1]
    if args.input is None:
        guess = _autodetect_input(script_dir)
        if guess is None:
            raise SystemExit(
                "No input dataset path provided and auto-detection failed.\n"
                "Provide an explicit path, e.g.:\n"
                "  python spatial_cloud_analysis.py runs/cloud.h5\n"
                "Or set SPATIAL_CLOUD_INPUT to the dataset path."
            )
        in_path = guess
        print(f"[info] Auto-detected input: {in_path}")
    else:
        in_path = Path(args.input).expanduser().resolve()
    if not in_path.exists():
        raise FileNotFoundError(str(in_path))

    # Default output directory: repository-level generated-output convention.
    if args.outdir is None or str(args.outdir).strip() == "":
        repo_root = script_dir.parent
        dataset_slug = "".join(
            ch if ch.isalnum() or ch in "._-" else "_" for ch in in_path.stem
        ).strip("._-") or "dataset"
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        outdir = (repo_root / "outputs" / "dataset_reports" / f"{dataset_slug}_{stamp}").resolve()
    else:
        repo_root = script_dir.parent
        p_out = Path(args.outdir).expanduser()
        # If relative, interpret against the repository root so outputs/ stays top-level.
        outdir = (p_out if p_out.is_absolute() else (repo_root / p_out)).resolve()

    # Load + sample
    ext = in_path.suffix.lower()
    h5_file = None
    dset = None

    if ext in (".h5", ".hdf5"):
        h5_file, dset, meta = _load_h5_meta(in_path)
        try:
            X = _sample_rows_h5_contiguous(dset, n_sample=int(args.sample), seed=int(args.seed))
        finally:
            try:
                h5_file.close()
            except Exception:
                pass
    elif ext == ".pt":
        arr, meta = _load_pt(in_path)
        rng = np.random.default_rng(int(args.seed))
        n = min(int(args.sample), int(arr.shape[0]))
        idx = rng.choice(arr.shape[0], size=n, replace=False)
        X = np.asarray(arr[idx, :])
    else:
        raise ValueError(f"Unsupported file extension: {ext!r}. Use .h5/.hdf5 or .pt")

    # Filter region (altitude band)
    X_f, filter_info = _apply_region_filter(
        X,
        r_ref_m=meta.r_ref_m,
        DU_m=meta.DU_m,
        unit_system=meta.unit_system,
        alt_min_km=args.alt_min_km,
        alt_max_km=args.alt_max_km,
    )

    if X_f.shape[0] < 10:
        raise RuntimeError(f"Too few points after filtering: {X_f.shape[0]} rows.")

    xyz = X_f[:, 0:3]
    U = X_f[:, 3]
    a = X_f[:, 4:7]
    r = np.linalg.norm(xyz, axis=1)
    a_norm = np.linalg.norm(a, axis=1)
    finite_mask = np.isfinite(X_f).all(axis=1)

    # Convert altitude if possible
    alt_km = None
    if meta.r_ref_m is not None and np.isfinite(meta.r_ref_m):
        if meta.unit_system == "canonical" and meta.DU_m is not None and np.isfinite(meta.DU_m):
            r_si = r * float(meta.DU_m)
            alt_km = (r_si - float(meta.r_ref_m)) / 1000.0
        elif meta.unit_system != "canonical":
            alt_km = (r - float(meta.r_ref_m)) / 1000.0

    # Summary
    est_bytes = meta.n_total * 7 * (4 if X_f.dtype == np.float32 else 8)

    # Determine labels based on target_mode
    _is_residual = (meta.target_mode == "residual") or (
        meta.degree_min is not None and meta.degree_min >= 0
    )
    _pot_label = "dU (residual potential)" if _is_residual else "U (full potential)"
    _acc_label = "|da| (residual accel)" if _is_residual else "|a| (full accel)"

    eps = 1e-30
    r_hat = xyz / np.maximum(r.reshape(-1, 1), eps)
    a_radial = np.sum(a * r_hat, axis=1)
    a_cross_vec = a - a_radial.reshape(-1, 1) * r_hat
    a_cross_norm = np.linalg.norm(a_cross_vec, axis=1)
    a_cos_radial = a_radial / np.maximum(a_norm, eps)
    abs_u = np.abs(U)
    dynamic_u_p99_p50 = _safe_div(
        float(np.nanpercentile(abs_u, 99.0)),
        float(np.nanpercentile(abs_u, 50.0)),
        default=float("inf"),
    )
    dynamic_a_p99_p50 = _safe_div(
        float(np.nanpercentile(a_norm, 99.0)),
        float(np.nanpercentile(a_norm, 50.0)),
        default=float("inf"),
    )

    direction_mean = np.nanmean(r_hat, axis=0)
    octant_ids = (
        (xyz[:, 0] >= 0).astype(np.int64) * 4
        + (xyz[:, 1] >= 0).astype(np.int64) * 2
        + (xyz[:, 2] >= 0).astype(np.int64)
    )
    octant_counts = np.bincount(octant_ids, minlength=8)

    declared_alt_min = _optional_float_attr(meta.raw, "alt_min_km", "train_alt_min_km", "ood_low_alt_min_km")
    declared_alt_max = _optional_float_attr(meta.raw, "alt_max_km", "train_alt_max_km", "ood_high_alt_max_km")
    dataset_role = _optional_str_attr(meta.raw, "dataset_role")
    sampling_strategy = _optional_str_attr(meta.raw, "sampling_strategy", "source_sampling_strategy")

    print("\n========== Spatial Cloud Analysis ==========")
    print(f"[file]     {in_path}")
    print(f"[dataset]  total_rows={meta.n_total:,} | columns={list(meta.columns)} | est_size~{_human_bytes(est_bytes)}")
    print(f"[body]     central_body={meta.central_body or 'unknown'} | target_mode={meta.target_mode or 'unknown'} | role={dataset_role or 'unknown'}")
    print(f"[degrees]  degree_min={meta.degree_min} | degree_max={meta.degree_max}")
    print(f"[units]    unit_system={meta.unit_system} | r_ref_m={meta.r_ref_m} | DU_m={meta.DU_m}")
    print(f"[analyzed] sampled_rows={X.shape[0]:,} | after_filter={X_f.shape[0]:,} | filter={filter_info}")
    print("")

    stats = {
        "file": str(in_path),
        "meta": {
            "n_total": int(meta.n_total),
            "columns": list(meta.columns),
            "unit_system": meta.unit_system,
            "central_body": meta.central_body,
            "target_mode": meta.target_mode,
            "degree_min": meta.degree_min,
            "degree_max": meta.degree_max,
            "a_sign_convention": meta.a_sign_convention,
            "dataset_role": dataset_role,
            "sampling_strategy": sampling_strategy,
            "mu_si": meta.mu_si,
            "r_ref_m": meta.r_ref_m,
            "DU_m": meta.DU_m,
            "TU_s": meta.TU_s,
            "VU_m_s": meta.VU_m_s,
        },
        "analyzed": {
            "sampled_rows": int(X.shape[0]),
            "rows_after_filter": int(X_f.shape[0]),
            "filter": filter_info,
        },
        "stats": {
            "radius": _np_stats(r),
            "potential": _np_stats(U),
            "accel_norm": _np_stats(a_norm),
            "x": _np_stats(xyz[:, 0]),
            "y": _np_stats(xyz[:, 1]),
            "z": _np_stats(xyz[:, 2]),
            "ax": _np_stats(a[:, 0]),
            "ay": _np_stats(a[:, 1]),
            "az": _np_stats(a[:, 2]),
        },
        "quality": {
            "finite": {
                "finite_row_fraction": float(np.mean(finite_mask)) if finite_mask.size else 0.0,
                "nonfinite_rows": int(finite_mask.size - np.count_nonzero(finite_mask)),
                "nan_values": int(np.count_nonzero(np.isnan(X_f))),
                "inf_values": int(np.count_nonzero(np.isinf(X_f))),
            },
            "field_dynamic_range": {
                "abs_potential_rms": _rms(U),
                "accel_norm_rms": _rms(a_norm),
                "abs_potential_p99_over_p50": float(dynamic_u_p99_p50),
                "accel_norm_p99_over_p50": float(dynamic_a_p99_p50),
            },
            "spatial_direction_balance": {
                "mean_unit_vector": [float(v) for v in direction_mean],
                "mean_unit_vector_norm": float(np.linalg.norm(direction_mean)),
                "octant_counts": [int(v) for v in octant_counts],
                "octant_entropy_score": _entropy_score(octant_counts),
                "octant_max_fraction": _safe_div(float(np.max(octant_counts)), float(np.sum(octant_counts))),
            },
            "acceleration_geometry": {
                "radial_component": _np_stats(a_radial),
                "cross_radial_norm": _np_stats(a_cross_norm),
                "cosine_with_radius": _np_stats(a_cos_radial),
                "cross_to_total_median": _safe_div(
                    float(np.nanpercentile(a_cross_norm, 50.0)),
                    float(np.nanpercentile(a_norm, 50.0)),
                ),
                "radial_abs_to_total_median": _safe_div(
                    float(np.nanpercentile(np.abs(a_radial), 50.0)),
                    float(np.nanpercentile(a_norm, 50.0)),
                ),
            },
            "warnings": [],
        },
    }

    if alt_km is not None:
        stats["stats"]["altitude_km"] = _np_stats(alt_km)
        alt_lo = declared_alt_min if declared_alt_min is not None else float(np.nanmin(alt_km))
        alt_hi = declared_alt_max if declared_alt_max is not None else float(np.nanmax(alt_km))
        alt_balance = _bin_balance(alt_km, alt_lo, alt_hi, n_bins=10)
        stats["quality"]["altitude_balance"] = alt_balance  # type: ignore[index]
        stats["quality"]["altitude_contract"] = {  # type: ignore[index]
            "declared_min_km": declared_alt_min,
            "declared_max_km": declared_alt_max,
            "observed_min_km": float(np.nanmin(alt_km)),
            "observed_max_km": float(np.nanmax(alt_km)),
            "below_declared_count": (
                int(np.count_nonzero(alt_km < float(declared_alt_min) - 1e-6))
                if declared_alt_min is not None else None
            ),
            "above_declared_count": (
                int(np.count_nonzero(alt_km > float(declared_alt_max) + 1e-6))
                if declared_alt_max is not None else None
            ),
        }

    warnings = stats["quality"]["warnings"]  # type: ignore[index]
    if stats["quality"]["finite"]["nonfinite_rows"] > 0:  # type: ignore[index]
        warnings.append("Dataset sample contains non-finite rows.")
    if "altitude_balance" in stats["quality"]:  # type: ignore[operator]
        alt_balance = stats["quality"]["altitude_balance"]  # type: ignore[index]
        if int(alt_balance["empty_bins"]) > 0:
            warnings.append("Altitude coverage has empty bins in the analyzed range.")
        if float(alt_balance["coefficient_of_variation"]) > 1.0:
            warnings.append("Altitude occupancy is highly imbalanced.")
    if float(stats["quality"]["spatial_direction_balance"]["octant_max_fraction"]) > 0.35:  # type: ignore[index]
        warnings.append("Directional sampling is concentrated in one octant.")
    if dynamic_a_p99_p50 > 50.0:
        warnings.append("Acceleration magnitude has a very high p99/p50 dynamic range.")

    # Print a compact sanity summary
    if alt_km is not None:
        s_alt = stats["stats"]["altitude_km"]
        print(f"[altitude km] min={s_alt['min']:.3f} | p50={s_alt['p50']:.3f} | max={s_alt['max']:.3f}")
    s_r = stats["stats"]["radius"]
    print(f"[radius] min={s_r['min']:.6g} | p50={s_r['p50']:.6g} | max={s_r['max']:.6g}")
    sU = stats["stats"]["potential"]
    print(f"[{_pot_label}] min={sU['min']:.6g} | p50={sU['p50']:.6g} | max={sU['max']:.6g}")
    sa = stats["stats"]["accel_norm"]
    print(f"[{_acc_label}] min={sa['min']:.6g} | p50={sa['p50']:.6g} | max={sa['max']:.6g}")
    q = stats["quality"]
    print(
        f"[quality] finite_rows={q['finite']['finite_row_fraction']:.3f} | "
        f"octant_entropy={q['spatial_direction_balance']['octant_entropy_score']:.3f} | "
        f"mean_dir_norm={q['spatial_direction_balance']['mean_unit_vector_norm']:.3f}"
    )
    if "altitude_balance" in q:
        ab = q["altitude_balance"]
        print(
            f"[altitude balance] empty_bins={ab['empty_bins']} | "
            f"cv={ab['coefficient_of_variation']:.3f} | entropy={ab['entropy_score']:.3f}"
        )
    ag = q["acceleration_geometry"]
    print(
        f"[accel geometry] cross/total median={ag['cross_to_total_median']:.3f} | "
        f"|radial|/total median={ag['radial_abs_to_total_median']:.3f}"
    )
    if warnings:
        print("[warnings]")
        for warning in warnings:
            print(f"  - {warning}")
    print("===========================================\n")

    # Plots
    plot_paths: Dict[str, str] = {}
    if not bool(args.no_plots):
        plot_paths = _make_plots(X_f, meta, outdir, scatter_n=int(args.scatter_n))
        print(f"[plots] saved to: {outdir}")
        for k, v in plot_paths.items():
            print(f"  - {k}: {v}")
        print("")

    stats["plots"] = plot_paths

    # JSON summary
    if bool(args.dump_json):
        outdir.mkdir(parents=True, exist_ok=True)
        out_json = outdir / "summary.json"
        out_json.write_text(json.dumps(stats, indent=2, sort_keys=True), encoding="utf-8")
        print(f"[json] wrote: {out_json}")


if __name__ == "__main__":
    main()
