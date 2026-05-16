# LUNAR_SIMULATION/analysis/extras/surface_topography_explorer.py
"""
Manual 2D/3D exploration for lunar surface products.

Purpose
-------
This is an optional exploratory analysis script, not part of the mission
propagation runtime.  It is useful when checking LOLA topography/albedo inputs
by eye before wiring them into surface-collision or radiation-pressure studies.

Capabilities
------------
1) Plot topography from LOLA LDEM_*_FLOAT .LBL/.TXT + .IMG products.
2) Plot albedo from LOLA LDAM_* cylindrical .LBL/.TXT + .IMG products.
3) Query the maximum topography height inside an optional latitude/longitude
   window and report its coordinates (lat, lon) + height (m).

This script is designed to run WITHOUT CLI arguments.
Edit the "USER INPUT" section below to provide file paths and query windows.
"""

from __future__ import annotations

import sys
import os
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


# =============================================================================
# USER INPUT (EDIT HERE)
# =============================================================================

# --- Topography (example defaults; edit for your machine) ---
TOPO_LBL: Optional[str] = r"C:\Users\ayber\Desktop\Ay Modeli\Modeller\Topografya\ldem_64_float.lbl.txt"
TOPO_IMG: Optional[str] = r"C:\Users\ayber\Desktop\Ay Modeli\Modeller\Topografya\ldem_64_float.img"

# --- Albedo (optional) ---
ALBEDO_LBL: Optional[str] = r"C:\Users\ayber\Desktop\Ay Modeli\Lunar_Simulation\models\albedo\ldam_8_float.lbl.txt"
ALBEDO_IMG: Optional[str] = None  # may be resolved from label if None

# --- Output ---
# Keep outputs next to this exploratory script regardless of current working dir.
SAVE_DIR: str = str(Path(__file__).resolve().parent / "outputs_surface")
SHOW_FIGS: bool = False

# --- Query window for maximum topography (degrees) ---
# Set to None to use the full extent on that axis.
LAT_RANGE_DEG: Optional[Tuple[float, float]] = (-10.1, 10.1)   # e.g., (-10, 10) or None
LON_RANGE_DEG: Optional[Tuple[float, float]] = (0.0, 359.0)    # e.g., (0, 180) or None  <-- set None for "all longitudes"

# --- Plot controls ---
PLOT_2D: bool = True
PLOT_3D: bool = True
PLOT_ALBEDO: bool = True

STRIDE_2D: int = 16
STRIDE_3D: int = 128
STRIDE_ALBEDO: int = 16
ZSCALE_3D: float = 2.0

CLIP_TOPO_M: Optional[Tuple[float, float]] = None   # e.g. (-8000, 9000) or None for auto
CLIP_ALBEDO: Optional[Tuple[float, float]] = None  # e.g. (0.05, 0.25) or None for auto

ELEV_3D: float = 25.0
AZIM_3D: float = 35.0


# ---------------------------------------------------------------------------
# Make imports reliable (analysis/extras/ -> add repo root so `models.*` imports work)
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
_CANDIDATE_ROOTS = [_THIS_DIR, _THIS_DIR.parent, _THIS_DIR.parent.parent]
for d in _CANDIDATE_ROOTS:
    if d.exists() and str(d) not in sys.path:
        sys.path.insert(0, str(d))

try:
    from models.surface_effects import TopographyGrid, LOLAAlbedoGrid  # type: ignore
except Exception as e:
    raise ImportError(
        "Could not import models.surface_effects (TopographyGrid, LOLAAlbedoGrid).\n"
        "Make sure your repo root is on PYTHONPATH and has a 'models/' folder.\n\n"
        f"Import error: {e}"
    )


# =============================================================================
# Helpers: paths, downsampling, etc.
# =============================================================================

def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _downsample(arr: np.ndarray, stride: int) -> np.ndarray:
    s = max(1, int(stride))
    return arr[::s, ::s]


def _autodetect_label_for_img(img_path: Path, *, preferred_stems: Tuple[str, ...] = ()) -> Optional[Path]:
    """Try to locate a plausible PDS3 label next to an IMG."""
    img_path = Path(img_path)

    candidates = [
        img_path.with_suffix(".LBL"),
        img_path.with_suffix(".lbl"),
        img_path.with_suffix(".TXT"),
        img_path.with_suffix(".txt"),
        img_path.with_name(img_path.stem + ".lbl.txt"),
        img_path.with_name(img_path.stem + ".LBL.TXT"),
    ]
    for p in candidates:
        if p.exists():
            return p

    for stem in preferred_stems:
        for suf in (".LBL", ".lbl", ".TXT", ".txt", ".lbl.txt", ".LBL.TXT"):
            p = img_path.parent / f"{stem}{suf}"
            if p.exists():
                return p

    pool = (
        list(img_path.parent.glob("*.lbl")) + list(img_path.parent.glob("*.LBL")) +
        list(img_path.parent.glob("*.txt")) + list(img_path.parent.glob("*.TXT"))
    )
    img_name = img_path.name.lower()

    for p in pool:
        try:
            raw = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if img_name in raw.lower():
            return p

    # last resort: try opening
    for p in pool:
        try:
            _ = TopographyGrid(p, img_path, mmap=True)
            return p
        except Exception:
            continue

    return None


def _resolve_topo_paths(lbl: Optional[str], img: Optional[str]) -> Tuple[Path, Path]:
    img_path: Optional[Path] = Path(img) if img else None
    if img_path is None and TOPO_IMG:
        img_path = Path(TOPO_IMG)

    if img_path is None or not img_path.exists():
        # lightweight fallback: scan script folder and parent for the common filename
        for base in (_THIS_DIR, _THIS_DIR.parent):
            for cand in base.rglob("ldem_64_float.img"):
                img_path = cand
                break
            if img_path and img_path.exists():
                break

    if img_path is None or not img_path.exists():
        raise FileNotFoundError(
            "Could not locate the topography IMG.\n"
            "Edit TOPO_IMG / TOPO_LBL in the USER INPUT section."
        )

    lbl_path = Path(lbl) if lbl else (Path(TOPO_LBL) if TOPO_LBL else Path())
    if not lbl_path.exists():
        auto = _autodetect_label_for_img(img_path, preferred_stems=("LDEM_64_FLOAT", "LDEM_16_FLOAT"))
        if auto is None:
            raise FileNotFoundError(
                "Could not locate the topography label (.lbl/.txt).\n"
                "Edit TOPO_LBL in the USER INPUT section.\n"
                f"IMG used: {img_path}"
            )
        lbl_path = auto

    return lbl_path, img_path


def _resolve_albedo_paths(lbl: Optional[str], img: Optional[str]) -> Tuple[Optional[Path], Optional[Path]]:
    lbl_path = Path(lbl) if lbl else (Path(ALBEDO_LBL) if ALBEDO_LBL else None)
    img_path = Path(img) if img else (Path(ALBEDO_IMG) if ALBEDO_IMG else None)

    if lbl_path is not None and not lbl_path.exists():
        lbl_path = None
    if img_path is not None and not img_path.exists():
        img_path = None

    if lbl_path is None:
        return None, None

    if img_path is None:
        # common: foo.lbl.txt -> foo.img
        name = lbl_path.name
        stem = name
        for suf in (".lbl.txt", ".LBL.TXT", ".lbl", ".LBL", ".txt", ".TXT"):
            if stem.endswith(suf):
                stem = stem[: -len(suf)]
                break
        cand = lbl_path.with_name(stem + ".img")
        if cand.exists():
            img_path = cand

    return lbl_path, img_path


# =============================================================================
# Windowed maximum query (lat/lon range) -- memory-safe
# =============================================================================

def _latlon_centers_deg(topo: TopographyGrid) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fetch latitude/longitude center arrays from TopographyGrid.
    surface_effects stores these as _lat_centers / _lon_centers internally.
    """
    lat = getattr(topo, "lat_centers_deg", None)
    lon = getattr(topo, "lon_centers_deg", None)

    if lat is None:
        lat = getattr(topo, "_lat_centers", None)
    if lon is None:
        lon = getattr(topo, "_lon_centers", None)

    if lat is None or lon is None:
        raise AttributeError("TopographyGrid does not expose lat/lon centers (expected _lat_centers/_lon_centers).")

    return np.asarray(lat, dtype=np.float64), np.asarray(lon, dtype=np.float64)


def _indices_to_slices(idx: np.ndarray) -> List[slice]:
    """
    Convert a sorted integer index array into 1 or more contiguous slices.
    This avoids fancy-index copies on memmap-backed arrays.
    """
    idx = np.asarray(idx, dtype=int)
    if idx.size == 0:
        return []
    idx_sorted = np.sort(idx)

    slices: List[slice] = []
    start = int(idx_sorted[0])
    prev = int(idx_sorted[0])

    for v in idx_sorted[1:]:
        v = int(v)
        if v == prev + 1:
            prev = v
            continue
        slices.append(slice(start, prev + 1))
        start = v
        prev = v

    slices.append(slice(start, prev + 1))
    return slices


def _select_lat_slices(lat_deg: np.ndarray, lat_range: Optional[Tuple[float, float]]) -> List[slice]:
    if lat_range is None:
        return [slice(0, lat_deg.size)]

    lat_min, lat_max = float(lat_range[0]), float(lat_range[1])
    lo, hi = (lat_min, lat_max) if lat_min <= lat_max else (lat_max, lat_min)

    idx = np.where((lat_deg >= lo) & (lat_deg <= hi))[0].astype(int)
    return _indices_to_slices(idx)


def _select_lon_slices(lon_deg: np.ndarray, lon_range: Optional[Tuple[float, float]]) -> Tuple[List[slice], bool]:
    """
    Longitude selection with optional 0..360 interpretation + wrap-around support.

    Returns:
      (slices, use_360_report)
    """
    n = lon_deg.size
    if lon_range is None:
        return [slice(0, n)], False

    lon_min, lon_max = float(lon_range[0]), float(lon_range[1])

    # If user gave both in [0, 360], interpret query in 0..360 regardless of dataset convention.
    use_360 = (0.0 <= lon_min <= 360.0) and (0.0 <= lon_max <= 360.0)
    if use_360:
        lon_vals = np.mod(lon_deg, 360.0)
        lo = lon_min % 360.0
        hi = lon_max % 360.0
    else:
        lon_vals = lon_deg
        lo, hi = lon_min, lon_max

    if lo <= hi:
        idx = np.where((lon_vals >= lo) & (lon_vals <= hi))[0].astype(int)
        return _indices_to_slices(idx), use_360

    # Wrap-around window, e.g. 350..10
    idx = np.where((lon_vals >= lo) | (lon_vals <= hi))[0].astype(int)
    return _indices_to_slices(idx), use_360


def report_topography_max(
    topo: TopographyGrid,
    *,
    lat_range: Optional[Tuple[float, float]],
    lon_range: Optional[Tuple[float, float]],
) -> Tuple[float, float, float]:
    """
    Find the maximum topography height within the given lat/lon bounds.

    Returns:
      (h_max_m, lat_deg, lon_deg_report)
    """
    # memmap-backed 2D raster
    dn = topo.dn_km  # raw DN values :contentReference[oaicite:4]{index=4}
    sf = float(getattr(topo.info, "scaling_factor", 1.0) or 1.0)

    lat_deg, lon_deg = _latlon_centers_deg(topo)

    lat_slices = _select_lat_slices(lat_deg, lat_range)
    lon_slices, use_360_report = _select_lon_slices(lon_deg, lon_range)

    if not lat_slices:
        raise ValueError(f"No latitude samples inside lat_range={lat_range}.")
    if not lon_slices:
        raise ValueError(f"No longitude samples inside lon_range={lon_range}.")

    # If scaling is negative (unlikely), max height corresponds to min DN.
    want_max = (sf >= 0.0)

    best_val_dn = None
    best_i = None
    best_j = None

    for sl_lat in lat_slices:
        for sl_lon in lon_slices:
            view = dn[sl_lat, sl_lon]  # view, no fancy-index copy
            try:
                seg_ext = np.nanmax(view) if want_max else np.nanmin(view)
            except ValueError:
                continue  # empty slice
            if not np.isfinite(seg_ext):
                continue

            if best_val_dn is None:
                best_val_dn = float(seg_ext)
                # refine to indices
                k = int(np.nanargmax(view) if want_max else np.nanargmin(view))
                ii, jj = np.unravel_index(k, view.shape)
                best_i = int(sl_lat.start) + int(ii)
                best_j = int(sl_lon.start) + int(jj)
            else:
                if (want_max and float(seg_ext) > best_val_dn) or ((not want_max) and float(seg_ext) < best_val_dn):
                    best_val_dn = float(seg_ext)
                    k = int(np.nanargmax(view) if want_max else np.nanargmin(view))
                    ii, jj = np.unravel_index(k, view.shape)
                    best_i = int(sl_lat.start) + int(ii)
                    best_j = int(sl_lon.start) + int(jj)

    if best_val_dn is None or best_i is None or best_j is None:
        raise ValueError("Selected window contains no finite height samples.")

    h_max_m = float(best_val_dn) * sf * 1000.0
    lat_max = float(lat_deg[best_i])

    lon_raw = float(lon_deg[best_j])
    lon_360 = float(np.mod(lon_raw, 360.0))
    lon_report = lon_360 if (lon_range is not None and use_360_report) else lon_raw

    return h_max_m, lat_max, lon_report


def _topo_height_grid_m(topo: TopographyGrid, stride: int) -> np.ndarray:
    """
    Build a downsampled height grid in meters for plotting.

    NOTE:
    surface_effects defines height conversion as h_km = DN * scaling_factor,
    h_m = h_km * 1000. :contentReference[oaicite:5]{index=5}
    """
    dn_ds = _downsample(topo.dn_km, stride).astype(np.float32, copy=False)
    sf = float(getattr(topo.info, "scaling_factor", 1.0) or 1.0)
    return (dn_ds * sf) * 1000.0


# =============================================================================
# Plotting
# =============================================================================

def plot_topography_2d(
    topo: TopographyGrid,
    *,
    stride: int,
    clip_m: Optional[Tuple[float, float]],
    save_path: Path,
    show: bool,
) -> None:
    h_m_ds = _topo_height_grid_m(topo, stride=stride)

    lat_deg, lon_deg = _latlon_centers_deg(topo)
    lon = lon_deg[::stride]
    lat = lat_deg[::stride]
    extent = [float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max())]

    if clip_m is not None:
        vmin, vmax = float(clip_m[0]), float(clip_m[1])
    else:
        p2, p98 = np.nanpercentile(h_m_ds, [2, 98])
        vmin, vmax = float(p2), float(p98)

    fig = plt.figure(figsize=(14, 6))
    ax = fig.add_subplot(111)
    im = ax.imshow(
        h_m_ds,
        extent=extent,
        origin="upper",
        aspect="auto",
        norm=Normalize(vmin=vmin, vmax=vmax),
        cmap="gray",
    )
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Height above reference sphere (m)")

    try:
        ppd = float(getattr(topo.info, "map_resolution_ppd", np.nan))
        ax.set_title(f"Lunar Topography — 2D (Simple Cylindrical, {ppd:g} pix/deg)")
    except Exception:
        ax.set_title("Lunar Topography — 2D (Simple Cylindrical)")

    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Latitude (deg)")

    fig.tight_layout()
    fig.savefig(save_path, dpi=220)
    if show:
        plt.show()
    plt.close(fig)


def plot_topography_3d(
    topo: TopographyGrid,
    *,
    stride: int,
    zscale: float,
    clip_m: Optional[Tuple[float, float]],
    save_path: Path,
    show: bool,
    elev: float,
    azim: float,
) -> None:
    h_m = _topo_height_grid_m(topo, stride=stride)

    lat_deg, lon_deg = _latlon_centers_deg(topo)
    lon_deg_ds = lon_deg[::stride]
    lat_deg_ds = lat_deg[::stride]

    lon = np.deg2rad(lon_deg_ds)
    lat = np.deg2rad(lat_deg_ds)
    Lon, Lat = np.meshgrid(lon, lat)

    R = float(getattr(topo, "reference_radius_m", 1737400.0))
    r = R + (zscale * h_m)

    X = r * np.cos(Lat) * np.cos(Lon)
    Y = r * np.cos(Lat) * np.sin(Lon)
    Z = r * np.sin(Lat)

    if clip_m is not None:
        vmin, vmax = float(clip_m[0]), float(clip_m[1])
    else:
        p2, p98 = np.nanpercentile(h_m, [2, 98])
        vmin, vmax = float(p2), float(p98)

    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap("gray")
    facecolors = cmap(norm(h_m))

    fig = plt.figure(figsize=(10.5, 10))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(
        X, Y, Z,
        facecolors=facecolors,
        linewidth=0,
        antialiased=False,
        shade=False,
    )

    ax.set_box_aspect((1, 1, 1))
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.grid(False)
    ax.view_init(elev=elev, azim=azim)

    try:
        ppd = float(getattr(topo.info, "map_resolution_ppd", np.nan))
        ax.set_title(f"Moon Topography — 3D (ppd={ppd:g}, stride={stride}, zscale={zscale:g})")
    except Exception:
        ax.set_title(f"Moon Topography — 3D (stride={stride}, zscale={zscale:g})")

    fig.tight_layout()
    fig.savefig(save_path, dpi=240)
    if show:
        plt.show()
    plt.close(fig)


def plot_albedo_2d(
    alb: LOLAAlbedoGrid,
    *,
    stride: int,
    clip: Optional[Tuple[float, float]],
    save_path: Path,
    show: bool,
) -> None:
    dn = alb.dn
    dn_ds = _downsample(dn, stride).astype(np.float64, copy=False)

    sf = float(alb.info.scaling_factor)
    off = float(alb.info.offset)
    missing = float(alb.info.missing_constant)

    val = dn_ds * sf + off
    if np.isfinite(missing):
        val = np.where(dn_ds == missing, np.nan, val)

    lon = alb.lon_centers_deg[::stride]
    lat = alb.lat_centers_deg[::stride]
    extent = [float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max())]

    if clip is not None:
        vmin, vmax = float(clip[0]), float(clip[1])
    else:
        p2, p98 = np.nanpercentile(val, [2, 98])
        vmin, vmax = float(p2), float(p98)

    fig = plt.figure(figsize=(14, 6))
    ax = fig.add_subplot(111)
    im = ax.imshow(
        val,
        extent=extent,
        origin="upper",
        aspect="auto",
        norm=Normalize(vmin=vmin, vmax=vmax),
        cmap="gray",
    )
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Albedo (scaled)")

    try:
        ppd = float(alb.info.map_resolution_ppd)
        ax.set_title(f"Lunar Albedo — 2D (Simple Cylindrical, {ppd:g} pix/deg)")
    except Exception:
        ax.set_title("Lunar Albedo — 2D (Simple Cylindrical)")

    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Latitude (deg)")

    fig.tight_layout()
    fig.savefig(save_path, dpi=220)
    if show:
        plt.show()
    plt.close(fig)


# =============================================================================
# Main runner (no CLI)
# =============================================================================

def run() -> Tuple[float, float, float]:
    topo_lbl, topo_img = _resolve_topo_paths(TOPO_LBL, TOPO_IMG)
    print(f"[SURFACE-PLOT] Topography:\n  LBL: {topo_lbl}\n  IMG: {topo_img}", flush=True)

    topo = TopographyGrid(topo_lbl, topo_img, mmap=True, flip_lat=False)

    # --- windowed maximum ---
    hmax_m, lat_max, lon_max = report_topography_max(
        topo,
        lat_range=LAT_RANGE_DEG,
        lon_range=LON_RANGE_DEG,
    )

    lat_msg = f"{LAT_RANGE_DEG}" if LAT_RANGE_DEG is not None else "(full)"
    lon_msg = f"{LON_RANGE_DEG}" if LON_RANGE_DEG is not None else "(full)"
    print(
        "[SURFACE-PLOT] Windowed maximum topography:\n"
        f"  lat_range: {lat_msg}\n"
        f"  lon_range: {lon_msg}\n"
        f"  max_height_m: {hmax_m:.3f}\n"
        f"  at (lat_deg, lon_deg): ({lat_max:.6f}, {lon_max:.6f})",
        flush=True
    )

    # --- outputs ---
    save_dir = Path(SAVE_DIR)
    if not save_dir.is_absolute():
        save_dir = _THIS_DIR / save_dir
    out_dir = _ensure_dir(save_dir)

    out_topo_2d = out_dir / "moon_topography_2d.png"
    out_topo_3d = out_dir / "moon_topography_3d.png"
    out_alb_2d = out_dir / "moon_albedo_2d.png"

    if PLOT_2D:
        plot_topography_2d(topo, stride=STRIDE_2D, clip_m=CLIP_TOPO_M, save_path=out_topo_2d, show=SHOW_FIGS)

    if PLOT_3D:
        plot_topography_3d(
            topo,
            stride=STRIDE_3D,
            zscale=ZSCALE_3D,
            clip_m=CLIP_TOPO_M,
            save_path=out_topo_3d,
            show=SHOW_FIGS,
            elev=ELEV_3D,
            azim=AZIM_3D,
        )

    if PLOT_ALBEDO:
        alb_lbl, alb_img = _resolve_albedo_paths(ALBEDO_LBL, ALBEDO_IMG)
        if alb_lbl is None:
            print("[SURFACE-PLOT] Albedo: label not provided / not found; skipping albedo plot.", flush=True)
        else:
            print(
                f"[SURFACE-PLOT] Albedo:\n  LBL: {alb_lbl}\n  IMG: {alb_img if alb_img else '(resolved from label or same-folder)'}",
                flush=True
            )
            alb = LOLAAlbedoGrid(alb_lbl, alb_img, mmap=True)
            plot_albedo_2d(alb, stride=STRIDE_ALBEDO, clip=CLIP_ALBEDO, save_path=out_alb_2d, show=SHOW_FIGS)

    print("[OK] Outputs:", flush=True)
    for p in (out_topo_2d, out_topo_3d, out_alb_2d):
        if p.exists():
            print(f"  {p}", flush=True)

    return hmax_m, lat_max, lon_max


if __name__ == "__main__":
    run()
