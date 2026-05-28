"""
Reusable module for 2D/3D visualization of lunar surface products.

Purpose
-------
Visualizes topography (LOLA LDEM_*_FLOAT) and albedo (LOLA LDAM_*) products.
Also includes utilities to query maximum topography inside coordinate windows.

Can be run via CLI or imported as a library for Jupyter notebooks or manual analysis.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from lunaris.loaders.io_surface import TopographyGrid, LOLAAlbedoGrid

# =============================================================================
# Helpers: paths, downsampling, etc.
# =============================================================================

def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def _downsample(arr: np.ndarray, stride: int) -> np.ndarray:
    s = max(1, int(stride))
    return arr[::s, ::s]

def resolve_topography_paths(label_path: Optional[str | Path], img_path: Optional[str | Path]) -> tuple[Path, Path]:
    """Resolve topography label and IMG paths."""
    if not label_path or not img_path:
        raise ValueError("Both label_path and img_path must be provided for topography.")
    lbl = Path(label_path)
    img = Path(img_path)
    if not lbl.exists():
        raise FileNotFoundError(f"Topography label not found: {lbl}")
    if not img.exists():
        raise FileNotFoundError(f"Topography IMG not found: {img}")
    return lbl, img

def resolve_albedo_paths(label_path: Optional[str | Path], img_path: Optional[str | Path]) -> tuple[Path | None, Path | None]:
    """Resolve albedo label and IMG paths."""
    if not label_path:
        return None, None
    lbl = Path(label_path)
    if not lbl.exists():
        return None, None
        
    img: Optional[Path] = None
    if img_path:
        img = Path(img_path)
    else:
        # try to guess img from label name
        name = lbl.name
        stem = name
        for suf in (".lbl.txt", ".LBL.TXT", ".lbl", ".LBL", ".txt", ".TXT"):
            if stem.endswith(suf):
                stem = stem[: -len(suf)]
                break
        cand = lbl.with_name(stem + ".img")
        if cand.exists():
            img = cand
            
    if img is not None and not img.exists():
        img = None
        
    if img is None:
        return None, None
        
    return lbl, img

# =============================================================================
# Windowed maximum query (lat/lon range) -- memory-safe
# =============================================================================

def _latlon_centers_deg(topo: TopographyGrid) -> Tuple[np.ndarray, np.ndarray]:
    lat = getattr(topo, "lat_centers_deg", None)
    lon = getattr(topo, "lon_centers_deg", None)
    if lat is None:
        lat = getattr(topo, "_lat_centers", None)
    if lon is None:
        lon = getattr(topo, "_lon_centers", None)
    if lat is None or lon is None:
        raise AttributeError("TopographyGrid does not expose lat/lon centers.")
    return np.asarray(lat, dtype=np.float64), np.asarray(lon, dtype=np.float64)

def _indices_to_slices(idx: np.ndarray) -> List[slice]:
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
    n = lon_deg.size
    if lon_range is None:
        return [slice(0, n)], False

    lon_min, lon_max = float(lon_range[0]), float(lon_range[1])
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

    idx = np.where((lon_vals >= lo) | (lon_vals <= hi))[0].astype(int)
    return _indices_to_slices(idx), use_360

def report_topography_max(
    topo: TopographyGrid,
    lat_range: Optional[Tuple[float, float]] = None,
    lon_range: Optional[Tuple[float, float]] = None,
) -> Tuple[float, float, float]:
    """Find max topography height within lat/lon bounds."""
    dn = topo.dn_km
    sf = float(getattr(topo.info, "scaling_factor", 1.0) or 1.0)
    lat_deg, lon_deg = _latlon_centers_deg(topo)

    lat_slices = _select_lat_slices(lat_deg, lat_range)
    lon_slices, use_360_report = _select_lon_slices(lon_deg, lon_range)

    if not lat_slices:
        raise ValueError(f"No latitude samples inside lat_range={lat_range}.")
    if not lon_slices:
        raise ValueError(f"No longitude samples inside lon_range={lon_range}.")

    want_max = (sf >= 0.0)
    best_val_dn = None
    best_i = None
    best_j = None

    for sl_lat in lat_slices:
        for sl_lon in lon_slices:
            view = dn[sl_lat, sl_lon]
            try:
                seg_ext = np.nanmax(view) if want_max else np.nanmin(view)
            except ValueError:
                continue
            if not np.isfinite(seg_ext):
                continue

            if best_val_dn is None:
                best_val_dn = float(seg_ext)
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
    dn_ds = _downsample(topo.dn_km, stride).astype(np.float32, copy=False)
    sf = float(getattr(topo.info, "scaling_factor", 1.0) or 1.0)
    return (dn_ds * sf) * 1000.0

# =============================================================================
# Plotting
# =============================================================================

def plot_topography_2d(
    topo: TopographyGrid,
    out_dir: str | Path,
    stride: int = 16,
    clip_m: Optional[Tuple[float, float]] = None,
    show: bool = False,
) -> Path:
    out_dir = _ensure_dir(Path(out_dir))
    save_path = out_dir / "moon_topography_2d.png"
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
    return save_path

def plot_topography_3d(
    topo: TopographyGrid,
    out_dir: str | Path,
    stride: int = 128,
    zscale: float = 2.0,
    clip_m: Optional[Tuple[float, float]] = None,
    show: bool = False,
    elev: float = 25.0,
    azim: float = 35.0,
) -> Path:
    out_dir = _ensure_dir(Path(out_dir))
    save_path = out_dir / "moon_topography_3d.png"
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
    return save_path

def plot_albedo_2d(
    alb: LOLAAlbedoGrid,
    out_dir: str | Path,
    stride: int = 16,
    clip: Optional[Tuple[float, float]] = None,
    show: bool = False,
) -> Path:
    out_dir = _ensure_dir(Path(out_dir))
    save_path = out_dir / "moon_albedo_2d.png"
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
    return save_path

# =============================================================================
# CLI
# =============================================================================

def run_surface_explorer(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir)

    lat_range = None
    if args.lat_min is not None and args.lat_max is not None:
        lat_range = (args.lat_min, args.lat_max)

    lon_range = None
    if args.lon_min is not None and args.lon_max is not None:
        lon_range = (args.lon_min, args.lon_max)

    # 1. Topography
    if args.topo_label and args.topo_img:
        lbl, img = resolve_topography_paths(args.topo_label, args.topo_img)
        print(f"Loading topography from {lbl} / {img}...")
        topo = TopographyGrid(lbl, img, mmap=True, flip_lat=False)

        hmax_m, lat_max, lon_max = report_topography_max(topo, lat_range, lon_range)
        print(f"Max height: {hmax_m:.2f}m at lat={lat_max:.4f}, lon={lon_max:.4f}")

        if args.plot_2d:
            print("Plotting Topography 2D...")
            p = plot_topography_2d(topo, out_dir, stride=args.stride_2d, show=args.show)
            print(f"  Saved to {p}")

        if args.plot_3d:
            print("Plotting Topography 3D...")
            p = plot_topography_3d(topo, out_dir, stride=args.stride_3d, show=args.show)
            print(f"  Saved to {p}")
    else:
        if args.plot_2d or args.plot_3d:
            print("Warning: Topography arguments missing, skipping topography plots.", file=sys.stderr)

    # 2. Albedo
    if args.plot_albedo:
        if args.albedo_label:
            lbl_a, img_a = resolve_albedo_paths(args.albedo_label, args.albedo_img)
            if lbl_a and img_a:
                print(f"Loading albedo from {lbl_a} / {img_a}...")
                alb = LOLAAlbedoGrid(lbl_a, img_a, mmap=True)
                print("Plotting Albedo 2D...")
                p = plot_albedo_2d(alb, out_dir, stride=args.stride_albedo, show=args.show)
                print(f"  Saved to {p}")
            else:
                print("Error: Could not resolve albedo files.", file=sys.stderr)
        else:
            print("Warning: Albedo label missing, skipping albedo plot.", file=sys.stderr)

    return 0

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Lunar Surface and Topography Explorer.")
    parser.add_argument("--topo-label", help="Path to topography .lbl file")
    parser.add_argument("--topo-img", help="Path to topography .img file")
    parser.add_argument("--albedo-label", help="Path to albedo .lbl file")
    parser.add_argument("--albedo-img", help="Path to albedo .img file (optional if inferable)")
    
    parser.add_argument("--out-dir", default="outputs_surface", help="Output directory for plots")
    
    parser.add_argument("--lat-min", type=float, help="Min latitude for query")
    parser.add_argument("--lat-max", type=float, help="Max latitude for query")
    parser.add_argument("--lon-min", type=float, help="Min longitude for query")
    parser.add_argument("--lon-max", type=float, help="Max longitude for query")

    parser.add_argument("--plot-2d", action="store_true", help="Plot 2D topography")
    parser.add_argument("--plot-3d", action="store_true", help="Plot 3D topography")
    parser.add_argument("--plot-albedo", action="store_true", help="Plot 2D albedo")
    
    parser.add_argument("--stride-2d", type=int, default=16, help="Downsample stride for 2D topo")
    parser.add_argument("--stride-3d", type=int, default=128, help="Downsample stride for 3D topo")
    parser.add_argument("--stride-albedo", type=int, default=16, help="Downsample stride for albedo")
    
    parser.add_argument("--show", action="store_true", help="Display GUI plot windows")

    args = parser.parse_args(argv)

    if not (args.topo_label and args.topo_img) and not args.albedo_label:
        parser.print_help()
        print("\nError: You must provide either topography inputs (--topo-label, --topo-img) or albedo input (--albedo-label).", file=sys.stderr)
        return 1

    return run_surface_explorer(args)

if __name__ == "__main__":
    sys.exit(main())
