# =============================================================================
# 13.                            CLI / SMOKE TEST
# =============================================================================

if __name__ == "__main__":

    import argparse
    import sys
    from pathlib import Path
    from typing import Optional, Tuple

    import numpy as np

    # ------------------------------------------------------------
    # 1) Environment setup (module-friendly)
    # ------------------------------------------------------------
    this_file = Path(__file__).resolve()
    project_root = this_file.parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    print("\n=== Starting Surface Effects Smoke Test ===")
    print("  Mode: python -m models.surface_effects")
    print(f"  Project root: {project_root}\n")

    # ------------------------------------------------------------
    # 2) CLI
    # ------------------------------------------------------------
    ap = argparse.ArgumentParser(
        description="Smoke test for LOLA LDEM (topography) and LDAM (albedo) loaders + SRP-albedo/thermal wrappers."
    )

    ap.add_argument("--img", default=None, help="Path to a PDS3 .IMG (optional).")
    ap.add_argument("--lbl", default=None, help="Path to a PDS3 label (.lbl/.LBL/.txt). If omitted, tries to infer.")

    ap.add_argument("--ldem-root", default=None, help="Directory containing LDEM products (label + IMG).")
    ap.add_argument("--albedo-root", default=None, help="Directory containing LDAM/Albedo products (label + IMG).")
    ap.add_argument("--ppd", type=int, default=None, help="Preferred LDEM resolution (e.g., 16, 64).")

    ap.add_argument("--lat", type=float, default=0.0, help="Test latitude [deg].")
    ap.add_argument("--lon", type=float, default=0.0, help="Test longitude [deg].")
    ap.add_argument("--method", choices=["nearest", "bilinear"], default="bilinear")

    ap.add_argument("--stats", action="store_true", help="Compute subsampled min/max stats (slow).")
    ap.add_argument("--stats-stride", type=int, default=200, help="Stride for subsampling when --stats is enabled.")
    ap.add_argument("--no-mmap", action="store_true", help="Disable memmap (load fully into RAM).")

    args = ap.parse_args()

    mmap = not bool(args.no_mmap)

    # ------------------------------------------------------------
    # 3) Path resolution helpers
    # ------------------------------------------------------------
    def _resolve_path(p: Optional[str]) -> Optional[Path]:
        if not p:
            return None
        q = Path(p)
        if q.exists():
            return q
        q2 = project_root / p
        if q2.exists():
            return q2
        return None

    def _infer_label_from_img(img: Path) -> Optional[Path]:
        """
        Best-effort: if user passes IMG, try to find a matching label next to it.
        Common patterns:
          - SAME_STEM.LBL / .lbl / .txt
          - IMG name referenced by a nearby label (harder; not handled here)
        """
        stem = img.with_suffix("")  # remove .IMG
        for suf in (".LBL", ".lbl", ".txt", ".label"):
            cand = stem.with_suffix(suf)
            if cand.exists():
                return cand
        return None

    def _auto_from_roots(
        ldem_root: Optional[Path],
        albedo_root: Optional[Path],
        ppd: Optional[int],
    ) -> Tuple[Optional[Path], Optional[Path]]:
        """
        Try to discover a usable (label, img) pair from provided roots.
        Preference order:
          1) LDEM if ldem_root is provided
          2) LDAM if albedo_root is provided
        """
        if ldem_root is not None:
            try:
                lbl, img = find_ldem_product(ldem_root, ppd=ppd)
                return lbl, img
            except Exception:
                pass

        if albedo_root is not None:
            try:
                lbl, img = find_lola_albedo_product(albedo_root)
                return lbl, img
            except Exception:
                pass

        return None, None

    img_path = _resolve_path(args.img)
    lbl_path = _resolve_path(args.lbl)
    ldem_root = _resolve_path(args.ldem_root)
    albedo_root = _resolve_path(args.albedo_root)

    # ------------------------------------------------------------
    # 3b) MOCK TESTS (no real files)
    # ------------------------------------------------------------
    if img_path is None and lbl_path is None and ldem_root is None and albedo_root is None:
        print("[INFO] No IMG/LBL/root provided -> running MOCK-only tests.\n")

        # Mock vectors (Moon-centered)
        r_sc = np.array([2_000e3, 0.0, 0.0], dtype=np.float64)      # 2000 km on +X
        r_sun = np.array([-150e9, 0.0, 0.0], dtype=np.float64)     # Sun on -X

        sc_props = SpacecraftProps(mass_kg=1000.0, area_m2=1.0, cr=1.5)

        print("[TEST] Albedo wrapper (mock geometry; eclipse disabled)")
        albedo_cfg = AlbedoConfig(A_moon=0.12, k_lambert=1.0)
        a_alb = albedo_accel(r_sc, r_sun, sc_props, albedo_cfg, model="simple", enable_eclipse=False)
        print(f"  a_albedo = {a_alb} m/s^2  |a|={np.linalg.norm(a_alb):.6e}")

        if (np.linalg.norm(a_alb) > 0.0) and (a_alb[0] > 0.0):
            print("  -> PASS ✅")
        else:
            print("  -> FAIL ❌")

        print("\n[TEST] Thermal wrapper (mock geometry; eclipse disabled)")
        thermal_cfg = ThermalConfig(k_thermal=1.0)
        a_th = thermal_accel(r_sc, r_sun, sc_props, thermal_cfg, enable_eclipse=False)
        print(f"  a_thermal = {a_th} m/s^2  |a|={np.linalg.norm(a_th):.6e}")

        # Thermal model can be tiny but should run
        if np.all(np.isfinite(a_th)):
            print("  -> PASS ✅")
        else:
            print("  -> FAIL ❌")

        print("\n[INFO] For real-file tests, pass either:")
        print("  - --ldem-root PATH (optionally --ppd 16/64/...)")
        print("  - --albedo-root PATH")
        print("  - or --img PATH/FILE.IMG (optionally --lbl PATH/FILE.LBL)\n")
        sys.exit(0)

    # ------------------------------------------------------------
    # 4) Resolve (label, img) pair
    # ------------------------------------------------------------
    # Priority:
    #   (A) explicit --lbl and --img
    #   (B) --img + inferred label next to it
    #   (C) roots discovery
    #   (D) label-only mode (resolve img from label)
    if lbl_path is not None and img_path is None:
        try:
            img_path, _ = _resolve_img_from_label(lbl_path)
        except Exception as e:
            print(f"FATAL: label provided but could not resolve IMG: {e}")
            sys.exit(1)

    if img_path is not None and lbl_path is None:
        lbl_path = _infer_label_from_img(img_path)

    if lbl_path is None or img_path is None:
        auto_lbl, auto_img = _auto_from_roots(ldem_root, albedo_root, args.ppd)
        if auto_lbl is not None and auto_img is not None:
            lbl_path, img_path = auto_lbl, auto_img

    if lbl_path is None:
        print("FATAL: Could not resolve a label file. Provide --lbl or --ldem-root/--albedo-root.")
        sys.exit(1)

    if img_path is None:
        try:
            img_path, _ = _resolve_img_from_label(lbl_path)
        except Exception as e:
            print(f"FATAL: Could not resolve IMG from label: {e}")
            sys.exit(1)

    assert lbl_path is not None
    assert img_path is not None

    print("-" * 70)
    print(f"LBL: {lbl_path}")
    print(f"IMG: {img_path}")

    # ------------------------------------------------------------
    # 5) Parse label -> decide type
    # ------------------------------------------------------------
    try:
        info = parse_pds3_label(lbl_path)
        is_topo = hasattr(info, "offset_km")  # LDEM map info has offset_km
        kind = "Topography (LDEM)" if is_topo else "Albedo (LDAM)"
        print(f"Type: {kind}")
        print(f"Grid: {info.lines} x {info.samples}")
        print(f"Res : {info.map_resolution_ppd} ppd")
    except Exception as e:
        print(f"FATAL: Failed to parse label: {e}")
        sys.exit(1)

    # ------------------------------------------------------------
    # 6) Load grid and sample
    # ------------------------------------------------------------
    try:
        if is_topo:
            grid = TopographyGrid(lbl_path, img_path, mmap=mmap)
            if args.method == "bilinear":
                val = float(grid.sample_bilinear(args.lat, args.lon, kind="height_m"))
            else:
                val = float(grid.sample_nearest(args.lat, args.lon, kind="height_m"))
            unit = "m (height above reference)"
        else:
            grid = LOLAAlbedoGrid(lbl_path, img_path, mmap=mmap)
            if args.method == "bilinear":
                val = float(grid.sample_bilinear(args.lat, args.lon))
            else:
                val = float(grid.sample_nearest(args.lat, args.lon))
            unit = "(albedo)"
    except Exception as e:
        print(f"FATAL: Failed to load/sample grid: {e}")
        sys.exit(1)

    print("-" * 70)
    print(f"Sample @ lat={args.lat:.6f} deg, lon={args.lon:.6f} deg  ({args.method})")
    print(f"Value  : {val:.6f} {unit}")

    # ------------------------------------------------------------
    # 7) Quick stats (subsampled)
    # ------------------------------------------------------------
    if args.stats:
        stride = max(1, int(args.stats_stride))
        print(f"\n[STATS] Subsample stride = {stride} (bigger -> faster, rougher)")

        if is_topo:
            dn = grid.dn_km  # raw DN (km) for topo
        else:
            dn = grid.dn      # raw DN for albedo grid

        sub = np.asarray(dn)[::stride, ::stride]
        with np.errstate(invalid="ignore"):
            print(f"  min = {np.nanmin(sub)}")
            print(f"  max = {np.nanmax(sub)}")

    print("\n✅ Surface Effects smoke test completed.\n")


