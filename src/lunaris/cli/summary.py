"""Summary-printing helpers for the main ``lunaris`` run command."""

from __future__ import annotations

from typing import Any

import numpy as np

from lunaris.core.config import SimConfig


def _extract_rv6(
    y0: Any,
) -> tuple[float | None, float | None, float | None, float | None, float | None, float | None]:
    """Extract (rx, ry, rz, vx, vy, vz) from strict initial-state container styles.

    Supported:
      - common.type_defs.InitialState: attributes x,y,z,vx,vy,vz or .to_array()
      - core.state.OrbitState: packed vector via .y (len>=6)
      - Generic containers: attributes position/velocity or r_m/v_ms (3,)
      - Array-like (len>=6)

    Returns (None,... ) if extraction fails.
    """
    try:
        if y0 is None:
            return (None, None, None, None, None, None)

        # SSOT initial state dataclass (common.type_defs.InitialState)
        if hasattr(y0, "to_array"):
            arr = y0.to_array()
            return (float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3]), float(arr[4]), float(arr[5]))

        if all(hasattr(y0, k) for k in ("x", "y", "z", "vx", "vy", "vz")):
            return (
                float(y0.x),
                float(y0.y),
                float(y0.z),
                float(y0.vx),
                float(y0.vy),
                float(y0.vz),
            )

        # Other state containers
        if hasattr(y0, "position") and hasattr(y0, "velocity"):
            r = y0.position
            v = y0.velocity
            return (float(r[0]), float(r[1]), float(r[2]), float(v[0]), float(v[1]), float(v[2]))

        if hasattr(y0, "r_m") and hasattr(y0, "v_ms"):
            r = y0.r_m
            v = y0.v_ms
            return (float(r[0]), float(r[1]), float(r[2]), float(v[0]), float(v[1]), float(v[2]))

        if hasattr(y0, "y"):
            y = y0.y
            return (float(y[0]), float(y[1]), float(y[2]), float(y[3]), float(y[4]), float(y[5]))

        y = y0  # assume array-like
        return (float(y[0]), float(y[1]), float(y[2]), float(y[3]), float(y[4]), float(y[5]))
    except Exception:
        return (None, None, None, None, None, None)


def print_summary(cfg: SimConfig, orbit_params: dict[str, float] | None, y0: Any) -> None:
    """Pretty-print a run summary (CLI-oriented)."""
    f = cfg.flags
    sc = cfg.spacecraft

    print("=" * 64)
    print("LUNARIS RUNNER (STRICT)")
    print("=" * 64)
    print("[Time]")
    print(f"  start_date   : {cfg.time.start_date}")
    print(f"  duration     : {cfg.time.duration_s:.1f} s  ({cfg.time.duration_days:.6f} days)")
    print(f"  output_dt_s  : {cfg.time.output_dt_s}")
    print(f"  samples/period (if output_dt_s is None): {cfg.time.samples_per_period}")
    print()
    print("[Output]")
    print(f"  out_dir      : {cfg.output.out_dir}")
    print(f"  make_3d_plots : {cfg.output.make_3d_plots}")
    print(f"  downsample_3d : {cfg.output.downsample_3d}")
    print()
    print("[Spacecraft]")
    print(f"  mass_kg      : {sc.mass_kg}")
    print(f"  area_m2      : {sc.area_m2}")
    print(f"  cd / cr      : {sc.cd} / {sc.cr}")
    print()
    print("[Gravity]")
    print(f"  backend      : {cfg.gravity.backend}")
    if cfg.gravity.uses_st_lrps:
        print(f"  model_dir    : {cfg.gravity.st_lrps_model_dir}")
    else:
        print(f"  file_path    : {cfg.gravity.file_path}")
        print(f"  degree       : {cfg.gravity.degree}")
        print(f"  adaptive     : enabled={cfg.gravity.adaptive.enabled} table={cfg.gravity.adaptive.altitude_table is not None}")
    print()
    print("[Forces]")
    print(f"  High-fidelity gravity: {f.enable_sh}")
    print(f"  Third-body Sun       : {f.enable_3rd_body_sun}")
    print(f"  Third-body Earth     : {f.enable_3rd_body_earth}")
    print(f"  Earth J2             : {f.enable_earth_j2}")
    print(f"  SRP                  : {f.enable_srp}")
    print(f"  Albedo               : {f.enable_albedo}")
    print(f"  Thermal              : {f.enable_thermal}")
    if f.enable_thermal and cfg.thermal is not None:
        print(
            "  Thermal mode/grid    : "
            f"{cfg.thermal.thermal_mode} / {cfg.thermal.facet_lat_count}x{cfg.thermal.facet_lon_count}"
        )
    tides_on = bool(f.enable_tides_k2 or f.enable_tides_k3)
    tides_kind = "k3" if f.enable_tides_k3 else ("k2" if f.enable_tides_k2 else "off")
    print(f"  Tides                : {tides_on} (kind={tides_kind})")
    if tides_on and cfg.solid_tides is not None:
        k3_str = "explicit" if cfg.solid_tides.k3 is not None else "unset"
        print(f"  Tides bodies/k2/k3   : {','.join(cfg.solid_tides.tide_bodies)} / {cfg.solid_tides.k2:g} / {k3_str}")
    print(f"  Relativity (1PN)     : {f.enable_relativity_1pn}")
    print()
    print("[Initial State]")
    if orbit_params:
        print(f"  COE: a={orbit_params['a_km']:.3f} km e={orbit_params['e']:.6f} i={orbit_params['inc_deg']:.3f} deg")
    else:
        print("  COE: (from cfg.initial_state)")

    rx, ry, rz, vx, vy, vz = _extract_rv6(y0)
    if rx is None:
        print(f"  r0 [m]  : (unavailable; initial_state={type(y0).__name__})")
        print("  v0 [m/s]: (unavailable)")
    else:
        print(f"  r0 [m]  : ({rx:.3f}, {ry:.3f}, {rz:.3f})")
        print(f"  v0 [m/s]: ({vx:.6f}, {vy:.6f}, {vz:.6f})")

    print("=" * 64)


def median_dt(t_arr: Any) -> float | None:
    """Median sampling interval for a time array."""
    try:
        t = np.asarray(t_arr, dtype=float).ravel()
        if t.size < 3:
            return None
        dt = np.diff(t)
        if dt.size == 0:
            return None
        return float(np.median(dt))
    except Exception:
        return None


__all__ = ["_extract_rv6", "print_summary", "median_dt"]
