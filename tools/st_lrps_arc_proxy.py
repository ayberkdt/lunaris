"""Trajectory-level error proxy for ST-LRPS checkpoints without propagation.

For a set of cached reference orbits (RK4 propagations of the SH reference
field, e.g. a ``lunaris-benchmark --gpu-batch-compare`` cache of the SH200
series), the first-order along-track position error at the final time T of a
propagation that replaces the residual field by a surrogate is

    s(T) = ∫_0^T p_v(τ) · f(τ) dτ,   f = a_surrogate_residual - a_true_residual,

where p = (p_r, p_v) solves the adjoint of the linearised dynamics around the
reference arc, ṗ_r = -G p_v, ṗ_v = -p_r, p(T) = (î(T), 0), î the along-track
unit vector at T and G the gravity gradient of the base field (degrees 0..N_base,
including μ/r). The response is linear in f, so ``build`` stores, once per
orbit, the reference positions in the body-fixed frame, the true residual
acceleration there and the quadrature-weighted adjoint p_v (rotated into the
body-fixed frame). ``score`` then needs one surrogate evaluation per stored
point and a dot product: seconds per checkpoint instead of a propagation.

The linearisation was checked on the cached 5-day RK4 (dt 15 s) propagations
of the 2026-09-20 Plan D model, using a forward linear-response integration
with the same gradient, resampling and frame: for the surrogate-minus-SH25
pair (the response to the full learned residual) it matched the propagated
along-track difference to within ~3% on 6 of 6 orbits (100-560 km perilune).
Errors of the integrator itself (RK4 step error that differs between the two
propagated fields) are not modelled.

Sub-commands::

    python tools/st_lrps_arc_proxy.py build --benchmark-dir DIR --out DIR [--orbits 32] [--step 5]
    python tools/st_lrps_arc_proxy.py score --proxy DIR --run-dir RUN [--device cuda]

``--benchmark-dir`` is a benchmark output directory (``run_metadata.json`` with
the original argv, ``benchmark_cache/models/<series>/scenario_*.npz``).
Set ``LUNARIS_DATA_DIR`` when this checkout has no ``data/`` directory.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

MU_SI = 4902800306330.2     # GL1800F header value, as used by the benchmark
R_REF_M = 1738000.0


def quat_to_matrix(q: Sequence[float]) -> np.ndarray:
    """Rotation matrix of a (w, x, y, z) quaternion (inertial -> body-fixed)."""
    w, x, y, z = (float(v) for v in q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def hermite_resample(t: np.ndarray, r: np.ndarray, v: np.ndarray, h: float):
    """Cubic Hermite interpolation of a (t, r, v) arc onto a uniform grid of step h.

    The residual field's shortest wavelengths (~58 km for degree 200 at 100 km)
    are crossed in ~36 s, so cached 60 s outputs must be refined before the
    error integral is sampled.
    """
    tt = np.arange(t[0], t[-1] + 0.5 * h, h)
    k = np.clip(np.searchsorted(t, tt, side="right") - 1, 0, len(t) - 2)
    dt = (t[k + 1] - t[k])[:, None]
    s = ((tt - t[k]) / dt[:, 0])[:, None]
    h00, h10 = 2 * s**3 - 3 * s**2 + 1, s**3 - 2 * s**2 + s
    h01, h11 = -2 * s**3 + 3 * s**2, s**3 - s**2
    rr = h00 * r[k] + h10 * dt * v[k] + h01 * r[k + 1] + h11 * dt * v[k + 1]
    d00, d10 = 6 * s**2 - 6 * s, 3 * s**2 - 4 * s + 1
    d01, d11 = -6 * s**2 + 6 * s, 3 * s**2 - 2 * s
    vv = (d00 * r[k] + d01 * r[k + 1]) / dt + d10 * v[k] + d11 * v[k + 1]
    return tt, rr, vv


def along_track_unit(r: np.ndarray, v: np.ndarray) -> np.ndarray:
    r_hat = r / np.linalg.norm(r)
    c = np.cross(r, v)
    return np.cross(c / np.linalg.norm(c), r_hat)


def simpson_weights(n: int, h: float) -> np.ndarray:
    """Composite Simpson weights for n (odd) uniformly spaced samples."""
    if n < 3 or n % 2 == 0:
        raise ValueError(f"Simpson needs an odd sample count >= 3, got {n}")
    w = np.full(n, 2.0)
    w[1::2] = 4.0
    w[0] = w[-1] = 1.0
    return w * h / 3.0


def adjoint_velocity_sensitivity(gradients_half: np.ndarray, c: np.ndarray, h: float) -> np.ndarray:
    """p_v on the grid for ṗ_r = -G p_v, ṗ_v = -p_r, p(T) = (c, 0), backward RK4.

    ``gradients_half`` is (2n-1, 3, 3) on the half-step grid: even rows are the
    n grid points, odd rows the RK4 mid-steps. The mid-step gradient must come
    from the true mid-step position. Averaging the neighbours shrinks the
    rotating Kepler gradient by ~(n h)^2/8, and the neutrally unstable
    along-track response amplifies that by ~(n T)^2: about 20% at h = 10 s
    over two days in the circular-orbit test.
    """
    if len(gradients_half) % 2 == 0:
        raise ValueError("gradients_half must hold 2n-1 half-step samples")
    gradients = gradients_half[0::2]
    mids = gradients_half[1::2]
    n = len(gradients)
    pv_out = np.empty((n, 3))
    pr = np.asarray(c, dtype=np.float64).copy()
    pv = np.zeros(3)
    pv_out[-1] = pv

    def rhs(g: np.ndarray, pr_: np.ndarray, pv_: np.ndarray):
        # d/ds with s = -t (backward): dpr/ds = G pv, dpv/ds = pr
        return g @ pv_, pr_

    for i in range(n - 1, 0, -1):
        g1, g2, g3 = gradients[i], mids[i - 1], gradients[i - 1]
        k1r, k1v = rhs(g1, pr, pv)
        k2r, k2v = rhs(g2, pr + 0.5 * h * k1r, pv + 0.5 * h * k1v)
        k3r, k3v = rhs(g2, pr + 0.5 * h * k2r, pv + 0.5 * h * k2v)
        k4r, k4v = rhs(g3, pr + h * k3r, pv + h * k3v)
        pr = pr + h / 6.0 * (k1r + 2 * k2r + 2 * k3r + k4r)
        pv = pv + h / 6.0 * (k1v + 2 * k2v + 2 * k3v + k4v)
        pv_out[i - 1] = pv
    return pv_out


# --------------------------------------------------------------------------- #
# build
# --------------------------------------------------------------------------- #


def _sh_eval(xyz: np.ndarray, tables: tuple, nmax: int, nmin: int) -> tuple[np.ndarray, np.ndarray]:
    from lunaris.physics.spherical_harmonics import sh_potential_accel_batch_serial

    C, S, a_nm, b_nm, diag_f, subdiag_f, k_ratio = tables
    u, a = sh_potential_accel_batch_serial(
        np.ascontiguousarray(xyz, dtype=np.float64), C, S, a_nm, b_nm, diag_f, subdiag_f, k_ratio,
        MU_SI, R_REF_M, nmax, nmin,
    )
    return np.asarray(u).reshape(-1), np.asarray(a).reshape(-1, 3)


def _base_gradient(r_fix: np.ndarray, tables: tuple, base_degree: int, step_m: float = 20.0) -> np.ndarray:
    """Body-fixed gravity gradient of degrees 0..base_degree (incl. μ/r), central differences."""
    g = np.empty((len(r_fix), 3, 3))
    for k in range(3):
        e = np.zeros(3)
        e[k] = step_m
        _, ap = _sh_eval(r_fix + e, tables, base_degree, -1)
        _, am = _sh_eval(r_fix - e, tables, base_degree, -1)
        g[:, :, k] = (ap - am) / (2.0 * step_m)
    return 0.5 * (g + np.transpose(g, (0, 2, 1)))


def build(args: argparse.Namespace) -> int:
    from lunaris.physics.ephemeris import EphemerisManager
    from lunaris.surrogate.st_lrps.data.spatial_cloud_generator import (
        load_coeffs_from_ssot,
        precompute_legendre_constants,
    )
    from lunaris.surrogate.st_lrps.evaluation._gravity_benchmark.types import build_base_config
    from lunaris.surrogate.st_lrps.evaluation.compare_gravity_models import parse_args

    bench = Path(args.benchmark_dir)
    meta = json.loads((bench / "run_metadata.json").read_text(encoding="utf-8-sig"))
    argv = [str(a) for a in meta["argv"]]
    # The recorded argv may start with the program path (".../lunaris-benchmark").
    while argv and not argv[0].startswith("-"):
        argv = argv[1:]
    cfg = build_base_config(parse_args(argv))
    ephem = EphemerisManager.from_time_and_spice(cfg.time, cfg.spice)
    series = bench / "benchmark_cache" / "models" / args.series
    ids = sorted(int(p.stem.split("_")[1]) for p in series.glob("scenario_*.npz"))[: args.orbits]
    if not ids:
        raise SystemExit(f"no scenario_*.npz under {series}")
    C, S, _ = load_coeffs_from_ssot(degree_max=args.ref_degree, gfc_path=None)
    tables = (C, S, *precompute_legendre_constants(args.ref_degree))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for sid in ids:
        t0 = time.time()
        z = np.load(series / f"scenario_{sid:06d}.npz", allow_pickle=True)
        t, r, v = (z[k].astype(np.float64) for k in ("t", "position", "velocity"))
        # Half-step grid for the adjoint's mid-step gradients; the quadrature
        # grid is every other sample and needs an odd count (Simpson).
        th, rh, vh = hermite_resample(t, r, v, 0.5 * args.step)
        keep = len(th) if (len(th) - 1) % 4 == 0 else len(th) - ((len(th) - 1) % 4)
        th, rh, vh = th[:keep], rh[:keep], vh[:keep]
        rot_h = np.stack([quat_to_matrix(ephem.get_inertial_to_fixed_rotation(float(x))) for x in th])
        r_fix_h = np.einsum("nij,nj->ni", rot_h, rh)
        g_in = np.einsum("nji,njk,nkl->nil", rot_h, _base_gradient(r_fix_h, tables, args.base_degree), rot_h)
        tt, rr, vv, rot, r_fix = th[0::2], rh[0::2], vh[0::2], rot_h[0::2], r_fix_h[0::2]
        _, a_true = _sh_eval(r_fix, tables, args.ref_degree, args.base_degree)
        c = along_track_unit(rr[-1], vv[-1])
        pv = adjoint_velocity_sensitivity(g_in, c, args.step)
        w_fix = np.einsum("nij,nj->ni", rot, pv) * simpson_weights(len(tt), args.step)[:, None]
        np.savez_compressed(
            out / f"orbit_{sid:06d}.npz",
            r_fix=r_fix, a_true=a_true, weight=w_fix.astype(np.float64),
            hp_km=float(np.min(np.linalg.norm(rr, axis=1)) - R_REF_M) / 1e3,
        )
        rows.append({"sid": sid, "points": len(tt), "seconds": round(time.time() - t0, 1)})
        print(f"orbit {sid:4d}: {len(tt)} points ({time.time() - t0:.1f} s)", flush=True)
    (out / "proxy_manifest.json").write_text(json.dumps({
        "benchmark_dir": str(bench), "series": args.series, "step_s": args.step,
        "base_degree": args.base_degree, "ref_degree": args.ref_degree,
        "duration_s": float(tt[-1]), "orbits": rows,
        "quantity": "first-order along-track position error at the final time [m]",
    }, indent=1), encoding="utf-8")
    return 0


# --------------------------------------------------------------------------- #
# score
# --------------------------------------------------------------------------- #


def score_arrays(a_model: np.ndarray, a_true: np.ndarray, weight: np.ndarray) -> float:
    """Predicted along-track error [m] of one orbit."""
    return float(np.sum(weight * (np.asarray(a_model, dtype=np.float64) - a_true)))


def score_run(proxy: Path, run_dir: Path, device: str = "cuda") -> dict:
    """Per-orbit predicted along-track errors and their summary for one run."""
    from lunaris.surrogate.st_lrps.runtime.canonical_runtime import load_surrogate_force_model

    proxy = Path(proxy)
    manifest = json.loads((proxy / "proxy_manifest.json").read_text(encoding="utf-8"))
    rt = load_surrogate_force_model(Path(run_dir), device=device)
    rows = []
    for p in sorted(proxy.glob("orbit_*.npz")):
        z = np.load(p)
        a_nn = np.asarray(rt.predict_residual_accel(z["r_fix"]), dtype=np.float64).reshape(-1, 3)
        rows.append({"sid": int(p.stem.split("_")[1]), "hp_km": float(z["hp_km"]),
                     "along_track_m": score_arrays(a_nn, z["a_true"], z["weight"])})
    s = np.array([r["along_track_m"] for r in rows])
    summary = {
        "run_dir": str(run_dir), "proxy_dir": str(proxy), "orbits": len(rows),
        "duration_days": manifest["duration_s"] / 86400.0,
        "rms_along_track_m": float(math.sqrt(np.mean(s**2))),
        "median_abs_along_track_m": float(np.median(np.abs(s))),
        "p95_abs_along_track_m": float(np.percentile(np.abs(s), 95)),
        "max_abs_along_track_m": float(np.max(np.abs(s))),
    }
    return {"summary": summary, "orbits": rows}


def score(args: argparse.Namespace) -> int:
    report = score_run(Path(args.proxy), Path(args.run_dir), args.device)
    summary = report["summary"]
    out = Path(args.out) if args.out else Path(args.run_dir) / "evals" / "arc_proxy.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--benchmark-dir", required=True)
    b.add_argument("--series", default="sh200", help="reference series under benchmark_cache/models")
    b.add_argument("--out", required=True)
    b.add_argument("--orbits", type=int, default=32)
    b.add_argument("--step", type=float, default=5.0, help="resampling step [s]")
    b.add_argument("--base-degree", type=int, default=25)
    b.add_argument("--ref-degree", type=int, default=200)
    s = sub.add_parser("score")
    s.add_argument("--proxy", required=True)
    s.add_argument("--run-dir", required=True)
    s.add_argument("--device", default="cuda")
    s.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    return build(a) if a.cmd == "build" else score(a)


if __name__ == "__main__":
    raise SystemExit(main())
