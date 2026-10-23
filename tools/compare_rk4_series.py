"""Pairwise comparison of cached RK4 benchmark trajectories, under equal conditions.

Each series is one model's trajectory cache from a ``lunaris-benchmark
--gpu-batch-compare --cache-trajectories`` run: ``<output-dir>/benchmark_cache/
models/<model>/scenario_NNNNNN.npz`` (``t``, ``position``, ``velocity`` in the
benchmark's frame, metres and seconds, plus metadata such as ``rk4_dt_s``).
A pair ``A,B`` takes ``A - B`` on every scenario both series hold, on identical
time grids, and only when both used the same RK4 step: the integrator error then
cancels to first order and the difference is the model difference. The
along-track/cross-track split uses B's state (RIC frame of the reference).

Typical pairs for an ST-LRPS model (series names are free; these are the ones of
the 2026-09-23 Plan D study)::

    A  stlrps vs sh200 : surrogate model error
    B  sh25   vs sh200 : what the degree-25 base alone misses
    C  sh25   vs cpu_sh25 : GPU/CPU float64 parity
    D  stlrps vs sh25  : the learned residual's effect on the orbit

Usage::

    python tools/compare_rk4_series.py --out OUT \\
        --series stlrps=RUN_GPU/benchmark_cache/models/st_lrps \\
        --series sh25=RUN_GPU/benchmark_cache/models/sh25 \\
        --series sh200=RUN_CPU/benchmark_cache/models/sh200 \\
        --pair A_model_error=stlrps,sh200 --pair B_base_deficiency=sh25,sh200

Writes ``summary.json`` (per pair: n, final/RMS position error mean, median,
P95, max [m], mean RIC RMS [m], final velocity error [mm/s], the common RK4 step),
``per_scenario.csv`` and ``ensemble_rms_vs_time.npz`` (``t_s`` plus one RMS
curve per pair). No figure: plot from these files.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

_META_KEYS = ("device", "dtype", "rk4_dt_s", "integrator", "hp_km", "ha_km", "e", "inc_deg", "backend", "frame_mode")


def load(series_dir: Path, sid: int):
    """Return ``(t, position, velocity, meta)`` for one cached scenario, or None."""
    p = series_dir / f"scenario_{sid:06d}.npz"
    if not p.exists():
        return None
    z = np.load(p, allow_pickle=True)
    meta = {k: (z[k].item() if k in z.files and z[k].shape == () else None) for k in _META_KEYS}
    return z["t"], z["position"].astype(np.float64), z["velocity"].astype(np.float64), meta


def ric(dpos: np.ndarray, pos_ref: np.ndarray, vel_ref: np.ndarray):
    """Project position differences on the reference's radial / in-track / cross-track axes."""
    r_hat = pos_ref / np.linalg.norm(pos_ref, axis=1, keepdims=True)
    h = np.cross(pos_ref, vel_ref)
    c_hat = h / np.linalg.norm(h, axis=1, keepdims=True)
    i_hat = np.cross(c_hat, r_hat)
    return (np.sum(dpos * r_hat, axis=1), np.sum(dpos * i_hat, axis=1), np.sum(dpos * c_hat, axis=1))


def scenario_ids(series_dir: Path) -> list[int]:
    return sorted(int(p.stem.split("_")[1]) for p in series_dir.glob("scenario_*.npz"))


def compare_pair(name: str, dir_a: Path, dir_b: Path):
    """Return ``(summary, rms_curve, t_s, per_rows)`` for ``A - B``; summary is None without overlap."""
    ids = sorted(set(scenario_ids(dir_a)) & set(scenario_ids(dir_b)))
    finals, rmss, rr, ii, cc, vfin, per_rows = [], [], [], [], [], [], []
    acc = t_ref = dt_common = None
    for sid in ids:
        A, B = load(dir_a, sid), load(dir_b, sid)
        if A is None or B is None:
            continue
        tA, pA, vA, mA = A
        tB, pB, vB, mB = B
        if tA.shape != tB.shape or not np.allclose(tA, tB):
            print(f"[{name}] scenario {sid}: time grids differ - skipped")
            continue
        dt_a, dt_b = mA["rk4_dt_s"], mB["rk4_dt_s"]
        if dt_a is None or dt_b is None or float(dt_a) != float(dt_b):
            raise SystemExit(f"[{name}] scenario {sid}: RK4 steps differ ({dt_a} vs {dt_b}); "
                             "the integrator error would not cancel")
        if dt_common is not None and float(dt_a) != dt_common:
            raise SystemExit(f"[{name}] scenario {sid}: RK4 step {dt_a} s differs from {dt_common} s "
                             "of the earlier scenarios")
        dt_common = float(dt_a)
        d = pA - pB
        dn = np.linalg.norm(d, axis=1)
        r_, i_, c_ = ric(d, pB, vB)
        finals.append(dn[-1])
        rmss.append(np.sqrt(np.mean(dn**2)))
        rr.append(np.sqrt(np.mean(r_**2)))
        ii.append(np.sqrt(np.mean(i_**2)))
        cc.append(np.sqrt(np.mean(c_**2)))
        vfin.append(np.linalg.norm((vA - vB)[-1]))
        acc = dn**2 if acc is None else acc + dn**2
        t_ref = tA
        per_rows.append({"pair": name, "scenario_id": sid, "hp_km": mB["hp_km"], "ha_km": mB["ha_km"],
                         "e": mB["e"], "inc_deg": mB["inc_deg"], "final_pos_err_m": dn[-1],
                         "rms_pos_err_m": rmss[-1], "max_pos_err_m": dn.max(), "radial_rms_m": rr[-1],
                         "along_rms_m": ii[-1], "cross_rms_m": cc[-1], "final_vel_err_mms": 1e3 * vfin[-1],
                         "dev_a": mA["device"], "dev_b": mB["device"]})
    if not finals:
        return None, None, None, []

    def q(x, p):
        return float(np.percentile(x, p))

    summary = {
        "n_scenarios": len(finals),
        "rk4_dt_s": dt_common,
        "final_pos_err_m": {"mean": float(np.mean(finals)), "median": q(finals, 50), "p95": q(finals, 95),
                            "max": float(np.max(finals))},
        "rms_pos_err_m": {"mean": float(np.mean(rmss)), "median": q(rmss, 50), "p95": q(rmss, 95),
                          "max": float(np.max(rmss))},
        "ric_rms_m": {"radial": float(np.mean(rr)), "along": float(np.mean(ii)), "cross": float(np.mean(cc))},
        "final_vel_err_mms": {"mean": 1e3 * float(np.mean(vfin)), "max": 1e3 * float(np.max(vfin))},
    }
    return summary, np.sqrt(acc / len(finals)), t_ref, per_rows


def _named(values: list[str], flag: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in values:
        name, sep, value = item.partition("=")
        if not sep or not name or not value:
            raise SystemExit(f"{flag} {item!r}: expected NAME=VALUE")
        if name in out:
            raise SystemExit(f"{flag} {name!r} given twice")
        out[name] = value
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--series", action="append", default=[], metavar="NAME=DIR",
                    help="a model's trajectory cache: <benchmark output>/benchmark_cache/models/<model>")
    ap.add_argument("--pair", action="append", default=[], metavar="NAME=A,B",
                    help="report A - B under this name (A and B are --series names)")
    ap.add_argument("--out", required=True, help="output directory")
    args = ap.parse_args(argv)

    series = {k: Path(v) for k, v in _named(args.series, "--series").items()}
    for name, path in series.items():
        if not path.is_dir():
            raise SystemExit(f"--series {name}: {path} is not a directory")
    pairs = []
    for name, value in _named(args.pair, "--pair").items():
        a, sep, b = value.partition(",")
        if not sep or a not in series or b not in series:
            raise SystemExit(f"--pair {name}: expected two --series names as A,B, got {value!r}")
        pairs.append((name, a, b))
    if not pairs:
        raise SystemExit("give at least one --pair")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict] = {}
    curves: dict[str, np.ndarray] = {}
    rows: list[dict] = []
    for name, a, b in pairs:
        result, curve, t_s, per_rows = compare_pair(name, series[a], series[b])
        if result is None:
            print(f"[{name}] no common scenarios ({a} vs {b}) - skipped")
            continue
        summary[name] = {"series": [a, b], **result}
        curves[name] = curve
        curves["t_s"] = t_s
        rows.extend(per_rows)
        f = result["final_pos_err_m"]
        ric_m = result["ric_rms_m"]
        print(f"[{name}] {a} vs {b}: n={result['n_scenarios']}, RK4 dt {result['rk4_dt_s']} s | final pos err "
              f"mean {f['mean']:.3g} m, median {f['median']:.3g} m, p95 {f['p95']:.3g} m, max {f['max']:.3g} m | "
              f"RIC rms {ric_m['radial']:.3g}/{ric_m['along']:.3g}/{ric_m['cross']:.3g} m")
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    if rows:
        with open(out / "per_scenario.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    if curves:
        np.savez(out / "ensemble_rms_vs_time.npz", **curves)
    print("wrote", out)
    return 0 if summary else 1


if __name__ == "__main__":
    sys.exit(main())
