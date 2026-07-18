"""Quantify spherical-harmonic degree-band shares of the perturbation field.

Answers, from the *actual* model coefficients (not a Kaula rule): what fraction
of the non-spherical gravity acceleration at a given altitude comes from the
degree band 61..100, and from the tail above 100? These numbers put the
"SH60 representation ceiling / SH100 numerical reference" framing of the
ST-LRPS benchmarks on a quantitative footing (see
``docs/ST_LRPS_VALIDATION_HYGIENE.md``).

Method: sample quasi-uniform random directions on the sphere at radius
``r_ref + h``, evaluate the serial Kahan kernel at truncation degrees
(60, 100, N_max) per point, and form vector band differences:

* ``pert_total``  = a(N_max) − a_pointmass   (all non-spherical signal)
* ``band_2_60``   = a(60)    − a_pointmass
* ``band_61_100`` = a(100)   − a(60)
* ``tail_>100``   = a(N_max) − a(100)

Shares are RMS-over-points of the band vector norm divided by the RMS of
``pert_total``. All evaluations reuse the production ``GravityModel`` kernel,
so the result inherits its validated normalization and pole-safe guards.

Usage::

    python -m validation.gravity.band_share_analysis \
        --altitudes-km 80,100 --max-degree 200 --n-points 1000 --seed 20260718

The output is a small Markdown table plus an optional ``--out-json`` payload
with full provenance (model path, seed, point count, degrees).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from lunaris.common.lunar_data import resolve_lunar_gravity_path
from lunaris.physics.spherical_harmonics import GravityModel

EVIDENCE_SCHEMA_VERSION = "lunaris_gravity_band_share_v1"
COEFFICIENT_NORMALIZATION = "fully_normalized_real_4pi"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_commit_sha() -> str | None:
    """Return the source commit when Git metadata is available."""
    repo_root = Path(__file__).resolve().parents[2]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = result.stdout.strip()
    return value if len(value) == 40 else None


def build_evidence_payload(
    *,
    gravity_file: Path,
    model: GravityModel,
    band_edges: tuple[int, int],
    n_points: int,
    seed: int,
    results: list[dict[str, Any]],
    repo_commit_sha: str | None = None,
    generated_at_utc: str | None = None,
) -> dict[str, Any]:
    """Build the self-contained, versioned evidence record written to JSON."""
    path = gravity_file.resolve()
    repo_root = Path(__file__).resolve().parents[2]
    try:
        path_hint = path.relative_to(repo_root).as_posix()
    except ValueError:
        path_hint = path.name
    return {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "generated_at_utc": generated_at_utc or datetime.now(timezone.utc).isoformat(),
        "repo_commit_sha": repo_commit_sha if repo_commit_sha is not None else _repo_commit_sha(),
        "method": {
            "sampler": "seeded_gaussian_normalized_directions",
            "acceleration_evaluator": "GravityModel.accel_fixed_serial_kahan",
            "share_denominator": "rms_norm_of_a_nmax_minus_point_mass",
        },
        "gravity_model": {
            "path_hint": path_hint,
            "file_name": path.name,
            "sha256": _sha256_file(path),
            "normalization": COEFFICIENT_NORMALIZATION,
            "mu_m3_s2": float(model.mu),
            "reference_radius_m": float(model.r_ref),
            "evaluated_max_degree": int(model.max_degree),
        },
        "band_edges": [int(band_edges[0]), int(band_edges[1])],
        "n_points_per_altitude": int(n_points),
        "seed": int(seed),
        "results": results,
    }


def _sample_directions(n_points: int, seed: int) -> np.ndarray:
    """Quasi-uniform unit vectors on S^2 (Gaussian-normalized, seeded)."""
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n_points, 3))
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    # Degenerate zero-norm draws are astronomically unlikely; guard anyway.
    valid = norms[:, 0] > 1e-12
    return v[valid] / norms[valid]


def compute_band_shares(
    model: GravityModel,
    *,
    altitude_km: float,
    n_points: int,
    seed: int,
    band_edges: tuple[int, int] = (60, 100),
) -> dict[str, Any]:
    """RMS acceleration of each degree band and its share of the perturbation."""
    lo, hi = band_edges
    n_max = model.max_degree
    if not (2 <= lo < hi <= n_max):
        raise ValueError(f"band edges {band_edges} must satisfy 2 <= lo < hi <= {n_max}")

    r = model.r_ref + altitude_km * 1000.0
    dirs = _sample_directions(n_points, seed)
    ws = model.make_workspace()

    sq = {"pert_total": 0.0, f"band_2_{lo}": 0.0, f"band_{lo + 1}_{hi}": 0.0, f"tail_gt_{hi}": 0.0}
    for u in dirs:
        pos = r * u
        a_lo = model.accel_fixed(pos, degree=lo, workspace=ws)
        a_hi = model.accel_fixed(pos, degree=hi, workspace=ws)
        a_full = model.accel_fixed(pos, degree=n_max, workspace=ws)
        a_pm = -(model.mu / (r * r)) * u

        sq["pert_total"] += float(np.sum((a_full - a_pm) ** 2))
        sq[f"band_2_{lo}"] += float(np.sum((a_lo - a_pm) ** 2))
        sq[f"band_{lo + 1}_{hi}"] += float(np.sum((a_hi - a_lo) ** 2))
        sq[f"tail_gt_{hi}"] += float(np.sum((a_full - a_hi) ** 2))

    n_used = int(dirs.shape[0])
    rms = {k: math.sqrt(v / n_used) for k, v in sq.items()}
    total = rms["pert_total"]
    shares = {k: (rms[k] / total if total > 0.0 else float("nan")) for k in rms}
    return {
        "altitude_km": float(altitude_km),
        "radius_m": float(r),
        "n_points": n_used,
        "rms_m_s2": rms,
        "share_of_pert_total": shares,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gravity-file", type=Path, default=None,
                        help="SHADR coefficient file (default: bundled JGGRX 1800F)")
    parser.add_argument("--altitudes-km", type=str, default="80,100")
    parser.add_argument("--max-degree", type=int, default=200,
                        help="Full-field truncation used as the perturbation total")
    parser.add_argument("--band", type=str, default="60,100",
                        help="Band edges lo,hi (default 60,100)")
    parser.add_argument("--n-points", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--out-json", type=Path, default=None)
    args = parser.parse_args(argv)

    gravity_file = resolve_lunar_gravity_path(args.gravity_file)
    lo, hi = (int(v) for v in args.band.split(","))
    altitudes = [float(v) for v in args.altitudes_km.split(",")]

    model = GravityModel.from_file(str(gravity_file), requested_degree=args.max_degree)

    results = [
        compute_band_shares(
            model,
            altitude_km=alt,
            n_points=args.n_points,
            seed=args.seed,
            band_edges=(lo, hi),
        )
        for alt in altitudes
    ]

    band_key = f"band_{lo + 1}_{hi}"
    tail_key = f"tail_gt_{hi}"
    print(f"# Degree-band shares of non-spherical acceleration (N_max={model.max_degree})")
    print(f"model: {gravity_file}")
    print(f"points/altitude: {args.n_points}  seed: {args.seed}\n")
    print(f"| alt (km) | pert RMS (m/s^2) | {lo + 1}-{hi} RMS | {lo + 1}-{hi} share | >{hi} RMS | >{hi} share |")
    print("|---|---|---|---|---|---|")
    for res in results:
        rms = res["rms_m_s2"]
        share = res["share_of_pert_total"]
        print(
            f"| {res['altitude_km']:.0f} | {rms['pert_total']:.3e} "
            f"| {rms[band_key]:.3e} | {share[band_key]:.4%} "
            f"| {rms[tail_key]:.3e} | {share[tail_key]:.4%} |"
        )

    if args.out_json is not None:
        payload = build_evidence_payload(
            gravity_file=gravity_file,
            model=model,
            band_edges=(lo, hi),
            n_points=args.n_points,
            seed=args.seed,
            results=results,
        )
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
