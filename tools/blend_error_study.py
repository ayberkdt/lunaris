"""Quantitative error study: altitude-aware SH degree blend vs fixed reference.

An external review asked for a *quantitative* error study of the adaptive
dual-fidelity blend (``GravityModel.accel_adaptive`` /
``sh_accel_adaptive_blend_numba``) against a fixed high-degree reference, to back
the policy decision that reference / paper runs use a single fixed degree while
the blend is a speed option for exploratory use.

This script isolates the gravity-model difference: it compares, on a synthetic
high-degree field (no external GRAIL file needed), three evaluations across an
altitude sweep at fixed latitude/longitude —

* ``fixed``       : full max-degree acceleration (the reference),
* ``blend``       : the smooth altitude-aware blend (near=max, far=coarse),
* ``hard_switch`` : a discontinuous degree switch at the mid altitude,

and reports, per altitude:

* relative acceleration error vs the fixed reference, and
* a finite-difference "jerk proxy" |d|a|/dh| that exposes the discontinuity a
  hard degree switch introduces and the blend is designed to smooth.

A short fixed-frame RK4 orbit (same frame for blend and fixed, so only the
gravity model differs) gives an integrated end-of-arc position error.

Outputs (written under ``outputs/blend_study/``): ``blend_error.csv``,
``blend_error.png`` (if matplotlib is present), and ``blend_error_summary.md``.

Run: ``python tools/blend_error_study.py``
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

from lunaris.physics.spherical_harmonics import GravityModel

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "outputs" / "blend_study"

# Synthetic field: modest degree for a fast, deterministic study.
DEG = 64
DEG_COARSE = 16
MU = 4.902800066e12  # m^3/s^2 (Moon-like)
R_REF = 1738000.0    # m
LAT_DEG = 25.0
LON_DEG = 37.0

# Altitude sweep and blend transition band (km).
ALT_MIN_KM, ALT_MAX_KM, N_ALT = 5.0, 400.0, 200
ALT_NEAR_KM, ALT_FAR_KM = 50.0, 250.0


def _synthetic_model() -> GravityModel:
    rng = np.random.default_rng(20260715)
    n_idx = np.arange(DEG + 1, dtype=np.float64)
    sigma = np.zeros(DEG + 1)
    sigma[2:] = 1.0e-4 / n_idx[2:] ** 2  # Kaula-like decay
    c = rng.standard_normal((DEG + 1, DEG + 1)) * sigma[:, None]
    s = rng.standard_normal((DEG + 1, DEG + 1)) * sigma[:, None]
    keep = np.tril(np.ones((DEG + 1, DEG + 1), dtype=bool))
    c[~keep] = 0.0
    s[~keep] = 0.0
    s[:, 0] = 0.0
    c[0, 0] = 1.0
    c[1, :2] = 0.0
    s[1, :2] = 0.0
    return GravityModel.from_arrays(
        degree_max=DEG, r_ref=R_REF, mu=MU, c_coeffs_full=c, s_coeffs_full=s
    )


def _point(alt_m: float) -> np.ndarray:
    r = R_REF + alt_m
    phi, lam = math.radians(LAT_DEG), math.radians(LON_DEG)
    return np.array(
        [r * math.cos(phi) * math.cos(lam), r * math.cos(phi) * math.sin(lam), r * math.sin(phi)]
    )


def _hard_switch_accel(model: GravityModel, p: np.ndarray, alt_m: float) -> np.ndarray:
    """Discontinuous degree switch at the mid transition altitude."""
    alt_mid = 0.5 * (ALT_NEAR_KM + ALT_FAR_KM) * 1000.0
    degree = DEG if alt_m <= alt_mid else DEG_COARSE
    return model.accel_fixed(p, degree=degree)


def _blend_accel(model: GravityModel, p: np.ndarray) -> np.ndarray:
    return model.accel_adaptive(
        p,
        degree_far=DEG_COARSE,
        degree_near=DEG,
        alt_far=ALT_FAR_KM * 1000.0,
        alt_near=ALT_NEAR_KM * 1000.0,
        degree_step=8,
    )


def _rk4_orbit_end_error(model: GravityModel, use_blend: bool, steps: int = 4000) -> float:
    """Fixed-frame RK4 orbit; returns end position vs the fixed-degree reference.

    Both the test and reference integrations share this frame, so the reported
    difference isolates the blend-vs-fixed gravity-model discrepancy, not any
    frame effect.
    """

    def accel(r: np.ndarray) -> np.ndarray:
        if use_blend:
            return _blend_accel(model, r)
        return model.accel_fixed(r)

    # Circular-ish initial state at 200 km altitude.
    r0 = _point(200_000.0)
    rmag = float(np.linalg.norm(r0))
    v_circ = math.sqrt(MU / rmag)
    # Velocity perpendicular to r0, in the x-y plane.
    radial = r0 / rmag
    v_dir = np.cross(np.array([0.0, 0.0, 1.0]), radial)
    v_dir /= np.linalg.norm(v_dir)
    y = np.concatenate([r0, v_circ * v_dir])
    period = 2.0 * math.pi * math.sqrt(rmag**3 / MU)
    dt = period / steps

    def deriv(state: np.ndarray) -> np.ndarray:
        return np.concatenate([state[3:6], accel(state[0:3])])

    for _ in range(steps):
        k1 = deriv(y)
        k2 = deriv(y + 0.5 * dt * k1)
        k3 = deriv(y + 0.5 * dt * k2)
        k4 = deriv(y + dt * k3)
        y = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return y[0:3]


def run_study() -> dict[str, float]:
    model = _synthetic_model()
    alts_km = np.linspace(ALT_MIN_KM, ALT_MAX_KM, N_ALT)
    rows: list[dict[str, float]] = []
    for alt_km in alts_km:
        alt_m = float(alt_km) * 1000.0
        p = _point(alt_m)
        a_ref = model.accel_fixed(p)
        a_blend = _blend_accel(model, p)
        a_hard = _hard_switch_accel(model, p, alt_m)
        ref_norm = float(np.linalg.norm(a_ref))
        rel_blend = float(np.linalg.norm(a_blend - a_ref)) / ref_norm
        rel_hard = float(np.linalg.norm(a_hard - a_ref)) / ref_norm
        rows.append(
            {
                "alt_km": float(alt_km),
                "a_ref_mag": ref_norm,
                "a_blend_mag": float(np.linalg.norm(a_blend)),
                "a_hard_mag": float(np.linalg.norm(a_hard)),
                "rel_err_blend": rel_blend,
                "rel_err_hard": rel_hard,
            }
        )

    # Discontinuity proxy: gradient of the *residual* (rel error vs the fixed
    # reference) per km. The monopole falloff cancels in the residual, so this
    # isolates the transition: a hard degree switch spikes here, the blend does
    # not.
    def residual_jerk(key: str) -> np.ndarray:
        mag = np.array([r[key] for r in rows])
        return np.abs(np.gradient(mag, alts_km))

    jerk_blend = residual_jerk("rel_err_blend")
    jerk_hard = residual_jerk("rel_err_hard")
    for i, r in enumerate(rows):
        r["jerk_blend"] = float(jerk_blend[i])
        r["jerk_hard"] = float(jerk_hard[i])

    # Integrated orbit-level end-position discrepancy (blend vs fixed reference).
    end_blend = _rk4_orbit_end_error(model, use_blend=True)
    end_fixed = _rk4_orbit_end_error(model, use_blend=False)
    orbit_end_pos_diff_m = float(np.linalg.norm(end_blend - end_fixed))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "blend_error.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "max_rel_err_blend": max(r["rel_err_blend"] for r in rows),
        "max_rel_err_hard": max(r["rel_err_hard"] for r in rows),
        "max_jerk_blend": float(jerk_blend.max()),
        "max_jerk_hard": float(jerk_hard.max()),
        "orbit_end_pos_diff_m": orbit_end_pos_diff_m,
    }

    _write_summary_md(summary)
    _maybe_plot(rows)
    return summary


def _write_summary_md(summary: dict[str, float]) -> None:
    md = OUT_DIR / "blend_error_summary.md"
    lines = [
        "# Adaptive SH blend vs fixed-degree reference — error study",
        "",
        f"Synthetic degree-{DEG} field (coarse={DEG_COARSE}), altitude "
        f"{ALT_MIN_KM:.0f}-{ALT_MAX_KM:.0f} km at lat={LAT_DEG}, lon={LON_DEG}. "
        f"Transition band {ALT_NEAR_KM:.0f}-{ALT_FAR_KM:.0f} km.",
        "",
        "| Metric | Blend | Hard switch |",
        "|---|---|---|",
        f"| Max rel. accel error vs fixed-{DEG} | "
        f"{summary['max_rel_err_blend']:.3e} | {summary['max_rel_err_hard']:.3e} |",
        f"| Max residual-gradient (discontinuity proxy, per km) | "
        f"{summary['max_jerk_blend']:.3e} | {summary['max_jerk_hard']:.3e} |",
        "",
        f"Integrated orbit end-position difference (blend vs fixed, one period): "
        f"**{summary['orbit_end_pos_diff_m']:.3f} m**.",
        "",
        "Interpretation: the blend keeps the acceleration continuous through the "
        "transition band (much smaller jerk proxy than a hard degree switch), but "
        "it is still a two-degree approximation of the full field, so a reference "
        "/ paper run must use a single fixed degree (enforced fail-closed under "
        "paper-safe / benchmark / strict). The blend is a speed option for "
        "exploratory use only.",
    ]
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _maybe_plot(rows: list[dict[str, float]]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    alts = [r["alt_km"] for r in rows]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 7), sharex=True)
    ax1.semilogy(alts, [r["rel_err_blend"] for r in rows], label="blend")
    ax1.semilogy(alts, [r["rel_err_hard"] for r in rows], label="hard switch")
    ax1.axvspan(ALT_NEAR_KM, ALT_FAR_KM, color="gray", alpha=0.15, label="transition band")
    ax1.set_ylabel(f"rel. accel error vs fixed-{DEG}")
    ax1.legend()
    ax1.set_title("Adaptive SH blend vs fixed-degree reference")
    ax2.semilogy(alts, [r["jerk_blend"] for r in rows], label="blend")
    ax2.semilogy(alts, [r["jerk_hard"] for r in rows], label="hard switch")
    ax2.axvspan(ALT_NEAR_KM, ALT_FAR_KM, color="gray", alpha=0.15)
    ax2.set_xlabel("altitude [km]")
    ax2.set_ylabel("|d(rel err)/dh| [per km]")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "blend_error.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    result = run_study()
    print("Blend error study complete. Summary:")
    for key, value in result.items():
        print(f"  {key}: {value:.6g}")
    print(f"Artifacts written to: {OUT_DIR}")
