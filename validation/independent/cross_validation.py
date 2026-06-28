"""Trajectory cross-validation harness: Lunaris vs an independent reference.

Why this exists
---------------
The in-tree benchmarks integrate an orbit with Lunaris and compare it against
*Lunaris* quantities, so a bug in the propagator can hide in both sides. This
harness compares a full Lunaris propagation against a reference produced by a
**deliberately independent** path, and reports the difference in the RIC
(radial / in-track / cross-track) frame -- the standard, paper-defensible way to
state orbit-propagation agreement and the language GMAT/STK validation uses.

Two reference modes
-------------------
1. ``kepler`` -- a closed-form two-body propagation via the classical
   Kepler-equation (eccentric-anomaly) formulation, implemented standalone here.
   It shares no code with the Lunaris integrator, so for a point-mass field it is
   an *exact* reference and isolates pure integrator error. Always available
   (numpy only).

2. ``gmat`` -- a GMAT (or any external tool) ephemeris ReportFile loaded from
   disk. GMAT is not required to be installed; you point the harness at a report
   it produced for the same initial state and force model. The file is parsed
   into ``(t, x, y, z, vx, vy, vz)`` and compared in RIC. This is how you back a
   "GMAT-grade or better" claim with a measured number rather than an assertion.

The reference here intentionally relies only on generic orbital-element /
orbital-element math, never on the Lunaris integrator or force kernels under
test. See :mod:`validation.independent` for the correlated-error rationale.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from lunaris.analysis.perturbation_budget.sampling import ric_frame

__all__ = [
    "propagate_kepler_analytic",
    "ric_error_report",
    "load_external_ephemeris",
    "Scenario",
    "cross_validate_integrator",
]


# =============================================================================
# 1) Independent reference: classical-element two-body propagation (elliptic)
# =============================================================================
#
# All bound orbits use the classical Kepler-equation path (solve M = E - e sinE
# for the eccentric anomaly, with guaranteed Newton convergence for e < 1). This
# is far more robust than the universal-variable iteration, which can converge to
# the wrong branch at intermediate anomalies. Hyperbolic/parabolic references are
# not needed for lunar orbits and raise a clear error.

_TWO_PI = 2.0 * math.pi


def _rv_to_elements(
    r_vec: np.ndarray, v_vec: np.ndarray, mu: float
) -> tuple[float, float, float, float, float, float]:
    """Inertial (r, v) -> classical elements (a, e, inc, raan, argp, nu) [SI, rad].

    Standalone Vallado RV2COE with the standard degenerate fallbacks (circular ->
    argument of latitude; equatorial -> measure from the x-axis), so it is valid
    for the inclined-circular and equatorial cases used by the harness.
    """
    eps = 1e-11
    r = float(np.linalg.norm(r_vec))
    v = float(np.linalg.norm(v_vec))
    vr = float(np.dot(r_vec, v_vec)) / r

    h_vec = np.cross(r_vec, v_vec)
    h = float(np.linalg.norm(h_vec))
    inc = math.acos(max(-1.0, min(1.0, h_vec[2] / h)))

    k = np.array([0.0, 0.0, 1.0])
    n_vec = np.cross(k, h_vec)
    n = float(np.linalg.norm(n_vec))

    e_vec = ((v * v - mu / r) * r_vec - r * vr * v_vec) / mu
    e = float(np.linalg.norm(e_vec))

    energy = 0.5 * v * v - mu / r
    if abs(energy) < 1e-20:
        raise ValueError("parabolic orbit (energy ~ 0) is not supported by the elliptic reference")
    a = -mu / (2.0 * energy)
    if e >= 1.0:
        raise ValueError(f"hyperbolic orbit (e={e:.3f}) is not supported by the elliptic reference")

    # RAAN
    if n > eps:
        raan = math.acos(max(-1.0, min(1.0, n_vec[0] / n)))
        if n_vec[1] < 0.0:
            raan = _TWO_PI - raan
    else:
        raan = 0.0  # equatorial: node undefined, measure from +x

    # Argument of periapsis + true anomaly, with circular/equatorial fallbacks.
    if e > eps and n > eps:                      # inclined elliptic
        argp = math.acos(max(-1.0, min(1.0, float(np.dot(n_vec, e_vec)) / (n * e))))
        if e_vec[2] < 0.0:
            argp = _TWO_PI - argp
        nu = math.acos(max(-1.0, min(1.0, float(np.dot(e_vec, r_vec)) / (e * r))))
        if vr < 0.0:
            nu = _TWO_PI - nu
    elif e > eps:                                # equatorial elliptic
        argp = math.acos(max(-1.0, min(1.0, e_vec[0] / e)))
        if e_vec[1] < 0.0:
            argp = _TWO_PI - argp
        nu = math.acos(max(-1.0, min(1.0, float(np.dot(e_vec, r_vec)) / (e * r))))
        if vr < 0.0:
            nu = _TWO_PI - nu
    elif n > eps:                                # inclined circular: arg. of latitude
        argp = 0.0
        nu = math.acos(max(-1.0, min(1.0, float(np.dot(n_vec, r_vec)) / (n * r))))
        if r_vec[2] < 0.0:
            nu = _TWO_PI - nu
    else:                                        # circular equatorial: true longitude
        argp = 0.0
        nu = math.acos(max(-1.0, min(1.0, r_vec[0] / r)))
        if r_vec[1] < 0.0:
            nu = _TWO_PI - nu
    return a, e, inc, raan, argp, nu


def _solve_kepler(mean_anom: float, e: float, tol: float = 1e-14, max_iter: int = 100) -> float:
    """Solve M = E - e sin E for the eccentric anomaly E (elliptic, e < 1)."""
    m = (mean_anom + math.pi) % _TWO_PI - math.pi  # wrap to [-pi, pi)
    e_anom = m if e < 0.8 else math.pi * math.copysign(1.0, m)
    for _ in range(int(max_iter)):
        f = e_anom - e * math.sin(e_anom) - m
        fp = 1.0 - e * math.cos(e_anom)
        d = f / fp
        e_anom -= d
        if abs(d) < tol:
            break
    return e_anom


def _elements_to_rv(
    a: float, e: float, inc: float, raan: float, argp: float, nu: float, mu: float
) -> tuple[np.ndarray, np.ndarray]:
    """Classical elements -> inertial (r, v) via the perifocal frame [SI, rad]."""
    p = a * (1.0 - e * e)
    r = p / (1.0 + e * math.cos(nu))
    r_pqw = np.array([r * math.cos(nu), r * math.sin(nu), 0.0])
    s_mu_p = math.sqrt(mu / p)
    v_pqw = np.array([-s_mu_p * math.sin(nu), s_mu_p * (e + math.cos(nu)), 0.0])

    co, so = math.cos(raan), math.sin(raan)
    ci, si = math.cos(inc), math.sin(inc)
    cw, sw = math.cos(argp), math.sin(argp)
    # 3-1-3 (raan, inc, argp) perifocal -> inertial rotation matrix.
    rot = np.array([
        [co * cw - so * sw * ci, -co * sw - so * cw * ci, so * si],
        [so * cw + co * sw * ci, -so * sw + co * cw * ci, -co * si],
        [sw * si, cw * si, ci],
    ])
    return rot @ r_pqw, rot @ v_pqw


def propagate_kepler_analytic(
    y0: np.ndarray,
    t_s: np.ndarray,
    mu_m3s2: float,
    *,
    t0_s: float = 0.0,
) -> np.ndarray:
    """Closed-form two-body propagation of a single elliptic state over a grid.

    Converts ``y0`` to classical elements once, advances the mean anomaly
    analytically to each time, solves Kepler's equation, and maps back to
    inertial ``(r, v)``. Independent of the Lunaris integrator: for a point-mass
    field this is the exact solution and isolates pure integrator error.

    Parameters
    ----------
    y0 : ndarray, shape (>=6,)
        Initial state ``[x, y, z, vx, vy, vz]`` in SI at ``t0_s``.
    t_s : ndarray, shape (N,)
        Absolute times (seconds) at which to evaluate the state.
    mu_m3s2 : float
        Central-body gravitational parameter [m^3/s^2].

    Returns
    -------
    ndarray, shape (N, 6)
        State rows ``[x, y, z, vx, vy, vz]`` at each ``t_s``.
    """
    y0 = np.asarray(y0, dtype=np.float64).reshape(-1)
    t_s = np.asarray(t_s, dtype=np.float64).reshape(-1)
    mu = float(mu_m3s2)
    if y0.size < 6:
        raise ValueError("y0 must have at least 6 components [r(3), v(3)]")
    if mu <= 0.0 or not math.isfinite(mu):
        raise ValueError(f"mu must be > 0, got {mu_m3s2!r}")

    a, e, inc, raan, argp, nu0 = _rv_to_elements(y0[0:3], y0[3:6], mu)
    n = math.sqrt(mu / (a * a * a))  # mean motion

    # Initial mean anomaly from the initial true anomaly.
    e_anom0 = 2.0 * math.atan2(math.sqrt(1.0 - e) * math.sin(nu0 / 2.0),
                               math.sqrt(1.0 + e) * math.cos(nu0 / 2.0))
    m0 = e_anom0 - e * math.sin(e_anom0)

    out = np.empty((t_s.size, 6), dtype=np.float64)
    for i, t in enumerate(t_s):
        m = m0 + n * (float(t) - float(t0_s))
        e_anom = _solve_kepler(m, e)
        nu = 2.0 * math.atan2(math.sqrt(1.0 + e) * math.sin(e_anom / 2.0),
                              math.sqrt(1.0 - e) * math.cos(e_anom / 2.0))
        r_vec, v_vec = _elements_to_rv(a, e, inc, raan, argp, nu, mu)
        out[i, 0:3] = r_vec
        out[i, 3:6] = v_vec
    return out


# =============================================================================
# 2) RIC error report
# =============================================================================

def ric_error_report(
    y_test: np.ndarray,
    y_ref: np.ndarray,
    *,
    label: str = "reference",
) -> dict[str, Any]:
    """RIC-frame position/velocity error of ``y_test`` against ``y_ref``.

    Both inputs are ``(N, >=6)`` state histories on the *same* time grid. The
    position difference at each sample is projected onto the reference orbit's
    radial / in-track / cross-track unit vectors (the RIC frame is built from the
    reference state, the trusted side). Returns per-axis and total RMS / max
    statistics in meters (position) and m/s (velocity-magnitude).
    """
    a = np.asarray(y_test, dtype=np.float64)
    b = np.asarray(y_ref, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] < 6:
        raise ValueError("y_test and y_ref must have identical (N, >=6) shapes")

    n = a.shape[0]
    ric = np.empty((n, 3), dtype=np.float64)
    for i in range(n):
        dr = a[i, 0:3] - b[i, 0:3]
        r_hat, t_hat, h_hat = ric_frame(b[i, 0:3], b[i, 3:6])
        ric[i, 0] = float(dr @ r_hat)
        ric[i, 1] = float(dr @ t_hat)
        ric[i, 2] = float(dr @ h_hat)

    pos_norm = np.linalg.norm(a[:, 0:3] - b[:, 0:3], axis=1)
    vel_norm = np.linalg.norm(a[:, 3:6] - b[:, 3:6], axis=1)

    def _rms(x: np.ndarray) -> float:
        return float(np.sqrt(np.mean(x * x)))

    return {
        "label": label,
        "n_samples": int(n),
        "pos_rms_m": _rms(pos_norm),
        "pos_max_m": float(np.max(pos_norm)),
        "pos_final_m": float(pos_norm[-1]),
        "vel_rms_m_s": _rms(vel_norm),
        "vel_max_m_s": float(np.max(vel_norm)),
        "ric_radial_rms_m": _rms(ric[:, 0]),
        "ric_intrack_rms_m": _rms(ric[:, 1]),
        "ric_crosstrack_rms_m": _rms(ric[:, 2]),
        "ric_radial_max_m": float(np.max(np.abs(ric[:, 0]))),
        "ric_intrack_max_m": float(np.max(np.abs(ric[:, 1]))),
        "ric_crosstrack_max_m": float(np.max(np.abs(ric[:, 2]))),
    }


# =============================================================================
# 3) External (GMAT/STK/SPICE) ephemeris ReportFile loader
# =============================================================================

def load_external_ephemeris(
    path: str | Path,
    *,
    length_unit_m: float = 1000.0,
    time_unit_s: float = 1.0,
    time_to_seconds_from_start: bool = True,
    columns: tuple[int, int, int, int, int, int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load a whitespace/CSV ephemeris report into ``(t_s, y)``.

    Designed for GMAT ReportFile output (default km, km/s) but works for any
    table whose rows are ``time x y z vx vy vz``. Header / comment lines (any
    line that does not parse as >=7 floats) are skipped. GMAT writes kilometers,
    so the default ``length_unit_m=1000.0`` converts to SI; pass ``1.0`` for a
    file already in meters.

    Parameters
    ----------
    columns : optional 7-tuple of int
        Zero-based column indices for ``(t, x, y, z, vx, vy, vz)``. If omitted,
        the first seven numeric columns are used in order.

    Returns
    -------
    (t_s, y) : (ndarray (N,), ndarray (N, 6))
        Times in seconds (relative to the first row if
        ``time_to_seconds_from_start``) and SI state rows.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"external ephemeris not found: {p}")

    rows: list[list[float]] = []
    sep = "," if p.suffix.lower() == ".csv" else None
    with p.open("r", encoding="utf-8", errors="ignore") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith(("#", "%", "//")):
                continue
            parts = line.split(sep) if sep else line.split()
            try:
                vals = [float(tok) for tok in parts]
            except ValueError:
                continue  # header / non-numeric line
            if len(vals) >= 7:
                rows.append(vals)

    if not rows:
        raise ValueError(f"no numeric 7+-column rows parsed from {p}")

    arr = np.asarray(rows, dtype=np.float64)
    idx = columns if columns is not None else (0, 1, 2, 3, 4, 5, 6)
    t = arr[:, idx[0]] * float(time_unit_s)
    if time_to_seconds_from_start:
        t = t - t[0]
    y = np.empty((arr.shape[0], 6), dtype=np.float64)
    y[:, 0:3] = arr[:, [idx[1], idx[2], idx[3]]] * float(length_unit_m)
    y[:, 3:6] = arr[:, [idx[4], idx[5], idx[6]]] * float(length_unit_m)
    return t, y


# =============================================================================
# 4) Scenario-driven integrator cross-validation
# =============================================================================

@dataclass(frozen=True)
class Scenario:
    """A circular/eccentric lunar orbit cross-validation case."""

    name: str
    alt_km: float          # periapsis altitude above the reference radius
    ecc: float = 0.0
    inc_deg: float = 0.0
    duration_s: float = 6.0 * 3600.0
    dt_s: float = 60.0


def _initial_state(scn: Scenario, mu: float, r_ref_m: float) -> np.ndarray:
    """Build an inertial initial state at periapsis for a scenario."""
    r_p = r_ref_m + scn.alt_km * 1000.0
    a = r_p / (1.0 - scn.ecc)
    # Speed at periapsis from vis-viva.
    v_p = math.sqrt(mu * (2.0 / r_p - 1.0 / a))
    inc = math.radians(scn.inc_deg)
    # Periapsis on +x; velocity in the orbit plane inclined about the x-axis.
    r_vec = np.array([r_p, 0.0, 0.0], dtype=np.float64)
    v_vec = np.array([0.0, v_p * math.cos(inc), v_p * math.sin(inc)], dtype=np.float64)
    return np.concatenate([r_vec, v_vec])


def cross_validate_integrator(
    scn: Scenario,
    mu_m3s2: float,
    r_ref_m: float,
    *,
    method: str = "DOP853",
    rtol: float = 1e-10,
    atol: float = 1e-12,
    atol_pos: float | None = None,
    atol_vel: float | None = None,
) -> dict[str, Any]:
    """Numerically integrate the two-body problem and compare to analytic Kepler.

    This isolates pure integrator error (the force model is an exact point mass),
    which is exactly what a "GMAT-grade integrator" claim needs to be measured
    against. Uses the same ``atol`` vector convention as the Lunaris propagator so
    scalar-vs-vector ``atol`` can be compared directly.
    """
    from scipy.integrate import solve_ivp

    y0 = _initial_state(scn, float(mu_m3s2), float(r_ref_m))
    t_eval = np.arange(0.0, scn.duration_s + 0.5 * scn.dt_s, scn.dt_s, dtype=np.float64)
    mu = float(mu_m3s2)

    def rhs(_t: float, y: np.ndarray) -> np.ndarray:
        r = y[0:3]
        rn = float(np.linalg.norm(r))
        a = -mu * r / (rn * rn * rn)
        return np.concatenate([y[3:6], a])

    if atol_pos is None and atol_vel is None:
        atol_arg: float | np.ndarray = float(atol)
    else:
        atol_arg = np.full(6, float(atol), dtype=np.float64)
        if atol_pos is not None:
            atol_arg[0:3] = float(atol_pos)
        if atol_vel is not None:
            atol_arg[3:6] = float(atol_vel)

    sol = solve_ivp(
        rhs, (0.0, float(t_eval[-1])), y0, method=method, t_eval=t_eval,
        rtol=float(rtol), atol=atol_arg,
    )
    if not bool(sol.success):
        raise RuntimeError(f"integration failed for {scn.name}: {sol.message}")

    y_ref = propagate_kepler_analytic(y0, t_eval, mu)
    report = ric_error_report(sol.y.T, y_ref, label=f"{scn.name} (numeric vs analytic Kepler)")
    report["scenario"] = scn.name
    report["method"] = method
    report["rtol"] = float(rtol)
    report["atol_mode"] = "vector" if isinstance(atol_arg, np.ndarray) else "scalar"
    report["atol_pos"] = atol_pos
    report["atol_vel"] = atol_vel
    report["period_s"] = float(2.0 * math.pi * math.sqrt((r_ref_m + scn.alt_km * 1000.0) ** 3 / mu))
    return report


# =============================================================================
# 5) CLI
# =============================================================================

DEFAULT_SCENARIOS: tuple[Scenario, ...] = (
    Scenario("LLO_50km_circular", alt_km=50.0, ecc=0.0, inc_deg=85.0, duration_s=6 * 3600.0, dt_s=30.0),
    Scenario("LLO_100km_circular", alt_km=100.0, ecc=0.0, inc_deg=45.0, duration_s=12 * 3600.0, dt_s=60.0),
    Scenario("elliptic_100x2000km", alt_km=100.0, ecc=0.36, inc_deg=30.0, duration_s=24 * 3600.0, dt_s=60.0),
    Scenario("high_500km_circular", alt_km=500.0, ecc=0.0, inc_deg=0.0, duration_s=24 * 3600.0, dt_s=120.0),
)


@dataclass
class _RunConfig:
    out_dir: Path
    method: str = "DOP853"
    rtol: float = 1e-10
    atol: float = 1e-12
    compare_vector_atol: bool = True
    gmat_report: Path | None = None
    gmat_units_km: bool = True
    extra: dict[str, Any] = field(default_factory=dict)


def main(argv: list[str] | None = None) -> int:
    import argparse

    from lunaris.common.constants import MU_MOON, R_MOON

    parser = argparse.ArgumentParser(description="Lunaris trajectory cross-validation vs an independent reference.")
    parser.add_argument("--out", type=Path, default=Path("validation/independent/outputs/cross_validation"))
    parser.add_argument("--method", default="DOP853")
    parser.add_argument("--rtol", type=float, default=1e-10)
    parser.add_argument("--atol", type=float, default=1e-12)
    parser.add_argument("--atol-pos", type=float, default=1e-6, help="position atol for the vector-atol comparison run")
    parser.add_argument("--atol-vel", type=float, default=1e-9, help="velocity atol for the vector-atol comparison run")
    parser.add_argument("--no-vector-atol", action="store_true", help="skip the scalar-vs-vector atol comparison")
    parser.add_argument("--gmat-report", type=Path, default=None,
                        help="optional external ephemeris (GMAT ReportFile) to compare a provided trajectory against")
    parser.add_argument("--gmat-traj", type=Path, default=None,
                        help="npz with arrays t,y (the Lunaris trajectory) to diff against --gmat-report")
    parser.add_argument("--gmat-meters", action="store_true", help="external report already in meters (default km)")
    args = parser.parse_args(argv)

    mu = float(MU_MOON)
    r_ref = float(R_MOON)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {"reference": "classical_kepler", "mu_m3s2": mu, "r_ref_m": r_ref, "scenarios": []}

    print(f"[xval] reference = analytic classical-element Kepler | method={args.method} "
          f"rtol={args.rtol:g} atol={args.atol:g}")
    for scn in DEFAULT_SCENARIOS:
        scalar = cross_validate_integrator(scn, mu, r_ref, method=args.method,
                                           rtol=args.rtol, atol=args.atol)
        entry: dict[str, Any] = {"scenario": asdict(scn), "scalar_atol": scalar}
        line = (f"  {scn.name:24s} scalar atol -> pos RMS {scalar['pos_rms_m']:.3e} m | "
                f"max {scalar['pos_max_m']:.3e} m | in-track RMS {scalar['ric_intrack_rms_m']:.3e} m")
        if not args.no_vector_atol:
            vec = cross_validate_integrator(scn, mu, r_ref, method=args.method, rtol=args.rtol,
                                            atol=args.atol, atol_pos=args.atol_pos, atol_vel=args.atol_vel)
            entry["vector_atol"] = vec
            ratio = (scalar["pos_rms_m"] / vec["pos_rms_m"]) if vec["pos_rms_m"] > 0 else float("nan")
            line += f"  ||  vector atol -> pos RMS {vec['pos_rms_m']:.3e} m (x{ratio:.2f})"
        results["scenarios"].append(entry)
        print(line)

    # Optional external (GMAT) report comparison against a provided Lunaris npz.
    if args.gmat_report is not None and args.gmat_traj is not None:
        t_ref, y_ref = load_external_ephemeris(
            args.gmat_report, length_unit_m=(1.0 if args.gmat_meters else 1000.0))
        data = np.load(args.gmat_traj)
        y_test = np.asarray(data["y"], dtype=np.float64)
        if y_test.shape[0] == 6 and y_test.shape[1] != 6:
            y_test = y_test.T
        n = min(y_test.shape[0], y_ref.shape[0])
        gmat = ric_error_report(y_test[:n], y_ref[:n], label="Lunaris vs external (GMAT) ephemeris")
        results["external_ephemeris"] = {"report": str(args.gmat_report), "metrics": gmat}
        print(f"[xval] external (GMAT) compare -> pos RMS {gmat['pos_rms_m']:.3e} m | "
              f"in-track RMS {gmat['ric_intrack_rms_m']:.3e} m | n={gmat['n_samples']}")

    out_path = out_dir / "cross_validation_report.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[xval] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
