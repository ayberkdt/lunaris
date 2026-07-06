"""Orbit-level drift harness for ST-LRPS force-model validation.

Pointwise field-error metrics are necessary but not sufficient: a small
per-point acceleration error can still integrate into a large trajectory error,
and any non-conservative content can pump or drain mechanical energy secularly
over many revolutions.

This module supplies the orbit-level half of that validation as a small,
model-independent harness:

* :func:`propagate_orbit` integrates ``r'' = a(r)`` with fixed-step RK4;
* :func:`orbit_drift` integrates two acceleration models from the *same* initial
  state with identical integrator settings and reports their trajectory
  divergence (position / velocity drift over time);
* :func:`energy_drift` tracks the mechanical energy ``0.5|v|^2 + U(r)`` of an
  orbit propagated under one field, relative to a supplied reference potential.
  For a conservative field this stays bounded by integrator truncation error;
  secular growth would expose any non-conservative content at the orbit level.

The acceleration callables are position-only, evaluated in a single Cartesian
frame (the Moon-fixed frame ST-LRPS predicts in). This is the honest scope of a
*drift comparison*: both models are integrated under identical simplifications,
so the reported divergence is attributable to the difference between the models,
not to frame or integrator mismatch. It is deliberately not a mission-grade
benchmark with rotating frames and a full perturbation stack.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

AccelFn = Callable[[np.ndarray], np.ndarray]
PotentialFn = Callable[[np.ndarray], float]


def _as_vec3(value: Any, name: str) -> np.ndarray:
    vec = np.asarray(value, dtype=np.float64).reshape(-1)
    if vec.size != 3:
        raise ValueError(f"{name} must have 3 elements, got {vec.size}.")
    return vec


def _accel_at(accel_fn: AccelFn, r: np.ndarray) -> np.ndarray:
    """Evaluate ``accel_fn`` at a single position, returning a finite ``(3,)``."""
    a = np.asarray(accel_fn(r.reshape(1, 3)), dtype=np.float64).reshape(-1)
    if a.size != 3:
        raise ValueError(f"accel_fn must return 3 components, got {a.size}.")
    if not np.all(np.isfinite(a)):
        raise FloatingPointError("accel_fn returned a non-finite acceleration.")
    return a


def propagate_orbit(
    accel_fn: AccelFn,
    r0: Any,
    v0: Any,
    *,
    dt_s: float,
    n_steps: int,
) -> dict[str, np.ndarray]:
    """Integrate ``r'' = a(r)`` with fixed-step classical RK4.

    Parameters
    ----------
    accel_fn:
        Maps ``(1, 3)`` position [m] to ``(1, 3)`` (or ``(3,)``) total
        acceleration [m/s^2]. Position-only and time-independent.
    r0, v0:
        Initial position [m] and velocity [m/s], each 3 elements.
    dt_s:
        Fixed step [s]; must be positive.
    n_steps:
        Number of steps; must be positive.

    Returns
    -------
    dict with ``times`` ``(n_steps + 1,)``, ``positions`` and ``velocities``
    ``(n_steps + 1, 3)``.
    """
    if dt_s <= 0.0:
        raise ValueError(f"dt_s must be positive, got {dt_s!r}.")
    if n_steps <= 0:
        raise ValueError(f"n_steps must be positive, got {n_steps!r}.")

    r = _as_vec3(r0, "r0")
    v = _as_vec3(v0, "v0")
    dt = float(dt_s)
    n = int(n_steps)

    positions = np.empty((n + 1, 3), dtype=np.float64)
    velocities = np.empty((n + 1, 3), dtype=np.float64)
    positions[0] = r
    velocities[0] = v

    for step in range(n):
        # State y = (r, v); y' = (v, a(r)). Classical RK4.
        k1r = v
        k1v = _accel_at(accel_fn, r)
        k2r = v + 0.5 * dt * k1v
        k2v = _accel_at(accel_fn, r + 0.5 * dt * k1r)
        k3r = v + 0.5 * dt * k2v
        k3v = _accel_at(accel_fn, r + 0.5 * dt * k2r)
        k4r = v + dt * k3v
        k4v = _accel_at(accel_fn, r + dt * k3r)
        r = r + (dt / 6.0) * (k1r + 2.0 * k2r + 2.0 * k3r + k4r)
        v = v + (dt / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
        positions[step + 1] = r
        velocities[step + 1] = v

    times = np.arange(n + 1, dtype=np.float64) * dt
    return {"times": times, "positions": positions, "velocities": velocities}


def _summ(series: np.ndarray) -> dict[str, float]:
    return {
        "final": float(series[-1]),
        "max": float(np.max(series)),
        "rms": float(np.sqrt(np.mean(series ** 2))),
    }


def orbit_drift(
    accel_ref: AccelFn,
    accel_test: AccelFn,
    r0: Any,
    v0: Any,
    *,
    dt_s: float,
    n_steps: int,
) -> dict[str, Any]:
    """Trajectory divergence between two acceleration models.

    Both models are propagated from the same initial state with identical
    integrator settings, so the divergence is attributable to the difference
    between ``accel_ref`` (e.g. a truth model or SH baseline) and ``accel_test``
    (e.g. the ``potential_autograd`` surrogate field).

    Returns position/velocity drift time series plus ``final/max/rms`` summaries
    and a scale-free ``position_drift_rel_max`` (max drift over the reference
    orbit radius range).
    """
    ref = propagate_orbit(accel_ref, r0, v0, dt_s=dt_s, n_steps=n_steps)
    test = propagate_orbit(accel_test, r0, v0, dt_s=dt_s, n_steps=n_steps)

    pos_drift = np.linalg.norm(test["positions"] - ref["positions"], axis=1)
    vel_drift = np.linalg.norm(test["velocities"] - ref["velocities"], axis=1)
    ref_radius = np.linalg.norm(ref["positions"], axis=1)
    scale = float(np.median(ref_radius)) if ref_radius.size else 0.0

    return {
        "schema_version": 1,
        "dt_s": float(dt_s),
        "n_steps": int(n_steps),
        "duration_s": float(dt_s * n_steps),
        "times": ref["times"],
        "position_drift_m": pos_drift,
        "velocity_drift_m_s": vel_drift,
        "position_drift_summary_m": _summ(pos_drift),
        "velocity_drift_summary_m_s": _summ(vel_drift),
        "position_drift_rel_max": float(np.max(pos_drift) / scale) if scale > 0.0 else float("inf"),
    }


def energy_drift(
    accel_fn: AccelFn,
    potential_fn: PotentialFn,
    r0: Any,
    v0: Any,
    *,
    dt_s: float,
    n_steps: int,
) -> dict[str, Any]:
    """Mechanical-energy drift of an orbit relative to a reference potential.

    The orbit is propagated under ``accel_fn``; specific mechanical energy
    ``E = 0.5 |v|^2 + U(r)`` is evaluated with ``potential_fn`` (specific
    potential ``U`` [m^2/s^2], such that the conservative field is ``-grad U``).
    For a conservative field consistent with ``U`` the energy stays bounded by
    integrator truncation error; a secular trend exposes non-conservative
    content. Returns the energy time series and the relative drift
    ``(E - E0) / |E0|``.
    """
    orbit = propagate_orbit(accel_fn, r0, v0, dt_s=dt_s, n_steps=n_steps)
    pos = orbit["positions"]
    vel = orbit["velocities"]
    kinetic = 0.5 * np.sum(vel ** 2, axis=1)
    potential = np.array([float(potential_fn(p.reshape(1, 3))) for p in pos], dtype=np.float64)
    energy = kinetic + potential
    e0 = float(energy[0])
    rel = (energy - e0) / max(abs(e0), 1e-300)
    return {
        "schema_version": 1,
        "dt_s": float(dt_s),
        "n_steps": int(n_steps),
        "times": orbit["times"],
        "energy": energy,
        "relative_energy_drift": rel,
        "relative_energy_drift_final": float(rel[-1]),
        "relative_energy_drift_max_abs": float(np.max(np.abs(rel))),
    }


def circular_orbit_state(mu: float, radius_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(r0, v0)`` for a planar circular orbit of a point-mass ``mu``.

    Convenience for callers/tests: ``r0 = (radius, 0, 0)``, circular speed
    ``sqrt(mu / radius)`` along ``+y``.
    """
    if mu <= 0.0 or radius_m <= 0.0:
        raise ValueError("mu and radius_m must be positive.")
    r0 = np.array([radius_m, 0.0, 0.0], dtype=np.float64)
    v0 = np.array([0.0, float(np.sqrt(mu / radius_m)), 0.0], dtype=np.float64)
    return r0, v0


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=(
            "Orbit-drift harness: propagate a circular orbit under an ST-LRPS "
            "surrogate artifact and a reference field and report trajectory drift."
        )
    )
    ap.add_argument("--model-dir", required=True, help="ST-LRPS run directory to evaluate.")
    ap.add_argument(
        "--ref-model-dir", default=None,
        help="Reference ST-LRPS run directory to compare against (default: point-mass base only).",
    )
    ap.add_argument("--altitude-km", type=float, default=200.0)
    ap.add_argument("--orbits", type=float, default=10.0)
    ap.add_argument("--steps-per-orbit", type=int, default=720)
    ap.add_argument("--out", default="outputs/orbit_drift/orbit_drift.json")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument(
        "--strict-domain", action="store_true",
        help="Hard-fail if the propagated orbit leaves the surrogate's trained domain "
             "(default: integrate anyway and report the extrapolation fraction).",
    )
    return ap


class _DomainTrackingTotalAccel:
    """Total-acceleration callable that records out-of-domain extrapolation.

    Wraps a loaded surrogate so every evaluated position is checked against the
    artifact's trained domain (:meth:`SurrogateForceModel.domain_status`). The
    counts let the orbit summary state how much of a trajectory left the training
    envelope, so a drift number can never silently rest on extrapolation. When
    ``strict_domain`` is True the wrapped runtime is loaded strict and the
    underlying prediction raises on extrapolation; this wrapper additionally
    raises early with a clear orbit-level message.
    """

    def __init__(self, runtime: Any, mu: float, *, strict_domain: bool = False) -> None:
        self.runtime = runtime
        self.mu = float(mu)
        self.strict_domain = bool(strict_domain)
        self.n_eval = 0
        self.n_extrapolating = 0

    def __call__(self, r: np.ndarray) -> np.ndarray:
        r = np.asarray(r, dtype=np.float64).reshape(-1, 3)
        try:
            status = self.runtime.domain_status(r)
            extrapolating = bool(status.get("recommended_fallback"))
        except Exception:
            status, extrapolating = None, False
        self.n_eval += int(r.shape[0])
        if extrapolating:
            self.n_extrapolating += int(r.shape[0])
            if self.strict_domain:
                raise RuntimeError(
                    "orbit_drift: trajectory left the surrogate's trained domain "
                    f"({(status or {}).get('reason', 'out of domain')}); refusing to "
                    "integrate an extrapolated orbit under strict_domain=True."
                )
        rn = np.linalg.norm(r, axis=1, keepdims=True)
        base = -self.mu * r / rn ** 3
        residual = np.asarray(
            self.runtime.predict_residual_accel_fixed(r), dtype=np.float64
        ).reshape(-1, 3)
        return base + residual

    @property
    def extrapolation_fraction(self) -> float:
        return float(self.n_extrapolating) / float(self.n_eval) if self.n_eval else 0.0

    def domain_report(self) -> dict[str, Any]:
        return {
            "evaluations": int(self.n_eval),
            "extrapolating_evaluations": int(self.n_extrapolating),
            "extrapolation_fraction": self.extrapolation_fraction,
            "strict_domain": self.strict_domain,
        }


def _runtime_total_accel(
    model_dir: str, device: str, *, strict_domain: bool = False
) -> tuple[_DomainTrackingTotalAccel, float, float]:
    """Build a total-acceleration callable (point-mass base + residual) for a run.

    Returns ``(accel_fn, mu, r_ref_m)`` using the artifact's own ``mu`` so the
    reference and test orbits share an identical Keplerian base. The returned
    callable tracks out-of-domain extrapolation (see
    :class:`_DomainTrackingTotalAccel`); ``strict_domain=True`` loads the runtime
    strict and hard-fails the propagation on extrapolation.
    """
    from lunaris.surrogate.st_lrps.runtime.force_model import load_surrogate_force_model

    runtime = load_surrogate_force_model(model_dir, device=device, strict_domain=strict_domain)
    contract = runtime.artifact_contract
    mu = float(getattr(contract, "mu_si", 0.0) or 0.0)
    r_ref = float(getattr(contract, "r_ref_m", 0.0) or 0.0)
    if mu <= 0.0:
        raise ValueError(f"Artifact {model_dir} has no usable mu_si for a Keplerian base.")

    return _DomainTrackingTotalAccel(runtime, mu, strict_domain=strict_domain), mu, r_ref


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    test_accel, mu, r_ref = _runtime_total_accel(
        args.model_dir, args.device, strict_domain=bool(args.strict_domain)
    )
    if args.ref_model_dir:
        ref_accel, _, _ = _runtime_total_accel(
            args.ref_model_dir, args.device, strict_domain=bool(args.strict_domain)
        )
    else:
        def ref_accel(r: np.ndarray) -> np.ndarray:  # point-mass base only
            r = np.asarray(r, dtype=np.float64).reshape(-1, 3)
            rn = np.linalg.norm(r, axis=1, keepdims=True)
            return -mu * r / rn ** 3

    radius = r_ref + float(args.altitude_km) * 1000.0
    r0, v0 = circular_orbit_state(mu, radius)
    period = 2.0 * np.pi * np.sqrt(radius ** 3 / mu)
    n_steps = max(1, int(round(args.orbits * args.steps_per_orbit)))
    dt = (args.orbits * period) / n_steps

    report = orbit_drift(ref_accel, test_accel, r0, v0, dt_s=dt, n_steps=n_steps)
    summary = {
        "model_dir": str(args.model_dir),
        "ref_model_dir": str(args.ref_model_dir) if args.ref_model_dir else "point_mass_base",
        "altitude_km": float(args.altitude_km),
        "orbits": float(args.orbits),
        "orbital_period_s": float(period),
        "dt_s": report["dt_s"],
        "n_steps": report["n_steps"],
        "position_drift_summary_m": report["position_drift_summary_m"],
        "velocity_drift_summary_m_s": report["velocity_drift_summary_m_s"],
        "position_drift_rel_max": report["position_drift_rel_max"],
        "test_domain": test_accel.domain_report(),
    }
    if isinstance(ref_accel, _DomainTrackingTotalAccel):
        summary["ref_domain"] = ref_accel.domain_report()
    out = Path(args.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
