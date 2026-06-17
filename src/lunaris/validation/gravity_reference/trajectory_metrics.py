"""Trajectory comparison metrics including RIC/RTN components."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np


def specific_energy_drift(
    states: np.ndarray,
    potential_fn: Callable[[np.ndarray], float],
) -> dict[str, Any]:
    """Relative drift of the conserved specific orbital energy along a trajectory.

    For a gravity-only conservative field the Hamiltonian
    ``E = 1/2 |v|^2 - U(r)`` is invariant, where ``U`` is the positive-geodesy
    potential (so that ``a = +grad U``). A non-constant ``E`` exposes a sign,
    frame, or integrator-stability defect that a one-shot state comparison can
    miss. This is an *internal* truth check: it needs no external reference.
    """
    s = np.asarray(states, dtype=np.float64)
    if s.ndim != 2 or s.shape[1] != 6:
        raise ValueError("states must have shape (N,6).")
    energy = np.empty(s.shape[0], dtype=np.float64)
    for i in range(s.shape[0]):
        v = s[i, 3:]
        energy[i] = 0.5 * float(np.dot(v, v)) - float(potential_fn(s[i, :3]))
    e0 = float(energy[0])
    rel = np.abs((energy - e0) / e0) if e0 != 0.0 else np.abs(energy - e0)
    return {
        "initial_specific_energy_m2_s2": e0,
        "max_abs_relative_drift": float(np.max(rel)),
        "final_abs_relative_drift": float(rel[-1]),
    }


def ric_basis(reference_state: np.ndarray) -> np.ndarray:
    """Return columns [R, I, C] from a Cartesian reference state."""
    state = np.asarray(reference_state, dtype=np.float64).reshape(6)
    r = state[:3]
    v = state[3:]
    r_norm = float(np.linalg.norm(r))
    h = np.cross(r, v)
    h_norm = float(np.linalg.norm(h))
    if r_norm <= 0.0 or h_norm <= 0.0:
        raise ValueError("Degenerate RIC basis.")
    radial = r / r_norm
    cross_track = h / h_norm
    in_track = np.cross(cross_track, radial)
    return np.column_stack((radial, in_track, cross_track))


def compute_trajectory_metrics(
    reference_states: np.ndarray,
    lunaris_states: np.ndarray,
) -> dict[str, Any]:
    """Compute Cartesian and RIC trajectory metrics."""
    ref = np.asarray(reference_states, dtype=np.float64)
    got = np.asarray(lunaris_states, dtype=np.float64)
    if ref.shape != got.shape or ref.ndim != 2 or ref.shape[1] != 6:
        raise ValueError("Trajectory state arrays must both have shape (N,6).")
    dr = got[:, :3] - ref[:, :3]
    dv = got[:, 3:] - ref[:, 3:]
    pos = np.linalg.norm(dr, axis=1)
    vel = np.linalg.norm(dv, axis=1)
    ric = np.zeros_like(dr)
    for i, state in enumerate(ref):
        ric[i] = ric_basis(state).T @ dr[i]
    traveled = float(np.sum(np.linalg.norm(np.diff(ref[:, :3], axis=0), axis=1)))
    return {
        "n_epochs": int(ref.shape[0]),
        "position_error_m": {
            "rms": float(np.sqrt(np.mean(pos**2))),
            "median": float(np.median(pos)),
            "p95": float(np.percentile(pos, 95)),
            "p99": float(np.percentile(pos, 99)),
            "max": float(np.max(pos)),
            "final": float(pos[-1]),
        },
        "velocity_error_m_s": {
            "rms": float(np.sqrt(np.mean(vel**2))),
            "median": float(np.median(vel)),
            "p95": float(np.percentile(vel, 95)),
            "p99": float(np.percentile(vel, 99)),
            "max": float(np.max(vel)),
            "final": float(vel[-1]),
        },
        "ric_error_m": {
            "radial_max_abs": float(np.max(np.abs(ric[:, 0]))),
            "in_track_max_abs": float(np.max(np.abs(ric[:, 1]))),
            "cross_track_max_abs": float(np.max(np.abs(ric[:, 2]))),
            "final": [float(v) for v in ric[-1]],
        },
        "normalized_position_error": {
            "max_over_traveled_distance": float(np.max(pos) / traveled) if traveled > 0 else None,
        },
        "maximum_error_epoch_index": int(np.argmax(pos)),
    }

