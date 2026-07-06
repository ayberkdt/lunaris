"""Ephemeris and runtime requirement helpers for batch propagation."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

import numpy as np

from lunaris.common.constants import DAY_S
from lunaris.common.math_utils import quat_rotate_np, quat_slerp_np


def _state_to_array(state_like: Any) -> np.ndarray:
    """
    Convert the configured nominal state to a plain row-major float64 vector.

    batch sampling works in Cartesian state space, so we accept the same state
    container styles as the single-run pipeline: ``InitialState``,
    ``OrbitState``-like objects exposing ``.y``, or raw array-likes.
    """

    if state_like is None:
        raise ValueError("Nominal state is None.")

    if hasattr(state_like, "to_array"):
        arr = np.asarray(state_like.to_array(), dtype=np.float64).reshape(-1)
    elif hasattr(state_like, "y"):
        arr = np.asarray(state_like.y, dtype=np.float64).reshape(-1)
    else:
        arr = np.asarray(state_like, dtype=np.float64).reshape(-1)

    if arr.size < 6:
        raise ValueError(f"Nominal state must contain at least 6 elements, got {arr.size}.")
    return np.ascontiguousarray(arr[:6], dtype=np.float64)


def _surface_topography_requested(surface_provider: Any, topo_grid: Any) -> bool:
    """
    Return True when the batch run needs Moon-fixed ephemeris because terrain is active.

    Topography can influence both surface-force sampling and impact detection, so
    the batch bootstrap mirrors the main runner by treating terrain availability as
    an ephemeris requirement even when third-body vectors are disabled.
    """

    return bool(surface_provider is not None or topo_grid is not None)


def _need_ephemeris(cfg: Any, *, topo_requested: bool) -> bool:
    """
    Match the main runner's ephemeris policy for consistent physics coverage.

    The batch/ensemble path should not secretly use a different decision tree than
    the single-run path.  Repeating the logic locally keeps this core module
    self-contained while preserving the same "SH/topography implies q_i2f"
    behavior.
    """

    flags = cfg.flags
    physics_need = (
        flags.enable_sh
        or flags.enable_3rd_body_sun
        or flags.enable_3rd_body_earth
        or flags.enable_earth_j2
        or flags.enable_srp
        or flags.enable_albedo
        or flags.enable_thermal
        or flags.enable_surface_forces
        or flags.enable_tides_k2
        or flags.enable_tides_k3
        or flags.enable_relativity_1pn
    )
    return bool(physics_need or topo_requested)


def _need_body_vectors(cfg: Any) -> bool:
    """
    Return True only when Sun/Earth position tables are physically required.

    SH-only or topo-only batch runs still need the Moon-fixed attitude
    quaternion table, but they do not need Sun/Earth vectors.  Using this split
    keeps ephemeris initialization lighter and avoids misleading SPICE warnings.
    """

    flags = cfg.flags
    thermal_mode = (
        str(getattr(getattr(cfg, "thermal", None), "thermal_mode", "constant_temperature")).strip().lower()
    )
    thermal_needs_sun = bool(
        flags.enable_thermal
        and thermal_mode in {"equilibrium", "equilibrium_temperature", "instantaneous_equilibrium"}
    )
    return bool(
        flags.enable_3rd_body_sun
        or flags.enable_3rd_body_earth
        or flags.enable_earth_j2
        or flags.enable_srp
        or flags.enable_albedo
        or thermal_needs_sun
        or flags.enable_tides_k2
        or flags.enable_tides_k3
        or flags.enable_relativity_1pn
    )


def _build_ephemeris_manager(cfg: Any) -> Any:
    """
    Build an ``EphemerisManager`` using the same buffered timeline as main.py.

    A small duration buffer protects interpolation near the last requested
    sample, which is especially helpful in ensemble runs where many samples
    stop at slightly different times due to impact events.
    """

    from lunaris.physics.ephemeris import EphemerisManager

    start_utc = str(cfg.time.start_date).strip()
    if not start_utc:
        raise ValueError("cfg.time.start_date is empty.")

    time_cfg = replace(cfg.time, duration_s=float(cfg.time.duration_s) + 0.1 * DAY_S)
    spice_cfg = replace(cfg.spice, include_third_body=_need_body_vectors(cfg))
    return EphemerisManager.from_time_and_spice(
        time_cfg,
        spice_cfg,
        auto_fix_kernel_paths=True,
        need_moon_fixed_rotation=True,
    )


def _impact_positions_fixed(
    ephem: Any,
    t_impact: np.ndarray,
    positions_inertial: np.ndarray,
) -> np.ndarray:
    """Rotate per-sample inertial impact positions into the Moon-fixed frame.

    Returns an all-NaN array when no ephemeris is available. Without the
    inertial->Moon-fixed rotation table there is no physically meaningful
    Moon-fixed impact position, so the engine must NOT fabricate one by treating
    inertial coordinates as if they were body-fixed: that silently produces a
    wrong geographic impact distribution (lat/lon) with no error signal.
    Consumers detect the NaN and skip lat/lon reporting instead.
    """
    out = np.full_like(positions_inertial, np.nan, dtype=np.float64)
    if ephem is None:
        return out
    finite = np.isfinite(t_impact) & np.isfinite(positions_inertial).all(axis=1)
    if not np.any(finite):
        return out

    provider = ephem.get_data_provider()
    q_tab = np.asarray(
        provider.get("q_i2f_tab", provider.get("rot_table")),
        dtype=np.float64,
    )
    dt_s = float(provider.get("dt_s", provider.get("dt")))
    for idx in np.where(finite)[0]:
        u = max(0.0, float(t_impact[idx]) / max(dt_s, 1e-12))
        i0 = min(int(math.floor(u)), q_tab.shape[0] - 1)
        if i0 >= q_tab.shape[0] - 1:
            q = q_tab[-1]
        else:
            q = quat_slerp_np(q_tab[i0], q_tab[i0 + 1], u - i0)
        out[idx] = quat_rotate_np(q, positions_inertial[idx])
    return out


__all__ = [
    "_state_to_array",
    "_surface_topography_requested",
    "_need_ephemeris",
    "_need_body_vectors",
    "_build_ephemeris_manager",
    "_impact_positions_fixed",
]
