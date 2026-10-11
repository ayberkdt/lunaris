"""Post-processing uses the run's gravity-model GM and reference radius.

The propagator takes GM and the impact/altitude reference radius from the
attached gravity model (``core.propagation.time_grid.get_ref_radius_and_mu``:
``grav.GM_m3s2`` / ``grav.R_ref_m``, constants only without a model). The
analysis paths read attributes the engine does not have (``mu_m3s2``,
``R_body_m``) and so always fell back: ``process_simulation_results`` to a
hard-coded 4.902801e12 m^3/s^2 and 1737.4 km, ``build_orbit_analysis`` to
``MU_MOON``.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from lunaris.analysis import orbit_analysis, postprocess
from lunaris.common.constants import MU_MOON, R_MOON

_GM = 4.9028003063302e12
_RREF = 1_738_100.0  # deliberately distinct from every constant


class _Captured(Exception):
    pass


def _result() -> SimpleNamespace:
    t = np.array([0.0, 10.0, 20.0])
    r = 1.85e6
    v = (_GM / r) ** 0.5
    y = np.array([[r, 0.0, 0.0, 0.0, v, 0.0]] * 3).T
    return SimpleNamespace(t=t, y=y)


def _capture(monkeypatch, module) -> dict:
    seen: dict = {}

    def fake(t, y, *, mu, R_body, **_kw):
        seen["mu"] = mu
        seen["R_body"] = R_body
        raise _Captured

    monkeypatch.setattr(module, "compute_history", fake)
    return seen


_ENGINE = SimpleNamespace(grav=SimpleNamespace(GM_m3s2=_GM, R_ref_m=_RREF))


def test_process_simulation_results_uses_engine_gravity_model(monkeypatch) -> None:
    seen = _capture(monkeypatch, postprocess)
    with pytest.raises(_Captured):
        postprocess.process_simulation_results(_result(), ctx=_ENGINE)
    assert seen == {"mu": _GM, "R_body": _RREF}


def test_process_simulation_results_falls_back_to_constants(monkeypatch) -> None:
    seen = _capture(monkeypatch, postprocess)
    with pytest.raises(_Captured):
        postprocess.process_simulation_results(_result(), ctx=SimpleNamespace(grav=None))
    assert seen == {"mu": float(MU_MOON), "R_body": float(R_MOON)}


def test_build_orbit_analysis_uses_engine_gravity_model(monkeypatch) -> None:
    seen = _capture(monkeypatch, orbit_analysis)
    with pytest.raises(_Captured):
        orbit_analysis.build_orbit_analysis(_result(), config=None, ctx=_ENGINE)
    assert seen == {"mu": _GM, "R_body": _RREF}



# ---------------------------------------------------------------------------
# Eclipse mask: Sun/Earth rows are interpolated at the sample time
# ---------------------------------------------------------------------------

def _rot_z(deg: float) -> np.ndarray:
    a = np.radians(deg)
    return np.array([np.cos(a), np.sin(a), 0.0])


def test_ephem_index_is_nearest_left() -> None:
    t_eph = np.array([0.0, 60.0, 120.0])
    j = postprocess._map_sim_time_to_ephem_index(np.array([0.0, 30.0, 60.0, 119.0, 500.0]), t_eph, 0.0)
    assert j.tolist() == [0, 0, 1, 1, 2]


def test_eclipse_mask_interpolates_sun_between_ephemeris_rows() -> None:
    """Halfway between two rows the interpolated Sun direction (80 deg) puts
    the spacecraft in the Moon's shadow; the next row (160 deg), which the mask
    used before, does not."""
    from lunaris.common.constants import AU, R_MOON_MEAN

    sun = np.vstack([AU * _rot_z(0.0), AU * _rot_z(160.0)])
    ctx = SimpleNamespace(
        ephem_data={
            "t_s": np.array([0.0, 3600.0]),
            "t0_s": 0.0,
            "q_i2f": np.array([[1.0, 0.0, 0.0, 0.0]] * 2),
            "r_sun_i_m": sun,
        }
    )
    r = -(R_MOON_MEAN + 100e3) * _rot_z(80.0)  # anti-Sun at the midpoint
    y = np.column_stack([np.r_[r, 0.0, 0.0, 0.0]] * 2)
    out = postprocess._eclipse_if_available(ctx, np.array([1800.0, 1800.5]), y, R_body=R_MOON_MEAN)
    assert out is not None
    assert bool(out["mask"][0]) is True
