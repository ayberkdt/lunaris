# tests/test_symplectic_strict_guard.py
"""Review finding #6: paper/validation-safe symplectic guard.

A symplectic integrator's bounded-energy-drift guarantee is void when a
non-conservative force (SRP / albedo / thermal IR / 1PN) is active. In normal use
the propagator only *warns*; ``PropagatorConfig.strict_symplectic=True`` escalates
that to a hard error so a benchmark/paper run cannot silently ship a symplectic
result whose central claim is invalid.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from lunaris.common.constants import MU_MOON, R_MOON
from lunaris.common.type_defs import PropagatorConfig, TimeConfig
from lunaris.core.propagation.propagator import propagate

MU = float(MU_MOON)
R = float(R_MOON)


class _FakeDyn:
    """Minimal point-mass dynamics with configurable perturbation flags."""

    grav = None
    ephem = None

    def __init__(self, **flagkw):
        self.flags = SimpleNamespace(**flagkw)

    def build_rhs(self):
        def rhs(_t, y):
            y = np.asarray(y, dtype=np.float64)
            r = y[:3]
            rn = float(np.linalg.norm(r))
            dy = np.empty_like(y)
            dy[:3] = y[3:6]
            dy[3:6] = -MU * r / rn**3
            return dy
        return rhs


def _state():
    r0 = R + 100e3
    v = math.sqrt(MU / r0)
    return np.array([r0, 0.0, 0.0, 0.0, v, 0.0])


def _tc():
    return TimeConfig(duration_s=600.0, output_dt_s=60.0)


def _cfg(method, strict):
    return PropagatorConfig(method=method, strict_symplectic=strict, verbose=False,
                            compute_2body_baseline=False)


def test_strict_symplectic_raises_with_nonconservative_force():
    dyn = _FakeDyn(enable_srp=True)
    with pytest.raises(ValueError, match="strict_symplectic"):
        propagate(dyn, _state(), _cfg("VV", True), time_cfg=_tc())


def test_nonstrict_only_warns():
    dyn = _FakeDyn(enable_srp=True)
    with pytest.warns(RuntimeWarning, match="bounded-energy-drift"):
        propagate(dyn, _state(), _cfg("VV", False), time_cfg=_tc())


def test_strict_symplectic_ok_when_conservative():
    dyn = _FakeDyn()  # no non-conservative flags -> guarantee holds
    res = propagate(dyn, _state(), _cfg("VV", True), time_cfg=_tc())
    assert res.t.size > 1


def test_strict_symplectic_does_not_affect_adaptive_method():
    # DOP853 is not symplectic, so there is no guarantee to void -> no error.
    dyn = _FakeDyn(enable_srp=True)
    res = propagate(dyn, _state(), _cfg("DOP853", True), time_cfg=_tc())
    assert res.t.size > 1


# ---------------------------------------------------------------------------
# Audit F1 — the gravity model itself can void symplecticity: a non-conservative
# surrogate (acceleration is not the gradient of a scalar potential) breaks the
# bounded-energy-drift guarantee even with every perturbation flag off. The
# supported potential_autograd surrogate and classical SH gravity stay exempt.
# The guard reads the provider's is_conservative taxonomy flag, never a class.
# ---------------------------------------------------------------------------

class _FakeSurrogateGrav:
    """Duck-typed ST-LRPS provider with only what the guard + strict SSOT need."""

    model_kind = "st_lrps"
    R_ref_m = R
    GM_m3s2 = MU
    degree_max = 2

    def __init__(self, kind: str):
        self.config = {"runtime_model_kind": kind}
        # Mirror the runtime taxonomy flag: only potential_autograd is conservative.
        self.is_conservative = kind == "potential_autograd"


def test_strict_symplectic_raises_with_nonconservative_gravity():
    dyn = _FakeDyn()
    dyn.grav = _FakeSurrogateGrav("force_direct")
    with pytest.raises(ValueError, match="strict_symplectic"):
        propagate(dyn, _state(), _cfg("VV", True), time_cfg=_tc())


def test_nonconservative_gravity_warns_when_not_strict():
    dyn = _FakeDyn()
    dyn.grav = _FakeSurrogateGrav("force_direct")
    with pytest.warns(RuntimeWarning, match="non-conservative"):
        propagate(dyn, _state(), _cfg("VV", False), time_cfg=_tc())


def test_potential_autograd_gravity_does_not_trip_guard():
    dyn = _FakeDyn()
    dyn.grav = _FakeSurrogateGrav("potential_autograd")
    res = propagate(dyn, _state(), _cfg("VV", True), time_cfg=_tc())
    assert res.t.size > 1


def test_nonconservative_gravity_ok_under_adaptive_method():
    # No symplectic guarantee to void under DOP853 -> no guard trip.
    dyn = _FakeDyn()
    dyn.grav = _FakeSurrogateGrav("force_direct")
    res = propagate(dyn, _state(), _cfg("DOP853", True), time_cfg=_tc())
    assert res.t.size > 1


def test_gravity_guard_reads_is_conservative_flag():
    from lunaris.core.propagation.integrators.fixed_step import (
        symplectic_nonconservative_gravity,
    )

    grav = _FakeSurrogateGrav("potential_autograd")
    grav.is_conservative = False  # taxonomy flag is authoritative
    assert symplectic_nonconservative_gravity("VV", grav)
    assert symplectic_nonconservative_gravity("DOP853", grav) == []
    assert symplectic_nonconservative_gravity("VV", None) == []
