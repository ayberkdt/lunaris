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

torch = pytest.importorskip('torch')

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


# ---------------------------------------------------------------------------
# Physics audit 2026-07-11 F1 — every acceleration-form method assumes
# a = f(t, r). RKN4 is acceleration-form but NOT symplectic, so the symplectic
# guard never fired for RKN4 + 1PN even though stage accelerations sample a
# stale velocity. The dedicated accel-form guard must cover it; the VV path
# keeps its original (symplectic) message and must not double-warn.
# ---------------------------------------------------------------------------

def test_rkn4_with_1pn_warns_stale_velocity_sampling():
    dyn = _FakeDyn(enable_relativity_1pn=True)
    with pytest.warns(RuntimeWarning, match="stale/inconsistent"):
        propagate(dyn, _state(), _cfg("RKN4", False), time_cfg=_tc())


def test_rkn4_with_1pn_strict_raises():
    dyn = _FakeDyn(enable_relativity_1pn=True)
    with pytest.raises(ValueError, match="strict_symplectic"):
        propagate(dyn, _state(), _cfg("RKN4", True), time_cfg=_tc())


def test_rkn4_without_velocity_dependent_force_is_silent():
    # SRP is non-conservative but position/attitude-only in this taxonomy sense
    # (not velocity-dependent); RKN4 carries no symplectic guarantee to void,
    # so neither guard may fire.
    import warnings as _warnings

    dyn = _FakeDyn(enable_srp=True)
    with _warnings.catch_warnings(record=True) as rec:
        _warnings.simplefilter("always")
        propagate(dyn, _state(), _cfg("RKN4", True), time_cfg=_tc())
    guard_msgs = [
        w for w in rec
        if "Acceleration-form" in str(w.message) or "bounded-energy-drift" in str(w.message)
    ]
    assert guard_msgs == []


def test_vv_with_1pn_keeps_single_symplectic_warning():
    # The symplectic message already carries the inconsistent-velocity note for
    # 1PN; the accel-form guard must not add a duplicate second warning.
    dyn = _FakeDyn(enable_relativity_1pn=True)
    with pytest.warns(RuntimeWarning, match="bounded-energy-drift") as rec:
        propagate(dyn, _state(), _cfg("VV", False), time_cfg=_tc())
    assert not [w for w in rec if "Acceleration-form" in str(w.message)]


def test_accel_form_guard_unit_semantics():
    from lunaris.core.propagation.integrators.fixed_step import (
        accel_form_velocity_dependence_violations,
    )

    flags_1pn = SimpleNamespace(enable_relativity_1pn=True)
    assert accel_form_velocity_dependence_violations("RKN4", flags_1pn) == ["1PN relativity"]
    assert accel_form_velocity_dependence_violations("VV", flags_1pn) == ["1PN relativity"]
    assert accel_form_velocity_dependence_violations("RK4", flags_1pn) == []
    assert accel_form_velocity_dependence_violations("DOP853", flags_1pn) == []
    assert accel_form_velocity_dependence_violations("RKN4", SimpleNamespace(enable_srp=True)) == []
    assert accel_form_velocity_dependence_violations("RKN4", None) == []


# ---------------------------------------------------------------------------
# Physics audit 2026-07-11 F3 — adaptive-degree SH gravity switches degree at
# discrete altitude thresholds, so the field is discontinuous in position; the
# symplectic bounded-drift argument assumes a smooth Hamiltonian. The guard
# reads the prepared gravity pack's ``adaptive_enabled`` flag.
# ---------------------------------------------------------------------------

def _adaptive_dyn():
    dyn = _FakeDyn()
    dyn._prep = {"grav": SimpleNamespace(adaptive_enabled=True)}
    return dyn


def test_symplectic_with_adaptive_degree_warns():
    with pytest.warns(RuntimeWarning, match="discontinuous"):
        propagate(_adaptive_dyn(), _state(), _cfg("VV", False), time_cfg=_tc())


def test_symplectic_with_adaptive_degree_strict_raises():
    with pytest.raises(ValueError, match="strict_symplectic"):
        propagate(_adaptive_dyn(), _state(), _cfg("PEFRL", True), time_cfg=_tc())


@pytest.mark.parametrize("method", ["RK4", "DOP853"])
def test_nonsymplectic_with_adaptive_degree_is_silent(method):
    import warnings as _warnings

    with _warnings.catch_warnings(record=True) as rec:
        _warnings.simplefilter("always")
        propagate(_adaptive_dyn(), _state(), _cfg(method, True), time_cfg=_tc())
    assert not [w for w in rec if "adaptive-degree" in str(w.message)]


def test_adaptive_guard_unit_semantics():
    from lunaris.core.propagation.integrators.fixed_step import (
        symplectic_discontinuous_gravity,
    )

    pack_on = SimpleNamespace(adaptive_enabled=True)
    pack_off = SimpleNamespace(adaptive_enabled=False)
    assert symplectic_discontinuous_gravity("VV", pack_on)
    assert symplectic_discontinuous_gravity("Y6", pack_on)
    assert symplectic_discontinuous_gravity("RKN4", pack_on) == []  # not symplectic
    assert symplectic_discontinuous_gravity("DOP853", pack_on) == []
    assert symplectic_discontinuous_gravity("VV", pack_off) == []
    assert symplectic_discontinuous_gravity("VV", None) == []
