# tests/test_gravity_provider_domain_and_baseline.py
"""GPU-tensor path domain guard + potential/acceleration baseline consistency.

Covers two fixes to ``lunaris.surrogate.runtime.gravity_provider``:

1. The tensor entry points (``predict_*_accel_torch``) used by the GPU batch
   propagator now enforce the training-altitude envelope (hard-fail under
   ``strict_domain``, warn-once otherwise), matching the CPU ``_fixed`` path.
2. ``_base_potential`` uses the SH(baseline_degree) potential -- consistent with
   ``_base_acceleration`` -- instead of the bare monopole, so the total field is
   self-consistent (``a = +grad(U)``).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from lunaris.common.constants import MU_MOON, R_MOON
from lunaris.physics.spherical_harmonics import GravityModel
from lunaris.surrogate.runtime import SurrogateGravityModel
from lunaris.surrogate.runtime.networks import _build_model_from_config

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_tiny_run(tmp_path: Path, run_name: str, extra_config: dict | None = None) -> Path:
    run_dir = tmp_path / run_name
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    config = {
        "hidden": 8, "depth": 1, "activation": "tanh", "dropout": 0.0,
        "resolved_mu_si": float(MU_MOON), "resolved_a_sign": 1.0,
        "scaler_kind": "isometric", "degree_min": 0, "degree_max": 50,
    }
    if extra_config:
        config.update(extra_config)
    scaler = {
        "x": {"mean": [0.0, 0.0, 0.0], "scale": 2_000_000.0},
        "u": {"mean": [0.0], "scale": 1.0},
        "a": {"mean": [0.0, 0.0, 0.0], "scale": 1.0},
    }
    model = _build_model_from_config(config)
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
    (run_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (run_dir / "scaler.json").write_text(json.dumps(scaler), encoding="utf-8")
    torch.save({"model": model.state_dict(), "config": config, "scaler": scaler}, ckpt_dir / "ckpt_best.pt")
    return run_dir


def _synthetic_field(degree: int = 6):
    rng = np.random.default_rng(11)
    c = np.zeros((degree + 1, degree + 1))
    s = np.zeros((degree + 1, degree + 1))
    c[0, 0] = 1.0
    for n in range(2, degree + 1):
        for m in range(n + 1):
            c[n, m] = rng.normal() * 1e-4
            if m > 0:
                s[n, m] = rng.normal() * 1e-4
    return c, s


def _bare(**attrs) -> SurrogateGravityModel:
    """A SurrogateGravityModel with only the attributes a method-under-test needs."""
    inst = SurrogateGravityModel.__new__(SurrogateGravityModel)
    for k, v in attrs.items():
        setattr(inst, k, v)
    return inst


# ---------------------------------------------------------------------------
# 1) GPU-tensor domain guard
# ---------------------------------------------------------------------------

def _guard_inst(strict: bool, lo=50.0, hi=150.0):
    return _bare(
        _alt_min_km=lo, _alt_max_km=hi, R_ref_m=float(R_MOON),
        strict_domain=strict, _warned_out_of_domain=False,
    )


def test_domain_guard_passes_inside_envelope():
    inst = _guard_inst(strict=True)
    x = torch.tensor([[R_MOON + 100e3, 0.0, 0.0]], dtype=torch.float32)
    inst._enforce_domain_torch(x, caller="t")  # must not raise


def test_domain_guard_strict_raises_outside_envelope():
    inst = _guard_inst(strict=True)
    x = torch.tensor([[R_MOON + 300e3, 0.0, 0.0]], dtype=torch.float32)  # 300 km > 150 km
    with pytest.raises(RuntimeError, match="outside the surrogate training envelope"):
        inst._enforce_domain_torch(x, caller="t")


def test_domain_guard_nonstrict_warns_once():
    inst = _guard_inst(strict=False)
    x = torch.tensor([[R_MOON + 10e3, 0.0, 0.0]], dtype=torch.float32)  # 10 km < 50 km
    with pytest.warns(RuntimeWarning, match="extrapolation"):
        inst._enforce_domain_torch(x, caller="t")
    # Second call is silent (warn-once) and still does not raise.
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        inst._enforce_domain_torch(x, caller="t")


def test_domain_guard_noop_without_envelope():
    inst = _guard_inst(strict=True, lo=None, hi=None)
    x = torch.tensor([[R_MOON + 5_000e3, 0.0, 0.0]], dtype=torch.float32)
    inst._enforce_domain_torch(x, caller="t")  # unknown envelope -> never blocks


def test_resolve_altitude_envelope_sources():
    R = SurrogateGravityModel._resolve_altitude_envelope
    assert R({"altitude_min_km": 40.0, "altitude_max_km": 200.0}, None) == (40.0, 200.0)
    assert R({"alt_min_km": 30.0, "alt_max_km": 180.0}, None) == (30.0, 180.0)
    assert R({"artifact_contract": {"altitude_min_km": 25.0, "altitude_max_km": 160.0}}, None) == (25.0, 160.0)
    assert R({}, None) == (None, None)

    class _FR:
        _train_alt_min_km = 60.0
        _train_alt_max_km = 140.0
    assert R({"altitude_min_km": 1.0, "altitude_max_km": 2.0}, _FR()) == (60.0, 140.0)  # runtime wins


def test_predict_total_accel_torch_strict_domain_end_to_end(tmp_path, monkeypatch):
    # The synthetic run is pre-contract by construction; the domain guard under
    # test lives on the legacy research loader, which needs the explicit opt-in
    # (default refusal is covered in test_gravity_provider_fallback_policy.py).
    monkeypatch.setenv("LUNARIS_ALLOW_LEGACY_ARTIFACT", "1")
    run = _make_tiny_run(tmp_path, "guard_run",
                         extra_config={"altitude_min_km": 50.0, "altitude_max_km": 150.0})
    model = SurrogateGravityModel.from_model_dir(
        run, mu_override=float(MU_MOON), r_ref_override=float(R_MOON),
        device_preference="cpu", strict_domain=True,
    )
    assert model.strict_domain is True
    assert (model._alt_min_km, model._alt_max_km) == (50.0, 150.0)

    inside = torch.tensor([[R_MOON + 100e3, 0.0, 0.0]], dtype=torch.float32)
    model.predict_total_accel_torch(inside)  # ok

    outside = torch.tensor([[R_MOON + 400e3, 0.0, 0.0]], dtype=torch.float32)
    with pytest.raises(RuntimeError, match="training envelope"):
        model.predict_total_accel_torch(outside)


# ---------------------------------------------------------------------------
# 2) Potential / acceleration baseline consistency
# ---------------------------------------------------------------------------

def _query_points():
    return np.array([
        [R_MOON + 50e3, 0.0, 0.0],
        [0.0, R_MOON + 120e3, 0.0],
        [(R_MOON + 80e3) * 0.6, (R_MOON + 80e3) * 0.5, (R_MOON + 80e3) * 0.62],
    ], dtype=np.float64)


def test_base_potential_uses_sh_baseline_not_monopole():
    N = 6
    c, s = _synthetic_field(N)
    gm = GravityModel.from_arrays(degree_max=N, r_ref=float(R_MOON), mu=float(MU_MOON),
                                  c_coeffs_full=c, s_coeffs_full=s)
    inst = _bare(baseline_gravity_model=gm, baseline_degree=N,
                 _mu_tensor=torch.tensor(float(MU_MOON), dtype=torch.float32))
    pts = _query_points()
    x = torch.as_tensor(pts, dtype=torch.float64)

    u_base = inst._base_potential(x).numpy().reshape(-1)
    # Matches the SH potential at the same degree (the acceleration baseline's field)...
    for i, p in enumerate(pts):
        assert u_base[i] == pytest.approx(gm.potential_fixed(p, degree=N), rel=1e-6)
    # ...and is NOT the bare monopole (the old, inconsistent behavior).
    monopole = np.array([float(MU_MOON) / np.linalg.norm(p) for p in pts])
    assert np.any(np.abs(u_base - monopole) / np.abs(monopole) > 1e-9)


def test_base_potential_gradient_matches_base_acceleration():
    """Finite-difference of the potential baseline must equal the accel baseline (a=+grad U)."""
    N = 6
    c, s = _synthetic_field(N)
    gm = GravityModel.from_arrays(degree_max=N, r_ref=float(R_MOON), mu=float(MU_MOON),
                                  c_coeffs_full=c, s_coeffs_full=s)
    inst = _bare(baseline_gravity_model=gm, baseline_degree=N,
                 _mu_tensor=torch.tensor(float(MU_MOON), dtype=torch.float64))
    p = np.array([R_MOON + 80e3, 1.0e5, -5.0e4], dtype=np.float64)
    a_base = inst._base_acceleration(torch.as_tensor(p.reshape(1, 3))).numpy().reshape(-1)

    h = 50.0  # m
    a_fd = np.empty(3)
    for k in range(3):
        pp, pm = p.copy(), p.copy()
        pp[k] += h
        pm[k] -= h
        up = inst._base_potential(torch.as_tensor(pp.reshape(1, 3))).item()
        um = inst._base_potential(torch.as_tensor(pm.reshape(1, 3))).item()
        a_fd[k] = (up - um) / (2.0 * h)  # a = +grad(U)
    assert np.linalg.norm(a_fd - a_base) / np.linalg.norm(a_base) < 1e-5


def _fd_total_accel(inst, p, h=50.0):
    """Central difference of the total potential (a = +grad U); h in m."""
    a_fd = np.empty(3)
    for k in range(3):
        pp, pm = p.copy(), p.copy()
        pp[k] += h
        pm[k] -= h
        a_fd[k] = (inst.predict_potential_and_acceleration_fixed(pp)[0].item()
                   - inst.predict_potential_and_acceleration_fixed(pm)[0].item()) / (2.0 * h)
    return a_fd


# dU = c . x  ->  grad(dU) = c  [m/s^2]. With a_sign = -1 the residual
# acceleration is -c; an unsigned dU puts +c into the potential gradient, a 2|c|
# (~7.5e-3 m/s^2, ~6e-3 of the ~1.3 m/s^2 monopole at 80 km) mismatch that misses
# the 1e-5 relative tolerance by ~500x. FD truncation of mu/r at h = 50 m is ~1e-9.
_C_RES = np.array([3.0e-3, -1.0e-3, 2.0e-3])
_P_TEST = np.array([R_MOON + 80e3, 1.0e5, -5.0e4], dtype=np.float64)  # Moon-fixed, m


def test_total_potential_and_acceleration_agree_for_a_negative_a_sign():
    """a_sign = -1 (physics convention): the residual potential joins the baseline as a_sign * dU.

    Ported from Sidera 4a8784c1. The baseline is geodesy-convention
    (a = +grad U); the network's residual acceleration is a_sign * grad(dU).
    Adding dU unsigned made the total potential's gradient disagree with the
    total acceleration.
    """

    class _Runtime:
        def predict_residual_potential(self, pos):
            return np.asarray(pos, dtype=np.float64).reshape(-1, 3) @ _C_RES

        def predict_residual_accel(self, pos):
            return -1.0 * np.tile(_C_RES, (np.asarray(pos).reshape(-1, 3).shape[0], 1))

    inst = _bare(
        _force_runtime=_Runtime(), a_sign=-1.0, training_mode="residual_potential", device="cpu",
        _torch_dtype=torch.float64, baseline_gravity_model=None,
        _mu_tensor=torch.tensor(float(MU_MOON), dtype=torch.float64),
    )
    _u, a = inst.predict_potential_and_acceleration_fixed(_P_TEST)
    a_fd = _fd_total_accel(inst, _P_TEST)
    assert np.linalg.norm(a_fd - a.reshape(-1)) / np.linalg.norm(a) < 1e-5


def test_legacy_path_potential_and_acceleration_agree_for_a_negative_a_sign():
    """Same consistency on the local (non-canonical-runtime) autograd path."""

    class _Linear(torch.nn.Module):
        def forward(self, x):
            return (x @ torch.as_tensor(_C_RES, dtype=x.dtype)).reshape(-1, 1)

    inst = _bare(
        _force_runtime=None, a_sign=-1.0, training_mode="residual_potential", device="cpu",
        _torch_dtype=torch.float64, baseline_gravity_model=None, model=_Linear(),
        _mu_tensor=torch.tensor(float(MU_MOON), dtype=torch.float64),
        _x_mean=torch.zeros(3, dtype=torch.float64), _x_scale=1.0,
        _u_mean=torch.zeros(1, dtype=torch.float64), _u_scale=1.0,
    )
    _u, a = inst.predict_potential_and_acceleration_fixed(_P_TEST)
    a_mono = -float(MU_MOON) * _P_TEST / np.linalg.norm(_P_TEST) ** 3
    np.testing.assert_allclose(a.reshape(-1), a_mono - _C_RES, rtol=1e-12)
    a_fd = _fd_total_accel(inst, _P_TEST)
    assert np.linalg.norm(a_fd - a.reshape(-1)) / np.linalg.norm(a) < 1e-5


def test_base_potential_monopole_when_no_baseline():
    inst = _bare(baseline_gravity_model=None,
                 _mu_tensor=torch.tensor(float(MU_MOON), dtype=torch.float64))
    p = np.array([R_MOON + 100e3, 0.0, 0.0], dtype=np.float64)
    u = inst._base_potential(torch.as_tensor(p.reshape(1, 3))).item()
    assert u == pytest.approx(float(MU_MOON) / np.linalg.norm(p), rel=1e-12)
