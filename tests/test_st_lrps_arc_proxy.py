"""Arc-proxy math (tools/st_lrps_arc_proxy.py): adjoint sensitivity against the
Clohessy-Wiltshire secular drift, Hermite resampling, Simpson weights, scoring."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location("st_lrps_arc_proxy", _REPO_ROOT / "tools" / "st_lrps_arc_proxy.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ap = _load()
MU = 4902800306330.2


def _circular(a: float, t: np.ndarray):
    n = math.sqrt(MU / a**3)
    r = np.stack([a * np.cos(n * t), a * np.sin(n * t), 0 * t], axis=1)
    v = np.stack([-a * n * np.sin(n * t), a * n * np.cos(n * t), 0 * t], axis=1)
    return n, r, v


def _point_mass_gradient(r: np.ndarray) -> np.ndarray:
    rn = np.linalg.norm(r, axis=1)
    rh = r / rn[:, None]
    eye = np.eye(3)[None]
    return (MU / rn**3)[:, None, None] * (3 * np.einsum("ni,nj->nij", rh, rh) - eye)


def test_adjoint_reproduces_the_cw_secular_along_track_drift():
    # Constant tangential force f on a circular orbit: along-track error grows
    # as -3/2 f t^2 plus a bounded periodic part of size ~4 f / n^2.
    a, f0, h = 1.838e6, 1e-8, 10.0
    t_half = np.arange(0.0, 2 * 86400.0 + 0.25 * h, 0.5 * h)
    t_half = t_half[: len(t_half) - (len(t_half) - 1) % 4]
    n, r_half, v_half = _circular(a, t_half)
    t, r, v = t_half[0::2], r_half[0::2], v_half[0::2]
    f = f0 * v / np.linalg.norm(v, axis=1, keepdims=True)
    c = ap.along_track_unit(r[-1], v[-1])
    pv = ap.adjoint_velocity_sensitivity(_point_mass_gradient(r_half), c, h)
    s = float(np.sum(ap.simpson_weights(len(t), h)[:, None] * pv * f))
    secular = -1.5 * f0 * t[-1] ** 2
    assert abs(s - secular) < 4 * f0 / n**2 + 1e-3 * abs(secular)
    with pytest.raises(ValueError, match="half-step"):
        ap.adjoint_velocity_sensitivity(_point_mass_gradient(r_half[:-1]), c, h)


def test_adjoint_is_linear_in_the_forcing():
    h = 10.0
    t_half = np.arange(0.0, 6 * 3600.0 + 0.25 * h, 0.5 * h)[:2161]
    _, r_half, v_half = _circular(1.9e6, t_half)
    t, r, v = t_half[0::2], r_half[0::2], v_half[0::2]
    pv = ap.adjoint_velocity_sensitivity(_point_mass_gradient(r_half), ap.along_track_unit(r[-1], v[-1]), h)
    w = ap.simpson_weights(len(t), h)[:, None] * pv
    g = np.random.default_rng(0)
    f1, f2 = g.normal(size=r.shape) * 1e-9, g.normal(size=r.shape) * 1e-9
    zero = np.zeros_like(f1)
    s1, s2 = ap.score_arrays(f1, zero, w), ap.score_arrays(f2, zero, w)
    assert ap.score_arrays(2 * f1 - 3 * f2, zero, w) == pytest.approx(2 * s1 - 3 * s2, rel=1e-12)
    assert ap.score_arrays(f1, f1, w) == 0.0


def test_hermite_resampling_is_exact_for_cubic_motion():
    t = np.arange(0.0, 601.0, 60.0)
    coef = np.array([[1.0, -2.0, 0.5], [3.0, 0.1, -1.0], [0.2, 0.3, 0.4], [1e-3, -2e-3, 5e-4]])
    r = sum(coef[k] * t[:, None] ** k for k in range(4))
    v = sum(k * coef[k] * t[:, None] ** (k - 1) for k in range(1, 4))
    tt, rr, vv = ap.hermite_resample(t, r, v, 5.0)
    r_exact = sum(coef[k] * tt[:, None] ** k for k in range(4))
    v_exact = sum(k * coef[k] * tt[:, None] ** (k - 1) for k in range(1, 4))
    np.testing.assert_allclose(rr, r_exact, rtol=1e-12, atol=1e-6)
    np.testing.assert_allclose(vv, v_exact, rtol=1e-10, atol=1e-8)


def test_simpson_weights_integrate_cubics_exactly_and_need_an_odd_count():
    x = np.linspace(0.0, 2.0, 21)
    w = ap.simpson_weights(len(x), x[1] - x[0])
    assert float(np.sum(w * (x**3 - x + 1))) == pytest.approx(4.0 - 2.0 + 2.0, rel=1e-12)
    with pytest.raises(ValueError, match="odd"):
        ap.simpson_weights(20, 0.1)


def test_quaternion_matrix_matches_the_repository_rotation_convention():
    from lunaris.common.math_utils import quat_rotate_np

    q = np.array([0.9, 0.1, -0.3, 0.2])
    q /= np.linalg.norm(q)
    v = np.array([1.0, -2.0, 0.5])
    np.testing.assert_allclose(ap.quat_to_matrix(q) @ v, quat_rotate_np(q, v), atol=1e-14)


@pytest.mark.parametrize("n", range(5, 60))
def test_half_grid_keep_gives_odd_grids_for_simpson_and_rk4(n):
    keep = ap.half_grid_keep(n)
    assert n - 3 <= keep <= n and keep % 4 == 1
    assert len(range(0, keep, 2)) % 2 == 1


def test_cache_key_tracks_the_proxy_and_the_checkpoint(tmp_path):
    proxy = tmp_path / "proxy"
    proxy.mkdir()
    (proxy / "proxy_manifest.json").write_text('{"step_s": 5}', encoding="utf-8")
    run = tmp_path / "run"
    (run / "checkpoints").mkdir(parents=True)
    (run / "checkpoints" / "ckpt_last.pt").write_bytes(b"a")
    k1 = ap.cache_key(proxy, run, "last")
    assert ap.cache_key(proxy, run, "last") == k1
    (proxy / "proxy_manifest.json").write_text('{"step_s": 10}', encoding="utf-8")
    assert ap.cache_key(proxy, run, "last") != k1
    assert ap.cache_key(proxy, run, "best") != ap.cache_key(proxy, run, "last")


def test_quaternion_matrix_is_a_rotation():
    q = np.array([0.9, 0.1, -0.3, 0.2])
    q /= np.linalg.norm(q)
    m = ap.quat_to_matrix(q)
    np.testing.assert_allclose(m @ m.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(m) == pytest.approx(1.0)
