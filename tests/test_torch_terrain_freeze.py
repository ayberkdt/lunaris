"""
Phase D core - torch terrain sampler + segment intersection (CPU).

These exercise the pure-tensor terrain logic that the CUDA-gated torch batch
propagators use, so they run on a CPU-only torch install (identity frame, no
CUDA). The terrain radius sampler must agree with the loader/CPU ground truth
(``sample_topo_radius_m``), and ``terrain_segment_intersection`` must freeze a
descending segment on the terrain surface - deeper for a crater, shallower for a
mountain - and degenerate to the sphere result for uniform terrain.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.requires_torch

from lunaris.common.constants import R_MOON  # noqa: E402
from lunaris.core.torch_frame import (  # noqa: E402
    line_sphere_intersection,
    sample_topo_radius_torch,
    terrain_segment_intersection,
)
from lunaris.loaders.io_surface import sample_topo_radius_m  # noqa: E402

# Synthetic regional grid (interior sampling only -> wrap conventions agree).
_RES = 1.0
_LAT_MIN, _LAT_MAX = 10.0, 20.0
_LON_W, _LON_E = 30.0, 50.0
_LINES = 10
_SAMPLES = 20
_SCALE_M = 0.001 * 1000.0       # DN -> m of height (radius scale)
_BIAS_M = 1737.4 * 1000.0       # reference radius [m]


def _varied_dn() -> np.ndarray:
    i = np.arange(_LINES, dtype=np.float64)[:, None]
    j = np.arange(_SAMPLES, dtype=np.float64)[None, :]
    return 100.0 * i + 10.0 * j  # min 0 @ (0,0), max 1090 @ (9,19)


def _payload(dn: np.ndarray) -> dict:
    """Build a `_grid_topo_payload`-shaped dict for a non-flipped regional grid."""
    lat_centers = np.linspace(_LAT_MAX - 0.5 * _RES, _LAT_MIN + 0.5 * _RES, _LINES)
    lon0 = _LON_W + 0.5 * _RES
    r = _SCALE_M * dn + _BIAS_M
    return {
        "dn": dn,
        "n_lines": _LINES,
        "n_samples": _SAMPLES,
        "res_deg": _RES,
        "lon0_deg": float(lon0),
        "lat0_deg": float(lat_centers[0]),
        "scale_m": _SCALE_M,
        "bias_m": _BIAS_M,
        "missing_dn": float("nan"),
        "flip_lat": 0,
        "lat_min_deg": float(lat_centers.min()),
        "lat_max_deg": float(lat_centers.max()),
        "r_terrain_min_m": float(r.min()),
        "r_terrain_max_m": float(r.max()),
        "radius_const_m": float(R_MOON),
    }


def _to_torch_payload(payload: dict) -> dict:
    tp = dict(payload)
    tp["dn"] = torch.tensor(np.asarray(payload["dn"], dtype=np.float64), dtype=torch.float64)
    return tp


class _IdentityFrame:
    uses_rotation = False

    def inertial_to_fixed(self, t_s, vectors):
        return vectors


_INTERIOR = [(15.0, 40.0), (12.3, 33.7), (18.9, 47.1), (11.0, 31.0), (19.0, 49.0)]


def test_sample_topo_radius_torch_matches_cpu_reference():
    np_payload = _payload(_varied_dn())
    tp = _to_torch_payload(np_payload)

    lat = torch.tensor([p[0] for p in _INTERIOR], dtype=torch.float64)
    lon = torch.tensor([p[1] for p in _INTERIOR], dtype=torch.float64)
    got = sample_topo_radius_torch(tp, lat, lon).numpy()

    for k, (la, lo) in enumerate(_INTERIOR):
        ref = sample_topo_radius_m(np_payload, la, lo)
        assert got[k] == pytest.approx(ref, abs=1e-6)


def test_sample_topo_radius_torch_constant_fallback():
    tp = {"radius_const_m": float(R_MOON)}
    lat = torch.zeros(3, dtype=torch.float64)
    lon = torch.zeros(3, dtype=torch.float64)
    out = sample_topo_radius_torch(tp, lat, lon).numpy()
    assert np.allclose(out, float(R_MOON))


def _uniform_payload(terrain_radius_m: float) -> dict:
    """Uniform terrain == sphere of the given radius (for clean assertions)."""
    dn = np.full((_LINES, _SAMPLES), (terrain_radius_m - _BIAS_M) / _SCALE_M, dtype=np.float64)
    return _payload(dn)


def _radial_segment(r_prev_m: float, r_curr_m: float):
    # Along +x so the sub-point lat/lon is well-defined and inside the patch is
    # irrelevant for a uniform grid.
    p_prev = torch.tensor([[r_prev_m, 0.0, 0.0]], dtype=torch.float64)
    p_curr = torch.tensor([[r_curr_m, 0.0, 0.0]], dtype=torch.float64)
    return p_prev, p_curr


@pytest.mark.parametrize("terrain_r", [R_MOON + 5000.0, R_MOON - 5000.0, R_MOON])
def test_terrain_freeze_matches_uniform_sphere(terrain_r):
    """For uniform terrain the bisection crossing must equal the sphere crossing."""
    tp = _to_torch_payload(_uniform_payload(terrain_r))
    p_prev, p_curr = _radial_segment(R_MOON + 8000.0, R_MOON - 8000.0)

    hit, alpha = terrain_segment_intersection(
        p_prev, p_curr,
        t_prev_s=0.0, dt_s=60.0,
        frame=_IdentityFrame(), topo=tp, impact_alt_m=0.0,
    )
    assert bool(hit[0])

    sph_hit, sph_alpha = line_sphere_intersection(p_prev, p_curr, terrain_r)
    assert bool(sph_hit[0])
    assert float(alpha[0]) == pytest.approx(float(sph_alpha[0]), abs=1e-6)

    # Crossing radius lands on the terrain surface.
    cross = p_prev + alpha.unsqueeze(1) * (p_curr - p_prev)
    r_cross = float(torch.linalg.norm(cross[0]))
    assert r_cross == pytest.approx(terrain_r, abs=1.0)


def test_terrain_freeze_crater_is_deeper_than_mountain():
    p_prev, p_curr = _radial_segment(R_MOON + 8000.0, R_MOON - 8000.0)
    frame = _IdentityFrame()

    _, a_mtn = terrain_segment_intersection(
        p_prev, p_curr, t_prev_s=0.0, dt_s=60.0,
        frame=frame, topo=_to_torch_payload(_uniform_payload(R_MOON + 5000.0)),
        impact_alt_m=0.0,
    )
    _, a_crater = terrain_segment_intersection(
        p_prev, p_curr, t_prev_s=0.0, dt_s=60.0,
        frame=frame, topo=_to_torch_payload(_uniform_payload(R_MOON - 5000.0)),
        impact_alt_m=0.0,
    )
    # Crater floor is lower -> crossing happens later along the descending segment.
    assert float(a_crater[0]) > float(a_mtn[0])


def test_terrain_freeze_no_impact_when_endpoint_above_terrain():
    """Segment that stays above the mountain must not register an impact."""
    tp = _to_torch_payload(_uniform_payload(R_MOON + 5000.0))
    p_prev, p_curr = _radial_segment(R_MOON + 9000.0, R_MOON + 6000.0)  # never below 5000
    hit, alpha = terrain_segment_intersection(
        p_prev, p_curr, t_prev_s=0.0, dt_s=60.0,
        frame=_IdentityFrame(), topo=tp, impact_alt_m=0.0,
    )
    assert not bool(hit[0])
    assert float(alpha[0]) == pytest.approx(1.0)


def test_terrain_freeze_respects_impact_alt_threshold():
    """impact_alt raises the trigger surface to terrain_r + impact_alt."""
    terrain_r = R_MOON + 1000.0
    impact_alt = 2000.0
    tp = _to_torch_payload(_uniform_payload(terrain_r))
    p_prev, p_curr = _radial_segment(R_MOON + 8000.0, R_MOON - 8000.0)
    hit, alpha = terrain_segment_intersection(
        p_prev, p_curr, t_prev_s=0.0, dt_s=60.0,
        frame=_IdentityFrame(), topo=tp, impact_alt_m=impact_alt,
    )
    assert bool(hit[0])
    cross = p_prev + alpha.unsqueeze(1) * (p_curr - p_prev)
    r_cross = float(torch.linalg.norm(cross[0]))
    assert r_cross == pytest.approx(terrain_r + impact_alt, abs=1.0)
