"""
Terrain-aware impact freeze - Phase A & B.

Phase A: ``loaders.io_surface.grid_topo_payload`` packages a topography grid into
a plain POD dict, and ``sample_topo_radius_m`` reconstructs the surface radius
[m] from it using the *exact* indexing convention the batch impact kernels use.
The reference sampler must round-trip against the loader's own
``TopographyGrid.radius_m`` so it can serve as the CPU ground truth for the
GPU/numba terrain-freeze kernels (Phase C/D).

Phase B: ``core.propagation.propagator.build_events`` must wire a real terrain-aware impact
event (``make_hybrid_impact_event``) when a topo grid is present, with the
near-field switch altitude sourced from the config single-source default
(11 km) - never the latent ``0.0`` fallback that silently disables terrain.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from lunaris.common.constants import R_MOON
from lunaris.common.type_defs import PropagatorConfig
from lunaris.core.propagation.events import _find_event_index
from lunaris.core.propagation.propagator import build_events
from lunaris.loaders.io_surface import (
    ConstantTopography,
    InMemorySurfaceProvider,
    TopographyGrid,
    grid_dem_provenance,
    grid_topo_payload,
    sample_topo_radius_m,
)

# Synthetic product geometry: a small regional LDEM patch (interior sampling
# only, so the loader's [west, east) longitude wrap and the payload's [0, 360)
# wrap agree exactly).
_PPD = 1.0
_LAT_MIN, _LAT_MAX = 10.0, 20.0
_LON_W, _LON_E = 30.0, 50.0
_LINES = int((_LAT_MAX - _LAT_MIN) * _PPD)      # 10
_SAMPLES = int((_LON_E - _LON_W) * _PPD)        # 20
_SCALING_KM = 0.001                              # DN -> km  (i.e. DN is height in m)
_OFFSET_KM = 1737.4                              # reference radius [km]


def _synthetic_dn() -> np.ndarray:
    """Deterministic, non-separable-but-smooth DN grid with known extremes."""
    i = np.arange(_LINES, dtype=np.float64)[:, None]
    j = np.arange(_SAMPLES, dtype=np.float64)[None, :]
    dn = 100.0 * i + 10.0 * j  # min at (0,0)=0, max at (9,19)=1090
    return np.ascontiguousarray(dn.astype("<i2"))


def _write_synthetic_ldem(tmp_path) -> TopographyGrid:
    """Write a minimal detached PDS3 LDEM label + IMG and open it."""
    dn = _synthetic_dn()
    img_path = tmp_path / "synthetic.img"
    img_path.write_bytes(dn.tobytes(order="C"))

    label = f"""
PDS_VERSION_ID = PDS3
RECORD_BYTES = {_SAMPLES * 2}
^IMAGE = "synthetic.img"
PRODUCT_ID = "LDEM_SYNTH"

OBJECT = IMAGE
  LINES = {_LINES}
  LINE_SAMPLES = {_SAMPLES}
  SAMPLE_TYPE = LSB_INTEGER
  SAMPLE_BITS = 16
  UNIT = METER
  SCALING_FACTOR = {_SCALING_KM}
  OFFSET = {_OFFSET_KM}
END_OBJECT = IMAGE

MAP_PROJECTION_TYPE = "SIMPLE_CYLINDRICAL"
MAP_RESOLUTION = {_PPD} <PIX/DEG>
MAXIMUM_LATITUDE = {_LAT_MAX}
MINIMUM_LATITUDE = {_LAT_MIN}
WESTERNMOST_LONGITUDE = {_LON_W}
EASTERNMOST_LONGITUDE = {_LON_E}
POSITIVE_LONGITUDE_DIRECTION = "EAST"
CENTER_LATITUDE = {(_LAT_MIN + _LAT_MAX) / 2.0} <DEG>
CENTER_LONGITUDE = {(_LON_W + _LON_E) / 2.0} <DEG>
A_AXIS_RADIUS = {_OFFSET_KM}
B_AXIS_RADIUS = {_OFFSET_KM}
C_AXIS_RADIUS = {_OFFSET_KM}
END
""".lstrip()
    lbl_path = tmp_path / "synthetic.lbl"
    lbl_path.write_text(label, encoding="utf-8")

    return TopographyGrid(lbl_path, img_path, mmap=False)


# ---------------------------------------------------------------------------
# Phase A - payload + reference sampler
# ---------------------------------------------------------------------------

# Interior points only (away from the patch boundary) so loader/payload wraps agree.
_INTERIOR_POINTS = [(15.0, 40.0), (12.3, 33.7), (18.9, 47.1), (11.0, 31.0), (19.0, 49.0)]


def test_dem_provenance_records_product_resolution_and_hash(tmp_path):
    topo = _write_synthetic_ldem(tmp_path)
    prov = grid_dem_provenance(topo, r_moon_m=float(R_MOON))

    assert prov is not None
    assert prov["label_name"] == "synthetic.lbl"
    assert prov["img_name"] == "synthetic.img"
    assert prov["img_sha256"] and len(prov["img_sha256"]) == 64
    assert prov["map_resolution_ppd"] == pytest.approx(_PPD)
    assert prov["res_deg"] == pytest.approx(1.0 / _PPD)
    # Ground sample distance = one pixel of arc on the datum radius (offset_km).
    expected_gsd = math.radians(1.0 / _PPD) * (_OFFSET_KM * 1000.0)
    assert prov["ground_sample_distance_m"] == pytest.approx(expected_gsd)
    assert prov["datum_a_axis_radius_km"] == pytest.approx(_OFFSET_KM)
    assert prov["lat_min_deg"] == pytest.approx(_LAT_MIN)
    assert prov["lat_max_deg"] == pytest.approx(_LAT_MAX)


def test_dem_provenance_none_for_constant_and_missing_grid():
    assert grid_dem_provenance(None) is None
    # ConstantTopography has no `.info`, so there is no DEM provenance to record.
    assert grid_dem_provenance(ConstantTopography(R_MOON + 100.0)) is None


def test_topo_payload_round_trips_against_loader(tmp_path):
    topo = _write_synthetic_ldem(tmp_path)
    payload = grid_topo_payload(topo, r_moon_m=float(R_MOON))

    # Sanity: it is a real grid payload, not the constant fallback.
    assert "dn" in payload
    assert payload["n_lines"] == _LINES
    assert payload["n_samples"] == _SAMPLES
    assert payload["scale_m"] == pytest.approx(_SCALING_KM * 1000.0)
    assert payload["bias_m"] == pytest.approx(_OFFSET_KM * 1000.0)

    for lat_deg, lon_deg in _INTERIOR_POINTS:
        from_payload = sample_topo_radius_m(payload, lat_deg, lon_deg)
        from_loader = topo.sample_bilinear(lat_deg, lon_deg, kind="radius_m")
        assert from_payload == pytest.approx(from_loader, rel=0.0, abs=1e-6)


def test_topo_payload_envelope_matches_dn_extremes(tmp_path):
    topo = _write_synthetic_ldem(tmp_path)
    payload = grid_topo_payload(topo, r_moon_m=float(R_MOON))

    dn = np.asarray(_synthetic_dn(), dtype=np.float64)
    scale_m = _SCALING_KM * 1000.0
    bias_m = _OFFSET_KM * 1000.0
    expected_min = float(dn.min()) * scale_m + bias_m
    expected_max = float(dn.max()) * scale_m + bias_m

    assert payload["r_terrain_min_m"] == pytest.approx(expected_min)
    assert payload["r_terrain_max_m"] == pytest.approx(expected_max)
    assert payload["r_terrain_min_m"] <= payload["r_terrain_max_m"]

    # Every interior sample must lie within the advertised envelope (the kernel
    # relies on this to bound near-field terrain lookups).
    for lat_deg, lon_deg in _INTERIOR_POINTS:
        r = sample_topo_radius_m(payload, lat_deg, lon_deg)
        assert payload["r_terrain_min_m"] - 1e-6 <= r <= payload["r_terrain_max_m"] + 1e-6


def test_topo_payload_prefers_explicit_ddeg_over_info_ppd(tmp_path):
    """Grid-like adapters may expose ``ddeg`` without repeating map_resolution_ppd."""

    topo = _write_synthetic_ldem(tmp_path)

    class _InfoWithoutPPD:
        lines = topo.info.lines
        samples = topo.info.samples
        scaling_factor = topo.info.scaling_factor
        offset_km = topo.info.offset_km
        missing_constant = getattr(topo.info, "missing_constant", float("nan"))

    class _GridLikeTopo:
        info = _InfoWithoutPPD()
        dn_km = topo.dn_km
        ddeg = topo.ddeg
        _lon_centers_deg = topo._lon_centers_deg
        _lat_centers_deg = topo._lat_centers_deg
        _flip_lat = topo._flip_lat

    payload = grid_topo_payload(_GridLikeTopo(), r_moon_m=float(R_MOON))

    assert "dn" in payload
    assert payload["res_deg"] == pytest.approx(topo.ddeg)


def test_topo_payload_none_and_constant_fall_back_to_radius_const():
    # No grid at all.
    p_none = grid_topo_payload(None, r_moon_m=float(R_MOON))
    assert p_none == {"radius_const_m": pytest.approx(float(R_MOON))}
    assert sample_topo_radius_m(p_none, 12.0, 34.0) == pytest.approx(float(R_MOON))

    # A non-grid provider (ConstantTopography lacks .info/.dn_km) -> constant.
    p_const = grid_topo_payload(ConstantTopography(R_MOON + 1234.0), r_moon_m=float(R_MOON))
    assert p_const == {"radius_const_m": pytest.approx(float(R_MOON))}


def test_inmemory_provider_topo_payload_wraps_injected_grid(tmp_path):
    topo = _write_synthetic_ldem(tmp_path)
    prov = InMemorySurfaceProvider(topo=topo, default_radius_m=float(R_MOON))
    payload = prov.topo_payload()
    assert "dn" in payload

    # Provider with no topo -> constant fallback.
    empty = InMemorySurfaceProvider(topo=None, default_radius_m=float(R_MOON))
    assert empty.topo_payload() == {"radius_const_m": pytest.approx(float(R_MOON))}


# ---------------------------------------------------------------------------
# Phase B - CPU build_events terrain-aware wiring (ground truth)
# ---------------------------------------------------------------------------

class _FakeTables:
    dt_s = 100.0
    # N=2 identity quaternions -> r_i_to_bf is the identity rotation.
    q_i2f_tab = np.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], dtype=np.float64)


class _FakeEphem:
    tables = _FakeTables()


class _FakeDynamics:
    """Minimal duck for build_events: point-mass ref radius + a rotation table."""
    grav = None  # -> get_ref_radius_and_mu falls back to (R_MOON, MU_MOON)
    ephem = _FakeEphem()


class _FlatTerrain:
    """radius_m_deg interface returning a constant terrain radius."""
    def __init__(self, terrain_radius_m: float) -> None:
        self._r = float(terrain_radius_m)

    def radius_m_deg(self, lat_deg: float, lon_deg: float) -> float:
        return self._r


def _impact_event(topo):
    # EventConfig defaults: detect_impact=True, impact_alt_km=0.0;
    # PropagatorConfig default: hybrid_switch_alt_m=11_000.0.
    cfg = PropagatorConfig()
    events = build_events(_FakeDynamics(), cfg, topo_grid=topo, add_stop_event=False)
    idx = _find_event_index(events, "impact")
    assert idx is not None
    return events[idx]


def test_build_events_uses_terrain_near_field_for_topo_grid():
    """At a sub-switch altitude the impact event must reflect the terrain radius,
    not the reference sphere - proving switch_alt is the 11 km default, not 0."""
    alt_m = 3000.0  # well below the 11 km hybrid switch altitude
    r = float(R_MOON) + alt_m
    y = np.array([r, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

    mountain = _impact_event(_FlatTerrain(float(R_MOON) + 5000.0))
    crater = _impact_event(_FlatTerrain(float(R_MOON) - 5000.0))

    val_mountain = mountain(0.0, y)
    val_crater = crater(0.0, y)

    # Terrain-aware: alt_terrain = ||r|| - terrain_radius.
    assert val_mountain == pytest.approx(alt_m - 5000.0)   # -2000: below the peak
    assert val_crater == pytest.approx(alt_m + 5000.0)     # +8000: above the floor

    # If switch_alt had defaulted to 0, both would equal the sphere altitude.
    assert val_mountain != pytest.approx(alt_m)


def test_build_events_far_field_is_sphere_regardless_of_terrain():
    """Above the switch altitude the event is the cheap reference-sphere altitude."""
    alt_m = 50_000.0  # far above the 11 km switch
    r = float(R_MOON) + alt_m
    y = np.array([r, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

    mountain = _impact_event(_FlatTerrain(float(R_MOON) + 5000.0))
    crater = _impact_event(_FlatTerrain(float(R_MOON) - 5000.0))

    assert mountain(0.0, y) == pytest.approx(alt_m)
    assert crater(0.0, y) == pytest.approx(alt_m)


def test_build_events_without_topo_is_pure_sphere():
    alt_m = 3000.0
    r = float(R_MOON) + alt_m
    y = np.array([r, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    ev = _impact_event(None)
    assert ev(0.0, y) == pytest.approx(alt_m)
