"""MOON_PA -> MOON_ME map rotation for terrain sampling, on every backend.

LOLA topography is tabulated on mean-Earth/polar axes (DE421) while the
attitude tables rotate into the DE440 principal-axes frame. The constant
rotation between them comes from the DE440 lunar frame kernel's TK definition
(``TKFRAME_31009``); these tests pin

* the matrix against its definition and, when the kernels are present,
  against SPICE ``pxform("MOON_PA", "MOON_ME", et)``;
* that the CPU impact/telemetry mapper, the torch latitude/longitude helper,
  the Numba CUDA terrain residual (under the simulator) and the batch engine's
  payload all apply the same rotation, so the four terrain paths sample the
  map at the same point.
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from lunaris.common.lunar_map_frame import (
    MOON_PA_DE440_TO_ME_ANGLES_ARCSEC,
    ephemeris_fixed_frame,
    moon_pa_de440_to_me_matrix,
    surface_map_rotation,
)


def _q_z(angle_rad: float) -> np.ndarray:
    return np.array([math.cos(angle_rad / 2), 0.0, 0.0, math.sin(angle_rad / 2)])


def test_pa_to_me_matrix_is_the_tk_frame_rotation() -> None:
    m = moon_pa_de440_to_me_matrix()
    np.testing.assert_allclose(m @ m.T, np.eye(3), atol=1e-15)
    assert abs(np.linalg.det(m) - 1.0) < 1e-15
    angle_arcsec = math.degrees(math.acos((np.trace(m) - 1.0) / 2.0)) * 3600.0
    # Rotation angle of the 3-2-1 sequence (67.85", 78.69", 0.28"): ~103.9".
    assert abs(angle_arcsec - 103.9) < 0.05, angle_arcsec
    # First-order check of the 3-2-1 frame-rotation sequence: ME->PA ~ I - [w]x
    # with w = (a3, a2, a1), so PA->ME ~ I + [w]x; second-order terms are
    # O(a^2) ~ 1.5e-7 for a ~ 3.8e-4 rad.
    a1, a2, a3 = (math.radians(a / 3600.0) for a in MOON_PA_DE440_TO_ME_ANGLES_ARCSEC)
    approx_pa_to_me = np.array([[1.0, -a1, a2], [a1, 1.0, -a3], [-a2, a3, 1.0]])
    np.testing.assert_allclose(m, approx_pa_to_me, rtol=0.0, atol=3e-7)
    # The transposed sequence would be off by the full first-order term.
    assert np.max(np.abs(m.T - approx_pa_to_me)) > 1e-4


def test_surface_map_rotation_only_for_de440_principal_axes() -> None:
    for name in ("MOON_PA", "moon_pa_de440", " MOON_PA "):
        np.testing.assert_array_equal(surface_map_rotation(name), moon_pa_de440_to_me_matrix())
    for name in (None, "MOON_ME", "IAU_MOON", "J2000", "MOON_PA_DE421"):
        assert surface_map_rotation(name) is None


def test_ephemeris_fixed_frame_reads_tables_then_owner() -> None:
    assert ephemeris_fixed_frame(None) is None
    assert ephemeris_fixed_frame(SimpleNamespace(tables=SimpleNamespace(fixed_frame="MOON_PA"))) == "MOON_PA"
    assert ephemeris_fixed_frame(SimpleNamespace(fixed_frame="MOON_ME")) == "MOON_ME"
    assert ephemeris_fixed_frame(SimpleNamespace(tables=None)) is None


@pytest.mark.requires_data
def test_pa_to_me_matrix_matches_spice_pxform() -> None:
    sp = pytest.importorskip("spiceypy")
    from lunaris.common.paths import data_dir_from_root, find_project_root

    eph = data_dir_from_root(find_project_root()) / "ephemeris_models"
    kernels = [eph / "naif0012.tls", eph / "moon_de440_250416.tf", eph / "moon_pa_de440_200625.bpc"]
    if not all(k.exists() for k in kernels):
        pytest.skip("DE440 lunar frame kernels not available")
    for k in kernels:
        sp.furnsh(str(k))
    try:
        m = moon_pa_de440_to_me_matrix()
        for utc in ("1990-06-01T00:00:00", "2000-01-01T12:00:00", "2026-10-11T00:00:00"):
            et = sp.str2et(utc)
            np.testing.assert_allclose(m, np.array(sp.pxform("MOON_PA", "MOON_ME", et)), atol=1e-15)
    finally:
        for k in kernels:
            sp.unload(str(k))


def _dynamics_with_frame(frame: str | None, q_tab: np.ndarray) -> SimpleNamespace:
    tables = SimpleNamespace(dt_s=600.0, q_i2f_tab=q_tab, fixed_frame=frame)
    return SimpleNamespace(ephem=SimpleNamespace(tables=tables))


def test_cpu_impact_mapper_composes_pa_to_me_rotation() -> None:
    from lunaris.common.math_utils import quat_rotate_np
    from lunaris.core.propagation.events import (
        _build_r_i_to_bf_from_rot_table,
        _build_r_i_to_surface_map,
    )

    q_tab = np.array([_q_z(0.3), _q_z(0.31)])
    r_i = np.array([1.2e6, -0.4e6, 1.1e6])
    pa = _dynamics_with_frame("MOON_PA", q_tab)
    mapper = _build_r_i_to_surface_map(pa)
    body = _build_r_i_to_bf_from_rot_table(pa)
    assert mapper is not None and body is not None
    np.testing.assert_allclose(mapper(0.0, r_i), moon_pa_de440_to_me_matrix() @ quat_rotate_np(q_tab[0], r_i), atol=1e-6)
    # Already-ME frame: the body-fixed mapper is returned unchanged.
    me = _dynamics_with_frame("MOON_ME", q_tab)
    np.testing.assert_allclose(_build_r_i_to_surface_map(me)(0.0, r_i), body(0.0, r_i), atol=0.0)


def test_torch_latlon_applies_payload_rotation() -> None:
    torch = pytest.importorskip("torch")
    from lunaris.core.torch_frame import topo_latlon_deg, topo_payload_to_torch

    m = moon_pa_de440_to_me_matrix()
    payload = {"dn": np.zeros((2, 2)), "map_rotation": m.tolist()}
    topo = topo_payload_to_torch(payload, device=torch.device("cpu"), dtype=torch.float64)
    p_bf = np.array([[1.0e6, 1.2e6, 0.5e6], [-0.3e6, 0.2e6, -1.6e6]])
    r = np.linalg.norm(p_bf, axis=1)
    lat, lon = topo_latlon_deg(topo, torch.as_tensor(p_bf), torch.as_tensor(r))
    p_me = p_bf @ m.T
    np.testing.assert_allclose(lat.numpy(), np.degrees(np.arcsin(p_me[:, 2] / r)), atol=1e-12)
    np.testing.assert_allclose(lon.numpy(), np.degrees(np.arctan2(p_me[:, 1], p_me[:, 0])), atol=1e-12)
    # Without a rotation the body-fixed angles are used unchanged.
    lat0, _ = topo_latlon_deg({"dn": None}, torch.as_tensor(p_bf), torch.as_tensor(r))
    np.testing.assert_allclose(lat0.numpy(), np.degrees(np.arcsin(p_bf[:, 2] / r)), atol=1e-12)


def test_batch_engine_payload_carries_rotation_for_moon_pa_only() -> None:
    from lunaris.batch.engine import BatchPropagationEngine

    eng = BatchPropagationEngine.__new__(BatchPropagationEngine)
    payload = {"dn": np.zeros((2, 2)), "r_terrain_max_m": 1.0}
    eng._dyn = _dynamics_with_frame("MOON_PA", np.array([_q_z(0.0)]))
    out = eng._with_map_rotation(payload)
    np.testing.assert_array_equal(np.asarray(out["map_rotation"]), moon_pa_de440_to_me_matrix())
    assert "map_rotation" not in payload  # caller's dict untouched
    eng._dyn = _dynamics_with_frame("MOON_ME", np.array([_q_z(0.0)]))
    assert "map_rotation" not in eng._with_map_rotation(payload)
    eng._dyn = SimpleNamespace(ephem=None)
    assert "map_rotation" not in eng._with_map_rotation(payload)


def test_numba_terrain_residual_samples_map_frame_like_cpu() -> None:
    """CUDA residual with the PA->ME rotation == CPU event sampling of the same map."""
    root = Path(__file__).resolve().parents[1]
    script = textwrap.dedent(
        """
        import math
        import numpy as np
        from numba import cuda

        from lunaris.common.lunar_map_frame import moon_pa_de440_to_me_matrix
        from lunaris.common.math_utils import quat_rotate_np
        from lunaris.core.batch_propagator import _terrain_residual_cuda, topo_meta_array
        from lunaris.loaders.io_surface import sample_topo_radius_m

        DATUM = 1737400.0
        res = 0.01  # 0.01 deg cells (~300 m): finer than the ~875 m frame offset
        lat_c = np.arange(25.0 - 0.5 * res, 15.0, -res)
        lon_c = np.arange(15.0 + 0.5 * res, 25.0, res)
        lat_g, lon_g = np.meshgrid(lat_c, lon_c, indexing="ij")
        dn = np.ascontiguousarray(3000.0 * np.sin(np.radians(40.0 * lon_g)) * np.cos(np.radians(30.0 * lat_g)))
        payload = {
            "dn": dn, "n_lines": lat_c.size, "n_samples": lon_c.size, "res_deg": res,
            "lon0_deg": float(lon_c[0]), "lat0_deg": float(lat_c[0]), "scale_m": 1.0,
            "bias_m": DATUM, "missing_dn": float("nan"), "flip_lat": 0,
            "lat_min_deg": float(lat_c.min()), "lat_max_deg": float(lat_c.max()),
            "r_terrain_min_m": DATUM - 3000.0, "r_terrain_max_m": DATUM + 3000.0,
            "radius_const_m": DATUM,
        }
        M = moon_pa_de440_to_me_matrix()
        payload_rot = dict(payload, map_rotation=M.tolist())
        q = np.array([math.cos(0.15), 0.0, 0.0, math.sin(0.15)])  # 0.3 rad about z

        @cuda.jit
        def launch(p, dn, meta, out):
            k = cuda.grid(1)
            if k < p.shape[0]:
                out[k] = _terrain_residual_cuda(
                    0.0, p[k, 0], p[k, 1], p[k, 2], 0.0, 0.0, 0.0,
                    q[0], q[1], q[2], q[3], dn, meta, 0.0,
                )

        rng = np.random.default_rng(3)
        pts = []
        for _ in range(32):
            lat = math.radians(rng.uniform(16.0, 24.0)); lon = math.radians(rng.uniform(16.0, 24.0))
            u_bf = np.array([math.cos(lat) * math.cos(lon), math.cos(lat) * math.sin(lon), math.sin(lat)])
            u_i = quat_rotate_np(np.array([q[0], -q[1], -q[2], -q[3]]), u_bf)
            pts.append((DATUM + 1000.0) * u_i)
        p = np.ascontiguousarray(np.array(pts))

        def cpu(r_i, rot):
            v = quat_rotate_np(q, r_i)
            if rot is not None:
                v = rot @ v
            rn = np.linalg.norm(r_i)
            lat = math.degrees(math.asin(v[2] / rn)); lon = math.degrees(math.atan2(v[1], v[0]))
            return rn - sample_topo_radius_m(payload, lat, lon)

        out_rot = np.zeros(len(pts)); out_id = np.zeros(len(pts))
        launch[1, 32](p, dn, topo_meta_array(payload_rot), out_rot)
        launch[1, 32](p, dn, topo_meta_array(payload), out_id)
        ref_rot = np.array([cpu(x, M) for x in p]); ref_id = np.array([cpu(x, None) for x in p])
        assert np.max(np.abs(out_rot - ref_rot)) < 1e-6, np.max(np.abs(out_rot - ref_rot))
        assert np.max(np.abs(out_id - ref_id)) < 1e-6
        # The ~875 m offset is resolved by this grid: the rotation changes the answer.
        shift = np.max(np.abs(ref_rot - ref_id))
        assert shift > 10.0, shift
        print("OK", shift)
        """
    )
    env = os.environ.copy()
    env["NUMBA_ENABLE_CUDASIM"] = "1"
    env["PYTHONPATH"] = str(root / "src") + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=root, env=env,
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout, result.stdout + result.stderr
