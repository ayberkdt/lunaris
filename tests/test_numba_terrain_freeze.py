"""Phase C: the Numba CUDA terrain-freeze device functions, under cudasim.

numba CUDA is unavailable on most CI/dev machines, so these run the *real*
device functions (``_sample_topo_radius_cuda`` / ``_terrain_residual_cuda``)
on Numba's CPU simulator and assert they agree with the CPU ground truth
(``loaders.io_surface.sample_topo_radius_m``) and the analytic impact residual.
This is the cross-backend contract: the numba kernel must sample terrain and
locate the crossing the same way the torch backend and the CPU event do.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _run_cudasim(body: str) -> subprocess.CompletedProcess[str]:
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["NUMBA_ENABLE_CUDASIM"] = "1"
    src_path = str(root / "src")
    env["PYTHONPATH"] = src_path + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-c", body],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_numba_topo_sampler_matches_cpu_reference() -> None:
    script = """
import numpy as np
from numba import cuda

from lunaris.core.mc_propagator import _sample_topo_radius_cuda, _terrain_residual_cuda
from lunaris.loaders.io_surface import sample_topo_radius_m

# --- Synthetic regional grid (interior sampling only) ---------------------
LINES, SAMPLES = 10, 20
RES = 1.0
LAT_MIN, LAT_MAX = 10.0, 20.0
LON_W = 30.0
SCALE_M = 1.0
BIAS_M = 1737400.0

i = np.arange(LINES, dtype=np.float64)[:, None]
j = np.arange(SAMPLES, dtype=np.float64)[None, :]
dn = np.ascontiguousarray(100.0 * i + 10.0 * j)

lat_centers = np.linspace(LAT_MAX - 0.5 * RES, LAT_MIN + 0.5 * RES, LINES)
lon0 = LON_W + 0.5 * RES
lat0 = float(lat_centers[0])
r = SCALE_M * dn + BIAS_M

meta = np.zeros(13, dtype=np.float64)
meta[0] = LINES; meta[1] = SAMPLES; meta[2] = RES
meta[3] = lon0; meta[4] = lat0; meta[5] = SCALE_M; meta[6] = BIAS_M
meta[7] = 0.0
meta[8] = float(lat_centers.min()); meta[9] = float(lat_centers.max())
meta[10] = float('nan'); meta[11] = float(r.max()); meta[12] = 1737400.0

payload = {
    "dn": dn, "n_lines": LINES, "n_samples": SAMPLES, "res_deg": RES,
    "lon0_deg": lon0, "lat0_deg": lat0, "scale_m": SCALE_M, "bias_m": BIAS_M,
    "missing_dn": float('nan'), "flip_lat": 0,
    "lat_min_deg": float(lat_centers.min()), "lat_max_deg": float(lat_centers.max()),
    "r_terrain_min_m": float(r.min()), "r_terrain_max_m": float(r.max()),
    "radius_const_m": 1737400.0,
}


@cuda.jit
def _sample_launch(lat, lon, dn, meta, out):
    k = cuda.grid(1)
    if k < lat.shape[0]:
        out[k] = _sample_topo_radius_cuda(lat[k], lon[k], dn, meta)


pts = [(15.0, 40.0), (12.3, 33.7), (18.9, 47.1), (11.0, 31.0), (19.0, 49.0)]
lat = np.array([p[0] for p in pts], dtype=np.float64)
lon = np.array([p[1] for p in pts], dtype=np.float64)
out = np.zeros(len(pts), dtype=np.float64)
_sample_launch[1, len(pts)](lat, lon, dn, meta, out)

max_err = 0.0
for k, (la, lo) in enumerate(pts):
    ref = sample_topo_radius_m(payload, la, lo)
    max_err = max(max_err, abs(out[k] - ref))
assert max_err < 1e-6, max_err

# --- Residual f(alpha) for a radial segment, identity rotation ------------
# Uniform terrain radius via a flat grid so terrain_r is constant.
TERR = 1737400.0 + 5000.0
dn_flat = np.full((LINES, SAMPLES), (TERR - BIAS_M) / SCALE_M, dtype=np.float64)
meta_flat = meta.copy(); meta_flat[11] = TERR

@cuda.jit
def _resid_launch(alphas, px0, py0, pz0, dx, dy, dz, dn, meta, impact_alt, out):
    k = cuda.grid(1)
    if k < alphas.shape[0]:
        out[k] = _terrain_residual_cuda(
            alphas[k], px0, py0, pz0, dx, dy, dz,
            1.0, 0.0, 0.0, 0.0, dn, meta, impact_alt,
        )

R0 = 1737400.0
r_prev = R0 + 8000.0
r_curr = R0 - 8000.0
px0, py0, pz0 = r_prev, 0.0, 0.0
dx, dy, dz = (r_curr - r_prev), 0.0, 0.0
alphas = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float64)
res = np.zeros_like(alphas)
_resid_launch[1, len(alphas)](alphas, px0, py0, pz0, dx, dy, dz, dn_flat, meta_flat, 0.0, res)

for k, a in enumerate(alphas):
    r_here = r_prev + a * (r_curr - r_prev)
    expected = r_here - TERR  # impact_alt = 0
    assert abs(res[k] - expected) < 1e-6, (a, res[k], expected)

# Residual crosses zero exactly at the terrain radius (sanity on monotonic sign).
assert res[0] > 0.0 and res[-1] < 0.0

print("OK", max_err)
"""
    result = _run_cudasim(script)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout, result.stdout + result.stderr
