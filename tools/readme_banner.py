"""Generate the Lunaris README banners (dark and light) as deterministic SVG.

Usage: python tools/readme_banner.py docs/assets [gravity_file]

The Moon disk shows real data: the radial gravity anomaly that a degree-25
field leaves out at 100 km altitude, i.e. degrees 26-200 of GRAIL GL1800F,
which is the residual ST-LRPS learns. It is drawn on the near side
(orthographic, centred on 0 N 0 E) as filled bands: warm where gravity is
stronger than SH25 predicts, cool where it is weaker. Needs the GL1800F file
(default data/gravity_models/jggrx_1800f_sha.tab.txt), pyshtools and
matplotlib; the coefficient convention (4-pi normalised, no Condon-Shortley
phase) was checked by the full field giving positive anomalies over the known
near-side mascons.

An inclined orbit is drawn in two parts (behind / in front of the Moon) with a
spacecraft marker, and the wordmark sits on the left. Colours are the Lunar
Graphite tokens (docs/UI_THEME.md).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

W, H = 1280, 320
CX, CY, R = 1010.0, 160.0, 104.0          # Moon centre and radius (px)
ORBIT_A = 178.0                           # orbit radius in the sky plane (px)
INC = math.radians(74.0)                  # tilt of the orbit plane towards the viewer
ROT = math.radians(-22.0)                 # rotation of the projected ellipse on the page
SC_U = math.radians(30.0)                 # spacecraft argument along the orbit
ALT_M = 100e3                             # altitude of the residual map (training-band floor)
DEG_LO, DEG_HI = 26, 200                  # residual = SH200 minus SH25
PIX = 1.5                                 # sampling step of the disk map (px)

THEMES = {
    "dark": dict(
        bg="#070A0F", bg2="#0D131C", fg="#F2F5FA", soft="#C4CEDB", muted="#95A3B5",
        accent="#6AA9FF", moon_lit="#C9D1DC", moon_mid="#8E9AAB", moon_dark="#151C27",
        warm=("#F2B27A", "#E8894A", "#D9622B"), cool=("#8FD0C8", "#4FAFA4", "#23877E"),
        field_op=0.62, orbit_back_op=0.35, star="#C4CEDB", n_stars=46,
    ),
    "light": dict(
        bg="#F5F7FB", bg2="#E9EEF5", fg="#0D131C", soft="#293747", muted="#4A5765",
        accent="#2E6AD6", moon_lit="#E7EBF1", moon_mid="#BCC5D1", moon_dark="#6F7C8D",
        warm=("#F0B98A", "#E08A50", "#C8642E"), cool=("#9ED3CC", "#5BB2A7", "#2B8C82"),
        field_op=0.70, orbit_back_op=0.40, star="#95A3B5", n_stars=0,
    ),
}


def orbit_point(u: float) -> tuple[float, float, float]:
    """Projected (x, y) and depth z (>0 towards the viewer) of the orbit at angle u."""
    x0, y0 = ORBIT_A * math.cos(u), ORBIT_A * math.sin(u)
    y1, z1 = y0 * math.cos(INC), y0 * math.sin(INC)
    xr = x0 * math.cos(ROT) - y1 * math.sin(ROT)
    yr = x0 * math.sin(ROT) + y1 * math.cos(ROT)
    return CX + xr, CY + yr, z1


def orbit_paths() -> tuple[str, str]:
    """Back (hidden behind the disk where it overlaps) and front polylines."""
    back, front = [], []
    n = 720
    for k in range(n + 1):
        u = 2 * math.pi * k / n
        x, y, z = orbit_point(u)
        inside = (x - CX) ** 2 + (y - CY) ** 2 < R ** 2
        (front if z >= 0 else back).append((x, y, inside and z < 0))
    def poly(points, skip_hidden=False):
        segs, cur = [], []
        for x, y, hidden in points:
            if skip_hidden and hidden:
                if cur:
                    segs.append(cur)
                    cur = []
                continue
            cur.append(f"{x:.1f},{y:.1f}")
        if cur:
            segs.append(cur)
        return " ".join("M" + " L".join(s) for s in segs if len(s) > 1)
    return poly(back, skip_hidden=True), poly(front)


def residual_anomaly(gravity_file: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(lat_deg, lon_deg, anomaly_mGal) of degrees DEG_LO..DEG_HI at ALT_M, positive = stronger pull."""
    import pyshtools as psh

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    from lunaris.loaders.io_gravity import load_gravity_model

    n, r0, gm, c, s = load_gravity_model(gravity_file, DEG_HI)
    cs = np.zeros((2, n + 1, n + 1))
    cs[0], cs[1] = c[: n + 1, : n + 1], s[: n + 1, : n + 1]
    cs[:, :DEG_LO, :] = 0.0
    grid = psh.SHGravCoeffs.from_array(cs, gm=gm, r0=r0, normalization="4pi", csphase=1).expand(
        a=r0 + ALT_M, lmax=2 * DEG_HI, normal_gravity=False)
    # pyshtools adds the central term; remove it. rad points outward, so a stronger
    # pull is a more negative rad: flip the sign to get the usual anomaly sign.
    central = -gm / (r0 + ALT_M) ** 2
    anomaly = -(grid.rad.to_array() - central) * 1e5
    return grid.rad.lats(), grid.rad.lons(), anomaly


def disk_bands(lats: np.ndarray, lons: np.ndarray, anomaly: np.ndarray) -> tuple[list[str], list[str]]:
    """SVG path data of the warm and cool bands on the near-side disk (3 levels each)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = np.arange(CX - R, CX + R + PIX, PIX)
    ys = np.arange(CY - R, CY + R + PIX, PIX)
    gx, gy = np.meshgrid(xs, ys)
    X, Y = (gx - CX) / R, -(gy - CY) / R
    rho2 = X ** 2 + Y ** 2
    Z = np.sqrt(np.clip(1.0 - rho2, 0.0, None))
    lat = np.degrees(np.arcsin(np.clip(Y, -1.0, 1.0)))
    lon = np.degrees(np.arctan2(X, Z)) % 360.0
    # nearest grid node (the DH grid is regular: lats descending from 90, lons from 0)
    i = np.clip(np.rint((lats[0] - lat) / (lats[0] - lats[1])).astype(int), 0, lats.size - 1)
    j = np.rint(lon / (lons[1] - lons[0])).astype(int) % lons.size
    field = anomaly[i, j]
    field[rho2 > 1.0] = 0.0
    scale = float(np.percentile(np.abs(anomaly), 99))
    # three bands per sign, light to dark; |anomaly| below scale/6 stays bare Moon
    edges = [scale / 6, scale / 3, 2 * scale / 3, float(np.abs(field).max()) + 1.0]
    fig, ax = plt.subplots()
    out = []
    for sign in (1.0, -1.0):
        cs = ax.contourf(gx, gy, sign * field, levels=edges)
        out.append([_svg_path(p) for p in cs.get_paths()])
    plt.close(fig)
    return out[0], out[1]


def _svg_path(path) -> str:
    """Matplotlib Path -> SVG path data, 0.1 px precision."""
    d, verts, codes = [], path.vertices, path.codes
    if codes is None:
        codes = [1] + [2] * (len(verts) - 1)
    for (x, y), code in zip(verts, codes, strict=True):
        if code == 1:
            d.append(f"M{x:.1f},{y:.1f}")
        elif code == 79:
            d.append("Z")
        else:
            d.append(f"L{x:.1f},{y:.1f}")
    return "".join(d)


def stars(n: int, seed: int = 7) -> list[tuple[float, float, float, float]]:
    """Deterministic sparse star field, kept clear of the text block."""
    import random
    rnd = random.Random(seed)
    out = []
    while len(out) < n:
        x, y = rnd.uniform(600, W - 12), rnd.uniform(12, H - 12)
        if (x - CX) ** 2 + (y - CY) ** 2 < (R + 14) ** 2:
            continue
        out.append((x, y, rnd.choice((0.7, 0.9, 1.1, 1.4)), rnd.uniform(0.25, 0.7)))
    return out


def build(theme: str, warm: list[str], cool: list[str]) -> str:
    t = THEMES[theme]
    back, front = orbit_paths()
    sx, sy, _ = orbit_point(SC_U)
    star_svg = "\n".join(
        f'    <circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{t["star"]}" fill-opacity="{o:.2f}"/>'
        for x, y, r, o in stars(t["n_stars"])
    )
    bands = "\n".join(
        f'    <path d="{d}" fill="{col}" fill-rule="evenodd"/>'
        for paths, cols in ((warm, t["warm"]), (cool, t["cool"]))
        for d, col in zip(paths, cols, strict=True) if d
    )
    sun = (-0.80, -0.60)  # unit light direction in the page plane (towards the upper left)
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-labelledby="t d">
  <title id="t">Lunaris</title>
  <desc id="d">Lunaris: lunar orbit propagation with a learned residual gravity field (ST-LRPS). The Moon's near side shows the radial gravity that a degree-25 field leaves out at 100 km altitude, degrees 26 to 200 of the GRAIL GL1800F model: warm where gravity is stronger, cool where it is weaker. A spacecraft is on an inclined orbit.</desc>
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0" stop-color="{t["bg"]}"/>
      <stop offset="1" stop-color="{t["bg2"]}"/>
    </linearGradient>
    <linearGradient id="moon" x1="{CX + sun[0] * R:.1f}" y1="{CY + sun[1] * R:.1f}" x2="{CX - sun[0] * R:.1f}" y2="{CY - sun[1] * R:.1f}" gradientUnits="userSpaceOnUse">
      <stop offset="0" stop-color="{t["moon_lit"]}"/>
      <stop offset="0.45" stop-color="{t["moon_lit"]}"/>
      <stop offset="0.68" stop-color="{t["moon_mid"]}"/>
      <stop offset="0.92" stop-color="{t["moon_dark"]}"/>
      <stop offset="1" stop-color="{t["moon_dark"]}"/>
    </linearGradient>
    <linearGradient id="shade" x1="{CX + sun[0] * R:.1f}" y1="{CY + sun[1] * R:.1f}" x2="{CX - sun[0] * R:.1f}" y2="{CY - sun[1] * R:.1f}" gradientUnits="userSpaceOnUse">
      <stop offset="0.55" stop-color="{t["moon_dark"]}" stop-opacity="0"/>
      <stop offset="1" stop-color="{t["moon_dark"]}" stop-opacity="0.85"/>
    </linearGradient>
    <clipPath id="disk"><circle cx="{CX}" cy="{CY}" r="{R}"/></clipPath>
  </defs>
  <rect width="{W}" height="{H}" fill="url(#bg)"/>
  <g>
{star_svg}
  </g>
  <path d="{back}" fill="none" stroke="{t["accent"]}" stroke-opacity="{t["orbit_back_op"]}" stroke-width="1.6" stroke-linecap="round"/>
  <circle cx="{CX}" cy="{CY}" r="{R}" fill="url(#moon)"/>
  <g clip-path="url(#disk)" fill-opacity="{t["field_op"]}">
{bands}
  </g>
  <circle cx="{CX}" cy="{CY}" r="{R}" fill="url(#shade)"/>
  <path d="{front}" fill="none" stroke="{t["accent"]}" stroke-width="2.2" stroke-linecap="round"/>
  <circle cx="{sx:.1f}" cy="{sy:.1f}" r="9" fill="{t["accent"]}" fill-opacity="0.18"/>
  <circle cx="{sx:.1f}" cy="{sy:.1f}" r="4.5" fill="{t["accent"]}"/>
  <g font-family="'Segoe UI', 'Helvetica Neue', Helvetica, Arial, sans-serif">
    <text x="72" y="142" font-size="76" font-weight="600" letter-spacing="1" fill="{t["fg"]}">Lunaris</text>
    <text x="76" y="188" font-size="24" fill="{t["soft"]}">Lunar orbit propagation with a learned</text>
    <text x="76" y="220" font-size="24" fill="{t["soft"]}">residual gravity field (ST-LRPS)</text>
    <text x="76" y="270" font-size="17" fill="{t["muted"]}">Moon: radial gravity that SH25 misses at 100 km (GRAIL GL1800F, degrees 26–200)</text>
  </g>
</svg>
'''


def main(out_dir: str, gravity_file: str | None = None) -> None:
    root = Path(__file__).resolve().parents[1]
    gravity_file = gravity_file or str(root / "data" / "gravity_models" / "jggrx_1800f_sha.tab.txt")
    warm, cool = disk_bands(*residual_anomaly(gravity_file))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for theme in THEMES:
        (out / f"lunaris-banner-{theme}.svg").write_text(build(theme, warm, cool), encoding="utf-8", newline="\n")
        print("wrote", out / f"lunaris-banner-{theme}.svg")


if __name__ == "__main__":
    main(*sys.argv[1:3])
