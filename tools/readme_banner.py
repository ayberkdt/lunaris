"""Generate the Lunaris README banners (dark and light) as deterministic SVG.

Usage: python tools/readme_banner.py docs/assets

Geometry: a Moon disk with a soft terminator and a faint graticule, an
inclined circular orbit drawn in two parts (behind / in front of the Moon), a
spacecraft marker on the front arc, and the wordmark on the left. Colours are
the Lunar Graphite tokens (docs/UI_THEME.md).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

W, H = 1280, 320
CX, CY, R = 1010.0, 160.0, 92.0          # Moon centre and radius (px)
ORBIT_A = 168.0                           # orbit radius in the sky plane (px)
INC = math.radians(74.0)                  # tilt of the orbit plane towards the viewer
ROT = math.radians(-22.0)                 # rotation of the projected ellipse on the page
SC_U = math.radians(30.0)                # spacecraft argument along the orbit

THEMES = {
    "dark": dict(
        bg="#070A0F", bg2="#0D131C", fg="#F2F5FA", soft="#C4CEDB", muted="#95A3B5",
        accent="#6AA9FF", moon_lit="#D3DAE4", moon_mid="#9AA6B6", moon_dark="#1A2330",
        mare="#3E4A5A", mare_op=0.26, orbit_back_op=0.35, star="#C4CEDB", n_stars=46,
    ),
    "light": dict(
        bg="#F5F7FB", bg2="#E9EEF5", fg="#0D131C", soft="#293747", muted="#4A5765",
        accent="#2E6AD6", moon_lit="#E4E9F0", moon_mid="#B9C3D0", moon_dark="#6F7C8D",
        mare="#5A6676", mare_op=0.20, orbit_back_op=0.40, star="#95A3B5", n_stars=0,
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


def graticule() -> str:
    """Faint meridians and parallels on the Moon (orthographic, tilted 12 deg)."""
    tilt = math.radians(12.0)
    lines = []
    for lon_deg in range(-60, 61, 30):
        lon = math.radians(lon_deg)
        pts = []
        for k in range(61):
            lat = math.radians(-90 + 3 * k)
            x = R * math.cos(lat) * math.sin(lon)
            y = -R * math.sin(lat)
            z = R * math.cos(lat) * math.cos(lon)
            yy, zz = y * math.cos(tilt) - z * math.sin(tilt), y * math.sin(tilt) + z * math.cos(tilt)
            if zz >= 0:
                pts.append(f"{CX + x:.1f},{CY + yy:.1f}")
        if len(pts) > 1:
            lines.append("M" + " L".join(pts))
    for lat_deg in range(-60, 61, 30):
        lat = math.radians(lat_deg)
        pts = []
        for k in range(73):
            lon = math.radians(-90 + 2.5 * k)
            x = R * math.cos(lat) * math.sin(lon)
            y = -R * math.sin(lat)
            z = R * math.cos(lat) * math.cos(lon)
            yy, zz = y * math.cos(tilt) - z * math.sin(tilt), y * math.sin(tilt) + z * math.cos(tilt)
            if zz >= 0:
                pts.append(f"{CX + x:.1f},{CY + yy:.1f}")
        if len(pts) > 1:
            lines.append("M" + " L".join(pts))
    return " ".join(lines)


def maria() -> str:
    """Soft dark patches placed like the near-side maria (Imbrium, Serenitatis,
    Tranquillitatis, Procellarum, Nubium), in disk-radius units."""
    patches = [
        (-0.30, -0.42, 0.30, 0.22, -20), (0.10, -0.36, 0.18, 0.15, 10),
        (0.22, -0.08, 0.22, 0.17, -15), (-0.55, -0.02, 0.22, 0.40, 15),
        (-0.18, 0.30, 0.20, 0.14, 0), (0.38, 0.12, 0.10, 0.08, 0),
    ]
    return "\n".join(
        f'    <ellipse cx="{CX + dx * R:.1f}" cy="{CY + dy * R:.1f}" rx="{rx * R:.1f}" ry="{ry * R:.1f}" '
        f'transform="rotate({a} {CX + dx * R:.1f} {CY + dy * R:.1f})"/>'
        for dx, dy, rx, ry, a in patches
    )


def craters() -> str:
    """A few small crater rims on the lit limb (disk-radius units)."""
    rims = [(-0.62, -0.55, 0.055), (-0.10, -0.72, 0.040), (-0.78, 0.28, 0.045), (0.05, 0.62, 0.050)]
    return "\n".join(
        f'      <circle cx="{CX + dx * R:.1f}" cy="{CY + dy * R:.1f}" r="{r * R:.1f}"/>' for dx, dy, r in rims
    )


def stars(n: int, seed: int = 7) -> list[tuple[float, float, float, float]]:
    """Deterministic sparse star field, kept clear of the text block."""
    import random
    rnd = random.Random(seed)
    out = []
    while len(out) < n:
        x, y = rnd.uniform(560, W - 12), rnd.uniform(12, H - 12)
        if (x - CX) ** 2 + (y - CY) ** 2 < (R + 14) ** 2:
            continue
        out.append((x, y, rnd.choice((0.7, 0.9, 1.1, 1.4)), rnd.uniform(0.25, 0.7)))
    return out


def build(theme: str) -> str:
    t = THEMES[theme]
    back, front = orbit_paths()
    sx, sy, _ = orbit_point(SC_U)
    star_svg = "\n".join(
        f'    <circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{t["star"]}" fill-opacity="{o:.2f}"/>'
        for x, y, r, o in stars(t["n_stars"])
    )
    sun = (-0.80, -0.60)  # unit light direction in the page plane (towards the upper left)
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-labelledby="t d">
  <title id="t">Lunaris</title>
  <desc id="d">Lunaris: lunar orbit propagation and ST-LRPS surrogate gravity. The picture shows a spacecraft on an inclined orbit around the Moon.</desc>
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0" stop-color="{t["bg"]}"/>
      <stop offset="1" stop-color="{t["bg2"]}"/>
    </linearGradient>
    <linearGradient id="moon" x1="{CX + sun[0] * R:.1f}" y1="{CY + sun[1] * R:.1f}" x2="{CX - sun[0] * R:.1f}" y2="{CY - sun[1] * R:.1f}" gradientUnits="userSpaceOnUse">
      <stop offset="0" stop-color="{t["moon_lit"]}"/>
      <stop offset="0.42" stop-color="{t["moon_lit"]}"/>
      <stop offset="0.62" stop-color="{t["moon_mid"]}"/>
      <stop offset="0.88" stop-color="{t["moon_dark"]}"/>
      <stop offset="1" stop-color="{t["moon_dark"]}"/>
    </linearGradient>
    <filter id="soft" x="-20%" y="-20%" width="140%" height="140%"><feGaussianBlur stdDeviation="5"/></filter>
    <clipPath id="disk"><circle cx="{CX}" cy="{CY}" r="{R}"/></clipPath>
  </defs>
  <rect width="{W}" height="{H}" fill="url(#bg)"/>
  <g>
{star_svg}
  </g>
  <path d="{back}" fill="none" stroke="{t["accent"]}" stroke-opacity="{t["orbit_back_op"]}" stroke-width="1.6" stroke-linecap="round"/>
  <circle cx="{CX}" cy="{CY}" r="{R}" fill="url(#moon)"/>
  <g clip-path="url(#disk)">
    <g fill="{t["mare"]}" fill-opacity="{t["mare_op"]}" filter="url(#soft)">
{maria()}
    </g>
    <g fill="none" stroke="{t["mare"]}" stroke-opacity="{t["mare_op"]}" stroke-width="1.2">
{craters()}
    </g>
  </g>
  <path d="{front}" fill="none" stroke="{t["accent"]}" stroke-width="2.2" stroke-linecap="round"/>
  <circle cx="{sx:.1f}" cy="{sy:.1f}" r="9" fill="{t["accent"]}" fill-opacity="0.18"/>
  <circle cx="{sx:.1f}" cy="{sy:.1f}" r="4.5" fill="{t["accent"]}"/>
  <g font-family="'Segoe UI', 'Helvetica Neue', Helvetica, Arial, sans-serif">
    <text x="72" y="150" font-size="76" font-weight="600" letter-spacing="1" fill="{t["fg"]}">Lunaris</text>
    <text x="76" y="196" font-size="24" fill="{t["soft"]}">Lunar orbit propagation and gravity modeling</text>
    <text x="76" y="230" font-size="24" fill="{t["muted"]}">with the ST-LRPS neural surrogate</text>
  </g>
</svg>
'''


def main(out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for theme in THEMES:
        (out / f"lunaris-banner-{theme}.svg").write_text(build(theme), encoding="utf-8", newline="\n")
        print("wrote", out / f"lunaris-banner-{theme}.svg")


if __name__ == "__main__":
    main(sys.argv[1])
