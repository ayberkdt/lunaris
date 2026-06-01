# Lunaris Orbit Propagation Framework - Showcase

Lunaris is a high-performance orbit analysis and Monte Carlo propagation framework utilizing **ST-LRPS** (a neural Sobolev-trained residual potential surrogate). This project is the interactive 3D web showcase for Lunaris, built with Next.js, React Three Fiber, and Three.js.

## Features

- **Premium 3D Lunar Visualization**: High-resolution texturing (Galileo false color & scientific gravity anomaly maps) with real-time displacement rendering.
- **Topographic Surface Relief**: Interactive UI control for visual exaggeration of surface displacement, using a glassmorphic sci-tech slider.
- **Cinematic Orbit Paths**: 
  - Low-inclination transfer orbit
  - Polar circular orbit
  - Highly elliptical inclined return orbit
  - Dynamic fading trails and premium geometric spacecraft markers.
- **Performant React Three Fiber Integration**: Optimized for smooth 60 FPS rendering in the browser.

## Getting Started

First, install the dependencies and run the development server:

```bash
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

## Desktop launcher embed (offline 3D Moon)

The Lunaris desktop launcher (`lunaris-launcher`) embeds a **headless** version of
this scene as a background visual. It uses the dedicated `/embed` route, which renders
only an edge-to-edge WebGL canvas (no header, no footer, no scroll, no marketing
content) and exposes a small JS bridge (`window.lunarisSetTextureMode`,
`window.lunarisSetOrbitVisible`, `window.lunarisSetPerformanceMode`) so the PySide6
host can drive it.

Build the offline static export consumed by the launcher:

```bash
cd web
npm install
npm run build      # output: 'export' -> writes ./out
```

This produces `out/embed/index.html` plus hashed assets under `out/_next/`. The Python
launcher serves `out/` from a local loopback HTTP server (no Node, no internet) and
loads `http://127.0.0.1:<port>/embed/`. If `out/` is missing or QtWebEngine/WebGL is
unavailable, the launcher falls back gracefully to a dark background — it never fails to
open. You can point the launcher at a custom build with the `LUNARIS_WEB_EMBED_DIR`
environment variable.

> **Note on the orbit:** the animated satellite path in `public/orbit-data.json` is a
> **demo orbit** — a Keplerian/Hohmann animation produced by `scripts/generate_orbit.py`
> for visual purposes. It is not the output of the Lunaris propagator.

## Project Structure

- `app/` - Next.js App Router pages and global CSS.
- `components/` - Modular React and 3D components for maintainability:
  - **3D Assets**: `Scene3D`, `MoonModel`, `OrbitPath`, `Spacecraft`
  - **UI & Layout**: `HeroTitle`, `ReliefControlPanel`, `OrbitalDynamicsBento`
- `public/` - Static assets, textures, and generated JSON data.
- `scripts/` - Python utilities for data generation and texture processing:
  - `generate_orbit.py`: Computes Keplerian constellation orbits and writes to `orbit-data.json`.
  - `tint_moon.py`, `tint_aesthetic.py`, `fix_textures.py`: Scripts for processing high-resolution lunar textures.

## Learn More

- Learn about the underlying ST-LRPS architecture via the in-app portal.
- [Next.js Documentation](https://nextjs.org/docs)
- [React Three Fiber](https://docs.pmnd.rs/react-three-fiber/getting-started/introduction)

---
*Created as part of the Lunaris ST-LRPS framework.*
