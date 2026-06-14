---
name: lunaris-web-3d
description: >-
  Work on the optional Lunaris Next.js/Three.js Moon preview under
  src/lunaris/ui/web the Lunaris way — offline static export, graceful WebGL
  fallback, and a clear demo-vs-solver boundary. Use when editing the web preview,
  the embed page, MoonModel/LauncherScene3D, camera/orbit controls, lighting,
  reduced-motion, GPU cleanup, or the static `out/` export, and on requests like
  "the 3D moon preview", "WebGL", "Three.js scene", "fix the embed", "offline
  build of the preview". The preview is optional and must never block the desktop
  app. NOT for the PySide6 OpenGL orbit widget (use lunaris-pyside6-ui) or 2D
  scientific charts (use scientific-figures).
---

# Lunaris Web / 3D Preview

The Next.js/Three.js preview (`src/lunaris/ui/web`) is an **optional** companion,
exported as static files (`out/`). It must degrade gracefully and never be
confused with solver output or block the desktop app.

## Invocation

Auto-trigger; inline. This is a separate Node subproject with its own
`src/lunaris/ui/web/AGENTS.md` and `package.json` — read those first. Hand 2D
charts to `scientific-figures` and the desktop OpenGL orbit to `lunaris-pyside6-ui`.

## Canonical sources

- `src/lunaris/ui/web/AGENTS.md`, `README.md`, `package.json`, `next.config.ts`.
- Components: `app/embed/page.tsx`, `app/st-lrps/page.tsx`,
  `components/MoonModel.tsx`, `components/LauncherScene3D.tsx`.

## Rules

1. **Offline static export.** The preview builds to `out/` and runs without a
   server or network. Don't add a runtime backend dependency or external CDN fetch
   required for it to render.
2. **Optional, non-blocking.** Its absence or a build failure must not break
   `lunaris-ui`/launcher. Keep it behind an optional path.
3. **WebGL fallback.** Detect WebGL-unavailable and show a static fallback (image/
   message), not a blank canvas or a crash.
4. **Frame budget + cleanup.** Target a steady frame budget; dispose geometries,
   materials, textures, and the renderer on unmount (no GPU leaks).
5. **Reduced motion.** Honor `prefers-reduced-motion` — stop/auto-rotate off.
6. **Demo vs solver.** Demonstration orbits in the scene are clearly labeled as
   illustrative; never present them as Lunaris-computed trajectories.
7. **Readability.** Depth cues, lighting, and orbit/Moon color separation so the
   orbit reads against the regolith; high-DPI correct; provide an accessible
   text/alternative.

## Procedure

1. Read the web `AGENTS.md`; make the minimal component/config change.
2. Preserve the static-export and optional-path guarantees.
3. Add/keep the WebGL fallback and resource cleanup.
4. Rebuild the static export if the task requires it.

## Verification

- `npm run build` (or the script in `package.json`) inside `src/lunaris/ui/web`
  produces `out/` without errors — if Node/npm is unavailable in the environment,
  say so and do not claim a successful build.
- Confirm the desktop app still launches with the preview absent.
- Manually verify reduced-motion and the WebGL-off fallback, or note no display.

## Stop conditions

- A change makes the desktop app depend on the web preview, or requires a network
  fetch to render → stop.

## Output

A change summary: components/config touched, how static-export + non-blocking +
WebGL-fallback + cleanup + reduced-motion are preserved, and the build/verify
result (or why it could not run here).

## Acceptance

Static offline export intact; preview optional and non-blocking; WebGL fallback
and GPU cleanup present; reduced-motion honored; demo orbits labeled; build runs
or its absence is explained.
