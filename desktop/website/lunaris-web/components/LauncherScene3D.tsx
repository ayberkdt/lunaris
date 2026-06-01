'use client';
import { Suspense, useEffect, useState } from 'react';
import { Canvas, useThree } from '@react-three/fiber';
import { OrbitControls, Stars, useProgress } from '@react-three/drei';
import { EffectComposer, Bloom } from '@react-three/postprocessing';
import MoonModel from './MoonModel';
import OrbitPath from './OrbitPath';

/**
 * LauncherScene3D — the headless visual engine for the desktop launcher.
 *
 * A trimmed, calmer variant of the marketing `Scene3D`:
 *  - slow auto-rotation, limited orbit controls (no zoom / no pan)
 *  - a single clean DEMO orbit (the satellite path in `public/orbit-data.json`
 *    is a Keplerian/Hohmann animation from `scripts/generate_orbit.py`, NOT a
 *    real solver trajectory)
 *  - no overlay HTML controls — the PySide6 launcher owns all navigation UI.
 *
 * The user can rotate and zoom (within tight limits); panning is disabled so
 * the composition can't be dragged off-center.
 *
 * It exposes a tiny global JS bridge so the Qt host (QWebEngineView) can drive
 * the scene without any web-side buttons:
 *   window.lunarisSetTextureMode("visual" | "gravity")
 *   window.lunarisSetOrbitVisible(true | false)
 *   window.lunarisSetPerformanceMode("quality" | "balanced" | "low")
 *   window.lunarisSetRelief(0..1)   // surface displacement exaggeration
 *   window.lunarisReady === true    // set after first successful render
 */

type TextureMode = 'aesthetic' | 'gravity';
type PerfMode = 'quality' | 'balanced' | 'low';

// Surface relief (displacement) range. 0 = flat, RELIEF_MAX = pronounced but
// still restrained. The slider sends 0..1 which maps linearly onto this.
const RELIEF_MAX = 0.06;

declare global {
  interface Window {
    lunarisSetTextureMode?: (mode: 'visual' | 'gravity') => void;
    lunarisSetOrbitVisible?: (visible: boolean) => void;
    lunarisSetPerformanceMode?: (mode: PerfMode) => void;
    lunarisSetRelief?: (value: number) => void;
    lunarisReady?: boolean;
  }
}

/** Sets window.lunarisReady = true once the GL context is live. */
function ReadyFlag() {
  const gl = useThree((s) => s.gl);
  useEffect(() => {
    if (typeof window !== 'undefined' && gl) {
      window.lunarisReady = true;
    }
  }, [gl]);
  return null;
}

/**
 * Pushes the rendered scene to the right of the canvas so the Moon clears the
 * launcher's left-anchored navigation panel. Uses the camera view offset, which
 * shifts the projection deterministically while OrbitControls keeps orbiting the
 * Moon at the origin (so it stays anchored on the right while it rotates).
 *
 * `fraction` is the share of the canvas width to shift by; a negative x offset
 * moves rendered content to the right.
 */
function ViewOffset({ fraction = 0.24 }: { fraction?: number }) {
  const camera = useThree((s) => s.camera);
  const size = useThree((s) => s.size);
  useEffect(() => {
    const w = size.width;
    const h = size.height;
    if (!w || !h) return;
    // On narrow/portrait canvases, recenter so the Moon is never cropped.
    const f = w < 720 ? 0 : fraction;
    // setViewOffset(fullW, fullH, x, y, w, h): negative x shifts content right.
    camera.setViewOffset(w, h, -w * f, 0, w, h);
    camera.updateProjectionMatrix();
    return () => {
      camera.clearViewOffset();
    };
  }, [camera, size, fraction]);
  return null;
}

export default function LauncherScene3D() {
  const [textureMode, setTextureMode] = useState<TextureMode>('aesthetic');
  const [orbitVisible, setOrbitVisible] = useState(true);
  const [perfMode, setPerfMode] = useState<PerfMode>('quality');
  const [relief, setRelief] = useState(0.5); // 0..1 (maps onto RELIEF_MAX)
  const { progress } = useProgress();
  const loaded = progress === 100;

  // Register the global JS bridge for the Qt host. Kept on window so the
  // launcher can call it via page.runJavaScript(...).
  useEffect(() => {
    if (typeof window === 'undefined') return;

    window.lunarisSetTextureMode = (mode) =>
      setTextureMode(mode === 'gravity' ? 'gravity' : 'aesthetic');
    window.lunarisSetOrbitVisible = (visible) => setOrbitVisible(!!visible);
    window.lunarisSetPerformanceMode = (mode) => {
      if (mode === 'low' || mode === 'balanced' || mode === 'quality') {
        setPerfMode(mode);
      }
    };
    window.lunarisSetRelief = (value) => {
      const v = Number(value);
      if (Number.isFinite(v)) setRelief(Math.max(0, Math.min(1, v)));
    };

    return () => {
      delete window.lunarisSetTextureMode;
      delete window.lunarisSetOrbitVisible;
      delete window.lunarisSetPerformanceMode;
      delete window.lunarisSetRelief;
    };
  }, []);

  const displacementScale = relief * RELIEF_MAX;

  // Performance tuning: lower DPR and disable bloom on weaker GPUs.
  const dpr: [number, number] =
    perfMode === 'low' ? [0.7, 1] : perfMode === 'balanced' ? [1, 1.5] : [1, 2];
  const enableBloom = perfMode !== 'low';

  return (
    <Canvas
      camera={{ position: [0, 0, 4.4], fov: 45 }}
      gl={{ alpha: false, antialias: perfMode !== 'low' }}
      dpr={dpr}
      style={{ background: '#000000' }}
    >
      <ReadyFlag />
      {/* Shift the Moon to the right so it clears the left-side nav panel. */}
      <ViewOffset fraction={0.26} />

      {/* Subtle, scientific starfield — depth without a neon/galaxy look. */}
      <Stars
        radius={120}
        depth={60}
        count={enableBloom ? 2600 : 1400}
        factor={3}
        saturation={0}
        fade
        speed={0.4}
      />

      {/* Scientific, restrained lighting — no neon wash. */}
      <ambientLight intensity={textureMode === 'aesthetic' ? 0.45 : 0.2} />
      <directionalLight
        position={[5, 3, 5]}
        intensity={textureMode === 'aesthetic' ? 1.8 : 1.4}
        color="#ffffff"
      />
      <pointLight position={[-5, -5, -5]} intensity={0.4} color="#334466" />

      <Suspense fallback={null}>
        <MoonModel textureMode={textureMode} displacementScale={displacementScale} />
      </Suspense>

      {loaded && orbitVisible && (
        <OrbitPath color="#00E5FF" glowColor="#00AEEF" speed={2.2} dataKey="path1" />
      )}

      {enableBloom && (
        <EffectComposer>
          <Bloom luminanceThreshold={2} mipmapBlur intensity={1.6} />
        </EffectComposer>
      )}

      {/* Calm, premium motion. Rotate + limited zoom; pan is locked so the
          composition can never be dragged off-center. The zoom range is tight
          so the Moon stays nicely framed. */}
      <OrbitControls
        enableZoom
        enablePan={false}
        minDistance={3.2}
        maxDistance={6.2}
        zoomSpeed={0.6}
        autoRotate
        autoRotateSpeed={0.35}
        rotateSpeed={0.4}
        minPolarAngle={Math.PI * 0.25}
        maxPolarAngle={Math.PI * 0.75}
      />
    </Canvas>
  );
}
