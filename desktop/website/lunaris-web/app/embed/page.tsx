'use client';
import dynamic from 'next/dynamic';

/**
 * Headless embed route for the Lunaris desktop launcher.
 *
 * This page is intentionally chrome-free: no Header (hidden via Header.tsx),
 * no footer, no scroll, no marketing content. It renders a single edge-to-edge
 * WebGL canvas that the PySide6 launcher overlays with its own glassmorphic
 * navigation cards. The web side is purely a visual engine.
 *
 * The R3F scene is loaded with `ssr: false` so the static export
 * (`output: 'export'`) does not try to prerender WebGL on the server.
 */
const LauncherScene3D = dynamic(() => import('@/components/LauncherScene3D'), {
  ssr: false,
});

export default function EmbedPage() {
  return (
    <div
      style={{
        position: 'fixed',
        inset: 0,
        width: '100vw',
        height: '100vh',
        background: '#000000',
        overflow: 'hidden',
      }}
    >
      <LauncherScene3D />
    </div>
  );
}
