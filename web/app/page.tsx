'use client';
import { useState } from 'react';
import dynamic from 'next/dynamic';
import { ReliefControlPanel } from '@/components/ReliefControlPanel';
import { HeroTitle } from '@/components/HeroTitle';
import { OrbitalDynamicsBento } from '@/components/OrbitalDynamicsBento';

// Load the R3F scene client-only so `output: 'export'` does not attempt to
// prerender a WebGL canvas on the server during static export.
const Scene3D = dynamic(() => import('@/components/Scene3D'), { ssr: false });

export default function Home() {
  const [textureMode, setTextureMode] = useState<'aesthetic' | 'gravity'>('aesthetic');
  const [displacementScale, setDisplacementScale] = useState(0.03);

  return (
    <main style={{ position: 'relative', background: '#05050A', overflowX: 'hidden' }}>
      
      {/* Absolute UI Layer (Scrolls away with the page) */}
      <ReliefControlPanel 
        textureMode={textureMode} 
        setTextureMode={setTextureMode} 
        displacementScale={displacementScale} 
        setDisplacementScale={setDisplacementScale} 
      />

      {/* Layer 0: LUNARIS Title (Scrolls behind the Moon) */}
      <HeroTitle />

      {/* Layer 5: Fixed 3D Scene Container */}
      <div style={{ position: 'fixed', top: 0, left: 0, width: '100vw', height: '100vh', zIndex: 5, pointerEvents: 'auto' }}>
        <Scene3D textureMode={textureMode} displacementScale={displacementScale} />
      </div>
      
      {/* First Page Spacer (Allows scrolling down) */}
      <div style={{ height: '100vh', width: '100%' }}></div>
      
      {/* Layer 10: Scroll Content Area (Scrolls over the Moon) */}
      <div style={{ position: 'relative', width: '100%', zIndex: 10 }}>
        <section style={{ position: 'relative', padding: '25vh 0 15vh 0', pointerEvents: 'auto' }}>
          
          {/* Smooth Blur Background Layer with Premium Gradient Mask */}
          <div style={{ 
            position: 'absolute', 
            top: 0, left: 0, width: '100%', height: '100%', 
            background: 'linear-gradient(180deg, rgba(5,5,10,0.0) 0%, rgba(5,5,10,0.85) 15%, rgba(5,5,10,1) 40%)',
            backdropFilter: 'blur(24px)',
            WebkitBackdropFilter: 'blur(24px)',
            maskImage: 'linear-gradient(to bottom, transparent 0%, black 15%, black 100%)',
            WebkitMaskImage: 'linear-gradient(to bottom, transparent 0%, black 15%, black 100%)',
            zIndex: -1
          }}></div>

          <OrbitalDynamicsBento />

        </section>
      </div>
    </main>
  );
}
