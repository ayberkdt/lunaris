'use client';
import { Suspense } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Stars, useProgress } from '@react-three/drei';
import { EffectComposer, Bloom } from '@react-three/postprocessing';
import MoonModel from './MoonModel';
import OrbitPath from './OrbitPath';
import { useReducedMotion } from './useReducedMotion';

interface Scene3DProps {
  textureMode?: 'aesthetic' | 'gravity';
  displacementScale?: number;
}

export default function Scene3D({ textureMode = 'aesthetic', displacementScale = 0.04 }: Scene3DProps) {
  const { progress } = useProgress();
  const loaded = progress === 100;
  const prefersReducedMotion = useReducedMotion();

  return (
    <Canvas camera={{ position: [0, 0, 5], fov: 45 }} gl={{ alpha: true }}>
      {/* Background removed to ensure Canvas transparency for Parallax text */}
      
      <ambientLight intensity={textureMode === 'aesthetic' ? 0.5 : 0.2} />
      <directionalLight position={[5, 3, 5]} intensity={textureMode === 'aesthetic' ? 2 : 1.5} color="#ffffff" />
      <pointLight position={[-5, -5, -5]} intensity={0.5} color="#445588" />
      
      <Stars radius={100} depth={50} count={5000} factor={4} saturation={0} fade speed={1} />
      
      <Suspense fallback={null}>
        <MoonModel
          textureMode={textureMode}
          displacementScale={displacementScale}
          animate={!prefersReducedMotion}
        />
      </Suspense>
      
      {loaded && (
        <>
          {/* Premium Single Satellite (Cyan / Electric Blue) */}
          <OrbitPath
            color="#00E5FF"
            glowColor="#00AEEF"
            speed={3.5}
            dataKey="path1"
            animate={!prefersReducedMotion}
          />
        </>
      )}
      
      <EffectComposer>
        <Bloom luminanceThreshold={2} mipmapBlur intensity={2} />
      </EffectComposer>
      
      <OrbitControls 
        enableZoom={false} 
        enablePan={false} 
        autoRotate={!prefersReducedMotion}
        autoRotateSpeed={0.5}
      />
    </Canvas>
  );
}
