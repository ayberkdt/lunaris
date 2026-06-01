'use client';
import { useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import { Sphere, useTexture } from '@react-three/drei';
import * as THREE from 'three';

interface MoonModelProps {
  textureMode?: 'aesthetic' | 'gravity';
  displacementScale?: number;
}

export default function MoonModel({ textureMode = 'aesthetic', displacementScale = 0.04 }: MoonModelProps) {
  const moonRef = useRef<THREE.Mesh>(null);
  
  // Pre-load all textures
  const [gravityMap, aestheticMap, dispMap] = useTexture([
    '/textures/gravity_moon_real.webp',
    '/textures/aesthetic_moon_real.webp',
    '/textures/moon_disp_real.webp'
  ]);
  
  gravityMap.colorSpace = THREE.SRGBColorSpace;
  aestheticMap.colorSpace = THREE.SRGBColorSpace;
  
  const currentMap = textureMode === 'aesthetic' ? aestheticMap : gravityMap;

  useFrame(({ clock }) => {
    if (moonRef.current) {
      moonRef.current.rotation.y = clock.getElapsedTime() * 0.05;
      moonRef.current.rotation.x = 0.1;
    }
  });

  return (
    <group>
      {/* Subtle Glow effect */}
      <Sphere args={[1.06, 64, 64]}>
        <meshBasicMaterial color={textureMode === 'aesthetic' ? "#A0C0D0" : "#00F0FF"} transparent opacity={0.04} side={THREE.BackSide} />
      </Sphere>
      
      {/* Main Moon Body */}
      <Sphere ref={moonRef} args={[1, 256, 256]}>
        <meshStandardMaterial 
          map={currentMap}
          displacementMap={dispMap}
          displacementScale={displacementScale}
          roughness={textureMode === 'aesthetic' ? 0.7 : 0.9}
          metalness={0.1}
        />
      </Sphere>
    </group>
  );
}
