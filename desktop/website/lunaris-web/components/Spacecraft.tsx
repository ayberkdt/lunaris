import React, { forwardRef } from 'react';
import { Box, Cylinder, Sphere, Cone } from '@react-three/drei';
import * as THREE from 'three';

export interface SpacecraftRefs {
  thrusterRef: React.RefObject<THREE.Mesh | null>;
  thrusterCoreRef: React.RefObject<THREE.Mesh | null>;
  lightRef: React.RefObject<THREE.PointLight | null>;
}

export const Spacecraft = forwardRef<THREE.Group, SpacecraftRefs>(({ thrusterRef, thrusterCoreRef, lightRef }, ref) => {
  return (
    <group ref={ref}>
      <group scale={3.4}>
        {/* Main Bus Body */}
        <Box args={[0.025, 0.025, 0.06]}>
          <meshStandardMaterial color="#eeeeee" metalness={0.6} roughness={0.2} />
        </Box>
        
        {/* Dark Radiator Panels (Sides of the bus) */}
        <Cylinder args={[0.013, 0.013, 0.04, 6]} rotation={[Math.PI / 2, 0, Math.PI / 6]}>
          <meshStandardMaterial color="#111111" metalness={0.5} roughness={0.8} />
        </Cylinder>

        {/* Forward Avionics Deck (Now +Z is forward) */}
        <Cylinder args={[0.008, 0.008, 0.015, 16]} position={[0, 0, 0.03]} rotation={[Math.PI / 2, 0, 0]}>
          <meshStandardMaterial color="#333333" metalness={0.8} roughness={0.2} />
        </Cylinder>
        
        {/* Forward Star Tracker / Optical Lens (Now +Z is forward) */}
        <Cylinder args={[0.004, 0.004, 0.01, 16]} position={[0, 0, 0.04]}>
          <meshStandardMaterial color="#1a1a1a" metalness={0.9} roughness={0.1} />
        </Cylinder>
        
        {/* Solar Panel Array Hinge (Y-Axis) */}
        <Cylinder args={[0.002, 0.002, 0.08, 8]} position={[0, 0, 0]} rotation={[0, 0, 0]}>
          <meshStandardMaterial color="#555555" metalness={0.8} roughness={0.4} />
        </Cylinder>

        {/* Solar Panel Top (Facing moon surface normal, parallel to surface) */}
        <Box args={[0.025, 0.07, 0.001]} position={[0, 0.05, 0]} rotation={[0, Math.PI / 2, 0]}>
          <meshStandardMaterial color="#0a192f" emissive="#001122" metalness={0.9} roughness={0.1} />
        </Box>
        
        {/* Solar Panel Bottom (Facing moon surface normal, parallel to surface) */}
        <Box args={[0.025, 0.07, 0.001]} position={[0, -0.05, 0]} rotation={[0, Math.PI / 2, 0]}>
          <meshStandardMaterial color="#0a192f" emissive="#001122" metalness={0.9} roughness={0.1} />
        </Box>

        {/* High-Gain Parabolic Antenna (Mounted on +X side to avoid solar panels) */}
        <group position={[0.015, 0, 0.01]} rotation={[0, -Math.PI / 6, -Math.PI / 2]}>
          {/* Dish Stem */}
          <Cylinder args={[0.001, 0.001, 0.01, 8]} position={[0, 0.005, 0]}>
             <meshStandardMaterial color="#888888" metalness={0.7} />
          </Cylinder>
          {/* Dish Hemisphere */}
          <Sphere args={[0.015, 16, 16, 0, Math.PI * 2, 0, Math.PI / 2]} position={[0, 0.01, 0]} rotation={[Math.PI, 0, 0]}>
            <meshStandardMaterial color="#ffffff" metalness={0.5} roughness={0.3} side={THREE.DoubleSide} />
          </Sphere>
          {/* Feed Horn */}
          <Cylinder args={[0.0005, 0.0005, 0.01, 8]} position={[0, 0.015, 0]}>
             <meshStandardMaterial color="#555555" metalness={0.9} />
          </Cylinder>
        </group>

        {/* Main Engine Bell (At the back: -Z, properly attached to body) */}
        <Cone args={[0.008, 0.015, 16]} position={[0, 0, -0.033]} rotation={[Math.PI / 2, 0, 0]}>
          <meshStandardMaterial color="#222222" metalness={0.9} roughness={0.5} />
        </Cone>

        {/* Thruster Plume outer — wide orange/amber envelope */}
        <Cylinder ref={thrusterRef} args={[0.022, 0.001, 0.14, 16]} position={[0, 0, -0.09]} rotation={[Math.PI / 2, 0, 0]}>
          <meshStandardMaterial color="#FF8800" emissive="#FF5500" emissiveIntensity={1.5} toneMapped={false} transparent opacity={0.6} />
        </Cylinder>

        {/* Thruster Core — inner white-hot streak */}
        <Cylinder ref={thrusterCoreRef} args={[0.009, 0.0005, 0.14, 8]} position={[0, 0, -0.09]} rotation={[Math.PI / 2, 0, 0]}>
          <meshStandardMaterial color="#FFFFFF" emissive="#FFEEAA" emissiveIntensity={3.0} toneMapped={false} transparent opacity={0.85} />
        </Cylinder>

        {/* Thruster Light — flickering warm glow */}
        <pointLight ref={lightRef} position={[0, 0, -0.05]} color="#FFAA00" intensity={0} distance={2.5} />
      </group>
    </group>
  );
});

Spacecraft.displayName = 'Spacecraft';
