'use client';
import { useRef, useMemo, useState } from 'react';
import { useFrame } from '@react-three/fiber';
import { Line, Box, Cylinder, Sphere, Cone } from '@react-three/drei';
import * as THREE from 'three';
import orbitData from '../public/orbit-data.json';
import { Spacecraft } from './Spacecraft';

interface OrbitPathProps {
  color?: string;
  glowColor?: string;
  speed?: number;
  dataKey?: string;
}

export default function OrbitPath({ color = "#00E5FF", glowColor = "#00AEEF", speed = 2, dataKey = "path1" }: OrbitPathProps) {
  const satelliteRef = useRef<THREE.Group>(null);
  const thrusterRef = useRef<THREE.Mesh>(null);
  const thrusterCoreRef = useRef<THREE.Mesh>(null);
  const lightRef = useRef<THREE.PointLight>(null);
  const dummy = useMemo(() => new THREE.Object3D(), []);
  const burnTime = useRef(0);
  
  const [futurePoints, setFuturePoints] = useState<THREE.Vector3[]>([]);
  const [trailPoints, setTrailPoints] = useState<THREE.Vector3[]>([]);

  const pathData = (orbitData as any)[dataKey];
  const pathRaw = pathData?.path as number[][];
  const futurePathsRaw = pathData?.future_paths as any;

  const points = useMemo(() => {
    if (!pathRaw) return [];
    return pathRaw.map(p => new THREE.Vector3(p[0], p[1], p[2]));
  }, [pathRaw]);

  const trailLength = 150;
  const trailSegments = 15; // More segments for smoother fade

  useFrame(({ clock }) => {
    if (satelliteRef.current && points.length > 1) {
      const timeOffset = 150; // Start in the middle of a coast phase to avoid confusing initial flips
      const t = ((clock.getElapsedTime() * speed * 2.5) + timeOffset) % points.length;
      const index = Math.floor(t);
      const nextIndex = (index + 1) % points.length;
      
      const p1 = points[index];
      const p2 = points[nextIndex];
      
      // Interpolate position
      const lerpFactor = t - index;
      satelliteRef.current.position.lerpVectors(p1, p2, lerpFactor);
      
      // ATTITUDE CONTROL — Direct quaternion from velocity vector
      // Velocity direction = from current point to next point
      const velDir = new THREE.Vector3().subVectors(p2, p1).normalize();

      // setFromUnitVectors(+Z, velDir) → satellite's +Z (nose/avionics) points along velocity
      // Engine is at -Z → fires BACKWARD during prograde = acceleration ✓
      const progradQ = new THREE.Quaternion().setFromUnitVectors(
        new THREE.Vector3(0, 0, 1), // satellite's nose axis
        velDir                       // target = velocity direction
      );

      // Retrograde attitude window — start ~8 seconds BEFORE burn for a graceful slow rotation
      // Burn 1: 870-900, prep from 820. Burn 2: 1170+, prep from 1120.
      const isRetrograde = (index >= 820 && index <= 930) || (index >= 1120 || index <= 30);
      if (isRetrograde) {
        const flip = new THREE.Quaternion().setFromEuler(new THREE.Euler(0, Math.PI, 0));
        progradQ.multiply(flip);
      }

      // Slerp factor 0.015 → smooth ~8-10 second attitude slew (realistic spacecraft turn rate)
      satelliteRef.current.quaternion.slerp(progradQ, 0.015);

      // Engine Burn Visuals (Prograde and Retrograde windows)
      const isBurning = 
        (index >= 270 && index <= 300) || // Prograde 1
        (index >= 570 && index <= 600) || // Prograde 2
        (index >= 870 && index <= 900) || // Retrograde 1
        (index >= 1170 || index < 5);     // Retrograde 2
      
      if (thrusterRef.current && thrusterCoreRef.current && lightRef.current) {
        if (isBurning) {
          burnTime.current += 0.05;
          const bt = burnTime.current;

          // Combustion instability: layered turbulent flicker
          const flicker1 = 0.85 + 0.15 * Math.sin(bt * 23.7);
          const flicker2 = 0.90 + 0.10 * Math.sin(bt * 41.3 + 1.2);
          const flicker3 = 0.95 + 0.05 * Math.sin(bt * 17.1 + 2.4);
          const combinedFlicker = flicker1 * flicker2 * flicker3;

          // Plume length oscillation (combustion chamber pressure variation)
          const plumeScale = combinedFlicker * (0.9 + 0.1 * Math.sin(bt * 8.3));
          const plumeWidth = 1.0 + 0.08 * Math.sin(bt * 31.0);

          thrusterRef.current.scale.set(plumeWidth, plumeWidth, plumeScale);
          thrusterCoreRef.current.scale.set(plumeWidth * 0.5, plumeWidth * 0.5, plumeScale * 1.1);

          // Dynamic material color shift (cool orange → white-hot core)
          const mat = thrusterRef.current.material as THREE.MeshStandardMaterial;
          mat.emissiveIntensity = 1.2 + 0.8 * combinedFlicker;
          mat.opacity = 0.55 + 0.15 * combinedFlicker;

          const coreMat = thrusterCoreRef.current.material as THREE.MeshStandardMaterial;
          coreMat.emissiveIntensity = 2.5 + 1.5 * combinedFlicker;
          coreMat.opacity = 0.7 + 0.2 * flicker1;

          // Flickering point light
          lightRef.current.intensity = 0.8 + 0.6 * combinedFlicker;
          lightRef.current.color.setHSL(0.08 + 0.02 * flicker2, 1.0, 0.6);
        } else {
          burnTime.current = 0;
          const targetScale = 0.001;
          thrusterRef.current.scale.lerp(new THREE.Vector3(targetScale, targetScale, targetScale), 0.2);
          thrusterCoreRef.current.scale.lerp(new THREE.Vector3(targetScale, targetScale, targetScale), 0.2);
          lightRef.current.intensity = THREE.MathUtils.lerp(lightRef.current.intensity, 0, 0.2);
        }
      }

      // Dynamic trail extraction (Seamless wraparound since path is physically continuous)
      const newTrail = [];
      for (let i = 0; i < trailLength; i++) {
        let pIdx = index - trailLength + i;
        if (pIdx < 0) pIdx += points.length;
        newTrail.push(points[pIdx]);
      }
      setTrailPoints(newTrail);

      // Morphing future trajectory (Full loop to see expansion/contraction)
      if (futurePathsRaw && futurePathsRaw[index]) {
        const currentFutureRaw = futurePathsRaw[index] as number[][];
        // Use all 150 points to show the complete predictive orbit shape
        const fPoints = currentFutureRaw.map(p => new THREE.Vector3(p[0], p[1], p[2]));
        setFuturePoints(fPoints);
      }
    }
  });

  if (points.length === 0) return null;

  return (
    <group>
      {/* Dynamic Future Path (Fading Predictive Ellipse) */}
      {futurePoints.length > 0 && Array.from({ length: 15 }).map((_, i) => {
        const segLen = futurePoints.length / 15;
        const start = Math.floor(i * segLen);
        const end = Math.floor((i + 1) * segLen) + 1;
        const segPoints = futurePoints.slice(start, Math.min(end, futurePoints.length));
        
        // Fades out as it goes further into the future
        const opacityRatio = 1.0 - (i / 14);
        const opacity = Math.max(0.02, Math.pow(opacityRatio, 1.5) * 0.4);

        if (segPoints.length < 2) return null;

        return (
          <Line key={`future-seg-${i}`} points={segPoints} color={glowColor} lineWidth={1.0} opacity={opacity} transparent depthWrite={false} />
        );
      })}
      
      {/* Cinematic Fading Trail */}
      {trailPoints.length > 0 && Array.from({ length: trailSegments }).map((_, i) => {
        const pointsPerSeg = trailPoints.length / trailSegments;
        const start = Math.floor(i * pointsPerSeg);
        const end = Math.floor((i + 1) * pointsPerSeg) + 1;
        const segPoints = trailPoints.slice(start, Math.min(end, trailPoints.length));
        
        // Non-linear opacity curve for a very premium comet-tail fade
        const opacityRatio = i / (trailSegments - 1);
        const opacity = Math.pow(opacityRatio, 1.5) * 0.9;
        const innerOpacity = Math.pow(opacityRatio, 2.0) * 1.0;

        if (segPoints.length < 2) return null;

        return (
          <group key={`trail-seg-${i}`}>
            <Line points={segPoints} color={color} lineWidth={2.5} opacity={opacity} transparent depthWrite={false} />
            <Line points={segPoints} color="#ffffff" lineWidth={1.0} opacity={innerOpacity} transparent depthWrite={false} />
          </group>
        );
      })}

      <Spacecraft
        ref={satelliteRef}
        thrusterRef={thrusterRef}
        thrusterCoreRef={thrusterCoreRef}
        lightRef={lightRef}
      />{/* end satelliteRef group */}
    </group>
  );
}
