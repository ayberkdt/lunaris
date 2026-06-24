'use client';

import { Suspense, useMemo, useRef, useState, type RefObject } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { Cone, Cylinder, Line, OrbitControls, Sphere, Stars } from '@react-three/drei';
import { Bloom, EffectComposer } from '@react-three/postprocessing';
import * as THREE from 'three';
import MoonModel from './MoonModel';
import { Spacecraft } from './Spacecraft';
import { useReducedMotion } from './useReducedMotion';
import orbitData from '../public/orbit-data.json';
import styles from './CinematicOrbitScene.module.css';

type CameraMode = 'director' | 'chase' | 'free';
type OrbitDataset = {
  path1?: {
    path?: number[][];
  };
};

type HudState = {
  phase: number;
  index: number;
  missionMin: number;
  altitudeKm: number;
  velocityKms: number;
  burnLabel: string;
};

const LUNAR_RADIUS_KM = 1737.4;
const ORBIT_PERIOD_MIN = 118;
const DEFAULT_HUD: HudState = {
  phase: 0,
  index: 0,
  missionMin: 0,
  altitudeKm: 0,
  velocityKms: 1.62,
  burnLabel: 'Coast arc',
};

const BURNS = [
  { index: 285, label: 'LOI trim 1', kind: 'prograde' },
  { index: 585, label: 'Apoapsis trim', kind: 'prograde' },
  { index: 885, label: 'Circularize', kind: 'retrograde' },
  { index: 1185, label: 'Plane cleanup', kind: 'retrograde' },
] as const;

function hasWebGL() {
  if (typeof document === 'undefined') return false;
  const canvas = document.createElement('canvas');
  return Boolean(canvas.getContext('webgl2') || canvas.getContext('webgl'));
}

function selectMimeType() {
  if (typeof MediaRecorder === 'undefined') return '';
  const candidates = [
    'video/webm;codecs=vp9',
    'video/webm;codecs=vp8',
    'video/webm',
  ];
  return candidates.find((type) => MediaRecorder.isTypeSupported(type)) ?? '';
}

function toVectors(rawPath: number[][] | undefined) {
  return (rawPath ?? [])
    .filter((sample) => sample.length >= 3)
    .map(([x, y, z]) => new THREE.Vector3(x, y, z));
}

function samplePoint(points: THREE.Vector3[], phase: number) {
  const scaled = phase * points.length;
  const index = Math.floor(scaled) % points.length;
  const nextIndex = (index + 1) % points.length;
  const lerp = scaled - Math.floor(scaled);
  const point = new THREE.Vector3().lerpVectors(points[index], points[nextIndex], lerp);
  const velocity = new THREE.Vector3().subVectors(points[nextIndex], points[index]).normalize();
  return { point, velocity, index };
}

function activeBurn(index: number) {
  return BURNS.find((burn) => {
    const wrappedDistance = Math.abs(index - burn.index);
    return Math.min(wrappedDistance, 1200 - wrappedDistance) <= 16;
  });
}

function OrbitMarker({ point, label }: { point: THREE.Vector3; label: string }) {
  return (
    <group position={point}>
      <Sphere args={[0.018, 16, 16]}>
        <meshStandardMaterial color="#f6fbff" emissive="#7dd8ff" emissiveIntensity={1.4} />
      </Sphere>
      <pointLight color="#7dd8ff" intensity={0.22} distance={0.5} />
      <mesh rotation={[Math.PI / 2, 0, 0]}>
        <torusGeometry args={[0.055, 0.0025, 8, 48]} />
        <meshBasicMaterial color="#7dd8ff" transparent opacity={0.55} />
      </mesh>
      <mesh name={label} visible={false} />
    </group>
  );
}

function ProfessionalSpacecraft({
  craftRef,
  plumeRef,
  coreRef,
  lightRef,
}: {
  craftRef: RefObject<THREE.Group | null>;
  plumeRef: RefObject<THREE.Mesh | null>;
  coreRef: RefObject<THREE.Mesh | null>;
  lightRef: RefObject<THREE.PointLight | null>;
}) {
  return (
    <group ref={craftRef}>
      <Spacecraft thrusterRef={plumeRef} thrusterCoreRef={coreRef} lightRef={lightRef} />
      <group scale={0.24}>
        <Cylinder args={[0.004, 0.004, 0.34, 10]} position={[0.22, 0, 0]} rotation={[0, 0, Math.PI / 2]}>
          <meshStandardMaterial color="#8aa0b5" metalness={0.7} roughness={0.32} />
        </Cylinder>
        <Cone args={[0.03, 0.08, 24]} position={[0.4, 0, 0]} rotation={[0, 0, -Math.PI / 2]}>
          <meshStandardMaterial color="#d9e8f2" metalness={0.42} roughness={0.28} />
        </Cone>
      </group>
    </group>
  );
}

function CinematicOrbitRig({
  points,
  playing,
  speed,
  cameraMode,
  onHud,
}: {
  points: THREE.Vector3[];
  playing: boolean;
  speed: number;
  cameraMode: CameraMode;
  onHud: (state: HudState) => void;
}) {
  const craftRef = useRef<THREE.Group>(null);
  const plumeRef = useRef<THREE.Mesh>(null);
  const coreRef = useRef<THREE.Mesh>(null);
  const lightRef = useRef<THREE.PointLight>(null);
  const lastHudUpdate = useRef(0);
  const progressRef = useRef(0.05);
  const { camera } = useThree();

  const ghostSegments = useMemo(() => {
    const segmentCount = 20;
    return Array.from({ length: segmentCount }, (_, i) => {
      const start = Math.floor((i / segmentCount) * points.length);
      const end = Math.floor(((i + 1) / segmentCount) * points.length) + 1;
      return points.slice(start, Math.min(end, points.length));
    }).filter((segment) => segment.length > 1);
  }, [points]);

  const burnMarkers = useMemo(
    () => BURNS.map((burn) => ({ ...burn, point: points[Math.min(burn.index, points.length - 1)] })),
    [points],
  );

  useFrame(({ clock }, delta) => {
    if (points.length < 2) return;

    if (playing) {
      progressRef.current = (progressRef.current + delta * speed * 0.025) % 1;
    }

    const phase = progressRef.current;
    const { point, velocity, index } = samplePoint(points, phase);
    const burn = activeBurn(index);

    if (craftRef.current) {
      craftRef.current.position.copy(point);
      const attitude = new THREE.Quaternion().setFromUnitVectors(
        new THREE.Vector3(0, 0, 1),
        velocity,
      );
      if (burn?.kind === 'retrograde') {
        attitude.multiply(new THREE.Quaternion().setFromEuler(new THREE.Euler(0, Math.PI, 0)));
      }
      craftRef.current.quaternion.slerp(attitude, 0.08);
    }

    const burnPulse = burn && playing ? 0.7 + 0.3 * Math.sin(clock.elapsedTime * 32) : 0;
    if (plumeRef.current && coreRef.current && lightRef.current) {
      const plumeScale = burnPulse > 0 ? 0.82 + burnPulse * 0.28 : 0.001;
      plumeRef.current.scale.lerp(new THREE.Vector3(1.05, 1.05, plumeScale), 0.22);
      coreRef.current.scale.lerp(new THREE.Vector3(0.5, 0.5, plumeScale * 1.08), 0.22);
      lightRef.current.intensity = THREE.MathUtils.lerp(lightRef.current.intensity, burnPulse * 1.6, 0.25);
    }

    if (cameraMode !== 'free') {
      const radial = point.clone().normalize();
      const side = new THREE.Vector3().crossVectors(radial, velocity).normalize();
      const directorAngle = clock.elapsedTime * 0.1;
      const target = point.clone().multiplyScalar(cameraMode === 'chase' ? 0.48 : 0.18);
      const desired =
        cameraMode === 'chase'
          ? point.clone().add(radial.multiplyScalar(0.48)).add(side.multiplyScalar(0.62)).sub(velocity.clone().multiplyScalar(0.82))
          : new THREE.Vector3(
              Math.sin(directorAngle) * 4.25,
              1.34 + Math.sin(directorAngle * 0.73) * 0.34,
              Math.cos(directorAngle) * 4.25,
            ).add(point.clone().multiplyScalar(0.12));
      camera.position.lerp(desired, 0.045);
      camera.lookAt(target);
    }

    if (clock.elapsedTime - lastHudUpdate.current > 0.1) {
      lastHudUpdate.current = clock.elapsedTime;
      const altitudeKm = Math.max(0, (point.length() - 1) * LUNAR_RADIUS_KM);
      onHud({
        phase,
        index,
        missionMin: phase * ORBIT_PERIOD_MIN,
        altitudeKm,
        velocityKms: 1.58 + 0.22 * (1 - Math.min(1, altitudeKm / 2000)),
        burnLabel: burn?.label ?? 'Coast arc',
      });
    }
  });

  return (
    <group>
      {ghostSegments.map((segment, i) => (
        <Line
          key={`ghost-${i}`}
          points={segment}
          color="#4d7691"
          lineWidth={1.1}
          transparent
          opacity={0.12 + i * 0.007}
          depthWrite={false}
        />
      ))}

      {ghostSegments.slice(0, 12).map((segment, i) => (
        <Line
          key={`glow-${i}`}
          points={segment}
          color="#7dd8ff"
          lineWidth={1.8}
          transparent
          opacity={0.04}
          depthWrite={false}
        />
      ))}

      {burnMarkers.map((burn) => (
        <OrbitMarker key={burn.label} point={burn.point} label={burn.label} />
      ))}

      <ProfessionalSpacecraft
        craftRef={craftRef}
        plumeRef={plumeRef}
        coreRef={coreRef}
        lightRef={lightRef}
      />
    </group>
  );
}

function OrbitCanvas({
  points,
  playing,
  speed,
  cameraMode,
  reducedMotion,
  onHud,
}: {
  points: THREE.Vector3[];
  playing: boolean;
  speed: number;
  cameraMode: CameraMode;
  reducedMotion: boolean;
  onHud: (state: HudState) => void;
}) {
  return (
    <Canvas
      camera={{ position: [3.5, 1.55, 4.1], fov: 38 }}
      dpr={[1, 2]}
      gl={{ alpha: false, antialias: true, powerPreference: 'high-performance' }}
      onCreated={({ gl }) => {
        gl.setClearColor('#03050a', 1);
        gl.outputColorSpace = THREE.SRGBColorSpace;
        gl.toneMapping = THREE.ACESFilmicToneMapping;
        gl.toneMappingExposure = 1.05;
      }}
    >
      <color attach="background" args={['#03050a']} />
      <fog attach="fog" args={['#03050a', 5.2, 9.5]} />

      <Stars radius={140} depth={70} count={reducedMotion ? 1200 : 3200} factor={3} saturation={0} fade speed={reducedMotion ? 0 : 0.22} />
      <ambientLight intensity={0.34} />
      <directionalLight position={[4, 3, 5]} intensity={2.15} color="#ffffff" />
      <pointLight position={[-3.4, -1.4, -3]} intensity={0.45} color="#55718e" />

      <Suspense fallback={null}>
        <MoonModel textureMode="aesthetic" displacementScale={0.045} animate={!reducedMotion && playing} />
      </Suspense>

      <CinematicOrbitRig
        points={points}
        playing={playing && !reducedMotion}
        speed={speed}
        cameraMode={cameraMode}
        onHud={onHud}
      />

      {cameraMode === 'free' && (
        <OrbitControls
          enablePan={false}
          enableZoom
          minDistance={2.6}
          maxDistance={7.2}
          rotateSpeed={0.42}
          zoomSpeed={0.5}
        />
      )}

      <EffectComposer>
        <Bloom intensity={1.2} luminanceThreshold={1.2} mipmapBlur />
      </EffectComposer>
    </Canvas>
  );
}

export default function CinematicOrbitScene() {
  const shellRef = useRef<HTMLDivElement>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<BlobPart[]>([]);
  const [webgl] = useState(() => hasWebGL());
  const [hud, setHud] = useState<HudState>(DEFAULT_HUD);
  const [playing, setPlaying] = useState(true);
  const [speed, setSpeed] = useState(1.0);
  const [cameraMode, setCameraMode] = useState<CameraMode>('director');
  const [recording, setRecording] = useState(false);
  const [recordStatus, setRecordStatus] = useState('Ready for browser WebM capture');
  const reducedMotion = useReducedMotion();

  const points = useMemo(
    () => toVectors((orbitData as unknown as OrbitDataset).path1?.path),
    [],
  );

  const canRecord = typeof window !== 'undefined' && 'MediaRecorder' in window;
  const effectivePlaying = playing && !reducedMotion;

  const startRecording = () => {
    const canvas = shellRef.current?.querySelector('canvas');
    if (!canvas || !canRecord) {
      setRecordStatus('Recording is not supported in this browser.');
      return;
    }

    const stream = canvas.captureStream(30);
    const mimeType = selectMimeType();
    chunksRef.current = [];
    const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
    recorderRef.current = recorder;
    recorder.ondataavailable = (event) => {
      if (event.data.size > 0) chunksRef.current.push(event.data);
    };
    recorder.onstop = () => {
      const blob = new Blob(chunksRef.current, { type: mimeType || 'video/webm' });
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement('a');
      anchor.href = url;
      anchor.download = 'lunaris-orbit-cinematic.webm';
      anchor.click();
      URL.revokeObjectURL(url);
      setRecording(false);
      setRecordStatus('Saved lunaris-orbit-cinematic.webm');
      stream.getTracks().forEach((track) => track.stop());
    };
    recorder.start(250);
    setRecording(true);
    setRecordStatus('Recording 12 seconds of canvas video...');
    window.setTimeout(() => {
      if (recorder.state !== 'inactive') recorder.stop();
    }, 12000);
  };

  const stopRecording = () => {
    if (recorderRef.current?.state === 'recording') {
      recorderRef.current.stop();
    }
  };

  if (webgl === false) {
    return (
      <section className={styles.shell}>
        <div className={styles.fallback}>
          <div className={styles.fallbackCard}>
            <h1>WebGL is unavailable</h1>
            <p>
              The cinematic orbit studio needs WebGL. Lunaris itself can still run;
              this route is an optional visual/export surface.
            </p>
          </div>
        </div>
      </section>
    );
  }

  return (
    <section className={styles.shell} ref={shellRef}>
      <div className={styles.canvasWrap} aria-hidden={webgl !== true}>
        {webgl && (
          <OrbitCanvas
            points={points}
            playing={effectivePlaying}
            speed={speed}
            cameraMode={cameraMode}
            reducedMotion={reducedMotion}
            onHud={setHud}
          />
        )}
      </div>

      <div className={styles.hud}>
        <p className={styles.kicker}>Lunaris cinematic orbit studio</p>
        <h1 className={styles.title}>Orbit Capture</h1>
        <p className={styles.subtitle}>
          A browser-native 3D shot builder for orbit videos. The route is offline,
          exportable, and separate from solver output.
        </p>
        <div className={styles.metrics}>
          <div className={styles.metric}>
            <span>Mission time</span>
            <strong>T+{hud.missionMin.toFixed(1)} min</strong>
          </div>
          <div className={styles.metric}>
            <span>Altitude</span>
            <strong>{hud.altitudeKm.toFixed(0)} km</strong>
          </div>
          <div className={styles.metric}>
            <span>Speed</span>
            <strong>{hud.velocityKms.toFixed(2)} km/s</strong>
          </div>
        </div>
        <p className={styles.notice}>
          Demo orbit for visualization only. Replace `public/orbit-data.json` with
          propagated samples before using the shot as mission evidence.
        </p>
      </div>

      <aside className={styles.controls} aria-label="Orbit video controls">
        <div className={styles.controlTitle}>
          <span>Director controls</span>
          <span>{reducedMotion ? 'Reduced motion' : 'Live'}</span>
        </div>
        <div className={styles.buttonRow}>
          {(['director', 'chase', 'free'] as CameraMode[]).map((mode) => (
            <button
              key={mode}
              type="button"
              className={`${styles.button} ${cameraMode === mode ? styles.active : ''}`}
              onClick={() => setCameraMode(mode)}
            >
              {mode}
            </button>
          ))}
        </div>
        <div className={styles.buttonRow}>
          <button
            type="button"
            className={`${styles.button} ${playing ? styles.active : ''}`}
            onClick={() => setPlaying((value) => !value)}
            disabled={reducedMotion}
          >
            {effectivePlaying ? 'Pause' : 'Play'}
          </button>
          <button type="button" className={styles.button} onClick={() => setSpeed(0.6)}>
            Slow
          </button>
          <button type="button" className={styles.button} onClick={() => setSpeed(1.4)}>
            Fast
          </button>
        </div>
        <label className={styles.field}>
          Playback speed: {speed.toFixed(1)}x
          <input
            type="range"
            min="0.4"
            max="1.8"
            step="0.1"
            value={speed}
            onChange={(event) => setSpeed(Number(event.target.value))}
          />
        </label>
        <button
          type="button"
          className={styles.primaryButton}
          onClick={recording ? stopRecording : startRecording}
          disabled={!canRecord}
        >
          {recording ? 'Stop recording' : 'Record 12s WebM'}
        </button>
        <p className={styles.recording}>{recordStatus}</p>
      </aside>

      <div className={styles.timeline}>
        <div className={styles.phase}>
          <span>Active segment</span>
          <strong>{hud.burnLabel}</strong>
        </div>
        <div className={styles.bar} aria-label="Orbit progress">
          <div className={styles.barFill} style={{ width: `${Math.max(0.02, hud.phase) * 100}%` }} />
        </div>
        <div className={styles.recording}>Frame index {hud.index}</div>
      </div>
    </section>
  );
}
