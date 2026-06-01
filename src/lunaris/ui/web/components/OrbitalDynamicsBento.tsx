'use client';
import Link from 'next/link';
import React, { useEffect, useState, useRef } from 'react';

// ─── Bezier helpers ────────────────────────────────────────────────────────
// Curve: M 20 200 Q 170 190 320 10   (P0, P1, P2)
const P0 = { x: 20,  y: 200 };
const P1 = { x: 170, y: 190 };
const P2 = { x: 320, y: 10  };

function bezierPoint(t: number) {
  const mt = 1 - t;
  return {
    x: mt * mt * P0.x + 2 * mt * t * P1.x + t * t * P2.x,
    y: mt * mt * P0.y + 2 * mt * t * P1.y + t * t * P2.y,
  };
}

function bezierTangent(t: number) {
  // B'(t) = 2(1-t)(P1-P0) + 2t(P2-P1)
  const mt = 1 - t;
  return {
    dx: 2 * mt * (P1.x - P0.x) + 2 * t * (P2.x - P1.x),
    dy: 2 * mt * (P1.y - P0.y) + 2 * t * (P2.y - P1.y),
  };
}

// Approximate arc-length for stroke-dasharray (Gauss 5-point)
const GAUSS_T = [0.046911, 0.230765, 0.5, 0.769235, 0.953089];
const GAUSS_W = [0.118463, 0.239314, 0.284444, 0.239314, 0.118463];
function bezierLength() {
  let len = 0;
  for (let i = 0; i < 5; i++) {
    const t = GAUSS_T[i];
    const { dx, dy } = bezierTangent(t);
    len += GAUSS_W[i] * Math.sqrt(dx * dx + dy * dy);
  }
  return len;
}
const CURVE_LEN = bezierLength(); // computed once at module level

export function OrbitalDynamicsBento() {
  const [progress, setProgress] = useState(0);
  const rafRef  = useRef<number | null>(null);
  const t0Ref   = useRef<number | null>(null);

  useEffect(() => {
    const RISE = 9000;
    const HOLD = 2000;
    const CYCLE = RISE + HOLD;

    const tick = (now: number) => {
      if (t0Ref.current === null) t0Ref.current = now;
      const elapsed = (now - t0Ref.current) % CYCLE;
      const raw = Math.min(elapsed / RISE, 1);
      // ease-in-cubic → strong "explosion" feel at the end
      setProgress(raw * raw * raw);
      rafRef.current = requestAnimationFrame(tick);
    };

    rafRef.current = requestAnimationFrame(tick);
    return () => { if (rafRef.current) cancelAnimationFrame(rafRef.current); };
  }, []);

  const pos = bezierPoint(progress);
  const tan = bezierTangent(progress);
  // +90 because rocket SVG is drawn pointing up (-Y), tangent points along travel
  const rocketAngle = Math.atan2(tan.dy, tan.dx) * (180 / Math.PI) + 90;

  const maxDegree   = 1200;
  const curDegree   = Math.floor(maxDegree * progress);
  const curOps      = curDegree * curDegree;
  const dashOffset  = CURVE_LEN * (1 - progress);

  return (
    <div className="premium-bento-container">

      {/* SECTION HEADER */}
      <div style={{ textAlign: 'center', marginBottom: '100px' }}>
        <h2 style={{ fontSize: '3.5rem', marginBottom: '24px', fontWeight: 300, letterSpacing: '0.02em', color: '#fff' }}>
          Orbital Dynamics Architecture
        </h2>
        <p style={{ fontSize: '1.15rem', lineHeight: 1.8, color: '#8A9FBD', maxWidth: '700px', margin: '0 auto', fontWeight: 300 }}>
          The Moon is not the passive, well-behaved sphere it appears to be. Beneath its silent surface lie ancient wounds — dense, irregular mass concentrations that silently hunt every satellite above. Lunaris was built to name each one.
        </p>
      </div>

      <div className="ambient-glow glow-1" />
      <div className="ambient-glow glow-2" />

      <div className="bento-grid">

        {/* ── CARD 1: MASCON CHAOS ───────────────────────────── */}
        <div className="glass-card card-full">
          <div className="card-bg">
            <svg width="100%" height="100%" viewBox="0 0 200 200"
                 preserveAspectRatio="xMidYMid slice"
                 style={{ opacity: 0.15, position: 'absolute', top: 0, left: 0 }}>
              <circle cx="100" cy="100" r="80" fill="none" stroke="#fff"
                      strokeWidth="1" strokeDasharray="2 4" />
              <path d="M 50 100 Q 80 150 120 70 T 170 120" fill="none"
                    stroke="#00E5FF" strokeWidth="2" />
              <path d="M 60 80 Q 100 120 140 60" fill="none"
                    stroke="#FF3366" strokeWidth="1.5" />
            </svg>
          </div>
          <div className="card-content">
            <h3 className="card-title">Invisible Lassos</h3>
            <p className="card-text">
              From afar, the Moon reads as a <span className="hl">serene, uniform sphere</span>.
              Up close, it is a minefield. Asymmetric mass concentrations — mascons — lurk beneath
              the regolith, the <span className="hl">fossilized scars</span> of ancient impacts.
              Their gravitational pull is not predictable like Earth's. It is{' '}
              <span className="hl">erratic, asymmetric, and lethal</span> to orbital stability —
              silently wrenching satellites off course until decay becomes inevitable.
            </p>
          </div>
        </div>

        {/* ── CARD 2: O(N²) COMPLEXITY GRAPH ────────────────── */}
        <div className="glass-card card-full">
          <div className="card-content chart-layout">

            <div className="text-section">
              <h3 className="card-title">The Cost of Precision</h3>
              <p className="card-text">
                To model this gravitational chaos with fidelity, traditional engines must expand
                the gravity field as a spherical-harmonic series — thousands of terms, recomputed{' '}
                <span className="hl">at every integration step</span>. The computation scales as
                O(N²). Each increase in precision doesn't add work; it{' '}
                <span className="hl">multiplies it</span>. Beyond a certain degree, the math
                doesn't just slow down — it{' '}
                <span className="hl">becomes impossible in real time</span>.
              </p>
            </div>

            {/* ── Chart row ── */}
            <div className="chart-section">

              {/* Left: live counters + milestones */}
              <div className="chart-left-panel">
                <div className="counter-box">
                  <div className="counter-row">
                    <span className="counter-label">HARMONIC DEGREE (N)</span>
                    <span className="counter-value">{curDegree.toLocaleString()}</span>
                  </div>
                  <div className="counter-divider" />
                  <div className="counter-row danger">
                    <span className="counter-label">OPERATIONS · O(N²)</span>
                    <span className="counter-value">{curOps.toLocaleString()}</span>
                  </div>
                </div>

                <div className="milestones-box">
                  <p className="milestones-title">COMPUTATION SCALE</p>
                  <div className="milestones-list">
                    {[
                      { name: 'SH20',   val: '400 ops',   pct: 0.05 },
                      { name: 'SH200',  val: '40 K ops',  pct: 0.25 },
                      { name: 'SH500',  val: '250 K ops', pct: 0.55 },
                      { name: 'SH1200', val: '1.44 M ops',pct: 1.0, danger: true },
                    ].map(m => (
                      <div key={m.name}
                           className={`milestone${m.danger ? ' danger-m' : ''}${progress >= m.pct ? ' active' : ''}`}>
                        <span className="m-name">{m.name}</span>
                        <span className="m-val">{m.val}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              {/* Right: SVG chart */}
              <div className="chart-right-panel">
                {/* Y-label as HTML – never clipped */}
                <div className="y-axis-label">COMPUTATIONAL LOAD</div>

                <div className="svg-wrapper">
                  <svg
                    viewBox="0 0 340 225"
                    preserveAspectRatio="xMidYMid meet"
                    style={{ width: '100%', height: '100%', display: 'block', overflow: 'visible' }}
                  >
                    <defs>
                      <linearGradient id="areaGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                        <stop offset="0%"   stopColor="rgba(255,51,102,0.0)" />
                        <stop offset="60%"  stopColor="rgba(255,51,102,0.3)" />
                        <stop offset="100%" stopColor="rgba(255,51,102,0.8)" />
                      </linearGradient>
                      <linearGradient id="curveGrad" x1="0%" y1="100%" x2="100%" y2="0%">
                        <stop offset="0%"   stopColor="#FF8A65" />
                        <stop offset="100%" stopColor="#FF0033" />
                      </linearGradient>
                      <filter id="glow" x="-30%" y="-30%" width="160%" height="160%">
                        <feGaussianBlur stdDeviation="4" result="b"/>
                        <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
                      </filter>
                      <filter id="softGlow" x="-50%" y="-50%" width="200%" height="200%">
                        <feGaussianBlur stdDeviation="7" result="b"/>
                        <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
                      </filter>
                      <clipPath id="chartArea">
                        <rect x="20" y="10" width="300" height="190" />
                      </clipPath>
                      <pattern id="grid" width="25" height="25" patternUnits="userSpaceOnUse">
                        <path d="M25 0 L0 0 0 25" fill="none"
                              stroke="rgba(255,255,255,0.04)" strokeWidth="1"/>
                      </pattern>
                    </defs>

                    {/* Grid background */}
                    <rect x="20" y="10" width="300" height="190" fill="url(#grid)" />

                    {/* Danger fill — clipped, opacity follows progress */}
                    <path
                      d="M 20 200 Q 170 190 320 10 L 320 200 Z"
                      fill="url(#areaGrad)"
                      opacity={progress * 0.85}
                      clipPath="url(#chartArea)"
                    />

                    {/* Main curve — drawn by dashoffset */}
                    <path
                      d="M 20 200 Q 170 190 320 10"
                      fill="none"
                      stroke="url(#curveGrad)"
                      strokeWidth="3.5"
                      strokeLinecap="round"
                      filter="url(#softGlow)"
                      strokeDasharray={CURVE_LEN}
                      strokeDashoffset={dashOffset}
                      clipPath="url(#chartArea)"
                    />

                    {/* Axes */}
                    <line x1="20" y1="200" x2="320" y2="200"
                          stroke="rgba(255,255,255,0.2)" strokeWidth="1.5"/>
                    <line x1="20" y1="10"  x2="20"  y2="200"
                          stroke="rgba(255,255,255,0.2)" strokeWidth="1.5"/>

                    {/* Rocket — placed on curve, invisible until progress > 3% */}
                    <g
                      transform={`translate(${pos.x},${pos.y}) rotate(${rocketAngle})`}
                      opacity={progress > 0.03 ? 1 : 0}
                      filter="url(#glow)"
                    >
                      {/* Body */}
                      <path d="M 0,-11 L 4,-4 L 4,6 L -4,6 L -4,-4 Z" fill="#fff"/>
                      {/* Nose */}
                      <path d="M 0,-11 L 3.5,-4 L -3.5,-4 Z" fill="#e0e0ff"/>
                      {/* Porthole */}
                      <circle cx="0" cy="0" r="1.6" fill="#0d1033" stroke="#aab" strokeWidth="0.6"/>
                      {/* Left fin */}
                      <path d="M -4,4 L -8,9 L -4,7 Z" fill="#c0c0d8"/>
                      {/* Right fin */}
                      <path d="M  4,4 L  8,9 L  4,7 Z" fill="#c0c0d8"/>
                      {/* Core flame */}
                      <path d="M -2.5,6 Q 0,14 2.5,6" fill="#FF4422" opacity="0.95"/>
                      {/* Outer flame */}
                      <path d="M -1.5,6 Q 0,20 1.5,6" fill="#FF9955" opacity="0.55"/>
                    </g>

                    {/* X-axis label */}
                    <text x="170" y="218" textAnchor="middle"
                          fill="rgba(138,159,189,0.65)"
                          fontSize="9" fontFamily="'Inter',sans-serif"
                          fontWeight="700" letterSpacing="2.5">
                      MODEL FIDELITY (SH DEGREE)
                    </text>

                    {/* Curve title */}
                    <text x="230" y="28" textAnchor="middle"
                          fill="rgba(255,255,255,0.9)"
                          fontSize="12" fontFamily="'Inter',sans-serif"
                          fontWeight="700" letterSpacing="1"
                          filter="url(#glow)">
                      O(N²) Explosion
                    </text>
                  </svg>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* ── CARD 3: CIS-LUNAR CHAOS ───────────────────────── */}
        <div className="glass-card card-full">
          <div className="card-bg">
            <svg width="100%" height="100%" viewBox="0 0 200 200"
                 preserveAspectRatio="xMidYMid slice"
                 style={{ opacity: 0.1, position: 'absolute', right: 0, bottom: 0 }}>
              <circle cx="150" cy="150" r="100" fill="none" stroke="#fff"
                      strokeWidth="0.5" strokeDasharray="5 5"/>
              <path d="M 0 200 Q 100 150 200 50" fill="none"
                    stroke="rgba(255,255,255,0.3)" strokeWidth="1"/>
              <path d="M 50 200 Q 150 150 200 100" fill="none"
                    stroke="rgba(255,255,255,0.2)" strokeWidth="1"/>
            </svg>
          </div>
          <div className="card-content">
            <h3 className="card-title">Cislunar Storms</h3>
            <p className="card-text">
              Mascons are only the beginning. In the broader cislunar environment, a spacecraft is
              simultaneously pulled apart by <span className="hl">Earth's tidal forces</span>,
              nudged off-axis by <span className="hl">solar radiation pressure</span>, and dragged
              through the unpredictable influence of a three-body system. These forces don't
              compound linearly — they interact, resonate, and cascade. What appears as a stable
              orbit at one moment can become{' '}
              <span className="hl">unrecoverably chaotic</span> in the next.
            </p>
          </div>
        </div>

        {/* ── CARD 4: ST-LRPS SAVIOR ─────────────────────────── */}
        <div className="glass-card savior-card">
          {/* Top accent line */}
          <div className="savior-accent" />

          {/* Ambient neural glow behind content */}
          <div className="savior-bg-glow" />

          <div className="card-content savior-layout">

            {/* Left: text column */}
            <div className="savior-text-col">
              <div className="savior-badge">ST-LRPS SYSTEM</div>
              <h3 className="savior-title">Taming the Impossible</h3>
              <p className="card-text" style={{ marginBottom: '28px' }}>
                Lunaris answers this challenge not by solving the equations faster — but by
                learning to <span className="hl">bypass them entirely</span>. Our Surrogate-based
                Trajectory Long-Range Prediction System replaces cascading matrix arithmetic with
                a neural architecture that has already internalized the physics. The result is a
                gravitational engine that answers in{' '}
                <span className="hl">microseconds</span> what classical solvers take minutes to
                compute.
              </p>
              <div className="savior-quote">
                Not a faster calculator. A fundamentally different kind of oracle — one that has
                memorized the shape of chaos and can navigate it at the speed of thought.
              </div>
            </div>

            {/* Right: visual panel + CTA */}
            <div className="savior-right-col">

              {/* Neural net / orbit visualization */}
              <div className="savior-visual">
                <svg viewBox="0 0 220 160" style={{ width: '100%', height: 'auto', display: 'block' }}>
                  <defs>
                    <radialGradient id="nodeGrad" cx="50%" cy="50%">
                      <stop offset="0%"   stopColor="#a5b4fc" stopOpacity="1"/>
                      <stop offset="100%" stopColor="#5A33FF" stopOpacity="0.4"/>
                    </radialGradient>
                    <radialGradient id="nodeGrad2" cx="50%" cy="50%">
                      <stop offset="0%"   stopColor="#00E5FF" stopOpacity="1"/>
                      <stop offset="100%" stopColor="#0066aa" stopOpacity="0.4"/>
                    </radialGradient>
                    <filter id="nodeGlow" x="-60%" y="-60%" width="220%" height="220%">
                      <feGaussianBlur stdDeviation="5" result="b"/>
                      <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
                    </filter>
                    <filter id="lineGlowSavior" x="-30%" y="-30%" width="160%" height="160%">
                      <feGaussianBlur stdDeviation="2" result="b"/>
                      <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
                    </filter>
                  </defs>

                  {/* Connection lines — input → hidden */}
                  {[30,60,90,120,150].map((y, i) =>
                    [50, 80, 110].map((hy, j) => (
                      <line key={`ih-${i}-${j}`}
                            x1="35" y1={y} x2="105" y2={hy}
                            stroke="rgba(165,180,252,0.12)" strokeWidth="1"
                            filter="url(#lineGlowSavior)"/>
                    ))
                  )}
                  {/* Hidden → output */}
                  {[50,80,110].map((y, i) =>
                    [65, 95].map((oy, j) => (
                      <line key={`ho-${i}-${j}`}
                            x1="105" y1={y} x2="175" y2={oy}
                            stroke="rgba(0,229,255,0.15)" strokeWidth="1"
                            filter="url(#lineGlowSavior)"/>
                    ))
                  )}

                  {/* Input layer */}
                  {[30,60,90,120,150].map((y, i) => (
                    <g key={`in-${i}`} filter="url(#nodeGlow)">
                      <circle cx="35" cy={y} r="7" fill="url(#nodeGrad)" opacity="0.8"/>
                      <circle cx="35" cy={y} r="3" fill="#a5b4fc"/>
                    </g>
                  ))}

                  {/* Hidden layer */}
                  {[50,80,110].map((y, i) => (
                    <g key={`h-${i}`} filter="url(#nodeGlow)">
                      <circle cx="105" cy={y} r="9"  fill="url(#nodeGrad)"  opacity="0.9"/>
                      <circle cx="105" cy={y} r="4"  fill="#c4b5fd"/>
                    </g>
                  ))}

                  {/* Output layer */}
                  {[65,95].map((y, i) => (
                    <g key={`out-${i}`} filter="url(#nodeGlow)">
                      <circle cx="175" cy={y} r="10" fill="url(#nodeGrad2)" opacity="0.9"/>
                      <circle cx="175" cy={y} r="4"  fill="#00E5FF"/>
                    </g>
                  ))}

                  {/* Labels */}
                  <text x="35" y="170" textAnchor="middle" fill="rgba(165,180,252,0.5)"
                        fontSize="8" fontFamily="'Inter',sans-serif" letterSpacing="1">INPUTS</text>
                  <text x="105" y="130" textAnchor="middle" fill="rgba(165,180,252,0.5)"
                        fontSize="8" fontFamily="'Inter',sans-serif" letterSpacing="1">SURROGATE</text>
                  <text x="175" y="115" textAnchor="middle" fill="rgba(0,229,255,0.5)"
                        fontSize="8" fontFamily="'Inter',sans-serif" letterSpacing="1">OUTPUT</text>
                </svg>

                {/* Speed badge */}
                <div className="speed-badge">
                  <span className="speed-icon">⚡</span>
                  <span className="speed-text">μs Inference</span>
                </div>
              </div>

              {/* CTA */}
              <Link href="/st-lrps" className="premium-cta">
                <div className="cta-content">
                  <span className="cta-text">INITIALIZE ARCHITECTURE</span>
                  <svg className="cta-arrow" width="20" height="20" viewBox="0 0 24 24"
                       fill="none" stroke="currentColor" strokeWidth="2.5"
                       strokeLinecap="round" strokeLinejoin="round">
                    <line x1="5" y1="12" x2="19" y2="12"/>
                    <polyline points="12 5 19 12 12 19"/>
                  </svg>
                </div>
                <div className="cta-glow"/>
              </Link>
            </div>

          </div>
        </div>

      </div>

      <style jsx>{`
        /* ── Container ──────────────────────────────────── */
        .premium-bento-container {
          max-width: 1200px;
          margin: 100px auto;
          padding: 0 24px;
          font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
          position: relative;
        }

        /* ── Ambient glows ──────────────────────────────── */
        .ambient-glow {
          position: absolute;
          border-radius: 50%;
          filter: blur(120px);
          opacity: 0.12;
          z-index: -1;
          pointer-events: none;
        }
        .glow-1 { top: 10%; left: -10%; width: 500px; height: 500px; background: #00E5FF; }
        .glow-2 { bottom: 10%; right: -10%; width: 600px; height: 600px; background: #5A33FF; }

        /* ── Grid ───────────────────────────────────────── */
        .bento-grid {
          display: flex;
          flex-direction: column;
          gap: 32px;
        }

        /* ── Glass card ─────────────────────────────────── */
        .glass-card {
          position: relative;
          background: rgba(12, 16, 28, 0.4);
          border: 1px solid rgba(255, 255, 255, 0.06);
          border-radius: 28px;
          padding: 48px;
          overflow: hidden;
          backdrop-filter: blur(20px);
          -webkit-backdrop-filter: blur(20px);
          transition: transform .5s cubic-bezier(.16,1,.3,1),
                      border-color .5s ease, box-shadow .5s ease, background .5s ease;
          box-shadow: 0 10px 40px rgba(0,0,0,0.15);
        }
        .glass-card:hover {
          transform: translateY(-4px);
          border-color: rgba(255,255,255,0.12);
          background: rgba(16,22,38,0.5);
          box-shadow: 0 20px 50px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.05);
        }

        .card-bg { position: absolute; top:0;left:0;right:0;bottom:0; pointer-events:none; z-index:0; }
        .card-content { position:relative; z-index:1; display:flex; flex-direction:column; height:100%; }

        /* ── Typography ─────────────────────────────────── */
        .card-title {
          font-size: 1.8rem; font-weight: 600; letter-spacing: -0.02em;
          color: #fff; margin: 0 0 24px; line-height: 1.2;
        }
        .card-text {
          font-size: 1.1rem; line-height: 1.8;
          color: rgba(255,255,255,0.65); font-weight: 300; margin: 0;
        }
        .hl {
          color: #fff; font-weight: 500;
          text-shadow: 0 0 12px rgba(255,255,255,0.5);
          transition: text-shadow .3s ease;
        }
        .glass-card:hover .hl { text-shadow: 0 0 20px rgba(255,255,255,0.8); }

        /* ── Chart card ─────────────────────────────────── */
        .chart-layout  { display:flex; flex-direction:column; }
        .text-section  { }

        .chart-section {
          display: flex;
          gap: 32px;
          margin-top: 40px;
          align-items: stretch;
          min-height: 340px;
          border-top: 1px solid rgba(255,255,255,0.05);
          padding-top: 32px;
        }

        /* Left panel */
        .chart-left-panel {
          display: flex; flex-direction: column;
          gap: 24px; flex-shrink: 0; width: 230px;
        }
        .counter-box {
          background: rgba(8,12,26,0.9);
          border: 1px solid rgba(255,255,255,0.08);
          border-radius: 14px;
          padding: 20px 22px;
          display: flex; flex-direction: column; gap: 14px;
        }
        .counter-divider { height: 1px; background: rgba(255,255,255,0.06); }
        .counter-row     { display:flex; flex-direction:column; gap:4px; }
        .counter-label   {
          font-size: 0.58rem; letter-spacing: 0.18em;
          color: #4A6080; font-weight: 700; text-transform: uppercase;
        }
        .counter-value   {
          font-size: 1.55rem; color: #fff;
          font-family: 'Courier New', monospace;
          font-weight: 600; letter-spacing: -0.02em;
        }
        .counter-row.danger .counter-value {
          color: #FF3366;
          text-shadow: 0 0 14px rgba(255,51,102,0.7);
        }

        .milestones-box  { padding: 4px 0; }
        .milestones-title {
          color: rgba(255,255,255,0.3); font-size: 0.58rem;
          letter-spacing: 0.18em; font-weight: 700; margin: 0 0 10px;
        }
        .milestones-list { display:flex; flex-direction:column; gap:6px; }

        .milestone {
          display: flex; justify-content: space-between; align-items: center;
          padding: 7px 11px;
          background: rgba(255,255,255,0.02);
          border-radius: 7px;
          border-left: 2px solid rgba(138,159,189,0.25);
          transition: border-color .4s ease, background .4s ease;
        }
        .milestone.active {
          border-left-color: rgba(138,159,189,0.7);
          background: rgba(138,159,189,0.05);
        }
        .milestone.danger-m { border-left-color: rgba(255,51,102,0.3); }
        .milestone.danger-m.active {
          border-left-color: #FF3366;
          background: rgba(255,51,102,0.07);
        }
        .m-name { color: #8A9FBD; font-weight: 600; font-size: 0.78rem; letter-spacing:.04em; }
        .m-val  { color: #fff; font-family: 'Courier New',monospace; font-size: 0.78rem; }
        .danger-m .m-name, .danger-m .m-val { color: #FF6680; }

        /* Right panel: y-label + SVG */
        .chart-right-panel {
          flex: 1; display: flex; align-items: stretch;
          gap: 0; position: relative; min-height: 280px;
        }
        .y-axis-label {
          writing-mode: vertical-rl;
          text-orientation: mixed;
          transform: rotate(180deg);
          font-size: 0.58rem; letter-spacing: 0.2em; font-weight: 700;
          color: rgba(138,159,189,0.5); text-transform: uppercase;
          display: flex; align-items: center; justify-content: center;
          padding-right: 10px; flex-shrink: 0; user-select: none;
        }
        .svg-wrapper { flex: 1; position: relative; }

        /* ── Savior card ─────────────────────────────────── */
        .savior-card {
          background: rgba(8, 10, 22, 0.65) !important;
          border-color: rgba(165,180,252,0.12) !important;
          overflow: visible !important;
        }
        .savior-accent {
          position: absolute; top: 0; left: 0; width: 100%; height: 1px;
          background: linear-gradient(90deg, transparent, rgba(165,180,252,0.5), rgba(0,229,255,0.3), transparent);
        }
        .savior-bg-glow {
          position: absolute;
          top: -80px; right: -80px;
          width: 400px; height: 400px;
          border-radius: 50%;
          background: radial-gradient(circle, rgba(90,51,255,0.15) 0%, transparent 70%);
          pointer-events: none;
          z-index: 0;
        }

        .savior-layout {
          flex-direction: column;
          gap: 48px;
        }
        @media (min-width: 900px) {
          .savior-layout { flex-direction: row; align-items: center; gap: 60px; }
        }

        .savior-text-col { flex: 1; }

        .savior-badge {
          display: inline-block;
          padding: 5px 14px;
          background: rgba(165,180,252,0.08);
          border: 1px solid rgba(165,180,252,0.2);
          border-radius: 30px;
          font-size: 0.65rem;
          letter-spacing: 0.2em;
          color: #a5b4fc;
          font-weight: 700;
          margin-bottom: 18px;
          text-transform: uppercase;
        }

        .savior-title {
          font-size: 2.2rem;
          font-weight: 600;
          letter-spacing: -0.02em;
          color: #fff;
          margin: 0 0 22px;
          line-height: 1.2;
          background: linear-gradient(135deg, #fff 0%, #a5b4fc 60%, #00E5FF 100%);
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
        }

        .savior-quote {
          margin-top: 24px;
          padding: 16px 20px;
          background: rgba(165,180,252,0.04);
          border-left: 2px solid rgba(165,180,252,0.3);
          border-radius: 0 10px 10px 0;
          font-size: 0.92rem;
          line-height: 1.7;
          color: rgba(255,255,255,0.4);
          font-style: italic;
        }

        /* Right column */
        .savior-right-col {
          flex-shrink: 0;
          width: 280px;
          display: flex;
          flex-direction: column;
          gap: 24px;
        }

        .savior-visual {
          position: relative;
          background: rgba(5,8,20,0.6);
          border: 1px solid rgba(165,180,252,0.1);
          border-radius: 18px;
          padding: 16px;
          overflow: hidden;
        }

        .speed-badge {
          position: absolute;
          top: 12px; right: 12px;
          display: flex; align-items: center; gap: 6px;
          padding: 4px 10px;
          background: rgba(0,229,255,0.08);
          border: 1px solid rgba(0,229,255,0.25);
          border-radius: 20px;
        }
        .speed-icon { font-size: 0.75rem; }
        .speed-text {
          font-size: 0.65rem; font-weight: 700;
          letter-spacing: 0.1em; color: #00E5FF;
        }

        /* CTA */
        .premium-cta {
          display: block; position: relative; text-decoration: none;
          border-radius: 14px;
          background: linear-gradient(135deg, rgba(165,180,252,0.08), rgba(0,229,255,0.05));
          border: 1px solid rgba(165,180,252,0.2);
          overflow: hidden;
          transition: all .4s ease;
        }
        .cta-content {
          position: relative; z-index: 2;
          display: flex; align-items: center; justify-content: space-between;
          padding: 18px 24px;
        }
        .cta-text {
          color: #fff; font-weight: 700; font-size: 0.82rem;
          letter-spacing: 0.12em;
        }
        .cta-arrow { color: #a5b4fc; transition: transform .4s ease; }
        .cta-glow {
          position: absolute; top:0;left:0;right:0;bottom:0;
          background: linear-gradient(135deg, rgba(165,180,252,0.15) 0%, rgba(0,229,255,0.08) 100%);
          opacity: 0; z-index: 1; transition: opacity .4s ease;
        }
        .premium-cta:hover {
          border-color: rgba(165,180,252,0.5);
          box-shadow: 0 8px 30px rgba(90,51,255,0.25);
          transform: translateY(-2px);
        }
        .premium-cta:hover .cta-glow  { opacity: 1; }
        .premium-cta:hover .cta-arrow { transform: translateX(5px); }
      `}</style>
    </div>
  );
}
