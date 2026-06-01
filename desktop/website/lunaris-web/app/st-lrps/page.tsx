'use client';
import ArchitectureBuilder from '@/components/ArchitectureBuilder';

export default function ST_LRPS() {
  return (
    <main style={{ minHeight: '100vh', background: '#050508', color: '#fff', fontFamily: 'inherit' }}>

      {/* ─── HERO ─── */}
      <section style={{ position: 'relative', minHeight: '100vh', display: 'flex', alignItems: 'center', overflow: 'hidden' }}>

        {/* Animated background grid */}
        <div style={{
          position: 'absolute', inset: 0, zIndex: 0,
          backgroundImage: `
            linear-gradient(rgba(0,229,255,0.025) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0,229,255,0.025) 1px, transparent 1px)
          `,
          backgroundSize: '80px 80px',
          maskImage: 'radial-gradient(ellipse 80% 80% at 50% 50%, black 30%, transparent 100%)',
          WebkitMaskImage: 'radial-gradient(ellipse 80% 80% at 50% 50%, black 30%, transparent 100%)',
        }} />

        {/* Glow blobs */}
        <div style={{ position: 'absolute', top: '15%', left: '-5%', width: '700px', height: '700px', borderRadius: '50%', background: 'radial-gradient(circle, rgba(0,229,255,0.07) 0%, transparent 65%)', pointerEvents: 'none', zIndex: 0 }} />
        <div style={{ position: 'absolute', bottom: '5%', right: '-10%', width: '900px', height: '600px', borderRadius: '50%', background: 'radial-gradient(circle, rgba(123,92,240,0.08) 0%, transparent 65%)', pointerEvents: 'none', zIndex: 0 }} />

        <div style={{ maxWidth: '1100px', margin: '0 auto', padding: '160px 40px 80px', position: 'relative', zIndex: 1, width: '100%' }}>

          {/* Eyebrow */}
          <div style={{ display: 'inline-flex', alignItems: 'center', gap: '12px', marginBottom: '40px', padding: '9px 20px', border: '1px solid rgba(0,229,255,0.18)', borderRadius: '40px', background: 'rgba(0,229,255,0.05)', backdropFilter: 'blur(8px)' }}>
            <span style={{ display: 'inline-block', width: '7px', height: '7px', borderRadius: '50%', background: '#00E5FF', boxShadow: '0 0 10px #00E5FF, 0 0 20px rgba(0,229,255,0.4)', animation: 'pulse 2s infinite' }} />
            <span style={{ fontSize: '0.7rem', fontWeight: 700, letterSpacing: '0.2em', textTransform: 'uppercase', color: '#00E5FF' }}>
              Sobolev-Trained Lunar Residual Potential Surrogate
            </span>
          </div>

          {/* Main headline */}
          <h1 style={{ fontSize: 'clamp(3rem, 6.5vw, 6rem)', fontWeight: 100, letterSpacing: '-0.04em', lineHeight: 1.02, marginBottom: '0', maxWidth: '900px' }}>
            The Moon pulls on<br />
            every satellite differently.<br />
            <span style={{
              fontWeight: 400,
              background: 'linear-gradient(100deg, #00E5FF 0%, #7B5CF0 50%, #FF7B54 100%)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
              backgroundClip: 'text',
            }}>
              ST-LRPS learned why.
            </span>
          </h1>

          {/* Sub-headline */}
          <p style={{ fontSize: '1.2rem', lineHeight: 1.8, color: '#7A90B0', maxWidth: '640px', marginTop: '40px', fontWeight: 300 }}>
            Traditional high-fidelity propagation evaluates 14,641 spherical harmonic terms at every integration step.
            Across millions of steps, that cost is existential. ST-LRPS replaces the expensive part
            with a physics-informed neural network that delivers the same accuracy — at a fraction of the compute.
          </p>

          {/* CTA metrics */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, auto)', gap: '0', width: 'fit-content', marginTop: '64px', border: '1px solid rgba(255,255,255,0.07)', borderRadius: '20px', overflow: 'hidden', backdropFilter: 'blur(12px)', background: 'rgba(255,255,255,0.02)' }}>
            {[
              { n: '~1000', unit: '×', label: 'Faster than SH120' },
              { n: '<2', unit: 'm', label: 'RMS error per orbit' },
              { n: 'O(1)', unit: '', label: 'Inference complexity' },
            ].map((s, i) => (
              <div key={i} style={{ padding: '32px 44px', borderRight: i < 2 ? '1px solid rgba(255,255,255,0.07)' : 'none', textAlign: 'center' }}>
                <div style={{ fontSize: '2.8rem', fontWeight: 200, color: '#fff', letterSpacing: '-0.03em', lineHeight: 1 }}>
                  {s.n}<span style={{ color: '#00E5FF', fontSize: '2rem' }}>{s.unit}</span>
                </div>
                <div style={{ fontSize: '0.72rem', fontWeight: 600, letterSpacing: '0.12em', textTransform: 'uppercase', color: '#4A6080', marginTop: '12px' }}>{s.label}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ─── PROBLEM → SOLUTION ─── */}
      <section style={{ padding: '100px 0', borderTop: '1px solid rgba(255,255,255,0.04)' }}>
        <div style={{ maxWidth: '1100px', margin: '0 auto', padding: '0 40px', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '80px', alignItems: 'center' }}>
          <div>
            <span style={{ fontSize: '0.7rem', fontWeight: 700, letterSpacing: '0.18em', textTransform: 'uppercase', color: '#FF7B54' }}>The Problem</span>
            <h2 style={{ fontSize: '2.4rem', fontWeight: 200, letterSpacing: '-0.02em', color: '#fff', margin: '20px 0 24px', lineHeight: 1.15 }}>
              Quadratic cost.<br />No escape.
            </h2>
            <p style={{ color: '#7A90B0', lineHeight: 1.8, fontSize: '1.05rem', marginBottom: '20px' }}>
              The GRAIL GRGM1200A model resolves the Moon's gravity at 120th degree and order — 14,641 harmonic coefficients.
              Each step of a numerical integrator must evaluate all of them. The cost scales as O(N²) with model degree.
            </p>
            <p style={{ color: '#5A6F8D', lineHeight: 1.8, fontSize: '1rem' }}>
              A single Monte Carlo campaign with 10,000 trajectories and 500,000 integration steps each
              requires <em style={{ color: '#8A9FBD' }}>73 billion</em> harmonic evaluations.
              Realism and speed have always been enemies. Until now.
            </p>
          </div>
          <div>
            <span style={{ fontSize: '0.7rem', fontWeight: 700, letterSpacing: '0.18em', textTransform: 'uppercase', color: '#00E5FF' }}>The Solution</span>
            <h2 style={{ fontSize: '2.4rem', fontWeight: 200, letterSpacing: '-0.02em', color: '#fff', margin: '20px 0 24px', lineHeight: 1.15 }}>
              Learn the residual.<br />Skip the rest.
            </h2>
            <p style={{ color: '#7A90B0', lineHeight: 1.8, fontSize: '1.05rem', marginBottom: '20px' }}>
              ST-LRPS computes a fast, exact low-degree baseline (SH20) analytically,
              then calls a compact neural network for the residual — everything the analytic model missed.
              The network runs in constant time regardless of fidelity.
            </p>
            <p style={{ color: '#5A6F8D', lineHeight: 1.8, fontSize: '1rem' }}>
              The key innovation: training with <em style={{ color: '#8A9FBD' }}>Sobolev losses</em> that
              penalize gradient error — not just value error — so the inferred forces are as trustworthy as the potential.
            </p>
          </div>
        </div>
      </section>

      {/* ─── HOW IT WORKS ─── */}
      <section style={{ padding: '80px 0 100px', background: 'rgba(255,255,255,0.01)' }}>
        <div style={{ maxWidth: '1100px', margin: '0 auto', padding: '0 40px' }}>
          <div style={{ textAlign: 'center', marginBottom: '72px' }}>
            <h2 style={{ fontSize: '2.2rem', fontWeight: 200, letterSpacing: '-0.02em', color: '#fff', marginBottom: '16px' }}>Three stages</h2>
            <p style={{ color: '#5A6F8D', fontSize: '1.05rem', maxWidth: '480px', margin: '0 auto', lineHeight: 1.7 }}>
              Model what physics gives you for free. Learn what it can't.
            </p>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '20px' }}>
            {[
              {
                step: '01',
                title: 'Analytic Baseline',
                color: '#00E5FF',
                body: 'A low-degree (20×20) spherical harmonic model computes the dominant gravitational potential analytically — fast, exact, deterministic. It strips away the easy part of the signal so the neural network only needs to learn what remains.',
                tags: ['SH 20×20', 'GRGM1200A', 'Deterministic'],
              },
              {
                step: '02',
                title: 'Residual Neural Potential',
                color: '#7B5CF0',
                body: 'A compact MLP ingests a physics-aware radial encoding of position and outputs a single scalar: the residual gravitational potential. Training minimizes both potential and gradient error simultaneously — ensuring forces, not just values, are accurate.',
                tags: ['Sobolev Loss', '∇φ penalized', 'MLP'],
              },
              {
                step: '03',
                title: 'Autograd Differentiation',
                color: '#FFB347',
                body: 'At runtime, PyTorch autograd differentiates the scalar output with respect to the input position — yielding the residual acceleration vector exactly, without finite differences or numerical noise. It is added to the analytic baseline and fed into the integrator.',
                tags: ['a = −∇φ_NN', 'Exact gradient', 'Zero noise'],
              },
            ].map((card) => (
              <div
                key={card.step}
                style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.05)', borderRadius: '24px', padding: '40px', backdropFilter: 'blur(12px)', transition: 'border-color 0.4s, transform 0.4s', position: 'relative', overflow: 'hidden', cursor: 'default' }}
                onMouseOver={(e) => { (e.currentTarget as HTMLDivElement).style.borderColor = card.color + '35'; (e.currentTarget as HTMLDivElement).style.transform = 'translateY(-5px)'; }}
                onMouseOut={(e) => { (e.currentTarget as HTMLDivElement).style.borderColor = 'rgba(255,255,255,0.05)'; (e.currentTarget as HTMLDivElement).style.transform = 'translateY(0)'; }}
              >
                <div style={{ position: 'absolute', top: '16px', right: '24px', fontSize: '4.5rem', fontWeight: 800, color: card.color, opacity: 0.04, lineHeight: 1, letterSpacing: '-0.05em', pointerEvents: 'none', userSelect: 'none' }}>{card.step}</div>

                <div style={{ display: 'inline-flex', alignItems: 'center', justifyContent: 'center', width: '32px', height: '32px', borderRadius: '8px', background: card.color + '15', border: `1px solid ${card.color}25`, marginBottom: '24px' }}>
                  <span style={{ fontSize: '0.65rem', fontWeight: 700, color: card.color }}>{card.step}</span>
                </div>

                <h3 style={{ fontSize: '1.35rem', fontWeight: 300, color: '#fff', marginBottom: '16px', letterSpacing: '-0.01em' }}>{card.title}</h3>
                <p style={{ color: '#6A8098', lineHeight: 1.75, fontSize: '0.95rem', marginBottom: '28px' }}>{card.body}</p>

                {/* Pill tags instead of code block */}
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
                  {card.tags.map(tag => (
                    <span key={tag} style={{
                      display: 'inline-block',
                      padding: '5px 14px',
                      borderRadius: '20px',
                      fontSize: '0.75rem',
                      fontWeight: 500,
                      letterSpacing: '0.03em',
                      color: card.color,
                      background: card.color + '12',
                      border: `1px solid ${card.color}22`,
                    }}>{tag}</span>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ─── SOBOLEV TRAINING ─── */}
      <section style={{ padding: '80px 0 120px' }}>
        <div style={{ maxWidth: '900px', margin: '0 auto', padding: '0 40px' }}>
          <div style={{ position: 'relative', background: 'linear-gradient(135deg, rgba(0,229,255,0.03) 0%, rgba(123,92,240,0.06) 100%)', border: '1px solid rgba(0,229,255,0.08)', borderRadius: '28px', padding: '64px', overflow: 'hidden' }}>
            <div style={{ position: 'absolute', top: '-100px', right: '-100px', width: '400px', height: '400px', borderRadius: '50%', background: 'radial-gradient(circle, rgba(123,92,240,0.08) 0%, transparent 70%)', pointerEvents: 'none' }} />
            <div style={{ position: 'relative', zIndex: 1 }}>
              <span style={{ fontSize: '0.7rem', fontWeight: 700, letterSpacing: '0.18em', textTransform: 'uppercase', color: '#7B5CF0' }}>Why Sobolev?</span>
              <h2 style={{ fontSize: '2.2rem', fontWeight: 200, letterSpacing: '-0.02em', color: '#fff', margin: '20px 0 28px', lineHeight: 1.2 }}>
                Most neural networks learn the wrong thing.
              </h2>
              <p style={{ color: '#7A90B0', lineHeight: 1.85, fontSize: '1.05rem', marginBottom: '24px' }}>
                Standard training minimizes prediction error on the output — in this case, gravitational potential values.
                But what actually drives spacecraft trajectories is the <em style={{ color: '#c0d0e8' }}>force</em>, which is the spatial gradient of the potential.
                A network that fits values perfectly but misrepresents their gradient injects unphysical accelerations into every integration step. The trajectory quietly diverges.
              </p>
              <p style={{ color: '#5A6F8D', lineHeight: 1.85, fontSize: '1.05rem' }}>
                Sobolev training solves this by adding the gradient error directly to the loss. The network is penalized for getting forces wrong, not just potentials. The result is a surrogate that earns its place in a flight-grade integrator: physically consistent, numerically trustworthy, and orders of magnitude faster than what it replaces.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* ─── ARCHITECTURE BUILDER ─── */}
      <section style={{ padding: '20px 0 120px' }}>
        <div style={{ maxWidth: '1100px', margin: '0 auto', padding: '0 40px' }}>
          <div style={{ textAlign: 'center', marginBottom: '56px' }}>
            <h2 style={{ fontSize: '2.2rem', fontWeight: 200, letterSpacing: '-0.02em', color: '#fff', marginBottom: '14px' }}>
              Interactive Architecture Builder
            </h2>
            <p style={{ color: '#5A6F8D', fontSize: '1rem', maxWidth: '480px', margin: '0 auto', lineHeight: 1.7 }}>
              Build the surrogate layer by layer. Watch how the signal transforms as it moves through the network.
            </p>
          </div>
          <ArchitectureBuilder />
        </div>
      </section>

    </main>
  );
}
