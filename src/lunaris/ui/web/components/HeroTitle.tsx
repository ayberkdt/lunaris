export function HeroTitle() {
  return (
    <section style={{ position: 'absolute', top: '0', left: 0, width: '100%', height: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 0, pointerEvents: 'none' }}>
      <div style={{ textAlign: 'center', transform: 'translateY(-22vh)' }}>
        <h1 style={{ fontSize: '12vw', letterSpacing: '0.2em', margin: 0, color: '#ffffff', opacity: 0.95, fontWeight: 300, lineHeight: 1, textShadow: '0 0 60px rgba(255,255,255,0.2)' }}>
          LUNARIS
        </h1>
        <p style={{ fontSize: '1.2rem', color: '#A0B0C0', letterSpacing: '0.4em', marginTop: '2rem', opacity: 0.8, fontWeight: 400 }}>
          ORBIT PROPAGATION FRAMEWORK
        </p>
      </div>
    </section>
  );
}
