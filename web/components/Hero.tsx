import styles from './Hero.module.css';

export default function Hero() {
  return (
    <section className={styles.heroSection}>
      <div className="container" style={{ position: 'relative', height: '100%' }}>
        <div className={styles.contentWrapper} style={{ pointerEvents: 'auto' }}>
          <div className={styles.badge}>LUNARIS ALPHA 0.1.0</div>
          <h1>
            Lunar orbit propagation<br/> 
            <span className={styles.gradientText}>& gravity-modeling</span> framework.
          </h1>
          <p>
            Lunaris bundles spherical-harmonic lunar gravity, configurable physical force models, 
            and <strong>ST-LRPS</strong> &mdash; a neural Sobolev-trained residual potential surrogate 
            for high-performance orbit analysis and Monte Carlo propagation.
          </p>
          <div className={styles.buttonGroup}>
            <button className={styles.primaryButton}>Explore ST-LRPS</button>
            <button className={styles.secondaryButton}>View Benchmarks</button>
          </div>
        </div>
      </div>
    </section>
  );
}
