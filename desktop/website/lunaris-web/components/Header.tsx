'use client';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import styles from './Header.module.css';

export default function Header() {
  const pathname = usePathname();

  // The /embed route is a headless visual engine for the desktop launcher:
  // no navigation chrome, edge-to-edge canvas only.
  if (pathname?.startsWith('/embed')) {
    return null;
  }

  return (
    <header className={styles.header}>
      <div className={styles.container}>
        <div className={styles.logo}>
          <Link href="/">LUNARIS</Link>
        </div>
        <nav className={styles.nav}>
          <Link href="/" className={pathname === '/' ? styles.active : ''}>
            Home
          </Link>
          <Link href="/st-lrps" className={pathname === '/st-lrps' ? styles.active : ''}>
            ST-LRPS
          </Link>
          <a href="https://github.com/ayberkdt/lunaris" target="_blank" rel="noopener noreferrer">
            GitHub
          </a>
        </nav>
      </div>
    </header>
  );
}
