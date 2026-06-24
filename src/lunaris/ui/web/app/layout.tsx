import type { Metadata } from 'next';
import './globals.css';

import Header from '@/components/Header';

export const metadata: Metadata = {
  title: 'Lunaris - Lunar Orbit Propagation',
  description: 'A framework for lunar-orbit propagation and gravity modeling.',
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>
        <Header />
        <div className="bg-glow"></div>
        {children}
      </body>
    </html>
  );
}
