import type { Metadata } from 'next';
import { Inter, Outfit } from 'next/font/google';
import './globals.css';

const inter = Inter({ subsets: ['latin'], variable: '--font-inter' });
const outfit = Outfit({ subsets: ['latin'], variable: '--font-outfit' });

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
      <body className={`${inter.variable} ${outfit.variable}`}>
        <Header />
        <div className="bg-glow"></div>
        {children}
      </body>
    </html>
  );
}
