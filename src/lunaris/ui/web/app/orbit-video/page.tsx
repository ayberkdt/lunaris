'use client';

import dynamic from 'next/dynamic';

const CinematicOrbitScene = dynamic(() => import('@/components/CinematicOrbitScene'), {
  ssr: false,
});

export default function OrbitVideoPage() {
  return <CinematicOrbitScene />;
}
