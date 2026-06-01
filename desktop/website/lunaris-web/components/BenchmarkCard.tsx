import React from 'react';

interface BenchmarkProps {
  title: string;
  value: string;
  description: string;
  delay?: string;
}

export default function BenchmarkCard({ title, value, description, delay = '' }: BenchmarkProps) {
  return (
    <div className={`glass-panel slide-up ${delay}`} style={{ pointerEvents: 'auto' }}>
      <h3 style={{ fontSize: '1.1rem', color: 'var(--text-secondary)', marginBottom: '12px' }}>{title}</h3>
      <div className="text-gradient" style={{ fontSize: '2.5rem', fontWeight: 700, fontFamily: 'var(--font-heading)', marginBottom: '8px' }}>
        {value}
      </div>
      <p style={{ fontSize: '0.9rem', color: 'var(--text-secondary)' }}>{description}</p>
    </div>
  );
}
