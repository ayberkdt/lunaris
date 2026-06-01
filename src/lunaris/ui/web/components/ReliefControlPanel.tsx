'use client';
import { useState } from 'react';

interface ReliefControlPanelProps {
  textureMode: 'aesthetic' | 'gravity';
  setTextureMode: (mode: 'aesthetic' | 'gravity') => void;
  displacementScale: number;
  setDisplacementScale: (scale: number) => void;
}

export function ReliefControlPanel({ textureMode, setTextureMode, displacementScale, setDisplacementScale }: ReliefControlPanelProps) {
  const [sliderExpanded, setSliderExpanded] = useState(false);
  const isExaggerated = displacementScale > 0.05;

  return (
    <div style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100vh', zIndex: 100, pointerEvents: 'none' }}>
      {/* Texture Toggle Button aligned with LUNARIS text */}
      <div style={{ position: 'absolute', top: 'calc(50vh - 22vh)', transform: 'translateY(-50%)', left: '40px', display: 'flex', flexDirection: 'column', gap: '8px', background: 'rgba(5,5,10,0.6)', padding: '8px', borderRadius: '24px', backdropFilter: 'blur(20px)', WebkitBackdropFilter: 'blur(20px)', border: '1px solid rgba(255,255,255,0.05)', pointerEvents: 'auto' }}>
        <button 
          onClick={() => setTextureMode('aesthetic')}
          style={{ 
            padding: '10px 24px', 
            borderRadius: '16px', 
            border: '1px solid', 
            borderColor: textureMode === 'aesthetic' ? 'rgba(255,255,255,0.8)' : 'transparent', 
            background: textureMode === 'aesthetic' ? 'radial-gradient(circle at 20% 30%, rgba(0,0,0,0.15) 0%, transparent 10%), radial-gradient(circle at 70% 60%, rgba(0,0,0,0.1) 0%, transparent 15%), radial-gradient(circle at 40% 80%, rgba(0,0,0,0.08) 0%, transparent 8%), radial-gradient(circle at 50% 50%, #dcdcdc 0%, #888888 100%)' : 'transparent', 
            color: textureMode === 'aesthetic' ? '#000' : '#8A9FBD', 
            cursor: 'pointer', 
            fontWeight: 700, 
            fontSize: '0.85rem', 
            transition: 'all 0.3s',
            boxShadow: textureMode === 'aesthetic' ? 'inset 0 0 10px rgba(0,0,0,0.2), 0 4px 15px rgba(255,255,255,0.2)' : 'none'
          }}
        >
          Aesthetic
        </button>
        <button 
          onClick={() => setTextureMode('gravity')}
          style={{ 
            padding: '10px 24px', 
            borderRadius: '16px', 
            border: '1px solid',
            borderColor: textureMode === 'gravity' ? 'rgba(255,255,255,0.8)' : 'transparent',
            background: textureMode === 'gravity' ? 'repeating-radial-gradient(circle at 30% 70%, rgba(255,255,255,0.1) 0px, rgba(255,255,255,0.1) 1px, transparent 1px, transparent 6px), linear-gradient(135deg, #000080 0%, #0000ff 20%, #00ffff 40%, #00ff00 60%, #ffff00 80%, #ff0000 100%)' : 'transparent', 
            color: textureMode === 'gravity' ? '#fff' : '#8A9FBD', 
            cursor: 'pointer', 
            fontWeight: 700, 
            fontSize: '0.85rem', 
            transition: 'all 0.3s',
            textShadow: textureMode === 'gravity' ? '0 1px 3px rgba(0,0,0,0.8)' : 'none',
            boxShadow: textureMode === 'gravity' ? 'inset 0 0 10px rgba(255,255,255,0.3), 0 4px 15px rgba(0, 229, 255, 0.4)' : 'none'
          }}
        >
          Gravity
        </button>
      </div>

      {/* Premium Collapsible Topographic Relief Control */}
      <div 
        className={`relief-control-panel ${sliderExpanded ? 'expanded' : 'collapsed'}`}
        onMouseEnter={() => setSliderExpanded(true)}
        onMouseLeave={() => setSliderExpanded(false)}
        style={{ 
          position: 'absolute', 
          bottom: '50px', 
          right: '50px', 
          display: 'flex', 
          flexDirection: 'column', 
          background: 'rgba(10, 15, 30, 0.85)', 
          padding: sliderExpanded ? '24px' : '16px', 
          borderRadius: sliderExpanded ? '20px' : '30px', 
          backdropFilter: 'blur(24px)', 
          WebkitBackdropFilter: 'blur(24px)',
          border: '1px solid rgba(255,255,255,0.08)', 
          boxShadow: sliderExpanded ? '0 30px 60px rgba(0,0,0,0.6), inset 0 0 0 1px rgba(255,255,255,0.05)' : '0 10px 30px rgba(0,0,0,0.5)',
          width: sliderExpanded ? '320px' : 'auto',
          minWidth: sliderExpanded ? '320px' : '60px',
          height: sliderExpanded ? 'auto' : '60px',
          alignItems: sliderExpanded ? 'stretch' : 'center',
          justifyContent: sliderExpanded ? 'flex-start' : 'center',
          transition: 'all 0.5s cubic-bezier(0.16, 1, 0.3, 1)',
          cursor: sliderExpanded ? 'default' : 'pointer',
          pointerEvents: 'auto'
        }}
      >
        {!sliderExpanded ? (
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#E0E0FF" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ opacity: 0.8 }}>
            <path d="M21 12H3m0 0l6-6m-6 6l6 6"/>
            <path d="M12 4v16m0 0l6-6m-6 6l-6-6"/>
            <circle cx="12" cy="12" r="10" stroke="#00E5FF" strokeOpacity="0.4" fill="none"/>
          </svg>
        ) : (
          <div style={{ opacity: 1, transition: 'opacity 0.4s ease 0.15s' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
              <label style={{ color: '#ffffff', fontSize: '0.85rem', letterSpacing: '0.15em', fontWeight: 600, textTransform: 'uppercase' }}>
                Surface Relief
              </label>
              <span style={{ color: isExaggerated ? '#FF4444' : '#00E5FF', fontSize: '0.85rem', fontWeight: 600, fontFamily: 'monospace', transition: 'color 0.3s' }}>
                {displacementScale.toFixed(3)}
              </span>
            </div>
            
            <p style={{ color: '#8A9FBD', fontSize: '0.75rem', marginBottom: '20px', lineHeight: 1.5 }}>
              Controls the visual depth exaggeration of the lunar surface displacement map.
            </p>

            <div style={{ position: 'relative', padding: '10px 0' }}>
              <input 
                type="range" 
                min="0" 
                max="0.08" 
                step="0.002" 
                value={displacementScale} 
                onChange={(e) => setDisplacementScale(parseFloat(e.target.value))}
                className={`premium-slider ${isExaggerated ? 'exaggerated' : ''}`}
                style={{
                  appearance: 'none',
                  width: '100%',
                  height: '3px',
                  background: `linear-gradient(90deg, ${isExaggerated ? '#FF4444' : '#00E5FF'} ${(displacementScale / 0.08) * 100}%, rgba(255,255,255,0.05) ${(displacementScale / 0.08) * 100}%)`,
                  borderRadius: '4px',
                  outline: 'none',
                  transition: 'background 0.3s'
                }}
              />
            </div>
            
            <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '12px', color: '#5A6F8D', fontSize: '0.65rem', textTransform: 'uppercase', letterSpacing: '0.15em', fontWeight: 600 }}>
              <span>Flat</span>
              <span>Realistic</span>
              <span style={{ color: '#FF4444', opacity: isExaggerated ? 1 : 0.3, transition: 'opacity 0.3s' }}>Extreme</span>
            </div>

            <div style={{
              marginTop: '16px',
              padding: '10px 14px',
              background: 'rgba(255, 68, 68, 0.08)',
              border: '1px solid rgba(255, 68, 68, 0.2)',
              borderRadius: '10px',
              color: '#FF6B6B',
              fontSize: '0.75rem',
              lineHeight: 1.4,
              display: 'flex',
              alignItems: 'flex-start',
              gap: '10px',
              opacity: isExaggerated ? 1 : 0,
              height: isExaggerated ? 'auto' : '0px',
              overflow: 'hidden',
              transition: 'opacity 0.4s ease, height 0.4s ease, margin 0.4s ease'
            }}>
              <span style={{ fontSize: '1rem' }}>⚠️</span> 
              <span><strong>Warning:</strong> Values above 0.05 create cinematic exaggeration and are physically inaccurate.</span>
            </div>
          </div>
        )}
      </div>
      <style jsx>{`
        .relief-control-panel:hover {
          border-color: rgba(0, 229, 255, 0.3);
          box-shadow: 0 30px 60px rgba(0, 0, 0, 0.8), 0 0 40px rgba(0, 229, 255, 0.1);
        }
        .premium-slider::-webkit-slider-thumb {
          -webkit-appearance: none;
          appearance: none;
          width: 14px;
          height: 14px;
          border-radius: 50%;
          background: #fff;
          cursor: pointer;
          box-shadow: 0 0 15px rgba(0, 229, 255, 0.6), 0 0 30px rgba(0, 229, 255, 0.3);
          border: 2px solid #00E5FF;
          transition: transform 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275), box-shadow 0.3s, border-color 0.3s;
        }
        .premium-slider.exaggerated::-webkit-slider-thumb {
          box-shadow: 0 0 15px rgba(255, 68, 68, 0.6), 0 0 30px rgba(255, 68, 68, 0.3);
          border-color: #FF4444;
        }
        .premium-slider::-webkit-slider-thumb:hover {
          transform: scale(1.4);
        }
      `}</style>
    </div>
  );
}
