'use client';
import { useState, useMemo } from 'react';

export default function ArchitectureBuilder() {
  const [layers, setLayers] = useState<number>(3);
  const [nodes, setNodes] = useState<number>(64);
  const [useSobolev, setUseSobolev] = useState<boolean>(true);
  const [useRadialDecay, setUseRadialDecay] = useState<boolean>(true);
  const [useGeopotential, setUseGeopotential] = useState<boolean>(false);

  const inputFeatures = useGeopotential ? 6 : 3;
  const outputFeatures = 1;

  // Calculate parameters
  const paramsInputToHidden = (inputFeatures * nodes) + nodes; // Weights + Biases
  const paramsHiddenToHidden = (layers > 1) ? (layers - 1) * ((nodes * nodes) + nodes) : 0;
  const paramsHiddenToOutput = (nodes * outputFeatures) + outputFeatures;
  const totalParams = paramsInputToHidden + paramsHiddenToHidden + paramsHiddenToOutput;

  // Helper to format large numbers
  const formatNum = (num: number) => num.toLocaleString('en-US');

  // SVG Drawing constants
  const svgWidth = 800;
  const svgHeight = 400;
  const colSpacing = svgWidth / (layers + 2); // +2 for input and output columns

  // Generate SVG Node Coordinates
  const getColumnNodes = (colIndex: number, totalNodes: number, isInput: boolean = false, isOutput: boolean = false) => {
    const x = colSpacing * 0.5 + colIndex * colSpacing;
    const visualNodesCount = Math.min(totalNodes, 5); // Draw max 5 nodes per column
    
    let yPositions = [];
    const ySpacing = isOutput ? 0 : 50;
    const startY = svgHeight / 2 - ((visualNodesCount - 1) * ySpacing) / 2;

    for (let i = 0; i < visualNodesCount; i++) {
      if (visualNodesCount === 5 && i === 2) {
         // This will be the "..." indicator node
         yPositions.push({ x, y: startY + i * ySpacing, isHiddenIndicator: true, label: '' });
      } else {
        let label = '';
        if (isInput) {
          if (inputFeatures === 3) label = ['X', 'Y', 'Z'][i];
          else label = ['X', 'Y', 'Z', 'r', 'lat', 'lon'][i];
        } else if (isOutput) {
          label = 'ΔU';
        }
        yPositions.push({ x, y: startY + i * ySpacing, isHiddenIndicator: false, label });
      }
    }
    return yPositions;
  };

  const allColumns = [
    getColumnNodes(0, inputFeatures, true, false), // Input
    ...Array.from({ length: layers }).map((_, i) => getColumnNodes(i + 1, nodes, false, false)), // Hidden
    getColumnNodes(layers + 1, outputFeatures, false, true) // Output
  ];

  // Generate paths between all visual nodes of consecutive columns
  const paths = [];
  for (let c = 0; c < allColumns.length - 1; c++) {
    const colA = allColumns[c];
    const colB = allColumns[c+1];
    
    for (const nodeA of colA) {
      if (nodeA.isHiddenIndicator) continue;
      for (const nodeB of colB) {
        if (nodeB.isHiddenIndicator) continue;
        
        const cx1 = nodeA.x + colSpacing / 3;
        const cy1 = nodeA.y;
        const cx2 = nodeB.x - colSpacing / 3;
        const cy2 = nodeB.y;
        
        paths.push(`M ${nodeA.x} ${nodeA.y} C ${cx1} ${cy1}, ${cx2} ${cy2}, ${nodeB.x} ${nodeB.y}`);
      }
    }
  }

  return (
    <div style={{ display: 'flex', gap: '30px', marginTop: '40px', flexWrap: 'wrap', alignItems: 'stretch' }}>
      
      {/* Controls Panel */}
      <div style={{ flex: '1', minWidth: '320px', background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.08)', borderRadius: '24px', padding: '35px', backdropFilter: 'blur(20px)', display: 'flex', flexDirection: 'column' }}>
        <h3 style={{ fontSize: '1.4rem', marginBottom: '30px', color: '#fff', fontWeight: 500, letterSpacing: '0.5px' }}>Hyperparameters</h3>
        
        <div style={{ marginBottom: '35px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '12px' }}>
            <span style={{ color: '#A0B0C0', fontWeight: 500, fontSize: '0.9rem' }}>Hidden Layers</span>
            <span style={{ color: '#00F0FF', fontWeight: 600 }}>{layers}</span>
          </div>
          <div style={{ display: 'flex', gap: '8px' }}>
            {[3, 5, 8, 12].map(num => (
              <button key={num} onClick={() => setLayers(num)} style={{ flex: 1, padding: '8px 0', borderRadius: '8px', border: '1px solid', borderColor: layers === num ? 'rgba(0,240,255,0.5)' : 'rgba(255,255,255,0.1)', background: layers === num ? 'rgba(0,240,255,0.1)' : 'transparent', color: layers === num ? '#00F0FF' : '#8A9FBD', cursor: 'pointer', transition: 'all 0.2s', fontWeight: 600 }}>
                {num}
              </button>
            ))}
          </div>
        </div>

        <div style={{ marginBottom: '40px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '12px' }}>
            <span style={{ color: '#A0B0C0', fontWeight: 500, fontSize: '0.9rem' }}>Nodes per Layer</span>
            <span style={{ color: '#00F0FF', fontWeight: 600 }}>{nodes}</span>
          </div>
          <div style={{ display: 'flex', gap: '8px' }}>
            {[32, 64, 128, 256].map(num => (
              <button key={num} onClick={() => setNodes(num)} style={{ flex: 1, padding: '8px 0', borderRadius: '8px', border: '1px solid', borderColor: nodes === num ? 'rgba(0,240,255,0.5)' : 'rgba(255,255,255,0.1)', background: nodes === num ? 'rgba(0,240,255,0.1)' : 'transparent', color: nodes === num ? '#00F0FF' : '#8A9FBD', cursor: 'pointer', transition: 'all 0.2s', fontWeight: 600 }}>
                {num}
              </button>
            ))}
          </div>
        </div>

        <h3 style={{ fontSize: '1.2rem', marginBottom: '20px', color: '#E0E0FF', fontWeight: 500 }}>Architecture Modules</h3>
        
        <div style={{ display: 'flex', flexDirection: 'column', gap: '15px' }}>
          {[
            { id: 'sobolev', label: 'Sobolev Training', sub: 'Calculates analytical gradients (Autograd)', state: useSobolev, set: setUseSobolev },
            { id: 'radial', label: 'Radial Decay', sub: 'Physical constraint (R₀ / r)ⁿ', state: useRadialDecay, set: setUseRadialDecay },
            { id: 'geo', label: 'Geopotential Features', sub: 'Extends inputs (r, lat, lon)', state: useGeopotential, set: setUseGeopotential }
          ].map(mod => (
            <div key={mod.id} onClick={() => mod.set(!mod.state)} style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '16px', background: mod.state ? 'rgba(0,240,255,0.05)' : 'rgba(0,0,0,0.2)', border: '1px solid', borderColor: mod.state ? 'rgba(0,240,255,0.3)' : 'rgba(255,255,255,0.05)', borderRadius: '12px', cursor: 'pointer', transition: 'all 0.3s' }}>
              <div>
                <div style={{ color: mod.state ? '#fff' : '#8A9FBD', fontWeight: 500, fontSize: '0.95rem', marginBottom: '4px' }}>{mod.label}</div>
                <div style={{ color: '#5A6F8D', fontSize: '0.8rem' }}>{mod.sub}</div>
              </div>
              <div style={{ width: '44px', height: '24px', background: mod.state ? '#00F0FF' : '#223344', borderRadius: '12px', position: 'relative', transition: 'background 0.3s' }}>
                <div style={{ width: '18px', height: '18px', background: mod.state ? '#05050A' : '#8A9FBD', borderRadius: '50%', position: 'absolute', top: '3px', left: mod.state ? '23px' : '3px', transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)' }}></div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Visualizer Panel */}
      <div style={{ flex: '2', minWidth: '400px', display: 'flex', flexDirection: 'column', gap: '20px' }}>
        
        {/* Parameter Dashboard */}
        <div style={{ background: 'linear-gradient(135deg, rgba(0,240,255,0.1), rgba(0,80,255,0.05))', border: '1px solid rgba(0,240,255,0.2)', borderRadius: '24px', padding: '25px 40px', display: 'flex', alignItems: 'center', justifyContent: 'space-between', backdropFilter: 'blur(10px)' }}>
          <div>
            <div style={{ color: '#00F0FF', fontSize: '0.9rem', fontWeight: 600, letterSpacing: '2px', textTransform: 'uppercase', marginBottom: '8px' }}>Total Trainable Parameters</div>
            <div style={{ fontSize: '3rem', fontWeight: 300, color: '#fff', letterSpacing: '1px', lineHeight: 1 }}>{formatNum(totalParams)}</div>
          </div>
          <div style={{ textAlign: 'right', opacity: 0.7 }}>
            <div style={{ color: '#A0B0C0', fontSize: '0.85rem', marginBottom: '5px' }}>Weights: {formatNum(totalParams - (nodes * layers + 1))}</div>
            <div style={{ color: '#A0B0C0', fontSize: '0.85rem' }}>Biases: {formatNum(nodes * layers + 1)}</div>
          </div>
        </div>

        {/* SVG Neural Network */}
        <div style={{ flex: 1, background: 'rgba(5,5,10,0.4)', border: '1px solid rgba(255,255,255,0.05)', borderRadius: '24px', position: 'relative', overflow: 'hidden' }}>
          
          <svg viewBox={`0 0 ${svgWidth} ${svgHeight}`} style={{ width: '100%', height: '100%', display: 'block' }}>
            <defs>
              <linearGradient id="synapseGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stopColor="#445566" stopOpacity="0.1" />
                <stop offset="50%" stopColor="#00F0FF" stopOpacity="0.4" />
                <stop offset="100%" stopColor="#0055FF" stopOpacity="0.1" />
              </linearGradient>
              <filter id="glow">
                <feGaussianBlur stdDeviation="3" result="coloredBlur"/>
                <feMerge>
                  <feMergeNode in="coloredBlur"/>
                  <feMergeNode in="SourceGraphic"/>
                </feMerge>
              </filter>
            </defs>

            {/* Render all paths (synapses) */}
            {paths.map((d, i) => (
              <path 
                key={`path-${i}`} 
                d={d} 
                fill="none" 
                stroke="url(#synapseGrad)" 
                strokeWidth="1.5" 
                className="synapse"
              />
            ))}

            {/* Render all nodes */}
            {allColumns.map((col, colIdx) => {
              const isInput = colIdx === 0;
              const isOutput = colIdx === allColumns.length - 1;
              const isHidden = !isInput && !isOutput;

              return col.map((node, nodeIdx) => {
                if (node.isHiddenIndicator) {
                  return (
                    <g key={`col-${colIdx}-node-${nodeIdx}`}>
                      <circle cx={node.x} cy={node.y - 10} r="2" fill="#8A9FBD" />
                      <circle cx={node.x} cy={node.y} r="2" fill="#8A9FBD" />
                      <circle cx={node.x} cy={node.y + 10} r="2" fill="#8A9FBD" />
                    </g>
                  );
                }

                return (
                  <g key={`col-${colIdx}-node-${nodeIdx}`}>
                    {/* Node glow/backdrop */}
                    <circle 
                      cx={node.x} 
                      cy={node.y} 
                      r={isOutput ? 25 : 15} 
                      fill={isInput ? '#1A2A3A' : isOutput ? '#003366' : '#051525'} 
                      stroke={isInput ? '#445566' : isOutput ? '#00F0FF' : '#00A0FF'} 
                      strokeWidth={isOutput ? 3 : 2}
                      filter={isOutput ? "url(#glow)" : ""}
                    />
                    
                    {/* Node text (if input/output) */}
                    {(isInput || isOutput) && (
                      <text 
                        x={node.x} 
                        y={node.y} 
                        textAnchor="middle" 
                        dominantBaseline="central" 
                        fill="#fff" 
                        fontSize={isOutput ? '16px' : '12px'} 
                        fontWeight="bold"
                      >
                        {node.label}
                      </text>
                    )}
                  </g>
                );
              });
            })}
          </svg>

          {/* Module Overlays (Sobolev / Radial Decay) */}
          <div style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', pointerEvents: 'none' }}>
            
            {useSobolev && (
              <div style={{ position: 'absolute', right: '40px', top: '40px', background: 'rgba(0,240,255,0.1)', border: '1px solid #00F0FF', padding: '10px 20px', borderRadius: '20px', color: '#00F0FF', display: 'flex', alignItems: 'center', gap: '8px', backdropFilter: 'blur(5px)', boxShadow: '0 0 20px rgba(0,240,255,0.2)', animation: 'slideIn 0.3s ease-out' }}>
                <span style={{ fontSize: '1.2rem', fontWeight: 'bold' }}>∇</span>
                <span style={{ fontSize: '0.8rem', fontWeight: 600, letterSpacing: '1px' }}>AUTOGRAD ENGINE</span>
              </div>
            )}

            {useRadialDecay && (
              <div style={{ position: 'absolute', right: '40px', bottom: '40px', background: 'rgba(255,255,255,0.05)', border: '1px dashed rgba(255,255,255,0.3)', padding: '10px 20px', borderRadius: '20px', color: '#fff', display: 'flex', alignItems: 'center', gap: '8px', backdropFilter: 'blur(5px)', animation: 'slideIn 0.3s ease-out' }}>
                <span style={{ fontSize: '0.85rem' }}>Decay: <code style={{ color: '#00F0FF' }}>(R₀ / r)ⁿ</code></span>
              </div>
            )}
          </div>
        </div>

      </div>

      {/* Global styles for SVG animations */}
      <style dangerouslySetInnerHTML={{__html: `
        @keyframes slideIn {
          from { opacity: 0; transform: translateY(10px); }
          to { opacity: 1; transform: translateY(0); }
        }
        .synapse {
          stroke-dasharray: 10;
          animation: flow 20s linear infinite;
        }
        @keyframes flow {
          to { stroke-dashoffset: -100; }
        }
      `}} />
    </div>
  );
}
