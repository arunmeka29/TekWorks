import React, { useState, useEffect, useRef } from 'react';
import { predictCluster } from '../services/api';
import Result from './Result';

export default function PredictionForm() {
  const [height, setHeight] = useState(180);
  const [weight, setWeight] = useState(80);
  const [cluster, setCluster] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [systemLogs, setSystemLogs] = useState(['Initializing Neural Network...', 'Loading model pkl...', 'Ready.']);
  
  const gridRef = useRef(null);

  // Auto-prediction trigger
  useEffect(() => {
    const delayDebounce = setTimeout(() => {
      handlePredict();
    }, 150);
    return () => clearTimeout(delayDebounce);
  }, [height, weight]);

  const handlePredict = async () => {
    setLoading(true);
    setError('');
    
    const logTime = new Date().toLocaleTimeString();
    setSystemLogs(prev => [
      `[${logTime}] Vector adjusted: H=${height}cm, W=${weight}kg`,
      `[${logTime}] Computing cluster centroids...`,
      ...prev.slice(0, 4)
    ]);

    try {
      const data = await predictCluster({
        height: parseFloat(height),
        weight: parseFloat(weight),
      });
      setCluster(data.cluster);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleGridInteraction = (e) => {
    if (!gridRef.current) return;
    const rect = gridRef.current.getBoundingClientRect();
    
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    const clientY = e.touches ? e.touches[0].clientY : e.clientY;
    
    let x = (clientX - rect.left) / rect.width;
    let y = (clientY - rect.top) / rect.height;
    
    x = Math.max(0, Math.min(1, x));
    y = Math.max(0, Math.min(1, y));
    
    const newHeight = Math.round(150 + x * 60);
    const newWeight = Math.round(110 - y * 60);
    
    setHeight(newHeight);
    setWeight(newWeight);
  };

  const handleMouseDown = (e) => {
    handleGridInteraction(e);
    const handleMouseMove = (moveEvent) => handleGridInteraction(moveEvent);
    const handleMouseUp = () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
    };
    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseup', handleMouseUp);
  };

  const handleTouchStart = (e) => {
    handleGridInteraction(e);
    const handleTouchMove = (moveEvent) => handleGridInteraction(moveEvent);
    const handleTouchEnd = () => {
      window.removeEventListener('touchmove', handleTouchMove);
      window.removeEventListener('touchend', handleTouchEnd);
    };
    window.addEventListener('touchmove', handleTouchMove);
    window.addEventListener('touchend', handleTouchEnd);
  };

  const pctX = ((height - 150) / 60) * 100;
  const pctY = ((110 - weight) / 60) * 100;

  return (
    <div className="page-wrapper dashboard-fullscreen">
      <div className="cyber-grid-overlay"></div>
      <div className="glow-sphere glow-sphere-1"></div>
      <div className="glow-sphere glow-sphere-2"></div>
      
      {/* HUD Header */}
      <header className="hud-header">
        <div className="hud-title-block">
          <div className="hud-badge">
            <span className="badge-dot"></span>
            <span>NEURAL MATRIX LAYER</span>
          </div>
          <h1 className="hud-main-title">CLUSTER TARGETER</h1>
        </div>
        <div className="hud-stats-block">
          <span>LATENCY: 14ms</span>
          <span className="divider">|</span>
          <span className={loading ? "status-scanning" : "status-active"}>
            {loading ? "SCANNING..." : "SYSTEM READY"}
          </span>
        </div>
      </header>

      {/* Full-screen Layout */}
      <main className="dashboard-content">
        {/* Left Telemetry Console */}
        <section className="control-panel">
          <div className="corner-bracket top-left"></div>
          <div className="corner-bracket top-right"></div>
          <div className="corner-bracket bottom-left"></div>
          <div className="corner-bracket bottom-right"></div>

          <h2 className="panel-title">TELEMETRY</h2>
          
          <div className="telemetry-cards">
            <div className="telemetry-card">
              <span className="card-lbl">HEIGHT</span>
              <div className="card-val-wrapper">
                <span className="card-num">{height}</span>
                <span className="card-unit">cm</span>
              </div>
              <input 
                type="range" 
                min="150" 
                max="210" 
                value={height}
                onChange={(e) => setHeight(parseInt(e.target.value))}
                className="neon-slider slider-cyan"
              />
            </div>
            <div className="telemetry-card">
              <span className="card-lbl">WEIGHT</span>
              <div className="card-val-wrapper">
                <span className="card-num">{weight}</span>
                <span className="card-unit">kg</span>
              </div>
              <input 
                type="range" 
                min="50" 
                max="110" 
                value={weight}
                onChange={(e) => setWeight(parseInt(e.target.value))}
                className="neon-slider slider-magenta"
              />
            </div>
          </div>

          <h2 className="panel-title">CLASSIFICATION</h2>
          <div className="result-display-area">
            {cluster !== null ? (
              <Result cluster={cluster} />
            ) : (
              <div className="empty-result-card">Awaiting coordinates...</div>
            )}
          </div>

          <h2 className="panel-title font-code">TERMINAL LOGS</h2>
          <div className="sys-logs-container">
            <div className="logs-list">
              {systemLogs.map((log, index) => (
                <div key={index} className="log-item">{log}</div>
              ))}
            </div>
          </div>
        </section>

        {/* Right Radar Plot Panel */}
        <section className="radar-panel">
          <div className="corner-bracket top-left"></div>
          <div className="corner-bracket top-right"></div>
          <div className="corner-bracket bottom-left"></div>
          <div className="corner-bracket bottom-right"></div>

          <div className="radar-grid-wrapper">
            <div className="axis-label-y">WEIGHT (kg)</div>
            
            <div className="grid-outer">
              <div className="axis-y-ticks">
                <span>110</span>
                <span>90</span>
                <span>70</span>
                <span>50</span>
              </div>
              
              <div 
                className="interactive-grid" 
                ref={gridRef}
                onMouseDown={handleMouseDown}
                onTouchStart={handleTouchStart}
              >
                <div className="grid-scanlines"></div>
                <div className="grid-radar-sweep"></div>
                
                {/* Concentric radar ranges */}
                <div className="radar-ring ring-1"></div>
                <div className="radar-ring ring-2"></div>
                <div className="radar-ring ring-3"></div>
                
                <div className="grid-crosshairs-bg"></div>
                
                {/* Targeter Crosshair */}
                <div 
                  className="target-crosshair"
                  style={{ left: `${pctX}%`, top: `${pctY}%` }}
                >
                  <div className="target-aim-box"></div>
                  <div className="pulse-ring"></div>
                  <div className="coord-readout">
                    H: {height} | W: {weight}
                  </div>
                </div>
              </div>
            </div>
            
            <div className="axis-label-x">HEIGHT (cm)</div>
            <div className="axis-x-ticks">
              <span>150</span>
              <span>170</span>
              <span>190</span>
              <span>210</span>
            </div>
          </div>
        </section>
      </main>

      {/* HUD Footer */}
      <footer className="hud-footer">
        <span>LOC: CLUSTER_STATION_B</span>
        <span>SYS_STATUS: ACTIVE</span>
      </footer>
    </div>
  );
}
