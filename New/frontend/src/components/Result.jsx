import React from 'react';

export default function Result({ cluster }) {
  const clusterColors = {
    0: 'cyan',
    1: 'magenta',
    2: 'lime',
    3: 'yellow'
  };

  const activeColor = clusterColors[cluster] || 'cyan';

  return (
    <div className={`result-card glow-${activeColor}`}>
      <span className="result-label">TARGET SEGMENT</span>
      <div className="cluster-value-wrapper">
        <span className="cluster-text">CLUSTER</span>
        <span className="cluster-id">{cluster}</span>
      </div>
      <div className="status-indicator">
        <span className="pulse-dot"></span>
        <span className="status-text">CLASSIFIED SUCCESSFUL</span>
      </div>
    </div>
  );
}

