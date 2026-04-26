export function MoodMeter({
  dataPoints,
}: {
  dataPoints: Array<{
    valence: number;
    arousal: number;
    minV: number;
    maxV: number;
    minA: number;
    maxA: number;
    label: string;
    color: string;
  }>;
}) {
  return (
    <div style={{ width: '100%', maxWidth: '300px', margin: '0 auto', position: 'relative' }}>
      <svg viewBox="0 0 200 200" width="100%" height="100%" style={{ background: 'rgba(0,0,0,0.2)', borderRadius: '12px', border: '1px solid rgba(255,255,255,0.05)' }}>
        {/* Quadrants */}
        {/* Top-Left: High Arousal, Low Valence (Stressed/Angry) */}
        <rect x="0" y="0" width="100" height="100" fill="rgba(239, 68, 68, 0.15)" />
        {/* Top-Right: High Arousal, High Valence (Excited/Happy) */}
        <rect x="100" y="0" width="100" height="100" fill="rgba(234, 179, 8, 0.15)" />
        {/* Bottom-Left: Low Arousal, Low Valence (Sad/Bored) */}
        <rect x="0" y="100" width="100" height="100" fill="rgba(59, 130, 246, 0.15)" />
        {/* Bottom-Right: Low Arousal, High Valence (Calm/Relaxed) */}
        <rect x="100" y="100" width="100" height="100" fill="rgba(16, 185, 129, 0.15)" />

        {/* Axes */}
        <line x1="100" y1="0" x2="100" y2="200" stroke="rgba(255,255,255,0.2)" strokeWidth="1" strokeDasharray="4 4" />
        <line x1="0" y1="100" x2="200" y2="100" stroke="rgba(255,255,255,0.2)" strokeWidth="1" strokeDasharray="4 4" />

        {/* Labels */}
        <text x="100" y="12" fill="rgba(255,255,255,0.5)" fontSize="8" textAnchor="middle">High Arousal</text>
        <text x="100" y="196" fill="rgba(255,255,255,0.5)" fontSize="8" textAnchor="middle">Low Arousal</text>
        <text x="5" y="103" fill="rgba(255,255,255,0.5)" fontSize="8" textAnchor="start">Negative</text>
        <text x="195" y="103" fill="rgba(255,255,255,0.5)" fontSize="8" textAnchor="end">Positive</text>

        {/* Data Points */}
        {dataPoints.map((pt, i) => {
          const vNorm = (pt.valence - pt.minV) / (pt.maxV - pt.minV);
          const aNorm = (pt.arousal - pt.minA) / (pt.maxA - pt.minA);
          
          const cx = Math.max(5, Math.min(195, vNorm * 200));
          const cy = Math.max(5, Math.min(195, (1 - aNorm) * 200));

          return (
            <g key={i}>
              <circle cx={cx} cy={cy} r="6" fill={pt.color} />
              <circle cx={cx} cy={cy} r="12" fill={pt.color} opacity="0.3">
                <animate attributeName="r" values="6;16;6" dur="2s" repeatCount="indefinite" />
                <animate attributeName="opacity" values="0.6;0;0.6" dur="2s" repeatCount="indefinite" />
              </circle>
              {/* Tooltip emulation simple text box */}
              <rect x={cx < 100 ? cx + 10 : cx - 60} y={cy < 20 ? cy + 10 : cy - 20} width="50" height="14" fill="rgba(0,0,0,0.8)" rx="4" />
              <text x={cx < 100 ? cx + 35 : cx - 35} y={cy < 20 ? cy + 19 : cy - 11} fill="#fff" fontSize="8" textAnchor="middle">{pt.label}</text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}
