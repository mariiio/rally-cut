'use client';

import { Box, Typography, Chip } from '@mui/material';

const COURT_W = 8;
const HALF_L = 8;
const SCALE = 100;
const SVG_W = COURT_W * SCALE;
const SVG_H = HALF_L * SCALE;
const PAD = 40;

const COLOR_STOPS = [
  { t: 0.0, r: 59, g: 130, b: 246 },
  { t: 0.5, r: 255, g: 107, b: 74 },
  { t: 1.0, r: 239, g: 68, b: 68 },
];

function interpolateColor(t: number): string {
  const clamped = Math.max(0, Math.min(1, t));
  let i = 0;
  for (; i < COLOR_STOPS.length - 1; i++) {
    if (clamped <= COLOR_STOPS[i + 1].t) break;
  }
  const a = COLOR_STOPS[i];
  const b = COLOR_STOPS[Math.min(i + 1, COLOR_STOPS.length - 1)];
  const range = b.t - a.t || 1;
  const f = (clamped - a.t) / range;
  const r = Math.round(a.r + (b.r - a.r) * f);
  const g = Math.round(a.g + (b.g - a.g) * f);
  const bl = Math.round(a.b + (b.b - a.b) * f);
  return `rgb(${r}, ${g}, ${bl})`;
}

interface CourtHeatmapProps {
  grid: number[][];
  count: number;
  points?: Array<{ courtX: number; courtY: number; type: string; team: string }>;
  title: string;
}

export default function CourtHeatmap({ grid, count, points, title }: CourtHeatmapProps) {
  if (!grid || grid.length === 0) return null;

  const rows = grid.length;
  const cols = grid[0].length;
  const cellW = SVG_W / cols;
  const cellH = SVG_H / rows;
  const maxVal = Math.max(...grid.flat(), 1e-9);

  return (
    <Box sx={{ display: 'inline-flex', flexDirection: 'column', alignItems: 'center' }}>
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, mb: 0.5 }}>
        <Typography variant="caption" sx={{ fontWeight: 600, color: 'text.primary' }}>
          {title}
        </Typography>
        <Chip label={`${count}`} size="small" sx={{ height: 16, fontSize: '0.625rem' }} />
      </Box>

      <svg
        viewBox={`${-PAD} ${-PAD} ${SVG_W + PAD * 2} ${SVG_H + PAD * 2}`}
        width={140}
        height={140}
        style={{ overflow: 'visible' }}
      >
        <rect x={0} y={0} width={SVG_W} height={SVG_H}
          fill="#1A1E28" stroke="rgba(255,255,255,0.3)" strokeWidth={3} rx={4} />

        {grid.map((row, gy) =>
          row.map((val, gx) => {
            if (val <= 0) return null;
            const intensity = val / maxVal;
            return (
              <rect key={`${gy}-${gx}`}
                x={gx * cellW} y={gy * cellH} width={cellW} height={cellH}
                fill={interpolateColor(intensity)} opacity={0.15 + intensity * 0.65} rx={2} />
            );
          }),
        )}

        {/* Net at bottom */}
        <line x1={0} y1={SVG_H} x2={SVG_W} y2={SVG_H}
          stroke="rgba(255,255,255,0.5)" strokeWidth={3} strokeDasharray="12,6" />

        <rect x={0} y={0} width={SVG_W} height={SVG_H}
          fill="none" stroke="rgba(255,255,255,0.3)" strokeWidth={3} rx={4} />

        {points?.map((pt, i) => {
          const cx = (pt.courtX / COURT_W) * SVG_W;
          const cy = (pt.courtY / HALF_L) * SVG_H;
          if (cx < -10 || cx > SVG_W + 10 || cy < -10 || cy > SVG_H + 10) return null;
          return (
            <circle key={i} cx={cx} cy={cy} r={12}
              fill={pt.type === 'serve' ? '#FF6B4A' : '#00D4AA'}
              opacity={0.7} stroke="rgba(255,255,255,0.4)" strokeWidth={1.5} />
          );
        })}

        <text x={SVG_W / 2} y={-12} textAnchor="middle"
          fill="rgba(255,255,255,0.5)" fontSize={28} fontFamily="Inter, sans-serif">
          Opponent
        </text>
        <text x={SVG_W / 2} y={SVG_H + 32} textAnchor="middle"
          fill="rgba(255,255,255,0.5)" fontSize={28} fontFamily="Inter, sans-serif">
          Net
        </text>
      </svg>
    </Box>
  );
}
