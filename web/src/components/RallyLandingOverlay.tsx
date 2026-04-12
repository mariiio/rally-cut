'use client';

import { Box } from '@mui/material';
import type { MatchStats } from '@/services/api';

const COURT_W = 8;
const COURT_L = 16;
const S = 100;
const SVG_W = COURT_W * S;
const SVG_H = COURT_L * S;

interface RallyLandingOverlayProps {
  rallyId: string;
  matchStats: MatchStats;
}

export function RallyLandingOverlay({ rallyId, matchStats }: RallyLandingOverlayProps) {
  const perRally = matchStats.landingHeatmaps?.perRally?.[rallyId];
  if (!perRally || perRally.points.length === 0) return null;

  return (
    <Box
      sx={{
        position: 'absolute',
        bottom: 48,
        right: 8,
        zIndex: 10,
        pointerEvents: 'none',
        opacity: 0.95,
      }}
    >
      <svg viewBox={`0 0 ${SVG_W} ${SVG_H}`} width={110} height={220}>
        {/* Far half — Team B */}
        <rect x={0} y={0} width={SVG_W} height={SVG_H / 2}
          fill="rgba(0,212,170,0.12)" rx={6} />
        {/* Near half — Team A */}
        <rect x={0} y={SVG_H / 2} width={SVG_W} height={SVG_H / 2}
          fill="rgba(255,107,74,0.12)" rx={6} />

        {/* Court outline */}
        <rect x={0} y={0} width={SVG_W} height={SVG_H}
          fill="none" stroke="rgba(255,255,255,0.35)" strokeWidth={5} rx={6} />

        {/* Net */}
        <line x1={0} y1={SVG_H / 2} x2={SVG_W} y2={SVG_H / 2}
          stroke="rgba(255,255,255,0.6)" strokeWidth={5} />

        {/* Team indicators */}
        <text x={20} y={42} fill="rgba(0,212,170,0.7)" fontSize={30}
          fontFamily="Inter, sans-serif" fontWeight={700}>B</text>
        <text x={20} y={SVG_H - 16} fill="rgba(255,107,74,0.7)" fontSize={30}
          fontFamily="Inter, sans-serif" fontWeight={700}>A</text>

        {/* Landing points */}
        {perRally.points.map((pt, i) => {
          const cx = pt.courtX * S;
          const cy = pt.courtY * S;
          if (cx < -20 || cx > SVG_W + 20 || cy < -20 || cy > SVG_H + 20) return null;
          const isServe = pt.type === 'serve';
          return (
            <g key={i}>
              <circle cx={cx} cy={cy} r={22}
                fill={isServe ? '#FF6B4A' : '#00D4AA'}
                opacity={0.95}
                stroke="#fff" strokeWidth={3} />
              <text x={cx} y={cy + 8} textAnchor="middle"
                fill="#fff" fontSize={18} fontWeight={700} fontFamily="Inter, sans-serif">
                {isServe ? 'S' : 'A'}
              </text>
            </g>
          );
        })}
      </svg>
    </Box>
  );
}
