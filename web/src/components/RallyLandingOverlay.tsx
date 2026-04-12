'use client';

import { Box } from '@mui/material';
import type { MatchStats } from '@/services/api';

const COURT_W = 8;
const COURT_L = 16;
const SVG_W = 120;
const SVG_H = 240;

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
        opacity: 0.85,
      }}
    >
      <svg viewBox={`0 0 ${COURT_W * 100} ${COURT_L * 100}`} width={SVG_W} height={SVG_H}>
        <rect x={0} y={0} width={COURT_W * 100} height={COURT_L * 100}
          fill="rgba(26,30,40,0.8)" stroke="rgba(255,255,255,0.3)" strokeWidth={4} rx={4} />

        <line x1={0} y1={COURT_L * 50} x2={COURT_W * 100} y2={COURT_L * 50}
          stroke="rgba(255,255,255,0.4)" strokeWidth={3} strokeDasharray="8,4" />

        {perRally.points.map((pt, i) => {
          const cx = (pt.courtX / COURT_W) * COURT_W * 100;
          const cy = (pt.courtY / COURT_L) * COURT_L * 100;
          if (cx < -20 || cx > COURT_W * 100 + 20 || cy < -20 || cy > COURT_L * 100 + 20) return null;
          const isServe = pt.type === 'serve';
          const isTeamA = pt.team === 'A';
          return (
            <circle key={i} cx={cx} cy={cy}
              r={isServe ? 16 : 14}
              fill={isServe ? '#FF6B4A' : '#00D4AA'}
              opacity={0.8}
              stroke={isTeamA ? 'rgba(255,255,255,0.6)' : 'rgba(255,255,255,0.3)'}
              strokeWidth={isTeamA ? 3 : 2}
              strokeDasharray={isTeamA ? undefined : '4,3'}
            />
          );
        })}
      </svg>
    </Box>
  );
}
