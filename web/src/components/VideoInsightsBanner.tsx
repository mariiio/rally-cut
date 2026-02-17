'use client';

import { useState, useMemo, useEffect } from 'react';
import { Alert, AlertTitle, IconButton, Box, Chip } from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';
import type { Match } from '@/types/rally';
import { usePlayerTrackingStore } from '@/stores/playerTrackingStore';
import type { QualityReport } from '@/services/api';

interface VideoInsightsBannerProps {
  currentMatch: Match | null;
}

function isDismissed(videoId: string): boolean {
  try {
    return localStorage.getItem(`insights-dismissed-${videoId}`) === '1';
  } catch {
    return false;
  }
}

function scoreColor(score: number): string {
  if (score >= 0.7) return '#4CAF50';
  if (score >= 0.4) return '#FF9800';
  return '#f44336';
}

export function VideoInsightsBanner({ currentMatch }: VideoInsightsBannerProps) {
  const videoId = currentMatch?.id ?? null;

  // Re-evaluate dismissed state when video changes
  const [dismissed, setDismissed] = useState(() => videoId ? isDismissed(videoId) : false);
  useEffect(() => {
    setDismissed(videoId ? isDismissed(videoId) : false);
  }, [videoId]);

  const playerTracks = usePlayerTrackingStore((s) => s.playerTracks);
  const calibrations = usePlayerTrackingStore((s) => s.calibrations);

  const { messages, qualityScore } = useMemo(() => {
    const msgs: string[] = [];
    let worstScore: number | null = null;

    // Video characteristics messages
    const chars = currentMatch?.characteristicsJson;
    if (chars) {
      if (chars.cameraDistance?.category === 'far') {
        msgs.push('Far camera angle — player tracking may be less accurate for distant players');
      }
      if (chars.cameraDistance?.category === 'close' || chars.cameraDistance?.category === 'medium') {
        msgs.push('Close camera angle — ball tracking may be less accurate when the ball leaves the frame');
      }
      if (chars.sceneComplexity?.category === 'complex') {
        msgs.push('Crowded scene detected — some spectators may appear in tracking results');
      }
    }

    // Quality report messages from tracked rallies
    if (currentMatch && videoId) {
      const hasCalibration = !!calibrations[videoId];
      const seen = new Set<string>();

      for (const rally of currentMatch.rallies) {
        const backendId = rally._backendId;
        if (!backendId) continue;
        const qr: QualityReport | undefined = playerTracks[backendId]?.tracksJson?.qualityReport;
        if (!qr) continue;

        // Track worst score
        if (worstScore === null || qr.trackabilityScore < worstScore) {
          worstScore = qr.trackabilityScore;
        }

        // Deduplicate suggestions across rallies
        for (const s of qr.suggestions) {
          // Skip calibration suggestion if already calibrated
          if (s.includes('Label court corners') && hasCalibration) continue;
          if (!seen.has(s)) {
            seen.add(s);
            msgs.push(s);
          }
        }
      }
    }

    return { messages: msgs, qualityScore: worstScore };
  }, [currentMatch, videoId, playerTracks, calibrations]);

  if (!currentMatch || dismissed || messages.length === 0) return null;

  const handleDismiss = () => {
    setDismissed(true);
    try {
      localStorage.setItem(`insights-dismissed-${currentMatch.id}`, '1');
    } catch {
      // Ignore storage errors
    }
  };

  return (
    <Alert
      severity="info"
      sx={{ mb: 1 }}
      action={
        <IconButton size="small" onClick={handleDismiss} aria-label="dismiss">
          <CloseIcon fontSize="small" />
        </IconButton>
      }
    >
      <Box sx={{ display: 'flex', alignItems: 'baseline', gap: 1 }}>
        <AlertTitle sx={{ mb: 0 }}>Video Insights</AlertTitle>
        {qualityScore !== null && (
          <Chip
            label={`Tracking: ${Math.round(qualityScore * 100)}%`}
            size="small"
            sx={{
              bgcolor: scoreColor(qualityScore),
              color: 'white',
              fontSize: '0.7rem',
              height: 20,
              fontWeight: 'bold',
              '& .MuiChip-label': { px: 0.75 },
            }}
          />
        )}
      </Box>
      {messages.map((msg, i) => (
        <div key={i}>{msg}</div>
      ))}
    </Alert>
  );
}
