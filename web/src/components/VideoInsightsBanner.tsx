'use client';

import { useState, useMemo, useEffect } from 'react';
import { Alert, AlertTitle, IconButton } from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';
import type { Match } from '@/types/rally';

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

export function VideoInsightsBanner({ currentMatch }: VideoInsightsBannerProps) {
  const videoId = currentMatch?.id ?? null;

  // Re-evaluate dismissed state when video changes
  const [dismissed, setDismissed] = useState(() => videoId ? isDismissed(videoId) : false);
  useEffect(() => {
    setDismissed(videoId ? isDismissed(videoId) : false);
  }, [videoId]);

  const messages = useMemo(() => {
    const chars = currentMatch?.characteristicsJson;
    if (!chars) return [];

    const msgs: string[] = [];
    if (chars.cameraDistance?.category === 'far') {
      msgs.push('Far camera angle — player tracking may be less accurate for distant players');
    }
    if (chars.cameraDistance?.category === 'close' || chars.cameraDistance?.category === 'medium') {
      // Medium/close cameras have worse ball tracking (38.7% vs 92.1% for far)
      msgs.push('Close camera angle — ball tracking may be less accurate when the ball leaves the frame');
    }
    if (chars.sceneComplexity?.category === 'complex') {
      msgs.push('Crowded scene detected — some spectators may appear in tracking results');
    }
    return msgs;
  }, [currentMatch?.characteristicsJson]);

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
      <AlertTitle>Video Insights</AlertTitle>
      {messages.map((msg, i) => (
        <div key={i}>{msg}</div>
      ))}
    </Alert>
  );
}
