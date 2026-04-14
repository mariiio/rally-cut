'use client';

import { Alert, AlertTitle, Box, Chip, IconButton } from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';
import { useState, useEffect, useMemo } from 'react';
import type { QualityIssue, QualityReport } from '@/types/rally';

const TIER_COLOR: Record<QualityIssue['tier'], string> = {
  block: '#f44336',
  gate: '#FF9800',
  advisory: '#2196F3',
};

const TIER_SEVERITY: Record<QualityIssue['tier'], 'error' | 'warning' | 'info'> = {
  block: 'error',
  gate: 'warning',
  advisory: 'info',
};

interface QualityReportBannerProps {
  report: QualityReport | null | undefined;
  videoId: string;
}

function isDismissed(videoId: string): boolean {
  try {
    return sessionStorage.getItem(`quality-banner-dismissed-${videoId}`) === '1';
  } catch {
    return false;
  }
}

export function QualityReportBanner({ report, videoId }: QualityReportBannerProps) {
  const [dismissed, setDismissed] = useState(() => isDismissed(videoId));
  useEffect(() => {
    setDismissed(isDismissed(videoId));
  }, [videoId]);

  const issues = useMemo(() => report?.issues ?? [], [report]);
  const topTier = useMemo<QualityIssue['tier']>(() => {
    if (issues.some((i) => i.tier === 'block')) return 'block';
    if (issues.some((i) => i.tier === 'gate')) return 'gate';
    return 'advisory';
  }, [issues]);

  if (!issues.length || dismissed) return null;

  const handleDismiss = () => {
    setDismissed(true);
    try {
      sessionStorage.setItem(`quality-banner-dismissed-${videoId}`, '1');
    } catch {
      // Ignore storage errors.
    }
  };

  return (
    <Alert
      severity={TIER_SEVERITY[topTier]}
      sx={{ mb: 1 }}
      action={
        topTier !== 'block' ? (
          <IconButton size="small" onClick={handleDismiss} aria-label="dismiss">
            <CloseIcon fontSize="small" />
          </IconButton>
        ) : undefined
      }
    >
      <Box sx={{ display: 'flex', alignItems: 'baseline', gap: 1, mb: 0.5 }}>
        <AlertTitle sx={{ mb: 0 }}>Video quality</AlertTitle>
        {issues.map((i) => (
          <Chip
            key={i.id}
            label={i.tier}
            size="small"
            sx={{
              bgcolor: TIER_COLOR[i.tier],
              color: 'white',
              fontSize: '0.7rem',
              height: 18,
              textTransform: 'capitalize',
              '& .MuiChip-label': { px: 0.75 },
            }}
          />
        ))}
      </Box>
      {issues.map((i) => (
        <div key={i.id}>{i.message}</div>
      ))}
    </Alert>
  );
}
