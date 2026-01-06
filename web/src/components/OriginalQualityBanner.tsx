'use client';

import { useEffect, useState, useMemo } from 'react';
import { Box, Button, IconButton, Typography } from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';
import UpgradeIcon from '@mui/icons-material/Upgrade';
import Link from 'next/link';
import { useTierStore } from '@/stores/tierStore';
import type { Match } from '@/types/rally';

const DISMISSED_KEY_PREFIX = 'rallycut_quality_banner_dismissed_';

interface OriginalQualityBannerProps {
  /** Current video/match being viewed */
  currentMatch: Match | null;
}

/**
 * Banner that warns FREE tier users when their video's original quality
 * is about to be downgraded (7 days after upload).
 *
 * Shows when:
 * - User is FREE tier
 * - Video is not yet quality-downgraded
 * - Within 3 days of downgrade
 * - Re-appears on last day regardless of dismissal
 */
export function OriginalQualityBanner({ currentMatch }: OriginalQualityBannerProps) {
  const [isDismissed, setIsDismissed] = useState(false);
  const isPremium = useTierStore((state) => state.isPremium());
  const originalQualityDays = useTierStore((state) => state.limits.originalQualityDays);

  const dismissedKey = currentMatch ? `${DISMISSED_KEY_PREFIX}${currentMatch.id}` : null;

  // Check dismissed state on mount and when match changes
  useEffect(() => {
    if (dismissedKey) {
      setIsDismissed(localStorage.getItem(dismissedKey) === 'true');
    }
  }, [dismissedKey]);

  // Calculate days until quality downgrade
  const daysUntilDowngrade = useMemo(() => {
    if (!currentMatch?.createdAt || !originalQualityDays) return null;

    const createdAt = new Date(currentMatch.createdAt);
    const now = new Date();
    const daysSinceUpload = Math.floor((now.getTime() - createdAt.getTime()) / (24 * 60 * 60 * 1000));
    return Math.max(0, originalQualityDays - daysSinceUpload);
  }, [currentMatch?.createdAt, originalQualityDays]);

  // Determine if banner should show
  const shouldShow = useMemo(() => {
    // Don't show for premium users
    if (isPremium) return false;

    // Don't show if no match or already downgraded
    if (!currentMatch || currentMatch.qualityDowngradedAt) return false;

    // Don't show if no createdAt or we can't calculate days
    if (daysUntilDowngrade === null) return false;

    // Only show when within 3 days
    if (daysUntilDowngrade > 3) return false;

    // On last day (or past), always show regardless of dismissal
    if (daysUntilDowngrade <= 1) return true;

    // Otherwise, respect dismissal
    return !isDismissed;
  }, [isPremium, currentMatch, daysUntilDowngrade, isDismissed]);

  const handleDismiss = () => {
    if (dismissedKey) {
      localStorage.setItem(dismissedKey, 'true');
      setIsDismissed(true);
    }
  };

  if (!shouldShow) {
    return null;
  }

  const urgencyMessage = daysUntilDowngrade === 0
    ? 'Today is the last day'
    : daysUntilDowngrade === 1
    ? '1 day left'
    : `${daysUntilDowngrade} days left`;

  return (
    <Box
      sx={{
        bgcolor: '#FFF3E0',
        borderBottom: '1px solid #FFB74D',
        py: 1,
        px: 2,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        gap: 2,
        flexWrap: 'wrap',
      }}
    >
      <Typography
        variant="body2"
        sx={{
          color: '#E65100',
          fontWeight: 500,
          textAlign: 'center',
        }}
      >
        <strong>{urgencyMessage}</strong> to keep original quality for exports.
        Your video will remain accessible at 720p.
      </Typography>
      <Button
        component={Link}
        href="/pricing"
        size="small"
        variant="contained"
        startIcon={<UpgradeIcon />}
        sx={{
          bgcolor: '#E65100',
          color: 'white',
          fontWeight: 600,
          '&:hover': {
            bgcolor: '#BF360C',
          },
        }}
      >
        Upgrade
      </Button>
      {daysUntilDowngrade !== null && daysUntilDowngrade > 1 && (
        <IconButton
          size="small"
          onClick={handleDismiss}
          sx={{
            color: '#E65100',
            opacity: 0.6,
            '&:hover': {
              opacity: 1,
              bgcolor: 'rgba(230, 81, 0, 0.1)',
            },
          }}
          aria-label="Dismiss banner"
        >
          <CloseIcon fontSize="small" />
        </IconButton>
      )}
    </Box>
  );
}
