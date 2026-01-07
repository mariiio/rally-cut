'use client';

import { useEffect, useState, useMemo } from 'react';
import { Box, Button, IconButton, Typography, Chip, Tooltip } from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';
import DiamondIcon from '@mui/icons-material/Diamond';
import AccessTimeIcon from '@mui/icons-material/AccessTime';
import InfoOutlinedIcon from '@mui/icons-material/InfoOutlined';
import Link from 'next/link';
import { useTierStore } from '@/stores/tierStore';
import { designTokens } from '@/app/theme';
import type { Match } from '@/types/rally';

const DISMISSED_KEY_PREFIX = 'rallycut_quality_banner_dismissed_';
const DOWNGRADED_DISMISSED_KEY_PREFIX = 'rallycut_downgraded_banner_dismissed_';

interface OriginalQualityBannerProps {
  /** Current video/match being viewed */
  currentMatch: Match | null;
}

type BannerState = 'none' | 'countdown' | 'downgraded';

/**
 * Banner for FREE tier users about video quality:
 *
 * 1. Countdown state (within 3 days of upload, not yet downgraded):
 *    - Shows time remaining for full quality exports
 *    - "Full quality exports available for X days"
 *
 * 2. Downgraded state (after 3 days OR qualityDowngradedAt set):
 *    - Subtle info message
 *    - "Full quality exports no longer available. Upgrade and re-upload to restore."
 *    - Permanently dismissible per video
 */
export function OriginalQualityBanner({ currentMatch }: OriginalQualityBannerProps) {
  const [isCountdownDismissed, setIsCountdownDismissed] = useState(false);
  const [isDowngradedDismissed, setIsDowngradedDismissed] = useState(false);
  const isPremium = useTierStore((state) => state.isPremium());
  const originalQualityDays = useTierStore((state) => state.limits.originalQualityDays);

  const countdownDismissedKey = currentMatch ? `${DISMISSED_KEY_PREFIX}${currentMatch.id}` : null;
  const downgradedDismissedKey = currentMatch ? `${DOWNGRADED_DISMISSED_KEY_PREFIX}${currentMatch.id}` : null;

  // Check dismissed state on mount and when match changes
  useEffect(() => {
    if (countdownDismissedKey) {
      setIsCountdownDismissed(localStorage.getItem(countdownDismissedKey) === 'true');
    }
    if (downgradedDismissedKey) {
      setIsDowngradedDismissed(localStorage.getItem(downgradedDismissedKey) === 'true');
    }
  }, [countdownDismissedKey, downgradedDismissedKey]);

  // Calculate days until quality downgrade
  const daysUntilDowngrade = useMemo(() => {
    if (!currentMatch?.createdAt || !originalQualityDays) return null;

    const createdAt = new Date(currentMatch.createdAt);
    const now = new Date();
    const daysSinceUpload = Math.floor((now.getTime() - createdAt.getTime()) / (24 * 60 * 60 * 1000));
    return originalQualityDays - daysSinceUpload;
  }, [currentMatch?.createdAt, originalQualityDays]);

  // Determine banner state
  const bannerState: BannerState = useMemo(() => {
    // Don't show for premium users
    if (isPremium) return 'none';

    // Don't show if no match
    if (!currentMatch) return 'none';

    // Already downgraded - show downgraded banner (if not dismissed)
    if (currentMatch.qualityDowngradedAt) {
      return isDowngradedDismissed ? 'none' : 'downgraded';
    }

    // No createdAt - can't calculate, don't show
    if (daysUntilDowngrade === null) return 'none';

    // Past grace period but not yet marked as downgraded in DB
    // (cleanup job hasn't run yet)
    if (daysUntilDowngrade < 0) {
      return isDowngradedDismissed ? 'none' : 'downgraded';
    }

    // Within 2 days of grace period ending - show countdown
    if (daysUntilDowngrade <= 2) {
      // On last day (or past), always show regardless of dismissal
      if (daysUntilDowngrade <= 1) return 'countdown';
      // Otherwise, respect dismissal
      return isCountdownDismissed ? 'none' : 'countdown';
    }

    return 'none';
  }, [isPremium, currentMatch, daysUntilDowngrade, isCountdownDismissed, isDowngradedDismissed]);

  const handleDismissCountdown = () => {
    if (countdownDismissedKey) {
      localStorage.setItem(countdownDismissedKey, 'true');
      setIsCountdownDismissed(true);
    }
  };

  const handleDismissDowngraded = () => {
    if (downgradedDismissedKey) {
      localStorage.setItem(downgradedDismissedKey, 'true');
      setIsDowngradedDismissed(true);
    }
  };

  if (bannerState === 'none') {
    return null;
  }

  // Downgraded state - subtle info banner
  if (bannerState === 'downgraded') {
    return (
      <Box
        sx={{
          background: 'rgba(255, 255, 255, 0.03)',
          borderBottom: '1px solid rgba(255, 255, 255, 0.08)',
          py: 0.5,
          px: 2,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          gap: 1.5,
        }}
      >
        <Typography
          variant="body2"
          sx={{
            color: 'text.disabled',
            fontSize: '0.75rem',
          }}
        >
          Full quality exports no longer available.{' '}
          <Typography
            component={Link}
            href="/pricing"
            sx={{
              color: designTokens.colors.tertiary.main,
              fontSize: 'inherit',
              textDecoration: 'none',
              '&:hover': {
                textDecoration: 'underline',
              },
            }}
          >
            Upgrade
          </Typography>
          {' '}and re-upload to restore.
        </Typography>
        <IconButton
          size="small"
          onClick={handleDismissDowngraded}
          sx={{
            color: 'text.disabled',
            p: 0.25,
            '&:hover': {
              color: 'text.secondary',
              bgcolor: 'rgba(255, 255, 255, 0.05)',
            },
          }}
          aria-label="Dismiss"
        >
          <CloseIcon sx={{ fontSize: 14 }} />
        </IconButton>
      </Box>
    );
  }

  // Countdown state
  const isUrgent = daysUntilDowngrade !== null && daysUntilDowngrade <= 1;
  const timeLabel = daysUntilDowngrade === 0
    ? 'Last day'
    : daysUntilDowngrade === 1
    ? '1 day'
    : `${daysUntilDowngrade} days`;

  return (
    <Box
      sx={{
        background: isUrgent
          ? 'linear-gradient(90deg, rgba(255, 107, 74, 0.15) 0%, rgba(255, 209, 102, 0.1) 100%)'
          : 'linear-gradient(90deg, rgba(255, 209, 102, 0.1) 0%, rgba(0, 212, 170, 0.05) 100%)',
        borderBottom: '1px solid',
        borderColor: isUrgent ? 'rgba(255, 107, 74, 0.3)' : 'rgba(255, 209, 102, 0.2)',
        py: 0.75,
        px: 2,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        gap: 1.5,
        flexWrap: 'wrap',
      }}
    >
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
        <Chip
          icon={<AccessTimeIcon sx={{ fontSize: 14 }} />}
          label={timeLabel}
          size="small"
          sx={{
            bgcolor: isUrgent ? 'rgba(255, 107, 74, 0.2)' : 'rgba(255, 209, 102, 0.2)',
            color: isUrgent ? designTokens.colors.tertiary.dark : designTokens.colors.tertiary.main,
            fontWeight: 600,
            fontSize: '0.7rem',
            height: 22,
            '& .MuiChip-icon': {
              color: 'inherit',
            },
          }}
        />
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
          <Typography
            variant="body2"
            sx={{
              color: 'text.secondary',
              fontSize: '0.8rem',
            }}
          >
            Full quality exports available
          </Typography>
          <Tooltip
            title="FREE tier includes full quality exports with watermark for the first 3 days. Upgrade to Pro for permanent full quality without watermark."
            arrow
            placement="top"
          >
            <InfoOutlinedIcon
              sx={{
                fontSize: 14,
                color: 'text.disabled',
                cursor: 'help',
                '&:hover': {
                  color: 'text.secondary',
                },
              }}
            />
          </Tooltip>
        </Box>
      </Box>

      <Button
        component={Link}
        href="/pricing"
        size="small"
        startIcon={<DiamondIcon sx={{ fontSize: 14 }} />}
        sx={{
          background: designTokens.gradients.tertiary,
          color: '#1a1a1a',
          fontWeight: 600,
          fontSize: '0.75rem',
          px: 1.5,
          py: 0.5,
          minHeight: 28,
          borderRadius: 1,
          textTransform: 'none',
          '&:hover': {
            background: designTokens.gradients.primary,
            color: 'white',
          },
        }}
      >
        Upgrade to Pro
      </Button>

      {!isUrgent && (
        <IconButton
          size="small"
          onClick={handleDismissCountdown}
          sx={{
            color: 'text.disabled',
            p: 0.5,
            '&:hover': {
              color: 'text.secondary',
              bgcolor: 'rgba(255, 255, 255, 0.05)',
            },
          }}
          aria-label="Dismiss"
        >
          <CloseIcon sx={{ fontSize: 16 }} />
        </IconButton>
      )}
    </Box>
  );
}
