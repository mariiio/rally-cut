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

interface OriginalQualityBannerProps {
  /** Current video/match being viewed */
  currentMatch: Match | null;
}

/**
 * Banner that warns FREE tier users when their video's original quality
 * is about to be downgraded (3 days after upload).
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
            Upgrade to unlock full quality exports
          </Typography>
          <Tooltip
            title="Pro unlocks original quality exports. After this period, the source file is removed and you'd need to re-upload."
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
          onClick={handleDismiss}
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
