'use client';

import { useEffect, useState } from 'react';
import { Box, Button, IconButton, Typography } from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';
import ArrowForwardIcon from '@mui/icons-material/ArrowForward';
import Link from 'next/link';
import { listSessions } from '@/services/api';
import { designTokens } from '@/app/designTokens';

const DISMISSED_KEY = 'rallycut_returning_banner_dismissed';

export function ReturningUserBanner() {
  const [visible, setVisible] = useState(false);
  const [sessionCount, setSessionCount] = useState(0);

  useEffect(() => {
    // Check if banner was dismissed this session
    if (sessionStorage.getItem(DISMISSED_KEY)) {
      return;
    }

    let isMounted = true;

    // Fetch sessions and count only REGULAR ones (exclude ALL_VIDEOS)
    listSessions(1, 100)
      .then((response) => {
        if (!isMounted) return;
        const regularSessions = response.data.filter((s) => s.type === 'REGULAR');
        if (regularSessions.length > 0) {
          setSessionCount(regularSessions.length);
          setVisible(true);
        }
      })
      .catch(() => {
        // Silently fail - don't show banner if API fails
      });

    return () => {
      isMounted = false;
    };
  }, []);

  const handleDismiss = () => {
    sessionStorage.setItem(DISMISSED_KEY, 'true');
    setVisible(false);
  };

  if (!visible) {
    return null;
  }

  return (
    <Box
      sx={{
        background: designTokens.gradients.primary,
        py: 1.5,
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
          color: 'white',
          fontWeight: 500,
          textAlign: 'center',
        }}
      >
        Welcome back! You have {sessionCount} session{sessionCount !== 1 ? 's' : ''} waiting.
      </Typography>
      <Button
        component={Link}
        href="/sessions"
        size="small"
        variant="outlined"
        endIcon={<ArrowForwardIcon />}
        sx={{
          color: 'white',
          borderColor: 'white',
          fontWeight: 600,
          '&:hover': {
            borderColor: 'white',
            bgcolor: 'rgba(255, 255, 255, 0.1)',
          },
        }}
      >
        Continue Editing
      </Button>
      <IconButton
        size="small"
        onClick={handleDismiss}
        sx={{
          color: 'white',
          opacity: 0.8,
          '&:hover': {
            opacity: 1,
            bgcolor: 'rgba(255, 255, 255, 0.1)',
          },
        }}
        aria-label="Dismiss banner"
      >
        <CloseIcon fontSize="small" />
      </IconButton>
    </Box>
  );
}
