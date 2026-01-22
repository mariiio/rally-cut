'use client';

import { Box, Stack, Typography } from '@mui/material';
import { ReactNode } from 'react';
import { designTokens } from '@/app/theme';

interface PageHeaderProps {
  icon: ReactNode;
  title: string;
  subtitle?: string;
  action?: ReactNode;
  gradient?: boolean;
}

export function PageHeader({ icon, title, subtitle, action, gradient = true }: PageHeaderProps) {
  return (
    <Box
      component="header"
      sx={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        mb: 4,
        py: 1,
        position: 'relative',
      }}
    >
      <Stack direction="row" alignItems="center" spacing={2}>
        <Box
          sx={{
            width: 48,
            height: 48,
            borderRadius: 2,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            background: 'rgba(255, 107, 74, 0.15)',
            border: '1px solid rgba(255, 107, 74, 0.2)',
            '& svg': {
              fontSize: 28,
              color: 'primary.main',
              filter: 'drop-shadow(0 2px 8px rgba(255, 107, 74, 0.4))',
            },
          }}
        >
          {icon}
        </Box>
        <Box>
          <Typography
            variant="h4"
            sx={{
              fontWeight: 700,
              letterSpacing: '-0.02em',
              ...(gradient && {
                background: designTokens.gradients.primary,
                backgroundClip: 'text',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
              }),
            }}
          >
            {title}
          </Typography>
          {subtitle && (
            <Typography
              variant="body2"
              color="text.secondary"
              sx={{ letterSpacing: '0.01em', mt: 0.25 }}
            >
              {subtitle}
            </Typography>
          )}
        </Box>
      </Stack>
      {action && <Box>{action}</Box>}
    </Box>
  );
}
