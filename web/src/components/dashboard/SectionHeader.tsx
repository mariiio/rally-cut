'use client';

import { Box, Chip, Stack, Typography } from '@mui/material';
import { ReactNode } from 'react';

interface SectionHeaderProps {
  icon?: ReactNode;
  title: string;
  count?: number;
  action?: ReactNode;
  color?: 'primary' | 'secondary';
}

export function SectionHeader({ icon, title, count, action, color = 'primary' }: SectionHeaderProps) {
  const iconBg = color === 'primary'
    ? 'rgba(255, 107, 74, 0.15)'
    : 'rgba(0, 212, 170, 0.15)';
  const countBg = color === 'primary'
    ? 'rgba(255, 107, 74, 0.2)'
    : 'rgba(0, 212, 170, 0.2)';
  const countColor = color === 'primary' ? 'primary.main' : 'secondary.main';

  return (
    <Stack
      direction="row"
      alignItems="center"
      justifyContent="space-between"
      sx={{ mb: 2.5 }}
    >
      <Stack direction="row" alignItems="center" spacing={1.5}>
        {icon && (
          <Box
            sx={{
              width: 32,
              height: 32,
              borderRadius: 1,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              bgcolor: iconBg,
              '& svg': {
                fontSize: 18,
                color: color === 'primary' ? 'primary.main' : 'secondary.main',
              },
            }}
          >
            {icon}
          </Box>
        )}
        <Typography
          variant="subtitle1"
          fontWeight={600}
          color="text.primary"
        >
          {title}
        </Typography>
        {count !== undefined && (
          <Chip
            label={count}
            size="small"
            sx={{
              bgcolor: countBg,
              color: countColor,
              fontWeight: 600,
              height: 22,
              minWidth: 28,
              '& .MuiChip-label': {
                px: 1,
              },
            }}
          />
        )}
      </Stack>
      {action && <Box>{action}</Box>}
    </Stack>
  );
}
