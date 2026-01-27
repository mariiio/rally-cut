'use client';

import { Box, Chip, Stack, Typography } from '@mui/material';
import { ReactNode } from 'react';
import { IconBox } from '@/components/IconBox';
import { designTokens } from '@/app/designTokens';

interface SectionHeaderProps {
  icon?: ReactNode;
  title: string;
  count?: number;
  action?: ReactNode;
  color?: 'primary' | 'secondary';
}

export function SectionHeader({ icon, title, count, action, color = 'primary' }: SectionHeaderProps) {
  const countBg = color === 'primary'
    ? designTokens.alpha.primary[20]
    : designTokens.alpha.secondary[20];
  const countColor = color === 'primary' ? 'primary.main' : 'secondary.main';

  return (
    <Stack
      direction="row"
      alignItems="center"
      justifyContent="space-between"
      sx={{ mb: 2.5 }}
    >
      <Stack direction="row" alignItems="center" spacing={1.5}>
        {icon && <IconBox icon={icon} color={color} size="sm" />}
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
