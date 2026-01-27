import { Chip, type SxProps, type Theme } from '@mui/material';
import DiamondIcon from '@mui/icons-material/Diamond';
import { designTokens } from '@/app/designTokens';

interface TierBadgeProps {
  tier: 'PRO' | 'ELITE';
  size?: 'small' | 'medium';
  sx?: SxProps<Theme>;
}

export function TierBadge({ tier, size = 'small', sx }: TierBadgeProps) {
  const iconSize = size === 'small' ? 12 : 14;
  const height = size === 'small' ? 22 : 24;
  const fontSize = size === 'small' ? '0.7rem' : '0.75rem';

  return (
    <Chip
      icon={<DiamondIcon sx={{ fontSize: iconSize }} />}
      label={tier === 'ELITE' ? 'Elite' : 'Pro'}
      size="small"
      variant="outlined"
      sx={{
        height,
        borderColor: designTokens.alpha.tertiary[50],
        color: designTokens.colors.tertiary.main,
        fontWeight: 600,
        fontSize,
        '& .MuiChip-icon': {
          color: designTokens.colors.tertiary.main,
        },
        ...sx,
      }}
    />
  );
}
