import { Box, type SxProps, type Theme } from '@mui/material';
import { ReactNode } from 'react';
import { designTokens } from '@/app/designTokens';

interface IconBoxProps {
  icon: ReactNode;
  color?: 'primary' | 'secondary';
  size?: 'sm' | 'md' | 'lg';
  sx?: SxProps<Theme>;
}

const sizeMap = {
  sm: { box: 32, icon: 18, radius: 1 },
  md: { box: 40, icon: 22, radius: 1.5 },
  lg: { box: 48, icon: 28, radius: 2 },
} as const;

export function IconBox({ icon, color = 'primary', size = 'sm', sx }: IconBoxProps) {
  const dims = sizeMap[size];
  const bg = color === 'primary'
    ? designTokens.alpha.primary[15]
    : designTokens.alpha.secondary[15];
  const svgColor = color === 'primary' ? 'primary.main' : 'secondary.main';

  return (
    <Box
      sx={{
        width: dims.box,
        height: dims.box,
        borderRadius: dims.radius,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        bgcolor: bg,
        '& svg': {
          fontSize: dims.icon,
          color: svgColor,
        },
        ...sx,
      }}
    >
      {icon}
    </Box>
  );
}
