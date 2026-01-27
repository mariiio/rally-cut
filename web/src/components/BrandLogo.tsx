import { Stack, Typography, Tooltip } from '@mui/material';
import SportsVolleyballIcon from '@mui/icons-material/SportsVolleyball';
import { designTokens } from '@/app/designTokens';

interface BrandLogoProps {
  onClick: () => void;
  tooltip?: string;
}

export function BrandLogo({ onClick, tooltip }: BrandLogoProps) {
  const content = (
    <Stack
      direction="row"
      alignItems="center"
      spacing={1}
      onClick={onClick}
      sx={{
        minWidth: 'fit-content',
        cursor: 'pointer',
        borderRadius: 1,
        px: 1,
        py: 0.5,
        mx: -1,
        transition: 'background-color 0.2s',
        '&:hover': {
          bgcolor: 'action.hover',
        },
      }}
    >
      <SportsVolleyballIcon
        sx={{
          fontSize: 28,
          color: 'primary.main',
          filter: `drop-shadow(0 2px 4px ${designTokens.alpha.primary[30]})`,
        }}
      />
      <Typography
        variant="h6"
        sx={{
          fontWeight: 700,
          background: designTokens.gradients.primary,
          backgroundClip: 'text',
          WebkitBackgroundClip: 'text',
          WebkitTextFillColor: 'transparent',
          letterSpacing: '-0.02em',
        }}
      >
        RallyCut
      </Typography>
    </Stack>
  );

  if (tooltip) {
    return <Tooltip title={tooltip}>{content}</Tooltip>;
  }

  return content;
}
