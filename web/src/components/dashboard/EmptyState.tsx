'use client';

import { Box, Button, Typography } from '@mui/material';
import SportsVolleyballIcon from '@mui/icons-material/SportsVolleyball';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import SearchOffIcon from '@mui/icons-material/SearchOff';
import FolderOffIcon from '@mui/icons-material/FolderOff';
import { designTokens } from '@/app/theme';

type EmptyStateVariant = 'no-videos' | 'no-sessions' | 'search-no-results';

interface EmptyStateProps {
  variant: EmptyStateVariant;
  onAction?: () => void;
  actionLabel?: string;
  title?: string;
  description?: string;
}

const variants: Record<EmptyStateVariant, {
  icon: typeof SportsVolleyballIcon;
  defaultTitle: string;
  defaultDescription: string;
  defaultActionLabel: string;
  showAnimation: boolean;
}> = {
  'no-videos': {
    icon: SportsVolleyballIcon,
    defaultTitle: 'Get started with RallyCut',
    defaultDescription: 'Upload your volleyball video to begin analyzing rallies and creating highlights',
    defaultActionLabel: 'Upload Your First Video',
    showAnimation: true,
  },
  'no-sessions': {
    icon: FolderOffIcon,
    defaultTitle: 'No sessions yet',
    defaultDescription: 'Create a session to organize your volleyball videos',
    defaultActionLabel: 'Create Session',
    showAnimation: false,
  },
  'search-no-results': {
    icon: SearchOffIcon,
    defaultTitle: 'No results found',
    defaultDescription: 'Try adjusting your search terms or filters',
    defaultActionLabel: 'Clear Search',
    showAnimation: false,
  },
};

export function EmptyState({
  variant,
  onAction,
  actionLabel,
  title,
  description,
}: EmptyStateProps) {
  const config = variants[variant];
  const Icon = config.icon;

  return (
    <Box sx={{ textAlign: 'center', py: { xs: 8, md: 12 } }}>
      <Box
        sx={{
          width: { xs: 100, md: 120 },
          height: { xs: 100, md: 120 },
          mx: 'auto',
          mb: 4,
          borderRadius: '50%',
          background: `linear-gradient(135deg, ${designTokens.colors.surface[2]} 0%, ${designTokens.colors.surface[1]} 100%)`,
          border: '2px dashed',
          borderColor: 'rgba(255, 107, 74, 0.3)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          position: 'relative',
          ...(config.showAnimation && {
            animation: 'pulse 3s ease-in-out infinite',
            '@keyframes pulse': {
              '0%, 100%': {
                borderColor: 'rgba(255, 107, 74, 0.3)',
                transform: 'scale(1)',
              },
              '50%': {
                borderColor: 'rgba(255, 107, 74, 0.6)',
                transform: 'scale(1.02)',
              },
            },
          }),
        }}
      >
        <Icon
          sx={{
            fontSize: { xs: 48, md: 56 },
            color: 'primary.main',
            filter: 'drop-shadow(0 4px 12px rgba(255, 107, 74, 0.3))',
          }}
        />
      </Box>

      <Typography
        variant="h5"
        sx={{
          mb: 1.5,
          fontWeight: 600,
          background: designTokens.gradients.primary,
          backgroundClip: 'text',
          WebkitBackgroundClip: 'text',
          WebkitTextFillColor: 'transparent',
        }}
      >
        {title || config.defaultTitle}
      </Typography>

      <Typography
        variant="body1"
        color="text.secondary"
        sx={{ mb: 4, maxWidth: 420, mx: 'auto', px: 2 }}
      >
        {description || config.defaultDescription}
      </Typography>

      {onAction && (
        <Button
          variant="contained"
          size="large"
          startIcon={variant === 'no-videos' ? <CloudUploadIcon /> : undefined}
          onClick={onAction}
          sx={{
            px: 4,
            py: 1.5,
            fontSize: '1rem',
            transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
            '&:hover': {
              transform: 'translateY(-3px)',
              boxShadow: '0 8px 24px rgba(255, 107, 74, 0.4)',
            },
          }}
        >
          {actionLabel || config.defaultActionLabel}
        </Button>
      )}
    </Box>
  );
}

export function EmptyStateCompact({
  title,
  description,
  onAction,
  actionLabel,
}: {
  title: string;
  description?: string;
  onAction?: () => void;
  actionLabel?: string;
}) {
  return (
    <Box
      sx={{
        textAlign: 'center',
        py: 6,
        px: 3,
        bgcolor: designTokens.colors.surface[1],
        borderRadius: 3,
        border: '1px dashed',
        borderColor: 'divider',
      }}
    >
      <Typography variant="subtitle1" fontWeight={600} sx={{ mb: 0.5 }}>
        {title}
      </Typography>
      {description && (
        <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
          {description}
        </Typography>
      )}
      {onAction && actionLabel && (
        <Button
          variant="outlined"
          size="small"
          onClick={onAction}
          sx={{
            borderColor: 'primary.main',
            color: 'primary.main',
            '&:hover': {
              bgcolor: 'rgba(255, 107, 74, 0.1)',
            },
          }}
        >
          {actionLabel}
        </Button>
      )}
    </Box>
  );
}
