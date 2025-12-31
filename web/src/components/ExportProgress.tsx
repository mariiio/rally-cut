'use client';

import {
  Box,
  LinearProgress,
  Typography,
  IconButton,
  Snackbar,
  Alert,
  CircularProgress,
} from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import { useExportStore } from '@/stores/exportStore';
import { designTokens } from '@/app/theme';

export function ExportProgress() {
  const { isExporting, progress, currentStep, error, clearError, reset } = useExportStore();

  // Show error snackbar
  if (error) {
    return (
      <Snackbar
        open={true}
        autoHideDuration={6000}
        onClose={clearError}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
        sx={{ mb: 1, mr: 1 }}
      >
        <Alert severity="error" onClose={clearError} sx={{ width: '100%' }}>
          {error}
        </Alert>
      </Snackbar>
    );
  }

  // Show nothing when not exporting
  if (!isExporting && progress === 0) {
    return null;
  }

  // Show success message briefly
  if (!isExporting && progress === 100) {
    return (
      <Box
        sx={{
          position: 'fixed',
          bottom: 16,
          right: 16,
          display: 'flex',
          alignItems: 'center',
          gap: 1,
          bgcolor: 'success.main',
          color: 'success.contrastText',
          borderRadius: 2,
          px: 2,
          py: 1,
          boxShadow: designTokens.shadows.lg,
          zIndex: 1300,
          animation: 'slideIn 0.2s ease-out',
          '@keyframes slideIn': {
            from: { opacity: 0, transform: 'translateY(8px)' },
            to: { opacity: 1, transform: 'translateY(0)' },
          },
        }}
        onClick={reset}
      >
        <CheckCircleIcon sx={{ fontSize: 18 }} />
        <Typography variant="body2" sx={{ fontWeight: 500 }}>
          Export complete
        </Typography>
      </Box>
    );
  }

  // Compact progress indicator in bottom-right corner
  return (
    <Box
      sx={{
        position: 'fixed',
        bottom: 16,
        right: 16,
        bgcolor: designTokens.colors.surface[3],
        borderRadius: 2,
        boxShadow: designTokens.shadows.lg,
        border: '1px solid',
        borderColor: 'divider',
        overflow: 'hidden',
        minWidth: 240,
        maxWidth: 280,
        zIndex: 1300,
        animation: 'slideIn 0.2s ease-out',
        '@keyframes slideIn': {
          from: { opacity: 0, transform: 'translateY(8px)' },
          to: { opacity: 1, transform: 'translateY(0)' },
        },
      }}
    >
      {/* Progress bar at top */}
      <LinearProgress
        variant="determinate"
        value={progress}
        sx={{
          height: 3,
          bgcolor: 'rgba(255,255,255,0.1)',
          '& .MuiLinearProgress-bar': {
            background: designTokens.gradients.primary,
          },
        }}
      />

      {/* Content */}
      <Box sx={{ px: 1.5, py: 1, display: 'flex', alignItems: 'center', gap: 1.5 }}>
        {/* Spinner */}
        <CircularProgress
          size={20}
          thickness={4}
          sx={{ color: 'primary.main' }}
        />

        {/* Text */}
        <Box sx={{ flex: 1, minWidth: 0 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <Typography
              variant="caption"
              sx={{
                fontWeight: 600,
                color: 'text.primary',
                fontSize: '0.75rem',
              }}
            >
              Exporting
            </Typography>
            <Typography
              variant="caption"
              sx={{
                fontFamily: 'monospace',
                color: 'primary.main',
                fontWeight: 600,
                fontSize: '0.75rem',
              }}
            >
              {progress}%
            </Typography>
          </Box>
          <Typography
            variant="caption"
            sx={{
              color: 'text.secondary',
              fontSize: '0.6875rem',
              display: 'block',
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              whiteSpace: 'nowrap',
            }}
          >
            {currentStep}
          </Typography>
        </Box>

        {/* Close button */}
        <IconButton
          size="small"
          onClick={reset}
          sx={{
            p: 0.25,
            color: 'text.secondary',
            '&:hover': { color: 'text.primary' },
          }}
        >
          <CloseIcon sx={{ fontSize: 16 }} />
        </IconButton>
      </Box>
    </Box>
  );
}
