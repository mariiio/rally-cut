'use client';

import { useEffect } from 'react';
import {
  Box,
  LinearProgress,
  Typography,
  Snackbar,
  Alert,
} from '@mui/material';
import CheckIcon from '@mui/icons-material/Check';
import { useExportStore } from '@/stores/exportStore';

export function ExportProgress() {
  const { isExporting, progress, currentStep, error, clearError, reset } = useExportStore();

  // Warn user before leaving page during export
  useEffect(() => {
    const handleBeforeUnload = (e: BeforeUnloadEvent) => {
      if (isExporting) {
        e.preventDefault();
        e.returnValue = 'Export in progress. Are you sure you want to leave?';
        return e.returnValue;
      }
    };

    window.addEventListener('beforeunload', handleBeforeUnload);
    return () => window.removeEventListener('beforeunload', handleBeforeUnload);
  }, [isExporting]);

  // Auto-reset after success
  useEffect(() => {
    if (!isExporting && progress === 100) {
      const timer = setTimeout(() => reset(), 2500);
      return () => clearTimeout(timer);
    }
  }, [isExporting, progress, reset]);

  // Show error snackbar
  if (error) {
    return (
      <Snackbar
        open={true}
        autoHideDuration={6000}
        onClose={clearError}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
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

  const isComplete = !isExporting && progress === 100;

  return (
    <Box
      sx={{
        width: '100%',
        opacity: isExporting || isComplete ? 1 : 0,
        transition: 'opacity 0.2s ease',
      }}
    >
      {/* Progress bar */}
      <Box sx={{ position: 'relative', height: 3 }}>
        <LinearProgress
          variant="determinate"
          value={progress}
          sx={{
            height: 3,
            bgcolor: 'rgba(255, 255, 255, 0.06)',
            '& .MuiLinearProgress-bar': {
              bgcolor: isComplete ? 'success.main' : 'primary.main',
              transition: 'transform 0.2s ease',
            },
          }}
        />
        {/* Glow sweep */}
        {!isComplete && progress > 0 && (
          <Box
            sx={{
              position: 'absolute',
              top: 0,
              bottom: 0,
              left: 0,
              width: `${progress}%`,
              overflow: 'hidden',
              pointerEvents: 'none',
            }}
          >
            <Box
              sx={{
                position: 'absolute',
                top: 0,
                bottom: 0,
                width: 40,
                background: 'linear-gradient(90deg, transparent, rgba(255,255,255,0.35), transparent)',
                animation: 'sweep 1.5s ease-in-out infinite',
                '@keyframes sweep': {
                  '0%': { left: '-40px' },
                  '100%': { left: '100%' },
                },
              }}
            />
          </Box>
        )}
      </Box>

      {/* Status text */}
      <Box
        sx={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          px: 1,
          py: 0.5,
          bgcolor: 'rgba(0, 0, 0, 0.3)',
        }}
      >
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.75 }}>
          {isComplete && (
            <CheckIcon sx={{ fontSize: 12, color: 'success.main' }} />
          )}
          <Typography
            variant="caption"
            sx={{
              color: isComplete ? 'success.main' : 'text.secondary',
              fontSize: 11,
            }}
          >
            {isComplete ? 'Export complete' : currentStep}
          </Typography>
        </Box>
        {!isComplete && (
          <Typography
            variant="caption"
            sx={{
              color: 'text.secondary',
              fontSize: 11,
              fontFamily: 'monospace',
            }}
          >
            {progress}%
          </Typography>
        )}
      </Box>
    </Box>
  );
}
