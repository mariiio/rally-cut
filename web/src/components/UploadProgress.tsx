'use client';

import { useEffect, useState } from 'react';
import {
  Box,
  LinearProgress,
  Typography,
  IconButton,
  Snackbar,
  Alert,
} from '@mui/material';
import CheckIcon from '@mui/icons-material/Check';
import CloseIcon from '@mui/icons-material/Close';
import { useUploadStore } from '@/stores/uploadStore';
import { ConfirmDialog } from './ConfirmDialog';

export function UploadProgress() {
  const { isUploading, progress, currentStep, error, cancel, clearError, reset } = useUploadStore();
  const [showCancelDialog, setShowCancelDialog] = useState(false);

  const isComplete = !isUploading && progress === 100;

  // Warn user before leaving page during upload
  useEffect(() => {
    const handleBeforeUnload = (e: BeforeUnloadEvent) => {
      if (isUploading) {
        e.preventDefault();
        e.returnValue = 'Upload in progress. Are you sure you want to leave?';
        return e.returnValue;
      }
    };

    window.addEventListener('beforeunload', handleBeforeUnload);
    return () => window.removeEventListener('beforeunload', handleBeforeUnload);
  }, [isUploading]);

  // Auto-reset after success
  useEffect(() => {
    if (isComplete) {
      const timer = setTimeout(() => reset(), 2500);
      return () => clearTimeout(timer);
    }
  }, [isComplete, reset]);

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

  // Show nothing when not uploading
  if (!isUploading && progress === 0) {
    return null;
  }

  return (
    <Box
      sx={{
        width: '100%',
        opacity: isUploading || isComplete ? 1 : 0,
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
            {isComplete ? 'Upload complete' : currentStep}
          </Typography>
        </Box>
        {!isComplete && (
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
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
            <IconButton
              size="small"
              onClick={() => setShowCancelDialog(true)}
              sx={{
                p: 0.25,
                color: 'text.secondary',
                '&:hover': {
                  color: 'error.main',
                },
              }}
            >
              <CloseIcon sx={{ fontSize: 14 }} />
            </IconButton>
          </Box>
        )}
      </Box>

      {/* Cancel confirmation dialog */}
      <ConfirmDialog
        open={showCancelDialog}
        title="Cancel upload?"
        message={`The upload is ${progress}% complete. Are you sure you want to cancel?`}
        confirmLabel="Cancel upload"
        cancelLabel="Continue"
        onConfirm={() => {
          cancel();
          setShowCancelDialog(false);
        }}
        onCancel={() => setShowCancelDialog(false)}
      />
    </Box>
  );
}
