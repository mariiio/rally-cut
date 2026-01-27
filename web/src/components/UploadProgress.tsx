'use client';

import { useEffect, useState } from 'react';
import {
  IconButton,
  Snackbar,
  Alert,
  Button,
} from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';
import UpgradeIcon from '@mui/icons-material/Upgrade';
import Link from 'next/link';
import { useUploadStore } from '@/stores/uploadStore';
import { useTierStore } from '@/stores/tierStore';
import { ConfirmDialog } from './ConfirmDialog';
import { ProgressBar } from './ProgressBar';

export function UploadProgress() {
  const { isUploading, progress, currentStep, error, cancel, clearError, reset } = useUploadStore();
  const [showCancelDialog, setShowCancelDialog] = useState(false);
  const [showSuccessInfo, setShowSuccessInfo] = useState(false);
  const isPaidTier = useTierStore((state) => state.isPaidTier());

  const isComplete = !isUploading && progress === 100;

  // Show success info snackbar for FREE users after upload completes
  useEffect(() => {
    if (isComplete && !isPaidTier) {
      // Small delay to let the progress bar finish animation
      const timer = setTimeout(() => setShowSuccessInfo(true), 500);
      return () => clearTimeout(timer);
    }
  }, [isComplete, isPaidTier]);

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

  // Show nothing when not uploading (but keep snackbar if open)
  if (!isUploading && progress === 0 && !showSuccessInfo) {
    return null;
  }

  // Only show snackbar when progress bar is hidden
  if (!isUploading && progress === 0 && showSuccessInfo) {
    return (
      <Snackbar
        open={showSuccessInfo}
        autoHideDuration={10000}
        onClose={() => setShowSuccessInfo(false)}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      >
        <Alert
          severity="info"
          onClose={() => setShowSuccessInfo(false)}
          sx={{ width: '100%', maxWidth: 500 }}
          action={
            <Button
              component={Link}
              href="/pricing"
              color="inherit"
              size="small"
              startIcon={<UpgradeIcon />}
            >
              Upgrade
            </Button>
          }
        >
          Video saved! Free accounts keep videos forever.
          Original quality exports available for 3 days.
        </Alert>
      </Snackbar>
    );
  }

  return (
    <>
      <ProgressBar
        progress={progress}
        isActive={isUploading}
        isComplete={isComplete}
        stepText={currentStep}
        completeText="Upload complete"
        actions={
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
        }
      />

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
    </>
  );
}
