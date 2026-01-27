'use client';

import { useEffect, useState } from 'react';
import {
  Snackbar,
  Alert,
  IconButton,
} from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';
import { useExportStore } from '@/stores/exportStore';
import { ConfirmDialog } from './ConfirmDialog';
import { ProgressBar } from './ProgressBar';

export function ExportProgress() {
  const { isExporting, progress, currentStep, error, clearError, reset, cancel } = useExportStore();
  const [showCancelDialog, setShowCancelDialog] = useState(false);

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
    <>
      <ProgressBar
        progress={progress}
        isActive={isExporting}
        isComplete={isComplete}
        stepText={currentStep}
        completeText="Export complete"
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
        title="Cancel export?"
        message={`The export is ${progress}% complete. Are you sure you want to cancel?`}
        confirmLabel="Cancel export"
        cancelLabel="Continue"
        onConfirm={cancel}
        onCancel={() => setShowCancelDialog(false)}
      />
    </>
  );
}
