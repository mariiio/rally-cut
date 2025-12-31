'use client';

import {
  Box,
  LinearProgress,
  Typography,
  IconButton,
  Snackbar,
  Alert,
} from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';
import { useExportStore } from '@/stores/exportStore';

export function ExportProgress() {
  const { isExporting, progress, currentStep, error, clearError, reset } = useExportStore();

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

  // Show progress bar when exporting
  if (!isExporting && progress === 0) {
    return null;
  }

  // Show success message briefly
  if (!isExporting && progress === 100) {
    return (
      <Snackbar
        open={true}
        autoHideDuration={2000}
        onClose={reset}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      >
        <Alert severity="success" sx={{ width: '100%' }}>
          Download complete
        </Alert>
      </Snackbar>
    );
  }

  return (
    <Box
      sx={{
        position: 'fixed',
        bottom: 16,
        left: '50%',
        transform: 'translateX(-50%)',
        bgcolor: 'background.paper',
        borderRadius: 2,
        boxShadow: 3,
        px: 3,
        py: 2,
        minWidth: 320,
        maxWidth: 400,
        zIndex: 1300,
      }}
    >
      <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
        <Typography variant="body2" sx={{ flex: 1, fontWeight: 500 }}>
          Exporting Video
        </Typography>
        <Typography variant="caption" color="text.secondary" sx={{ mr: 1 }}>
          {progress}%
        </Typography>
        <IconButton size="small" onClick={reset} sx={{ ml: 0.5 }}>
          <CloseIcon fontSize="small" />
        </IconButton>
      </Box>

      <LinearProgress
        variant="determinate"
        value={progress}
        sx={{ mb: 1, borderRadius: 1 }}
      />

      <Typography variant="caption" color="text.secondary">
        {currentStep}
      </Typography>
    </Box>
  );
}
