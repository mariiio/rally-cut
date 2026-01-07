'use client';

import { useState, useEffect, useCallback } from 'react';
import {
  Box,
  Button,
  Typography,
  LinearProgress,
  Tooltip,
  Alert,
  CircularProgress,
} from '@mui/material';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import LockIcon from '@mui/icons-material/Lock';
import RestoreIcon from '@mui/icons-material/Restore';
import { useEditorStore, ConfirmationStatus } from '@/stores/editorStore';
import { confirmRallies, getConfirmationStatus, restoreOriginalVideo } from '@/services/api';
import { ConfirmDialog } from './ConfirmDialog';

interface ConfirmRalliesProps {
  matchId: string;
  isPremium: boolean;
}

export function ConfirmRallies({ matchId, isPremium }: ConfirmRalliesProps) {
  const {
    confirmationStatus,
    setConfirmationStatus,
    isConfirming,
    setIsConfirming,
    updateConfirmationProgress,
    rallies,
    reloadCurrentMatch,
  } = useEditorStore();

  const [error, setError] = useState<string | null>(null);
  const [showConfirmDialog, setShowConfirmDialog] = useState(false);
  const [showRestoreDialog, setShowRestoreDialog] = useState(false);
  const [isRestoring, setIsRestoring] = useState(false);

  const status = confirmationStatus[matchId];
  const isLocked = status?.status === 'CONFIRMED';
  const isProcessing = status?.status === 'PENDING' || status?.status === 'PROCESSING';

  // Load initial status and poll when processing
  useEffect(() => {
    let isMounted = true;
    let intervalId: ReturnType<typeof setInterval> | null = null;

    const fetchStatus = async () => {
      if (!isMounted) return;

      try {
        const result = await getConfirmationStatus(matchId);
        if (!isMounted) return;

        if (result.confirmation) {
          setConfirmationStatus(matchId, {
            id: result.confirmation.id,
            status: result.confirmation.status,
            progress: result.confirmation.progress,
            error: result.confirmation.error,
            confirmedAt: result.confirmation.confirmedAt,
            originalDurationMs: result.confirmation.originalDurationMs,
            trimmedDurationMs: result.confirmation.trimmedDurationMs,
          });

          // If completed or failed, stop polling and handle result
          if (result.confirmation.status === 'CONFIRMED') {
            setIsConfirming(false);
            if (intervalId) {
              clearInterval(intervalId);
              intervalId = null;
            }
            await reloadCurrentMatch();
          } else if (result.confirmation.status === 'FAILED') {
            setIsConfirming(false);
            if (intervalId) {
              clearInterval(intervalId);
              intervalId = null;
            }
            setError(result.confirmation.error || 'Confirmation failed');
          }
        }
      } catch (e) {
        // Ignore errors - status may not exist yet
        console.debug('No confirmation status for match:', matchId, e);
      }
    };

    // Initial load
    fetchStatus();

    // Start polling if processing
    if (isProcessing) {
      intervalId = setInterval(fetchStatus, 2000);
    }

    return () => {
      isMounted = false;
      if (intervalId) {
        clearInterval(intervalId);
      }
    };
  }, [isProcessing, matchId, setConfirmationStatus, setIsConfirming, reloadCurrentMatch]);

  const handleConfirm = useCallback(async () => {
    if (!isPremium) {
      setError('Rally confirmation requires Premium tier');
      return;
    }

    if (rallies.length === 0) {
      setError('No rallies to confirm');
      return;
    }

    setError(null);
    setIsConfirming(true);

    try {
      const result = await confirmRallies(matchId);
      setConfirmationStatus(matchId, {
        id: result.confirmationId,
        status: result.status,
        progress: result.progress,
        error: null,
        confirmedAt: null,
        originalDurationMs: 0, // Will be updated on next poll
        trimmedDurationMs: null,
      });
    } catch (e) {
      setIsConfirming(false);
      setError(e instanceof Error ? e.message : 'Failed to confirm rallies');
    }
  }, [isPremium, rallies.length, matchId, setConfirmationStatus, setIsConfirming]);

  const handleRestore = useCallback(async () => {
    setShowRestoreDialog(false);
    setIsRestoring(true);
    setError(null);

    try {
      await restoreOriginalVideo(matchId);
      setConfirmationStatus(matchId, null);
      // Reload match to get original rally timestamps
      await reloadCurrentMatch();
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to restore original');
    } finally {
      setIsRestoring(false);
    }
  }, [matchId, setConfirmationStatus, reloadCurrentMatch]);

  // Not Premium - show disabled button
  if (!isPremium) {
    return (
      <Tooltip title="Rally confirmation requires Premium tier">
        <span>
          <Button
            variant="outlined"
            size="small"
            disabled
            startIcon={<LockIcon />}
            sx={{ fontSize: 11, py: 0.5 }}
          >
            Confirm Rallies
          </Button>
        </span>
      </Tooltip>
    );
  }

  // Show error alert with helpful messages
  if (error) {
    const isNoRalliesError = error.toLowerCase().includes('at least one rally') ||
                             error.toLowerCase().includes('no rallies');
    return (
      <Alert
        severity={isNoRalliesError ? 'info' : 'error'}
        onClose={() => setError(null)}
        sx={{ py: 0.5, fontSize: 11 }}
      >
        {isNoRalliesError ? (
          <>
            <strong>No rallies to confirm.</strong> Add rallies by clicking the{' '}
            <strong>+ Add Rally</strong> button or pressing <kbd>Cmd+Enter</kbd> while
            playing the video.
          </>
        ) : (
          error
        )}
      </Alert>
    );
  }

  // Processing - show progress
  if (isProcessing) {
    return (
      <Box sx={{ width: '100%' }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 0.5 }}>
          <CircularProgress size={14} />
          <Typography variant="caption" sx={{ fontSize: 11, color: 'text.secondary' }}>
            Generating trimmed video...
          </Typography>
          <Typography
            variant="caption"
            sx={{ fontSize: 11, color: 'text.secondary', fontFamily: 'monospace' }}
          >
            {status?.progress ?? 0}%
          </Typography>
        </Box>
        <LinearProgress
          variant="determinate"
          value={status?.progress ?? 0}
          sx={{
            height: 3,
            bgcolor: 'rgba(255, 255, 255, 0.06)',
            '& .MuiLinearProgress-bar': {
              bgcolor: 'primary.main',
            },
          }}
        />
      </Box>
    );
  }

  // Confirmed - show locked status with restore button
  if (isLocked) {
    return (
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
        <Box
          sx={{
            display: 'flex',
            alignItems: 'center',
            gap: 0.5,
            px: 1,
            py: 0.25,
            bgcolor: 'success.main',
            borderRadius: 1,
            opacity: 0.9,
          }}
        >
          <CheckCircleIcon sx={{ fontSize: 14 }} />
          <Typography variant="caption" sx={{ fontSize: 11, fontWeight: 500 }}>
            Confirmed
          </Typography>
        </Box>
        <Tooltip title="Restore original video and unlock editing">
          <Button
            variant="text"
            size="small"
            onClick={() => setShowRestoreDialog(true)}
            startIcon={isRestoring ? <CircularProgress size={12} /> : <RestoreIcon />}
            disabled={isRestoring}
            sx={{
              fontSize: 11,
              py: 0.25,
              px: 1,
              minWidth: 0,
              color: 'text.secondary',
              '&:hover': { color: 'warning.main' },
            }}
          >
            Restore
          </Button>
        </Tooltip>

        <ConfirmDialog
          open={showRestoreDialog}
          title="Restore original video?"
          message="This will delete the trimmed video and restore rally timestamps to their original values. You'll need to confirm again to create a new trimmed video."
          confirmLabel="Restore"
          cancelLabel="Cancel"
          onConfirm={handleRestore}
          onCancel={() => setShowRestoreDialog(false)}
        />
      </Box>
    );
  }

  // Default - show confirm button
  const hasRallies = rallies.length > 0;
  const tooltipTitle = hasRallies
    ? 'Generate a trimmed video with only rally segments'
    : 'Add rallies first using + Add Rally or Cmd+Enter';

  return (
    <>
      <Tooltip title={tooltipTitle}>
        <span>
          <Button
            variant="outlined"
            size="small"
            onClick={() => setShowConfirmDialog(true)}
            disabled={isConfirming || !hasRallies}
            startIcon={isConfirming ? <CircularProgress size={12} /> : <CheckCircleIcon />}
            sx={{
              fontSize: 11,
              py: 0.5,
              borderColor: 'divider',
              '&:hover': {
                borderColor: 'primary.main',
                bgcolor: 'rgba(144, 202, 249, 0.08)',
              },
            }}
          >
            Confirm Rallies
          </Button>
        </span>
      </Tooltip>

      <ConfirmDialog
        open={showConfirmDialog}
        title="Confirm rallies?"
        message="This will create a new trimmed video containing only the rally segments. Dead time between rallies will be removed, and rally editing will be locked."
        confirmLabel="Confirm"
        cancelLabel="Cancel"
        onConfirm={handleConfirm}
        onCancel={() => setShowConfirmDialog(false)}
      />
    </>
  );
}
