'use client';

import { useEffect, useCallback } from 'react';
import {
  Box,
  Button,
  Typography,
  CircularProgress,
  Stack,
  Alert,
  AlertTitle,
  Link,
  Tooltip,
} from '@mui/material';
import AutoFixHighIcon from '@mui/icons-material/AutoFixHigh';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import CancelIcon from '@mui/icons-material/Cancel';
import { useAnalysisStore, type AnalysisPhase } from '@/stores/analysisStore';
import { useEditorStore } from '@/stores/editorStore';
import { useAuthStore } from '@/stores/authStore';

interface AnalysisPipelineProps {
  /** If true, the video already has rallies detected */
  hasRallies: boolean;
  /** If true, the video is locked (confirmed) */
  isLocked: boolean;
}

/** Phase progress steps for the step indicator */
const PHASE_ORDER: AnalysisPhase[] = ['quality_check', 'detecting', 'tracking', 'done'];

function PhaseIndicator({ currentPhase }: { currentPhase: AnalysisPhase }) {
  const labels = ['Check', 'Detect', 'Track', 'Done'];

  return (
    <Stack direction="row" spacing={0.5} alignItems="center">
      {PHASE_ORDER.map((phase, i) => {
        const phaseIndex = PHASE_ORDER.indexOf(currentPhase);
        const isComplete = i < phaseIndex || currentPhase === 'done';
        const isCurrent = phase === currentPhase || (currentPhase === 'quality_warning' && phase === 'quality_check') || (currentPhase === 'completing' && phase === 'tracking');
        const color = isComplete ? 'success.main' : isCurrent ? 'primary.main' : 'text.disabled';

        return (
          <Stack key={phase} direction="row" alignItems="center" spacing={0.5}>
            {i > 0 && (
              <Box
                sx={{
                  width: 12,
                  height: 1,
                  bgcolor: isComplete ? 'success.main' : 'divider',
                }}
              />
            )}
            <Typography
              variant="caption"
              sx={{
                fontSize: 10,
                fontWeight: isCurrent ? 600 : 400,
                color,
              }}
            >
              {labels[i]}
            </Typography>
          </Stack>
        );
      })}
    </Stack>
  );
}

function TimeEstimate({ ralliesFound }: { ralliesFound?: number }) {
  if (!ralliesFound) return null;
  // ~6s per rally on Modal GPU
  const estimatedSeconds = ralliesFound * 6;
  const minutes = Math.ceil(estimatedSeconds / 60);
  return (
    <Typography variant="caption" sx={{ color: 'text.disabled', fontSize: 10 }}>
      ~{minutes} min remaining
    </Typography>
  );
}

export function AnalysisPipeline({ hasRallies, isLocked }: AnalysisPipelineProps) {
  const activeMatchId = useEditorStore((s) => s.activeMatchId);
  const isAuthenticated = useAuthStore((s) => s.isAuthenticated);
  const promptSignIn = useAuthStore((s) => s.promptSignIn);

  const pipeline = useAnalysisStore((s) => activeMatchId ? s.getPipeline(activeMatchId) : null);
  const startAnalysis = useAnalysisStore((s) => s.startAnalysis);
  const dismissWarnings = useAnalysisStore((s) => s.dismissWarnings);
  const cancelAnalysis = useAnalysisStore((s) => s.cancelAnalysis);
  const resumeIfNeeded = useAnalysisStore((s) => s.resumeIfNeeded);

  const phase = pipeline?.phase ?? 'idle';

  // Resume polling on mount if pipeline is in-progress
  useEffect(() => {
    if (activeMatchId && (phase === 'detecting' || phase === 'tracking' || phase === 'completing')) {
      resumeIfNeeded(activeMatchId);
    }
  }, [activeMatchId, phase, resumeIfNeeded]);

  const handleStart = useCallback(() => {
    if (!activeMatchId) return;
    if (!isAuthenticated) {
      promptSignIn('Create an account to analyze your match');
      return;
    }
    startAnalysis(activeMatchId);
  }, [activeMatchId, isAuthenticated, promptSignIn, startAnalysis]);

  const handleCancel = useCallback(() => {
    if (activeMatchId) {
      cancelAnalysis(activeMatchId);
    }
  }, [activeMatchId, cancelAnalysis]);

  const handleDismissWarnings = useCallback(() => {
    if (activeMatchId) {
      dismissWarnings(activeMatchId);
    }
  }, [activeMatchId, dismissWarnings]);

  // Don't show for locked videos or when no match is active
  if (!activeMatchId || isLocked) return null;

  // Idle — show "Analyze Match" button only when no rallies exist
  if (phase === 'idle') {
    // If video already has rallies and no pipeline is running, show nothing
    // (detection button was already handled)
    if (hasRallies) return null;

    return (
      <Tooltip title="Detect rallies, track players, and generate match stats">
        <Button
          data-tutorial="detect-rallies"
          size="small"
          variant="outlined"
          startIcon={<AutoFixHighIcon sx={{ fontSize: 16 }} />}
          onClick={handleStart}
          disabled={!activeMatchId}
          sx={{
            fontSize: 12,
            py: 0.25,
            textTransform: 'none',
          }}
        >
          Analyze Match
        </Button>
      </Tooltip>
    );
  }

  // Quality warning — show warnings with continue/cancel
  if (phase === 'quality_warning' && pipeline?.qualityResult) {
    return (
      <Box sx={{ maxWidth: 400 }}>
        <Alert
          severity="warning"
          variant="outlined"
          sx={{
            py: 0.25,
            '& .MuiAlert-message': { py: 0.25 },
            '& .MuiAlertTitle-root': { fontSize: 12, mb: 0.25 },
            fontSize: 11,
          }}
          action={
            <Stack direction="row" spacing={0.5}>
              <Button
                size="small"
                onClick={handleDismissWarnings}
                sx={{ fontSize: 11, textTransform: 'none' }}
              >
                Continue
              </Button>
              <Link
                component="button"
                variant="caption"
                onClick={handleCancel}
                sx={{ fontSize: 11, color: 'text.secondary' }}
              >
                Cancel
              </Link>
            </Stack>
          }
        >
          <AlertTitle>Quality Issues</AlertTitle>
          {pipeline.qualityResult.quality.warnings.slice(0, 3).map((w, i) => (
            <Typography key={i} variant="caption" display="block" sx={{ fontSize: 10 }}>
              {w}
            </Typography>
          ))}
        </Alert>
      </Box>
    );
  }

  // In-progress phases: detecting, tracking, completing
  if (['quality_check', 'detecting', 'tracking', 'completing'].includes(phase)) {
    return (
      <Stack direction="row" alignItems="center" spacing={1.5} sx={{
        bgcolor: 'action.hover',
        px: 1.5,
        py: 0.5,
        borderRadius: 2,
      }}>
        <Box sx={{ position: 'relative', display: 'inline-flex' }}>
          <CircularProgress
            size={32}
            variant={pipeline!.progress > 5 ? 'determinate' : 'indeterminate'}
            value={pipeline!.progress}
            color="primary"
            thickness={4}
          />
          {pipeline!.progress > 5 && (
            <Box
              sx={{
                position: 'absolute',
                top: 0,
                left: 0,
                bottom: 0,
                right: 0,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
              }}
            >
              <Typography
                variant="caption"
                sx={{ fontSize: 9, fontWeight: 600, color: 'text.secondary' }}
              >
                {Math.round(pipeline!.progress)}%
              </Typography>
            </Box>
          )}
        </Box>
        <Stack spacing={0}>
          <Typography variant="caption" sx={{ color: 'text.primary', fontWeight: 500, fontSize: 12 }}>
            {pipeline!.stepMessage}
          </Typography>
          <Stack direction="row" spacing={1} alignItems="center">
            <PhaseIndicator currentPhase={phase as AnalysisPhase} />
            {phase === 'tracking' && (
              <TimeEstimate ralliesFound={pipeline!.trackingProgress?.total} />
            )}
          </Stack>
        </Stack>
        <Tooltip title="Cancel analysis">
          <Button
            size="small"
            onClick={handleCancel}
            sx={{ minWidth: 'auto', px: 0.5, color: 'text.secondary' }}
          >
            <CancelIcon sx={{ fontSize: 16 }} />
          </Button>
        </Tooltip>
      </Stack>
    );
  }

  // Error
  if (phase === 'error') {
    return (
      <Stack direction="row" alignItems="center" spacing={0.5}>
        <Typography variant="caption" sx={{ color: 'warning.main', maxWidth: 250, fontSize: 11 }}>
          {pipeline!.error || 'Analysis failed'}
        </Typography>
        <Button
          size="small"
          onClick={handleStart}
          sx={{ minWidth: 'auto', px: 1, fontSize: 11, textTransform: 'none' }}
        >
          Retry
        </Button>
        <Button
          size="small"
          onClick={handleCancel}
          sx={{ minWidth: 'auto', px: 0.5, color: 'text.secondary' }}
        >
          OK
        </Button>
      </Stack>
    );
  }

  // Done
  if (phase === 'done') {
    return (
      <Stack direction="row" alignItems="center" spacing={1} sx={{
        bgcolor: pipeline!.ralliesFound === 0 ? 'warning.main' : 'success.main',
        color: 'white',
        px: 1.5,
        py: 0.5,
        borderRadius: 2,
      }}>
        <CheckCircleIcon sx={{ fontSize: 16 }} />
        <Typography variant="caption" sx={{ fontWeight: 500, fontSize: 12 }}>
          {pipeline!.stepMessage}
        </Typography>
        <Button
          size="small"
          onClick={handleCancel}
          sx={{ minWidth: 'auto', px: 0.5, color: 'inherit', opacity: 0.7, fontSize: 11 }}
        >
          Dismiss
        </Button>
      </Stack>
    );
  }

  return null;
}
