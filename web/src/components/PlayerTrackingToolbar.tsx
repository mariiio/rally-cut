'use client';

import { useEffect, useState } from 'react';
import { Box, Button, CircularProgress, Tooltip, Typography, Chip, Dialog, DialogTitle, DialogContent, DialogActions, TextField } from '@mui/material';
import CropFreeIcon from '@mui/icons-material/CropFree';
import PersonSearchIcon from '@mui/icons-material/PersonSearch';
import VisibilityIcon from '@mui/icons-material/Visibility';
import VisibilityOffIcon from '@mui/icons-material/VisibilityOff';
import SportsVolleyballIcon from '@mui/icons-material/SportsVolleyball';
import EditIcon from '@mui/icons-material/Edit';
import SaveIcon from '@mui/icons-material/Save';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import { usePlayerTrackingStore } from '@/stores/playerTrackingStore';
import { useEditorStore } from '@/stores/editorStore';
import type { ActionsData } from '@/services/api';
import { getLabelStudioStatus, exportToLabelStudio, importFromLabelStudio, API_BASE_URL } from '@/services/api';

// Action colors for classified contacts
const ACTION_COLORS: Record<string, string> = {
  serve: '#4CAF50',
  receive: '#2196F3',
  set: '#FFC107',
  spike: '#f44336',
  block: '#9C27B0',
  dig: '#FF9800',
  unknown: '#9e9e9e',
};

// Format phase for display
function formatPhase(phase: string): string {
  return phase.charAt(0).toUpperCase() + phase.slice(1);
}

export function PlayerTrackingToolbar() {
  const [labelStudioLoading, setLabelStudioLoading] = useState(false);
  const [hasGroundTruth, setHasGroundTruth] = useState(false);
  const [labelStudioTaskId, setLabelStudioTaskId] = useState<number | null>(null);
  const [showImportDialog, setShowImportDialog] = useState(false);
  const [importTaskId, setImportTaskId] = useState('');

  const activeMatchId = useEditorStore((state) => state.activeMatchId);
  const selectedRallyId = useEditorStore((state) => state.selectedRallyId);
  const rallies = useEditorStore((state) => state.rallies);
  const getActiveMatch = useEditorStore((state) => state.getActiveMatch);

  const {
    isCalibrating,
    setIsCalibrating,
    calibrations,
    isTracking,
    isLoadingTrack,
    playerTracks,
    trackPlayersForRally,
    loadPlayerTrack,
    showPlayerOverlay,
    togglePlayerOverlay,
    showBallOverlay,
    toggleBallOverlay,
  } = usePlayerTrackingStore();

  // Get backend rally ID from selected rally
  const selectedRally = rallies.find((r) => r.id === selectedRallyId);
  const backendRallyId = selectedRally?._backendId ?? null;
  const activeMatch = getActiveMatch();
  const fps = activeMatch?.video?.fps || 30;

  // Compute track data variables (needed for hooks below)
  const hasCalibration = activeMatchId ? !!calibrations[activeMatchId] : false;
  const isTrackingRally = backendRallyId ? isTracking[backendRallyId] : false;
  const isLoadingTrackData = backendRallyId ? isLoadingTrack[backendRallyId] : false;
  const trackData = backendRallyId ? playerTracks[backendRallyId]?.tracksJson : null;
  const hasTrackingData = !!trackData?.tracks?.length;

  // Load existing tracking data when rally is selected
  useEffect(() => {
    if (backendRallyId) {
      loadPlayerTrack(backendRallyId, fps);
    }
  }, [backendRallyId, fps, loadPlayerTrack]);

  // Check Label Studio status when tracking data is available
  useEffect(() => {
    const checkLabelStudioStatus = async () => {
      if (!backendRallyId || !hasTrackingData) {
        setHasGroundTruth(false);
        setLabelStudioTaskId(null);
        return;
      }
      try {
        const status = await getLabelStudioStatus(backendRallyId);
        setHasGroundTruth(status.hasGroundTruth);
        setLabelStudioTaskId(status.taskId ?? null);
      } catch (error) {
        console.error('Failed to get Label Studio status:', error);
      }
    };
    checkLabelStudioStatus();
  }, [backendRallyId, hasTrackingData]);

  // Don't show if no video loaded
  if (!activeMatchId) {
    return null;
  }

  const handleStartCalibration = () => {
    setIsCalibrating(true);
  };

  const handleTrackPlayers = async () => {
    if (!backendRallyId || !activeMatchId) return;
    await trackPlayersForRally(backendRallyId, activeMatchId, fps);
  };

  // Open tracking data in Label Studio for correction
  const handleOpenInLabelStudio = async () => {
    if (!backendRallyId || !activeMatch?.videoUrl) return;

    setLabelStudioLoading(true);
    try {
      // Convert relative URL to absolute URL for Label Studio (must use API server, not web server)
      const videoUrl = new URL(activeMatch.videoUrl, API_BASE_URL).href;
      // Reuse existing task if available (task ID stored in groundTruthTaskId)
      // Frame timing is now calculated at fixed 30fps for correct Label Studio sync
      const result = await exportToLabelStudio(backendRallyId, videoUrl);
      if (result.success && result.taskUrl) {
        setLabelStudioTaskId(result.taskId ?? null);
        // Open Label Studio in new tab
        window.open(result.taskUrl, '_blank');
      } else {
        console.error('Failed to export to Label Studio:', result.error);
        alert(result.error || 'Failed to export to Label Studio');
      }
    } catch (error) {
      console.error('Failed to open Label Studio:', error);
      alert(error instanceof Error ? error.message : 'Failed to open Label Studio');
    } finally {
      setLabelStudioLoading(false);
    }
  };

  // Import corrected annotations from Label Studio
  const handleSaveGroundTruth = async () => {
    if (!backendRallyId) return;

    // If we have a known task ID, use it directly
    if (labelStudioTaskId) {
      setLabelStudioLoading(true);
      try {
        const result = await importFromLabelStudio(backendRallyId, labelStudioTaskId);
        if (result.success) {
          setHasGroundTruth(true);
          alert(`Ground truth saved! ${result.playerCount} player annotations, ${result.ballCount} ball annotations across ${result.frameCount} frames.`);
        } else {
          console.error('Failed to import from Label Studio:', result.error);
          alert(result.error || 'Failed to save ground truth');
        }
      } catch (error) {
        console.error('Failed to save ground truth:', error);
        alert(error instanceof Error ? error.message : 'Failed to save ground truth');
      } finally {
        setLabelStudioLoading(false);
      }
    } else {
      // Show dialog to enter task ID manually
      setShowImportDialog(true);
    }
  };

  // Handle import dialog submission
  const handleImportSubmit = async () => {
    if (!backendRallyId || !importTaskId) return;

    const taskId = parseInt(importTaskId, 10);
    if (isNaN(taskId) || taskId <= 0) {
      alert('Please enter a valid task ID');
      return;
    }

    setShowImportDialog(false);
    setLabelStudioLoading(true);
    try {
      const result = await importFromLabelStudio(backendRallyId, taskId);
      if (result.success) {
        setHasGroundTruth(true);
        setLabelStudioTaskId(taskId);
        setImportTaskId('');
        alert(`Ground truth saved! ${result.playerCount} player annotations, ${result.ballCount} ball annotations across ${result.frameCount} frames.`);
      } else {
        console.error('Failed to import from Label Studio:', result.error);
        alert(result.error || 'Failed to save ground truth');
      }
    } catch (error) {
      console.error('Failed to save ground truth:', error);
      alert(error instanceof Error ? error.message : 'Failed to save ground truth');
    } finally {
      setLabelStudioLoading(false);
    }
  };

  const hasBallPositions = !!trackData?.ballPositions?.length;

  return (
    <Box
      sx={{
        display: 'flex',
        flexDirection: 'column',
        py: 1,
        px: 0.5,
      }}
    >
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, flexWrap: 'wrap' }}>
      {/* Calibration button */}
      {!isCalibrating && (
        <Button
          size="small"
          variant="outlined"
          startIcon={<CropFreeIcon />}
          onClick={handleStartCalibration}
        >
          {hasCalibration ? 'Re-calibrate Court' : 'Calibrate Court'}
        </Button>
      )}

      {/* Track Players button */}
      {!isCalibrating && selectedRallyId && backendRallyId && (
        <Tooltip title={!selectedRallyId ? 'Select a rally first' : ''}>
          <span>
            <Button
              size="small"
              variant="outlined"
              startIcon={
                isTrackingRally || isLoadingTrackData ? (
                  <CircularProgress size={16} />
                ) : (
                  <PersonSearchIcon />
                )
              }
              onClick={handleTrackPlayers}
              disabled={isTrackingRally || isLoadingTrackData || !selectedRallyId}
            >
              {isTrackingRally
                ? 'Tracking...'
                : isLoadingTrackData
                  ? 'Loading...'
                  : hasTrackingData
                    ? 'Re-track Players'
                    : 'Track Players'}
            </Button>
          </span>
        </Tooltip>
      )}

      {/* Toggle player overlay visibility */}
      {hasTrackingData && !isCalibrating && (
        <Tooltip title={showPlayerOverlay ? 'Hide player overlay' : 'Show player overlay'}>
          <Button
            size="small"
            variant={showPlayerOverlay ? 'contained' : 'outlined'}
            onClick={togglePlayerOverlay}
            sx={{ minWidth: 'auto', px: 1 }}
          >
            {showPlayerOverlay ? <VisibilityIcon /> : <VisibilityOffIcon />}
          </Button>
        </Tooltip>
      )}

      {/* Toggle ball overlay visibility - only show if ball positions available */}
      {hasTrackingData && hasBallPositions && !isCalibrating && (
        <Tooltip title={showBallOverlay ? 'Hide ball track' : 'Show ball track'}>
          <Button
            size="small"
            variant={showBallOverlay ? 'contained' : 'outlined'}
            onClick={toggleBallOverlay}
            sx={{
              minWidth: 'auto',
              px: 1,
              bgcolor: showBallOverlay ? '#FFC107' : undefined,
              '&:hover': showBallOverlay ? { bgcolor: '#FFB300' } : undefined,
            }}
          >
            <SportsVolleyballIcon fontSize="small" />
          </Button>
        </Tooltip>
      )}

      {/* Label Studio Integration - Ground Truth Labeling */}
      {hasTrackingData && !isCalibrating && (
        <>
          <Tooltip title="Open in Label Studio to correct tracking">
            <span>
              <Button
                size="small"
                variant="outlined"
                startIcon={labelStudioLoading ? <CircularProgress size={14} /> : <EditIcon />}
                onClick={handleOpenInLabelStudio}
                disabled={labelStudioLoading}
                sx={{ ml: 1 }}
              >
                Label
              </Button>
            </span>
          </Tooltip>
          <Tooltip title={hasGroundTruth ? 'Ground truth saved' : 'Save corrected labels as ground truth'}>
            <span>
              <Button
                size="small"
                variant={hasGroundTruth ? 'contained' : 'outlined'}
                startIcon={hasGroundTruth ? <CheckCircleIcon /> : <SaveIcon />}
                onClick={handleSaveGroundTruth}
                disabled={labelStudioLoading}
                color={hasGroundTruth ? 'success' : 'primary'}
              >
                {hasGroundTruth ? 'GT Saved' : 'Save GT'}
              </Button>
            </span>
          </Tooltip>
        </>
      )}
      </Box>

      {/* Action Sequence - from contact detection + classification */}
      {hasTrackingData && trackData?.actions?.actions?.length && !isCalibrating && (
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, mt: 0.5, px: 0.5, flexWrap: 'wrap' }}>
          <Typography variant="caption" sx={{ color: 'text.secondary', fontWeight: 'bold', mr: 0.5 }}>
            Actions:
          </Typography>
          {(trackData.actions as ActionsData).actions.map((action, index) => (
            <Box key={index} sx={{ display: 'flex', alignItems: 'center' }}>
              {index > 0 && (
                <Typography variant="caption" sx={{ color: 'text.disabled', mx: 0.25 }}>
                  {'\u2192'}
                </Typography>
              )}
              <Tooltip title={`Frame ${action.frame} | Player #${action.playerTrackId} | ${action.courtSide} court`}>
                <Chip
                  label={formatPhase(action.action)}
                  size="small"
                  sx={{
                    bgcolor: ACTION_COLORS[action.action] || ACTION_COLORS.unknown,
                    color: 'white',
                    fontSize: '0.7rem',
                    height: 22,
                    fontWeight: 'bold',
                    '& .MuiChip-label': { px: 0.75 },
                  }}
                />
              </Tooltip>
            </Box>
          ))}
          <Typography variant="caption" sx={{ color: 'text.secondary', ml: 1 }}>
            ({(trackData.actions as ActionsData).numContacts} contacts)
          </Typography>
        </Box>
      )}

      {/* Import Task ID Dialog */}
      <Dialog open={showImportDialog} onClose={() => setShowImportDialog(false)}>
        <DialogTitle>Import Ground Truth from Label Studio</DialogTitle>
        <DialogContent>
          <Typography variant="body2" sx={{ mb: 2 }}>
            Enter the Label Studio task ID to import corrected annotations.
          </Typography>
          <TextField
            autoFocus
            label="Task ID"
            type="number"
            fullWidth
            value={importTaskId}
            onChange={(e) => setImportTaskId(e.target.value)}
            helperText="Find the task ID in the Label Studio URL (e.g., .../task=123)"
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setShowImportDialog(false)}>Cancel</Button>
          <Button onClick={handleImportSubmit} variant="contained" disabled={!importTaskId}>
            Import
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}
