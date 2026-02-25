'use client';

import { useEffect, useState } from 'react';
import { Box, Button, IconButton, CircularProgress, Tooltip, Typography, Chip, Divider, Dialog, DialogTitle, DialogContent, DialogActions, TextField, FormControl, InputLabel, Select, MenuItem, FormControlLabel, Checkbox } from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';
import CropFreeIcon from '@mui/icons-material/CropFree';
import GridOnIcon from '@mui/icons-material/GridOn';
import PersonSearchIcon from '@mui/icons-material/PersonSearch';
import PlaylistPlayIcon from '@mui/icons-material/PlaylistPlay';
import VisibilityIcon from '@mui/icons-material/Visibility';
import VisibilityOffIcon from '@mui/icons-material/VisibilityOff';
import SportsVolleyballIcon from '@mui/icons-material/SportsVolleyball';
import EditIcon from '@mui/icons-material/Edit';
import OpenInNewIcon from '@mui/icons-material/OpenInNew';
import SaveIcon from '@mui/icons-material/Save';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import SwapHorizIcon from '@mui/icons-material/SwapHoriz';
import LabelIcon from '@mui/icons-material/Label';
import DeleteOutlineIcon from '@mui/icons-material/DeleteOutline';
import { usePlayerTrackingStore } from '@/stores/playerTrackingStore';
import { useEditorStore } from '@/stores/editorStore';
import { usePlayerStore } from '@/stores/playerStore';
import type { ActionsData } from '@/services/api';
import { getLabelStudioStatus, exportToLabelStudio, importFromLabelStudio, API_BASE_URL } from '@/services/api';

// Action colors for classified contacts
const ACTION_COLORS: Record<string, string> = {
  serve: '#4CAF50',
  receive: '#2196F3',
  set: '#FFC107',
  attack: '#f44336',
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
  const [showSwapDialog, setShowSwapDialog] = useState(false);
  const [calibrationPromptDismissed, setCalibrationPromptDismissed] = useState<Record<string, boolean>>({});
  const [swapTrackA, setSwapTrackA] = useState<number | ''>('');
  const [swapTrackB, setSwapTrackB] = useState<number | ''>('');
  const [swapFromCurrent, setSwapFromCurrent] = useState(true);
  const [isSwapping, setIsSwapping] = useState(false);

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
    showCourtDebugOverlay,
    toggleCourtDebugOverlay,
    swapTracks,
    isLabelingActions,
    setIsLabelingActions,
    actionGroundTruth,
    actionGtDirty,
    actionGtSaving,
    loadActionGroundTruth,
    saveActionGroundTruth,
    removeActionLabel,
    batchTracking,
    trackAllRalliesForVideo,
    pollBatchTrackingStatus,
  } = usePlayerTrackingStore();

  const currentTime = usePlayerStore((state) => state.currentTime);
  const seek = usePlayerStore((state) => state.seek);

  // Get backend rally ID from selected rally
  const selectedRally = rallies.find((r) => r.id === selectedRallyId);
  const backendRallyId = selectedRally?._backendId ?? null;
  const activeMatch = getActiveMatch();
  const fps = activeMatch?.video?.fps ?? 30;

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

  // Load action ground truth when tracking data is available
  useEffect(() => {
    if (backendRallyId && hasTrackingData) {
      loadActionGroundTruth(backendRallyId);
    }
  }, [backendRallyId, hasTrackingData, loadActionGroundTruth]);

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

  // Resume polling on mount if batch was in progress
  const batchStatus = activeMatchId ? batchTracking[activeMatchId] : undefined;
  const isBatchActive = batchStatus?.status === 'pending' || batchStatus?.status === 'processing';
  useEffect(() => {
    if (activeMatchId && isBatchActive) {
      pollBatchTrackingStatus(activeMatchId, fps);
    }
  }, [activeMatchId, isBatchActive, fps, pollBatchTrackingStatus]);

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

  // Open Label Studio — fresh export (forceRegenerate) or resume existing task
  const handleOpenLabelStudio = async (forceRegenerate: boolean) => {
    if (!backendRallyId || !activeMatch?.videoUrl) return;

    setLabelStudioLoading(true);
    try {
      const videoUrl = new URL(activeMatch.videoUrl, API_BASE_URL).href;
      const opts = forceRegenerate ? { forceRegenerate: true as const } : undefined;
      const result = await exportToLabelStudio(backendRallyId, videoUrl, opts);
      if (result.success && result.taskUrl) {
        setLabelStudioTaskId(result.taskId ?? null);
        window.open(result.taskUrl, '_blank');
      } else {
        console.error('Failed to open Label Studio:', result.error);
        alert(result.error || 'Failed to open Label Studio');
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

  // Available track IDs for swap dialog
  const availableTrackIds = trackData?.tracks?.map(t => t.trackId) ?? [];

  const handleOpenSwapDialog = () => {
    setSwapTrackA('');
    setSwapTrackB('');
    setSwapFromCurrent(true);
    setShowSwapDialog(true);
  };

  const handleSwapSubmit = async () => {
    if (!backendRallyId || swapTrackA === '' || swapTrackB === '' || swapTrackA === swapTrackB) return;

    const rallyStart = selectedRally?.start_time ?? 0;
    const fromFrame = swapFromCurrent
      ? Math.max(0, Math.round((currentTime - rallyStart) * fps))
      : 0;

    setIsSwapping(true);
    try {
      await swapTracks(backendRallyId, swapTrackA, swapTrackB, fromFrame, fps);
      setShowSwapDialog(false);
    } catch (error) {
      console.error('Failed to swap tracks:', error);
      alert(error instanceof Error ? error.message : 'Failed to swap tracks');
    } finally {
      setIsSwapping(false);
    }
  };

  const handleTrackAllRallies = async () => {
    if (!activeMatchId) return;
    await trackAllRalliesForVideo(activeMatchId);
  };

  const hasBallPositions = !!trackData?.ballPositions?.length;

  const showTrackingTools = hasTrackingData && !isCalibrating;

  const handleToggleActionLabeling = () => {
    setIsLabelingActions(!isLabelingActions);
  };

  const handleSeekToLabel = (frame: number) => {
    if (!selectedRally || !trackData) return;
    const time = selectedRally.start_time + frame / trackData.fps;
    seek(time);
  };

  const handleDeleteLabel = (frame: number) => {
    if (!backendRallyId) return;
    removeActionLabel(backendRallyId, frame);
  };

  const handleSaveActionGt = async () => {
    if (!backendRallyId) return;
    try {
      await saveActionGroundTruth(backendRallyId);
    } catch (error) {
      console.error('Failed to save action GT:', error);
      alert(error instanceof Error ? error.message : 'Failed to save action ground truth');
    }
  };

  const gtLabels = backendRallyId ? actionGroundTruth[backendRallyId] : undefined;
  const gtCount = gtLabels?.length ?? 0;
  const isDirty = backendRallyId ? actionGtDirty[backendRallyId] : false;
  const isSaving = backendRallyId ? actionGtSaving[backendRallyId] : false;

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', py: 0.75, px: 1 }}>
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.75, flexWrap: 'wrap' }}>
        {/* ── Setup ── */}
        {!isCalibrating && (
          <>
            <Button
              size="small"
              variant="outlined"
              startIcon={<CropFreeIcon />}
              onClick={handleStartCalibration}
            >
              {hasCalibration ? 'Re-calibrate' : 'Calibrate'}
            </Button>

            {selectedRallyId && backendRallyId && (
              <Button
                size="small"
                variant={hasTrackingData ? 'outlined' : 'contained'}
                startIcon={
                  isTrackingRally || isLoadingTrackData
                    ? <CircularProgress size={16} />
                    : <PersonSearchIcon />
                }
                onClick={handleTrackPlayers}
                disabled={isTrackingRally || isLoadingTrackData || isBatchActive}
              >
                {isTrackingRally
                  ? 'Tracking...'
                  : isLoadingTrackData
                    ? 'Loading...'
                    : hasTrackingData
                      ? 'Re-track'
                      : 'Track Players'}
              </Button>
            )}

            <Tooltip title="Track all rallies in this video (batch processing)">
              <span>
                <Button
                  size="small"
                  variant={isBatchActive ? 'contained' : 'outlined'}
                  startIcon={
                    isBatchActive
                      ? <CircularProgress size={16} />
                      : <PlaylistPlayIcon />
                  }
                  onClick={handleTrackAllRallies}
                  disabled={isBatchActive || isTrackingRally}
                  color={isBatchActive ? 'warning' : 'primary'}
                >
                  {isBatchActive
                    ? `Tracking ${batchStatus?.completedRallies ?? 0}/${batchStatus?.totalRallies ?? '?'}`
                    : 'Track All'}
                </Button>
              </span>
            </Tooltip>
          </>
        )}

        {/* ── Overlays ── */}
        {showTrackingTools && (
          <>
            <Divider orientation="vertical" flexItem sx={{ mx: 0.25 }} />
            <Tooltip title={showPlayerOverlay ? 'Hide players' : 'Show players'}>
              <IconButton
                size="small"
                onClick={togglePlayerOverlay}
                color={showPlayerOverlay ? 'primary' : 'default'}
              >
                {showPlayerOverlay ? <VisibilityIcon fontSize="small" /> : <VisibilityOffIcon fontSize="small" />}
              </IconButton>
            </Tooltip>
            {hasBallPositions && (
              <Tooltip title={showBallOverlay ? 'Hide ball track' : 'Show ball track'}>
                <IconButton
                  size="small"
                  onClick={toggleBallOverlay}
                  sx={{
                    color: showBallOverlay ? '#FFC107' : 'action.active',
                    '&:hover': { color: showBallOverlay ? '#FFB300' : undefined },
                  }}
                >
                  <SportsVolleyballIcon fontSize="small" />
                </IconButton>
              </Tooltip>
            )}
            <Tooltip title={showCourtDebugOverlay ? 'Hide court debug' : 'Show court debug'}>
              <IconButton
                size="small"
                onClick={toggleCourtDebugOverlay}
                sx={{
                  color: showCourtDebugOverlay ? '#00BCD4' : 'action.active',
                  '&:hover': { color: showCourtDebugOverlay ? '#00ACC1' : undefined },
                }}
              >
                <GridOnIcon fontSize="small" />
              </IconButton>
            </Tooltip>
          </>
        )}

        {/* ── Action Labeling ── */}
        {showTrackingTools && hasBallPositions && (
          <>
            <Divider orientation="vertical" flexItem sx={{ mx: 0.25 }} />
            <Button
              size="small"
              variant={isLabelingActions ? 'contained' : 'outlined'}
              startIcon={<LabelIcon />}
              onClick={handleToggleActionLabeling}
              color={isLabelingActions ? 'warning' : 'primary'}
            >
              {isLabelingActions ? 'Labeling...' : 'Label Actions'}
            </Button>
            {gtCount > 0 && (
              <Button
                size="small"
                variant={isDirty ? 'contained' : 'outlined'}
                startIcon={isSaving ? <CircularProgress size={14} /> : <SaveIcon />}
                onClick={handleSaveActionGt}
                disabled={isSaving || !isDirty}
                color={isDirty ? 'primary' : 'success'}
              >
                {isSaving ? 'Saving...' : isDirty ? `Save GT (${gtCount})` : `GT Saved (${gtCount})`}
              </Button>
            )}
          </>
        )}

        {/* ── Tools ── */}
        {showTrackingTools && availableTrackIds.length >= 2 && (
          <>
            <Divider orientation="vertical" flexItem sx={{ mx: 0.25 }} />
            <Button
              size="small"
              variant="outlined"
              startIcon={<SwapHorizIcon />}
              onClick={handleOpenSwapDialog}
              disabled={isSwapping}
            >
              Swap Tracks
            </Button>
          </>
        )}

        {/* ── Label Studio ── */}
        {showTrackingTools && (
          <>
            <Divider orientation="vertical" flexItem sx={{ mx: 0.25 }} />
            <Tooltip title="Export tracking to Label Studio">
              <span>
                <Button
                  size="small"
                  variant="outlined"
                  startIcon={labelStudioLoading ? <CircularProgress size={14} /> : <EditIcon />}
                  onClick={() => handleOpenLabelStudio(true)}
                  disabled={labelStudioLoading}
                >
                  Label
                </Button>
              </span>
            </Tooltip>
            {labelStudioTaskId && (
              <Tooltip title="Resume existing annotations">
                <span>
                  <IconButton
                    size="small"
                    onClick={() => handleOpenLabelStudio(false)}
                    disabled={labelStudioLoading}
                    color="primary"
                  >
                    <OpenInNewIcon fontSize="small" />
                  </IconButton>
                </span>
              </Tooltip>
            )}
            <Tooltip title={hasGroundTruth ? 'Ground truth saved' : 'Save labels as ground truth'}>
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

      {/* Batch Tracking Status */}
      {batchStatus && batchStatus.status !== 'idle' && (
        <Box sx={{
          display: 'flex',
          alignItems: 'center',
          gap: 0.75,
          mt: 0.75,
          px: 0.5,
          py: 0.25,
          borderRadius: 1,
          bgcolor: batchStatus.status === 'failed' ? 'error.dark' : batchStatus.status === 'completed' ? 'success.dark' : 'action.hover',
        }}>
          {isBatchActive && <CircularProgress size={14} sx={{ color: 'warning.main' }} />}
          <Typography variant="caption" sx={{ color: batchStatus.status === 'failed' || batchStatus.status === 'completed' ? 'white' : 'text.secondary' }}>
            {batchStatus.status === 'completed'
              ? `All ${batchStatus.totalRallies} rallies tracked${batchStatus.failedRallies ? ` (${batchStatus.failedRallies} failed)` : ''}`
              : batchStatus.status === 'failed'
                ? `Batch tracking failed: ${batchStatus.error ?? 'Unknown error'}`
                : `Tracking rally ${(batchStatus.completedRallies ?? 0) + (batchStatus.failedRallies ?? 0) + 1} of ${batchStatus.totalRallies ?? '?'}...`}
          </Typography>
          {batchStatus.rallyStatuses && isBatchActive && (
            <Box sx={{ display: 'flex', gap: 0.25, ml: 'auto' }}>
              {batchStatus.rallyStatuses.map((rs) => (
                <Box
                  key={rs.rallyId}
                  sx={{
                    width: 8,
                    height: 8,
                    borderRadius: '50%',
                    bgcolor: rs.status === 'COMPLETED' ? 'success.main'
                      : rs.status === 'FAILED' ? 'error.main'
                      : rs.status === 'PROCESSING' ? 'warning.main'
                      : 'action.disabled',
                  }}
                />
              ))}
            </Box>
          )}
        </Box>
      )}

      {/* Calibration Recommendation Prompt */}
      {activeMatchId && !hasCalibration && !isCalibrating
        && !calibrationPromptDismissed[activeMatchId]
        && trackData?.qualityReport?.calibrationRecommended && (
        <Box sx={{
          display: 'flex',
          alignItems: 'center',
          gap: 1,
          mt: 0.75,
          px: 1,
          py: 0.5,
          borderRadius: 1,
          bgcolor: 'info.dark',
          color: 'white',
        }}>
          <CropFreeIcon sx={{ fontSize: 16 }} />
          <Typography variant="caption" sx={{ flex: 1 }}>
            Court calibration recommended — improves tracking accuracy and enables real-world stats
          </Typography>
          <Button
            size="small"
            variant="outlined"
            onClick={handleStartCalibration}
            sx={{ color: 'white', borderColor: 'rgba(255,255,255,0.5)', fontSize: '0.7rem', py: 0, minHeight: 24 }}
          >
            Calibrate
          </Button>
          <IconButton
            size="small"
            onClick={() => setCalibrationPromptDismissed((prev) => ({ ...prev, [activeMatchId!]: true }))}
            sx={{ color: 'rgba(255,255,255,0.7)', p: 0.25 }}
          >
            <CloseIcon sx={{ fontSize: 14 }} />
          </IconButton>
        </Box>
      )}

      {/* Action Sequence */}
      {showTrackingTools && trackData?.actions?.actions?.length && (
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, mt: 0.75, flexWrap: 'wrap' }}>
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

      {/* Action Labeling Shortcut Legend */}
      {isLabelingActions && (
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mt: 0.75, flexWrap: 'wrap' }}>
          <Typography variant="caption" sx={{ color: 'warning.main', fontWeight: 'bold' }}>
            Keys:
          </Typography>
          {[
            { key: 'S', label: 'Serve', color: '#4CAF50' },
            { key: 'R', label: 'Receive', color: '#2196F3' },
            { key: 'T', label: 'Set', color: '#FFC107' },
            { key: 'A', label: 'Attack', color: '#f44336' },
            { key: 'B', label: 'Block', color: '#9C27B0' },
            { key: 'D', label: 'Dig', color: '#FF9800' },
          ].map(({ key, label, color }) => (
            <Chip
              key={key}
              label={`${key} = ${label}`}
              size="small"
              sx={{
                bgcolor: color,
                color: 'white',
                fontSize: '0.65rem',
                height: 20,
                fontWeight: 'bold',
                '& .MuiChip-label': { px: 0.5 },
              }}
            />
          ))}
          <Typography variant="caption" sx={{ color: 'text.secondary' }}>
            , . = step frame | ESC = exit
          </Typography>
        </Box>
      )}

      {/* GT Label List */}
      {isLabelingActions && gtLabels && gtLabels.length > 0 && (
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, mt: 0.5, flexWrap: 'wrap' }}>
          <Typography variant="caption" sx={{ color: 'text.secondary', fontWeight: 'bold', mr: 0.5 }}>
            GT Labels:
          </Typography>
          {gtLabels.map((label) => (
            <Box key={label.frame} sx={{ display: 'flex', alignItems: 'center' }}>
              <Chip
                label={`${formatPhase(label.action)} f${label.frame}`}
                size="small"
                onClick={() => handleSeekToLabel(label.frame)}
                onDelete={() => handleDeleteLabel(label.frame)}
                deleteIcon={<DeleteOutlineIcon sx={{ fontSize: '14px !important' }} />}
                sx={{
                  bgcolor: ACTION_COLORS[label.action] || ACTION_COLORS.unknown,
                  color: 'white',
                  fontSize: '0.65rem',
                  height: 22,
                  fontWeight: 'bold',
                  cursor: 'pointer',
                  '& .MuiChip-label': { px: 0.5 },
                  '& .MuiChip-deleteIcon': { color: 'rgba(255,255,255,0.7)', '&:hover': { color: 'white' } },
                }}
              />
            </Box>
          ))}
        </Box>
      )}

      {/* Swap Tracks Dialog */}
      <Dialog open={showSwapDialog} onClose={() => !isSwapping && setShowSwapDialog(false)}>
        <DialogTitle>Swap Player Tracks</DialogTitle>
        <DialogContent>
          <Typography variant="body2" sx={{ mb: 2 }}>
            Swap track IDs between two players to fix ID switches.
          </Typography>
          <Box sx={{ display: 'flex', gap: 2, mb: 2 }}>
            <FormControl fullWidth size="small">
              <InputLabel>Track A</InputLabel>
              <Select
                value={swapTrackA}
                label="Track A"
                onChange={(e) => setSwapTrackA(e.target.value as number)}
              >
                {availableTrackIds.map((id) => (
                  <MenuItem key={id} value={id} disabled={id === swapTrackB}>
                    Player #{id}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
            <FormControl fullWidth size="small">
              <InputLabel>Track B</InputLabel>
              <Select
                value={swapTrackB}
                label="Track B"
                onChange={(e) => setSwapTrackB(e.target.value as number)}
              >
                {availableTrackIds.map((id) => (
                  <MenuItem key={id} value={id} disabled={id === swapTrackA}>
                    Player #{id}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </Box>
          <FormControlLabel
            control={
              <Checkbox
                checked={swapFromCurrent}
                onChange={(e) => setSwapFromCurrent(e.target.checked)}
              />
            }
            label={`From current time (frame ${Math.max(0, Math.round((currentTime - (selectedRally?.start_time ?? 0)) * fps))})`}
          />
          {!swapFromCurrent && (
            <Typography variant="caption" color="text.secondary" sx={{ display: 'block', ml: 4 }}>
              Swaps for entire rally
            </Typography>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setShowSwapDialog(false)} disabled={isSwapping}>Cancel</Button>
          <Button
            onClick={handleSwapSubmit}
            variant="contained"
            disabled={swapTrackA === '' || swapTrackB === '' || swapTrackA === swapTrackB || isSwapping}
            startIcon={isSwapping ? <CircularProgress size={16} /> : undefined}
          >
            {isSwapping ? 'Swapping...' : 'Swap'}
          </Button>
        </DialogActions>
      </Dialog>

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
