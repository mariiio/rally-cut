'use client';

import { useEffect, useMemo, useState } from 'react';
import { Box, Button, IconButton, CircularProgress, Tooltip, Typography, Chip, Divider, Dialog, DialogTitle, DialogContent, DialogActions, TextField, FormControl, InputLabel, Select, MenuItem, FormControlLabel, Checkbox, ToggleButtonGroup, ToggleButton, Menu } from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';
import CropFreeIcon from '@mui/icons-material/CropFree';
import GridOnIcon from '@mui/icons-material/GridOn';
import PersonSearchIcon from '@mui/icons-material/PersonSearch';
import SportsVolleyballIcon from '@mui/icons-material/SportsVolleyball';
import EditIcon from '@mui/icons-material/Edit';
import OpenInNewIcon from '@mui/icons-material/OpenInNew';
import SaveIcon from '@mui/icons-material/Save';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import SwapHorizIcon from '@mui/icons-material/SwapHoriz';
import LabelIcon from '@mui/icons-material/Label';
import DeleteOutlineIcon from '@mui/icons-material/DeleteOutline';
import PeopleIcon from '@mui/icons-material/People';
import MoreVertIcon from '@mui/icons-material/MoreVert';
import LayersIcon from '@mui/icons-material/Layers';
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

interface PlayerTrackingToolbarProps {
  onOpenPlayerMatching?: () => void;
  onOpenReferenceCrops?: () => void;
}

export function PlayerTrackingToolbar({ onOpenPlayerMatching, onOpenReferenceCrops }: PlayerTrackingToolbarProps = {}) {
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
  const [swapMode, setSwapMode] = useState<'swap' | 'promote'>('swap');
  const [demoteTrackId, setDemoteTrackId] = useState<number | ''>('');
  const [promoteTrackId, setPromoteTrackId] = useState<number | ''>('');

  const [toolsMenuAnchor, setToolsMenuAnchor] = useState<null | HTMLElement>(null);

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
    showRawTracks,
    toggleRawTracks,
    swapTracks,
    promoteTracks,
    isLabelingActions,
    setIsLabelingActions,
    actionGroundTruth,
    actionGtDirty,
    actionGtSaving,
    loadActionGroundTruth,
    saveActionGroundTruth,
    removeActionLabel,
    batchTracking,
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

  const batchStatus = activeMatchId ? batchTracking[activeMatchId] : undefined;
  const isBatchActive = batchStatus?.status === 'pending' || batchStatus?.status === 'processing';

  // Player number mapping: sorted trackIds → 1-based display numbers
  const playerNumberMap = useMemo(() => {
    if (!trackData?.tracks?.length) return new Map<number, number>();
    const sorted = [...trackData.tracks].sort((a, b) => a.trackId - b.trackId);
    const map = new Map<number, number>();
    sorted.forEach((t, i) => map.set(t.trackId, i + 1));
    return map;
  }, [trackData?.tracks]);

  const gtLabels = backendRallyId ? actionGroundTruth[backendRallyId] : undefined;
  const gtCount = gtLabels?.length ?? 0;
  const isDirty = backendRallyId ? actionGtDirty[backendRallyId] : false;
  const isSaving = backendRallyId ? actionGtSaving[backendRallyId] : false;

  // Find which action is active based on current playback time
  const currentActionIndex = useMemo(() => {
    const actions = trackData?.actions as ActionsData | undefined;
    if (!actions?.actions?.length || !selectedRally) return -1;
    const currentFrame = Math.round((currentTime - selectedRally.start_time) * fps);
    // Find the last action whose frame is <= currentFrame (or closest upcoming one)
    let bestIdx = -1;
    for (let i = 0; i < actions.actions.length; i++) {
      if (actions.actions[i].frame <= currentFrame) {
        bestIdx = i;
      }
    }
    // If before all actions, highlight the first one if we're close (within 15 frames)
    if (bestIdx === -1 && actions.actions.length > 0 && actions.actions[0].frame - currentFrame < 15) {
      return 0;
    }
    // Don't highlight if we're far past the last action (>30 frames)
    if (bestIdx >= 0 && bestIdx === actions.actions.length - 1) {
      const dist = currentFrame - actions.actions[bestIdx].frame;
      if (dist > 30) return -1;
    }
    return bestIdx;
  }, [trackData?.actions, currentTime, selectedRally, fps]);

  // Find the GT label nearest to current playback time
  const nearestGtFrame = useMemo(() => {
    if (!gtLabels?.length || !selectedRally || !trackData) return null;
    const currentFrame = Math.round((currentTime - selectedRally.start_time) * trackData.fps);
    let bestFrame = gtLabels[0].frame;
    let bestDist = Math.abs(currentFrame - bestFrame);
    for (const label of gtLabels) {
      const dist = Math.abs(currentFrame - label.frame);
      if (dist < bestDist) {
        bestDist = dist;
        bestFrame = label.frame;
      }
    }
    return bestFrame;
  }, [gtLabels, currentTime, selectedRally, trackData]);

  // Compute frame ranges for raw tracks (for display hints in dropdown)
  const rawTrackFrameRanges = useMemo(() => {
    if (!trackData?.rawTracks) return new Map<number, { first: number; last: number; count: number }>();
    const ranges = new Map<number, { first: number; last: number; count: number }>();
    for (const rt of trackData.rawTracks) {
      if (rt.positions.length > 0) {
        const sorted = [...rt.positions].sort((a, b) => a.frame - b.frame);
        ranges.set(rt.trackId, {
          first: sorted[0].frame,
          last: sorted[sorted.length - 1].frame,
          count: sorted.length,
        });
      }
    }
    return ranges;
  }, [trackData?.rawTracks]);

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
  const availableRawTrackIds = trackData?.rawTracks?.map(t => t.trackId) ?? [];
  const hasRawTracks = availableRawTrackIds.length > 0;

  const handleOpenSwapDialog = () => {
    setSwapTrackA('');
    setSwapTrackB('');
    setDemoteTrackId('');
    setPromoteTrackId('');
    setSwapFromCurrent(true);
    setSwapMode('swap');
    setShowSwapDialog(true);
  };

  const handleSwapSubmit = async () => {
    if (!backendRallyId) return;
    if (swapMode === 'promote' && (demoteTrackId === '' || promoteTrackId === '')) return;
    if (swapMode === 'swap' && (swapTrackA === '' || swapTrackB === '' || swapTrackA === swapTrackB)) return;

    const rallyStart = selectedRally?.start_time ?? 0;
    const fromFrame = swapFromCurrent
      ? Math.max(0, Math.round((currentTime - rallyStart) * fps))
      : 0;

    setIsSwapping(true);
    try {
      if (swapMode === 'promote') {
        await promoteTracks(backendRallyId, demoteTrackId as number, promoteTrackId as number, fromFrame, fps);
      } else {
        await swapTracks(backendRallyId, swapTrackA as number, swapTrackB as number, fromFrame, fps);
      }
      setShowSwapDialog(false);
    } catch (error) {
      console.error('Failed to swap/promote tracks:', error);
      alert(error instanceof Error ? error.message : 'Failed to swap/promote tracks');
    } finally {
      setIsSwapping(false);
    }
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


  // Keycap style for keyboard shortcuts
  const kbdSx = {
    display: 'inline-flex',
    alignItems: 'center',
    justifyContent: 'center',
    minWidth: 20,
    height: 20,
    px: 0.5,
    borderRadius: '4px',
    fontSize: '0.65rem',
    fontWeight: 700,
    fontFamily: 'inherit',
    lineHeight: 1,
    bgcolor: 'rgba(255,255,255,0.12)',
    border: '1px solid rgba(255,255,255,0.2)',
    boxShadow: '0 1px 0 rgba(0,0,0,0.3)',
    color: 'text.primary',
  } as const;

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
                      ? 'Re-track Rally'
                      : 'Track Rally'}
              </Button>
            )}

            {onOpenReferenceCrops && (
              <Tooltip title="Select player reference crops for attribution">
                <IconButton
                  size="small"
                  onClick={onOpenReferenceCrops}
                  color="default"
                >
                  <PersonSearchIcon fontSize="small" />
                </IconButton>
              </Tooltip>
            )}

            {onOpenPlayerMatching && (
              <Tooltip title="Label cross-rally player matching ground truth">
                <IconButton
                  size="small"
                  onClick={onOpenPlayerMatching}
                  color="default"
                >
                  <PeopleIcon fontSize="small" />
                </IconButton>
              </Tooltip>
            )}
          </>
        )}

        {/* ── Overlay Toggles ── */}
        {showTrackingTools && (
          <>
            <Divider orientation="vertical" flexItem sx={{ mx: 0.25 }} />
            <ToggleButtonGroup size="small" sx={{ height: 28 }}>
              <ToggleButton
                value="players"
                selected={showPlayerOverlay}
                onClick={togglePlayerOverlay}
                sx={{ px: 0.75, textTransform: 'none', fontSize: '0.7rem', gap: 0.5 }}
              >
                <PersonSearchIcon sx={{ fontSize: 16 }} />
                Players
              </ToggleButton>
              {hasRawTracks && showPlayerOverlay && (
                <Tooltip title="Show all detected tracks (including non-primary)">
                  <ToggleButton
                    value="raw"
                    selected={showRawTracks}
                    onClick={toggleRawTracks}
                    sx={{
                      px: 0.75, textTransform: 'none', fontSize: '0.7rem', gap: 0.5,
                      '&.Mui-selected': { color: '#B39DDB', bgcolor: 'rgba(179,157,219,0.12)' },
                    }}
                  >
                    <LayersIcon sx={{ fontSize: 16 }} />
                    All
                  </ToggleButton>
                </Tooltip>
              )}
              {hasBallPositions && (
                <ToggleButton
                  value="ball"
                  selected={showBallOverlay}
                  onClick={toggleBallOverlay}
                  sx={{
                    px: 0.75, textTransform: 'none', fontSize: '0.7rem', gap: 0.5,
                    '&.Mui-selected': { color: '#FFC107', bgcolor: 'rgba(255,193,7,0.12)' },
                  }}
                >
                  <SportsVolleyballIcon sx={{ fontSize: 16 }} />
                  Ball
                </ToggleButton>
              )}
              <ToggleButton
                value="court"
                selected={showCourtDebugOverlay}
                onClick={toggleCourtDebugOverlay}
                sx={{
                  px: 0.75, textTransform: 'none', fontSize: '0.7rem', gap: 0.5,
                  '&.Mui-selected': { color: '#00BCD4', bgcolor: 'rgba(0,188,212,0.12)' },
                }}
              >
                <GridOnIcon sx={{ fontSize: 16 }} />
                Court
              </ToggleButton>
            </ToggleButtonGroup>
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

        {/* ── Tools Menu (Swap Tracks, Label Studio) ── */}
        {showTrackingTools && (
          <>
            <Divider orientation="vertical" flexItem sx={{ mx: 0.25 }} />
            <Tooltip title="More tools">
              <IconButton
                size="small"
                onClick={(e) => setToolsMenuAnchor(e.currentTarget)}
              >
                <MoreVertIcon fontSize="small" />
              </IconButton>
            </Tooltip>
            <Menu
              anchorEl={toolsMenuAnchor}
              open={!!toolsMenuAnchor}
              onClose={() => setToolsMenuAnchor(null)}
              slotProps={{ paper: { sx: { minWidth: 200 } } }}
            >
              {availableTrackIds.length >= 2 && (
                <MenuItem
                  onClick={() => { setToolsMenuAnchor(null); handleOpenSwapDialog(); }}
                  disabled={isSwapping}
                >
                  <SwapHorizIcon sx={{ mr: 1, fontSize: 18 }} />
                  Swap Tracks
                </MenuItem>
              )}
              <MenuItem
                onClick={() => { setToolsMenuAnchor(null); handleOpenLabelStudio(true); }}
                disabled={labelStudioLoading}
              >
                <EditIcon sx={{ mr: 1, fontSize: 18 }} />
                Export to Label Studio
              </MenuItem>
              {labelStudioTaskId && (
                <MenuItem
                  onClick={() => { setToolsMenuAnchor(null); handleOpenLabelStudio(false); }}
                  disabled={labelStudioLoading}
                >
                  <OpenInNewIcon sx={{ mr: 1, fontSize: 18 }} />
                  Resume Label Studio
                </MenuItem>
              )}
              <MenuItem
                onClick={() => { setToolsMenuAnchor(null); handleSaveGroundTruth(); }}
                disabled={labelStudioLoading}
              >
                {hasGroundTruth
                  ? <CheckCircleIcon sx={{ mr: 1, fontSize: 18, color: 'success.main' }} />
                  : <SaveIcon sx={{ mr: 1, fontSize: 18 }} />}
                {hasGroundTruth ? 'GT Saved' : 'Save GT from Label Studio'}
              </MenuItem>
            </Menu>
          </>
        )}
      </Box>


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
          {(trackData.actions as ActionsData).actions.map((action, index) => {
            const isActive = index === currentActionIndex;
            const pNum = action.playerTrackId >= 0 ? playerNumberMap.get(action.playerTrackId) : undefined;
            return (
            <Box key={index} sx={{ display: 'flex', alignItems: 'center' }}>
              {index > 0 && (
                <Typography variant="caption" sx={{ color: isActive ? 'text.primary' : 'text.disabled', mx: 0.25 }}>
                  {'\u2192'}
                </Typography>
              )}
              <Tooltip title={`Frame ${action.frame} | Player ${pNum != null ? `#${pNum}` : `#${action.playerTrackId}`} | ${action.courtSide} court`}>
                <Chip
                  label={pNum != null ? `${formatPhase(action.action)} P${pNum}` : formatPhase(action.action)}
                  size="small"
                  onClick={() => handleSeekToLabel(action.frame)}
                  sx={{
                    bgcolor: isActive
                      ? ACTION_COLORS[action.action] || ACTION_COLORS.unknown
                      : 'transparent',
                    color: isActive ? 'white' : 'text.secondary',
                    border: `1.5px solid ${ACTION_COLORS[action.action] || ACTION_COLORS.unknown}`,
                    fontSize: '0.7rem',
                    height: 22,
                    fontWeight: 'bold',
                    cursor: 'pointer',
                    transition: 'all 0.15s ease',
                    '& .MuiChip-label': { px: 0.75 },
                    ...(isActive && {
                      transform: 'scale(1.1)',
                      boxShadow: `0 0 8px ${ACTION_COLORS[action.action] || ACTION_COLORS.unknown}80`,
                    }),
                    ...(!isActive && {
                      opacity: currentActionIndex >= 0 ? 0.5 : 0.85,
                    }),
                  }}
                />
              </Tooltip>
            </Box>
            );
          })}
          <Typography variant="caption" sx={{ color: 'text.secondary', ml: 1 }}>
            ({(trackData.actions as ActionsData).numContacts} contacts)
          </Typography>
        </Box>
      )}

      {/* Action Labeling Shortcut Legend */}
      {isLabelingActions && (
        <Box sx={{
          display: 'flex',
          alignItems: 'center',
          gap: 0.75,
          mt: 0.75,
          px: 0.75,
          py: 0.5,
          borderRadius: 1,
          bgcolor: 'rgba(255, 152, 0, 0.08)',
          border: '1px solid rgba(255, 152, 0, 0.2)',
          flexWrap: 'wrap',
        }}>
          {[
            { key: 'S', label: 'Serve', color: '#4CAF50' },
            { key: 'R', label: 'Receive', color: '#2196F3' },
            { key: 'T', label: 'Set', color: '#FFC107' },
            { key: 'A', label: 'Attack', color: '#f44336' },
            { key: 'B', label: 'Block', color: '#9C27B0' },
            { key: 'D', label: 'Dig', color: '#FF9800' },
          ].map(({ key, label, color }) => (
            <Box key={key} sx={{ display: 'inline-flex', alignItems: 'center', gap: 0.25 }}>
              <Box component="kbd" sx={{ ...kbdSx, bgcolor: color, borderColor: color, color: 'white' }}>
                {key}
              </Box>
              <Typography variant="caption" sx={{ fontSize: '0.65rem', color: 'text.secondary' }}>
                {label}
              </Typography>
            </Box>
          ))}
          <Divider orientation="vertical" flexItem sx={{ mx: 0.25 }} />
          <Box sx={{ display: 'inline-flex', alignItems: 'center', gap: 0.5 }}>
            {['1', '2', '3', '4'].map((k) => (
              <Box key={k} component="kbd" sx={kbdSx}>{k}</Box>
            ))}
            <Typography variant="caption" sx={{ fontSize: '0.65rem', color: 'text.secondary' }}>
              player
            </Typography>
          </Box>
          <Divider orientation="vertical" flexItem sx={{ mx: 0.25 }} />
          <Box sx={{ display: 'inline-flex', alignItems: 'center', gap: 0.25 }}>
            <Box component="kbd" sx={kbdSx}>,</Box>
            <Box component="kbd" sx={kbdSx}>.</Box>
            <Typography variant="caption" sx={{ fontSize: '0.65rem', color: 'text.secondary', ml: 0.25 }}>
              step
            </Typography>
          </Box>
          <Box component="kbd" sx={{ ...kbdSx, ml: 0.5 }}>ESC</Box>
        </Box>
      )}

      {/* GT Label List */}
      {isLabelingActions && gtLabels && gtLabels.length > 0 && (
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, mt: 0.5, flexWrap: 'wrap' }}>
          <Typography variant="caption" sx={{ color: 'text.secondary', fontWeight: 'bold', mr: 0.5 }}>
            GT ({gtLabels.length}):
          </Typography>
          {gtLabels.map((label) => {
            const pNum = label.playerTrackId >= 0 ? playerNumberMap.get(label.playerTrackId) : undefined;
            const chipLabel = pNum != null
              ? `${formatPhase(label.action)} f${label.frame} P${pNum}`
              : `${formatPhase(label.action)} f${label.frame}`;
            const isNearest = label.frame === nearestGtFrame;
            return (
            <Box key={label.frame} sx={{ display: 'flex', alignItems: 'center' }}>
              <Chip
                label={chipLabel}
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
                  ...(isNearest && {
                    outline: '2px solid rgba(255,255,255,0.8)',
                    outlineOffset: 1,
                    boxShadow: '0 0 8px rgba(255,255,255,0.3)',
                  }),
                }}
              />
            </Box>
            );
          })}
        </Box>
      )}

      {/* Swap Tracks Dialog */}
      <Dialog open={showSwapDialog} onClose={() => !isSwapping && setShowSwapDialog(false)}>
        <DialogTitle>Swap Player Tracks</DialogTitle>
        <DialogContent>
          {hasRawTracks && (
            <ToggleButtonGroup
              value={swapMode}
              exclusive
              onChange={(_, v) => v && setSwapMode(v)}
              size="small"
              sx={{ mb: 2 }}
            >
              <ToggleButton value="swap" sx={{ textTransform: 'none', fontSize: '0.75rem' }}>
                Swap Primary
              </ToggleButton>
              <ToggleButton value="promote" sx={{ textTransform: 'none', fontSize: '0.75rem' }}>
                Replace with Raw
              </ToggleButton>
            </ToggleButtonGroup>
          )}

          {swapMode === 'swap' ? (
            <>
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
            </>
          ) : (
            <>
              <Typography variant="body2" sx={{ mb: 2 }}>
                Replace a primary track with a raw (non-primary) track.
              </Typography>
              <Box sx={{ display: 'flex', gap: 2, mb: 2 }}>
                <FormControl fullWidth size="small">
                  <InputLabel>Primary to replace</InputLabel>
                  <Select
                    value={demoteTrackId}
                    label="Primary to replace"
                    onChange={(e) => setDemoteTrackId(e.target.value as number)}
                  >
                    {availableTrackIds.map((id) => (
                      <MenuItem key={id} value={id}>
                        Player #{id}
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>
                <FormControl fullWidth size="small">
                  <InputLabel>Raw track to promote</InputLabel>
                  <Select
                    value={promoteTrackId}
                    label="Raw track to promote"
                    onChange={(e) => setPromoteTrackId(e.target.value as number)}
                  >
                    {availableRawTrackIds.map((id) => {
                      const range = rawTrackFrameRanges.get(id);
                      const hint = range ? ` (frames ${range.first}-${range.last})` : '';
                      return (
                        <MenuItem key={id} value={id}>
                          Raw #{id}{hint}
                        </MenuItem>
                      );
                    })}
                  </Select>
                </FormControl>
              </Box>
            </>
          )}

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
              {swapMode === 'swap' ? 'Swaps for entire rally' : 'Replaces for entire rally'}
            </Typography>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setShowSwapDialog(false)} disabled={isSwapping}>Cancel</Button>
          <Button
            onClick={handleSwapSubmit}
            variant="contained"
            disabled={
              isSwapping || (swapMode === 'swap'
                ? (swapTrackA === '' || swapTrackB === '' || swapTrackA === swapTrackB)
                : (demoteTrackId === '' || promoteTrackId === ''))
            }
            startIcon={isSwapping ? <CircularProgress size={16} /> : undefined}
          >
            {isSwapping ? (swapMode === 'swap' ? 'Swapping...' : 'Promoting...') : (swapMode === 'swap' ? 'Swap' : 'Replace')}
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
