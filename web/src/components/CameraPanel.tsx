'use client';

import { useMemo, useCallback, memo, useState, useEffect } from 'react';
import {
  Box,
  Typography,
  IconButton,
  Slider,
  ToggleButton,
  ToggleButtonGroup,
  Tooltip,
  Divider,
  Stack,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogContentText,
  DialogActions,
} from '@mui/material';
import DeleteIcon from '@mui/icons-material/Delete';
import CheckIcon from '@mui/icons-material/Check';
import CloseIcon from '@mui/icons-material/Close';
import CropPortraitIcon from '@mui/icons-material/CropPortrait';
import Crop169Icon from '@mui/icons-material/Crop169';
import VideocamIcon from '@mui/icons-material/Videocam';
import RestartAltIcon from '@mui/icons-material/RestartAlt';
import VisibilityIcon from '@mui/icons-material/Visibility';
import VisibilityOffIcon from '@mui/icons-material/VisibilityOff';
import { useEditorStore } from '@/stores/editorStore';
import { usePlayerStore } from '@/stores/playerStore';
import { useCameraStore, createDefaultKeyframe, selectCameraEdit, selectSelectedKeyframeId } from '@/stores/cameraStore';
import { designTokens } from '@/app/theme';
import type { AspectRatio, CameraKeyframe } from '@/types/camera';
import { ZOOM_MAX, ZOOM_STEP, KEYFRAME_TIME_THRESHOLD, ROTATION_MIN, ROTATION_MAX, ROTATION_STEP, DEFAULT_GLOBAL_CAMERA } from '@/types/camera';

// Format time as MM:SS.ms
function formatTime(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  const ms = Math.floor((seconds % 1) * 10);
  return `${mins}:${secs.toString().padStart(2, '0')}.${ms}`;
}

// Format offset percentage
function formatOffset(offset: number): string {
  return `${Math.round(offset * 100)}%`;
}

// Memoized keyframe list item
const KeyframeItem = memo(function KeyframeItem({
  keyframe,
  isSelected,
  isDeleting,
  onSelect,
  onDelete,
  onCancelDelete,
  rallyStartTime,
  rallyDuration,
}: {
  keyframe: CameraKeyframe;
  isSelected: boolean;
  isDeleting: boolean;
  onSelect: (id: string) => void;
  onDelete: (id: string) => void;
  onCancelDelete: () => void;
  rallyStartTime: number;
  rallyDuration: number;
}) {
  const handleSelect = useCallback(() => onSelect(keyframe.id), [onSelect, keyframe.id]);

  const handleDelete = useCallback((e: React.MouseEvent) => {
    e.stopPropagation();
    onDelete(keyframe.id);
  }, [onDelete, keyframe.id]);

  const handleCancelDelete = useCallback((e: React.MouseEvent) => {
    e.stopPropagation();
    onCancelDelete();
  }, [onCancelDelete]);

  // Calculate absolute time for display
  const absoluteTime = rallyStartTime + keyframe.timeOffset * rallyDuration;

  return (
    <Box
      onClick={handleSelect}
      sx={{
        p: 1,
        borderRadius: 1,
        bgcolor: isSelected ? 'primary.dark' : designTokens.colors.surface[3],
        cursor: 'pointer',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        '&:hover': {
          bgcolor: isSelected ? 'primary.dark' : designTokens.colors.surface[4],
        },
      }}
    >
      <Box sx={{ flex: 1 }}>
        <Typography variant="caption" sx={{ fontWeight: 500 }}>
          {formatTime(absoluteTime)}
        </Typography>
        <Typography variant="caption" sx={{ color: 'text.secondary', ml: 1 }}>
          {keyframe.zoom.toFixed(1)}x
          {keyframe.rotation !== 0 && ` · ${keyframe.rotation.toFixed(0)}°`}
        </Typography>
      </Box>
      {/* Delete with confirmation */}
      {isDeleting ? (
        <Stack direction="row" spacing={0.25}>
          <Tooltip title="Confirm delete">
            <IconButton
              size="small"
              onClick={handleDelete}
              sx={{
                color: 'white',
                bgcolor: 'error.main',
                width: 24,
                height: 24,
                '&:hover': { bgcolor: 'error.light' },
              }}
            >
              <CheckIcon sx={{ fontSize: 14 }} />
            </IconButton>
          </Tooltip>
          <Tooltip title="Cancel">
            <IconButton
              size="small"
              onClick={handleCancelDelete}
              sx={{ width: 24, height: 24 }}
            >
              <CloseIcon sx={{ fontSize: 14 }} />
            </IconButton>
          </Tooltip>
        </Stack>
      ) : (
        <IconButton
          size="small"
          onClick={handleDelete}
          sx={{ opacity: 0.6, '&:hover': { opacity: 1, color: 'error.main' } }}
        >
          <DeleteIcon fontSize="small" />
        </IconButton>
      )}
    </Box>
  );
});

export function CameraPanel() {
  // Local state for reset confirmation dialogs
  const [showResetConfirm, setShowResetConfirm] = useState(false);
  const [showGlobalResetConfirm, setShowGlobalResetConfirm] = useState(false);
  // Local state for keyframe delete confirmation
  const [deleteConfirmId, setDeleteConfirmId] = useState<string | null>(null);

  // Editor store - get selected rally and camera tab state
  const selectedRallyId = useEditorStore((state) => state.selectedRallyId);
  const rallies = useEditorStore((state) => state.rallies);
  const activeMatchId = useEditorStore((state) => state.activeMatchId);
  const setIsCameraTabActive = useEditorStore((state) => state.setIsCameraTabActive);

  const selectedRally = useMemo(
    () => rallies.find((r) => r.id === selectedRallyId) ?? null,
    [rallies, selectedRallyId]
  );

  // Player state
  const applyCameraEdits = usePlayerStore((state) => state.applyCameraEdits);
  const toggleApplyCameraEdits = usePlayerStore((state) => state.toggleApplyCameraEdits);
  const currentTime = usePlayerStore((state) => state.currentTime);
  const seek = usePlayerStore((state) => state.seek);

  // Camera store - use optimized selectors
  const cameraEdit = useCameraStore(selectCameraEdit(selectedRallyId));
  const selectedKeyframeId = useCameraStore(selectSelectedKeyframeId);

  // Get stable action references
  const setAspectRatio = useCameraStore((state) => state.setAspectRatio);
  const addKeyframe = useCameraStore((state) => state.addKeyframe);
  const updateKeyframe = useCameraStore((state) => state.updateKeyframe);
  const removeKeyframe = useCameraStore((state) => state.removeKeyframe);
  const selectKeyframe = useCameraStore((state) => state.selectKeyframe);
  const resetCamera = useCameraStore((state) => state.resetCamera);
  const getCameraStateAtTime = useCameraStore((state) => state.getCameraStateAtTime);
  const setIsAdjustingRotation = useCameraStore((state) => state.setIsAdjustingRotation);

  // Global camera settings
  const setGlobalSettings = useCameraStore((state) => state.setGlobalSettings);
  const resetGlobalSettings = useCameraStore((state) => state.resetGlobalSettings);

  // Get active keyframes for the current aspect ratio
  const activeKeyframes = useMemo(() => {
    if (!cameraEdit) return [];
    return cameraEdit.keyframes[cameraEdit.aspectRatio] ?? [];
  }, [cameraEdit]);

  // Get selected keyframe
  const selectedKeyframe = useMemo(() => {
    if (!selectedKeyframeId) return null;
    return activeKeyframes.find((kf) => kf.id === selectedKeyframeId) ?? null;
  }, [activeKeyframes, selectedKeyframeId]);

  // Calculate current time offset within rally
  const currentTimeOffset = useMemo(() => {
    if (!selectedRally) return 0;
    const duration = selectedRally.end_time - selectedRally.start_time;
    if (duration <= 0) return 0;
    const offset = (currentTime - selectedRally.start_time) / duration;
    return Math.max(0, Math.min(1, offset));
  }, [selectedRally, currentTime]);

  const rallyDuration = useMemo(
    () => selectedRally ? selectedRally.end_time - selectedRally.start_time : 0,
    [selectedRally]
  );

  // Get current video/match ID - use activeMatchId (works even when no rally selected)
  const currentVideoId = useMemo(() => {
    // First try to get from selected rally for consistency
    if (selectedRallyId) {
      // Rally IDs are formatted as `${matchId}_rally_${n}`
      const parts = selectedRallyId.split('_rally_');
      if (parts.length > 0) return parts[0];
    }
    // Fall back to activeMatchId (for when no rally is selected)
    return activeMatchId;
  }, [selectedRallyId, activeMatchId]);

  // Get current global settings for this video (targeted selector — only recomputes when THIS video's settings change)
  const currentVideoGlobalSettings = useCameraStore(
    useCallback((s) => s.globalCameraSettings[currentVideoId ?? ''], [currentVideoId])
  );
  const currentGlobalSettings = currentVideoGlobalSettings ?? DEFAULT_GLOBAL_CAMERA;

  // Check if current video has non-default global settings
  const videoHasGlobalSettings = currentGlobalSettings.zoom !== 1.0 ||
    currentGlobalSettings.positionX !== 0.5 ||
    currentGlobalSettings.positionY !== 0.5 ||
    currentGlobalSettings.rotation !== 0;

  // Get drag position to detect active dragging
  const dragPosition = useCameraStore((state) => state.dragPosition);

  // Auto-deselect keyframe when playhead moves away
  useEffect(() => {
    // Skip if no selection, no edit, or during drag
    if (!selectedKeyframeId || !cameraEdit || dragPosition) return;

    const selectedKf = activeKeyframes.find((kf) => kf.id === selectedKeyframeId);
    if (!selectedKf || !selectedRally) return;

    const rallyDuration = selectedRally.end_time - selectedRally.start_time;
    const thresholdOffset = rallyDuration > 0 ? KEYFRAME_TIME_THRESHOLD / rallyDuration : 0;
    const distance = Math.abs(currentTimeOffset - selectedKf.timeOffset);

    if (distance > thresholdOffset) {
      selectKeyframe(null);
    }
  }, [currentTimeOffset, selectedKeyframeId, cameraEdit, selectedRally, activeKeyframes, dragPosition, selectKeyframe]);

  // Handlers
  const handleAspectRatioChange = useCallback(
    (_: React.MouseEvent<HTMLElement>, newRatio: AspectRatio | null) => {
      if (newRatio && selectedRallyId) {
        setAspectRatio(selectedRallyId, newRatio);
        // Enter camera edit mode when changing aspect ratio
        setIsCameraTabActive(true);
      }
    },
    [selectedRallyId, setAspectRatio, setIsCameraTabActive]
  );

  // Keyframe delete with confirmation
  const handleKeyframeDeleteClick = useCallback((keyframeId: string) => {
    if (deleteConfirmId === keyframeId) {
      // Confirm delete
      if (selectedRallyId) {
        removeKeyframe(selectedRallyId, keyframeId);
      }
      setDeleteConfirmId(null);
    } else {
      // Show confirmation
      setDeleteConfirmId(keyframeId);
    }
  }, [deleteConfirmId, selectedRallyId, removeKeyframe]);

  const handleCancelKeyframeDelete = useCallback(() => {
    setDeleteConfirmId(null);
  }, []);

  const handleKeyframeSelect = useCallback((kfId: string) => {
    selectKeyframe(kfId);
    const kf = activeKeyframes.find(k => k.id === kfId);
    if (kf && selectedRally) {
      const keyframeTime = selectedRally.start_time + kf.timeOffset * rallyDuration;
      seek(keyframeTime);
    }
    setIsCameraTabActive(true);
  }, [selectKeyframe, activeKeyframes, selectedRally, rallyDuration, seek, setIsCameraTabActive]);

  const handleZoomChange = useCallback(
    (_: Event, value: number | number[]) => {
      if (!selectedRallyId || !selectedRally) return;
      const newZoom = value as number;

      // Enter camera edit mode when changing zoom
      setIsCameraTabActive(true);

      if (selectedKeyframeId) {
        // Keyframe is selected: update it
        updateKeyframe(selectedRallyId, selectedKeyframeId, { zoom: newZoom });
      } else {
        // No keyframe selected - check for nearby keyframe or create new
        const rallyDuration = selectedRally.end_time - selectedRally.start_time;
        const thresholdOffset = rallyDuration > 0 ? KEYFRAME_TIME_THRESHOLD / rallyDuration : 0;
        const nearestKeyframe = activeKeyframes.find(
          (kf) => Math.abs(kf.timeOffset - currentTimeOffset) < thresholdOffset
        );

        if (nearestKeyframe) {
          // Update nearby keyframe and select it
          updateKeyframe(selectedRallyId, nearestKeyframe.id, { zoom: newZoom });
          selectKeyframe(nearestKeyframe.id);
        } else {
          // Create new keyframe at current position with new zoom
          const currentState = getCameraStateAtTime(selectedRallyId, currentTimeOffset);
          const keyframe = createDefaultKeyframe(currentTimeOffset, {
            positionX: currentState.positionX,
            positionY: currentState.positionY,
            zoom: newZoom,
          });
          addKeyframe(selectedRallyId, keyframe);
        }
      }
    },
    [selectedRallyId, selectedRally, selectedKeyframeId, currentTimeOffset, activeKeyframes, updateKeyframe, addKeyframe, selectKeyframe, getCameraStateAtTime, setIsCameraTabActive]
  );

  const handleRotationChange = useCallback(
    (_: Event, value: number | number[]) => {
      if (!selectedRallyId || !selectedRally) return;
      const newRotation = value as number;

      // Enter camera edit mode when changing rotation
      setIsCameraTabActive(true);

      if (selectedKeyframeId) {
        // Keyframe is selected: update it
        updateKeyframe(selectedRallyId, selectedKeyframeId, { rotation: newRotation });
      } else {
        // No keyframe selected - check for nearby keyframe or create new
        const rallyDuration = selectedRally.end_time - selectedRally.start_time;
        const thresholdOffset = rallyDuration > 0 ? KEYFRAME_TIME_THRESHOLD / rallyDuration : 0;
        const nearestKeyframe = activeKeyframes.find(
          (kf) => Math.abs(kf.timeOffset - currentTimeOffset) < thresholdOffset
        );

        if (nearestKeyframe) {
          // Update nearby keyframe and select it
          updateKeyframe(selectedRallyId, nearestKeyframe.id, { rotation: newRotation });
          selectKeyframe(nearestKeyframe.id);
        } else {
          // Create new keyframe at current position with new rotation
          const currentState = getCameraStateAtTime(selectedRallyId, currentTimeOffset);
          const keyframe = createDefaultKeyframe(currentTimeOffset, {
            positionX: currentState.positionX,
            positionY: currentState.positionY,
            zoom: currentState.zoom,
            rotation: newRotation,
          });
          addKeyframe(selectedRallyId, keyframe);
        }
      }
    },
    [selectedRallyId, selectedRally, selectedKeyframeId, currentTimeOffset, activeKeyframes, updateKeyframe, addKeyframe, selectKeyframe, getCameraStateAtTime, setIsCameraTabActive]
  );

  // Global settings handlers
  const handleGlobalZoomChange = useCallback(
    (_: Event, value: number | number[]) => {
      if (!currentVideoId) return;
      setGlobalSettings(currentVideoId, { zoom: value as number });
    },
    [currentVideoId, setGlobalSettings]
  );

  const handleGlobalRotationChange = useCallback(
    (_: Event, value: number | number[]) => {
      if (!currentVideoId) return;
      setGlobalSettings(currentVideoId, { rotation: value as number });
    },
    [currentVideoId, setGlobalSettings]
  );

  const handleResetGlobalSettings = useCallback(() => {
    if (!currentVideoId) return;
    resetGlobalSettings(currentVideoId);
    setShowGlobalResetConfirm(false);
  }, [currentVideoId, resetGlobalSettings]);

  const handleResetCamera = useCallback(() => {
    if (selectedRallyId) {
      resetCamera(selectedRallyId);
      setShowResetConfirm(false);
    }
  }, [selectedRallyId, resetCamera]);

  // Check if rally has camera keyframes for the active aspect ratio
  const hasCameraKeyframes = activeKeyframes.length > 0;

  // Check if any rally in the session has camera edits (for disabling preview toggle)
  const cameraEdits = useCameraStore((state) => state.cameraEdits);
  const hasAnyCameraEdits = useMemo(() => {
    return rallies.some((rally) => {
      const edit = cameraEdits[rally.id];
      return (
        edit &&
        ((edit.keyframes.ORIGINAL?.length ?? 0) > 0 || (edit.keyframes.VERTICAL?.length ?? 0) > 0)
      );
    });
  }, [rallies, cameraEdits]);

  // Min zoom is always 1.0 - zoom out causes objectFit issues
  const zoomMin = 1.0;

  const rallyIndex = selectedRally ? rallies.indexOf(selectedRally) + 1 : 0;

  return (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
      {/* Global Video Settings - shown when no rally is selected */}
      {!selectedRally ? (
        currentVideoId ? (
          <Box sx={{ flex: 1, overflowY: 'auto', p: 2 }}>
            <Box sx={{ mb: 2, display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between' }}>
              <Box>
                <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 0.5 }}>
                  Video Camera Settings
                </Typography>
                <Typography variant="caption" sx={{ color: 'text.disabled', display: 'block' }}>
                  Base settings that apply to all rallies
                </Typography>
              </Box>
              {hasAnyCameraEdits && (
                <Tooltip title={applyCameraEdits ? 'Hide preview' : 'Show preview'}>
                  <IconButton
                    size="small"
                    onClick={toggleApplyCameraEdits}
                    sx={{ color: applyCameraEdits ? 'primary.main' : 'text.disabled', mt: -0.5 }}
                  >
                    {applyCameraEdits ? <VisibilityIcon fontSize="small" /> : <VisibilityOffIcon fontSize="small" />}
                  </IconButton>
                </Tooltip>
              )}
            </Box>

            {/* Global Zoom */}
            <Box sx={{ mb: 3, opacity: applyCameraEdits ? 1 : 0.5 }}>
              <Typography variant="caption" sx={{ color: 'text.secondary' }}>
                Zoom: {currentGlobalSettings.zoom.toFixed(1)}x
              </Typography>
              <Slider
                value={currentGlobalSettings.zoom}
                onChange={handleGlobalZoomChange}
                min={1.0}
                max={2.0}
                step={0.1}
                size="small"
                disabled={!applyCameraEdits}
              />
            </Box>

            {/* Global Rotation */}
            <Box sx={{ mb: 3, opacity: applyCameraEdits ? 1 : 0.5 }}>
              <Typography variant="caption" sx={{ color: 'text.secondary' }}>
                Rotation: {currentGlobalSettings.rotation.toFixed(1)}°
              </Typography>
              <Slider
                value={currentGlobalSettings.rotation}
                onChange={handleGlobalRotationChange}
                onChangeCommitted={() => setIsAdjustingRotation(false)}
                onMouseDown={() => setIsAdjustingRotation(true)}
                min={ROTATION_MIN}
                max={ROTATION_MAX}
                step={ROTATION_STEP}
                size="small"
                disabled={!applyCameraEdits}
                marks={[
                  { value: ROTATION_MIN, label: `${ROTATION_MIN}°` },
                  { value: 0, label: '0°' },
                  { value: ROTATION_MAX, label: `${ROTATION_MAX}°` },
                ]}
              />
            </Box>

            {videoHasGlobalSettings && (
              <Button
                size="small"
                variant="text"
                onClick={() => setShowGlobalResetConfirm(true)}
                disabled={!applyCameraEdits}
                startIcon={<RestartAltIcon sx={{ fontSize: 16 }} />}
                sx={{ color: 'text.secondary' }}
              >
                Reset to Defaults
              </Button>
            )}

            {/* Global settings reset confirmation dialog */}
            <Dialog
              open={showGlobalResetConfirm}
              onClose={() => setShowGlobalResetConfirm(false)}
              maxWidth="xs"
            >
              <DialogTitle>Reset Video Camera Settings?</DialogTitle>
              <DialogContent>
                <DialogContentText>
                  This will reset zoom, position, and rotation to their default values for this video.
                </DialogContentText>
              </DialogContent>
              <DialogActions>
                <Button onClick={() => setShowGlobalResetConfirm(false)}>Cancel</Button>
                <Button onClick={handleResetGlobalSettings} color="error" variant="contained">
                  Reset
                </Button>
              </DialogActions>
            </Dialog>

            <Divider sx={{ my: 2 }} />

            <Typography variant="caption" sx={{ color: 'text.disabled', display: 'block', textAlign: 'center' }}>
              Select a rally for per-rally camera editing
            </Typography>
          </Box>
        ) : (
          <Box
            sx={{
              flex: 1,
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              p: 3,
              color: 'text.secondary',
            }}
          >
            <VideocamIcon sx={{ fontSize: 48, mb: 2, opacity: 0.3 }} />
            <Typography variant="body2" sx={{ textAlign: 'center' }}>
              No video loaded
            </Typography>
          </Box>
        )
      ) : (
        <>
          {/* Rally-specific header */}
          <Box sx={{ px: 2, py: 1.5, borderBottom: 1, borderColor: 'divider', display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between' }}>
            <Box>
              <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 0.5 }}>
                Camera Edit
              </Typography>
              <Typography variant="caption" sx={{ color: 'text.secondary' }}>
                Rally {rallyIndex}: {formatTime(selectedRally.start_time)} -{' '}
                {formatTime(selectedRally.end_time)}
              </Typography>
            </Box>
            {hasAnyCameraEdits && (
              <Tooltip title={applyCameraEdits ? 'Hide preview' : 'Show preview'}>
                <IconButton
                  size="small"
                  onClick={toggleApplyCameraEdits}
                  sx={{ color: applyCameraEdits ? 'primary.main' : 'text.disabled', mt: -0.5 }}
                >
                  {applyCameraEdits ? <VisibilityIcon fontSize="small" /> : <VisibilityOffIcon fontSize="small" />}
                </IconButton>
              </Tooltip>
            )}
          </Box>

          {/* Controls */}
          <Box sx={{ flex: 1, overflowY: 'auto', p: 2, opacity: applyCameraEdits ? 1 : 0.5 }}>
            {/* Aspect ratio selector */}
            <Box sx={{ mb: 3 }}>
              <Typography variant="caption" sx={{ color: 'text.secondary', mb: 1, display: 'block' }}>
                Aspect Ratio
              </Typography>
              <ToggleButtonGroup
                value={cameraEdit?.aspectRatio ?? 'ORIGINAL'}
                exclusive
                onChange={handleAspectRatioChange}
                size="small"
                disabled={!applyCameraEdits}
                sx={{ width: '100%' }}
              >
                <ToggleButton value="ORIGINAL" sx={{ flex: 1 }}>
                  <Crop169Icon sx={{ mr: 0.5, fontSize: 18 }} />
                  16:9
                </ToggleButton>
                <ToggleButton value="VERTICAL" sx={{ flex: 1 }}>
                  <CropPortraitIcon sx={{ mr: 0.5, fontSize: 18 }} />
                  9:16
                </ToggleButton>
              </ToggleButtonGroup>
            </Box>

            <Divider sx={{ my: 2 }} />

            {/* Keyframes section */}
            <Box sx={{ mb: 2 }}>
              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 1 }}>
                <Typography variant="caption" sx={{ color: 'text.secondary' }}>
                  Keyframes
                </Typography>
                {hasCameraKeyframes && (
                  <Tooltip title="Reset all camera settings">
                    <span>
                      <IconButton
                        size="small"
                        onClick={() => setShowResetConfirm(true)}
                        disabled={!applyCameraEdits}
                        sx={{ color: 'error.main', opacity: 0.7, '&:hover': { opacity: 1 } }}
                      >
                        <RestartAltIcon fontSize="small" />
                      </IconButton>
                    </span>
                  </Tooltip>
                )}
              </Box>

              {/* Keyframe list */}
              <Stack spacing={0.5}>
                {activeKeyframes.map((kf) => (
                  <KeyframeItem
                    key={kf.id}
                    keyframe={kf}
                    isSelected={selectedKeyframeId === kf.id}
                    isDeleting={deleteConfirmId === kf.id}
                    onSelect={handleKeyframeSelect}
                    onDelete={handleKeyframeDeleteClick}
                    onCancelDelete={handleCancelKeyframeDelete}
                    rallyStartTime={selectedRally.start_time}
                    rallyDuration={rallyDuration}
                  />
                ))}
              </Stack>

              {activeKeyframes.length === 0 && (
                <Box sx={{ textAlign: 'center', py: 2 }}>
                  <Typography variant="caption" sx={{ color: 'text.disabled', display: 'block' }}>
                    Drag on video to create keyframes
                  </Typography>
                </Box>
              )}
            </Box>

            <Divider sx={{ my: 2 }} />

            {/* Camera controls - always visible */}
            <Box>
              {/* Selected keyframe info */}
              {selectedKeyframe && (
                <>
                  <Typography variant="caption" sx={{ color: 'text.secondary', mb: 1, display: 'block' }}>
                    Selected Keyframe
                  </Typography>
                  <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block', mb: 2 }}>
                    Time: {formatTime(selectedRally.start_time + selectedKeyframe.timeOffset * rallyDuration)}
                    {' '}({formatOffset(selectedKeyframe.timeOffset)} of rally)
                  </Typography>
                </>
              )}

              {/* Zoom slider - always available */}
              <Box sx={{ mb: 2 }}>
                <Typography variant="caption" sx={{ color: 'text.secondary' }}>
                  Zoom: {(selectedKeyframe?.zoom ?? getCameraStateAtTime(selectedRallyId!, currentTimeOffset).zoom).toFixed(1)}x
                </Typography>
                <Slider
                  value={selectedKeyframe?.zoom ?? getCameraStateAtTime(selectedRallyId!, currentTimeOffset).zoom}
                  onChange={handleZoomChange}
                  min={zoomMin}
                  max={ZOOM_MAX}
                  step={ZOOM_STEP}
                  size="small"
                  disabled={!applyCameraEdits}
                />
              </Box>

              {/* Rotation slider */}
              <Box sx={{ mb: 2 }}>
                <Typography variant="caption" sx={{ color: 'text.secondary' }}>
                  Rotation: {(selectedKeyframe?.rotation ?? getCameraStateAtTime(selectedRallyId!, currentTimeOffset).rotation).toFixed(1)}°
                </Typography>
                <Slider
                  value={selectedKeyframe?.rotation ?? getCameraStateAtTime(selectedRallyId!, currentTimeOffset).rotation}
                  onChange={handleRotationChange}
                  onChangeCommitted={() => setIsAdjustingRotation(false)}
                  onMouseDown={() => setIsAdjustingRotation(true)}
                  min={ROTATION_MIN}
                  max={ROTATION_MAX}
                  step={ROTATION_STEP}
                  size="small"
                  disabled={!applyCameraEdits}
                  marks={[
                    { value: ROTATION_MIN, label: `${ROTATION_MIN}°` },
                    { value: 0, label: '0°' },
                    { value: ROTATION_MAX, label: `${ROTATION_MAX}°` },
                  ]}
                />
              </Box>

              {/* Position hint */}
              <Typography variant="caption" sx={{ color: 'text.disabled', fontStyle: 'italic' }}>
                {selectedKeyframe
                  ? 'Drag on video to reposition keyframe'
                  : 'Drag on video or adjust zoom to create keyframe'}
              </Typography>
            </Box>
          </Box>

          {/* Current position indicator */}
          <Box
            sx={{
              px: 2,
              py: 1,
              borderTop: 1,
              borderColor: 'divider',
              bgcolor: designTokens.colors.surface[2],
            }}
          >
            <Typography variant="caption" sx={{ color: 'text.secondary' }}>
              Playhead: {formatTime(currentTime)} ({formatOffset(currentTimeOffset)} of rally)
            </Typography>
          </Box>

          {/* Reset confirmation dialog */}
          <Dialog
            open={showResetConfirm}
            onClose={() => setShowResetConfirm(false)}
            maxWidth="xs"
          >
            <DialogTitle>Reset Camera Settings?</DialogTitle>
            <DialogContent>
              <DialogContentText>
                This will remove all keyframes and camera settings for this rally. This action cannot be undone.
              </DialogContentText>
            </DialogContent>
            <DialogActions>
              <Button onClick={() => setShowResetConfirm(false)}>Cancel</Button>
              <Button onClick={handleResetCamera} color="error" variant="contained">
                Reset
              </Button>
            </DialogActions>
          </Dialog>
        </>
      )}
    </Box>
  );
}
