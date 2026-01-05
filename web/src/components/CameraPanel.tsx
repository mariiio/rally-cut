'use client';

import { useMemo, useCallback, memo, useState, useEffect } from 'react';
import {
  Box,
  Typography,
  IconButton,
  Switch,
  FormControlLabel,
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
import { useEditorStore } from '@/stores/editorStore';
import { usePlayerStore } from '@/stores/playerStore';
import { useCameraStore, createDefaultKeyframe, selectCameraEdit, selectSelectedKeyframeId } from '@/stores/cameraStore';
import { designTokens } from '@/app/theme';
import type { AspectRatio, CameraKeyframe } from '@/types/camera';
import { ZOOM_MAX, ZOOM_STEP, KEYFRAME_TIME_THRESHOLD } from '@/types/camera';

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
  onSelect: () => void;
  onDelete: () => void;
  onCancelDelete: () => void;
  rallyStartTime: number;
  rallyDuration: number;
}) {
  const handleDelete = useCallback((e: React.MouseEvent) => {
    e.stopPropagation();
    onDelete();
  }, [onDelete]);

  const handleCancelDelete = useCallback((e: React.MouseEvent) => {
    e.stopPropagation();
    onCancelDelete();
  }, [onCancelDelete]);

  // Calculate absolute time for display
  const absoluteTime = rallyStartTime + keyframe.timeOffset * rallyDuration;

  return (
    <Box
      onClick={onSelect}
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
  // Local state for reset confirmation dialog
  const [showResetConfirm, setShowResetConfirm] = useState(false);
  // Local state for keyframe delete confirmation
  const [deleteConfirmId, setDeleteConfirmId] = useState<string | null>(null);

  // Editor store - get selected rally and camera tab state
  const selectedRallyId = useEditorStore((state) => state.selectedRallyId);
  const rallies = useEditorStore((state) => state.rallies);
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

  // Get drag position to detect active dragging
  const dragPosition = useCameraStore((state) => state.dragPosition);

  // Auto-deselect keyframe when playhead moves away
  useEffect(() => {
    // Skip if no selection, no edit, or during drag
    if (!selectedKeyframeId || !cameraEdit || dragPosition) return;

    const selectedKf = activeKeyframes.find((kf) => kf.id === selectedKeyframeId);
    if (!selectedKf || !selectedRally) return;

    const distance = Math.abs(currentTimeOffset - selectedKf.timeOffset);

    if (distance > KEYFRAME_TIME_THRESHOLD) {
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

  const handleZoomChange = useCallback(
    (_: Event, value: number | number[]) => {
      if (!selectedRallyId) return;
      const newZoom = value as number;

      // Enter camera edit mode when changing zoom
      setIsCameraTabActive(true);

      if (selectedKeyframeId) {
        // Keyframe is selected: update it
        updateKeyframe(selectedRallyId, selectedKeyframeId, { zoom: newZoom });
      } else {
        // No keyframe selected - check for nearby keyframe or create new
        const nearestKeyframe = activeKeyframes.find(
          (kf) => Math.abs(kf.timeOffset - currentTimeOffset) < KEYFRAME_TIME_THRESHOLD
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
    [selectedRallyId, selectedKeyframeId, currentTimeOffset, activeKeyframes, updateKeyframe, addKeyframe, selectKeyframe, getCameraStateAtTime, setIsCameraTabActive]
  );

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

  // Empty state - no rally selected
  if (!selectedRally) {
    return (
      <Box
        sx={{
          height: '100%',
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
          Select a rally to edit camera
        </Typography>
        <Typography variant="caption" sx={{ textAlign: 'center', mt: 1, color: 'text.disabled' }}>
          Click on a rally from the list to start editing
        </Typography>
      </Box>
    );
  }

  const rallyIndex = rallies.indexOf(selectedRally) + 1;

  return (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
      {/* Global preview toggle - separate from rally-specific settings */}
      <Box sx={{ px: 2, py: 1.5, borderBottom: 1, borderColor: 'divider' }}>
        <FormControlLabel
          control={
            <Switch
              checked={applyCameraEdits}
              onChange={toggleApplyCameraEdits}
              color="primary"
              size="small"
              disabled={!hasAnyCameraEdits}
            />
          }
          label={
            <Typography
              variant="body2"
              sx={{ color: hasAnyCameraEdits ? 'text.primary' : 'text.disabled' }}
            >
              Preview camera edits
            </Typography>
          }
          sx={{ m: 0 }}
        />
        {!hasAnyCameraEdits && (
          <Typography variant="caption" sx={{ color: 'text.disabled', display: 'block', mt: 0.5 }}>
            Add keyframes to enable preview
          </Typography>
        )}
      </Box>

      {/* Rally-specific header */}
      <Box sx={{ px: 2, py: 1.5, borderBottom: 1, borderColor: 'divider' }}>
        <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 0.5 }}>
          Camera Edit
        </Typography>
        <Typography variant="caption" sx={{ color: 'text.secondary' }}>
          Rally {rallyIndex}: {formatTime(selectedRally.start_time)} -{' '}
          {formatTime(selectedRally.end_time)}
        </Typography>
      </Box>

      {/* Controls */}
      <Box sx={{ flex: 1, overflowY: 'auto', p: 2 }}>
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
              Keyframes ({activeKeyframes.length})
            </Typography>
            {hasCameraKeyframes && (
              <Tooltip title="Reset all camera settings">
                <IconButton
                  size="small"
                  onClick={() => setShowResetConfirm(true)}
                  sx={{ color: 'error.main', opacity: 0.7, '&:hover': { opacity: 1 } }}
                >
                  <RestartAltIcon fontSize="small" />
                </IconButton>
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
                onSelect={() => {
                  selectKeyframe(kf.id);
                  // Seek to keyframe position
                  const keyframeTime = selectedRally.start_time + kf.timeOffset * rallyDuration;
                  seek(keyframeTime);
                  // Enter camera edit mode when clicking a keyframe
                  setIsCameraTabActive(true);
                }}
                onDelete={() => handleKeyframeDeleteClick(kf.id)}
                onCancelDelete={handleCancelKeyframeDelete}
                rallyStartTime={selectedRally.start_time}
                rallyDuration={rallyDuration}
              />
            ))}
          </Stack>

          {activeKeyframes.length === 0 && (
            <Box sx={{ textAlign: 'center', py: 2 }}>
              <Typography variant="caption" sx={{ color: 'text.disabled', display: 'block' }}>
                Drag on video or adjust zoom to add keyframes
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
    </Box>
  );
}
