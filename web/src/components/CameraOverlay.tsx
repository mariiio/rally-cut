'use client';

import { useRef, useState, useCallback, useEffect } from 'react';
import { Box, Typography } from '@mui/material';
import { useCameraStore, selectCameraEdit, selectSelectedKeyframeId, createDefaultKeyframe } from '@/stores/cameraStore';
import { useEditorStore } from '@/stores/editorStore';
import { usePlayerStore } from '@/stores/playerStore';
import { getValidPositionRange } from '@/utils/cameraInterpolation';
import type { AspectRatio } from '@/types/camera';
import { KEYFRAME_TIME_THRESHOLD, DEFAULT_GLOBAL_CAMERA } from '@/types/camera';

interface DragState {
  isDragging: boolean;
  startX: number;
  startY: number;
  startPosX: number;
  startPosY: number;
}

interface CameraOverlayProps {
  containerRef: React.RefObject<HTMLElement | null>;
}

export function CameraOverlay({ containerRef }: CameraOverlayProps) {
  const overlayRef = useRef<HTMLDivElement>(null);
  const [dragState, setDragState] = useState<DragState | null>(null);
  const [localPosition, setLocalPosition] = useState<{ x: number; y: number } | null>(null);
  // Track if user has explicitly exited camera edit mode for current rally
  const [exitedForRally, setExitedForRally] = useState<string | null>(null);

  // Get selected rally and active match
  const selectedRallyId = useEditorStore((state) => state.selectedRallyId);
  const activeMatchId = useEditorStore((state) => state.activeMatchId);
  const rallies = useEditorStore((state) => state.rallies);
  const isCameraTabActive = useEditorStore((state) => state.isCameraTabActive);
  const setIsCameraTabActive = useEditorStore((state) => state.setIsCameraTabActive);
  const selectedRally = rallies.find((r) => r.id === selectedRallyId) ?? null;

  // Global camera settings
  const globalCameraSettings = useCameraStore((state) => state.globalCameraSettings);
  const setGlobalSettings = useCameraStore((state) => state.setGlobalSettings);
  const currentGlobalSettings = activeMatchId ? globalCameraSettings[activeMatchId] ?? DEFAULT_GLOBAL_CAMERA : DEFAULT_GLOBAL_CAMERA;

  // Track when user exits camera edit mode - hide overlay until they select a different rally
  const prevIsCameraTabActive = useRef(isCameraTabActive);
  useEffect(() => {
    if (prevIsCameraTabActive.current && !isCameraTabActive && selectedRallyId) {
      // User exited camera edit mode - remember this rally
      // eslint-disable-next-line react-hooks/set-state-in-effect -- intentional: tracking previous state
      setExitedForRally(selectedRallyId);
    }
    prevIsCameraTabActive.current = isCameraTabActive;
  }, [isCameraTabActive, selectedRallyId]);

  // Reset exitedForRally when selecting a different rally
  useEffect(() => {
    if (selectedRallyId !== exitedForRally) {
      // eslint-disable-next-line react-hooks/set-state-in-effect -- intentional: syncing derived state
      setExitedForRally(null);
    }
  }, [selectedRallyId, exitedForRally]);

  // Player state
  const currentTime = usePlayerStore((state) => state.currentTime);
  const isFullscreen = usePlayerStore((state) => state.isFullscreen);

  // Camera state
  const cameraEdit = useCameraStore(selectCameraEdit(selectedRallyId));
  const selectedKeyframeId = useCameraStore(selectSelectedKeyframeId);
  const updateKeyframe = useCameraStore((state) => state.updateKeyframe);
  const addKeyframe = useCameraStore((state) => state.addKeyframe);
  const selectKeyframe = useCameraStore((state) => state.selectKeyframe);
  const setDragPosition = useCameraStore((state) => state.setDragPosition);
  const getCameraStateAtTime = useCameraStore((state) => state.getCameraStateAtTime);

  // Get the selected keyframe from the active aspect ratio
  const activeKeyframes = cameraEdit ? cameraEdit.keyframes[cameraEdit.aspectRatio] ?? [] : [];
  const selectedKeyframe = activeKeyframes.find((kf) => kf.id === selectedKeyframeId);

  // Calculate current time offset within rally
  const currentTimeOffset = selectedRally
    ? Math.max(0, Math.min(1, (currentTime - selectedRally.start_time) / (selectedRally.end_time - selectedRally.start_time)))
    : 0;

  // Check if rally has camera edits (keyframes in any aspect ratio)
  const hasCameraEdits = cameraEdit && (
    (cameraEdit.keyframes.ORIGINAL?.length ?? 0) > 0 ||
    (cameraEdit.keyframes.VERTICAL?.length ?? 0) > 0
  );

  // Check if we have global camera settings
  const hasGlobalCameraSettings = currentGlobalSettings.zoom !== 1.0 ||
    currentGlobalSettings.positionX !== 0.5 ||
    currentGlobalSettings.positionY !== 0.5 ||
    currentGlobalSettings.rotation !== 0;

  // Check if position dragging would have any visible effect in global mode
  // (only if there's zoom > 1 or rotation, otherwise position changes aren't visible)
  const canDragInGlobalMode = currentGlobalSettings.zoom > 1.0 || currentGlobalSettings.rotation !== 0;

  // Mode: are we editing global settings (no rally selected) or per-rally?
  const isGlobalMode = !selectedRallyId && activeMatchId;

  // Show overlay when:
  // - Not in fullscreen mode, AND
  // - In camera edit mode (isCameraTabActive) with a rally selected, OR
  // - Rally with camera edits is selected (for quick editing), unless user exited edit mode for this rally, OR
  // - In global mode (no rally selected) with zoom or rotation that makes position dragging useful
  const isActive = !isFullscreen && ((selectedRallyId && selectedRally && (
    isCameraTabActive || (hasCameraEdits && exitedForRally !== selectedRallyId)
  )) || (isGlobalMode && canDragInGlobalMode));

  // Get aspect ratio
  const aspectRatio: AspectRatio = cameraEdit?.aspectRatio ?? 'ORIGINAL';
  const isVertical = aspectRatio === 'VERTICAL';

  // Handle mouse down - start drag
  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    // Need either a selected rally or global mode
    if (!selectedRallyId && !isGlobalMode) return;

    e.preventDefault();
    e.stopPropagation();

    // Enter camera edit mode when user starts dragging
    if (!isCameraTabActive) {
      setIsCameraTabActive(true);
    }

    // Get starting position
    let startPosX: number;
    let startPosY: number;

    if (isGlobalMode) {
      // Global mode: use global settings position
      startPosX = currentGlobalSettings.positionX;
      startPosY = currentGlobalSettings.positionY;
    } else if (selectedKeyframe) {
      // Per-rally mode with selected keyframe
      startPosX = selectedKeyframe.positionX;
      startPosY = selectedKeyframe.positionY;
    } else {
      // Per-rally mode: get current camera state at playhead position
      const currentState = getCameraStateAtTime(selectedRallyId!, currentTimeOffset);
      startPosX = currentState.positionX;
      startPosY = currentState.positionY;
    }

    setDragState({
      isDragging: true,
      startX: e.clientX,
      startY: e.clientY,
      startPosX,
      startPosY,
    });
    setLocalPosition({
      x: startPosX,
      y: startPosY,
    });
  }, [selectedKeyframe, selectedRallyId, isGlobalMode, currentGlobalSettings, getCameraStateAtTime, currentTimeOffset, isCameraTabActive, setIsCameraTabActive]);

  // Get current zoom level from selected keyframe, global settings, or current camera state
  const currentZoom = isGlobalMode
    ? currentGlobalSettings.zoom
    : (selectedKeyframe?.zoom ?? (selectedRallyId ? getCameraStateAtTime(selectedRallyId, currentTimeOffset).zoom : 1));

  // Handle mouse move - update position locally AND in store for live preview
  useEffect(() => {
    if (!dragState?.isDragging || !containerRef.current) return;

    const handleMouseMove = (e: MouseEvent) => {
      // containerRef.current is guaranteed by the early return at line 102
      const rect = containerRef.current!.getBoundingClientRect();
      // Calculate drag delta as percentage of container
      const deltaX = (e.clientX - dragState.startX) / rect.width;
      const deltaY = (e.clientY - dragState.startY) / rect.height;

      // Get valid position range based on aspect ratio and zoom
      const { minX, maxX, minY, maxY } = getValidPositionRange(
        isVertical ? 'VERTICAL' : 'ORIGINAL',
        currentZoom
      );

      // Calculate new position with clamping to valid bounds
      // Drag direction is inverted (drag right = look left in video)
      const newPosX = Math.max(minX, Math.min(maxX, dragState.startPosX - deltaX));
      const newPosY = Math.max(minY, Math.min(maxY, dragState.startPosY - deltaY));

      const newPos = { x: newPosX, y: newPosY };
      setLocalPosition(newPos);
      // Also update store so VideoPlayer can show live preview
      setDragPosition(newPos);
    };

    const handleMouseUp = () => {
      if (!localPosition) {
        setDragState(null);
        setLocalPosition(null);
        setDragPosition(null);
        return;
      }

      // Check current mode (can't rely on closure for isGlobalMode)
      const currentSelectedRallyId = useEditorStore.getState().selectedRallyId;
      const currentActiveMatchId = useEditorStore.getState().activeMatchId;
      const isCurrentlyGlobalMode = !currentSelectedRallyId && currentActiveMatchId;

      if (isCurrentlyGlobalMode && currentActiveMatchId) {
        // Global mode: update global settings
        setGlobalSettings(currentActiveMatchId, {
          positionX: localPosition.x,
          positionY: localPosition.y,
        });
      } else if (currentSelectedRallyId) {
        // Per-rally mode
        if (selectedKeyframeId) {
          // Keyframe is selected: update it
          updateKeyframe(currentSelectedRallyId, selectedKeyframeId, {
            positionX: localPosition.x,
            positionY: localPosition.y,
          });
        } else {
          // No keyframe selected - check for nearby keyframe or create new
          // Access activeKeyframes via cameraEdit to avoid stale closure
          const edit = useCameraStore.getState().cameraEdits[currentSelectedRallyId];
          const keyframes = edit ? edit.keyframes[edit.aspectRatio] ?? [] : [];

          // Find nearest keyframe within threshold (convert seconds to timeOffset)
          const rallyDuration = selectedRally ? selectedRally.end_time - selectedRally.start_time : 0;
          const thresholdOffset = rallyDuration > 0 ? KEYFRAME_TIME_THRESHOLD / rallyDuration : 0;
          const nearestKeyframe = keyframes.find(
            (kf) => Math.abs(kf.timeOffset - currentTimeOffset) < thresholdOffset
          );

          if (nearestKeyframe) {
            // Update nearby keyframe and select it
            updateKeyframe(currentSelectedRallyId, nearestKeyframe.id, {
              positionX: localPosition.x,
              positionY: localPosition.y,
            });
            selectKeyframe(nearestKeyframe.id);
          } else {
            // Create new keyframe at current position
            const currentState = getCameraStateAtTime(currentSelectedRallyId, currentTimeOffset);
            const newKeyframe = createDefaultKeyframe(currentTimeOffset, {
              positionX: localPosition.x,
              positionY: localPosition.y,
              zoom: currentState.zoom,
            });
            addKeyframe(currentSelectedRallyId, newKeyframe);
            // addKeyframe automatically selects the new keyframe
          }
        }
      }

      setDragState(null);
      setLocalPosition(null);
      // Clear drag position from store
      setDragPosition(null);
    };

    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseup', handleMouseUp);

    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
    };
  }, [dragState, containerRef, isVertical, currentZoom, localPosition, selectedRallyId, selectedKeyframeId, currentTimeOffset, updateKeyframe, addKeyframe, selectKeyframe, getCameraStateAtTime, setDragPosition, setGlobalSettings]);

  if (!isActive) return null;

  return (
    <Box
      ref={overlayRef}
      onMouseDown={handleMouseDown}
      sx={{
        position: 'absolute',
        inset: 0,
        cursor: dragState?.isDragging ? 'grabbing' : 'grab',
        zIndex: 10,
        // Show visual feedback
        '&::before': {
          content: '""',
          position: 'absolute',
          inset: 0,
          border: '2px dashed',
          borderColor: dragState?.isDragging ? 'primary.main' : 'rgba(255,255,255,0.5)',
          borderRadius: 1,
          pointerEvents: 'none',
          transition: 'border-color 0.15s',
        },
      }}
    >
      {/* Instructions */}
      {!dragState?.isDragging && (
        <Box
          sx={{
            position: 'absolute',
            bottom: 8,
            left: '50%',
            transform: 'translateX(-50%)',
            bgcolor: 'rgba(0,0,0,0.7)',
            px: 1.5,
            py: 0.5,
            borderRadius: 1,
            pointerEvents: 'none',
          }}
        >
          <Typography variant="caption" sx={{ color: 'white', whiteSpace: 'nowrap' }}>
            {isGlobalMode
              ? 'Drag to position camera'
              : selectedKeyframe
                ? 'Drag to reposition'
                : 'Drag to add keyframe'}
          </Typography>
        </Box>
      )}
    </Box>
  );
}
