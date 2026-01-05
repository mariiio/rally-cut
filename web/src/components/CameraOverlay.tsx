'use client';

import { useRef, useState, useCallback, useEffect } from 'react';
import { Box, Typography } from '@mui/material';
import { useCameraStore, selectCameraEdit, selectSelectedKeyframeId, createDefaultKeyframe } from '@/stores/cameraStore';
import { useEditorStore } from '@/stores/editorStore';
import { usePlayerStore } from '@/stores/playerStore';
import { getValidPositionRange } from '@/utils/cameraInterpolation';
import type { AspectRatio } from '@/types/camera';
import { KEYFRAME_TIME_THRESHOLD } from '@/types/camera';

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

  // Get selected rally
  const selectedRallyId = useEditorStore((state) => state.selectedRallyId);
  const rallies = useEditorStore((state) => state.rallies);
  const selectedRally = rallies.find((r) => r.id === selectedRallyId) ?? null;

  // Player state
  const currentTime = usePlayerStore((state) => state.currentTime);
  const applyCameraEdits = usePlayerStore((state) => state.applyCameraEdits);

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

  // Show overlay when camera preview is on AND there's a selected rally
  const isActive = applyCameraEdits && selectedRallyId && selectedRally;

  // Get aspect ratio
  const aspectRatio: AspectRatio = cameraEdit?.aspectRatio ?? 'ORIGINAL';
  const isVertical = aspectRatio === 'VERTICAL';

  // Handle mouse down - start drag
  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    if (!selectedRallyId) return;

    e.preventDefault();
    e.stopPropagation();

    // Get starting position - from selected keyframe or current camera state
    let startPosX: number;
    let startPosY: number;

    if (selectedKeyframe) {
      startPosX = selectedKeyframe.positionX;
      startPosY = selectedKeyframe.positionY;
    } else {
      // Get current camera state at playhead position
      const currentState = getCameraStateAtTime(selectedRallyId, currentTimeOffset);
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
  }, [selectedKeyframe, selectedRallyId, getCameraStateAtTime, currentTimeOffset]);

  // Get current zoom level from selected keyframe or current camera state
  const currentZoom = selectedKeyframe?.zoom ?? (selectedRallyId ? getCameraStateAtTime(selectedRallyId, currentTimeOffset).zoom : 1);

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
      if (!localPosition || !selectedRallyId) {
        setDragState(null);
        setLocalPosition(null);
        setDragPosition(null);
        return;
      }

      if (selectedKeyframeId) {
        // Keyframe is selected: update it
        updateKeyframe(selectedRallyId, selectedKeyframeId, {
          positionX: localPosition.x,
          positionY: localPosition.y,
        });
      } else {
        // No keyframe selected - check for nearby keyframe or create new
        // Access activeKeyframes via cameraEdit to avoid stale closure
        const edit = useCameraStore.getState().cameraEdits[selectedRallyId];
        const keyframes = edit ? edit.keyframes[edit.aspectRatio] ?? [] : [];

        // Find nearest keyframe within threshold
        const nearestKeyframe = keyframes.find(
          (kf) => Math.abs(kf.timeOffset - currentTimeOffset) < KEYFRAME_TIME_THRESHOLD
        );

        if (nearestKeyframe) {
          // Update nearby keyframe and select it
          updateKeyframe(selectedRallyId, nearestKeyframe.id, {
            positionX: localPosition.x,
            positionY: localPosition.y,
          });
          selectKeyframe(nearestKeyframe.id);
        } else {
          // Create new keyframe at current position
          const currentState = getCameraStateAtTime(selectedRallyId, currentTimeOffset);
          const newKeyframe = createDefaultKeyframe(currentTimeOffset, {
            positionX: localPosition.x,
            positionY: localPosition.y,
            zoom: currentState.zoom,
          });
          addKeyframe(selectedRallyId, newKeyframe);
          // addKeyframe automatically selects the new keyframe
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
  }, [dragState, containerRef, isVertical, currentZoom, localPosition, selectedRallyId, selectedKeyframeId, currentTimeOffset, updateKeyframe, addKeyframe, selectKeyframe, getCameraStateAtTime, setDragPosition]);

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
            {selectedKeyframe ? 'Drag to reposition' : 'Drag to add keyframe'}
          </Typography>
        </Box>
      )}
    </Box>
  );
}
